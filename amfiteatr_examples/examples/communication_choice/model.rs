use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use amfiteatr_core::agent::{AgentGen, AutomaticAgent, RandomPolicy, ReseedAgent};
use amfiteatr_core::comm::StdEnvironmentEndpoint;
use amfiteatr_core::demo::{DemoDomain, DemoInfoSet, DemoState};
use amfiteatr_core::env::{HashMapEnvironment, ReseedEnvironment, RoundRobinUniversalEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_net_ext::{ComplexComm512, DomainCommA512, DomainCommE512};
use amfiteatr_net_ext::tcp::BoundedTcpEnvironmentEndpoint;
use crate::options::{CCOptions, CommunicationMedium};

pub type ErrorAmfi = AmfiteatrError<DemoDomain>;


type CEnvironment = HashMapEnvironment<DemoDomain, DemoState, DomainCommE512<DemoDomain>>;
type CAgent = AgentGen<DemoDomain, RandomPolicy<DemoDomain, DemoInfoSet>, DomainCommA512<DemoDomain>>;

pub const BANDITS: [(f32, f32);3] = [(1.0, 3.0), (5.6, 6.7), (0.1, 9.0)];

pub struct CCModel{

    env: CEnvironment,
    agents: Vec<Arc<Mutex<CAgent>>>
}

impl CCModel{
    pub fn new(
        options: &CCOptions) -> Self{

        let number_of_bandits = BANDITS.len();

        let mut agents = Vec::new();
        let mut env_comms = HashMap::new();

        let random_policy = RandomPolicy::<DemoDomain, DemoInfoSet>::new();
        
        for id in 0..options.number_of_players{
            
            let (env_comm, agent_comm) = match options.comm{
                CommunicationMedium::Mpsc => {
                    let (e,a) = StdEnvironmentEndpoint::new_pair();
                    (DomainCommE512::StdSync(e), DomainCommA512::StdSync(a))
                }
                CommunicationMedium::Tcp => {
                    let (e,a) = BoundedTcpEnvironmentEndpoint::<DemoDomain, 512>::create_local_pair(20000+id as u16).unwrap();
                    (DomainCommE512::Tcp(e), DomainCommA512::Tcp(a))
                }
            };
            env_comms.insert(id, env_comm);
            let info_set = DemoInfoSet::new(id, number_of_bandits);
            agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, agent_comm, random_policy.clone()))));
            
        }
        let player_set = (0..options.number_of_players).collect();
        let state = DemoState::new_with_players(BANDITS.into(), options.rounds, &player_set);
        let env = CEnvironment::new(state, env_comms);
        
        Self{
            env,
            agents,
        }
        
        


    }

    pub fn run_single_game(&mut self){
        std::thread::scope(|s|{
            s.spawn(||{
                self.env.run_round_robin_with_rewards().unwrap()
            });

            for agent in self.agents.iter(){
                s.spawn(||{
                    let mut guard = agent.as_ref().lock().unwrap();
                    guard.run().unwrap();
                });
            }
        });
    }

    pub fn run_several_games(&mut self, number_of_games: usize){


        for i in 0..number_of_games{
            log::info!("Playing game {i}");
            self.env.reseed(()).unwrap();
            for agent in self.agents.iter(){
                let mut guard = agent.as_ref().lock().unwrap();
                guard.reseed(()).unwrap()
            }
            self.run_single_game();
        }
    }
}
