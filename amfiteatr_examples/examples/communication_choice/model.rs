use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use amfiteatr_core::agent::{AgentGen, AutomaticAgent, RandomPolicy, ReseedAgent};
use amfiteatr_core::comm::{AgentEndpoint, AgentMpscAdapter, EnvironmentAdapter, EnvironmentEndpoint, EnvironmentMpscPort, StdEnvironmentEndpoint};
use amfiteatr_core::demo::{DemoDomain, DemoInfoSet, DemoState};
use amfiteatr_core::env::{AutoEnvironmentWithScores, BasicEnvironment, HashMapEnvironment, ReseedEnvironment, RoundRobinUniversalEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_net_ext::{DomainCommA512, DomainCommE512};
use amfiteatr_net_ext::tcp::PairedTcpEnvironmentEndpoint;
use crate::options::{CCOptions, CommunicationMedium};

pub type ErrorAmfi = AmfiteatrError<DemoDomain>;


//type MapEnvironment = HashMapEnvironment<DemoDomain, DemoState, DomainCommE512<DemoDomain>>;
type MapEnvironment<C: EnvironmentEndpoint<DemoDomain> + Send> = HashMapEnvironment<DemoDomain, DemoState, C>;
type CentralEnvironment = BasicEnvironment<DemoDomain, DemoState, EnvironmentMpscPort<DemoDomain>>;

type MappedAgent<C: AgentEndpoint<DemoDomain> + Send> = AgentGen<DemoDomain, RandomPolicy<DemoDomain, DemoInfoSet>, C>;
type CAgent = AgentGen<DemoDomain, RandomPolicy<DemoDomain, DemoInfoSet>, AgentMpscAdapter<DemoDomain>>;
pub const BANDITS: [(f32, f32);3] = [(1.0, 3.0), (5.6, 6.7), (0.1, 9.0)];

pub struct MapModel<CE:EnvironmentEndpoint<DemoDomain>, CA:  AgentEndpoint<DemoDomain>> {

    env: MapEnvironment<CE>,
    agents: Vec<Arc<Mutex<MappedAgent<CA>>>>
}

impl MapModel<DomainCommE512<DemoDomain>, DomainCommA512<DemoDomain>>{
    pub fn new(
        options: &CCOptions) -> Result<Self, anyhow::Error>{

        let number_of_bandits = BANDITS.len();

        let mut agents = Vec::new();
        let mut env_comms = HashMap::new();

        let random_policy = RandomPolicy::<DemoDomain, DemoInfoSet>::new();


        match options.comm{
            CommunicationMedium::StaticMpsc => {
                for id in 0..options.number_of_players{
                    let (e,a) = StdEnvironmentEndpoint::new_pair();
                    env_comms.insert(id, DomainCommE512::StdSync(e));
                    let info_set = DemoInfoSet::new(id, number_of_bandits);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, DomainCommA512::StdSync(a), random_policy.clone()))))
                }
            }
            CommunicationMedium::StaticTcp => {
                let agents_vec: Vec<usize> = (0..options.number_of_players).collect();
                let (mapped_env_comms, mapped_agent_comms) = PairedTcpEnvironmentEndpoint::create_local_net(28000, agents_vec.iter()).unwrap();
                env_comms = mapped_env_comms.into_iter().map(|(i, ep)|{
                    (i, DomainCommE512::Tcp(ep))
                }).collect();

                for (id, a_comm) in mapped_agent_comms{
                    let info_set = DemoInfoSet::new(id, number_of_bandits);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, DomainCommA512::Tcp(a_comm), random_policy.clone()))))
                }


            }
            m => {
                return Err(anyhow::Error::msg(format!("Comm type not implemented for 1-1 model: {:?}", m)))
            }
        }

        let player_set = (0..options.number_of_players).collect();
        let state = DemoState::new_with_players(BANDITS.into(), options.rounds, &player_set);
        let env = MapEnvironment::new(state, env_comms);

        Ok(Self{
            env,
            agents,
        })




    }
}

impl<
    CE:EnvironmentEndpoint<DemoDomain> + Send,
    CA:  AgentEndpoint<DemoDomain> + Send
> MapModel<CE, CA> {


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


pub struct CentralModel{
    env: CentralEnvironment,
    agents: Vec<Arc<Mutex<CAgent>>>
}

impl CentralModel{

    pub fn new(
        options: &CCOptions) -> Result<Self, anyhow::Error>{
        let number_of_bandits = BANDITS.len();

        let mut agents = Vec::new();

        let mut env_communicator = EnvironmentMpscPort::new();
        let random_policy = RandomPolicy::<DemoDomain, DemoInfoSet>::new();

        if options.comm == CommunicationMedium::CentralMpsc{

            for id in 0..options.number_of_players{
                let agent_comm = env_communicator.register_agent(id)?;
                let info_set = DemoInfoSet::new(id, number_of_bandits);
                agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, agent_comm, random_policy.clone()))))
            }



        }
        else {
            return Err(anyhow::Error::msg(format!("Cannot use central model with {:?}", options.comm)))
        }

        let player_set = (0..options.number_of_players).collect();
        let state = DemoState::new_with_players(BANDITS.into(), options.rounds, &player_set);
        let env = CentralEnvironment::new(state, env_communicator);

        Ok(Self{
            env, agents
        })


    }

    pub fn run_single_game(&mut self){
        std::thread::scope(|s|{
            s.spawn(||{
                self.env.run_with_scores().unwrap()
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
