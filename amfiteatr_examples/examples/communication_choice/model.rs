use std::cmp::min;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use amfiteatr_core::agent::{AgentGen, AutomaticAgent, InformationSet, Policy, RandomPolicy, ReseedAgent};
use amfiteatr_core::comm::{AgentEndpoint, AgentMpscAdapter, BidirectionalEndpoint, EnvironmentAdapter, EnvironmentEndpoint, EnvironmentMpscPort, StdEnvironmentEndpoint};
use amfiteatr_core::scheme::{AgentMessage, Scheme, EnvironmentMessage, Renew};
use amfiteatr_core::env::{AutoEnvironmentWithScores, BasicEnvironment, HashMapEnvironment, ReseedEnvironment, RoundRobinUniversalEnvironment, SequentialGameState};
use amfiteatr_core::error::{AmfiteatrError, CommunicationError};
use amfiteatr_examples::expensive_update::agent::ExpensiveUpdateInformationSet;
use amfiteatr_examples::expensive_update::scheme::ExpensiveUpdateScheme;
use amfiteatr_examples::expensive_update::env::ExpensiveUpdateState;
use amfiteatr_net_ext::{SchemeCommA512, SchemeCommE512};
use amfiteatr_net_ext::tcp::PairedTcpEnvironmentEndpoint;
use crate::options::{CCOptions, CommunicationMedium};


pub type EUD = ExpensiveUpdateScheme;
pub type EUS = ExpensiveUpdateState;
pub type EUSI = ExpensiveUpdateInformationSet;


//type MapEnvironment = HashMapEnvironment<EUD, EUS, DomainCommE512<EUD>>;
type MapEnvironment<C: EnvironmentEndpoint<EUD> + Send> = HashMapEnvironment<EUD, EUS, C>;
type CentralEnvironment = BasicEnvironment<EUD, EUS, EnvironmentMpscPort<EUD>>;

type MappedAgent<C> = AgentGen<EUD, RandomPolicy<EUD, EUSI>, C>;
type CAgent = AgentGen<EUD, RandomPolicy<EUD, EUSI>, AgentMpscAdapter<EUD>>;
pub const BANDITS: [(f32, f32);3] = [(1.0, 3.0), (5.6, 6.7), (0.1, 9.0)];

pub struct MapModel<CE:EnvironmentEndpoint<EUD>, CA:  AgentEndpoint<EUD>> {

    env: MapEnvironment<CE>,
    agents: Vec<Arc<Mutex<MappedAgent<CA>>>>
}

impl MapModel<
    Box<dyn BidirectionalEndpoint<Error=CommunicationError<EUD>, InwardType=AgentMessage<EUD>, OutwardType=EnvironmentMessage<EUD>> + Send>,
    Box<dyn BidirectionalEndpoint<Error=CommunicationError<EUD>, InwardType=EnvironmentMessage<EUD>, OutwardType=AgentMessage<EUD>> + Send>>{
    pub fn new_boxing(
        options: &CCOptions) -> Result<Self, anyhow::Error>{

        let tcp_agents = min(options.number_of_dynamic_tcp_agents, options.number_of_players);

        let mut agents: Vec<Arc<Mutex<AgentGen<EUD, RandomPolicy<EUD, EUSI>, Box<dyn BidirectionalEndpoint<Error=CommunicationError<EUD>, InwardType=EnvironmentMessage<EUD>, OutwardType=AgentMessage<EUD>> + Send>>>>> = Vec::new();
        let mut env_comms: HashMap<u64, Box<dyn BidirectionalEndpoint<Error=CommunicationError<EUD>, InwardType=AgentMessage<EUD>, OutwardType=EnvironmentMessage<EUD>> + Send>> = HashMap::new();

        let random_policy = RandomPolicy::<EUD, EUSI>::new();


        match options.comm{
            CommunicationMedium::Dynamic => {


                let agents_vec: Vec<u64> = (0..tcp_agents).collect();
                let (mapped_env_comms_tcp, mapped_agent_comms_tcp) = PairedTcpEnvironmentEndpoint::<EUD, 512>::create_local_net(28000, agents_vec.iter()).unwrap();

                /*env_comms = mapped_env_comms_tcp.into_iter().map(|(i, ep)|{
                    (i, Box::new(ep))
                }).collect();

                 */
                for (i,c) in mapped_env_comms_tcp.into_iter(){
                    env_comms.insert(i, Box::new(c));
                }

                for (id, a_comm) in mapped_agent_comms_tcp {
                    let info_set = EUSI::new(id);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, Box::new(a_comm), random_policy.clone()))))
                }

                for id in (tcp_agents..options.number_of_players){
                    let (e,a) = StdEnvironmentEndpoint::new_pair();
                    env_comms.insert(id, Box::new(e));
                    let info_set = EUSI::new(id);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, Box::new(a), random_policy.clone()))))
                }


            }
            /*
            CommunicationMedium::StaticTcp => {
                let agents_vec: Vec<u64> = (0..options.number_of_players).collect();
                let (mapped_env_comms, mapped_agent_comms) = PairedTcpEnvironmentEndpoint::create_local_net(28000, agents_vec.iter()).unwrap();
                env_comms = mapped_env_comms.into_iter().map(|(i, ep)|{
                    (i, DomainCommE512::Tcp(ep))
                }).collect();

                for (id, a_comm) in mapped_agent_comms{
                    let info_set = EUSI::new(id);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, DomainCommA512::Tcp(a_comm), random_policy.clone()))))
                }


            }

             */
            m => {
                return Err(anyhow::Error::msg(format!("Comm type not implemented for 1-1 dynamic model: {:?}", m)))
            }
        }

        //let player_set = (0..options.number_of_players).collect();
        let state = EUS::new(options.rounds, options.number_of_players, options.small_update_cost_per_agent, options.big_update_cost_per_agent, options.big_update_cost_flat);
        let env = MapEnvironment::new(state, env_comms);

        Ok(Self{
            env,
            agents,
        })




    }
}

impl MapModel<SchemeCommE512<EUD>, SchemeCommA512<EUD>>{
    pub fn new(
        options: &CCOptions) -> Result<Self, anyhow::Error>{

        let number_of_bandits = BANDITS.len();

        let mut agents = Vec::new();
        let mut env_comms = HashMap::new();

        let random_policy = RandomPolicy::<EUD, EUSI>::new();
        //let policy = ExpensiveUpdatePolicy::


        match options.comm{
            CommunicationMedium::StaticMpsc => {
                for id in 0..options.number_of_players{
                    let (e,a) = StdEnvironmentEndpoint::new_pair();
                    env_comms.insert(id, SchemeCommE512::StdSync(e));
                    let info_set = EUSI::new(id);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, SchemeCommA512::StdSync(a), random_policy.clone()))))
                }
            }
            CommunicationMedium::StaticTcp => {
                let agents_vec: Vec<u64> = (0..options.number_of_players).collect();
                let (mapped_env_comms, mapped_agent_comms) = PairedTcpEnvironmentEndpoint::create_local_net(28000, agents_vec.iter()).unwrap();
                env_comms = mapped_env_comms.into_iter().map(|(i, ep)|{
                    (i, SchemeCommE512::Tcp(ep))
                }).collect();

                for (id, a_comm) in mapped_agent_comms{
                    let info_set = EUSI::new(id);
                    agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, SchemeCommA512::Tcp(a_comm), random_policy.clone()))))
                }


            }
            m => {
                return Err(anyhow::Error::msg(format!("Comm type not implemented for 1-1 model: {:?}", m)))
            }
        }

        //let player_set = (0..options.number_of_players).collect();
        let state = EUS::new(options.rounds, options.number_of_players, options.small_update_cost_per_agent, options.big_update_cost_per_agent, options.big_update_cost_flat);
        let env = MapEnvironment::new(state, env_comms);

        Ok(Self{
            env,
            agents,
        })




    }
}

impl<
    CE:EnvironmentEndpoint<EUD> + Send,
    CA:  AgentEndpoint<EUD> + Send
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

    pub fn run_several_games(&mut self, number_of_games: u64){


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
        //let number_of_bandits = BANDITS.len();

        let mut agents = Vec::new();

        let mut env_communicator = EnvironmentMpscPort::new();
        let random_policy = RandomPolicy::<EUD, EUSI>::new();

        if options.comm == CommunicationMedium::CentralMpsc{

            for id in 0..options.number_of_players{
                let agent_comm = env_communicator.register_agent(id)?;
                let info_set = EUSI::new(id);
                agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, agent_comm, random_policy.clone()))))
            }



        }
        else {
            return Err(anyhow::Error::msg(format!("Cannot use central model with {:?}", options.comm)))
        }

        //let player_set = (0..options.number_of_players).collect();
        let state = EUS::new(options.rounds, options.number_of_players, options.small_update_cost_per_agent, options.big_update_cost_per_agent, options.big_update_cost_flat);
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

    pub fn run_several_games(&mut self, number_of_games: u64){


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

pub struct NoCommModel{
    env_state: EUS,
    agent_info_sets: HashMap<<EUD as Scheme>::AgentId ,EUSI>,
    agent_policies: HashMap<<EUD as Scheme>::AgentId, RandomPolicy<EUD, EUSI>>,
}

impl NoCommModel{

    pub fn new(options: &CCOptions) -> Result<Self, anyhow::Error>{
        let mut agent_info_sets = HashMap::new();
        let env_state = EUS::new(options.rounds, options.number_of_players, options.small_update_cost_per_agent, options.big_update_cost_per_agent, options.big_update_cost_flat);
        let mut agent_policies = HashMap::new();
        for id in 0..options.number_of_players{
            agent_policies.insert(id, RandomPolicy::new());
            agent_info_sets.insert(id, EUSI::new(id));
        }
        Ok(Self{
            env_state,
            agent_info_sets,
            agent_policies
        })
    }

    pub fn run_single_game(&mut self){

        while !self.env_state.is_finished(){
            let current_agent = self.env_state.current_player().unwrap();
            let action = self.agent_policies.get(&current_agent).unwrap()
                .select_action(self.agent_info_sets.get(&current_agent).unwrap()).unwrap();
            let updates  = self.env_state.forward(current_agent, action).unwrap();
            for (id, update) in updates{
                self.agent_info_sets.get_mut(&id).unwrap().update(update).unwrap();
            }
        }
    }

    pub fn run_several_games(&mut self, number_of_games: u64){
        for i in 0..number_of_games{
            log::info!("Playing game {i}");
            self.env_state.renew_from(()).unwrap();
            for agent in self.agent_info_sets.values_mut(){
                agent.renew_from(()).unwrap();
            }
            self.run_single_game();
        }
    }

}
