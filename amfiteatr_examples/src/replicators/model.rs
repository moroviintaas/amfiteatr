use std::collections::HashMap;
use std::fs::File;
use std::sync::RwLock;
use amfiteatr_classic::agent::{LocalHistoryInfoSet, ReplInfoSetAgentNum};
use amfiteatr_classic::{AsymmetricRewardTable, SymmetricRewardTable};
use amfiteatr_classic::domain::{AgentNum, ClassicAction, ClassicGameDomainNumbered};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_core::agent::{AgentGen, Policy, TracingAgentGen};
use amfiteatr_core::comm::{EnvironmentMpscPort, StdAgentEndpoint, StdEnvironmentEndpoint};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::env::TracingHashMapEnvironment;
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::reexport::nom::Parser;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::policy::LearningNetworkPolicy;
use crate::replicators::error::ReplError;
use crate::replicators::error::ReplError::OddAgentNumber;
use amfiteatr_classic::agent::ReplInfoSet;

pub type ReplDomain = ClassicGameDomainNumbered;

pub type PurePolicy = ClassicPureStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
pub type AgentPure = AgentGen<ReplDomain, PurePolicy, StdAgentEndpoint<ReplDomain>>;
pub type StaticBehavioralAgent = AgentGen<ReplDomain, ClassicMixedStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>, StdAgentEndpoint<ReplDomain>>;
pub type LearningAgent<
    P: LearningNetworkPolicy<ReplDomain, InfoSetType=LocalHistoryInfoSet<ReplDomain>>
        + TensorboardSupport<ReplDomain>> =
    TracingAgentGen<ReplDomain, P, StdAgentEndpoint<ReplDomain>>;

pub type ModelEnvironment = TracingHashMapEnvironment<ReplDomain, PairingState<<ReplDomain as DomainParameters>::AgentId>, StdEnvironmentEndpoint<ReplDomain>>;


pub trait ReplicatorNetworkPolicy: LearningNetworkPolicy<ReplDomain, InfoSetType: ReplInfoSetAgentNum> + TensorboardSupport<ReplDomain>{

}

impl <P: LearningNetworkPolicy<ReplDomain, InfoSetType: ReplInfoSetAgentNum> + TensorboardSupport<ReplDomain>> ReplicatorNetworkPolicy for P {}
pub struct ReplicatorModelBuilder<LP: ReplicatorNetworkPolicy>{
    pure_hawks: Vec<RwLock<AgentPure>>,
    pure_doves: Vec<RwLock<AgentPure>>,
    mixed_agents: Vec<RwLock<StaticBehavioralAgent>>,
    learning_agents: Vec<RwLock<LearningAgent<LP>>>,
    communication_map: HashMap<AgentNum, StdEnvironmentEndpoint<ReplDomain>>,
    thread_pool: Option<rayon::ThreadPool>,
    reward_table: AsymmetricRewardTable<i64>,
    target_rounds: usize,
    tboard_writer: Option<tboard::EventWriter<File>>,
    //new_agent_index: u32,
}


impl<LP: ReplicatorNetworkPolicy>ReplicatorModelBuilder<LP> {
    pub fn new() -> Self {
        Self{
            pure_hawks: Vec::new(),
            pure_doves: Vec::new(),
            mixed_agents: Vec::new(),
            learning_agents: Vec::new(),
            communication_map: HashMap::new(),
            thread_pool: None,
            reward_table: SymmetricRewardTable::new(2, 1, 4, 0).into(),
            target_rounds: 100,
            tboard_writer: None
        }
    }
    pub fn add_learning_agent(&mut self, agent_id: u32, policy: LP) -> Result<(), ReplError>{
        if self.communication_map.contains_key(&agent_id) {
            Err(ReplError::AgentDuplication(agent_id))
        } else{
            let info_set = <LP as Policy<ReplDomain>>::InfoSetType::create(agent_id, self.reward_table.clone());
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
            let agent = TracingAgentGen::new(info_set, agent_comm, policy);
            self.communication_map.insert(agent_id, env_comm);
            self.learning_agents.push(RwLock::new(agent));
            Ok(())
        }
    }
    pub fn with_learning_agent(mut self, agent_id: u32, policy: LP) -> Result<Self, ReplError>{

        self.add_learning_agent(agent_id, policy)?;
        Ok(self)
    }

    pub fn add_dove_agent(&mut self, agent_id: u32) -> Result<(), ReplError>{
        if self.communication_map.contains_key(&agent_id) {
            Err(ReplError::AgentDuplication(agent_id))
        } else{
            let policy = PurePolicy::new(ClassicAction::Down);
            let info_set = LocalHistoryInfoSet::new(agent_id, self.reward_table.clone());
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
            let agent = AgentGen::new(info_set, agent_comm, policy);
            self.communication_map.insert(agent_id, env_comm);
            self.pure_doves.push(RwLock::new(agent));
            Ok(())
        }
    }

    pub fn add_hawk_agent(&mut self, agent_id: u32) -> Result<(), ReplError>{
        if self.communication_map.contains_key(&agent_id) {
            Err(ReplError::AgentDuplication(agent_id))
        } else{
            let policy = PurePolicy::new(ClassicAction::Up);
            let info_set = LocalHistoryInfoSet::new(agent_id, self.reward_table.clone());
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
            let agent = AgentGen::new(info_set, agent_comm, policy);
            self.communication_map.insert(agent_id, env_comm);
            self.pure_hawks.push(RwLock::new(agent));
            Ok(())
        }
    }

    pub fn build(self) -> Result<ReplicatorModel<LP>, ReplError>{
        let number_of_agents = self.learning_agents.len() + self.mixed_agents.len()
            +self.pure_doves.len() + self.pure_hawks.len();
        if number_of_agents %2 != 0{
            return Err(OddAgentNumber(number_of_agents))
        }

        let env_state = PairingState::new_even(number_of_agents, self.target_rounds, self.reward_table)
            .map_err(|e| AmfiteatrError::Game { source: e })?;
        let environment = ModelEnvironment::new(env_state, self.communication_map);

        Ok(ReplicatorModel::_create(
            environment,
            self.pure_hawks,
            self.pure_doves,
            self.mixed_agents,
            self.learning_agents,
            self.tboard_writer,
            self.thread_pool,
        ))
    }



}

pub struct ReplicatorModel<LP: ReplicatorNetworkPolicy>{
    tboard_writer: Option<tboard::EventWriter<File>>,
    thread_pool: Option<rayon::ThreadPool>,

    pure_hawks: Vec<RwLock<AgentPure>>,
    pure_doves: Vec<RwLock<AgentPure>>,
    mixed_agents: Vec<RwLock<StaticBehavioralAgent>>,
    learning_agents: Vec<RwLock<LearningAgent<LP>>>,
    environment: ModelEnvironment,


    /*
    min_learning_index: u32,
    max_learning_index: u32,

    min_hawk_index: u32,
    max_hawk_index: u32,

    min_doves_index: u32,
    max_doves_index: u32,

    min_

     */

}

impl<LP: ReplicatorNetworkPolicy> ReplicatorModel<LP>{

    //num_learning_agents: usize, num_pure_hawks: usize, num_pure_doves: usize

    pub(crate) fn _create(
        environment : ModelEnvironment,
        pure_hawks: Vec<RwLock<AgentPure>>,
        pure_doves: Vec<RwLock<AgentPure>>,
        mixed_agents: Vec<RwLock<StaticBehavioralAgent>>,
        learning_agents: Vec<RwLock<LearningAgent<LP>>>,
        tboard_writer: Option<tboard::EventWriter<File>>,
        thread_pool: Option<rayon::ThreadPool>,
    ) -> Self{
        Self{
            tboard_writer,
            thread_pool,
            pure_hawks,
            pure_doves,
            mixed_agents,
            learning_agents,
            environment,
        }
    }
    /*
    pub fn new() -> Self{
        let environment_ports = HashMap::new();

        let env_state = PairingState::new_even(0, self.t, ())

        for index in 0..num_learning_agents as u32 {
            let (env_point, agent_point) = StdEnvironmentEndpoint::new_pair();

        }



    }

     */
}