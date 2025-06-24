use std::fs::File;
use std::sync::RwLock;
use amfiteatr_classic::agent::LocalHistoryInfoSet;
use amfiteatr_classic::domain::{AgentNum, ClassicGameDomainNumbered};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_core::agent::{AgentGen, TracingAgentGen};
use amfiteatr_core::comm::{StdAgentEndpoint, StdEnvironmentEndpoint};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::env::TracingHashMapEnvironment;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::policy::LearningNetworkPolicy;

pub type ReplDomain = ClassicGameDomainNumbered;

pub type PurePolicy = ClassicPureStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
pub type AgentPure = AgentGen<ReplDomain, PurePolicy, StdAgentEndpoint<ReplDomain>>;
pub type StaticBehavioralAgent = AgentGen<ReplDomain, ClassicMixedStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>, StdAgentEndpoint<ReplDomain>>;
pub type LearningAgent<
    P: LearningNetworkPolicy<ReplDomain, InfoSetType=LocalHistoryInfoSet<ReplDomain>>
        + TensorboardSupport<ReplDomain>> =
    TracingAgentGen<ReplDomain, P, StdAgentEndpoint<ReplDomain>>;



pub trait ReplicatorNetworkPolicy: LearningNetworkPolicy<ReplDomain> + TensorboardSupport<ReplDomain>{

}

impl <P: LearningNetworkPolicy<ReplDomain> + TensorboardSupport<ReplDomain>> ReplicatorNetworkPolicy for P {}

pub struct ReplicatorModel<LP: ReplicatorNetworkPolicy>{
    tboard_writer: Option<tboard::EventWriter<File>>,
    thread_pool: Option<rayon::ThreadPool>,

    pure_hawks: Vec<RwLock<AgentPure>>,
    pure_doves: Vec<RwLock<AgentPure>>,
    mixed_agents: Vec<RwLock<StaticBehavioralAgent>>,
    learning_agents: Vec<RwLock<LP>>,
    environment: TracingHashMapEnvironment<ReplDomain, PairingState<<ReplDomain as DomainParameters>::AgentId>, StdEnvironmentEndpoint<ReplDomain>>,


}

impl<LP: ReplicatorNetworkPolicy> ReplicatorModel<LP>{
    pub fn new(num_learning_agents: usize, num_pure_hawks: usize, num_pure_doves: usize) -> Self{
        todo!()

    }
}