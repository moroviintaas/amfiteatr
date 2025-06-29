use std::collections::HashMap;
use std::fs::File;
use std::sync::{Arc, mpsc};
use parking_lot::{Mutex, RwLock};
use amfiteatr_classic::agent::{LocalHistoryConversionToTensor, LocalHistoryInfoSet, ReplInfoSetAgentNum};
use amfiteatr_classic::{AsymmetricRewardTable, ClassicActionTensorRepresentation, SymmetricRewardTable};
use amfiteatr_classic::domain::{AgentNum, ClassicAction, ClassicGameDomainNumbered};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_core::agent::{AgentGen, AutomaticAgent, IdAgent, InformationSet, MultiEpisodeAutoAgent, Policy, PolicyAgent, StatefulAgent, TracingAgent, TracingAgentGen};
use amfiteatr_core::comm::{EnvironmentMpscPort, StdAgentEndpoint, StdEnvironmentEndpoint};
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::env::{AutoEnvironmentWithScores, ReseedEnvironment, ScoreEnvironment, TracingHashMapEnvironment};
use amfiteatr_core::error::{AmfiteatrError, CommunicationError};
use amfiteatr_core::reexport::nom::Parser;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::policy::{LearningNetworkPolicy, LearnSummary, PolicyDiscretePPO};
use crate::replicators::error::ReplError;
use crate::replicators::error::ReplError::OddAgentNumber;
use amfiteatr_classic::agent::ReplInfoSet;
use crate::replicators::epoch_description::{EpochDescription, EpochDescriptionMean, SessionDescription, SessionLearningSummaries};

pub type ReplDomain = ClassicGameDomainNumbered;

pub type PurePolicy = ClassicPureStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
pub type AgentPure = AgentGen<ReplDomain, PurePolicy, StdAgentEndpoint<ReplDomain>>;
pub type StaticBehavioralAgent = AgentGen<ReplDomain, ClassicMixedStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>, StdAgentEndpoint<ReplDomain>>;
pub type LearningAgent<
    P: LearningNetworkPolicy<ReplDomain, Summary=LearnSummary> + Policy<ReplDomain, InfoSetType=LocalHistoryInfoSet<AgentNum>>
        + TensorboardSupport<ReplDomain>
> = TracingAgentGen<ReplDomain, P, StdAgentEndpoint<ReplDomain>>;



//pub type PpoAgent = TracingAgentGen<ReplDomain, PolicyDiscretePPO<ReplDomain, LocalHistoryInfoSet<AgentNum>, LocalHistoryConversionToTensor, ClassicActionTensorRepresentation>, StdAgentEndpoint<ReplDomain>>;

pub type ModelEnvironment = TracingHashMapEnvironment<ReplDomain, PairingState<<ReplDomain as DomainParameters>::AgentId>, StdEnvironmentEndpoint<ReplDomain>>;


pub trait ReplicatorNetworkPolicy: LearningNetworkPolicy<ReplDomain, InfoSetType= LocalHistoryInfoSet<AgentNum>, Summary = LearnSummary> + TensorboardSupport<ReplDomain>{

}

impl <P: LearningNetworkPolicy<ReplDomain, InfoSetType= LocalHistoryInfoSet<AgentNum>, Summary = LearnSummary> + TensorboardSupport<ReplDomain>>
ReplicatorNetworkPolicy for P {}
pub struct ReplicatorModelBuilder<LP: ReplicatorNetworkPolicy>
{
    pure_hawks: Vec<Arc<Mutex<AgentPure>>>,
    pure_doves: Vec<Arc<Mutex<AgentPure>>>,
    mixed_agents: Vec<Arc<Mutex<StaticBehavioralAgent>>>,
    network_learning_agents: Vec<Arc<Mutex<LearningAgent<LP>>>>,
    communication_map: HashMap<AgentNum, StdEnvironmentEndpoint<ReplDomain>>,
    thread_pool: Option<rayon::ThreadPool>,
    reward_table: AsymmetricRewardTable<i64>,
    target_rounds: Option<usize>,
    tboard_writer: Option<tboard::EventWriter<File>>,
    //new_agent_index: u32,
}


impl <LP: ReplicatorNetworkPolicy>ReplicatorModelBuilder<LP>
//ReplicatorModelBuilder

{
    pub fn new() -> Self {
        Self{
            pure_hawks: Vec::new(),
            pure_doves: Vec::new(),
            mixed_agents: Vec::new(),
            network_learning_agents: Vec::new(),
            communication_map: HashMap::new(),
            thread_pool: None,
            reward_table: SymmetricRewardTable::new(2, 1, 4, 0).into(),
            target_rounds: None,
            tboard_writer: None
        }
    }

    /// How many times agent must involve in encounter in episode
    pub fn encounters_in_episode(mut self, encounters: usize) -> Self{
        self.target_rounds = Some(encounters);
        self
    }

    pub fn add_learning_agent(&mut self, agent_id: u32, policy: LP) -> Result<(), ReplError>{
        if self.communication_map.contains_key(&agent_id) {
            Err(ReplError::AgentDuplication(agent_id))
        } else{
            let info_set = <LP as Policy<ReplDomain>>::InfoSetType::create(agent_id, self.reward_table.clone());
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
            let agent = TracingAgentGen::new(info_set, agent_comm, policy);
            self.communication_map.insert(agent_id, env_comm);
            self.network_learning_agents.push(Arc::new(Mutex::new(agent)));
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
            self.pure_doves.push(Arc::new(Mutex::new(agent)));
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
            self.pure_hawks.push(Arc::new(Mutex::new(agent)));
            Ok(())
        }
    }

    pub fn build(self) -> Result<ReplicatorModel<LP>, ReplError>
    where <LP as Policy<ReplDomain>>::InfoSetType: Renew<ReplDomain, ()> + InformationSet<ReplDomain>{
    //pub fn build(self) -> Result<ReplicatorModel, ReplError>{
        let number_of_agents = self.network_learning_agents.len() + self.mixed_agents.len()
            +self.pure_doves.len() + self.pure_hawks.len();
        if number_of_agents %2 != 0{
            return Err(OddAgentNumber(number_of_agents))
        }

        if self.target_rounds.is_none(){
            return Err(ReplError::MissingParameter("Number of encounters in episode (use method `encounters_in_episode`)".to_string()))
        }
        let rounds = self.target_rounds.unwrap();
        let env_state = PairingState::new_even(number_of_agents, rounds, self.reward_table)
            .map_err(|e| AmfiteatrError::Game { source: e })?;
        let environment = ModelEnvironment::new(env_state, self.communication_map);

        Ok(ReplicatorModel::_create(
            environment,
            self.pure_hawks,
            self.pure_doves,
            self.mixed_agents,
            self.network_learning_agents,
            self.tboard_writer,
            self.thread_pool,
        ))
    }



}

pub struct ReplicatorModel<LP: ReplicatorNetworkPolicy> {
    //pub struct ReplicatorModel{
    tboard_writer: Option<tboard::EventWriter<File>>,
    thread_pool: Option<rayon::ThreadPool>,

    pure_hawks: Vec<Arc<Mutex<AgentPure>>>,
    pure_doves: Vec<Arc<Mutex<AgentPure>>>,
    mixed_agents: Vec<Arc<Mutex<StaticBehavioralAgent>>>,
    network_learning_agents: Vec<Arc<Mutex<LearningAgent<LP>>>>,
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

impl<LP: ReplicatorNetworkPolicy> ReplicatorModel<LP> {
//impl ReplicatorModel{

    //num_learning_agents: usize, num_pure_hawks: usize, num_pure_doves: usize

    pub(crate) fn _create(
        environment : ModelEnvironment,
        pure_hawks: Vec<Arc<Mutex<AgentPure>>>,
        pure_doves: Vec<Arc<Mutex<AgentPure>>>,
        mixed_agents: Vec<Arc<Mutex<StaticBehavioralAgent>>>,
        learning_agents: Vec<Arc<Mutex<LearningAgent<LP>>>>,
        tboard_writer: Option<tboard::EventWriter<File>>,
        thread_pool: Option<rayon::ThreadPool>,
    ) -> Self{
        Self{
            tboard_writer,
            thread_pool,
            pure_hawks,
            pure_doves,
            mixed_agents,
            network_learning_agents: learning_agents,
            environment,
        }
    }



    pub fn run_episode(&mut self) -> Result<(), AmfiteatrError<ReplDomain>>{
        match &mut self.thread_pool{
            Some(pool) => {
                todo!()
            },
            None => {
                std::thread::scope(|s|{
                    s.spawn(||{
                        self.environment.reseed(()).unwrap();
                        self.environment.run_with_scores().unwrap();
                    });

                    for hawk in &self.pure_hawks{
                        let agent = hawk.clone();
                        s.spawn(move ||{
                            let mut hawk_guard = agent.lock();
                            hawk_guard.run_episode(()).unwrap();

                        });
                    }

                    for dove in &self.pure_doves{
                        let agent = dove.clone();
                        s.spawn(move ||{
                            let mut guard = agent.lock();
                            guard.run_episode(()).unwrap();

                        });
                    }

                    for nla in &self.network_learning_agents{
                        let agent = nla.clone();
                        s.spawn(move ||{
                            let mut guard = agent.lock();
                            guard.run_episode(()).unwrap();

                        });
                    }

                    for mixed_agent in &self.mixed_agents{
                        let agent = mixed_agent.clone();
                        s.spawn(move ||{
                            let mut guard = agent.lock();
                            guard.run_episode(()).unwrap();

                        });
                    }

                });

            }
        }
        Ok(())
    }

    pub fn clear_episodes_trajectories(&mut self) -> Result<(), AmfiteatrError<ReplDomain>>{
        for agent in &self.network_learning_agents{
            let mut guard = agent.as_ref().lock();
            guard.clear_episodes()?;

        }
        Ok(())
    }


    fn create_empty_description(&self, capacity: usize) -> EpochDescription{

        let mut description = EpochDescription::default();
        for agent in self.network_learning_agents.iter(){
            let guard = agent.lock();
            description.scores.insert(guard.id().clone(), Vec::with_capacity(capacity));
            description.network_learning_hawk_moves.insert(guard.id().clone(), Vec::with_capacity(capacity));
            description.network_learning_dove_moves.insert(guard.id().clone(), Vec::with_capacity(capacity));
        }
        for agent in self.pure_doves.iter(){
            let guard = agent.lock();
            description.scores.insert(guard.id().clone(), Vec::with_capacity(capacity));
        }
        for agent in self.pure_hawks.iter(){
            let guard = agent.lock();
            description.scores.insert(guard.id().clone(), Vec::with_capacity(capacity));
        }

        description
    }

    fn update_epoch_description_with_episode(&self, description: &mut EpochDescription){
        for agent in self.network_learning_agents.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_penalty_score_of_player(guard.id()))
            }
            if let Some(a) = description.network_learning_dove_moves.get_mut(guard.id()){
                let dove_action = guard.trajectory().iter()
                    .filter(|s| *s.action() == ClassicAction::Down)
                    .count();
                a.push(dove_action);
            }
            if let Some(a) = description.network_learning_hawk_moves.get_mut(guard.id()){
                let hawk_action = guard.trajectory().iter()
                    .filter(|s| *s.action() == ClassicAction::Up)
                    .count();
                a.push(hawk_action);
            }
        }
        for agent in self.pure_doves.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_penalty_score_of_player(guard.id()))
            }
        }
        for agent in self.pure_hawks.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_penalty_score_of_player(guard.id()))
            }
        }
        for agent in self.mixed_agents.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_penalty_score_of_player(guard.id()))
            }
        }
    }

    pub fn train_network_agents_parallel(&mut self) -> Result<HashMap<AgentNum, <LP as LearningNetworkPolicy<ReplDomain>>::Summary>, AmfiteatrError<ReplDomain>>{
        match self.thread_pool{
            Some(_) => todo!(),
            None => {

                let summaries = Arc::new(Mutex::new(HashMap::new()));
                let summaries2 = summaries.clone();
                std::thread::scope(|s|{
                    let (tx, rx) = mpsc::channel();
                    s.spawn(move ||{
                        let mut summaries_guard = summaries.lock();
                        while let Ok((id, summary)) = rx.recv(){
                            summaries_guard.insert(id, summary);
                        }
                    });

                    for agent in &self.network_learning_agents{
                        let agentc = agent.clone();
                        let txc = tx.clone();
                        s.spawn(move || {
                            let mut guard = agentc.lock();
                            let trajectories = guard.take_episodes();
                            let summary = guard.policy_mut().train_on_trajectories_env_reward(&trajectories[..])?;
                            txc.send((guard.info_set().agent_id().clone(), summary )).map_err(|e|{
                               AmfiteatrError::Communication {
                                   source: CommunicationError::SendError(guard.info_set().agent_id().clone(), e.to_string())
                               }
                            })?;
                            Ok::<_, AmfiteatrError<ReplDomain>>(())
                        });
                    }
                    drop(tx);

                });
                let m = Arc::try_unwrap(summaries2).map_err(|e|{
                    AmfiteatrError::Lock { description: "Failed unwrapping Arc".to_string(), object: "Summaries in training".to_string() }
                })?;
                let summaries = m.into_inner();
                Ok(summaries)
            }
        }

    }
    pub fn run_epoch(&mut self, episodes: usize) -> Result<EpochDescription, AmfiteatrError<ReplDomain>>{

        let mut description = self.create_empty_description(episodes);
        self.clear_episodes_trajectories();
        for i in 0..episodes{
            log::debug!("Running episode {}", i);
            self.run_episode();
            self.update_epoch_description_with_episode(&mut description);
        }
        Ok(description)
        //let description_mean = description.mean();



    }

    pub fn run_training_session(&mut self, episodes: usize, epochs: usize) -> Result<(Vec<EpochDescriptionMean>, Vec<HashMap<AgentNum, LearnSummary>>),AmfiteatrError<ReplDomain>>{
    //pub fn run_training_session(&mut self, episodes: usize, epochs: usize) -> Result<HashMap<AgentNum, (SessionDescription, Option<SessionLearningSummaries>)>, AmfiteatrError<ReplDomain>>{
        /*let mut summaries = self.network_learning_agents.iter()
            .zip(self.

         */
        let (mut epoch_results, mut epoch_learn_results) = (Vec::with_capacity(epochs), Vec::with_capacity(epochs));
        for e in 0..epochs{
            let description = self.run_epoch(episodes)?.mean();
            let learn_summary = self.train_network_agents_parallel()?;

            epoch_results.push(description);
            epoch_learn_results.push(learn_summary);
        }
        Ok((epoch_results, epoch_learn_results))
    }
}