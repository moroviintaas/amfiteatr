use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use parking_lot::Mutex;
use amfiteatr_classic::agent::LocalHistoryInfoSet;
use amfiteatr_classic::{AsymmetricRewardTable, SymmetricRewardTable};
use amfiteatr_classic::domain::{AgentNum, ClassicAction, ClassicGameDomainNumbered};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_core::agent::{AgentGen, IdAgent, InformationSet, MultiEpisodeAutoAgent, Policy, PolicyAgent, StatefulAgent, TracingAgentGen};
use amfiteatr_core::comm::{StdAgentEndpoint, StdEnvironmentEndpoint};
use amfiteatr_core::scheme::{Scheme, Renew};
use amfiteatr_core::env::{AutoEnvironmentWithScores, ReseedEnvironment, ScoreEnvironment, StatefulEnvironment, TracingHashMapEnvironment};
use amfiteatr_core::error::{AmfiteatrError, CommunicationError};
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::policy::{LearningNetworkPolicy, LearningNetworkPolicyGeneric, LearnSummary};
use crate::replicators::error::ReplError;
use crate::replicators::error::ReplError::OddAgentNumber;
use amfiteatr_classic::agent::ReplInfoSet;
use crate::replicators::epoch_description::{EpochDescription, EpochDescriptionMean};
use crate::replicators::options::ReplicatorOptions;

pub type ReplDomain = ClassicGameDomainNumbered;

pub type PurePolicy = ClassicPureStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>;
pub type AgentPure = AgentGen<ReplDomain, PurePolicy, StdAgentEndpoint<ReplDomain>>;
pub type StaticBehavioralAgent = AgentGen<ReplDomain, ClassicMixedStrategy<AgentNum, LocalHistoryInfoSet<AgentNum>>, StdAgentEndpoint<ReplDomain>>;
pub type LearningAgent<P> = TracingAgentGen<ReplDomain, P, StdAgentEndpoint<ReplDomain>>;



//pub type PpoAgent = TracingAgentGen<ReplDomain, PolicyDiscretePPO<ReplDomain, LocalHistoryInfoSet<AgentNum>, LocalHistoryConversionToTensor, ClassicActionTensorRepresentation>, StdAgentEndpoint<ReplDomain>>;

pub type ModelEnvironment = TracingHashMapEnvironment<ReplDomain, PairingState<<ReplDomain as Scheme>::AgentId>, StdEnvironmentEndpoint<ReplDomain>>;


pub trait ReplicatorNetworkPolicy: LearningNetworkPolicy<ReplDomain, InfoSetType= LocalHistoryInfoSet<AgentNum>> + TensorboardSupport<ReplDomain>{

}

impl <P: LearningNetworkPolicy<ReplDomain, InfoSetType= LocalHistoryInfoSet<AgentNum>> + TensorboardSupport<ReplDomain>>
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

    pub fn add_mixed_agent(&mut self, agent_id: u32, probability: f64) -> Result<(), ReplError>{
        if self.communication_map.contains_key(&agent_id) {
            Err(ReplError::AgentDuplication(agent_id))
        } else{
            let policy = ClassicMixedStrategy::new_checked(probability)
                .map_err(|_e| ReplError::PolicyBuilderError("Mixed probability not in range (0.0, 1.0)".to_string()))?;
            let info_set = LocalHistoryInfoSet::new(agent_id, self.reward_table.clone());
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
            let agent = AgentGen::new(info_set, agent_comm, policy);
            self.communication_map.insert(agent_id, env_comm);
            self.mixed_agents.push(Arc::new(Mutex::new(agent)));
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
    tboard_writer_hawks: Option<tboard::EventWriter<File>>,
    tboard_writer_doves: Option<tboard::EventWriter<File>>,
    tboard_writer_mixes: Option<tboard::EventWriter<File>>,
    tboard_writer_learners: Option<tboard::EventWriter<File>>,

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
            tboard_writer_hawks: None,
            tboard_writer_doves: None,
            tboard_writer_mixes: None,
            tboard_writer_learners: None,
            thread_pool,
            pure_hawks,
            pure_doves,
            mixed_agents,
            network_learning_agents: learning_agents,
            environment,
        }
    }

    pub fn register_tboard_writers_from_program_args(&mut self, options: &ReplicatorOptions) -> Result<(), ReplError> {
        if let Some(tboard_path) = &options.tboard{
            self.tboard_writer = Some(tboard::EventWriter::create(tboard_path)
                .map_err(|e| ReplError::Amfiteatr(AmfiteatrError::TboardFlattened {
                    context: "Registering generic tboard writer".to_string(),
                    error: e.to_string() }))?);

        }
        if let Some(tboard_agent_base_path) = &options.agent_tboard{
            let path_hawk: PathBuf = [tboard_agent_base_path.as_ref(), std::path::Path::new(&"hawks")].iter().collect();
            self.tboard_writer_hawks = Some(tboard::EventWriter::create(&path_hawk)
                .map_err(|e| ReplError::Amfiteatr(AmfiteatrError::TboardFlattened {
                    context: "Registering hawk tboard writer".to_string(),
                    error: e.to_string() }))?);

            let path_dove: PathBuf = [tboard_agent_base_path.as_ref(), std::path::Path::new(&"doves")].iter().collect();
            self.tboard_writer_doves = Some(tboard::EventWriter::create(&path_dove)
                .map_err(|e| ReplError::Amfiteatr(AmfiteatrError::TboardFlattened {
                    context: "Registering dove tboard dove".to_string(),
                    error: e.to_string() }))?);
            let path_mixes: PathBuf = [tboard_agent_base_path.as_ref(), std::path::Path::new(&"mixes")].iter().collect();
            self.tboard_writer_mixes = Some(tboard::EventWriter::create(&path_mixes)
                .map_err(|e| ReplError::Amfiteatr(AmfiteatrError::TboardFlattened {
                    context: "Registering mixes tboard writer".to_string(),
                    error: e.to_string() }))?);
            let path_learners: PathBuf = [tboard_agent_base_path.as_ref(), std::path::Path::new(&"learners")].iter().collect();
            self.tboard_writer_learners = Some(tboard::EventWriter::create(&path_learners)
                .map_err(|e| ReplError::Amfiteatr(AmfiteatrError::TboardFlattened {
                    context: "Registering learners tboard writer".to_string(),
                    error: e.to_string() }))?);
        }

        Ok(())
    }


    pub fn run_episode(&mut self) -> Result<(), AmfiteatrError<ReplDomain>>{
        match &mut self.thread_pool{
            Some(_pool) => {
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
        for agent in self.mixed_agents.iter(){
            let guard = agent.lock();
            description.scores.insert(guard.id().clone(), Vec::with_capacity(capacity));
        }

        description
    }

    /*
    fn set_epoch_description_hawk_and_doves_with_episode(&self, description: &mut EpochDescription){
        for agent in self.network_learning_agents.iter(){
            let guard = agent.lock();
            if let Some(a) = description.network_learning_dove_moves.get_mut(guard.id()){
                a.clear();
                a.extend(guard.episodes().iter()
                    .map(|t|
                        t.iter().filter(|s| *s.action() == ClassicAction::Down ).count()
                    )
                );

            }
            if let Some(a) = description.network_learning_hawk_moves.get_mut(guard.id()){
                a.clear();
                a.extend(guard.episodes().iter()
                    .map(|t|
                        t.iter().filter(|s| *s.action() == ClassicAction::Up ).count()
                    )
                );

            }
        }

    }

     */

    fn update_epoch_description_score_with_episode(&self, description: &mut EpochDescription){
        for agent in self.network_learning_agents.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_score_of_player(guard.id()))
            }

            //debug!("Agent {} actions : {:?}", guard.id(), guard.trajectory().iter().map(|t|*t.action()).collect::<Vec<_>>());
            if let Some(a) = description.network_learning_dove_moves.get_mut(guard.id()){

                if let Some(trajectory) = guard.episodes().last(){
                    let dove_actions = trajectory.iter()
                        .filter(|s| *s.action() == ClassicAction::Down)
                        .count();
                    a.push(dove_actions);
                }


            }
            if let Some(a) = description.network_learning_hawk_moves.get_mut(guard.id()){


                if let Some(trajectory) = guard.episodes().last(){
                    let hawk_actions = trajectory.iter()
                        .filter(|s| *s.action() == ClassicAction::Up)
                        .count();
                    a.push(hawk_actions);
                }
            }


        }
        for agent in self.pure_doves.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_score_of_player(guard.id()))
            }
        }
        for agent in self.pure_hawks.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_score_of_player(guard.id()))
            }
        }
        for agent in self.mixed_agents.iter(){
            let guard = agent.lock();
            if let Some(a) = description.scores.get_mut(guard.id()){
                a.push(self.environment.actual_score_of_player(guard.id()))
            }
        }
    }

    pub fn train_network_agents_parallel(&mut self) -> Result<HashMap<AgentNum, <LP as LearningNetworkPolicyGeneric<ReplDomain>>::Summary>, AmfiteatrError<ReplDomain>>{
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
                            let summary = guard.policy_mut().train(&trajectories[..])?;
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
                let m = Arc::try_unwrap(summaries2).map_err(|_e|{
                    AmfiteatrError::Lock { description: "Failed unwrapping Arc".to_string(), object: "Summaries in training".to_string() }
                })?;
                let summaries = m.into_inner();
                Ok(summaries)
            }
        }

    }
    pub fn run_epoch(&mut self, episodes: usize) -> Result<EpochDescription, AmfiteatrError<ReplDomain>>{

        let mut description = self.create_empty_description(episodes);
        self.clear_episodes_trajectories()?;
        for i in 0..episodes{
            log::debug!("Running episode {}", i);
            self.run_episode()?;
            self.update_epoch_description_score_with_episode(&mut description);
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
        for e in 0..epochs as i64{
            log::info!("Running epoch {}", e);
            let description = self.run_epoch(episodes)?;
            //debug!("Epoch {e} description : {:?}", description);
            let description_mean = description.mean_divide_round(self.environment.state().target_rounds());
            /*
            if let Some(mut writer) = &self.tboard_writer{

            }*/
            let mut network_learning_score_sum: f32 = 0.0;
            let mut network_learning_mean_hawk_moves_sum: f32 = 0.0;
            let mut network_learning_mean_dove_moves_sum: f32 = 0.0;
            let mut hawk_score_sum: f32 = 0.0;
            let mut dove_score_sum: f32 = 0.0;
            let mut mixes_score_sum: f32 = 0.0;
            for learner in &self.network_learning_agents{
                let mut guard = learner.lock();

                if let Some(mean_score) = description_mean.mean_scores.get(guard.id()){
                    guard.policy_mut().t_write_scalar(e, "mean_score", *mean_score as f32)?;
                    network_learning_score_sum += *mean_score as f32;
                }
                if let Some(mean_hawks) = description_mean.mean_network_learning_hawk_moves.get(guard.id()){
                    guard.policy_mut().t_write_scalar(e, "mean_hawk_moves", *mean_hawks as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened { context: "Reporting hawk moves".to_string(), error: e.to_string() })?;
                    network_learning_mean_hawk_moves_sum += *mean_hawks as f32;
                }
                if let Some(mean_doves) = description_mean.mean_network_learning_dove_moves.get(guard.id()){
                    guard.policy_mut().t_write_scalar(e, "mean_dove_moves", *mean_doves as f32)?;
                    network_learning_mean_dove_moves_sum += *mean_doves as f32;
                }
            }

            if let Some(twriter) = &mut self.tboard_writer_learners{
                if !self.network_learning_agents.is_empty(){
                    twriter.write_scalar(e, "mean_score", network_learning_score_sum / self.network_learning_agents.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Learners".to_string(), error: e.to_string() })?;
                    twriter.write_scalar(e, "mean_hawk_moves", network_learning_mean_hawk_moves_sum / self.network_learning_agents.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Learners".to_string(), error: e.to_string() })?;
                    twriter.write_scalar(e, "mean_dove_moves", network_learning_mean_dove_moves_sum / self.network_learning_agents.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Learners".to_string(), error: e.to_string() })?;
                }
            }

            for dove in &self.pure_doves{
                let guard = dove.lock();

                if let Some(mean_score) = description_mean.mean_scores.get(guard.id()){
                    dove_score_sum += *mean_score as f32;
                }
            }
            if let Some(twriter) = &mut self.tboard_writer_doves{
                if !self.pure_doves.is_empty(){
                    twriter.write_scalar(e, "mean_score", dove_score_sum / self.pure_doves.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Doves".to_string(), error: e.to_string() })?;
                }

            }

            for hawk in &self.pure_hawks{
                let guard = hawk.lock();

                if let Some(mean_score) = description_mean.mean_scores.get(guard.id()){
                    hawk_score_sum += *mean_score as f32;
                }
            }

            if let Some(twriter) = &mut self.tboard_writer_hawks{
                if !self.pure_hawks.is_empty(){
                    twriter.write_scalar(e, "mean_score", hawk_score_sum / self.pure_hawks.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Hawks".to_string(), error: e.to_string() })?;
                }

            }
            for mixed in &self.mixed_agents{
                let guard = mixed.lock();

                if let Some(mean_score) = description_mean.mean_scores.get(guard.id()){
                    mixes_score_sum += *mean_score as f32;
                }
            }

            if let Some(twriter) = &mut self.tboard_writer_mixes{
                if !self.mixed_agents.is_empty(){
                    twriter.write_scalar(e, "mean_score", mixes_score_sum / self.mixed_agents.len() as f32)
                        .map_err(|e|AmfiteatrError::TboardFlattened { context: "Reporting Mixes".to_string(), error: e.to_string() })?;
                }

            }


            let learn_summary = self.train_network_agents_parallel()?;

            epoch_results.push(description_mean);
            epoch_learn_results.push(learn_summary);
        }
        Ok((epoch_results, epoch_learn_results))
    }
}