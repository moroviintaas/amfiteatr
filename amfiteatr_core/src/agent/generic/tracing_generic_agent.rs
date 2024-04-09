use std::marker::PhantomData;

use crate::agent::*;
use crate::comm::BidirectionalEndpoint;
use crate::error::{AmfiteatrError, CommunicationError};
use crate::domain::{AgentMessage, DomainParameters, EnvironmentMessage, Renew, Reward};



/// Generic agent implementing traits proposed in this crate.
/// This agent implements minimal functionality to work automatically with environment.
/// This agents  collects trace of game, for are agent not collecting it look for [AgentGen](crate::agent::AgentGen).
/// This agent can be built if used Policy operates on information set that is [`ScoringInformationSet`](crate::agent::EvaluatedInformationSet)
pub struct TracingAgentGen<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{


    information_set: <P as Policy<DP>>::InfoSetType,
    comm: Comm,
    policy: P,
    _phantom: PhantomData<DP>,

    constructed_universal_reward: <DP as DomainParameters>::UniversalReward,
    committed_universal_score: <DP as DomainParameters>::UniversalReward,

    game_trajectory: AgentTrajectory<DP, P::InfoSetType>,
    last_action: Option<DP::ActionType>,
    state_before_last_action: Option<<P as Policy<DP>>::InfoSetType>,
    episodes: Vec<AgentTrajectory<DP, P::InfoSetType>>,
}

impl <DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{

    pub fn new(state: <P as Policy<DP>>::InfoSetType, comm: Comm, policy: P) -> Self{
        Self{
            information_set: state,
            comm,
            policy,
            _phantom:PhantomData::default(),
            constructed_universal_reward: Reward::neutral(),
            committed_universal_score: Reward::neutral(),
            game_trajectory: AgentTrajectory::new(),
            state_before_last_action: None,
            last_action: None,
            //explicit_subjective_reward_component: <P::InfoSetType as EvaluatedInformationSet<DP>>::RewardType::neutral(),
            episodes: vec![],
        }
    }

    /// Given new policy consumes this agent producing replacement agent (with moved internal state).
    /// New agent has now provided policy. Previous policy is dropped.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::{TracingAgentGen, RandomPolicy};
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DEMO_AGENT_RED, DemoAgentID, DemoInfoSet, DemoPolicySelectFirst};
    /// let (_, comm) = StdEnvironmentEndpoint::new_pair();
    /// let agent = TracingAgentGen::new(DemoInfoSet::new(DEMO_AGENT_RED, 10), comm, RandomPolicy::new());
    /// let agent_2 = agent.transform_replace_policy(DemoPolicySelectFirst{});
    /// ```
    pub fn transform_replace_policy<P2: Policy<DP, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> TracingAgentGen<DP, P2, Comm>
    {
        TracingAgentGen::<DP, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            committed_universal_score: self.committed_universal_score,
            comm: self.comm,
            last_action: self.last_action,
            state_before_last_action: self.state_before_last_action,
            game_trajectory: self.game_trajectory,
            //explicit_subjective_reward_component: self.explicit_subjective_reward_component,
            episodes: vec![],
        }
    }


    /// Given new policy consumes this agent producing replacement agent (with moved internal state).
    /// New agent has now provided policy. Previous policy is returned as second element in tuple.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::{TracingAgentGen, RandomPolicy};
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DEMO_AGENT_RED, DemoAgentID, DemoInfoSet, DemoPolicySelectFirst};
    /// let (_, comm) = StdEnvironmentEndpoint::new_pair();
    /// let agent = TracingAgentGen::new(DemoInfoSet::new(DEMO_AGENT_RED, 10), comm, RandomPolicy::new());
    /// let (agent_2, old_policy) = agent.transform_replace_policy_ret(DemoPolicySelectFirst{});
    /// ```
    pub fn transform_replace_policy_ret<P2: Policy<DP, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> (TracingAgentGen<DP, P2, Comm>, P)
    {
        let p = self.policy;
        (TracingAgentGen::<DP, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            comm: self.comm,
            last_action: self.last_action,
            state_before_last_action: self.state_before_last_action,
            game_trajectory: self.game_trajectory,
            committed_universal_score: self.committed_universal_score,
            episodes: vec![],
        }, p)
    }

    /// Replaces communication endpoint returning old in return;
    pub fn replace_comm(&mut self, mut comm: Comm) -> Comm{
        std::mem::swap(&mut self.comm, &mut comm);
        comm
    }
    /// Using [`std::mem::swap`](::std::mem::swap) swaps communication endpoints between two instances.
    pub fn swap_comms<P2: Policy<DP>>(&mut self, other: &mut TracingAgentGen<DP, P2, Comm>)
    where <P2 as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone{
        std::mem::swap(&mut self.comm, &mut other.comm)
    }

    /// Using [`std::mem::swap`](::std::mem::swap) swaps communication endpoints with instance of [`AgentGent`](crate::agent::AgentGen).
    pub fn swap_comms_with_basic<P2: Policy<DP>>(&mut self, other: &mut AgentGen<DP, P2, Comm>)
    where <P2 as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone{
        std::mem::swap(&mut self.comm, &mut other.comm_mut())
    }

    pub(crate) fn comm_mut(&mut self) -> &mut Comm{
        &mut self.comm
    }


    pub fn episodes(&self) -> &Vec<AgentTrajectory<DP, P::InfoSetType>>{
        &self.episodes
    }

    pub fn take_episodes(&mut self) -> Vec<AgentTrajectory<DP, P::InfoSetType>>{
        let mut v = Vec::with_capacity(self.episodes.len());
        std::mem::swap(&mut v, &mut self.episodes);
        v
    }

}



impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
    CommunicatingAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone{

    type CommunicationError = CommunicationError<DP>;

    fn send(&mut self, message: AgentMessage<DP>) -> Result<(), Self::CommunicationError> {
        self.comm.send(message)
    }

    fn recv(&mut self) -> Result<EnvironmentMessage<DP>, Self::CommunicationError> {
        self.comm.receive_blocking()
    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
StatefulAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{

    type InfoSetType = <P as Policy<DP>>::InfoSetType;

    fn update(&mut self, state_update: DP::UpdateType) -> Result<(), DP::GameErrorType> {
        self.information_set.update(state_update)
    }

    fn info_set(&self) -> &Self::InfoSetType {
        &self.information_set
    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>,
    Seed> ReseedAgent<DP, Seed> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: Renew<DP, Seed>
    + InformationSet<DP>,
<Self as StatefulAgent<DP>>::InfoSetType: Renew<DP, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>>{

        self.game_trajectory.clear();
        self.constructed_universal_reward = DP::UniversalReward::neutral();
        self.committed_universal_score = DP::UniversalReward::neutral();
        self.information_set.renew_from(seed)

    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
ActingAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone{

    /// Firstly, agent commits last step to stack.
    fn take_action(&mut self) -> Result<Option<DP::ActionType>, AmfiteatrError<DP>> {
        self.commit_trace()?;

        let action = self.policy.select_action(&self.information_set);
        self.last_action = action.clone();
        self.state_before_last_action = Some(self.information_set.clone());
        Ok(action)
    }

    fn finalize(&mut self) -> Result<(), AmfiteatrError<DP>>{
        /*
        self.commit_trace();
        self.game_trajectory.finalize(self.information_set.clone());
        self.state_before_last_action = Some(self.information_set.clone())

         */
        //self.game_trajectory.finish(self.information_set.clone(), self.last_action.take()).unwrap()
        self.finalize_trajectory()
    }

    fn react_refused_action(&mut self) -> Result<(), AmfiteatrError<DP>> {
        todo!()
    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
TracingAgent<DP, <P as Policy<DP>>::InfoSetType> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone,
//for <'a> &'a<DP as DomainParameters>::UniversalReward: Sub<&'a <DP as DomainParameters>::UniversalReward, Output=<DP as DomainParameters>::UniversalReward>,
//for<'a> &'a <<P as Policy<DP>>::StateType as ScoringInformationSet<DP>>::RewardType: Sub<&'a  <<P as Policy<DP>>::StateType as ScoringInformationSet<DP>>::RewardType, Output = <<P as Policy<DP>>::StateType as ScoringInformationSet<DP>>::RewardType>
{
    fn reset_trajectory(&mut self) {
        self.game_trajectory.clear();
        self.last_action = None;
    }

    fn take_trajectory(&mut self) -> AgentTrajectory<DP, <P as Policy<DP>>::InfoSetType> {
        std::mem::take(&mut self.game_trajectory)
    }

    fn game_trajectory(&self) -> &AgentTrajectory<DP, <P as Policy<DP>>::InfoSetType> {
        &self.game_trajectory
    }

    /// Commits information set change to trajectory. Adds [
    fn commit_trace(&mut self) -> Result<(), AmfiteatrError<DP>> {
        if let Some(prev_action) = self.last_action.take(){

            let universal_score_before_update = self.committed_universal_score.clone();
            self.commit_partial_rewards();
            //let universal_score_after_update = self.committed_universal_score.clone();
            let initial_state = self.state_before_last_action.take().unwrap();

            self.game_trajectory.register_step_point(initial_state, prev_action, universal_score_before_update)

        } else {
            Ok(())
        }

    }

    fn finalize_trajectory(&mut self) -> Result<(), AmfiteatrError<DP>> {
        if let (Some(action), Some(info_set_before)) = (&self.last_action, &self.state_before_last_action){
            self.game_trajectory.register_step_point(info_set_before.clone(), action.clone(), self.committed_universal_score.clone())?;
            self.commit_partial_rewards();
            self.game_trajectory.finish(self.information_set.clone(), self.committed_universal_score.clone())
        } else {
            #[cfg(feature = "log_warn")]
            log::warn!("Finalizing trajectory with no previous step");
            Ok(())
        }


    }
}




impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>,
    Seed>
MultiEpisodeAutoAgentRewarded<DP, Seed> for TracingAgentGen<DP, P, Comm>
where Self: ReseedAgent<DP, Seed> + AutomaticAgentRewarded<DP>,
      <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone,
{
    fn store_episode(&mut self) {
        let mut new_trajectory = AgentTrajectory::with_capacity(self.game_trajectory.number_of_steps());
        std::mem::swap(&mut new_trajectory, &mut self.game_trajectory);
        self.episodes.push(new_trajectory);

    }

    fn clear_episodes(&mut self) {
        self.episodes.clear();
    }
}



impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>,
    Seed>
MultiEpisodeTracingAgent<DP, <P as Policy<DP>>::InfoSetType, Seed> for TracingAgentGen<DP, P, Comm>
    where <P as Policy<DP>>::InfoSetType: InformationSet<DP> + Clone,
    Self: ReseedAgent<DP, Seed>
    //+ SelfEvaluatingAgent<DP>
    + AutomaticAgentRewarded<DP>,
          <Self as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>{



    fn take_episodes(&mut self) -> Vec<AgentTrajectory<DP, <P as Policy<DP>>::InfoSetType>> {
        let mut episodes = Vec::with_capacity(self.episodes.len());
        std::mem::swap(&mut episodes, &mut self.episodes);
        episodes
    }
}
impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
PolicyAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn policy_mut(&mut self) -> &mut Self::Policy {
        &mut self.policy
    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
RewardedAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{

    fn current_universal_reward(&self) -> DP::UniversalReward {
        self.constructed_universal_reward.clone()
    }

    fn current_universal_reward_add(&mut self, reward_fragment: &DP::UniversalReward) {
        self.constructed_universal_reward += reward_fragment;
    }


    fn current_universal_score(&self) -> DP::UniversalReward {
        self.committed_universal_score.clone() + &self.constructed_universal_reward
    }
    fn commit_partial_rewards(&mut self) {
        self.committed_universal_score += &self.constructed_universal_reward;
        self.constructed_universal_reward = DP::UniversalReward::neutral();
    }
}

impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
ReinitAgent<DP> for TracingAgentGen<DP, P, Comm>
where <P as Policy<DP>>::InfoSetType: InformationSet<DP>{

    fn reinit(&mut self, initial_state: <Self as StatefulAgent<DP>>::InfoSetType) {
        self.information_set = initial_state;
        self.game_trajectory.clear();
        self.constructed_universal_reward = DP::UniversalReward::neutral();
        self.committed_universal_score = DP::UniversalReward::neutral();
        self.state_before_last_action = None;
        self.last_action = None;

    }
}

/*
impl<
    DP: DomainParameters,
    P: Policy<DP>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<DP>,
        InwardType=EnvironmentMessage<DP>,
        Error=CommunicationError<DP>>>
SelfEvaluatingAgent<DP> for TracingAgentGen<DP, P, Comm>
where <Self as StatefulAgent<DP>>::InfoSetType: EvaluatedInformationSet<DP>,
<P as Policy<DP>>::InfoSetType: EvaluatedInformationSet<DP>{
    type Assessment = <<Self as StatefulAgent<DP>>::InfoSetType as EvaluatedInformationSet<DP>>::RewardType;
    fn current_assessment_total(&self) ->  Self::Assessment {
        self.information_set.current_subjective_score() + &self.explicit_subjective_reward_component
    }

    fn add_explicit_assessment(&mut self, explicit_reward: &Self::Assessment) {
        self.explicit_subjective_reward_component += explicit_reward
    }
    fn penalty_for_illegal_action(&self) -> Self::Assessment {
        self.information_set.penalty_for_illegal()
    }
}



 */