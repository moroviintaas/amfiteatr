use std::marker::PhantomData;

use crate::agent::*;
use crate::comm::BidirectionalEndpoint;
use crate::error::{AmfiteatrError, CommunicationError};
use crate::scheme::{AgentMessage, Scheme, EnvironmentMessage, Renew, Reward};



/// Generic agent implementing traits proposed in this crate.
/// This agent implements minimal functionality to work automatically with environment.
/// This agents  collects trace of game, for are agent not collecting it look for [AgentGen](crate::agent::AgentGen).
/// This agent can be built if used Policy operates on information set that is [`ScoringInformationSet`](crate::agent::EvaluatedInformationSet)
pub struct TracingAgentGen<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{


    information_set: <P as Policy<S>>::InfoSetType,
    comm: Comm,
    policy: P,
    _phantom: PhantomData<S>,

    constructed_universal_reward: <S as Scheme>::UniversalReward,
    committed_universal_score: <S as Scheme>::UniversalReward,

    game_trajectory: AgentTrajectory<S, P::InfoSetType>,
    //last_action: Option<S::ActionType>,
    //state_before_last_action: Option<<P as Policy<S>>::InfoSetType>,
    episodes: Vec<AgentTrajectory<S, P::InfoSetType>>,
}

impl <S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    pub fn new(state: <P as Policy<S>>::InfoSetType, comm: Comm, policy: P) -> Self{
        Self{
            information_set: state,
            comm,
            policy,
            _phantom:PhantomData,
            constructed_universal_reward: Reward::neutral(),
            committed_universal_score: Reward::neutral(),
            game_trajectory: AgentTrajectory::new(),
            //state_before_last_action: None,
            //last_action: None,
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
    pub fn transform_replace_policy<P2: Policy<S, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> TracingAgentGen<S, P2, Comm>
    {
        TracingAgentGen::<S, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            committed_universal_score: self.committed_universal_score,
            comm: self.comm,
            //last_action: self.last_action,
            //state_before_last_action: self.state_before_last_action,
            game_trajectory: self.game_trajectory,
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
    pub fn transform_replace_policy_ret<P2: Policy<S, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> (TracingAgentGen<S, P2, Comm>, P)
    {
        let p = self.policy;
        (TracingAgentGen::<S, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            comm: self.comm,
            //last_action: self.last_action,
            //state_before_last_action: self.state_before_last_action,
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
    pub fn swap_comms<P2: Policy<S>>(&mut self, other: &mut TracingAgentGen<S, P2, Comm>)
    where <P2 as Policy<S>>::InfoSetType: InformationSet<S> + Clone{
        std::mem::swap(&mut self.comm, &mut other.comm)
    }

    /// Using [`std::mem::swap`](::std::mem::swap) swaps communication endpoints with instance of [`AgentGent`](crate::agent::AgentGen).
    pub fn swap_comms_with_basic<P2: Policy<S>>(&mut self, other: &mut AgentGen<S, P2, Comm>)
    where <P2 as Policy<S>>::InfoSetType: InformationSet<S> + Clone{
        std::mem::swap(&mut self.comm, other.comm_mut())
    }

    pub(crate) fn comm_mut(&mut self) -> &mut Comm{
        &mut self.comm
    }


    pub fn episodes(&self) -> &Vec<AgentTrajectory<S, P::InfoSetType>>{
        &self.episodes
    }

    pub fn take_episodes(&mut self) -> Vec<AgentTrajectory<S, P::InfoSetType>>{
        let mut v = Vec::with_capacity(self.episodes.len());
        std::mem::swap(&mut v, &mut self.episodes);
        v
    }


}



impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
    CommunicatingAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S> + Clone{

    //type CommunicationError = CommunicationError<S>;

    fn send(&mut self, message: AgentMessage<S>) -> Result<(), CommunicationError<S>> {
        self.comm.send(message)
    }

    fn recv(&mut self) -> Result<EnvironmentMessage<S>, CommunicationError<S>> {
        self.comm.receive_blocking()
    }
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
StatefulAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    type InfoSetType = <P as Policy<S>>::InfoSetType;

    fn update(&mut self, state_update: S::UpdateType) -> Result<(), S::GameErrorType> {
        self.information_set.update(state_update)
    }

    fn info_set(&self) -> &Self::InfoSetType {
        &self.information_set
    }
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>,
    Seed> ReseedAgent<S, Seed> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: Renew<S, Seed>
    + InformationSet<S>,
<Self as StatefulAgent<S>>::InfoSetType: Renew<S, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<S>>{

        self.game_trajectory.clear();
        self.constructed_universal_reward = S::UniversalReward::neutral();
        self.committed_universal_score = S::UniversalReward::neutral();
        self.information_set.renew_from(seed)

    }
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
ActingAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S> + Clone{



    /// Firstly, agent commits last step to stack.
    fn select_action(&mut self) -> Result<S::ActionType, AmfiteatrError<S>> {
        //self.commit_trace()?;
        //self.merge_partial_rewards();
        self.commit_partial_rewards();

        let r_action = self.policy.select_action(&self.information_set);
        //self.last_action = action.clone();
        //self.state_before_last_action = Some(self.information_set.clone());
        if let Ok(ref action) = r_action {
            self.game_trajectory.register_step_point(self.information_set.clone(), action.clone(), self.committed_universal_score.clone())?;
        } else {
            #[cfg(feature = "log_warn")]
            log::warn!("Agent {} does not select any action, therefore nothing is registered in trajectory", self.information_set.agent_id());
        }

        r_action
    }

    fn finalize(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.commit_partial_rewards();

        //self.finalize_trajectory()
        /*
        if let (Some(action), Some(info_set_before)) = (&self.last_action, &self.state_before_last_action){
            self.game_trajectory.register_step_point(info_set_before.clone(), action.clone(), self.committed_universal_score.clone())?;
            self.commit_partial_rewards();
            self.game_trajectory.finish(self.information_set.clone(), self.committed_universal_score.clone())
        } else {
            #[cfg(feature = "log_warn")]
            log::warn!("Finalizing trajectory with no previous step");
            Ok(())
        }

         */
        self.game_trajectory.finish(self.information_set.clone(), self.committed_universal_score.clone())

    }

    fn react_refused_action(&mut self) -> Result<(), AmfiteatrError<S>> {
        self.game_trajectory.mark_previous_action_illegal();
        #[cfg(feature = "log_error")]
        log::error!("Agent: {0} Action  has been refused", self.information_set.agent_id());
        Ok(())
    }
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
TracingAgent<S, <P as Policy<S>>::InfoSetType> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S> + Clone,
//for <'a> &'a<S as DomainParameters>::UniversalReward: Sub<&'a <S as DomainParameters>::UniversalReward, Output=<S as DomainParameters>::UniversalReward>,
//for<'a> &'a <<P as Policy<S>>::StateType as ScoringInformationSet<S>>::RewardType: Sub<&'a  <<P as Policy<S>>::StateType as ScoringInformationSet<S>>::RewardType, Output = <<P as Policy<S>>::StateType as ScoringInformationSet<S>>::RewardType>
{
    fn reset_trajectory(&mut self) {
        self.game_trajectory.clear();
        //self.last_action = None;
    }

    fn take_trajectory(&mut self) -> AgentTrajectory<S, <P as Policy<S>>::InfoSetType> {
        std::mem::take(&mut self.game_trajectory)
    }

    fn trajectory(&self) -> &AgentTrajectory<S, <P as Policy<S>>::InfoSetType> {
        &self.game_trajectory
    }

}




impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>,
    Seed>
MultiEpisodeAutoAgent<S, Seed> for TracingAgentGen<S, P, Comm>
where Self: ReseedAgent<S, Seed> + AutomaticAgent<S>,
      <P as Policy<S>>::InfoSetType: InformationSet<S> + Clone,
{
    fn initialize_episode(&mut self) -> Result<(), AmfiteatrError<S>> {
        self.policy_mut().call_on_episode_start()
    }

    fn store_episode(&mut self) -> Result<(), AmfiteatrError<S>> {
        let payoff = self.committed_universal_score.clone();
        self.policy_mut().call_on_episode_finish(payoff)?;
        let mut new_trajectory = AgentTrajectory::with_capacity(self.game_trajectory.number_of_steps());
        std::mem::swap(&mut new_trajectory, &mut self.game_trajectory);
        self.episodes.push(new_trajectory);
        Ok(())

    }

    fn clear_episodes(&mut self) -> Result<(), AmfiteatrError<S>> {
        self.policy_mut().call_between_epochs()?;
        self.episodes.clear();
        Ok(())
    }
}



impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>,
    Seed>
MultiEpisodeTracingAgent<S, <P as Policy<S>>::InfoSetType, Seed> for TracingAgentGen<S, P, Comm>
    where <P as Policy<S>>::InfoSetType: InformationSet<S> + Clone,
    Self: ReseedAgent<S, Seed>
    //+ SelfEvaluatingAgent<S>
    + AutomaticAgent<S>,
          <Self as StatefulAgent<S>>::InfoSetType: InformationSet<S>{



    fn take_episodes(&mut self) -> Vec<AgentTrajectory<S, <P as Policy<S>>::InfoSetType>> {
        let mut episodes = Vec::with_capacity(self.episodes.len());
        std::mem::swap(&mut episodes, &mut self.episodes);
        episodes
    }
}
impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
PolicyAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn policy_mut(&mut self) -> &mut Self::Policy {
        &mut self.policy
    }
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
RewardedAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    fn current_universal_reward(&self) -> S::UniversalReward {
        self.constructed_universal_reward.clone()
    }

    fn current_universal_reward_add(&mut self, reward_fragment: &S::UniversalReward) {
        self.constructed_universal_reward += reward_fragment;
    }


    fn current_universal_score(&self) -> S::UniversalReward {
        self.committed_universal_score.clone() + &self.constructed_universal_reward
    }
    fn commit_partial_rewards(&mut self) {
        self.committed_universal_score += &self.constructed_universal_reward;
        self.constructed_universal_reward = S::UniversalReward::neutral();
    }


}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
ReinitAgent<S> for TracingAgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    fn reinit(&mut self, initial_state: <Self as StatefulAgent<S>>::InfoSetType) {
        self.information_set = initial_state;
        self.game_trajectory.clear();
        self.constructed_universal_reward = S::UniversalReward::neutral();
        self.committed_universal_score = S::UniversalReward::neutral();
        //self.state_before_last_action = None;
        //self.last_action = None;

    }
}
