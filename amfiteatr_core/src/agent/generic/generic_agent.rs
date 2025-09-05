use std::marker::PhantomData;
use crate::agent::*;
use crate::agent::info_set::{InformationSet};
use crate::agent::policy::Policy;
use crate::comm::BidirectionalEndpoint;
use crate::error::{AmfiteatrError, CommunicationError};
use crate::scheme::{AgentMessage, EnvironmentMessage, Scheme, Reward, Renew};

/// Generic agent implementing common traits needed by agent.
/// This agent implements minimal functionality to work automatically with environment.
/// This agents does not collect trace of game, for are agent collecting it look for [AgentGenT](crate::agent::TracingAgentGen).
/// This agent can be built if used Policy operates on information set that is [`ScoringInformationSet`](crate::agent::EvaluatedInformationSet)
pub struct AgentGen<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{
    /// Information Set (State as viewed by agent)
    information_set: <P as Policy<S>>::InfoSetType,
    /// Communication endpoint, should be paired with environment's.
    comm: Comm,
    /// Agent's policy
    policy: P,
    _phantom: PhantomData<S>,

    constructed_universal_reward: <S as Scheme>::UniversalReward,
    committed_universal_score: <S as Scheme>::UniversalReward,
    //explicit_subjective_reward_component: <P::InfoSetType as EvaluatedInformationSet<S>>::RewardType,
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>
    >
>
    AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>
{


    pub fn new(state: <P as Policy<S>>::InfoSetType, comm: Comm, policy: P) -> Self{
        Self{
            information_set: state,
            comm,
            policy,
            _phantom:PhantomData,
            constructed_universal_reward: Reward::neutral(),
            committed_universal_score: Reward::neutral(),
            //explicit_subjective_reward_component: <P::InfoSetType as EvaluatedInformationSet<S>>::RewardType::neutral()
        }
    }

    /// Replaces information set in agent. Does not anything beside it.
    pub fn replace_info_set(&mut self, state: <P as Policy<S>>::InfoSetType){
        self.information_set = state
    }

    /// Given new policy consumes this agent producing replacement agent (with moved internal state).
    /// New agent has now provided policy. Previous policy is dropped.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::{AgentGen, RandomPolicy};
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DEMO_AGENT_RED, DemoAgentID, DemoInfoSet, DemoPolicySelectFirst};
    /// let (_, comm) = StdEnvironmentEndpoint::new_pair();
    /// let agent = AgentGen::new(DemoInfoSet::new(DEMO_AGENT_RED, 10), comm, RandomPolicy::new());
    /// let agent_2 = agent.transform_replace_policy(DemoPolicySelectFirst{});
    /// ```
    pub fn transform_replace_policy<P2: Policy<S, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> AgentGen<S, P2, Comm>
    {
        AgentGen::<S, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            committed_universal_score: self.committed_universal_score,
            comm: self.comm,
            //explicit_subjective_reward_component: self.explicit_subjective_reward_component
        }
    }
    /// Given new policy consumes this agent producing replacement agent (with moved internal state).
    /// New agent has now provided policy. Previous policy is returned as second element in tuple.
    /// # Example:
    /// ```
    /// use amfiteatr_core::agent::{AgentGen, RandomPolicy};
    /// use amfiteatr_core::comm::StdEnvironmentEndpoint;
    /// use amfiteatr_core::demo::{DEMO_AGENT_RED, DemoAgentID, DemoInfoSet, DemoPolicySelectFirst};
    /// let (_, comm) = StdEnvironmentEndpoint::new_pair();
    /// let agent = AgentGen::new(DemoInfoSet::new(DEMO_AGENT_RED, 10), comm, RandomPolicy::new());
    /// let (agent_2, old_policy) = agent.transform_replace_policy_ret(DemoPolicySelectFirst{});
    /// ```
    pub fn transform_replace_policy_ret<P2: Policy<S, InfoSetType=P::InfoSetType>>(self, new_policy: P2) -> (AgentGen<S, P2, Comm>, P)
    {
        let p = self.policy;
        (AgentGen::<S, P2, Comm>{
            information_set: self.information_set,
            policy: new_policy,
            _phantom: Default::default(),
            constructed_universal_reward: self.constructed_universal_reward,
            committed_universal_score: self.committed_universal_score,
            comm: self.comm,
            //explicit_subjective_reward_component: self.explicit_subjective_reward_component
        }, p)
    }

    /// Replaces communication endpoint returning old in return;
    pub fn replace_comm(&mut self, mut comm: Comm) -> Comm{
        std::mem::swap(&mut self.comm, &mut comm);
        comm
    }
    /// Using [`std::mem::swap`](::std::mem::swap) swaps communication endpoints between two instances.
    pub fn swap_comms<P2: Policy<S>>(&mut self, other: &mut AgentGen<S, P2, Comm>)
    where <P2 as Policy<S>>::InfoSetType: InformationSet<S>{
        std::mem::swap(&mut self.comm, &mut other.comm)
    }
    /// Using [`std::mem::swap`](::std::mem::swap) swaps communication endpoints with instance of [`AgentGentT`](crate::agent::TracingAgentGen).
    pub fn swap_comms_with_tracing<P2: Policy<S>>(&mut self, other: &mut TracingAgentGen<S, P2, Comm>)
    where <P2 as Policy<S>>::InfoSetType: InformationSet<S>{
        std::mem::swap(&mut self.comm, other.comm_mut())
    }

    pub(crate) fn comm_mut(&mut self) -> &mut Comm{
        &mut self.comm
    }

    /*
    /// Adds current partial reward to actual score, and then neutralises universal reward
    fn commit_reward_to_score(&mut self){
        self.actual_universal_score += &self.constructed_universal_reward;
        self.constructed_universal_reward = S::UniversalReward::neutral();
    }

     */
}

impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>
    >
>
    CommunicatingAgent<S> for AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>
{

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
        Error=CommunicationError<S>
    >
>
StatefulAgent<S> for AgentGen<S, P, Comm>
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
    Seed> ReseedAgent<S, Seed> for AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: Renew<S, Seed>
    + InformationSet<S>,
<Self as StatefulAgent<S>>::InfoSetType: Renew<S, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<S>> {


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
        Error=CommunicationError<S>>,
    Seed>
MultiEpisodeAutoAgent<S, Seed> for AgentGen<S, P, Comm>
    where Self: ReseedAgent<S, Seed> + AutomaticAgent<S>,
          <P as Policy<S>>::InfoSetType: InformationSet<S>,
{
    fn initialize_episode(&mut self) -> Result<(), AmfiteatrError<S>> {
        self.policy_mut().call_on_episode_start()
    }

    fn store_episode(&mut self)  -> Result<(), AmfiteatrError<S>> {
        let payoff = self.committed_universal_score.clone();
        self.policy_mut().call_on_episode_finish(payoff)
    }

    fn clear_episodes(&mut self)  -> Result<(), AmfiteatrError<S>> {
        self.policy_mut().call_between_epochs()
    }
}



impl<
    S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>>
ActingAgent<S> for AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    fn select_action(&mut self) -> Result<S::ActionType, AmfiteatrError<S>> {
        //self.commit_reward_to_score();
        self.commit_partial_rewards();
        self.policy.select_action(&self.information_set)

    }

    fn finalize(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.commit_partial_rewards();
        Ok(())
        //self.commit_reward_to_score();
    }

    fn react_refused_action(&mut self) -> Result<(), AmfiteatrError<S>>{
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
PolicyAgent<S> for AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{
    type Policy = P;

    fn policy(&self) -> &Self::Policy {
        &self.policy
    }

    fn policy_mut(&mut self) -> &mut Self::Policy {
        &mut self.policy
    }
}



impl<S: Scheme,
    P: Policy<S>,
    Comm: BidirectionalEndpoint<
        OutwardType=AgentMessage<S>,
        InwardType=EnvironmentMessage<S>,
        Error=CommunicationError<S>>> RewardedAgent<S> for AgentGen<S, P, Comm>
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
ReinitAgent<S> for AgentGen<S, P, Comm>
where <P as Policy<S>>::InfoSetType: InformationSet<S>{

    fn reinit(&mut self, initial_state: <Self as StatefulAgent<S>>::InfoSetType) {
        self.information_set = initial_state;
        self.constructed_universal_reward = S::UniversalReward::neutral();
        self.committed_universal_score = S::UniversalReward::neutral();
    }
}
