use crate::agent::{Policy, StatefulAgent};
use crate::domain::DomainParameters;

/// Trait for agents that performs actions, possibly mutating some attributes of agent.
///
pub trait ActingAgent<DP: DomainParameters>{

    /// Agent selects action and performs optional changing operations.
    /// It has ability to mutate agent when he/she selects action.
    /// Method is invoked in [`AutomaticAgent::run`](crate::agent::AutomaticAgent::run)
    /// in the very moment when agent needs to select action.
    /// In the same moment agent may want to mutate itself for example to separate steps of game for purpose of saving trajectory.
    /// This behaviour is used by [`AgentGen`](crate::agent::AgentGen) to add rewards obtained since last action to current store.
    /// [`AgentGenT`](crate::agent::TracingAgentGen) uses it also to add new step entry to his/her game trajectory.
    /// __Note__ that this method should not affect agents _information set_, as the way of changing it is through [`DomainParameters::UpdateType`](crate::domain::DomainParameters::UpdateType)
    /// provided by _environment_.
    fn take_action(&mut self) -> Option<DP::ActionType>;

    /// This method is meant to do optional actions of [`take_action`](crate::agent::ActingAgent::take_action)
    /// without selecting new action. Usually to be invoked at the end of game to commit last step to trace.
    fn finalize(&mut self);
}
/// Agent that follows some policy, which can be referenced.
pub trait PolicyAgent<DP: DomainParameters>: StatefulAgent<DP>{
    /// Policy type that is used by agent
    type Policy: Policy<DP, InfoSetType= <Self as StatefulAgent<DP>>::InfoSetType>;

    /// Returns reference to policy followed by this instance of agent
    fn policy(&self) -> &Self::Policy;
    /// Returns mutable reference to policy followed by this instance of agent
    fn policy_mut(&mut self) -> &mut Self::Policy;
    /// Selects action, by default it runs [`Policy::select_action`](crate::agent::Policy::select_action).
    /// However one may want to perform some additional actions on agent then.
    /// For example agent may want to log this event or store selected action
    /// (self reference is not mutable, however it can be cheated for example with [Cell](::std::cell::Cell))
    fn policy_select_action(&self)
        -> Option<DP::ActionType>{
        self.policy().select_action(self.info_set())
    }
}