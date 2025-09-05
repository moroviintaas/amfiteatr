use std::fmt::Debug;
use crate::scheme::{Scheme};


/// Game state to be used in sequential games (where in single time step
/// only one player is allowed to play).
pub trait SequentialGameState<S: Scheme>: Send + Debug{
    type Updates: IntoIterator<Item = (S::AgentId, S::UpdateType)>;

    fn current_player(&self) -> Option<S::AgentId>;
    fn is_finished(&self) -> bool;

    fn forward(&mut self, agent: S::AgentId, action: S::ActionType)
        -> Result<Self::Updates, S::GameErrorType>;

    //fn transform(&mut self, agent_id: &Spec::AgentId, action: Spec::ActionType) -> Result<Self::UpdatesCollection, Spec::GameErrorType>;

    /// Return initial observations for agents (possibly `None`). If called  after first step should probably return `None`.
    fn first_observations(&self) -> Option<Self::Updates>{
        None
    }
}

//pub trait EnvStateSimultaneous{}


impl<S: Scheme, T: SequentialGameState<S>> SequentialGameState<S> for Box<T>{
    type Updates = T::Updates;

    fn current_player(&self) -> Option<S::AgentId> {
        self.as_ref().current_player()
    }

    fn is_finished(&self) -> bool {
        self.as_ref().is_finished()
    }

    fn forward(&mut self, agent: S::AgentId, action: S::ActionType) -> Result<Self::Updates, S::GameErrorType> {
        self.as_mut().forward(agent, action)
    }
}


/// Combination of traits [`SequentialGameState`] and [`From`]
pub trait ConstructedSequentialGameState<S: Scheme, B>:
    SequentialGameState<S> + From<B>{}


impl<S: Scheme, B, T: SequentialGameState<S> + From<B>>
    ConstructedSequentialGameState<S, B> for T{}


//impl<S: DomainParameters, B, T: ConstructedEnvState<S, B>> ConstructedEnvState<S, B> for Box<T>{}


/// Trait adding interface to get current payoff of selected agent.
pub trait GameStateWithPayoffs<S: Scheme>: SequentialGameState<S>{

    fn state_payoff_of_player(&self, agent: &S::AgentId) -> S::UniversalReward;

}

/*
pub trait ExpandingState<S: DomainParameters>{

    fn register_agent(&mut self, agent_id: S::AgentId) -> Result<(), AmfiError<S>>;

}

 */
