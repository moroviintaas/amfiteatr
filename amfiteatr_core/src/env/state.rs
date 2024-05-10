use std::fmt::Debug;
use crate::domain::{DomainParameters};


/// Game state to be used in sequential games (where in single time step
/// only one player is allowed to play).
pub trait SequentialGameState<DP: DomainParameters>: Send + Debug{
    type Updates: IntoIterator<Item = (DP::AgentId, DP::UpdateType)>;

    fn current_player(&self) -> Option<DP::AgentId>;
    fn is_finished(&self) -> bool;

    fn forward(&mut self, agent: DP::AgentId, action: DP::ActionType)
        -> Result<Self::Updates, DP::GameErrorType>;

    //fn transform(&mut self, agent_id: &Spec::AgentId, action: Spec::ActionType) -> Result<Self::UpdatesCollection, Spec::GameErrorType>;

}

//pub trait EnvStateSimultaneous{}


impl<DP: DomainParameters, T: SequentialGameState<DP>> SequentialGameState<DP> for Box<T>{
    type Updates = T::Updates;

    fn current_player(&self) -> Option<DP::AgentId> {
        self.as_ref().current_player()
    }

    fn is_finished(&self) -> bool {
        self.as_ref().is_finished()
    }

    fn forward(&mut self, agent: DP::AgentId, action: DP::ActionType) -> Result<Self::Updates, DP::GameErrorType> {
        self.as_mut().forward(agent, action)
    }
}


/// Combination of traits [`SequentialGameState`] and [`From`]
pub trait ConstructedSequentialGameState<DP: DomainParameters, B>:
    SequentialGameState<DP> + From<B>{}


impl<DP: DomainParameters, B, T: SequentialGameState<DP> + From<B>>
    ConstructedSequentialGameState<DP, B> for T{}


//impl<DP: DomainParameters, B, T: ConstructedEnvState<DP, B>> ConstructedEnvState<DP, B> for Box<T>{}


/// Trait adding interface to get current payoff of selected agent.
pub trait GameStateWithPayoffs<DP: DomainParameters>: SequentialGameState<DP>{

    fn state_payoff_of_player(&self, agent: &DP::AgentId) -> DP::UniversalReward;

}

/*
pub trait ExpandingState<DP: DomainParameters>{

    fn register_agent(&mut self, agent_id: DP::AgentId) -> Result<(), AmfiError<DP>>;

}

 */
