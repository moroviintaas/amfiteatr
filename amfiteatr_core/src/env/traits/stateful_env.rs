use crate::domain::DomainParameters;
use crate::env::EnvironmentStateSequential;

/// Environment for games with some state, actually it is for almost every game.
pub trait StatefulEnvironment<DP: DomainParameters>{
    type State: EnvironmentStateSequential<DP>;

    /// Returns reference to current game state.
    fn state(&self) -> &Self::State;


    /// Returns Option for actual player, when no player is allowed to play it is None (game
    /// is finished).
    ///
    fn current_player(&self) -> Option<DP::AgentId>{
        self.state().current_player()
    }

    /// Processes action of agent, if result is Ok, iterator of updates for every player is
    /// returned.
    fn process_action(&mut self, agent: &DP::AgentId, action: &DP::ActionType) 
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, DP::GameErrorType>;


}