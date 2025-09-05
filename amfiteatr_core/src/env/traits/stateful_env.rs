use crate::scheme::Scheme;
use crate::env::SequentialGameState;
use crate::error::AmfiteatrError;

/// Environment for games with some state, actually it is for almost every game.
pub trait StatefulEnvironment<S: Scheme>{
    type State: SequentialGameState<S>;

    /// Returns reference to current game state.
    fn state(&self) -> &Self::State;


    /// Returns Option for actual player, when no player is allowed to play it is None (game
    /// is finished).
    ///
    fn current_player(&self) -> Option<S::AgentId>{
        self.state().current_player()
    }

    /// Processes action of agent, if result is Ok, iterator of updates for every player is
    /// returned.
    fn process_action(&mut self, agent: &S::AgentId, action: &S::ActionType)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>>;


}