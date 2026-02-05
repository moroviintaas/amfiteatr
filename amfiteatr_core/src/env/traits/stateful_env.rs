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


    /// Implement this method to enable pointing agent that violated rules in the game.
    /// Default implementation always returns `None` which is considered that no rules were violated,
    /// and the game is in proper state.
    fn game_violator(&self) -> Option<&S::AgentId>;
    /*{
        None
    }*/


    /// Implement this method to enable setting agent that violated rules in the game.
    /// Default implementation does not do anything.
    /// This method will be used by automatic agents to set this information if
    /// method [`forward`](SequentialGameState::forward) returns an error. If it is not implemented.
    fn set_game_violator(&mut self, game_violator: Option<S::AgentId>);
    /*
    {

    }

     */
}