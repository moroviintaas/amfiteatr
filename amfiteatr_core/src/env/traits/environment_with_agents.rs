
use crate::scheme::Scheme;

/// Environment interface to list agents taking part in game or simulations
///
pub trait EnvironmentWithAgents<S: Scheme>{
    type PlayerIterator: IntoIterator<Item = S::AgentId>;

    /// Method returning `IntoIterator` of players in game.
    /// This method is used to propagate error so it should include all players.
    fn players(&self) -> Self::PlayerIterator;




}