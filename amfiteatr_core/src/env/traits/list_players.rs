use crate::scheme::Scheme;

/// Environment interface to list agents taking part in game or simulations
pub trait ListPlayers<S: Scheme>{
    type IterType: Iterator<Item = S::AgentId>;

    /// Method returning `IntoIterator` of players in game.
    /// This method is used to propagate error so it should include all players.
    fn players(&self) -> Self::IterType;
}
