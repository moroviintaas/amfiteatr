use crate::domain::DomainParameters;

/// Environment interface to list agents taking part in game or simulations
pub trait ListPlayers<DP: DomainParameters>{
    type IterType: Iterator<Item = DP::AgentId>;

    /// Method returning `IntoIterator` of players in game.
    /// This method is used to propagate error so it should include all players.
    fn players(&self) -> Self::IterType;
}
