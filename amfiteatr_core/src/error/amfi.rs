use thiserror::Error;
use crate::error::{CommunicationError, ProtocolError, WorldError};
use crate::domain::{DomainParameters};

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum AmfiError<DP: DomainParameters>{
    #[error("Game error: {0}")]
    Game(DP::GameErrorType),
    #[error("Agent {1} caused game error: {0}")]
    GameA(DP::GameErrorType, DP::AgentId),
    #[error("Communication error: {0}")]
    Communication(CommunicationError<DP>),
    #[error("Protocol error: {0}")]
    Protocol(ProtocolError<DP>),

    //#[error("Setup error: {0}")]
    //Setup(SetupError<DP>),
    #[error("Data convert")]
    DataConvert(),
    #[error("World maintenance error: {0}")]
    World(WorldError<DP>),
    #[error("Custom: {0}")]
    Custom(String)
    //#[error("External: {0}")]
    //External(String)
}
