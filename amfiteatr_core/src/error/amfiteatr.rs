use thiserror::Error;
use crate::error::{CommunicationError, ProtocolError, TrajectoryError, WorldError};
use crate::domain::{DomainParameters};

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum AmfiteatrError<DP: DomainParameters>{
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
    Custom(String),
    #[error("Lock error on {object:} with {description:}")]
    Lock{
        description: String,
        object: String
    },
    #[error("Trajectory maintanance error: {source:}")]
    Trajectory{
        #[source]
        source: TrajectoryError<DP>
    }
    //#[error("External: {0}")]
    //External(String)
}

