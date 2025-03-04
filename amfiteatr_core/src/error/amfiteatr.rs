use thiserror::Error;
use crate::error::{CommunicationError, DataError, ProtocolError, TrajectoryError, WorldError};
use crate::domain::{DomainParameters};
use crate::error::tensor::TensorError;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum AmfiteatrError<DP: DomainParameters>{
    #[error("Game error: {source}")]
    Game{
        #[source]
        source: DP::GameErrorType
    },
    #[error("Agent {agent} caused game error: {source}")]
    GameA{
        #[source]
        source: DP::GameErrorType,
        agent: DP::AgentId
    },
    #[error("Communication error: {source}")]
    Communication{
        #[source]
        source: CommunicationError<DP>
    },
    #[error("Protocol error: {source}")]
    Protocol{
        #[source]
        source: ProtocolError<DP>
    },

    //#[error("Setup error: {0}")]
    //Setup(SetupError<DP>),
    #[error("Data convert")]
    DataConvert(),
    #[error("World maintenance error: {source}")]
    World{
        #[source]
        source: WorldError<DP>
    },
    #[error("Custom: {0}")]
    Custom(String),
    #[error("Lock error on {object:} with {description:}")]
    Lock{
        description: String,
        object: String,
    },
    #[error("Trajectory maintenance error: {source:}")]
    Trajectory{
        #[source]
        source: TrajectoryError<DP>
    },
    #[error("Error in nom parser: {explanation:}")]
    Nom{
        explanation: String
    },
    #[error("Error in I/O operation: {explanation:}")]
    IO{
        explanation: String
    },
    #[error("Impossible action")]
    NoActionAvailable{
        context: String
    },
    #[error("Tensor operation error: {error}")]
    Tensor{
        #[source]
        error: TensorError,
    },
    #[error("Data error: {error}")]
    Data{
        #[source]
        error: DataError,
    },
    //#[error("External: {0}")]
    //External(String)
}

