use thiserror::Error;
use crate::error::{CommunicationError, ConvertError, DataError, ProtocolError, TrajectoryError, ModelError};
use crate::domain::{DomainParameters};
use crate::error::tensor::TensorError;

/// Top level crate error, constructed from more specific error.
#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum AmfiteatrError<DP: DomainParameters>{
    /// Error occurring in specific game logic, defined in generic parameter `DP:` [`DomainParameters`](crate::domain::DomainParameters).
    #[error("Game error: {source}")]
    Game{
        #[source]
        source: DP::GameErrorType
    },
    #[error("Agent {agent} caused game error: {source}")]
    /// Similarly like [`Game`](crate::error::AmfiteatrError::Game), but also with pointing out agent who caused error.
    GameA{
        #[source]
        source: DP::GameErrorType,
        agent: DP::AgentId
    },
    /// Error in communication between agent and environment.
    #[error("Communication error: {source}")]
    Communication{
        #[source]
        source: CommunicationError<DP>
    },
    /// General protocol violation, e.g .when agent makes action when it is not his turn.
    #[error("Protocol error: {source}")]
    Protocol{
        #[source]
        source: ProtocolError<DP>
    },

    //#[error("Setup error: {0}")]
    //Setup(SetupError<DP>),
    /// Error during data conversion - typically when encoding as tensors or decoding from them.
    #[error("Data convert - {0}")]
    DataConvert(ConvertError),

    /// Higher level model error.
    #[error("World maintenance error: {source}")]
    Model {
        #[source]
        source: ModelError<DP>
    },

    /// Custom error to return if error does not fir any other category.
    #[error("Custom: {0}")]
    Custom(String),

    /// Error on locking shared object. This is a type to build from [`TryLockError`](std::sync::TryLockError).
    #[error("Lock error on {object:} with {description:}")]
    Lock{
        description: String,
        object: String,
    },

    /// Error in maintaining trajectory of game or agent.
    #[error("Trajectory maintenance error: {source:}")]
    Trajectory{
        #[source]
        source: TrajectoryError<DP>
    },
    /// Error originating in [`nom`](nom) crate.
    #[error("Error in nom parser: {explanation:}")]
    Nom{
        explanation: String
    },
    /// Standard Input/Output error.
    #[error("Error in I/O operation: {explanation:}")]
    IO{
        explanation: String
    },
    /// Special error for policies to generate when no action is available.
    #[error("Impossible action")]
    NoActionAvailable{
        context: String
    },
    /// Errors in tensor processing.
    #[error("Tensor operation error: \"{error}\"")]
    Tensor{
        #[source]
        error: TensorError,
    },

    /// Error in general data processing - **maybe merged with ConverError** in the future.
    #[error("Data error: {error}")]
    Data{
        #[source]
        error: DataError,
    },
    //#[error("External: {0}")]
    //External(String)
}

