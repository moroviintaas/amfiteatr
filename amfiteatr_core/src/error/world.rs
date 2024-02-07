use thiserror::Error;
use crate::error::AmfiError;
use crate::domain::DomainParameters;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum WorldError<DP: DomainParameters>{

    //#[error("Failed joining thread for agent: {0}")]
    //FailedJoinAgent(DP::AgentId),
    #[error("Agent's Id: {0} is duplicated")]
    DuplicateId(DP::AgentId),
    #[error("Missing Agent's Id: {0}")]
    MissingId(DP::AgentId),
    #[error("Missing environment initial state")]
    MissingState,
    //#[error("Missing action processing function")]
    //MissingActionProcessingFunction,
    //#[error("Failed locking mutex for agent")]
    //AgentMutexLock,
}

impl<DP: DomainParameters> From<WorldError<DP>> for AmfiError<DP>{
    fn from(value: WorldError<DP>) -> Self {
        Self::World(value)
    }
}