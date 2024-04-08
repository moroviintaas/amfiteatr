use thiserror::Error;
use crate::error::AmfiteatrError;
use crate::domain::DomainParameters;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum WorldError<DP: DomainParameters>{

    #[error("Agent's Id: {0} is duplicated")]
    DuplicateId(DP::AgentId),
    #[error("Missing Agent's Id: {0}")]
    MissingId(DP::AgentId),
    #[error("Missing environment initial state")]
    MissingState,
}

impl<DP: DomainParameters> From<WorldError<DP>> for AmfiteatrError<DP>{
    fn from(value: WorldError<DP>) -> Self {
        Self::World{ source: value}
    }
}