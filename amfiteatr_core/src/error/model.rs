use thiserror::Error;
use crate::error::AmfiteatrError;
use crate::domain::DomainParameters;

/// Errors in handling higher level models. For example when using builder to create structs with agents and environment, but some data is missing to construct model.

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum ModelError<DP: DomainParameters>{

    #[error("Agent's Id: {0} is duplicated")]
    DuplicateId(DP::AgentId),
    #[error("Missing Agent's Id: {0}")]
    MissingId(DP::AgentId),
    #[error("Missing environment initial state")]
    MissingState,
}

impl<DP: DomainParameters> From<ModelError<DP>> for AmfiteatrError<DP>{
    fn from(value: ModelError<DP>) -> Self {
        Self::Model { source: value}
    }
}