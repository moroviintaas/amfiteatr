use thiserror::Error;
use crate::error::AmfiteatrError;
use crate::scheme::Scheme;

/// Errors in handling higher level models. For example when using builder to create structs with agents and environment, but some data is missing to construct model.

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum ModelError<S: Scheme>{

    #[error("Agent's Id: {0} is duplicated")]
    DuplicateId(S::AgentId),
    #[error("Missing Agent's Id: {0}")]
    MissingId(S::AgentId),
    #[error("Missing environment initial state")]
    MissingState,
}

impl<S: Scheme> From<ModelError<S>> for AmfiteatrError<S>{
    fn from(value: ModelError<S>) -> Self {
        Self::Model { source: value}
    }
}