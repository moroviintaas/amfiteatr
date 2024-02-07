
use crate::error::AmfiError;
use crate::domain::DomainParameters;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum SetupError<Spec: DomainParameters>{
    #[error("Agent's Id: {0} is duplicated")]
    DuplicateId(Spec::AgentId),
    #[error("Missing Agent's Id: {0}")]
    MissingId(Spec::AgentId),
    #[error("Missing environment initial state")]
    MissingState,
    #[error("Missing action processing function")]
    MissingActionProcessingFunction,
    #[error("Failed locking mutex for agent")]
    AgentMutexLock,

}
impl<Spec: DomainParameters> From<SetupError<Spec>> for AmfiError<Spec>{
    fn from(value: SetupError<Spec>) -> Self {
        AmfiError::Setup(value)
    }
}