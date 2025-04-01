use thiserror::Error;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum LearningError{
    #[error("Empty trajectory")]
    EmptyTrajectory,
}


impl<DP: DomainParameters> From<LearningError> for AmfiteatrError<DP>{
    fn from(err: LearningError) -> Self {
        Self::Learning {
            error: err
        }
    }
}