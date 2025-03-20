use thiserror::Error;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum DataError{
    #[error("Data should have the same length, but left is {left:} and right is {right:}. {context:}")]
    LengthMismatch{
        left: usize,
        right: usize,
        context: String,
    }
}
impl<DP: DomainParameters> From<DataError> for AmfiteatrError<DP>{
    fn from(source: DataError) -> AmfiteatrError<DP>{
        AmfiteatrError::Data{
            error: source,
        }
    }
}