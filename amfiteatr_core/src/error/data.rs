use thiserror::Error;
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;


/// Data processing error, **likely to be merged in some different category later. **
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
impl<S: Scheme> From<DataError> for AmfiteatrError<S>{
    fn from(source: DataError) -> AmfiteatrError<S>{
        AmfiteatrError::Data{
            error: source,
        }
    }
}