use thiserror::Error;
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;


/// Error from tensor processing crate.
#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum TensorError{
    #[error("Action convert in context: \"{context}\", origin: \"{origin:}\"")]
    Torch{
        origin: String,
        context: String,
    }
}



impl<S: Scheme> From<TensorError> for AmfiteatrError<S>{
    fn from(source: TensorError) -> AmfiteatrError<S>{
        AmfiteatrError::Tensor{
            error: source,
        }
    }
}


impl TensorError{
    #[cfg(feature = "torch")]
    pub fn from_tch_with_context(error: tch::TchError, context: String) -> Self{
        Self::Torch { origin: format!("{error}"), context }
    }


}
/*
#[cfg(feature = "torch")]
impl From<tch::TchError> for TensorError{
    fn from(source: tch::TchError) -> TensorError{
        TensorError::Torch {
            transcript: format!("{}", source),

            context: "Unspecified - probably due to '?' cast.".to_string(),
        }
    }
}




#[cfg(feature = "torch")]
impl<S: DomainParameters>  From<tch::TchError> for AmfiteatrError<S>{
    fn from(source: tch::TchError) -> AmfiteatrError<S>{
        AmfiteatrError::Tensor {
            error: TensorError::Torch {
                transcript: format!("{}", source),
                context: "Unspecified - probably due to '?' cast.".to_string(),
            },
        }
    }
}
*/