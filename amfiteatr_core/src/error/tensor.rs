use thiserror::Error;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;

#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum TensorError{
    #[error("Action convert from {context:}")]
    Torch{
        context: String
    }
}

impl<DP: DomainParameters> From<TensorError> for AmfiteatrError<DP>{
    fn from(source: TensorError) -> AmfiteatrError<DP>{
        AmfiteatrError::Tensor{
            error: source,
        }
    }
}
#[cfg(feature = "torch")]
impl From<tch::TchError> for TensorError{
    fn from(source: tch::TchError) -> TensorError{
        TensorError::Torch {
            context: format!("{}", source)
        }
    }
}

#[cfg(feature = "torch")]
impl<DP: DomainParameters>  From<tch::TchError> for AmfiteatrError<DP>{
    fn from(source: tch::TchError) -> AmfiteatrError<DP>{
        AmfiteatrError::Tensor {
            error: TensorError::Torch {
                context: format!("{}", source)
            },
        }
    }
}