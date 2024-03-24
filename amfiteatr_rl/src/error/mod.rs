mod tensor_repr;
pub use tensor_repr::*;

use tch::TchError;
use thiserror::Error;
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::domain::DomainParameters;


/// Error trait that wraps standard [`AmfiteatrError`]
#[derive(Error, Debug)]
pub enum AmfiteatrRlError<DP: DomainParameters>{
    /// Variant - [`AmfiteatrError`]
    #[error("Basic amfiteatr error: {0}")]
    Amfi(AmfiteatrError<DP>),
    /// Variant wrapping error captured by [`tch`]
    #[error("Torch error: {error} in context: {context:}")]
    Torch{
        error: TchError,
        context: String
    },
    /// Error with tensor representation
    #[error("Tensor representation: {0}")]
    TensorRepresentation(TensorRepresentationError),
    #[error("Input/Output Error")]
    IO(String),
    #[error("Empty training data")]
    NoTrainingData,


}

impl<DP: DomainParameters> From<TchError> for AmfiteatrRlError<DP>{
    fn from(value: TchError) -> Self {
        Self::Torch{
            error: value,
            context: String::from("unspecified")
        }
    }
}

impl<DP: DomainParameters> From<AmfiteatrError<DP>> for AmfiteatrRlError<DP>{
    fn from(value: AmfiteatrError<DP>) -> Self {
        Self::Amfi(value)
    }
}

impl<DP: DomainParameters> From<AmfiteatrRlError<DP>> for AmfiteatrError<DP>{
    fn from(value: AmfiteatrRlError<DP>) -> Self {
        match value{
            AmfiteatrRlError::Amfi(n) => n,
            any => AmfiteatrError::Custom(format!("{:?}", any))
        }
    }
}