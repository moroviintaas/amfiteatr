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
    #[error("Basic amfiteatr error: {source}")]
    Amfiteatr {
        #[source]
        source:AmfiteatrError<DP>
    },
    /// Variant wrapping error captured by [`tch`]
    #[error("Torch error: {source} in context: {context:}")]
    Torch{
        #[source]
        source: TchError,
        context: String
    },
    /// Error with tensor representation
    #[error("Tensor representation: {source}")]
    TensorRepresentation{
        #[source]
        source:TensorRepresentationError
    },
    #[error("Input/Output Error")]
    IO(String),
    #[error("Empty training data")]
    NoTrainingData,


}

impl<DP: DomainParameters> From<TchError> for AmfiteatrRlError<DP>{
    fn from(value: TchError) -> Self {
        Self::Torch{
            source: value,
            context: String::from("unspecified")
        }
    }
}

impl<DP: DomainParameters> From<AmfiteatrError<DP>> for AmfiteatrRlError<DP>{
    fn from(value: AmfiteatrError<DP>) -> Self {
        Self::Amfiteatr{source: value}
    }
}

impl<DP: DomainParameters> From<AmfiteatrRlError<DP>> for AmfiteatrError<DP>{
    fn from(value: AmfiteatrRlError<DP>) -> Self {
        match value{
            AmfiteatrRlError::Amfiteatr{source: n} => n,
            any => AmfiteatrError::Custom(format!("{:?}", any))
        }
    }
}