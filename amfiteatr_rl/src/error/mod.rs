mod tensor_repr;

pub use tensor_repr::*;

use tch::TchError;
use thiserror::Error;
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_core::scheme::Scheme;


/// Error trait that wraps standard [`AmfiteatrError`]
#[derive(Error, Debug)]
pub enum AmfiteatrRlError<S: Scheme>{
    /// Variant - [`AmfiteatrError`]
    #[error("Basic amfiteatr error: {source}")]
    Amfiteatr {
        #[source]
        source:AmfiteatrError<S>
    },
    /// Variant wrapping error captured by [`tch`]
    #[error("Torch error: {source} in context: {context:}")]
    Torch{
        #[source]
        source: TchError,
        context: String
    },
    /// Error with tensor representation
    ///  #[deprecated(since = "0.8.0", note = "Migrating to [`ConvertError`] in [`AmfiteatrError`]")]
    #[error("Tensor representation: {source}")]
    TensorRepresentation{
        #[source]
        source:TensorRepresentationError
    },
    #[error("Mismatched length {shape1:?} and {shape2:?} in {context:}.")]
    MismatchedLengthsOfData{
        shape1: usize,
        shape2: usize,
        context: String
    },
    #[error("Batch size is 0 with context {context:}")]
    ZeroBatchSize{
        context: String
    },
    #[error("Input/Output Error")]
    IO(String),
    #[error("Empty training data")]
    NoTrainingData,


}

impl<S: Scheme> From<TchError> for AmfiteatrRlError<S>{
    fn from(value: TchError) -> Self {
        Self::Torch{
            source: value,
            context: String::from("unspecified")
        }
    }
}

impl<S: Scheme> From<AmfiteatrError<S>> for AmfiteatrRlError<S>{
    fn from(value: AmfiteatrError<S>) -> Self {
        Self::Amfiteatr{source: value}
    }
}

impl<S: Scheme> From<AmfiteatrRlError<S>> for AmfiteatrError<S>{
    fn from(value: AmfiteatrRlError<S>) -> Self {
        match value{
            AmfiteatrRlError::Amfiteatr{source: n} => n,
            any => AmfiteatrError::Custom(format!("{:?}", any))
        }
    }
}

impl<S: Scheme> From<ConvertError> for AmfiteatrRlError<S>{
    fn from(value: ConvertError) -> Self {
        let ae = AmfiteatrError::from(value);
        ae.into()
    }
}