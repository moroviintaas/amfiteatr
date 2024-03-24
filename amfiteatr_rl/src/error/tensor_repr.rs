use tch::TchError;
use amfiteatr_core::domain::DomainParameters;
use crate::error::AmfiteatrRlError;
use thiserror::Error;


/// Error in vectorisation of data to tensor
#[derive(Error, Debug)]
pub enum TensorRepresentationError{
    #[error("Information set {info_set:?} cannot be fit into tensor of shape {shape:?}.")]
    InfoSetNotFit{
        info_set: String,
        shape: Vec<i64>
    },
    #[error("Error originating in tch crate's function: {error:}, in context: {context:}")]
    Torch{
        error: TchError,
        context: String
    },
    #[error("Conversion of value to tensor is not supported ({comment:})")]
    ConversionToTensor{
        comment: String,

    }

}

impl<DP: DomainParameters> From<TensorRepresentationError> for AmfiteatrRlError<DP>{
    fn from(value: TensorRepresentationError) -> Self {
        AmfiteatrRlError::TensorRepresentation(value)
    }
}
impl From<TchError> for TensorRepresentationError{
    fn from(value: TchError) -> Self {
        Self::Torch{
            error: value,
            context: "unspecified".into()
        }
    }
}