use tch::TchError;
use amfiteatr_core::domain::DomainParameters;
use crate::error::AmfiRLError;
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

}

impl<DP: DomainParameters> From<TensorRepresentationError> for AmfiRLError<DP>{
    fn from(value: TensorRepresentationError) -> Self {
        AmfiRLError::TensorRepresentation(value)
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