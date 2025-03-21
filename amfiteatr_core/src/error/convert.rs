use tch::TchError;
use thiserror::Error;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;


/// Error type for data conversion - mainly to tensor for ML operations.
#[derive(Debug, Clone, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum ConvertError{
    #[error("Converting from tensor {context}. Source context: {origin}")]
    ConvertFromTensor{
        origin: String,
        context: String
    },
    #[error("Converting to tensor: {context}. Source context: {origin}.")]
    ConvertToTensor{
        origin: String,
        context: String
    },
    #[error("Information set {info_set:?} cannot be fit into tensor of shape {shape:?}.")]
    InfoSetNotFit{
        info_set: String,
        shape: Vec<i64>
    },
    #[error("Vector normalisation error: {comment:}")]
    VectorNormalisation{
        comment: String
    },
    #[error("Bad parameter index {index:}")]
    BadParameterIndex{
        index: usize,
    },
    #[error("Entity: {entity:}, conversion to Tensor with context {context} is illegal")]
    IllegalConversion{
        entity: String,
        context: String,
    },

    /// Dump of `TchError` from crate `tch` (optional dependency) to string.
    #[error("Torch string")]
    TorchStr{
        origin: String
    }

}



impl<DP: DomainParameters> From<ConvertError> for AmfiteatrError<DP>{
    fn from(error: ConvertError) -> Self {
        AmfiteatrError::DataConvert(error.into())
    }
}


#[cfg(feature = "torch")]

impl From<tch::TchError> for ConvertError{
    fn from(value: TchError) -> Self {
        Self::TorchStr {origin: format!("{value}")}
    }
}

/*

impl<DP: DomainParameters, E: Into<ConvertError>> From<E> for AmfiteatrError<DP>{
    fn from(error: E) -> Self {
        AmfiteatrError::DataConvert(error.into())
    }
}


 */