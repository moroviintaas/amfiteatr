use thiserror::Error;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;

#[derive(Debug, Clone, Error)]
pub enum ConvertError{
    #[error("Action convert from {0}")]
    ConvertFromTensor(String)
}

impl<DP: DomainParameters> From<ConvertError> for AmfiteatrError<DP>{
    fn from(err: ConvertError) -> AmfiteatrError<DP>{
        AmfiteatrError::DataConvert()
    }
}