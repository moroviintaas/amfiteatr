use thiserror::Error;

#[derive(Debug, Clone, Error)]
pub enum ConvertError{
    #[error("Action convert from {0}")]
    ActionDeserialize(String)
}