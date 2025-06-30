use tboard::Error;
use thiserror::Error;
use amfiteatr_core::error::AmfiteatrError;
use crate::replicators::model::ReplDomain;

#[derive(Error, Debug, Clone)]
pub enum ReplError {
    #[error("Amfiteatr error: {0}")]
    Amfiteatr(AmfiteatrError<ReplDomain>),
    #[error("Builder missing parameter {0}")]
    MissingParameter(String),
    #[error("Agent duplication {0}")]
    AgentDuplication(u32),
    #[error("The number of agents cant be odd number ({0})")]
    OddAgentNumber(usize),
    #[error("Policy builder error: {0}")]
    PolicyBuilderError(String),
    //#[error("Tensorboard error: {0}")]
    //TensorBoard(tboard::Error),
    //#[error("Classic)]
}

impl From<AmfiteatrError<ReplDomain>> for ReplError {
    fn from(value: AmfiteatrError<ReplDomain>) -> Self {
        Self::Amfiteatr(value)
    }
}
/*
impl From<tboard::Error> for ReplError {
    fn from(value: Error) -> Self {
        Self::TensorBoard(value.into())
    }
}

 */