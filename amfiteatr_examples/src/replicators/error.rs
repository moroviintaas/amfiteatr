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
    //#[error("Classic)]
}

impl From<AmfiteatrError<ReplDomain>> for ReplError {
    fn from(value: AmfiteatrError<ReplDomain>) -> Self {
        Self::Amfiteatr(value)
    }
}