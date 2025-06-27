use amfiteatr_classic::agent::{LocalHistoryConversionToTensor, LocalHistoryInfoSet, LocalHistoryInfoSetNumbered};
use amfiteatr_classic::ClassicActionTensorRepresentation;
use amfiteatr_rl::policy::PolicyDiscretePPO;
use crate::replicators::model::ReplDomain;

pub type ReplPPO = PolicyDiscretePPO<
    ReplDomain,
    LocalHistoryInfoSetNumbered,
    LocalHistoryConversionToTensor,
    ClassicActionTensorRepresentation
>;