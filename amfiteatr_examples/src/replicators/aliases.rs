use amfiteatr_classic::agent::{LocalHistoryConversionToTensor, LocalHistoryInfoSetNumbered};
use amfiteatr_classic::ClassicActionTensorRepresentation;
use amfiteatr_rl::policy::PolicyDiscretePPO;
use crate::replicators::model::ReplScheme;

pub type ReplPPO = PolicyDiscretePPO<
    ReplScheme,
    LocalHistoryInfoSetNumbered,
    LocalHistoryConversionToTensor,
    ClassicActionTensorRepresentation
>;