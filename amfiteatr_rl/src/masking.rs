use tch::Tensor;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::tensor_data::{TensorIndexI64Encoding, MultiTensorIndexI64Encoding};

pub trait MaskingInformationSetAction<DP: DomainParameters, Ctx: TensorIndexI64Encoding>{

    fn try_build_mask(&self, ctx: &Ctx) -> Result<Tensor, AmfiteatrError<DP>>;
}

pub trait MaskingInformationSetActionMultiParameter<DP: DomainParameters, Ctx: MultiTensorIndexI64Encoding>{
    fn try_build_masks(&self, ctx: &Ctx) -> Result<Vec<Tensor>, AmfiteatrError<DP>>;
}

