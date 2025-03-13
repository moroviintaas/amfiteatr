use std::error::Error;
use tch::Tensor;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use crate::tensor_data::{ConversionToIndexI64, ConversionToMultiIndexI64};

pub trait MaskingInformationSetAction<DP: DomainParameters, Ctx: ConversionToIndexI64>{

    fn try_build_mask(&self, ctx: &Ctx) -> Result<Tensor, AmfiteatrError<DP>>;
}

pub trait MaskingInformationSetActionMultiParameter<DP: DomainParameters, Ctx: ConversionToMultiIndexI64>{
    fn try_build_masks(&self, ctx: &Ctx) -> Result<Vec<Tensor>, AmfiteatrError<DP>>;
}

