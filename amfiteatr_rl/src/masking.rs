use std::error::Error;
use tch::Tensor;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::tensor_data::{ConversionToIndexI64, ConversionToMultiIndexI64};

pub trait MaskingInformationSetAction<Ctx: ConversionToIndexI64>{
    type MaskingError: Error;

    fn try_build_mask(&self, ctx: &Ctx) -> Result<Option<Tensor>, Self::MaskingError>;
}

pub trait MaskingInformationSetActionMultiParameter<Ctx: ConversionToMultiIndexI64>{
    type Error: Error;

    fn try_build_masks(&self, ctx: &Ctx) -> Result<Option<Vec<Tensor>>, Self::Error>;
}