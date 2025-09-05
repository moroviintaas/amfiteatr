use tch::Tensor;
use amfiteatr_core::scheme::Scheme;
use amfiteatr_core::error::AmfiteatrError;
use crate::tensor_data::{TensorIndexI64Encoding, MultiTensorIndexI64Encoding};


/// Trait for information sets to build illegal action mask.
/// Mask is in form of [`Tensor`] with [`kind::Bool`](tch::Kind::Bool) in the size matching action distribution size.
pub trait MaskingInformationSetAction<DP: Scheme, Ctx: TensorIndexI64Encoding>{

    fn try_build_mask(&self, ctx: &Ctx) -> Result<Tensor, AmfiteatrError<DP>>;
}

/// Trait for information sets to build illegal action masks for actions built from more than one parameter..
/// Each parameter mask is in form of [`Tensor`] with [`kind::Bool`](tch::Kind::Bool) in the size matching parameter distribution size.
/// Masks are gathered in `[Vec]` of length matching the number of action parameters.
pub trait MaskingInformationSetActionMultiParameter<DP: Scheme, Ctx: MultiTensorIndexI64Encoding>{
    fn try_build_masks(&self, ctx: &Ctx) -> Result<Vec<Tensor>, AmfiteatrError<DP>>;
}

