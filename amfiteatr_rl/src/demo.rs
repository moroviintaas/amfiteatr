use tch::Tensor;
use amfiteatr_core::demo::{DemoInfoSet};
use crate::error::TensorRepresentationError;
use crate::tensor_data::{CtxTryIntoTensor, ConversionToTensor};

/// Demonstration conversion to tensor meant for [`DemoInfoSet`].
/// This is made only to demonstrate syntax for examples making sense please refer to
/// examples linked in crate's root.
#[derive(Default, Copy, Clone, Debug)]
pub struct DemoConversionToTensor {}



impl ConversionToTensor for DemoConversionToTensor {
    fn desired_shape(&self) -> &[i64] {
        &[1]
    }
}

impl CtxTryIntoTensor<DemoConversionToTensor> for DemoInfoSet{
    fn try_to_tensor(&self, _way: &DemoConversionToTensor) -> Result<Tensor, TensorRepresentationError> {
        Ok(Tensor::from_slice(&[1.0]))
    }
}

