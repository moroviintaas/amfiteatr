use tch::Tensor;
use amfiteatr_core::demo::{DemoAction, DemoInfoSet};
use amfiteatr_core::error::ConvertError;
use crate::error::TensorRepresentationError;
use crate::tensor_data::{ActionTensor, ConvertToTensor, ConversionToTensor};

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

impl ConvertToTensor<DemoConversionToTensor> for DemoInfoSet{
    fn try_to_tensor(&self, _way: &DemoConversionToTensor) -> Result<Tensor, TensorRepresentationError> {
        Ok(Tensor::from_slice(&[1.0]))
    }
}

impl ActionTensor for DemoAction{
    fn to_tensor(&self) -> Tensor {
        Tensor::from_slice(&[self.0 as f32])
    }

    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError> {
        let v: Vec<f32> = Vec::try_from(t).unwrap();
        Ok(DemoAction{0: v[0] as u8})
    }
}