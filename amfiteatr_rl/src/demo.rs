use tch::Tensor;
use amfiteatr_core::demo::{DemoAction, DemoInfoSet};
use amfiteatr_core::error::ConvertError;
use crate::tensor_data::{ContextDecodeTensor, ContextEncodeIndexI64, ContextEncodeTensor, TensorDecoding, TensorEncoding, TensorIndexI64Encoding};

/// Demonstration conversion to tensor meant for [`DemoInfoSet`].
/// This is made only to demonstrate syntax for examples making sense please refer to
/// examples linked in crate's root.
#[derive(Default, Copy, Clone, Debug)]
pub struct DemoConversionToTensor {}



impl TensorEncoding for DemoConversionToTensor {
    fn desired_shape(&self) -> &[i64] {
        &[1]
    }
}

impl ContextEncodeTensor<DemoConversionToTensor> for DemoInfoSet{

    fn try_to_tensor(&self, _way: &DemoConversionToTensor) -> Result<Tensor, ConvertError> {
        Ok(Tensor::from_slice(&[1.0]))
    }
}

pub struct DemoActionConversionContext{

}

impl TensorDecoding for DemoActionConversionContext{
    fn expected_input_shape(&self) -> &[i64] {
        &[1]
    }
}

impl TensorIndexI64Encoding for DemoActionConversionContext{
    fn min(&self) -> i64 {
        0
    }

    fn limit(&self) -> i64 {
        2
    }
}

impl ContextDecodeTensor<DemoActionConversionContext> for DemoAction{
    fn try_from_tensor(tensor: &Tensor, _way: &DemoActionConversionContext) -> Result<Self, ConvertError> where Self: Sized {
        let t = tensor.int64_value(&[0]);
        match t{
            i @ 0..=255 => Ok(Self(i as u8)),
            e => Err(ConvertError::BadParameterIndex {index: e as usize})
        }
    }
}

impl ContextEncodeIndexI64<DemoActionConversionContext> for DemoAction{
    fn try_to_index(&self, _way: &DemoActionConversionContext) -> Result<i64, ConvertError> {
        Ok(self.0 as i64)
    }
}
