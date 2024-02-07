use std::error::Error;
use std::fmt::{Debug};
use tch::{Tensor};
use amfiteatr_core::domain::{Action, Reward};
use amfiteatr_core::error::{ConvertError};
use crate::error::TensorRepresentationError;


/// Extension for [`ConversionToTensor`] to actually produce tensor. However implementing this trait
/// is optional, because generic implementations require [`ConvertToTensor`] on type converted,
/// and it is probably what you need to implement.
/// However it may be sometimes convenient to implement this trait for converter and then use it on
/// converted types.
pub trait SimpleConvertToTensor<T>: Send + ConversionToTensor{
    /// Take reference to associated type and output tensor
    fn make_tensor(&self, t: &T) -> Tensor;
}




/// Trait representing structs (maybe 0-sized) that tell what is desired shape of tensor produced
/// by conversion to tensor when using [`ConvertToTensor`].
pub trait ConversionToTensor: Send + Default{
    /// Returns shape (slice of i64 numbers) of shape that must be produced for network
    fn desired_shape(&self) -> &[i64];

    /// Returns shape (slice of i64 numbers) of shape that must be produced for network but flatten
    /// to one dimension
    fn desired_shape_flatten(&self) -> i64{
        self.desired_shape().iter().product()
    }
}

/// Trait for structs that can be represented as traits, for example game information sets (game states).
/// It could be resolved by requiring `&T: Into<Tensor>`, however using associated type
/// [`ConversionToTensor`] allows to implement different conversion. This may be useful when you
/// implement information set however you want several tensor forms for experiments.
/// For example in some card game you may represent only current _certain_ information
/// and compare it with model utilizing some assumptions on _uncertain_ information like
/// _"enemy has king of hearts with probability of 40%"_.
/// In this case you can implement one [`InformationSet`](amfiteatr_core::agent::InformationSet) and two ways
/// of converting it.
pub trait ConvertToTensor<W: ConversionToTensor> : Debug{
    fn try_to_tensor(&self, way: &W) -> Result<Tensor, TensorRepresentationError>;

    fn to_tensor(&self, way: &W) -> Tensor{
        self.try_to_tensor(way).unwrap()
    }
    fn try_to_tensor_flat(&self, way: &W) -> Result<Tensor, TensorRepresentationError>{
        let t1 = self.try_to_tensor(way)?;
        t1.f_flatten(0, -1).map_err(|e|{
            TensorRepresentationError::Torch {
                error: e,
                context: format!("Flattening tensor {t1:?} from information set: {:?}", self)
            }
        })
    }

    fn to_tensor_flat(&self, way: &W) -> Tensor{
        let t1 = self.to_tensor(way);
        //let dim = t1.dim() as i64;
        t1.flatten(0, -1)
    }
    fn tensor_shape(way: &W) -> &[i64]{
        way.desired_shape()
    }
    fn tensor_length_flatten(way: &W) -> i64{
        way.desired_shape().iter().product()
    }
}

impl<W: ConversionToTensor, T: ConvertToTensor<W>> ConvertToTensor<W> for Box<T>{
    fn try_to_tensor(&self, way: &W) -> Result<Tensor, TensorRepresentationError> {
        self.as_ref().try_to_tensor(way)
    }
}

/// Trait representing structs (maybe 0-sized) that tell what is expected size of tensor to be
/// used to create data struct using [`TryConvertFromTensor`].
pub trait ConversionFromTensor: Send{
    fn expected_input_shape() -> &'static[i64];
}

/// Implemented by structs that can be converted from tensors.
/// Certain data type can have different tensor representation, then it is needed to specify
/// what particular representation is used (done by using correct [`ConversionFromTensor`].
pub trait TryConvertFromTensor<W: ConversionFromTensor>{
    type ConvertError: Error;
    fn try_from_tensor(tensor: &Tensor, way: &W) -> Result<Self, Self::ConvertError> where Self: Sized;
}

/*
pub trait ConvertToTensorD<W: ConversionToTensor>: ConvertToTensor<W> + Display + Debug{}
impl<W: ConversionToTensor, T: ConvertToTensor<W> + Display + Debug> ConvertToTensorD<W> for T{}
*/

/// Trait dedicated for actions that can be written as tensor and read from tensors (usually it
/// will be some index in action space. Actual implementations expects that actions are represented
/// as Tensor with type [`Float`](tch::Kind::Float).
pub trait ActionTensor: Action{

    fn to_tensor(&self) -> Tensor;
    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError>;
}

/// Trait dedicated to rewards (payoffs) that can be represented as tensor of floats
pub trait FloatTensorReward: Reward{
    //type Dims: IntList;
    fn to_tensor(&self) -> Tensor;
    //fn shape(&self) -> Dims;
    fn shape() -> Vec<i64>;
    fn total_size() -> i64{
        Self::shape().iter().fold(0, |acc, x| acc+x)
    }
}

macro_rules! impl_reward_std_f {
    ($($x: ty), +) => {
        $(
        impl FloatTensorReward for $x{

            fn to_tensor(&self) -> Tensor {
                let s = [*self as f32;1];
                Tensor::from_slice(&s[..])

            }

            fn shape() -> Vec<i64> {
                vec![1]
            }
        }

        )*

    }
}

impl_reward_std_f![f32, f64];


impl FloatTensorReward for i64{

    fn to_tensor(&self) -> Tensor {
        let s = [*self as f32;1];
        Tensor::from_slice(&s[..])

    }

    fn shape() -> Vec<i64> {
        vec![1]
    }
}

impl FloatTensorReward for i32{

    fn to_tensor(&self) -> Tensor {
        let s = [*self as f32];
        Tensor::from_slice(&s[..])

    }

    fn shape() -> Vec<i64> {
        vec![1]
    }
}

