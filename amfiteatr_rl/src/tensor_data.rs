use std::error::Error;
use std::fmt::{Debug};
use generic_array::{ArrayLength, GenericArray};
use tch::{TchError, Tensor};
use amfiteatr_core::domain::{Action, DomainParameters, Reward};
use amfiteatr_core::error::{AmfiteatrError, ConvertError, TensorError};
use crate::error::TensorRepresentationError;


/// Extension for [`ConversionToTensor`] to actually produce tensor. However, implementing this trait
/// is optional, because generic implementations require [`ContextTryIntoTensor`] on type converted,
/// and it is probably what you need to implement.
/// However, it may be sometimes convenient to implement this trait for converter and then use it on
/// converted types.
pub trait SimpleConvertToTensor<T>: Send + ConversionToTensor {
    /// Take reference to associated type and output tensor
    fn make_tensor(&self, t: &T) -> Tensor;
}




/// Trait representing structs (maybe 0-sized) that tell what is desired shape of tensor produced
/// by conversion to tensor when using [`ContextTryIntoTensor`].
pub trait ConversionToTensor: Send{
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
pub trait ContextTryIntoTensor<Ctx: ConversionToTensor> : Debug{
    fn try_to_tensor(&self, way: &Ctx) -> Result<Tensor, ConvertError>;

    fn to_tensor(&self, way: &Ctx) -> Tensor{
        self.try_to_tensor(way).unwrap()
    }
    fn try_to_tensor_flat(&self, way: &Ctx) -> Result<Tensor, ConvertError>{
        let t1 = self.try_to_tensor(way)?;
        t1.f_flatten(0, -1).map_err(|e|{
            ConvertError::ConvertToTensor {
                origin: format!("{e}"),
                context: format!("Flattening tensor {t1:?} from information set: {:?}", self)
            }.into()
        })
    }

    fn to_tensor_flat(&self, way: &Ctx) -> Tensor{
        let t1 = self.to_tensor(way);
        //let dim = t1.dim() as i64;
        t1.flatten(0, -1)
    }
    fn tensor_shape(way: &Ctx) -> &[i64]{
        way.desired_shape()
    }
    fn tensor_length_flatten(way: &Ctx) -> i64{
        way.desired_shape().iter().product()
    }
}

/// Represents format for converting element to set of Tensors
pub trait ConversionToMultipleTensors: Send{
    fn expected_outputs_shape(&self) -> &[Vec<i64>];
}

pub trait ContextTryIntoMultipleTensors<Ctx: ConversionToMultipleTensors> : Debug{
    fn try_to_multiple_tensors(&self, form: &Ctx) -> Result<Vec<Tensor>, ConvertError>;
    fn to_multiple_tensors(&self, form: &Ctx) -> Vec<Tensor>{
        self.try_to_multiple_tensors(form).unwrap()
    }
    /*
    fn shapes(&self, form: &W) -> &[Vec<i64>]{
        form.expected_outputs_shape()
    }

     */
}

pub trait ConversionToIndexI64: Send{
    fn min(&self) -> i64;
    fn limit(&self) -> i64;
}

pub trait ContextTryIntoIndexI64<Ctx: ConversionToIndexI64> : Debug{

    fn try_to_index(&self, way: &Ctx) -> Result<i64, ConvertError>;
}

pub trait ConversionToMultiIndexI64{
    fn min(&self, param_index: usize) -> Option<i64>;
    fn limit(&self, param_index: usize) -> Option<i64>;

    fn number_of_params() -> usize;


}

impl<T: ConversionToMultiIndexI64 + ConversionFromMultipleTensors> ActionTensorFormat for T{
    type TensorForm = Vec<Tensor>;

    fn param_dimension_size(&self) -> i64 {
        Self::number_of_params() as i64
    }
}

pub trait ContextTryIntoMultiIndexI64<Ctx: ConversionToMultiIndexI64>{
    fn param_value(&self, context: &Ctx, param_index: usize) -> Result<Option<i64>, ConvertError>;
    fn default_value(&self, _context: &Ctx, _param_index: usize) -> i64{
        0
    }





    fn action_indexes(&self, context: &Ctx) -> Result<Vec<Option<i64>>, ConvertError>{

        Ok((0..Ctx::number_of_params()).map(|i|{
            self.param_value(context, i)
        }).collect::<Result<Vec<Option<i64>>, _>>()?)
    }



    fn action_index_and_mask_tensor_vecs(&self, context: &Ctx) -> Result<(Vec<Tensor>, Vec<Tensor>),ConvertError>{
        let mut params = Vec::new();
        let mut usage_masks = Vec::new();

        for i in 0..Ctx::number_of_params(){
            match self.param_value(context, i)?{
                Some(p) => {
                    params.push(Tensor::from(p));
                    usage_masks.push(Tensor::from(true));
                },
                None => {
                    params.push(Tensor::from(self.default_value(context, i)));
                    usage_masks.push(Tensor::from(false));
                }
            }

        }

        #[cfg(feature = "log_trace")]
        log::trace!("Action param[0] = {:?}", params[0]);
        Ok((params, usage_masks))

    }

    fn batch_index_and_mask_tensor_vecs(actions: &[&Self], context: &Ctx) -> Result<(Vec<Tensor>, Vec<Tensor>), ConvertError>
    {
        let mut params = Vec::new();
        let mut usage_masks = Vec::new();

        for i in 0..Ctx::number_of_params(){

            let mut action_params = Vec::new();
            let mut action_usage_masks = Vec::new();
            for action in actions{
                match action.param_value(context, i)?{
                    Some(p) => {
                        action_params.push(p);
                        action_usage_masks.push(true);
                    },
                    None => {
                        action_params.push(action.default_value(context, i));
                        action_usage_masks.push(false);
                    }
                }
            }
            let t_actions = Tensor::from_slice(&action_params);
            let t_usage_masks = Tensor::from_slice(&action_usage_masks);
            params.push(t_actions);
            params.push(t_usage_masks);

        }
        Ok((params, usage_masks))
    }
    /*
    fn try_to_multi_index(&self, ctx: &W) -> Result<Vec<Option<i64>>, TensorRepresentationError>;
    fn try_multi_index_tensor(&self, ctx: &W) -> Result<Vec<Option<Tensor>>, TensorRepresentationError>{
        self.try_to_multi_index(ctx).map(|r|{
            r.iter().map(|option| option.and_then(|i| Some(Tensor::from(i)))).collect()
        })
    }

     */


}

pub trait TensorFormat: ConversionToTensor + ConversionFromTensor{}
impl<T> TensorFormat for T where T: ConversionToTensor + ConversionFromTensor{}
pub trait MultipleTensorFormat: ConversionToMultipleTensors + ConversionFromMultipleTensors{}
impl<T> MultipleTensorFormat for T where T: ConversionToMultipleTensors + ConversionFromMultipleTensors{}



pub trait TryIntoTensor: Debug{


    fn try_to_tensor(&self) -> Result<Tensor, TensorRepresentationError>;
}

impl<W: ConversionToTensor, T: ContextTryIntoTensor<W>> ContextTryIntoTensor<W> for Box<T>{

    fn try_to_tensor(&self, way: &W) -> Result<Tensor, ConvertError> {
        self.as_ref().try_to_tensor(way)
    }
}

/// Trait representing structs (maybe 0-sized) that tell what is expected size of tensor to be
/// used to create data struct using [`ContextTryFromTensor`].
pub trait ConversionFromTensor: Send{
    fn expected_input_shape(&self) -> &[i64];
}

/// Implemented by structs that can be converted from tensors.
/// Certain data type can have different tensor representation, then it is needed to specify
/// what particular contextual representation is used (done by using correct [`ConversionFromTensor`]).
pub trait ContextTryFromTensor<Ctx: ConversionFromTensor>{
    fn try_from_tensor(tensor: &Tensor, way: &Ctx) -> Result<Self, ConvertError> where Self: Sized;
}


/// Implemented by structs that can be converted from tensors.).
pub trait TryFromTensor: for<'a> TryFrom<&'a Tensor, Error=ConvertError> + Sized{
    fn try_from_tensor(tensor: &Tensor) -> Result<Self, ConvertError> where Self: Sized{
        Self::try_from(tensor)
    }
}



impl<T: for<'a> TryFrom<&'a Tensor, Error=ConvertError> + Sized> TryFromTensor for T{

}


pub trait TryFromMultiTensors: for<'a> TryFrom<&'a [Tensor], Error=ConvertError> + Sized{

    fn try_from_multi_tensors(tensors: &[Tensor]) -> Result<Self, ConvertError> where Self: Sized{
        Self::try_from(tensors)
    }
}

impl<T: for<'a> TryFrom<&'a [Tensor], Error=ConvertError> + Sized> TryFromMultiTensors for T{

}


/// Trait representing structs (maybe 0-sized) that tell what is expected size of tensor to be
/// used to create data struct using [`crate::tensor_data::ContextTryFromMultipleTensors`].
pub trait ConversionFromMultipleTensors: Send{
    fn expected_inputs_shape(&self) -> &[Vec<i64>];
}

/// Implemented by structs that can be converted from tensors.
/// Certain data type can have different tensor representation, then it is needed to specify
/// what particular contextual representation is used (done by using correct [`crate::tensor_data::ConversionFromTensor`]).
pub trait ContextTryFromMultipleTensors<Ctx: crate::tensor_data::ConversionFromMultipleTensors>{
    fn try_from_tensors(tensors: &[Tensor], way: &Ctx)
        -> Result<Self, ConvertError> where Self: Sized;
}



pub trait ActionTensorFormat {

    /// Typically it will be just `Tensor` or `Vec<Tensor>`
    type TensorForm;



    //type BatchVecTensorForm;





    /// Set to 1 for `TensorForm` being `Tensor`, and `v.len()` for `Vec<Tensor>`
    fn param_dimension_size(&self) -> i64;





}




/// Trait dedicated for actions that can be written as tensor and read from tensors (usually it
/// will be some index in action space. Actual implementations expects that actions are represented
/// as Tensor with type [`Float`](tch::Kind::Float).
#[deprecated(since = "0.3.0", note = "Implement rather [`TryFromTensor`]")]
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
        Self::shape().iter().sum::<i64>()
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



impl<N: ArrayLength> ConversionToTensor for GenericArray<i64, N>{
    fn desired_shape(&self) -> &[i64] {
        &self[..]
    }
}
impl<N: ArrayLength> ConversionFromTensor for GenericArray<i64, N>{
    fn expected_input_shape(&self) -> &[i64] {
        &self[..]
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

