use std::fmt::{Debug};
use generic_array::{ArrayLength, GenericArray};
use tch::Tensor;
use amfiteatr_core::domain::{Action, Reward};
use amfiteatr_core::error::ConvertError;
use crate::error::TensorRepresentationError;


/// Extension for [`TensorEncoding`] to actually produce tensor. However, implementing this trait
/// is optional, because generic implementations require [`ContextEncodeTensor`] on type converted,
/// and it is probably what you need to implement.
/// However, it may be sometimes convenient to implement this trait for converter and then use it on
/// converted types.
pub trait SimpleConvertToTensor<T>: Send + TensorEncoding {
    /// Take reference to associated type and output tensor
    fn make_tensor(&self, t: &T) -> Tensor;
}




/// Trait representing structs (maybe 0-sized) that tell what is desired shape of tensor produced
/// by conversion to tensor when using [`ContextEncodeTensor`].
pub trait TensorEncoding: Send{
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
/// [`TensorEncoding`] allows to implement different conversion. This may be useful when you
/// implement information set however you want several tensor forms for experiments.
/// For example in some card game you may represent only current _certain_ information
/// and compare it with model utilizing some assumptions on _uncertain_ information like
/// _"enemy has king of hearts with probability of 40%"_.
/// In this case you can implement one [`InformationSet`](amfiteatr_core::agent::InformationSet) and two ways
/// of converting it.
pub trait ContextEncodeTensor<Ctx: TensorEncoding> : Debug{

    /// Try convert to tensor using encoding.
    fn try_to_tensor(&self, encoding: &Ctx) -> Result<Tensor, ConvertError>;

    /// Convert to tensor using encoding.
    fn to_tensor(&self, encoding: &Ctx) -> Tensor{
        self.try_to_tensor(encoding).unwrap()
    }

    /// Tries to encode as tensor and then flatten it to one dimension.
    fn try_to_tensor_flat(&self, encoding: &Ctx) -> Result<Tensor, ConvertError>{
        let t1 = self.try_to_tensor(encoding)?;
        t1.f_flatten(0, -1).map_err(|e|{
            ConvertError::ConvertToTensor {
                origin: format!("{e}"),
                context: format!("Flattening tensor {t1:?} from information set: {:?}", self)
            }
        })
    }

    /// Converts to tensor ant then flattens it to one dimension.
    fn to_tensor_flat(&self, encoding: &Ctx) -> Tensor{
        let t1 = self.to_tensor(encoding);
        //let dim = t1.dim() as i64;
        t1.flatten(0, -1)
    }
    /// Returns desired shape of data encoded as [`Tensor`].
    fn tensor_shape(encoding: &Ctx) -> &[i64]{
        encoding.desired_shape()
    }

    /// Length of flatten encoded tensor.
    fn tensor_length_flatten(encoding: &Ctx) -> i64{
        encoding.desired_shape().iter().product()
    }
}

/// Represents format for converting element to set of Tensors. Currently not used in this crate.
pub trait MultiTensorEncoding: Send{
    /// Returns expected shape of output in this format.
    /// Vector is indexed with category, and i64 is length of [`Tensor`] representing parameter.
    fn expected_outputs_shape(&self) -> &[Vec<i64>];
}

/// For data to be encoded into multiple tensors. Currently without use in this crate.
pub trait ContextMultiTensorEncode<Ctx: MultiTensorEncoding> : Debug{
    /// Try encode to tensors using format.
    fn try_to_multiple_tensors(&self, form: &Ctx) -> Result<Vec<Tensor>, ConvertError>;
    /// Encode to tensors using format.
    fn to_multiple_tensors(&self, form: &Ctx) -> Vec<Tensor>{
        self.try_to_multiple_tensors(form).unwrap()
    }
    /*
    fn shapes(&self, form: &W) -> &[Vec<i64>]{
        form.expected_outputs_shape()
    }

     */
}

/// Encoding format of data into index [`Tensor`], usually for mapping discrete actions to index.
pub trait TensorIndexI64Encoding: Send{
    /// Minimal value of index
    fn min(&self) -> i64;
    /// Limit of index value - the maximal index is `limit() -1`
    fn limit(&self) -> i64;
}
/// For data implementing encoding into index (usually actions from discrete space).
pub trait ContextEncodeIndexI64<Ctx: TensorIndexI64Encoding> : Debug{

    /// Tries encoding data (from discrete space) into index using specific format.
    fn try_to_index(&self, encoding: &Ctx) -> Result<i64, ConvertError>;
}
/// For decoding data from index [`Tensor`] - usually discrete actions mapped to numbers.
pub trait ContextDecodeIndexI64<Ctx: TensorIndexI64Encoding> : Debug + Sized{

    fn try_from_index(index: i64, encoding: &Ctx) -> Result<Self, ConvertError>;
}


/// Encoding into multiple [`Tensors`](tch::Tensor) format.
pub trait MultiTensorIndexI64Encoding {
    /// Minimum index for a category.
    fn min(&self, param_index: usize) -> Option<i64>;
    /// Limiting index for a category - maximal index is this value -1.
    fn limit(&self, param_index: usize) -> Option<i64>;
    /// Number of categories that are in this encoding format.
    fn number_of_params() -> usize;


}


impl<T: MultiTensorIndexI64Encoding> ActionTensorFormat<Vec<Tensor>> for T{

    fn param_dimension_size(&self) -> i64 {
        Self::number_of_params() as i64
    }
}


impl<T: TensorIndexI64Encoding> ActionTensorFormat<Tensor> for T{

    fn param_dimension_size(&self) -> i64 {
        1
    }
}

/// Trait for type (usually action) into vector of indices.
/// The main purpose is to generate index of parameter choice in every category.
pub trait ContextEncodeMultiIndexI64<Ctx: MultiTensorIndexI64Encoding>{
    /// Outputs index of parameter at given category.
    /// Let's say that an action has parameter in category 1 of value that is mapped to index 2.
    /// Then this function will return Some(2) for category 1.
    fn param_value(&self, context: &Ctx, param_index: usize) -> Result<Option<i64>, ConvertError>;
    /// Default value when the index must be generated and no info is preserved about this.
    /// It could be 0 or random value in proper range. Default implementation gives index 0.
    fn default_value(&self, _context: &Ctx, _param_index: usize) -> i64{
        0
    }




    /// Returns vector of choice indexes in every category.
    fn action_indexes(&self, context: &Ctx) -> Result<Vec<Option<i64>>, ConvertError>{

        (0..Ctx::number_of_params()).map(|i|{
            self.param_value(context, i)
        }).collect::<Result<Vec<Option<i64>>, _>>()
    }


    /// Returns vector of index for every category and mask for category - if it is important.
    /// There may be a case where action does not have value in some category. This is
    /// indicated by setting mask to `Tensor([False])` in this category.
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

    /// Just like `action_index_and_mask_tensor_vecs` but for slice of actions. Outputting vectors (indexed by category)
    /// of Tensors with first (0) dimension being batch dimension.
    fn batch_index_and_mask_tensor_vecs(actions: &[&Self], context: &Ctx) -> Result<(Vec<Tensor>, Vec<Tensor>), ConvertError>
    {
        let mut params = Vec::new();
        let usage_masks = Vec::new();

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

/// For types to be decoded from multiple index [`Tensors`](tch::Tensor).
/// E.g. action build from `vec![Tensor([1]), Tensor([4]), Tensor([0])]`.
/// Where these numbers `1,4,0` are indices of parameter choices. In category 0 parameter 1 is chosen,
/// in category 1 parameter of index 4 is chosen and in category 2 parameter is chosen from index 0.
pub trait ContextDecodeMultiIndexI64<Ctx: MultiTensorIndexI64Encoding> : Debug + Sized{

    fn try_from_indices(indices: &[i64], way: &Ctx) -> Result<Self, ConvertError>;
}

/// Trait combining [`TensorEncoding`] and [`TensorDecoding`].
pub trait TensorFormat: TensorEncoding + TensorDecoding {}
impl<T> TensorFormat for T where T: TensorEncoding + TensorDecoding {}

/// Trait combining [`MultiTensorDecoding`] and [`MultiTensorEncoding`].
pub trait MultipleTensorFormat: MultiTensorEncoding + MultiTensorDecoding {}
impl<T> MultipleTensorFormat for T where T: MultiTensorEncoding + MultiTensorDecoding {}

/// This is to be deprecated in the future.
pub trait TryIntoTensor: Debug{


    fn try_to_tensor(&self) -> Result<Tensor, TensorRepresentationError>;
}



impl<W: TensorEncoding, T: ContextEncodeTensor<W>> ContextEncodeTensor<W> for Box<T>{

    fn try_to_tensor(&self, way: &W) -> Result<Tensor, ConvertError> {
        self.as_ref().try_to_tensor(way)
    }
}

/// Trait representing structs (maybe 0-sized) that tell what is expected size of tensor to be
/// used to create data struct using [`ContextDecodeTensor`].
pub trait TensorDecoding: Send{
    /// Expected tensor shape that is decoded into value.
    fn expected_input_shape(&self) -> &[i64];
}

/// Implemented by structs that can be converted from tensors.
/// Certain data type can have different tensor representation, then it is needed to specify
/// what particular contextual representation is used (done by using correct [`TensorDecoding`]).
pub trait ContextDecodeTensor<Ctx: TensorDecoding>{
    /// Try decode data from tensor using decoding format.
    fn try_from_tensor(tensor: &Tensor, decoding: &Ctx) -> Result<Self, ConvertError> where Self: Sized;
}


/// Implemented by structs that can be converted from tensors.).
pub trait TryFromTensor: for<'a> TryFrom<&'a Tensor, Error=ConvertError> + Sized{
    fn try_from_tensor(tensor: &Tensor) -> Result<Self, ConvertError> where Self: Sized{
        Self::try_from(tensor)
    }
}



impl<T: for<'a> TryFrom<&'a Tensor, Error=ConvertError> + Sized> TryFromTensor for T{

}

/// Trait for data to be converted from multiple tensors. Currently without use in this library.
/// In A2C and PPO policies actions are converted from index tensors using trait [`ContextDecodeMultiIndexI64`].
pub trait TryFromMultiTensors: for<'a> TryFrom<&'a [Tensor], Error=ConvertError> + Sized{

    fn try_from_multi_tensors(tensors: &[Tensor]) -> Result<Self, ConvertError> where Self: Sized{
        Self::try_from(tensors)
    }
}

impl<T: for<'a> TryFrom<&'a [Tensor], Error=ConvertError> + Sized> TryFromMultiTensors for T{

}


/// Trait representing structs (maybe 0-sized) that tell what is expected size of tensor to be
/// used to create data struct using [`crate::tensor_data::ContextDecodeMultiTensor`].
pub trait MultiTensorDecoding: Send{
    /// Expected shape of vec of tensors to decode. `Vec` is indexed with category numbers,
    /// and `i64` is the length of [`Tensor`] in category.
    fn expected_inputs_shape(&self) -> &[Vec<i64>];
}

/// Implemented by structs that can be converted from tensors.
/// Certain data type can have different tensor representation, then it is needed to specify
/// what particular contextual representation is used (done by using correct [`crate::tensor_data::TensorDecoding`]).
pub trait ContextDecodeMultiTensor<Ctx: crate::tensor_data::MultiTensorDecoding>{
    fn try_from_tensors(tensors: &[Tensor], way: &Ctx)
        -> Result<Self, ConvertError> where Self: Sized;
}


/// Typically TensorForm will be just `Tensor` or `Vec<Tensor>`
pub trait ActionTensorFormat<TensorForm> {



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



impl<N: ArrayLength> TensorEncoding for GenericArray<i64, N>{
    fn desired_shape(&self) -> &[i64] {
        &self[..]
    }
}
impl<N: ArrayLength> TensorDecoding for GenericArray<i64, N>{
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

