use tch::Tensor;
use amfiteatr_core::error::ConvertError;


/// Marker trait describing output format for neural network. For example Actor-Critic methods output
/// two Tensors (one for Action distribution and other to evaluate current state (information set).
pub trait NetOutput{}

/// Struct to aggregate both actor and critic output tensors from network.
pub struct TensorA2C{
    pub critic: Tensor,
    pub actor: Tensor
}

/// Struct to aggregate output for actor-critic networks with multi parameter actor
pub struct TensorCriticMultiActor{
    pub critic: Tensor,
    pub actor: Vec<Tensor>
}

pub type MultiDiscreteTensor = Vec<Tensor>;

impl NetOutput for MultiDiscreteTensor{}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}
impl NetOutput for TensorA2C{}

impl NetOutput for TensorCriticMultiActor{}


/// Converts tensor of shape (1,) and type i64 to i64. Technically it will work
/// with  shape (n,), but it will take the very first element. It is used to when we have single
/// value in Tensor that we want numeric.
///
/// Used usually when converting discrete distribution to action index.
/// Consider distribution of 4 actions: `[0.3, 0.5, 0.1, 0.1]`
/// 1. First we sample one number of `(0,1,2,3)` with probabilities above.
/// Let's say we sampled `1` (here it has 1/2 chance to be so);
/// 2. At this moment we have Tensor of shape `(1,)` of type `i64`. But we want just number `i64`.
/// 3. So we need to do this conversion;
///
/// # Example:
/// ```
/// use tch::Kind::{Double, Float};
/// use tch::Tensor;
/// use amfiteatr_rl::torch_net::index_tensor_to_i64;
/// let t = Tensor::from_slice(&[0.3f64, 0.5, 0.1, 0.1]);
/// let index_tensor = t.multinomial(1, true).softmax(-1, Double);
/// assert_eq!(index_tensor.size(), vec![1]);
/// let index = index_tensor_to_i64(&index_tensor, "context message if error").unwrap();
/// assert!(index >=0 && index <= 3);
/// ```
#[inline]
pub fn index_tensor_to_i64(tensor: &Tensor, additional_context: &str) -> Result<i64, ConvertError>{
    /*
    let v: Vec<i64> = match Vec::try_from(tensor){
        Ok(v) => v,
        Err(_) => {
            return Err(ConvertError::ActionDeserialize(format!("From tensor {} in context \"{}\"", tensor, additional_context)))
        }
    };
    Ok(v[0])

     */

    tensor.f_int64_value(&[0]).map_err(|e|{
        ConvertError::ConvertFromTensor(
            format!("From tensor {} in context \"{}\". The error itself: {}",
                    tensor, additional_context, e))
    })
}