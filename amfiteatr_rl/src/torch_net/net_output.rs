use tch::Tensor;


/// Marker trait describing output format for neural network. For example Actor-Critic methods output
/// two Tensors (one for Action distribution and other to evaluate current state (information set).
pub trait NetOutput{}

/// Struct to aggregate both actor and critic output tensors from network.
pub struct TensorA2C{
    pub critic: Tensor,
    pub actor: Tensor
}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}
impl NetOutput for TensorA2C{}