use std::fmt::{Display, Formatter};
use amfiteatr_core::error::{AmfiteatrError, ConvertError};
use amfiteatr_core::scheme::{Action, Renew, Scheme};
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{ContextDecodeIndexI64, ContextEncodeIndexI64, TensorDecoding, TensorIndexI64Encoding, TryIntoTensor};

pub const SINGLE_PLAYER_ID: u64 = 0;
#[derive(Debug, Clone)]
pub struct CartPoleScheme{}


#[derive(Clone, Debug, Default)]
pub struct CartPoleObservation{
    pub position: f32,
    pub velocity: f32,
    pub angle: f32,
    pub angular_velocity: f32,
}

impl CartPoleObservation{
    pub fn new(position: f32, velocity: f32, angle: f32, angular_velocity: f32) -> Self{
        Self{
            position, velocity, angle, angular_velocity
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum CartPoleAction{
    Left,
    Right
}

impl CartPoleAction{
    pub fn index(&self) -> usize{
        match self{
            CartPoleAction::Left => 0,
            CartPoleAction::Right => 1
        }
    }
}

impl Renew<CartPoleScheme, ()> for CartPoleObservation{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<CartPoleScheme>> {
        self.angle = 0.0;
        self.velocity = 0.0;
        self.angular_velocity = 0.0;
        self.position = 0.0;
        Ok(())
    }
}




impl Action for CartPoleAction {}

impl Display for CartPoleAction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self{
            CartPoleAction::Left => write!(f, "Left"),
            CartPoleAction::Right => write!(f, "Right")
        }
    }
}



impl TryIntoTensor for CartPoleAction {
    fn try_to_tensor(&self) -> Result<Tensor, TensorRepresentationError> {
        match self {
            CartPoleAction::Left => Ok(Tensor::from_slice(&[0.0f32])),
            CartPoleAction::Right => Ok(Tensor::from_slice(&[1.0f32])),
        }
    }
}

pub struct CartPoleActionEncoding{}

impl TensorIndexI64Encoding for CartPoleActionEncoding{
    fn min(&self) -> i64 {
        0
    }

    fn limit(&self) -> i64 {
        1
    }
}

impl TensorDecoding for CartPoleActionEncoding{
    fn expected_input_shape(&self) -> &[i64] {
        &[1]
    }
}

impl ContextDecodeIndexI64<CartPoleActionEncoding> for CartPoleAction{
    fn try_from_index(index: i64, _encoding: &CartPoleActionEncoding) -> Result<Self, ConvertError> {
        match index{
            0 => Ok(CartPoleAction::Left),
            1 => Ok(CartPoleAction::Right),
            any =>  Err(ConvertError::ConvertFromTensor{ origin: "".to_string(), context: format!("Failed converting number {any:} to CartPoleAction") })
        }
    }
}

impl ContextEncodeIndexI64<CartPoleActionEncoding> for CartPoleAction{
    fn try_to_index(&self, _encoding: &CartPoleActionEncoding) -> Result<i64, ConvertError> {
        match self{
            CartPoleAction::Left => Ok(0),
            CartPoleAction::Right => Ok(1)
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum CartPoleRustError {
    #[error("Game not initialized. Use reseed().")]
    GameStateNotInitialized,
    #[error("Custom {0}")]
    Custom(String),

}
impl Scheme for CartPoleScheme{
    type ActionType = CartPoleAction;
    type GameErrorType = CartPoleRustError;
    type UpdateType = CartPoleObservation;
    type AgentId = u64;
    type UniversalReward = f32;
}