use std::fmt::{Display, Formatter};
use amfiteatr_core::domain::{Action, DomainParameters};
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::TryIntoTensor;


pub const _SINGLE_PLAYER_ID: u64 = 1;
#[derive(Debug, Clone)]
pub struct CartPoleDomain{}


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

#[derive(Clone, Debug, thiserror::Error)]
pub enum CartPoleRustError {
    #[error("Game not initialized. Use reseed().")]
    GameStateNotInitialized,

}
impl DomainParameters for CartPoleDomain{
    type ActionType = CartPoleAction;
    type GameErrorType = CartPoleRustError;
    type UpdateType = CartPoleObservation;
    type AgentId = u64;
    type UniversalReward = f32;
}