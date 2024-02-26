use std::fmt::{Display, Formatter, write};
use pyo3::{PyDowncastError, PyErr};
use amfiteatr_core::domain::{Action, DomainParameters};
use amfiteatr_core::error::ConvertError;
use amfiteatr_rl::tch::{TchError, Tensor};
use amfiteatr_rl::tensor_data::ActionTensor;

pub const SINGLE_PLAYER_ID: u64 = 1;

#[derive(Debug, Clone)]
pub struct CartPoleDomain{}

/*
#[derive(Debug, Clone)]
pub enum CartPoleAction{
    Left,
    Right
}

 */


#[derive(thiserror::Error, Debug, Clone)]
pub enum CartPoleError {
    #[error("Internal Game error from gymnasium: {description:}")]
    InsidePython {
        description: String
    },
    #[error("Interpreting python function output: {description}")]
    InterpretingPythonData{
        description: String
    }

}

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
/*
impl<T: Display> From<T> for GymnasiumError{
    fn from(value: T) -> Self {
        Self::Bail {description: format!("{}", value)}
    }
}

 */

impl From<PyErr> for CartPoleError {
    fn from(value: PyErr) -> Self {
        Self::InsidePython {description: format!("{}", value)}
    }
}
impl<'a> From<PyDowncastError<'a>> for CartPoleError {
    fn from(value: PyDowncastError<'a>) -> Self {
        Self::InsidePython {description: format!("Python downcast error {}", value)}
    }
}

#[derive(Debug, Copy, Clone)]
pub enum CartPoleAction{
    Left,
    Right
}

impl  From<CartPoleAction> for i64{
    fn from(value: CartPoleAction) -> Self {
        match value{
            CartPoleAction::Left => 0,
            CartPoleAction::Right => 1
        }
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

impl ActionTensor for CartPoleAction{
    fn to_tensor(&self) -> Tensor {
        match self{
            CartPoleAction::Left => Tensor::from_slice(&[0.0f32]),
            CartPoleAction::Right => Tensor::from_slice(&[1.0f32]),
        }
    }

    fn try_from_tensor(t: &Tensor) -> Result<Self, ConvertError> {
        let v = Vec::<i64>::try_from(t)
            .map_err(|e| ConvertError::ActionDeserialize(format!("{}", t)))?;
        match v.get(0){
            Some(0) => Ok(CartPoleAction::Left),
            Some(1) => Ok(CartPoleAction::Right),
            Some(n) => Err(ConvertError::ActionDeserialize(format!("Expected action number 0 or 1, got {}",n))) ,
            None => Err(ConvertError::ActionDeserialize("Trying to convert tensor of size 0".to_string()))
        }
    }
}

impl DomainParameters for CartPoleDomain{
    type ActionType = CartPoleAction;
    type GameErrorType = CartPoleError;
    type UpdateType = CartPoleObservation;
    type AgentId = u64;
    type UniversalReward = f32;
}