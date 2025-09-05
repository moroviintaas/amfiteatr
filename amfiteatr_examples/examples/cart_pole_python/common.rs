use std::fmt::{Display, Formatter};
use pyo3::{DowncastError, PyErr};
use amfiteatr_core::scheme::{Action, Scheme};
use amfiteatr_core::error::ConvertError;
use amfiteatr_rl::error::TensorRepresentationError;
use amfiteatr_rl::tch::{Tensor};
use amfiteatr_rl::tensor_data::{TryIntoTensor};

pub const SINGLE_PLAYER_ID: u64 = 1;

#[derive(Debug, Clone)]
pub struct CartPoleScheme {}



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

impl From<PyErr> for CartPoleError {
    fn from(value: PyErr) -> Self {
        Self::InsidePython {description: format!("{}", value)}
    }
}
impl<'a, 'py> From<DowncastError<'a, 'py>> for CartPoleError {
    fn from(value: DowncastError<'a, 'py>) -> Self {
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

impl TryFrom<&Tensor> for CartPoleAction{
    type Error = ConvertError;

    fn try_from(tensor: &Tensor) -> Result<Self, Self::Error> {
        let v: Vec<i64> = match Vec::try_from(tensor){
            Ok(v) => v,
            Err(e) =>{
                return Err(ConvertError::ConvertFromTensor { origin: format!("{e}"), context: "Cart Pole Action".into()})
            }
        };
        match v[0]{
            0 => Ok(CartPoleAction::Left),
            1 => Ok(CartPoleAction::Right),
            e => Err(ConvertError::ConvertFromTensor{ origin: format!("{e:}"), context: "Bad action index {e}".into()})
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



impl TryIntoTensor for CartPoleAction {
    fn try_to_tensor(&self) -> Result<Tensor, TensorRepresentationError> {
        match self {
            CartPoleAction::Left => Ok(Tensor::from_slice(&[0.0f32])),
            CartPoleAction::Right => Ok(Tensor::from_slice(&[1.0f32])),
        }
    }
}



impl Scheme for CartPoleScheme {
    type ActionType = CartPoleAction;
    type GameErrorType = CartPoleError;
    type UpdateType = CartPoleObservation;
    type AgentId = u64;
    type UniversalReward = f32;
}