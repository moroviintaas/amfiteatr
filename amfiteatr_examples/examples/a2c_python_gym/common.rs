use pyo3::{PyDowncastError, PyErr};
use amfiteatr_core::domain::DomainParameters;

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

#[derive(Clone, Debug)]
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

impl DomainParameters for CartPoleDomain{
    type ActionType = i64;
    type GameErrorType = CartPoleError;
    type UpdateType = CartPoleObservation;
    type AgentId = u64;
    type UniversalReward = f32;
}