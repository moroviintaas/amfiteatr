use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use pyo3::prelude::*;
use pyo3::PyDowncastError;
use pyo3::types::PyTuple;
use amfiteatr_core::domain::{DomainParameters, RenewWithSideEffect};
use amfiteatr_core::env::EnvironmentStateSequential;
use amfiteatr_core::error::AmfiError;


pub const SINGLE_PLAYER_ID: u64 = 1;
#[derive(Debug, Clone)]
#[pyclass]
pub struct PythonGymnasiumCartPoleState {
    internal: PyObject,
    action_space: i64,
    observation_space: Vec<f32>,
    terminated: bool,
    truncated: bool,
    latest_observation: Vec<f32>,
    player_reward: f32,
}

#[derive(Debug, Clone)]
pub struct CartPoleDomain{}


#[derive(thiserror::Error, Debug, Clone)]
pub enum GymnasiumError{
    #[error("Internal Game error from gymnasium: {description:}")]
    Bail{
        description: String
    }

}
/*
impl<T: Display> From<T> for GymnasiumError{
    fn from(value: T) -> Self {
        Self::Bail {description: format!("{}", value)}
    }
}

 */

impl From<PyErr> for GymnasiumError{
    fn from(value: PyErr) -> Self {
        Self::Bail {description: format!("{}", value)}
    }
}
impl<'a> From<PyDowncastError<'a>> for GymnasiumError{
    fn from(value: PyDowncastError<'a>) -> Self {
        Self::Bail {description: format!("Python downcast error {}", value)}
    }
}

impl DomainParameters for CartPoleDomain{
    type ActionType = i64;
    type GameErrorType = GymnasiumError;
    type UpdateType = Vec<f32>;
    type AgentId = u64;
    type UniversalReward = f32;
}

#[pymethods]
impl PythonGymnasiumCartPoleState {
    #[new]
    pub fn new() -> PyResult<Self>{
        Python::with_gil(|py|{

            let gymnasium = py.import("gymnasium")?;


            let fn_make = gymnasium.getattr("make")?;
            let env_obj = fn_make.call(("CartPole-v1",), None)?;
            let action_space = env_obj.getattr("action_space")?;
            let action_space = if let Ok(val) = action_space.getattr("n"){
                val.extract()?
            } else {
                let action_space: Vec<i64> = action_space.getattr("shape")?.extract()?;
                action_space[0]
            };

            let observation_space = env_obj.getattr("observation_space")?;
            let observation_space = observation_space.getattr("shape")?.extract()?;
            let internal_obj: PyObject = env_obj.to_object(py);
            let step0 = internal_obj.call_method0(py, "reset")?;
            let step0_t = step0.downcast::<PyTuple>(py)?;
            //let step0 = PyTuple::from
            let obs = step0_t.get_item(0)?;
            let v = obs.extract()?;

            Ok(PythonGymnasiumCartPoleState {
                internal: internal_obj,
                action_space, observation_space,
                truncated: false, terminated: false,
                latest_observation: v,
                player_reward: 0.0
            })

        })


    }


    pub fn __forward(&mut self, action: <CartPoleDomain as DomainParameters>::ActionType)
        -> PyResult<Vec<f32>>{

        Python::with_gil(|py|{
            let result = self.internal.call_method1(py, "step", (action, ))?;


            let result_tuple: &PyTuple = result.downcast(py)?;

            let observation = result_tuple.get_item(0)?;
            let reward = result_tuple.get_item(1)?;
            let truncated = result_tuple.get_item(2)?;
            let terminated = result_tuple.get_item(3)?;
            let info = result_tuple.get_item(4)?;

            self.terminated = terminated.extract()?;
            self.truncated = truncated.extract()?;
            let v = observation.extract()?;
            let r: f32 = reward.extract()?;
            self.player_reward += r;

            Ok(v)

        })
    }

    pub fn __reset(&mut self) -> PyResult<<CartPoleDomain as DomainParameters>::UpdateType>{
        Python::with_gil(|py|{
            let result = self.internal.call_method0(py, "reset")?;
            let result_tuple: &PyTuple = result.downcast(py)?;
            let observation = result_tuple.get_item(0)?;
            self.truncated = false;
            self.terminated = false;
            self.player_reward = 0.0;
            let v = observation.extract()?;
            Ok(v)
        })
    }
}


impl PythonGymnasiumCartPoleState{
    pub fn latest_observation(&self) -> &Vec<f32>{
        &self.latest_observation
    }
}

impl EnvironmentStateSequential<CartPoleDomain> for PythonGymnasiumCartPoleState{
    type Updates = [(<CartPoleDomain as DomainParameters>::AgentId, Vec<f32> );1];

    fn current_player(&self) -> Option<<CartPoleDomain as DomainParameters>::AgentId> {
        if self.terminated || self.truncated{
            None
        } else {
            Some(SINGLE_PLAYER_ID)
        }
    }

    fn is_finished(&self) -> bool {
        self.terminated || self.truncated
    }

    fn forward(&mut self, _agent: <CartPoleDomain as DomainParameters>::AgentId, action: i64) -> Result<Self::Updates, <CartPoleDomain as DomainParameters>::GameErrorType> {

        self.__forward(action)
            .map(|observation| [(SINGLE_PLAYER_ID, observation);1])
            .map_err(|e| e.into())
    }
}

impl RenewWithSideEffect<CartPoleDomain, ()> for PythonGymnasiumCartPoleState{
    type SideEffect = [(<CartPoleDomain as DomainParameters>::AgentId, <CartPoleDomain as DomainParameters>::UpdateType);1];

    fn renew_with_side_effect_from(&mut self, _base: ()) -> Result<Self::SideEffect, AmfiError<CartPoleDomain>> {
        self.__reset()
            .map(|observation| [(SINGLE_PLAYER_ID, observation);1])
            .map_err(|e| AmfiError::Game(e.into()))
    }
}
