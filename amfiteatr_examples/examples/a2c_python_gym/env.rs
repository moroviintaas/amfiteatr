use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use pyo3::prelude::*;
use pyo3::PyDowncastError;
use pyo3::types::PyTuple;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::env::EnvironmentStateSequential;



pub const SINGLE_PLAYER_ID: u64 = 1;
#[derive(Debug, Clone)]
#[pyclass]
pub struct PythonGymnasiumCartPoleState {
    internal: PyObject,
    action_space: i64,
    observation_space: Vec<i64>,
    terminated: bool,
    truncated: bool,
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
    type UpdateType = Vec<i64>;
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
            Ok(PythonGymnasiumCartPoleState {
                internal: env_obj.into(),
                action_space, observation_space,
                truncated: false, terminated: false,
            })

        })


    }
    fn __forward(&mut self, action: <CartPoleDomain as DomainParameters>::ActionType)
        -> PyResult<Vec<i64>>{

        Python::with_gil(|py|{
            let r = self.internal.call1(py, (action, ))?;


            let r_tuple: &PyTuple = r.downcast(py)?;

            todo!()
        })
    }
}


impl EnvironmentStateSequential<CartPoleDomain> for PythonGymnasiumCartPoleState{
    type Updates = [(<CartPoleDomain as DomainParameters>::AgentId, Vec<i64> );1];

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
        /*Python::with_gil(|py|{
            let r = self.internal.call1(py, (action, ))?;

            let r_tuple: PyTuple = r.try_into()?;

            todo!()
        })

         */
        todo!()
    }
}

