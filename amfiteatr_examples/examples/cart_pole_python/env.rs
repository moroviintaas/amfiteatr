//! Wrapping Farama gymnasium [CartPole environment](https://github.com/Farama-Foundation/Gymnasium)
use std::fmt::{Debug};
use pyo3::prelude::*;
use amfiteatr_core::domain::{DomainParameters, RenewWithSideEffect};
use amfiteatr_core::env::{EnvironmentStateSequential, EnvironmentStateUniScore};
use amfiteatr_core::error::AmfiteatrError;
use crate::common::{CartPoleDomain, CartPoleObservation, CartPoleError, SINGLE_PLAYER_ID, CartPoleAction};



#[derive(Debug, Clone)]
#[pyclass]
pub struct PythonGymnasiumCartPoleState {
    internal: PyObject,
    #[allow(unused_variables)]
    _action_space: i64,
    terminated: bool,
    truncated: bool,
    latest_observation: Vec<f32>,
    player_reward: f32,
    #[allow(unused_variables)]
    _observation_space: Vec<i64>,
}



#[pymethods]
impl PythonGymnasiumCartPoleState {
    #[new]
    pub fn new() -> PyResult<Self>{
        Python::with_gil(|py|{

            let gymnasium = py.import_bound("gymnasium")?;


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
            let step0_t = step0.downcast_bound::<pyo3::types::PyTuple>(py)?;
            //let step0 = PyTuple::from
            let obs = step0_t.get_item(0)?;
            let v = obs.extract()?;

            Ok(PythonGymnasiumCartPoleState {
                internal: internal_obj,
                _action_space: action_space,
                _observation_space: observation_space,
                truncated: false, terminated: false,
                latest_observation: v,
                player_reward: 0.0
            })

        })


    }


    pub fn __forward(&mut self, action: i64)
        -> PyResult<Vec<f32>>{

        Python::with_gil(|py|{
            let result = self.internal.call_method1(py, "step", (action, ))?;


            //let result_tuple: &pyo3::types::PyTuple = result.downcast(py)?.into();
            let result_tuple: &Bound<'_, pyo3::types::PyTuple> = result.downcast_bound(py)?;

            let observation = result_tuple.get_item(0)?;
            let reward = result_tuple.get_item(1)?;
            let truncated = result_tuple.get_item(3)?;
            let terminated = result_tuple.get_item(2)?;
            let _info = result_tuple.get_item(4)?;

            self.terminated = terminated.extract()?;
            self.truncated = truncated.extract()?;
            let v = observation.extract()?;
            let r: f32 = reward.extract()?;
            self.player_reward += r;

            Ok(v)

        })
    }

    pub fn __reset(&mut self) -> PyResult<Vec<f32>>{
        Python::with_gil(|py|{
            let result = self.internal.call_method0(py, "reset")?;
            let result_tuple: &Bound<'_, pyo3::types::PyTuple> = result.downcast_bound(py)?;
            //let result_tuple: &pyo3::types::PyTuple = result.downcast(py)?;
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
    type Updates = [(<CartPoleDomain as DomainParameters>::AgentId, <CartPoleDomain as DomainParameters>::UpdateType );1];

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

    fn forward(&mut self, _agent: <CartPoleDomain as DomainParameters>::AgentId, action: CartPoleAction) -> Result<Self::Updates, <CartPoleDomain as DomainParameters>::GameErrorType> {


        match self.__forward(action.into()){
            Err(e) => Err(e.into()),
            Ok(observation_vec) => {
                if observation_vec.len() >= 4{
                    let observation = CartPoleObservation::new(
                        observation_vec[0],
                    observation_vec[1],
                    observation_vec[2],
                    observation_vec[3]);
                    Ok([(SINGLE_PLAYER_ID, observation)])
                } else {
                    Err(CartPoleError::InterpretingPythonData {
                        description: format!("Expected observation containing 4 f32 values, observed {}", observation_vec.len())
                    })
                }
            }
        }





    }
}

impl RenewWithSideEffect<CartPoleDomain, ()> for PythonGymnasiumCartPoleState{
    type SideEffect = [(<CartPoleDomain as DomainParameters>::AgentId, <CartPoleDomain as DomainParameters>::UpdateType);1];

    fn renew_with_side_effect_from(&mut self, _base: ()) -> Result<Self::SideEffect, AmfiteatrError<CartPoleDomain>> {
        match self.__reset(){
            Err(e) => Err(AmfiteatrError::Game{source: e.into()}),
            Ok(observation_vec) => {
                if observation_vec.len() >= 4{
                    let observation = CartPoleObservation::new(
                        observation_vec[0],
                        observation_vec[1],
                        observation_vec[2],
                        observation_vec[3]);
                    Ok([(SINGLE_PLAYER_ID, observation)])
                } else {
                    Err(AmfiteatrError::Game{source: CartPoleError::InterpretingPythonData {
                        description: format!("Expected observation containing 4 f32 values, observed {}", observation_vec.len())
                    }})
                }
            }
        }
    }
}

impl EnvironmentStateUniScore<CartPoleDomain> for PythonGymnasiumCartPoleState{
    fn state_score_of_player(&self, _agent: &<CartPoleDomain as DomainParameters>::AgentId) -> <CartPoleDomain as DomainParameters>::UniversalReward {
        self.player_reward
    }
}