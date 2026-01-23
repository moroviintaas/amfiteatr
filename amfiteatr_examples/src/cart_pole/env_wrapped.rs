use pyo3::{intern, pyclass, pymethods, Bound, PyObject, PyResult, Python};
use pyo3::prelude::PyDictMethods;
use pyo3::types::PyDict;
use rand::Rng;
use amfiteatr_core::env::{GameStateWithPayoffs, SequentialGameState};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::scheme::{Renew, Scheme};
use crate::cart_pole::common::{CartPoleObservation, CartPoleRustError, CartPoleScheme, SINGLE_PLAYER_ID};
use crate::cart_pole::env::CartPoleEnvStateRust;
use crate::cart_pole::model::CartPoleModelOptions;
use pyo3::types::PyAnyMethods;
use pyo3::IntoPyObject;

#[derive(Debug, Clone)]
#[pyclass]
pub struct PythonGymnasiumWrap {
    internal: PyObject,
    #[allow(unused_variables)]
    terminated: bool,
    truncated: bool,
    current_agent: u64,
    reward: f32,
    steps_made: usize,
}

#[pymethods]
impl PythonGymnasiumWrap{
    #[new]
    pub fn new() -> PyResult<Self>{
        Python::with_gil(|py|{
            let pettingzoo = py.import("pettingzoo.classic")?;
            let fn_env = pettingzoo.getattr("CartPole-v1")?.getattr("env")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "render_mode"), "None")?;
            let env_obj = fn_env.call((), Some(&kwargs))?;
            env_obj.call_method0("reset")?;

            let internal_obj: PyObject = env_obj.into_pyobject(py)?.into();

            Ok(Self{
                internal: internal_obj,

                terminated: false,
                truncated: false,
                current_agent: SINGLE_PLAYER_ID,
                reward: 0.0,
                steps_made: 0,
            })
        })

    }

    pub fn __reset(&mut self) -> PyResult<()>{
        Python::with_gil(|py|{
            let pettingzoo = py.import("pettingzoo.classic")?;
            let fn_env = pettingzoo.getattr("CartPole-v1")?.getattr("env")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "render_mode"), "None")?;
            let env_obj = fn_env.call((), Some(&kwargs))?;
            env_obj.call_method0("reset")?;


            let internal_obj: PyObject = env_obj.into_pyobject(py)?.into();
            self.internal = internal_obj;

            Ok(())

        })
    }

    pub fn __forward(&self, action: u8) -> PyResult<(Vec<f32>, f32, bool, bool)>{
        Python::with_gil(|py| {
            let result = self.internal.call_method1(py, "step", (action,))?;
            let result_tuple: &Bound<'_, pyo3::types::PyTuple> = result.downcast_bound(py)?;

            let observation = result_tuple.get_item(0)?
                .get_item("observation")?
                .call_method0("flatten")?
                .call_method0("tolist")?;
            let reward = result_tuple.get_item(1)?;
            let truncated = result_tuple.get_item(3)?;
            let terminated = result_tuple.get_item(2)?;

            let o: Vec<f32> = observation.extract()?;
            let r = reward.extract()?;
            let term = terminated.extract()?;
            let trunc = truncated.extract()?;
            

            Ok((o, r, trunc, term))
        })



    }

    pub fn __observation(&self) -> PyResult<Vec<f32>> {
        Python::with_gil(|py| {
            let result = self.internal.getattr(py, "state")?;
            let v: &Bound<'_, pyo3::types::PyAny> = &result.downcast_bound(py)?
                .call_method0("flatten")?
                .call_method0("tolist")?;
            let o = v.extract()?;
            Ok(o)
        })
    }
}

impl SequentialGameState<CartPoleScheme> for PythonGymnasiumWrap{
    type Updates = [(<CartPoleScheme as Scheme>::AgentId, <CartPoleScheme as Scheme>::UpdateType );1];


    fn current_player(&self) -> Option<<CartPoleScheme as Scheme>::AgentId> {

        if self.terminated || self.truncated{
            None
        } else {
            Some(self.current_agent)
        }
    }



    fn is_finished(&self) -> bool {
        self.terminated || self.truncated
    }

    fn forward(&mut self, agent: <CartPoleScheme as Scheme>::AgentId, action: <CartPoleScheme as Scheme>::ActionType)
        -> Result<Self::Updates, <CartPoleScheme as Scheme>::GameErrorType> {
        if self.is_finished(){
            return Err(CartPoleRustError::Custom("Game is finished".to_string()))
        }
        if agent != self.current_agent{
            return Err(CartPoleRustError::Custom("Violated order".to_string()))
        }
        let (obs, reward, terminated, truncated) = self.__forward(action.index() as u8)
            .map_err(|e| CartPoleRustError::Custom("Py error {e}".to_string()))?;

        self.reward += reward;
        self.terminated = terminated;
        self.truncated = truncated;

        let observation = CartPoleObservation{
            position: obs[0],
            velocity: obs[1],
            angle: obs[2],
            angular_velocity: obs[3],
        };
        //self.current_agent = self.current_agent.other();
        Ok([(SINGLE_PLAYER_ID, observation)])



    }

    fn first_observations(&self) -> Option<Self::Updates> {
        let obs = self.__observation().unwrap();

        let observation = CartPoleObservation{
            position: obs[0],
            velocity: obs[1],
            angle: obs[2],
            angular_velocity: obs[3],
        };

        Some([(SINGLE_PLAYER_ID, observation)])
    }

}
/*
impl Renew<CartPoleScheme, (), > for PythonGymnasiumWrap {
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<CartPoleScheme>> {

        let d = rand::distr::Uniform::new(-0.05, 0.05).unwrap();
        let mut rng = rand::rng();
        let state = CartPoleObservation{
            position: rng.sample(d),
            velocity: rng.sample(d),
            angle: rng.sample(d),
            angular_velocity: rng.sample(d),
        };

        self.terminated = false;
        self.truncated = false;
        self.steps_made = 0;
        self.state = Some(state);
        self.reward = 0.0;


        Ok(())
    }
}

impl GameStateWithPayoffs<CartPoleScheme> for PythonGymnasiumWrap {
    fn state_payoff_of_player(&self, _agent: &<CartPoleScheme as Scheme>::AgentId) -> <CartPoleScheme as Scheme>::UniversalReward {
        self.reward
    }
}

 */