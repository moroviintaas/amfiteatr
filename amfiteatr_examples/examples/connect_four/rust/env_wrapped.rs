use pyo3::{Bound, intern, pyclass, pymethods, PyObject, PyResult, Python,  IntoPyObject};
use pyo3::prelude::PyAnyMethods;
use pyo3::types::PyDict;
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::env::{SequentialGameState, GameStateWithPayoffs};
use amfiteatr_core::error::AmfiteatrError;
use crate::common::{
    ConnectFourBinaryObservation,
    ConnectFourDomain,
    ConnectFourError,
    ConnectFourPlayer
};



#[derive(Debug, Clone)]
#[pyclass]
pub struct PythonPettingZooStateWrap {
    internal: PyObject,
    #[allow(unused_variables)]
    terminated: bool,
    truncated: bool,
    current_agent: ConnectFourPlayer,
    rewards: [f32;2],
}

#[pymethods]
impl PythonPettingZooStateWrap{

    #[new]
    pub fn new() -> PyResult<Self>{
        Python::with_gil(|py|{
            let pettingzoo = py.import("pettingzoo.classic")?;
            let fn_env = pettingzoo.getattr("connect_four_v3")?.getattr("env")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "render_mode"), "None")?;
            let env_obj = fn_env.call((), Some(&kwargs))?;
            env_obj.call_method0("reset")?;

            let internal_obj: PyObject = env_obj.into_pyobject(py)?.into();

            Ok(Self{
                internal: internal_obj,

                terminated: false,
                truncated: false,
                current_agent: ConnectFourPlayer::One,
                rewards: [0.0, 0.0],
            })
        })

    }
    pub fn __reset(&mut self) -> PyResult<()>{
        Python::with_gil(|py|{
            let pettingzoo = py.import("pettingzoo.classic")?;
            let fn_env = pettingzoo.getattr("connect_four_v3")?.getattr("env")?;
            let kwargs = PyDict::new(py);
            kwargs.set_item(intern!(py, "render_mode"), "None")?;
            let env_obj = fn_env.call((), Some(&kwargs))?;
            env_obj.call_method0("reset")?;


            let internal_obj: PyObject = env_obj.into_pyobject(py)?.into();
            self.internal = internal_obj;

            Ok(())

        })
    }


    pub fn __last(&self) -> PyResult<(Vec<u8>, f32, bool, bool)>{
        Python::with_gil(|py|{
            let result = self.internal.call_method0(py, "last")?;
            let result_tuple: &Bound<'_, pyo3::types::PyTuple> = result.downcast_bound(py)?;

            let observation = result_tuple.get_item(0)?
                .get_item("observation")?
                .call_method0("flatten")?
                .call_method0("tolist")?;
            let reward = result_tuple.get_item(1)?;
            let truncated = result_tuple.get_item(3)?;
            let terminated = result_tuple.get_item(2)?;

            let o: Vec<u8> = observation.extract()?;
            let r = reward.extract()?;
            let term = terminated.extract()?;
            let trunc = truncated.extract()?;

            Ok((o,r,term,trunc))

        })
    }
    pub fn __forward(&self, action: u8) -> PyResult<(Vec<u8>, f32, bool, bool)>{
        Python::with_gil(|py| {
            self.internal.call_method1(py, "step", (action,))?;
            self.__last()
        })



    }


    /*
    pub fn __agent_selection(&self) -> PyResult<ConnectFourDomain>{
        Python::with_gil(|py|{
            let
        })
    }

     */
}

impl SequentialGameState<ConnectFourDomain> for PythonPettingZooStateWrap{
    type Updates = [(<ConnectFourDomain as DomainParameters>::AgentId, <ConnectFourDomain as DomainParameters>::UpdateType );1];


    fn current_player(&self) -> Option<<ConnectFourDomain as DomainParameters>::AgentId> {

        if self.terminated || self.truncated{
            None
        } else {
            Some(self.current_agent)
        }
    }



    fn is_finished(&self) -> bool {
        self.terminated || self.truncated
    }

    fn forward(&mut self, agent: <ConnectFourDomain as DomainParameters>::AgentId, action: <ConnectFourDomain as DomainParameters>::ActionType) -> Result<Self::Updates, <ConnectFourDomain as DomainParameters>::GameErrorType> {
        if self.is_finished(){
            return Err(ConnectFourError::PlayerDeadStep {player: agent})
        }
        if agent != self.current_agent{
            return Err(ConnectFourError::PlayerViolatedOrder {player: agent})
        }
        let (obs, reward, terminated, truncated) = self.__forward(action.index() as u8)?;

        self.rewards[agent.index()] += reward;
        self.terminated = terminated;
        self.truncated = truncated;

        let mut observation = ConnectFourBinaryObservation::default();
        for row in 0..6{
            for column in 0..7{
                for i in 0..2{
                    observation.board[row][column][i] = obs[(row*7*2) +(column * 2)+ i];
                }
            }
        }
        self.current_agent = self.current_agent.other();
        Ok([(agent.other(), observation)])



    }
}

impl GameStateWithPayoffs<ConnectFourDomain> for PythonPettingZooStateWrap{
    fn state_payoff_of_player(&self, agent: &ConnectFourPlayer) -> <ConnectFourDomain as DomainParameters>::UniversalReward {
        self.rewards[agent.index()]
    }
}

impl Renew<ConnectFourDomain, ()> for PythonPettingZooStateWrap{
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<ConnectFourDomain>> {
        self.__reset().map_err(|e| AmfiteatrError::Game { source: e.into() })?;
        self.rewards = [0.0, 0.0];
        self.terminated = false;
        self.truncated = false;
        self.current_agent = ConnectFourPlayer::One;
        //for i in self.state_vec.iter_mut(){
        //    *i = 0;
        //};
        Ok(())
    }
}

impl Default for PythonPettingZooStateWrap{
    fn default() -> Self {
        Self::new().unwrap()
    }
}