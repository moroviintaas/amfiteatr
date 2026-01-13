use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_core::agent::TracingAgentGen;
use amfiteatr_core::comm::{AgentMpscAdapter, EnvironmentMpscPort};
use amfiteatr_core::env::{BasicEnvironment, SequentialGameState};
use amfiteatr_rl::policy::PolicyDiscretePPO;
use amfiteatr_rl::tch;
use amfiteatr_rl::tch::nn::{AdamW, VarStore};
use amfiteatr_rl::torch_net::{build_network_operator_ac, Layer, NeuralNetActorCritic};
use crate::cart_pole::agent::CartPoleObservationEncoding;
use crate::cart_pole::common::{CartPoleActionEncoding, CartPoleObservation, CartPoleScheme, SINGLE_PLAYER_ID};
use crate::cart_pole::env::CartPoleEnvStateRust;
use amfiteatr_rl::tch::nn::OptimizerConfig;


pub type CartPolePolicy = PolicyDiscretePPO<CartPoleScheme, CartPoleObservation, CartPoleObservationEncoding, CartPoleActionEncoding>;

pub struct CartPoleModelRust{
    env: BasicEnvironment<CartPoleScheme, CartPoleEnvStateRust, EnvironmentMpscPort<CartPoleScheme>>,
    agent: TracingAgentGen<CartPoleScheme, CartPolePolicy, AgentMpscAdapter<CartPoleScheme>>
}


impl CartPoleModelRust{
    pub fn new_simple() -> anyhow::Result<Self>{
        let initial_state = CartPoleEnvStateRust::new(true);
        let mut env_communicator = EnvironmentMpscPort::new();
        let obs = initial_state.first_observations().unwrap()[0].1.clone();
        let agent_comm = env_communicator.register_agent(SINGLE_PLAYER_ID)?;
        let env = BasicEnvironment::new(initial_state, env_communicator);

        let operator = build_network_operator_ac(vec![Layer::Linear(64), Layer::Linear(64)],
                                                 vec![4], 2);
        let vs = VarStore::new(tch::Device::Cpu);
        let optimizer = AdamW::default().build(&vs, 0.001)?;
        let network = NeuralNetActorCritic::new(vs, operator);

        let policy = CartPolePolicy::new(
            ConfigPPO::default(),
            network,
            optimizer,
            CartPoleObservationEncoding {},
            CartPoleActionEncoding {}
        );

        let agent = TracingAgentGen::new(obs, agent_comm, policy);

        Ok(Self{
            env, agent
        })
    }
}