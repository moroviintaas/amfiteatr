pub mod env;
pub mod common;
pub mod agent;

use amfiteatr_core::agent::TracingAgentGen;
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{BasicEnvironment, EnvironmentStateSequential};
use amfiteatr_rl::policy::{ActorCriticPolicy, TrainConfig};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorA2C};
use crate::agent::{CART_POLE_TENSOR_REPR, PythonGymnasiumCartPoleInformationSet};
use crate::common::SINGLE_PLAYER_ID;
use crate::env::PythonGymnasiumCartPoleState;


fn main() {
    println!("Hello");
    let device = Device::Cpu;
    let var_store = VarStore::new(device);
    let  env_state = PythonGymnasiumCartPoleState::new().unwrap();
    let agent_state = PythonGymnasiumCartPoleInformationSet::default();

    let mut env_comm = EnvironmentMpscPort::new();
    let agent_comm = env_comm.register_agent(SINGLE_PLAYER_ID).unwrap();

    let mut environment = BasicEnvironment::new(env_state, env_comm);

    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", 4, 128, Default::default()))
            .add_fn(|xs| xs.relu());

        let actor = nn::linear(path / "al", 128, 2, Default::default());
        let critic = nn::linear(path / "ac", 128, 1, Default::default());
        { move | input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });
    let net =  A2CNet::new(var_store, net_template.get_net_closure());
    let optimizer = net.build_optimizer(Adam::default(), 1e-4).unwrap();

    let policy = ActorCriticPolicy::new(net,
                                        optimizer,
                                        CART_POLE_TENSOR_REPR,
                                        TrainConfig {gamma: 0.99});

    let mut agent = TracingAgentGen::new(agent_state, agent_comm, policy);



    println!("world")
}