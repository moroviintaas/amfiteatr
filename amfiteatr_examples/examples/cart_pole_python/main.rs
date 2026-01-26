pub mod env;
pub mod common;
pub mod agent;

use std::thread;
use amfiteatr_core::agent::{PolicyAgent, TracingAgentGen};
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{AutoEnvironmentWithScores, BasicEnvironment, ReseedEnvironmentWithObservation};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::agent::RlModelAgent;
use amfiteatr_rl::error::AmfiteatrRlError;
#[allow(deprecated)]
use amfiteatr_rl::policy::{ActorCriticPolicy, LearningNetworkPolicyGeneric, TrainConfig};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::torch_net::{A2CNet, TensorActorCritic};
use crate::agent::{CART_POLE_TENSOR_REPR, PythonGymnasiumCartPoleInformationSet};
use crate::common::{CartPoleScheme, CartPoleObservation, SINGLE_PLAYER_ID};
use crate::env::PythonGymnasiumCartPoleState;

fn test<R: RlModelAgent<CartPoleScheme, CartPoleObservation, PythonGymnasiumCartPoleInformationSet>>(
    env: &mut BasicEnvironment<CartPoleScheme, PythonGymnasiumCartPoleState, EnvironmentMpscPort<CartPoleScheme>>,
    agent: &mut R,
    number_of_tests: usize)
    -> Result<f32, AmfiteatrRlError<CartPoleScheme>>
where <R as PolicyAgent<CartPoleScheme>>::Policy: LearningNetworkPolicyGeneric<CartPoleScheme>{

    let mut result_sum = 0.0f64;
    for _ in 0..number_of_tests{
        let mut observation = env.reseed_with_observation(())?;
        let observation = observation.remove(&SINGLE_PLAYER_ID)
            .ok_or(AmfiteatrRlError::Amfiteatr{source: AmfiteatrError::Custom(
                String::from("No observation for only player in resetting game")
            )})?;

        thread::scope(|s|{
            s.spawn(||{
                env.run_with_scores()
            });
            s.spawn(||{
                agent.run_episode(observation).unwrap()
            });
        });
        result_sum += agent.current_universal_score() as f64;
    }

    Ok((result_sum / (number_of_tests as f64)) as f32)


}

fn train_epoch<R: RlModelAgent<CartPoleScheme, CartPoleObservation, PythonGymnasiumCartPoleInformationSet>>(
    env: &mut BasicEnvironment<CartPoleScheme, PythonGymnasiumCartPoleState, EnvironmentMpscPort<CartPoleScheme>>,
    agent: &mut R,
    number_of_games: usize)
    -> Result<(), AmfiteatrRlError<CartPoleScheme>>
where <R as PolicyAgent<CartPoleScheme>>::Policy: LearningNetworkPolicyGeneric<CartPoleScheme>{

    agent.clear_episodes()?;
    for _ in 0..number_of_games{
        let mut observation = env.reseed_with_observation(())?;
        let observation = observation.remove(&SINGLE_PLAYER_ID)
            .ok_or(AmfiteatrRlError::Amfiteatr{source: AmfiteatrError::Custom(
                String::from("No observation for only player in resetting game")
            )})?;

        thread::scope(|s|{
            s.spawn(||{
                env.run_with_scores()
            });
            s.spawn(||{
                agent.run_episode(observation).unwrap()
            });
        });
    }
    let trajectories = agent.take_episodes();

    agent.policy_mut().train(&trajectories)?;

    Ok(())
}

fn main() {
    let device = Device::Cpu;
    let var_store = VarStore::new(device);
    let  env_state = PythonGymnasiumCartPoleState::new().unwrap();
    let agent_state = PythonGymnasiumCartPoleInformationSet::default();

    let mut env_comm = EnvironmentMpscPort::new();
    let agent_comm = env_comm.register_agent(SINGLE_PLAYER_ID).unwrap();

    let mut environment = BasicEnvironment::new(env_state, env_comm);

    /* old
    let net_template = NeuralNetTemplate::new(|path|{
        let seq = nn::seq()
            .add(nn::linear(path / "input", 4, 128, Default::default()))
            .add_fn(|xs| xs.relu());

        let actor = nn::linear(path / "al", 128, 2, Default::default());
        let critic = nn::linear(path / "ac", 128, 1, Default::default());
        { move | input: &Tensor|{
            let xs = input.to_device(device).apply(&seq);
            TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });

     */

    let operator = Box::new(move |vs: &VarStore, tensor: &Tensor|{
        let seq = nn::seq()
            .add(nn::linear(vs.root() / "input", 4, 128, Default::default()))
            .add_fn(|xs| xs.relu());

        let actor = nn::linear(vs.root() / "al", 128, 2, Default::default());
        let critic = nn::linear(vs.root() / "ac", 128, 1, Default::default());

        let xs = tensor.to_device(device).apply(&seq);
        TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}

    });

    let net =  A2CNet::new_concept_1(var_store, operator);
    let optimizer = net.build_optimizer(Adam::default(), 1e-4).unwrap();

    #[allow(deprecated)]
    let policy = ActorCriticPolicy::new(net,
                                        optimizer,
                                        CART_POLE_TENSOR_REPR,
                                        TrainConfig {gamma: 0.99});

    let mut agent = TracingAgentGen::new(agent_state, agent_comm, policy);

    let avg = test(&mut environment, &mut agent, 100).unwrap();
    println!("Average result before learning: {}.", avg);
    for epoch in 0..100{
        train_epoch(&mut environment, &mut agent, 10).unwrap();
        let avg = test(&mut environment, &mut agent, 100).unwrap();
        println!("Average result after epoch {}: {}.", epoch+1, avg);
    }

}