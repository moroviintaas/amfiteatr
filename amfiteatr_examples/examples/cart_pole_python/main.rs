
pub mod env;
pub mod common;
pub mod agent;

use amfiteatr_examples::cart_pole::model_wrapped::CartPoleModelPython;
use std::thread;
use amfiteatr_core::agent::{PolicyAgent, TracingAgentGen};
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{AutoEnvironmentWithScores, BasicEnvironment, ReseedEnvironmentWithObservation, StatefulEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_examples::cart_pole::agent::CartPoleObservationEncoding;
use amfiteatr_examples::cart_pole::common::CartPoleActionEncoding;
use amfiteatr_rl::agent::RlModelAgent;
use amfiteatr_rl::error::AmfiteatrRlError;
#[allow(deprecated)]
use amfiteatr_rl::policy::{ActorCriticPolicy, LearningNetworkPolicyGeneric, TrainConfig};
use amfiteatr_rl::policy::{ConfigPPO, PolicyDiscretePPO};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::torch_net::{build_network_model_ac_discrete, A2CNet, TensorActorCritic, VariableStorage};
use amfiteatr_rl::torch_net::Layer::{Linear, Relu, Tanh};
use crate::agent::{CART_POLE_TENSOR_REPR, PythonGymnasiumCartPoleInformationSet};
use crate::common::{CartPoleScheme, CartPoleObservation, SINGLE_PLAYER_ID};
use crate::env::PythonGymnasiumCartPoleState;
use amfiteatr_rl::tch::nn::OptimizerConfig;
use amfiteatr_core::env::SequentialGameState;
use amfiteatr_examples::cart_pole;
use clap::Parser;
/*
fn test<R: RlModelAgent<CartPoleScheme, CartPoleObservation, PythonGymnasiumCartPoleInformationSet>>(
    env: &mut BasicEnvironment<CartPoleScheme, PythonGymnasiumCartPoleState, EnvironmentMpscPort<CartPoleScheme>>,
    agent: &mut R,
    number_of_tests: usize)
    -> Result<f32, AmfiteatrRlError<CartPoleScheme>>
where <R as PolicyAgent<CartPoleScheme>>::Policy: LearningNetworkPolicyGeneric<CartPoleScheme>
{
    let mut result_sum = 0.0f64;
    for _ in 0..number_of_tests {
        let mut observation = env.reseed_with_observation(())?;
        let observation = observation.remove(&SINGLE_PLAYER_ID)
            .ok_or(AmfiteatrRlError::Amfiteatr {
                source: AmfiteatrError::Custom(
                    String::from("No observation for only player in resetting game")
                )
            })?;

        thread::scope(|s| {
            s.spawn(|| {
                env.run_with_scores()
            });
            s.spawn(|| {
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


 */

pub fn setup_logger(options: &cart_pole::model::CartPoleModelOptions) -> Result<(), fern::InitError> {
    println!("Options: {:?}", options);
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level_for("amfiteatr_examples", options.log_level)
        .level_for("amfiteatr_core", options.log_level_amfiteatr)
        .level_for("amfiteatr_rl", options.log_level_amfiteatr_rl)
        //.level(options.log_level)

        ;
    //.level_for("amfiteatr_core", options.log_level_amfi);

    match &options.log_file{
        None => dispatch.chain(std::io::stdout()),
        Some(f) => dispatch.chain(fern::log_file(f)?)
    }

        .apply()?;
    Ok(())
}
fn main() -> anyhow::Result<()>{
    /*
    let device = Device::Cpu;
    let var_store = VarStore::new(device);
    let env_state = PythonGymnasiumCartPoleState::new().unwrap();
    let agent_state = CartPoleObservation::new(0.0, 0.0, 0.0, 0.0);

    let mut env_comm = EnvironmentMpscPort::new();
    let agent_comm = env_comm.register_agent(SINGLE_PLAYER_ID).unwrap();

    let mut environment = BasicEnvironment::new(env_state, env_comm);

    let info_set = environment.state().first_observations().unwrap()[1].1;



    let var_store = VarStore::new(device);
    let layers = vec![Linear(128)];
    let model = build_network_model_ac_discrete(layers.clone(), vec![4], 2, &var_store.root());
    let net =  A2CNet::new(VariableStorage::Owned(var_store), model);
    //let optimizer = net.build_optimizer(Adam::default(), 1e-4).unwrap();
    let optimizer = Adam::default().build(&var_store, 1e-4)?;

    #[allow(deprecated)]
    let policy = PolicyDiscretePPO::new(
        ConfigPPO::default(),
        net,
        optimizer,
        CartPoleObservationEncoding{},
        CartPoleActionEncoding{}
    );

    //let mut agent = TracingAgentGen::new(info_set, agent_comm, policy);

    let avg = test(&mut environment, &mut agent, 100).unwrap();
    println!("Average result before learning: {}.", avg);
    for epoch in 0..100{
        train_epoch(&mut environment, &mut agent, 10).unwrap();
        let avg = test(&mut environment, &mut agent, 100).unwrap();
        println!("Average result after epoch {}: {}.", epoch+1, avg);
    }



    Ok(())

     */

    let options = cart_pole::model::CartPoleModelOptions::parse();
    setup_logger(&options)?;

    let mut model = CartPoleModelPython::new_simple(&options)?;

    let summary = model.run_session(options)?;
    println!("summary:\n{:?}", summary);

    Ok(())


}