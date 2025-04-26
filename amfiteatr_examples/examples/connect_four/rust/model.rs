use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::ops::{Add, Div};
use std::path::PathBuf;
use log::info;
use serde::{Deserialize, Serialize};
use tboard::EventWriter;
use amfiteatr_core::agent::{AutomaticAgent, MultiEpisodeAutoAgent, Policy, PolicyAgent, ReseedAgent, TracingAgentGen};
use amfiteatr_core::comm::{
    StdAgentEndpoint,
    StdEnvironmentEndpoint
};
use amfiteatr_core::domain::Renew;
use amfiteatr_core::env::{GameStateWithPayoffs, HashMapEnvironment, ReseedEnvironment, RoundRobinPenalisingUniversalEnvironment, StatefulEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::{ActorCriticPolicy, ConfigA2C, ConfigPPO, LearningNetworkPolicy, PolicyDiscreteA2C, PolicyMaskingDiscreteA2C, PolicyMaskingDiscretePPO, PolicyDiscretePPO, TrainConfig, PolicyHelperA2C};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, OptimizerConfig, VarStore};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorActorCritic};
use crate::common::{ConnectFourDomain, ConnectFourPlayer, ErrorRL};
use crate::rust::agent::{ConnectFourActionTensorRepresentation, ConnectFourInfoSet, ConnectFourTensorReprD1};

use amfiteatr_rl::policy::LearnSummary;
use crate::options::{ComputeDevice, ConnectFourOptions};

#[derive(Default, Copy, Clone, Serialize, Deserialize)]

pub struct EpochSummary {
    pub games_played: f64,
    pub scores: [f64;2],
    pub invalid_actions: [f64;2],
    pub game_steps: f64,
}



impl EpochSummary {
    pub fn describe_as_collected(&self) -> String{
        format!("games played: {}, average game steps: {} | average score: P1: {}, P2: {}, number of illegal actions: P1: {}, P2: {}",
        self.games_played, self.game_steps, self.scores[0], self.scores[1], self.invalid_actions[0], self.invalid_actions[1])
    }
}
/*
impl Display for Summary{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Game steps: {} | Player One: score: {}, faul | ", self.game_steps)
    }
}

 */

impl Add<EpochSummary> for EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            scores: [self.scores[0] + rhs.scores[0], self.scores[1] + rhs.scores[1]],
            invalid_actions: [self.invalid_actions[0] + rhs.invalid_actions[0], self.invalid_actions[1] + rhs.invalid_actions[1]],
            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}
impl Add<&EpochSummary> for &EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: &EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            scores: [self.scores[0] + rhs.scores[0], self.scores[1] + rhs.scores[1]],
            invalid_actions: [self.invalid_actions[0] + rhs.invalid_actions[0], self.invalid_actions[1] + rhs.invalid_actions[1]],
            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}


impl Div<f64> for EpochSummary {
    type Output = EpochSummary;

    fn div(self, rhs: f64) -> Self::Output {
        Self{
            games_played: self.games_played / rhs,
            scores: [self.scores[0]/ rhs, self.scores[1]/ rhs],
            invalid_actions: [self.invalid_actions[0]/ rhs, self.invalid_actions[1]/ rhs],
            game_steps: self.game_steps/ rhs,
        }
    }
}

fn build_a2c_policy_old(layer_sizes: &[i64], device: Device) -> Result<C4A2CPolicyOld, AmfiteatrRlError<ConnectFourDomain>>{
    let var_store = VarStore::new(device);
    let input_shape = ConnectFourTensorReprD1{}.desired_shape()[0];
    let hidden_layers = &layer_sizes;
    let network_pattern = NeuralNetTemplate::new(|path| {
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(path / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(path / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(path / "al", input_shape, 7, Default::default()),
                 nn::linear(path / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(path / "al", ld, 7, Default::default()),
                 nn::linear(path / "cl", ld, 1, Default::default()))
            }
        };
        let device = path.device();
        {move |xs: &Tensor|{
            if seq.is_empty(){
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            } else {
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }
        }}
    });

    let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, 1e-4)?;
    let net = A2CNet::new(var_store, net, );

    Ok(ActorCriticPolicy::new(
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        TrainConfig{gamma: 0.9})
    )
}

fn build_a2c_policy(layer_sizes: &[i64], device: Device, config: ConfigA2C, learning_rate: f64) -> Result<C4A2CPolicy, AmfiteatrRlError<ConnectFourDomain>>{
    let var_store = VarStore::new(device);
    let input_shape = ConnectFourTensorReprD1{}.desired_shape()[0];
    let hidden_layers = &layer_sizes;
    let network_pattern = NeuralNetTemplate::new(|path| {
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(path / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(path / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(path / "al", input_shape, 7, Default::default()),
                 nn::linear(path / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(path / "al", ld, 7, Default::default()),
                 nn::linear(path / "cl", ld, 1, Default::default()))
            }
        };
        let device = path.device();
        {move |xs: &Tensor|{
            if seq.is_empty(){
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            } else {
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }
        }}
    });

    let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, learning_rate)?;
    let net = A2CNet::new(var_store, net, );

    //let mut config = ConfigA2C::default();
    //config.gae_lambda = gae_lambda;
    Ok(PolicyDiscreteA2C::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{}
        )
    )
}
#[allow(dead_code)]
fn build_a2c_policy_masking(layer_sizes: &[i64], device: Device, config: ConfigA2C, learning_rate: f64) -> Result<C4A2CPolicyMasking, AmfiteatrRlError<ConnectFourDomain>>{
    let var_store = VarStore::new(device);
    let input_shape = ConnectFourTensorReprD1{}.desired_shape()[0];
    let hidden_layers = &layer_sizes;
    let network_pattern = NeuralNetTemplate::new(|path| {
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(path / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(path / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(path / "al", input_shape, 7, Default::default()),
                 nn::linear(path / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(path / "al", ld, 7, Default::default()),
                 nn::linear(path / "cl", ld, 1, Default::default()))
            }
        };
        let device = path.device();
        {move |xs: &Tensor|{
            if seq.is_empty(){
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            } else {
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }
        }}
    });

    let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, learning_rate)?;
    let net = A2CNet::new(var_store, net, );

    //let mut config = ConfigA2C::default();
    // config.gae_lambda = gae_lambda;
    Ok(PolicyMaskingDiscreteA2C::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{}
    )
    )
}
fn build_ppo_policy_masking(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicyMasking, AmfiteatrRlError<ConnectFourDomain>>{
    let var_store = VarStore::new(device);
    //let var_store = VarStore::new(Device::Cuda(0));
    let input_shape = ConnectFourTensorReprD1{}.desired_shape()[0];
    let hidden_layers = &layer_sizes;
    let network_pattern = NeuralNetTemplate::new(|path| {
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(path / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(path / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(path / "al", input_shape, 7, Default::default()),
                 nn::linear(path / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(path / "al", ld, 7, Default::default()),
                 nn::linear(path / "cl", ld, 1, Default::default()))
            }
        };
        let device = path.device();
        {move |xs: &Tensor|{
            if seq.is_empty(){
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            } else {
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }
        }}
    });

    let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, learning_rate)?;
    let net = A2CNet::new(var_store, net, );

    Ok(PolicyMaskingDiscretePPO::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{})
    )
}
fn build_ppo_policy(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicy, AmfiteatrRlError<ConnectFourDomain>>{
    Ok(build_ppo_policy_masking(layer_sizes, device, config, learning_rate)?.base)

}


pub type C4A2CPolicyOld = ActorCriticPolicy<ConnectFourDomain, ConnectFourInfoSet, ConnectFourTensorReprD1>;
pub type C4A2CPolicy = PolicyDiscreteA2C<ConnectFourDomain, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4A2CPolicyMasking = PolicyMaskingDiscreteA2C<ConnectFourDomain, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4PPOPolicy = PolicyDiscretePPO<ConnectFourDomain, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
pub type C4PPOPolicyMasking = PolicyMaskingDiscretePPO<ConnectFourDomain, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
type Environment<S> = HashMapEnvironment<ConnectFourDomain, S, StdEnvironmentEndpoint<ConnectFourDomain>>;
type Agent<P> = TracingAgentGen<ConnectFourDomain, P, StdAgentEndpoint<ConnectFourDomain>>;
pub struct ConnectFourModelRust<S: GameStateWithPayoffs<ConnectFourDomain>, P: LearningNetworkPolicy<ConnectFourDomain, Summary=LearnSummary>>{

    env: Environment<S>,
    agent0: Agent<P>,
    agent1: Agent<P>,

    //model_tboard: Option<tboard::EventWriter<File>>

}
/*
impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S, C4A2CPolicyOld>{
    #[allow(dead_code)]
    pub fn new_a2c_old(agent_layers_1: &[i64], agent_layers_2: &[i64], device: Device) -> Self
    where S: Default{

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let agent_policy_1 = build_a2c_policy_old(agent_layers_1, device).unwrap();
        let agent_policy_2 = build_a2c_policy_old(agent_layers_2, device).unwrap();
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_1);
        let agent_2 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_2);

        Self{
            env,
            agent1: agent_1,
            agent2: agent_2
        }
    }
}
*/

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S, C4A2CPolicy>{
    #[allow(dead_code)]
    pub fn new_a2c(options: &ConnectFourOptions) -> Self
        where S: Default{

        let mut config_a2c = ConfigA2C::default();
        config_a2c.gae_lambda = options.gae_lambda;

        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let mut agent_policy_0 = build_a2c_policy(&options.layer_sizes_0[..], device, config_a2c, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_a2c_policy(&options.layer_sizes_1[..], device, config_a2c, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }
        let agent_0 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0);
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1);

        //let model_tboard = options.tboard.as_ref()
        //    .and_then(|p| Some(EventWriter::create(p).unwrap()));

        Self{
            env,
            agent0: agent_0,
            agent1: agent_1,
            //model_tboard,
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S, C4A2CPolicyMasking>{
    #[allow(dead_code)]
    pub fn new_a2c_masking(options: &ConnectFourOptions) -> Self
        where S: Default{

        let mut config_a2c = ConfigA2C::default();
        config_a2c.gae_lambda = options.gae_lambda;

        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let mut agent_policy_0 = build_a2c_policy_masking(&options.layer_sizes_0[..], device, config_a2c, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_a2c_policy_masking(&options.layer_sizes_1[..], device, config_a2c, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }
        let agent_0 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0);
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1);

        //let model_tboard = options.tboard.as_ref()
        //    .and_then(|p| Some(EventWriter::create(p).unwrap()));

        Self{
            env,
            agent0: agent_0,
            agent1: agent_1,
            //model_tboard,
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S,C4PPOPolicy>{
    #[allow(dead_code)]
    pub fn new_ppo(options: &ConnectFourOptions) -> Self
    where S: Default{

        let mut config_ppo = ConfigPPO::default();
        config_ppo.gae_lambda = options.gae_lambda;

        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let mut agent_policy_0 = build_ppo_policy(&options.layer_sizes_0[..], device, config_ppo, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_ppo_policy(&options.layer_sizes_1[..], device, config_ppo, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }
        let agent_0 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0);
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1);

        //let model_tboard = options.tboard.as_ref()
        //    .and_then(|p| Some(EventWriter::create(p).unwrap()));

        Self{
            env,
            agent0: agent_0,
            agent1: agent_1,
            //model_tboard,
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S,C4PPOPolicyMasking>{
    #[allow(dead_code)]
    pub fn new_ppo_masking(options: &ConnectFourOptions) -> Self
    where S: Default{

        let mut config_ppo = ConfigPPO::default();
        config_ppo.gae_lambda = options.gae_lambda;

        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let mut agent_policy_0 = build_ppo_policy_masking(&options.layer_sizes_0[..], device, config_ppo, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_ppo_policy_masking(&options.layer_sizes_1[..], device, config_ppo, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }
        let agent_0 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0);
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1);

        //let model_tboard = options.tboard.as_ref()
        //    .and_then(|p| Some(EventWriter::create(p).unwrap()));

        Self{
            env,
            agent0: agent_0,
            agent1: agent_1,
            //model_tboard,
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
    P: LearningNetworkPolicy<ConnectFourDomain, Summary = LearnSummary> + PolicyHelperA2C<ConnectFourDomain>
> ConnectFourModelRust<S,P>
where <P as Policy<ConnectFourDomain>>::InfoSetType: Renew<ConnectFourDomain, ()> + Clone{



    pub fn play_one_game(&mut self, store_episode: bool) -> Result<EpochSummary, AmfiteatrRlError<ConnectFourDomain>>{
        let mut summary = EpochSummary::default();
        self.env.reseed(())?;
        self.agent0.reseed(())?;
        self.agent1.reseed(())?;

        std::thread::scope(|s|{
            s.spawn(|| {
                let r = self.env.run_round_robin_with_rewards_penalise(|_,_| -10.0);
                if let Err(e) = r{
                    if let AmfiteatrError::Game {source: game_error} = e{
                        if let Some(fauler) = game_error.fault_of_player(){
                            summary.invalid_actions[fauler.index()] = 1.0;
                        }
                    }
                }
            });
            s.spawn(||{
                self.agent0.run().unwrap()
            });
            s.spawn(||{
                self.agent1.run().unwrap()
            });
        });

        summary.scores = [
            self.env.state().state_payoff_of_player(&ConnectFourPlayer::One) as f64,
            self.env.state().state_payoff_of_player(&ConnectFourPlayer::Two) as f64,
        ];

        summary.games_played = 1.0;
        summary.game_steps = self.env.completed_steps() as f64;

        if store_episode{
            self.agent0.store_episode();
            self.agent1.store_episode();
        }

        Ok(summary)
    }

    pub fn play_epoch(&mut self, number_of_games: usize, summarize: bool, training_epoch: bool) -> Result<EpochSummary, AmfiteatrRlError<ConnectFourDomain>>{
        self.agent0.clear_episodes();
        self.agent1.clear_episodes();
        let mut vec = match summarize{
            true => Vec::with_capacity(number_of_games as usize),
            false => Vec::with_capacity(0),
        };
        for _ in 0..number_of_games{
            let summary = self.play_one_game(training_epoch)?;
            if summarize{
                vec.push(summary);
            }
        }
        if summarize{
            let summary_sum: EpochSummary = vec.iter().fold(EpochSummary::default(), |acc, x| &acc+x);
            let n = number_of_games as f64;
            return Ok(EpochSummary {
                games_played: summary_sum.games_played,
                scores: [summary_sum.scores[0]/n, summary_sum.scores[1]/n],
                invalid_actions: [summary_sum.invalid_actions[0]/n, summary_sum.invalid_actions[1]/n],
                game_steps: summary_sum.game_steps / n,
            });
        }
        Ok(EpochSummary::default())
    }

    //pub fn train_epoch(&mut self, number_of_games: usize) -> Result<Summary, AmfiteatrRlError<ConnectFourDomain>>{

    //}

    pub fn train_agents_on_experience(&mut self) -> Result<(LearnSummary,LearnSummary), ErrorRL>{
        let t1 = self.agent0.take_episodes();
        let s1 = self.agent0.policy_mut().train_on_trajectories_env_reward(&t1)?;
        let t2 = self.agent1.take_episodes();
        let s2 = self.agent1.policy_mut().train_on_trajectories_env_reward(&t2)?;
        //self.agent2.policy_mut().train_on_trajectories(&t2, |step| Tensor::from(-1.0 + (2.0 * step.reward())))?;

        Ok((s1, s2))
    }

    pub fn train_agent1_on_experience(&mut self) -> Result<LearnSummary, ErrorRL>{
        let t1 = self.agent0.take_episodes();
        let s1 = self.agent0.policy_mut().train_on_trajectories_env_reward(&t1)?;
        let _t2 = self.agent1.take_episodes();
        //self.agent2.policy_mut().train_on_trajectories_env_reward(&t2)?;

        Ok(s1)
    }


    pub fn run_session(&mut self, options: &ConnectFourOptions)
        -> Result<(), ErrorRL>{
        info!("Starting session");
        let pre_training_summary = self.play_epoch(options.num_test_episodes, true, false)?;
        info!("Summary before training: {}", pre_training_summary.describe_as_collected());

        let mut results_learn = Vec::with_capacity(options.epochs);
        let mut results_test = Vec::with_capacity(options.epochs);
        let mut l_summaries_1 = Vec::with_capacity(options.epochs);
        let mut l_summaries_2 = Vec::with_capacity(options.epochs);

        for e in 0..options.epochs{
            let s = self.play_epoch(options.num_episodes, true, true)?;
            if let Some(tboard) = self.agent0.policy_mut().tboard_writer(){
                tboard.write_scalar(e as i64, "train_epoch/score", s.scores[0] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent0 scores".into(), error: format!("{e}")})?;
                tboard.write_scalar(e as i64, "train_epoch/illegal_moves", s.invalid_actions[0] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent0 bad action count".into(), error: format!("{e}")})?;

            }
            if let Some(tboard) = self.agent1.policy_mut().tboard_writer(){

                tboard.write_scalar(e as i64, "train_epoch/score", s.scores[1] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent1 scores".into(), error: format!("{e}")})?;
               tboard.write_scalar(e as i64, "train_epoch/illegal_moves", s.invalid_actions[1] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent1 bad action count".into(), error: format!("{e}")})?;
            }
            results_learn.push(s);
            let (s1,s2) = self.train_agents_on_experience()?;

            info!("Training epoch {}: Critic losses {:.3}, {:.3}; Actor loss {:.3}, {:.3}; entropy loss {:.3}, {:.3}",
                e+1, s1.value_loss.unwrap(), s2.value_loss.unwrap(),
                s1.policy_gradient_loss.unwrap(), s2.policy_gradient_loss.unwrap(),
                s1.entropy_loss.unwrap(), s2.entropy_loss.unwrap()
            );
            let s =self.play_epoch(options.num_test_episodes, true, false)?;
            results_test.push(s);
            l_summaries_1.push(s1);
            l_summaries_2.push(s2);



            info!("Summary of tests after epoch {}: {}", e+1, s.describe_as_collected())
        }
        if let Some(result_file) = &options.save_path_train_param_summary{
            let yaml = serde_yaml::to_string(&(l_summaries_1, l_summaries_2)).unwrap();
            let mut file = File::create(&result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        if let Some(result_file) = &options.save_path_test_epoch{
            let yaml = serde_yaml::to_string(&results_test).unwrap();
            let mut file = File::create(&result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        if let Some(result_file) = &options.save_path_train_epoch{
            let yaml = serde_yaml::to_string(&results_learn).unwrap();
            let mut file = File::create(&result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        for e in 0..options.extended_epochs{
            let s = self.play_epoch(options.num_episodes, true, true)?;
            if let Some(tboard) = self.agent0.policy_mut().tboard_writer(){
                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/score", s.scores[0] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent0 scores".into(), error: format!("{e}")})?;
                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/illegal_moves", s.invalid_actions[0] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent0 bad action count".into(), error: format!("{e}")})?;

            }
            if let Some(tboard) = self.agent1.policy_mut().tboard_writer(){

                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/score", s.scores[1] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent1 scores".into(), error: format!("{e}")})?;
                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/illegal_moves", s.invalid_actions[1] as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving Agent1 bad action count".into(), error: format!("{e}")})?;
            }
            let s1 = self.train_agent1_on_experience()?;
            info!("Training only agent 1 epoch {}: Critic losses {:.3}; Actor loss {:.3}; entropy loss {:.3}",
                e+1, s1.value_loss.unwrap(),
                s1.policy_gradient_loss.unwrap(),
                s1.entropy_loss.unwrap(),
            );
            let s =self.play_epoch(options.num_test_episodes, true, false)?;
            info!("Summary of tests after extended epoch {}: {}", e+1, s.describe_as_collected())
        }




        Ok(())
    }


}