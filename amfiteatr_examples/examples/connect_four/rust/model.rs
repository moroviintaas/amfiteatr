use std::collections::HashMap;
use std::ops::{Add, Div};
use log::info;
use amfiteatr_core::agent::{AutomaticAgent, MultiEpisodeAutoAgent, Policy, PolicyAgent, ReseedAgent, TracingAgentGen};
use amfiteatr_core::comm::{
    StdAgentEndpoint,
    StdEnvironmentEndpoint
};
use amfiteatr_core::domain::Renew;
use amfiteatr_core::env::{GameStateWithPayoffs, HashMapEnvironment, ReseedEnvironment, RoundRobinPenalisingUniversalEnvironment, StatefulEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::{ActorCriticPolicy, ConfigA2C, ConfigPPO, LearningNetworkPolicy, PolicyDiscreteA2C, PolicyMaskingDiscreteA2C, PolicyMaskingDiscretePPO, PolicyDiscretePPO, TrainConfig};
use amfiteatr_rl::tch::{Device, nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, OptimizerConfig, VarStore};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{A2CNet, NeuralNetTemplate, TensorActorCritic};
use crate::common::{ConnectFourDomain, ConnectFourPlayer, ErrorRL};
use crate::rust::agent::{ConnectFourActionTensorRepresentation, ConnectFourInfoSet, ConnectFourTensorReprD1};


#[derive(Default, Copy, Clone)]

pub struct Summary{
    pub games_played: f64,
    pub scores: [f64;2],
    pub invalid_actions: [f64;2],
    pub game_steps: f64,
}


impl Summary{
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

impl Add<Summary> for Summary{
    type Output = Summary;

    fn add(self, rhs: Summary) -> Self::Output {
        Summary{
            games_played: self.games_played + rhs.games_played,
            scores: [self.scores[0] + rhs.scores[0], self.scores[1] + rhs.scores[1]],
            invalid_actions: [self.invalid_actions[0] + rhs.invalid_actions[0], self.invalid_actions[1] + rhs.invalid_actions[1]],
            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}
impl Add<&Summary> for &Summary{
    type Output = Summary;

    fn add(self, rhs: &Summary) -> Self::Output {
        Summary{
            games_played: self.games_played + rhs.games_played,
            scores: [self.scores[0] + rhs.scores[0], self.scores[1] + rhs.scores[1]],
            invalid_actions: [self.invalid_actions[0] + rhs.invalid_actions[0], self.invalid_actions[1] + rhs.invalid_actions[1]],
            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}


impl Div<f64> for Summary{
    type Output = Summary;

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

fn build_a2c_policy(layer_sizes: &[i64], device: Device, gae_lambda: Option<f64>) -> Result<C4A2CPolicy, AmfiteatrRlError<ConnectFourDomain>>{
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

    let mut config = ConfigA2C::default();
    config.gae_lambda = gae_lambda;
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
fn build_a2c_policy_masking(layer_sizes: &[i64], device: Device, gae_lambda: Option<f64>) -> Result<C4A2CPolicyMasking, AmfiteatrRlError<ConnectFourDomain>>{
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

    let mut config = ConfigA2C::default();
    config.gae_lambda = gae_lambda;
    Ok(PolicyMaskingDiscreteA2C::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{}
    )
    )
}
fn build_ppo_policy_masking(layer_sizes: &[i64], device: Device, config: ConfigPPO) -> Result<C4PPOPolicyMasking, AmfiteatrRlError<ConnectFourDomain>>{
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
    let optimiser = Adam::default().build(&var_store, 1e-4)?;
    let net = A2CNet::new(var_store, net, );

    Ok(PolicyMaskingDiscretePPO::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{})
    )
}
fn build_ppo_policy(layer_sizes: &[i64], device: Device, config: ConfigPPO) -> Result<C4PPOPolicy, AmfiteatrRlError<ConnectFourDomain>>{
    Ok(build_ppo_policy_masking(layer_sizes, device, config)?.base)

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
pub struct ConnectFourModelRust<S: GameStateWithPayoffs<ConnectFourDomain>, P: LearningNetworkPolicy<ConnectFourDomain>>{

    env: Environment<S>,
    agent1: Agent<P>,
    agent2: Agent<P>,

}

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
impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S, C4A2CPolicy>{
    #[allow(dead_code)]
    pub fn new_a2c(agent_layers_1: &[i64], agent_layers_2: &[i64], device: Device, gae_lambda: Option<f64>) -> Self
        where S: Default{

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let agent_policy_1 = build_a2c_policy(agent_layers_1, device, gae_lambda).unwrap();
        let agent_policy_2 = build_a2c_policy(agent_layers_2, device, gae_lambda).unwrap();
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_1);
        let agent_2 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_2);

        Self{
            env,
            agent1: agent_1,
            agent2: agent_2
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S, C4A2CPolicyMasking>{
    #[allow(dead_code)]
    pub fn new_a2c_masking(agent_layers_1: &[i64], agent_layers_2: &[i64], device: Device, gae_lambda: Option<f64>) -> Self
        where S: Default{

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let agent_policy_1 = build_a2c_policy_masking(agent_layers_1, device, gae_lambda).unwrap();
        let agent_policy_2 = build_a2c_policy_masking(agent_layers_2, device, gae_lambda).unwrap();
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_1);
        let agent_2 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_2);

        Self{
            env,
            agent1: agent_1,
            agent2: agent_2
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S,C4PPOPolicy>{
    #[allow(dead_code)]
    pub fn new_ppo(agent_layers_1: &[i64], agent_layers_2: &[i64], device: Device, config_ppo: ConfigPPO) -> Self
    where S: Default{

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let agent_policy_1 = build_ppo_policy(agent_layers_1, device, config_ppo).unwrap();
        let agent_policy_2 = build_ppo_policy(agent_layers_2, device, config_ppo).unwrap();
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_1);
        let agent_2 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_2);

        Self{
            env,
            agent1: agent_1,
            agent2: agent_2
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRust<S,C4PPOPolicyMasking>{
    #[allow(dead_code)]
    pub fn new_ppo_masking(agent_layers_1: &[i64], agent_layers_2: &[i64], device: Device, config_ppo: ConfigPPO) -> Self
    where S: Default{

        let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

        let mut hm = HashMap::new();
        hm.insert(ConnectFourPlayer::One, c_env1);
        hm.insert(ConnectFourPlayer::Two, c_env2);


        let env = Environment::new(S::default(), hm, );
        let agent_policy_1 = build_ppo_policy_masking(agent_layers_1, device, config_ppo).unwrap();
        let agent_policy_2 = build_ppo_policy_masking(agent_layers_2, device, config_ppo).unwrap();
        let agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_1);
        let agent_2 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_2);

        Self{
            env,
            agent1: agent_1,
            agent2: agent_2
        }
    }
}

impl<
    S:  GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
    P: LearningNetworkPolicy<ConnectFourDomain>
> ConnectFourModelRust<S,P>
where <P as Policy<ConnectFourDomain>>::InfoSetType: Renew<ConnectFourDomain, ()> + Clone{



    pub fn play_one_game(&mut self, store_episode: bool) -> Result<Summary, AmfiteatrRlError<ConnectFourDomain>>{
        let mut summary = Summary::default();
        self.env.reseed(())?;
        self.agent1.reseed(())?;
        self.agent2.reseed(())?;

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
                self.agent1.run().unwrap()
            });
            s.spawn(||{
                self.agent2.run().unwrap()
            });
        });

        summary.scores = [
            self.env.state().state_payoff_of_player(&ConnectFourPlayer::One) as f64,
            self.env.state().state_payoff_of_player(&ConnectFourPlayer::Two) as f64,
        ];

        summary.games_played = 1.0;
        summary.game_steps = self.env.completed_steps() as f64;

        if store_episode{
            self.agent1.store_episode();
            self.agent2.store_episode();
        }

        Ok(summary)
    }

    pub fn play_epoch(&mut self, number_of_games: usize, summarize: bool, training_epoch: bool) -> Result<Summary, AmfiteatrRlError<ConnectFourDomain>>{
        self.agent1.clear_episodes();
        self.agent2.clear_episodes();
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
            let summary_sum: Summary = vec.iter().fold(Summary::default(), |acc,x| &acc+x);
            let n = number_of_games as f64;
            return Ok(Summary{
                games_played: summary_sum.games_played,
                scores: [summary_sum.scores[0]/n, summary_sum.scores[1]/n],
                invalid_actions: [summary_sum.invalid_actions[0], summary_sum.invalid_actions[1]],
                game_steps: summary_sum.game_steps / n,
            });
        }
        Ok(Summary::default())
    }

    pub fn train_agents_on_experience(&mut self) -> Result<(), ErrorRL>{
        let t1 = self.agent1.take_episodes();
        self.agent1.policy_mut().train_on_trajectories_env_reward(&t1)?;
        let t2 = self.agent2.take_episodes();
        self.agent2.policy_mut().train_on_trajectories_env_reward(&t2)?;
        //self.agent2.policy_mut().train_on_trajectories(&t2, |step| Tensor::from(-1.0 + (2.0 * step.reward())))?;

        Ok(())
    }

    pub fn train_agent1_on_experience(&mut self) -> Result<(), ErrorRL>{
        let t1 = self.agent1.take_episodes();
        self.agent1.policy_mut().train_on_trajectories_env_reward(&t1)?;
        let _t2 = self.agent2.take_episodes();
        //self.agent2.policy_mut().train_on_trajectories_env_reward(&t2)?;

        Ok(())
    }


    pub fn run_session(&mut self, epochs: usize, episodes: usize, test_episodes: usize, extended_epochs: usize) -> Result<(), ErrorRL>{
        info!("Starting session");
        let pre_training_summary = self.play_epoch(test_episodes, true, false)?;
        info!("Summary before training: {}", pre_training_summary.describe_as_collected());

        for e in 0..epochs{
            self.play_epoch(episodes, false, true)?;
            self.train_agents_on_experience()?;
            let s =self.play_epoch(test_episodes, true, false)?;
            info!("Summary of tests after epoch {}: {}", e+1, s.describe_as_collected())
        }

        for e in 0..extended_epochs{
            self.play_epoch(episodes, false, true)?;
            self.train_agent1_on_experience()?;
            let s =self.play_epoch(test_episodes, true, false)?;
            info!("Summary of tests after extended epoch {}: {}", e+1, s.describe_as_collected())
        }


        Ok(())
    }


}