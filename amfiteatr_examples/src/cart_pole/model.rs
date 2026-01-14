use std::ops::Div;
use std::ops::Add;
use std::fs::File;
use std::path::PathBuf;
use clap::Parser;
use log::LevelFilter;
use tboard::EventWriter;
use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_core::agent::{AutomaticAgent, MultiEpisodeAutoAgent, ReseedAgent, TracingAgentGen};
use amfiteatr_core::comm::{AgentMpscAdapter, EnvironmentMpscPort};
use amfiteatr_core::env::{AutoEnvironment, AutoEnvironmentWithScores, BasicEnvironment, ReseedEnvironment, SequentialGameState, StatefulEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::PolicyDiscretePPO;
use amfiteatr_rl::tch;
use amfiteatr_rl::tch::nn::{AdamW, VarStore};
use amfiteatr_rl::torch_net::{build_network_operator_ac, Layer, NeuralNetActorCritic};
use crate::cart_pole::agent::CartPoleObservationEncoding;
use crate::cart_pole::common::{CartPoleActionEncoding, CartPoleObservation, CartPoleScheme, SINGLE_PLAYER_ID};
use crate::cart_pole::env::CartPoleEnvStateRust;
use amfiteatr_rl::tch::nn::OptimizerConfig;
use crate::connect_four::common::{ConnectFourPlayer, ConnectFourScheme};
use amfiteatr_core::env::GameStateWithPayoffs;

pub type CartPolePolicy = PolicyDiscretePPO<CartPoleScheme, CartPoleObservation, CartPoleObservationEncoding, CartPoleActionEncoding>;


#[derive(Clone, Debug, Default)]
pub struct EpochSummary {
    pub games_played: f64,
    pub score: f64,
    pub game_steps: f64,
}

impl Add<EpochSummary> for EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            score: self.score + rhs.score,

            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}
impl Add<&EpochSummary> for &EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: &EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            score: self.score + rhs.score,

            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}


impl Div<f64> for EpochSummary {
    type Output = EpochSummary;

    fn div(self, rhs: f64) -> Self::Output {
        Self{
            games_played: self.games_played / rhs,
            score: self.score/ rhs,
            game_steps: self.game_steps/ rhs,
        }
    }
}

#[derive(Clone, Parser, Debug)]
pub struct CartPoleModelOptions{
    #[arg(long = "tensorboard", help = "directory for tensorboard summary")]
    pub tboard_writer_path: Option<PathBuf>,
    #[arg(long = "tensorboard-agent", help = "directory for tensorboard agent summary")]
    pub tboard_agent_writer_path: Option<PathBuf>,
    #[arg(short = 'b', long = "sutton-barto-reward", help = "Use Sutton-Barto reward")]
    pub sutton_barto: bool,

    #[arg(short = 'v', long = "log-level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

}

pub struct CartPoleModelRust{
    env: BasicEnvironment<CartPoleScheme, CartPoleEnvStateRust, EnvironmentMpscPort<CartPoleScheme>>,
    agent: TracingAgentGen<CartPoleScheme, CartPolePolicy, AgentMpscAdapter<CartPoleScheme>>,
    tboard_writer: Option<tboard::EventWriter<File>>,
}




impl CartPoleModelRust{
    pub fn new_simple(options: &CartPoleModelOptions) -> anyhow::Result<Self>{
        let initial_state = CartPoleEnvStateRust::new(options.sutton_barto);
        let mut env_communicator = EnvironmentMpscPort::new();
        let obs = initial_state.first_observations().unwrap()[0].1.clone();
        let agent_comm = env_communicator.register_agent(SINGLE_PLAYER_ID)?;
        let env = BasicEnvironment::new(initial_state, env_communicator);

        let operator = build_network_operator_ac(vec![Layer::Linear(64), Layer::Linear(64)],
                                                 vec![4], 2);
        let vs = VarStore::new(tch::Device::Cpu);
        let optimizer = AdamW::default().build(&vs, 0.001)?;
        let network = NeuralNetActorCritic::new(vs, operator);

        let mut policy = CartPolePolicy::new(
            ConfigPPO::default(),
            network,
            optimizer,
            CartPoleObservationEncoding {},
            CartPoleActionEncoding {}
        );

        if let Some(p) = &options.tboard_agent_writer_path{
            policy.add_tboard_directory(p)?;
        }

        let agent = TracingAgentGen::new(obs, agent_comm, policy);

        let tboard_writer = match &options.tboard_agent_writer_path {
            Some(p) => Some(EventWriter::create(p)?),
            None => None
        };

        Ok(Self{
            env, agent, tboard_writer
        })
    }

    pub fn play_one_game(&mut self, store_episode: bool, truncate_at_step: Option<usize>) -> Result<EpochSummary, AmfiteatrRlError<CartPoleScheme>>{
        let mut summary = EpochSummary::default();
        self.env.reseed(())?;
        self.agent.reseed(())?;

        std::thread::scope(|s|{
            s.spawn(|| {
                let r = self.env.run_with_scores_truncating( truncate_at_step);
                r.inspect_err(|e|{
                    log::error!("Error {e}")
                })

            });
            s.spawn(||{
                self.agent.run().unwrap()
            });
        });

        summary.score = self.env.state().state_payoff_of_player(&SINGLE_PLAYER_ID) as f64;
        summary.games_played = 1.0;
        summary.game_steps = self.env.completed_steps() as f64;

        if store_episode{
            self.agent.store_episode()?;
        }

        Ok(summary)
    }
}