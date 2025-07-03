use std::path::PathBuf;
use log::LevelFilter;
use clap::Parser;
use crate::common::{ComputeDevice, PolicySelect};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ReplicatorOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    /*
    #[arg(short = 'V', long = "general_log_level", value_enum, default_value = "warn")]
    pub general_log_level: LevelFilter,


     */
    #[arg(short = 'R', long = "rl_log_level", value_enum, default_value = "warn")]
    pub rl_log_level: LevelFilter,

    #[arg(short = 'C', long = "classic_log_level", value_enum, default_value = "warn")]
    pub classic_log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfi", value_enum, default_value = "warn")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 'd', long = "device", default_value = "cpu")]
    pub device: ComputeDevice,


    #[arg(short = 'p', long = "policy-select", default_value = "ppo")]
    pub policy_algo: PolicySelect,

    //#[arg(short = 'l', long = "gae-lambda")]
    //pub gae_lambda: Option<f64>,

    //#[arg(short = 'm', long = "minibatch-size", default_value = "16")]
    //pub mini_batch_size: usize,


    /*
    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,


     */
    #[arg(short = 'e', long = "epochs", default_value = "100")]
    pub epochs: usize,

    #[arg(short = 'g', long = "games", default_value = "128")]
    pub games: usize,

    #[arg(short = 'n', long = "rounds", default_value = "32")]
    pub number_of_rounds: usize,

    #[arg(short = 'H', long = "hawks", default_value = "0")]
    pub number_of_hawks: usize,

    #[arg(short = 'D', long = "doves", default_value = "0")]
    pub number_of_doves: usize,

    #[arg(short = 'm', long = "mixes", default_value = "0")]
    pub number_of_mixes: usize,

    #[arg(short = 'M', long = "mix-hawk-probability", default_value = "0.5")]
    pub mix_probability_of_hawk: f64,

    #[arg(short = 'l', long = "learners", default_value = "10")]
    pub number_of_learning: usize,

    #[arg(short = 'b', long = "batch", default_value = "64")]
    pub batch_size: usize,

    #[arg(long = "tensorboard", help = "Directory to save tensorboard output for epoch scores")]
    pub tboard: Option<PathBuf>,

    #[arg(long = "agent-tensorboard", help = "Directory to save tensorboard output for epoch scores")]
    pub agent_tboard: Option<PathBuf>,

    #[arg( long = "layer-sizes", value_delimiter = ',',  value_terminator = "!", num_args = 1.., default_value = "128,128")]
    pub layer_sizes: Vec<i64>,

    #[arg(long = "learning-rate", default_value = "0.001")]
    pub learning_rate: f64,

    #[arg(short = 'V', long = "value-loss-coefficient", default_value = "0.5")]
    pub value_loss_coef: f64,

    #[arg(short = 'E', long = "entropy-coefficient", default_value = "0.01")]
    pub entropy_coefficient: f64,



    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}