use std::path::PathBuf;
use clap::Parser;
use log::LevelFilter;
use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Implementation{
    Rust,
    Wrap,
    RustNd,
}

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum ComputeDevice{
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ConnectFourOptions{

    #[arg(short = 'v', long = "log-level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log-level-amfiteatr", value_enum, default_value = "off")]
    pub log_level_amfiteatr: LevelFilter,

    #[arg(short = 'A', long = "log-level-amfiteatr-rl", value_enum, default_value = "off")]
    pub log_level_amfiteatr_rl: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 'l', long = "gae-lambda")]
    pub gae_lambda: Option<f64>,

    #[arg(short = 'm', long = "minibatch-size", default_value = "16")]
    pub mini_batch_size: usize,

    /*
    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,


     */

    #[arg(short = 'd', long = "device", default_value = "cpu")]
    pub device: ComputeDevice,

    #[arg(short = 'e', long = "epochs", default_value = "100")]
    pub epochs: usize,

    #[arg(long = "limit-steps-in-epoch")]
    pub limit_steps_in_epochs: Option<usize>,

    #[arg(short = 'E', long = "extended-epochs", default_value = "0")]
    pub extended_epochs: usize,

    #[arg(short = 'g', long = "games", default_value = "128")]
    pub num_episodes: usize,

    #[arg(short = 'u', long = "updates-per-epoch", default_value = "4")]
    pub ppo_update_epochs: usize,

    #[arg(short = 't', long = "test-games", default_value = "0")]
    pub num_test_episodes: usize,

    #[arg(short = 'P', long = "penalty", default_value = "-10")]
    pub penalty_for_illegal: f32,

    #[arg(short = 'M', long = "mode", default_value = "rust")]
    pub implementation: Implementation,
    #[arg(long = "learning-rate", default_value = "1e-4")]
    pub learning_rate: f64,

    #[arg( long = "layer-sizes-0", value_delimiter = ',',  value_terminator = "!", num_args = 1.., default_value = "64,64")]
    pub layer_sizes_0: Vec<i64>,
    #[arg( long = "layer-sizes-1", value_delimiter = ',', value_terminator = "!", num_args = 1.., default_value = "64,64")]
    pub layer_sizes_1: Vec<i64>,

    #[arg(long = "save-train-params-summary", help = "File to save learn policy summary for epochs")]
    pub save_path_train_param_summary: Option<PathBuf>,

    #[arg(long = "save-test-epoch-summary", help = "File to save test epoch average results")]
    pub save_path_test_epoch: Option<PathBuf>,

    #[arg(long = "save-train-epoch-summary", help = "File to save train epoch average results")]
    pub save_path_train_epoch: Option<PathBuf>,


    #[arg(long = "tensorboard-policy-agent-0", help = "Directory to save tensorboard output for agent 0")]
    pub tboard_agent0: Option<PathBuf>,
    #[arg(long = "tensorboard-policy-agent-1", help = "Directory to save tensorboard output for agent 1")]
    pub tboard_agent1: Option<PathBuf>,

    #[arg(long = "tensorboard", help = "Directory to save tensorboard output for epoch scores")]
    pub tboard: Option<PathBuf>,

    #[arg(short = 'p', long = "rayon-pool-size", help = "Use threading pool of rayon with N threads") ]
    pub rayon_pool: Option<usize>
    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}