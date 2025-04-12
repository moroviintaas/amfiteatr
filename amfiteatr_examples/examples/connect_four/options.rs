use std::path::PathBuf;
use clap::Parser;
use log::LevelFilter;
use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum Implementation{
    Rust,
    Wrap
}

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum ComputeDevice{
    Cpu,
    Cuda,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ConnectFourOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfiteatr", value_enum, default_value = "OFF")]
    pub log_level_amfiteatr: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 'l', long = "gae-lambda")]
    pub gae_lambda: Option<f64>,

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

    #[arg(short = 'E', long = "extended-epochs", default_value = "0")]
    pub extended_epochs: usize,

    #[arg(short = 'g', long = "games", default_value = "128")]
    pub num_episodes: usize,

    #[arg(short = 't', long = "test-games", default_value = "100")]
    pub num_test_episodes: usize,

    #[arg(short = 'P', long = "penalty", default_value = "-10")]
    pub penalty_for_illegal: f32,

    #[arg(short = 'm', long = "mode", default_value = "rust")]
    pub implementation: Implementation,

    #[arg( long = "layer-sizes-1", value_delimiter = ',',  value_terminator = "!", num_args = 1.., default_value = "64,64")]
    pub layer_sizes_1: Vec<i64>,
    #[arg( long = "layer-sizes-2", value_delimiter = ',', value_terminator = "!", num_args = 1.., default_value = "64,64")]
    pub layer_sizes_2: Vec<i64>,

    #[arg(long = "save-train-params-summary", help = "File to save learn policy summary for epochs")]
    pub save_path_train_param_summary: Option<PathBuf>,

    #[arg(long = "save-test-epoch-summary", help = "File to save test epoch average results")]
    pub save_path_test_epoch: Option<PathBuf>,

    #[arg(long = "save-train-epoch-summary", help = "File to save train epoch average results")]
    pub save_path_train_epoch: Option<PathBuf>


    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}