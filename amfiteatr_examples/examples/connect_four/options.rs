use std::path::PathBuf;
use clap::{Parser, Subcommand};
use log::LevelFilter;

#[derive(Subcommand)]
pub enum Implementation{
    Rust
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ConnectFourOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfi", value_enum, default_value = "OFF")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,

    #[arg(short = 'e', long = "epochs", default_value = "100")]
    pub epochs: usize,

    #[arg(short = 'g', long = "games", default_value = "128")]
    pub num_episodes: usize,

    #[arg(short = 't', long = "test_games", default_value = "100")]
    pub num_test_episodes: usize,

    #[arg(short = 'P', long = "penalty", default_value = "-10")]
    pub penalty_for_illegal: f32,

    #[command(subcommand)]
    pub implementation: Implementation,

    #[arg( long = "layer_sizes_1", value_delimiter = ',',  value_terminator = "!", num_args = 1.., default_value = "128,256,128")]
    pub layer_sizes_1: Vec<i64>,
    #[arg( long = "layer_sizes_2", value_delimiter = ',', value_terminator = "!", num_args = 1.., default_value = "128,256,128")]
    pub layer_sizes_2: Vec<i64>,




    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}