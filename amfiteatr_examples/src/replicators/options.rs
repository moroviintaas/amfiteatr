use std::path::PathBuf;
use log::LevelFilter;
use clap::Parser;

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

    /*
    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,


     */
    #[arg(short = 'e', long = "epochs", default_value = "10")]
    pub epochs: usize,

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

    #[arg(short = 'l', long = "learners", default_value = "100")]
    pub number_of_learning: usize,

    #[arg(short = 'b', long = "batch", default_value = "64")]
    pub batch_size: usize,






    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}