use std::path::PathBuf;
use log::LevelFilter;
use clap::{ValueEnum, Parser};

#[derive(ValueEnum, Debug, Clone)]
pub enum SecondPolicy{
    Mixed,
    SwitchTwo,
    FibonacciForgive,
    ForgiveAfterTwo,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct EducatorOptions{

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

    #[arg(short = 'b', long = "batch", default_value = "64")]
    pub batch_size: usize,

    #[arg(short = 'n', long = "rounds", default_value = "10")]
    pub number_of_rounds: usize,

    #[arg(short = 'p', long = "policy", default_value = "std")]
    pub policy: SecondPolicy,

    #[arg(long = "reward_bias_scale", default_value = "0")]
    pub reward_bias_scale: f32,

    #[arg(short = 'M', long = "defect-proba", default_value = "0.5")]
    pub defect_proba: f32,

    #[arg(long = "defect-defect", default_value = "3")]
    pub defect_versus_defect: i64,

    #[arg(long = "coop-defect", default_value = "1")]
    pub coop_versus_defect: i64,

    #[arg(long = "defect-coop", default_value = "10")]
    pub defect_versus_coop: i64,

    #[arg(long = "coop-coop", default_value = "5")]
    pub coop_versus_coop: i64,


    //#[arg(short = 'r', long = "reward", default_value = "env")]
    //pub reward_source: RewardSource,

}