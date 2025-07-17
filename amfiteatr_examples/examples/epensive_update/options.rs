use std::path::PathBuf;
use log::LevelFilter;
use clap::Parser;
use amfiteatr_examples::expensive_update::domain::UpdateCost;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct ExpensiveUpdateOptions{
    #[arg(short = 'v', long = "log_level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    //#[arg(short = 'R', long = "rl_log_level", value_enum, default_value = "warn")]
    //pub rl_log_level: LevelFilter,

    #[arg(short = 'C', long = "classic_log_level", value_enum, default_value = "warn")]
    pub classic_log_level: LevelFilter,

    #[arg(short = 'a', long = "log_level_amfi", value_enum, default_value = "warn")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 'g', long = "games", default_value = "1")]
    pub games: u64,

    #[arg(short = 'n', long = "rounds", default_value = "32")]
    pub number_of_rounds: u64,

    #[arg(short = 'p', long = "agents", default_value = "32")]
    pub agents: u64,

    #[arg(short = 's', long = "small-update-cost-per-agent", default_value = "0")]
    pub small_update_cost_per_agent: UpdateCost,

    #[arg(short = 'b', long = "big-update-cost-per-agent", default_value = "1024")]
    pub big_update_cost_per_agent: UpdateCost,


}