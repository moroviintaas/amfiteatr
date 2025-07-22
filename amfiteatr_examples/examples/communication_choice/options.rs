use std::path::PathBuf;
use clap::{Parser, ValueEnum};
use log::LevelFilter;
use amfiteatr_examples::expensive_update::domain::UpdateCost;

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommunicationMedium{
    StaticMpsc,
    StaticTcp,
    CentralMpsc,
    Dynamic
}
impl Default for CommunicationMedium{
    fn default() -> Self {
        Self::StaticMpsc
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct CCOptions{
    #[arg(short = 'v', long = "log-level", value_enum, default_value = "info")]
    pub log_level: LevelFilter,

    #[arg(short = 'a', long = "log-level-amfi", value_enum, default_value = "OFF")]
    pub log_level_amfi: LevelFilter,

    #[arg(short = 'p', long = "number-of-players", default_value = "10")]
    pub number_of_players: u64,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,


    #[arg(short = 'P', long = "port", default_value = "20000")]
    pub port: u16,

    #[arg(short = 'g', long = "games", default_value = "100")]
    pub games: u64,

    #[arg(short = 'r', long = "rounds", default_value = "10")]
    pub rounds: u64,

    #[arg(short = 'c', long = "communication", default_value = "static-mpsc")]
    pub comm: CommunicationMedium,

    #[arg(short = 's', long = "small-update-cost-per-agent", default_value = "0")]
    pub small_update_cost_per_agent: UpdateCost,

    #[arg(short = 'b', long = "big-update-cost-per-agent", default_value = "1024")]
    pub big_update_cost_per_agent: UpdateCost,

    #[arg(short = 'b', long = "big-update-cost-flat", default_value = "0")]
    pub big_update_cost_flat: UpdateCost,

    #[arg(short = 't', long = "number-of-dyn-tcp", default_value = "0")]
    pub number_of_dynamic_tcp_agents: u64,
}