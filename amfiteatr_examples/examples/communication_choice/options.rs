use std::path::PathBuf;
use clap::{Parser, ValueEnum};
use log::LevelFilter;

#[derive(ValueEnum, Copy, Clone, Debug, PartialEq, Eq)]
pub enum CommunicationMedium{
    Mpsc,
    Tcp,
    CentralMpsc,
}
impl Default for CommunicationMedium{
    fn default() -> Self {
        Self::Mpsc
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
    pub number_of_players: usize,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,


    #[arg(short = 'P', long = "port", default_value = "20000")]
    pub port: u16,

    #[arg(short = 'g', long = "games", default_value = "100")]
    pub games: usize,

    #[arg(short = 'r', long = "rounds", default_value = "10")]
    pub rounds: usize,

    #[arg(short = 'c', long = "communication", default_value = "mpsc")]
    pub comm: CommunicationMedium,
}