use clap::Parser;
use crate::model::{MapModel, CentralModel};
use crate::options::{CCOptions, CommunicationMedium};

mod model;
pub mod options;

pub fn setup_logger(options: &CCOptions) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(options.log_level)
        .level_for("amfiteatr_examples", options.log_level)
        .level_for("amfiteatr_core", options.log_level_amfi);

    match &options.log_file{
        None => dispatch.chain(std::io::stdout()),
        Some(f) => dispatch.chain(fern::log_file(f)?)
    }

        .apply()?;
    Ok(())
}
fn main() -> Result<(), anyhow::Error>{

    let cli = CCOptions::parse();
    setup_logger(&cli)?;

    match cli.comm{
        CommunicationMedium::StaticMpsc | CommunicationMedium::StaticTcp => {
            let mut model = MapModel::new(&cli)?;
            model.run_several_games(cli.games);
        },

        CommunicationMedium::CentralMpsc => {
            let mut model = CentralModel::new(&cli)?;
            model.run_several_games(cli.games);
        },
        CommunicationMedium::Dynamic => {
            let mut model = MapModel::new_boxing(&cli)?;
            model.run_several_games(cli.games);
        }
    }

    Ok(())
}
