use clap::Parser;
use amfiteatr_rl::error::AmfiteatrRlError;
use crate::common::ErrorRL;
use crate::options::{ConnectFourOptions, Implementation};
use crate::rust::env::ConnectFourRustEnvState;
use crate::rust::model::ConnectFourModelRust;

mod rust;
pub mod common;
mod options;



pub fn setup_logger(options: &ConnectFourOptions) -> Result<(), fern::InitError> {
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
fn main() -> Result<(), ErrorRL>{

    let cli = ConnectFourOptions::parse();
    setup_logger(&cli).unwrap();

    match cli.implementation{
        Implementation::Rust => {
            let mut model = ConnectFourModelRust::<ConnectFourRustEnvState>::new(
                &cli.layer_sizes_1[..], &cli.layer_sizes_2[..]
            );
            model.run_session(cli.epochs, cli.num_episodes, cli.num_test_episodes)?;

        }
    }
    Ok(())


}
