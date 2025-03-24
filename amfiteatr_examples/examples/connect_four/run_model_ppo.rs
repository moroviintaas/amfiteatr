use clap::Parser;
use pyo3::prelude::PyAnyMethods;
use pyo3::Python;
use amfiteatr_rl::policy::ConfigPpo;
use amfiteatr_rl::tch::Device;
use crate::common::ErrorRL;
use crate::options::{ComputeDevice, ConnectFourOptions, Implementation};
use crate::rust::env::ConnectFourRustEnvState;
use crate::rust::env_wrapped::PythonPettingZooStateWrap;
use crate::rust::model::{C4PPOPolicy, ConnectFourModelRust};

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
        .level_for("amfiteatr_core", options.log_level_amfiteatr)
        .level_for("amfiteatr_rl", options.log_level_amfiteatr);

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

    let ppo_config = ConfigPpo::default();


    let device = match cli.device{
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::Cuda(0),
    };

    match cli.implementation{
        Implementation::Rust => {
            let mut model = ConnectFourModelRust::<ConnectFourRustEnvState, C4PPOPolicy>::new_ppo(
                &cli.layer_sizes_1[..], &cli.layer_sizes_2[..], device, ppo_config
            );
            model.run_session(cli.epochs, cli.num_episodes, cli.num_test_episodes)?;

        },

        Implementation::Wrap => {
            let mut model = ConnectFourModelRust::<PythonPettingZooStateWrap, C4PPOPolicy>::new_ppo(
                &cli.layer_sizes_1[..], &cli.layer_sizes_2[..], device, ppo_config
            );
            Python::with_gil(|py|{
                let pylogger = py.import("pettingzoo.utils.env_logger").unwrap();
                pylogger.getattr("EnvLogger").unwrap()
                   .call_method0("suppress_output").unwrap();

            });
            model.run_session(cli.epochs, cli.num_episodes, cli.num_test_episodes).unwrap();


        }



    }
    Ok(())


}
