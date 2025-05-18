use clap::Parser;
use pyo3::prelude::PyAnyMethods;
use pyo3::Python;
use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_rl::tch::Device;
use crate::common::ErrorRL;
use crate::options::{ComputeDevice, ConnectFourOptions, Implementation};
use crate::rust::env::ConnectFourRustEnvState;
use crate::rust::env_wrapped::PythonPettingZooStateWrap;
use crate::rust::model::{build_ppo_policy_shared, C4PPOPolicy, C4PPOPolicyShared, ConnectFourModelRust};

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
        .level_for("amfiteatr_rl", options.log_level_amfiteatr_rl);

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

    let mut ppo_config = ConfigPPO::default();

    ppo_config.gae_lambda = cli.gae_lambda;


    let device = match cli.device{
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::Cuda(0),
    };

    let policy = build_ppo_policy_shared(&cli.layer_sizes_0[..], device, ppo_config, cli.learning_rate)?;

    match cli.implementation{
        Implementation::Rust => {
            let mut model = ConnectFourModelRust::<ConnectFourRustEnvState, C4PPOPolicyShared>::new_ppo_generic(
                &cli,
                policy.clone(),
                policy,
                true
            );
            model.run_session(&cli)?;

        },

        Implementation::Wrap => {
            let mut model = ConnectFourModelRust::<PythonPettingZooStateWrap, C4PPOPolicy>::new_ppo(
                &cli
            );
            Python::with_gil(|py|{
                let pylogger = py.import("pettingzoo.utils.env_logger").unwrap();
                pylogger.getattr("EnvLogger").unwrap()
                   .call_method0("suppress_output").unwrap();

            });
            model.run_session(&cli).unwrap();


        }



    }
    Ok(())


}
