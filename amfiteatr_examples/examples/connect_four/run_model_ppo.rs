use clap::Parser;
use pyo3::prelude::PyAnyMethods;
use pyo3::Python;
use amfiteatr_examples::common::ComputeDevice;
use amfiteatr_examples::connect_four::common::ErrorRL;
use amfiteatr_examples::connect_four::env::ConnectFourRustEnvState;
use amfiteatr_examples::connect_four::env_nd::ConnectFourRustNdEnvState;
use amfiteatr_examples::connect_four::env_wrapped::PythonPettingZooStateWrap;
use amfiteatr_examples::connect_four::model::{build_ppo_policy, C4PPOPolicy, C4PPOPolicyMasking, ConnectFourModelRust};
use amfiteatr_examples::connect_four::no_protocol_model::ConnectFourModelRustNoProtocol;
use amfiteatr_examples::connect_four::options::{ConnectFourOptions, Implementation};
use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_rl::tch::Device;

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
    let agent_policy_0 = build_ppo_policy(&cli.layer_sizes_0[..], device, ppo_config, cli.learning_rate).unwrap();
    let agent_policy_1 = build_ppo_policy(&cli.layer_sizes_1[..], device, ppo_config, cli.learning_rate).unwrap();



    match cli.implementation{
        Implementation::Rust => {
            let mut model = ConnectFourModelRust::<ConnectFourRustEnvState, C4PPOPolicy>::new_ppo_generic(
                &cli,
                agent_policy_0,
                agent_policy_1,
                false
            );
            model.run_session(&cli)?;

        },
        Implementation::RustNd => {
            let mut model = ConnectFourModelRust::<ConnectFourRustNdEnvState, C4PPOPolicy>::new_ppo_generic(
                &cli,
                agent_policy_0,
                agent_policy_1,
                false
            );
            model.run_session(&cli).unwrap();

        },


        Implementation::Wrap => {
            let mut model = ConnectFourModelRust::<PythonPettingZooStateWrap, C4PPOPolicy>::new_ppo_generic(
                &cli,
                agent_policy_0,
                agent_policy_1,
                true
            );
            Python::with_gil(|py|{
                let pylogger = py.import("pettingzoo.utils.env_logger").unwrap();
                pylogger.getattr("EnvLogger").unwrap()
                   .call_method0("suppress_output").unwrap();

            });
            model.run_session(&cli).unwrap();


        },
        Implementation::RustNoProtocol => {
            let mut model = ConnectFourModelRustNoProtocol::<ConnectFourRustEnvState, C4PPOPolicy>::new_ppo_no_protocol(
                &cli
            );
            model.run_session(&cli)?;
        }



    }
    Ok(())


}
