use clap::Parser;
use pyo3::prelude::PyAnyMethods;
use pyo3::Python;
use amfiteatr_examples::connect_four::common::ErrorRL;
use amfiteatr_examples::connect_four::env::ConnectFourRustEnvState;
use amfiteatr_examples::connect_four::env_nd::ConnectFourRustNdEnvState;
use amfiteatr_examples::connect_four::env_wrapped::PythonPettingZooStateWrap;
use amfiteatr_examples::connect_four::model::{C4A2CPolicy, C4A2CPolicyMasking, ConnectFourModelRust};
use amfiteatr_examples::connect_four::no_protocol_model::ConnectFourModelRustNoProtocol;
use amfiteatr_examples::connect_four::options::{ConnectFourOptions, Implementation};


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
        .level_for("amfiteatr_rl", options.log_level_amfiteatr_rl)
        ;

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

    /*
    let device = match cli.device{
        ComputeDevice::Cpu => Device::Cpu,
        ComputeDevice::Cuda => Device::Cuda(0),
    };

     */

    match cli.implementation{
        Implementation::Rust => {
            let mut model = ConnectFourModelRust::<ConnectFourRustEnvState, C4A2CPolicy>::new_a2c(
                &cli
            );
            model.run_session(&cli)?;

        },
        Implementation::RustNd => {
            let mut model = ConnectFourModelRust::<ConnectFourRustNdEnvState, C4A2CPolicy>::new_a2c(
                &cli
            );
            model.run_session(&cli)?;

        },

        Implementation::Wrap => {
            let mut model = ConnectFourModelRust::<PythonPettingZooStateWrap, C4A2CPolicy>::new_a2c(
                &cli
            );
            Python::with_gil(|py|{
                let pylogger = py.import("pettingzoo.utils.env_logger").unwrap();
                pylogger.getattr("EnvLogger").unwrap()
                   .call_method0("suppress_output").unwrap();

            });
            model.run_session(&cli).unwrap();


        },
        Implementation::RustNoProtocol => {
            let mut model = ConnectFourModelRustNoProtocol::<ConnectFourRustEnvState, C4A2CPolicy>::new_a2c_no_protocol(
                &cli
            );
            model.run_session(&cli)?;
        }



    }
    Ok(())


}
