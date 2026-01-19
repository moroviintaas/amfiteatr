//mod env;
//mod common;
//mod agent;
use clap::Parser;
use amfiteatr_examples::cart_pole;


pub fn setup_logger(options: &cart_pole::model::CartPoleModelOptions) -> Result<(), fern::InitError> {
    println!("Options: {:?}", options);
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
        .level_for("amfiteatr_examples", options.log_level)
        .level_for("amfiteatr_core", options.log_level_amfiteatr)
        .level_for("amfiteatr_rl", options.log_level_amfiteatr_rl)
        //.level(options.log_level)

    ;
        //.level_for("amfiteatr_core", options.log_level_amfi);

    match &options.log_file{
        None => dispatch.chain(std::io::stdout()),
        Some(f) => dispatch.chain(fern::log_file(f)?)
    }

        .apply()?;
    Ok(())
}
fn main() -> anyhow::Result<()> {

    let options = cart_pole::model::CartPoleModelOptions::parse();
    setup_logger(&options)?;

    let mut model = cart_pole::model::CartPoleModelRust::new_simple(&options)?;

    let summary = model.run_session(options)?;
    println!("summary:\n{:?}", summary);


    Ok(())

}