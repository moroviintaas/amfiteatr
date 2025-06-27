use amfiteatr_examples::replicators::options::ReplicatorOptions;
use clap::Parser;
use amfiteatr_classic::domain::AgentNum;
use amfiteatr_examples::common::PolicySelect;
use amfiteatr_examples::replicators::error::ReplError;
use amfiteatr_examples::replicators::model::{ReplicatorModelBuilder, ReplicatorNetworkPolicy};
use amfiteatr_examples::replicators::policy_builder::{LearningPolicyBuilder, ReplPolicyBuilderPPO};
use amfiteatr_rl::tch::Device;
pub fn setup_logger(options: &ReplicatorOptions) -> Result<(), fern::InitError> {
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
        //.level(options.general_log_level)
        .level_for("amfiteatr_rl", options.rl_log_level)
        .level_for("replicator_dynamics_old", options.log_level)
        .level_for("amfiteatr_classic", options.classic_log_level)
        .level_for("amfiteatr_core", options.log_level_amfi);

    match &options.log_file{
        None => dispatch.chain(std::io::stdout()),
        Some(f) => dispatch.chain(fern::log_file(f)?)
    }

        .apply()?;
    Ok(())
}

fn create_and_run_model<PB: LearningPolicyBuilder>(options: &ReplicatorOptions, builders: Vec<(AgentNum, PB)>) -> Result<(), ReplError>{
    let mut model_builder = ReplicatorModelBuilder::new()
        .encounters_in_episode(options.number_of_rounds);

    for (l, b) in builders{
        model_builder.add_learning_agent(l,b.build()?)?;
    }
    let start = options.number_of_learning;
    for h in start..start + options.number_of_hawks{
        model_builder.add_hawk_agent(h as AgentNum)?;
    }
    let start = options.number_of_learning + options.number_of_hawks;
    for d in start..start + options.number_of_doves{
        model_builder.add_hawk_agent(d as AgentNum)?;
    }

    let mut model = model_builder.build()?;
    model.run_episode()?;
    Ok(())
}

fn main() -> Result<(), anyhow::Error>{

    let args = ReplicatorOptions::parse();
    setup_logger(&args).unwrap();
    //let device = Device::Cpu;
    match args.policy_algo{
        PolicySelect::PPO => {

            let policy_builders: Vec<_> = (0..args.number_of_learning as AgentNum)
                .map(|i|{(
                        i,
                        ReplPolicyBuilderPPO{
                            options: &args,
                            agent_id: i,
                        }
                    )

                }).collect();

            create_and_run_model(&args, policy_builders)?;

        }
        any_other => {
            todo!()
        }
    }



    Ok(())
}