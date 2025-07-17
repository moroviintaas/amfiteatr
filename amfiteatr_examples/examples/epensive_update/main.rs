use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use clap::Parser;
use log::info;
use amfiteatr_core::agent::{AgentGen, MultiEpisodeAutoAgent, RandomPolicy};
use amfiteatr_core::comm::StdEnvironmentEndpoint;
use amfiteatr_core::env::{AutoEnvironmentWithScores, HashMapEnvironment, ReseedEnvironment};
use amfiteatr_examples::expensive_update::agent::ExpensiveUpdateInformationSet;
use amfiteatr_examples::expensive_update::env::ExpensiveUpdateState;
use crate::options::ExpensiveUpdateOptions;

mod options;

pub fn setup_logger(options: &ExpensiveUpdateOptions) -> Result<(), fern::InitError> {
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
        //.level_for("amfiteatr_rl", options.rl_log_level)
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

fn main() -> anyhow::Result<()> {

    let options = ExpensiveUpdateOptions::parse();
    setup_logger(&options)?;
    let mut comm_map = HashMap::new();

    let mut agents = Vec::new();

    for i in 0..options.agents{
        let (env_comm, agent_comm) = StdEnvironmentEndpoint::new_pair();
        comm_map.insert(i, env_comm);
        let agent = Arc::new(Mutex::new(AgentGen::new(ExpensiveUpdateInformationSet::new(i), agent_comm, RandomPolicy::new())));
        //agents.push(env_comm);
        agents.push(agent);

    }

    let environment_state = ExpensiveUpdateState::new(
        options.number_of_rounds,
        options.agents,
        options.small_update_cost_per_agent,
        options.big_update_cost_per_agent
    );
    let mut environment = HashMapEnvironment::new(environment_state, comm_map);

    for g in 0..options.games{
        std::thread::scope(|scope| {
            scope.spawn(||{
                environment.reseed(()).unwrap();
                environment.run_with_scores().unwrap();
            });

            for a in agents.iter(){
                let agent = a.clone();
                scope.spawn(move ||{
                    let mut guard = agent.lock().unwrap();
                    guard.run_episode(()).unwrap();
                });
            }
        });
        info!("Finished game {g}");

    }

    Ok(())


}

