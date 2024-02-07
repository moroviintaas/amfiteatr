
use std::collections::HashMap;
use std::path::PathBuf;
use std::thread;
use log::LevelFilter;
use amfiteatr_core::agent::{TracingAgentGen, IdAgent, AutomaticAgentRewarded, ReinitAgent, StatefulAgent, TracingAgent};
use amfiteatr_core::comm::StdEnvironmentEndpoint;
use amfiteatr_core::env::TracingHashMapEnvironment;
use amfiteatr_core::env::{ReinitEnvironment, RoundRobinUniversalEnvironment, TracingEnv};
use amfiteatr_core::error::AmfiError;
use amfiteatr_classic::agent::{Forgive1Policy, PrisonerInfoSet, RandomPrisonerPolicy, SwitchOnTwoSubsequent};
use amfiteatr_classic::domain::ClassicAction::Defect;
use amfiteatr_classic::domain::{ClassicAction, ClassicGameDomainNamed};
use amfiteatr_classic::domain::PrisonerId::{Alice, Bob};
use amfiteatr_classic::policy::ClassicPureStrategy;
use amfiteatr_classic::SymmetricRewardTableInt;
use amfiteatr_examples::classic::env::PrisonerEnvState;


pub fn setup_logger(log_level: LevelFilter, log_file: &Option<PathBuf>) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log_level);

        match log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        //.chain(std::io::stdout())
        //.chain(fern::log_file("output.log")?)
        .apply()?;
    Ok(())
}




fn main() -> Result<(), AmfiError<ClassicGameDomainNamed>>{
    println!("Hello prisoners;");
    setup_logger(LevelFilter::Debug, &None).unwrap();

    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);



    let env_state = PrisonerEnvState::new(reward_table,  10);

    let (comm_env_0, comm_prisoner_0) = StdEnvironmentEndpoint::new_pair();
    let (comm_env_1, comm_prisoner_1) = StdEnvironmentEndpoint::new_pair();

    let mut prisoner0 = TracingAgentGen::new(
        PrisonerInfoSet::new(Alice, reward_table.clone()), comm_prisoner_0, ClassicPureStrategy::new(ClassicAction::Cooperate));

    let mut prisoner1 = TracingAgentGen::new(
        PrisonerInfoSet::new(Bob, reward_table.clone()), comm_prisoner_1, Forgive1Policy{});

    let mut env_coms = HashMap::new();
    env_coms.insert(Alice, comm_env_0);
    env_coms.insert(Bob, comm_env_1);

    let mut env = TracingHashMapEnvironment::new(env_state, env_coms);

    thread::scope(|s|{
        s.spawn(||{
            env.run_round_robin_with_rewards().unwrap();
        });
        s.spawn(||{
            prisoner0.run_rewarded().unwrap();
        });
        s.spawn(||{
            prisoner1.run_rewarded().unwrap();
        });
    });

    println!("Scenario 2");




    env.reinit(PrisonerEnvState::new(reward_table.clone(), 10));
    let mut prisoner0 = prisoner0.transform_replace_policy(RandomPrisonerPolicy{});
    //let mut prisoner1 = prisoner1.do_change_policy(BetrayRatioPolicy{});
    let mut prisoner1 = prisoner1.transform_replace_policy(SwitchOnTwoSubsequent{});
    prisoner0.reinit(PrisonerInfoSet::new(*prisoner0.id(), reward_table.clone()));
    prisoner1.reinit(PrisonerInfoSet::new(*prisoner1.id(), reward_table.clone()));

    thread::scope(|s|{
        s.spawn(||{
            env.run_round_robin_with_rewards().unwrap();
        });
        s.spawn(||{
            prisoner0.run_rewarded().unwrap();
        });
        s.spawn(||{
            prisoner1.run_rewarded().unwrap();
        });
    });

    let prisoner0_betrayals = prisoner0.info_set().count_actions(Defect);
    let prisoner1_betrayals = prisoner1.info_set().count_actions(Defect);

    println!("Prisoner 0 betrayed {:?} times and Prisoner 1 betrayed {:?} times.", prisoner0_betrayals, prisoner1_betrayals);

    for elem in env.trajectory().list(){
        println!("{}", elem);
    }

    for trace in prisoner1.game_trajectory().list(){
        println!("{}", trace);
    }



    Ok(())
}


