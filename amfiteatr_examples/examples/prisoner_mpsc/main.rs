use std::thread;
use amfiteatr_classic::agent::{LocalHistoryInfoSet};
use amfiteatr_classic::domain::ClassicAction::Down;
use amfiteatr_classic::domain::TwoPlayersStdName::{Alice, Bob};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_classic::SymmetricRewardTableInt;
use amfiteatr_core::agent::{AgentGen, AutomaticAgent, StatefulAgent, TracingAgent, TracingAgentGen};
use amfiteatr_core::comm::EnvironmentMpscPort;
use amfiteatr_core::env::{AutoEnvironmentWithScores, StatefulEnvironment, TracingBasicEnvironment, TracingEnvironment};

fn main() {
    let number_of_players = 2;
    let mut env_adapter = EnvironmentMpscPort::new();



    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);



    let alice_comm = env_adapter.register_agent(Alice).unwrap();
    let alice_policy = ClassicMixedStrategy::new(0.7);
    let alice_state = LocalHistoryInfoSet::new(Alice, reward_table.into());
    let mut alice = TracingAgentGen::new(alice_state, alice_comm, alice_policy);

    let comm_bob = env_adapter.register_agent(Bob).unwrap();
    let bob_policy = ClassicPureStrategy::new(Down);
    let bob_state = LocalHistoryInfoSet::new(Bob, reward_table.into());
    let mut bob = AgentGen::new(bob_state, comm_bob, bob_policy);

    let env_state = PairingState::new_even(number_of_players, 1, reward_table.into()).unwrap();
    let mut environment = TracingBasicEnvironment::new(env_state, env_adapter);

    thread::scope(|s|{
        s.spawn(||{
            environment.run_with_scores().unwrap();
        });
        s.spawn(||{
            alice.run().unwrap();
        });
        s.spawn(||{
            bob.run().unwrap();
        });
    });

    println!("Final state: {}", environment.state());
    println!("Trajectory of environment: {:?}", environment.trajectory());

    println!("Alice final information set: {}", alice.info_set());
    println!("Trajectory of Alice: {:?}", alice.game_trajectory());

}