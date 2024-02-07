/*
use std::collections::HashMap;
use std::thread;
use amfiteatr_classic::agent::{LocalHistoryInfoSet, MinimalInfoSet};
use amfiteatr_classic::domain::ClassicAction::Down;
use amfiteatr_classic::domain::TwoPlayersStdName::{Alice, Bob};
use amfiteatr_classic::env::PairingState;
use amfiteatr_classic::policy::{ClassicMixedStrategy, ClassicPureStrategy};
use amfiteatr_classic::SymmetricRewardTableInt;
use amfiteatr_core::agent::{AgentGen, AutomaticAgentRewarded, StatefulAgent, TracingAgent, TracingAgentGen};
use amfiteatr_core::comm::{DynEndpoint, EnvironmentMpscPort, StdEnvironmentEndpoint};
use amfiteatr_core::env::{AutoEnvironmentWithScores, RoundRobinUniversalEnvironment, StatefulEnvironment, TracingEnv, TracingEnvironment, TracingHashMapEnvironment};
use amfiteatr_net_ext::tcp::{TcpComm32, TcpComm512};

fn main() {
    let number_of_players = 2;

    //alice connected locally
    let (env_alice_comm, alice_comm) = StdEnvironmentEndpoint::new_pair();
    let env_alice_comm = DynEndpoint::Std(env_alice_comm);

    let (t_bob_stream, r_bob_stream) = std::sync::mpsc::channel();
    //create listener for
    let tcp_listener = std::net::TcpListener::bind("127.0.0.1:8420").unwrap();
    let wait_connection = thread::spawn(move ||{
        let(stream_listen_bob, _) = tcp_listener.accept().unwrap();
        t_bob_stream.send(stream_listen_bob).unwrap();
    });
    let stream_bob_env = std::net::TcpStream::connect("127.0.0.1:8420").unwrap();
    let stram_env_bob = r_bob_stream.recv().unwrap();
    wait_connection.join().unwrap();

    let bob_comm = TcpComm512::new(stream_bob_env);


    let env_bob_comm = DynEndpoint::Dynamic(Box::new(TcpComm512::new(stram_env_bob)));



    let reward_table = SymmetricRewardTableInt::new(5, 1, 10, 3);



    //let alice_comm = env_adapter.register_agent(Alice).unwrap();
    let alice_policy = ClassicMixedStrategy::new(0.7);
    let alice_state = LocalHistoryInfoSet::new(Alice, reward_table.into());
    let mut alice = TracingAgentGen::new(alice_state, alice_comm, alice_policy);


    let bob_policy = ClassicPureStrategy::new(Down);
    let bob_state = LocalHistoryInfoSet::new(Bob, reward_table.into());
    let mut bob = AgentGen::new(bob_state, bob_comm, bob_policy);

    let env_state = PairingState::new_even(number_of_players, 1, reward_table.into()).unwrap();
    // Environment will try receive on any endpoints switching listening in round robin manner
    let mut env_endpoints = HashMap::new();
    env_endpoints.insert(Alice, env_alice_comm);
    env_endpoints.insert(Bob, env_bob_comm);
    let mut environment= TracingHashMapEnvironment::new(
        env_state, env_endpoints);

    thread::scope(|s|{
        s.spawn(||{
            environment.run_round_robin_with_rewards().unwrap();
        });
        s.spawn(||{
            alice.run_rewarded().unwrap();
        });
        s.spawn(||{
            bob.run_rewarded().unwrap();
        });
    });

    println!("Final state: {}", environment.state());
    println!("Trajectory of environment: {:?}", environment.trajectory());

    println!("Alice final information set: {}", alice.info_set());
    println!("Trajectory of Alice: {:?}", alice.game_trajectory());

}
*/


fn main() {}