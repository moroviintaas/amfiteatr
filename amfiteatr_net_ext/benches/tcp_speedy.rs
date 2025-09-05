use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use amfiteatr_core::agent::{AgentGen, AutomaticAgent, RandomPolicy};
use amfiteatr_core::comm::{EnvironmentMpscPort, StdEnvironmentEndpoint};
use amfiteatr_core::demo::{DemoScheme, DemoInfoSet, DemoState};
use amfiteatr_core::env::{AutoEnvironmentWithScores, BasicEnvironment, HashMapEnvironment, RoundRobinUniversalEnvironment};
use amfiteatr_net_ext::tcp::{PairedTcpEnvironmentEndpoint};

pub fn bench_demo_game_tcp_speedy_hashmap(c: &mut Criterion){


    let player_nums = [2,4,6,8,10];

    let mut group = c.benchmark_group("Benchmark hashmap/tcp+speedy communication");

    for player_number_setup in player_nums{
        let parameter_string = format!("with {} players", player_number_setup);
        group.bench_function(BenchmarkId::new("tcp_speedy ", parameter_string),
        |b|{
            b.iter_batched(
                ||{
                    let bandits = vec![(3.0, 5.0), (10.0, 11.5), (3.0, 6.0), (2.0, 7.0)];
                    let number_of_bandits = bandits.len();

                    let mut agents = Vec::new();
                    let mut env_comms = HashMap::new();
                    let random_policy = RandomPolicy::<DemoScheme, DemoInfoSet>::new();
                    for id in 0..{

                        let (env_comm, agent_comm) = PairedTcpEnvironmentEndpoint::<DemoScheme, 512>::create_local_pair(11000+id as u16).unwrap();
                        env_comms.insert(id, env_comm);
                        let info_set = DemoInfoSet::new(id, number_of_bandits);

                        agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, agent_comm, random_policy.clone()))));


                    }
                    let player_set = (0..player_number_setup).collect();
                    let state = DemoState::new_with_players(bandits, 10, &player_set);
                    let env = HashMapEnvironment::new(state, env_comms);
                    (env, agents)
                },
            |routine_input|{
                    let mut env = routine_input.0;
                    let agents = routine_input.1;

                    std::thread::scope(|s|{
                        s.spawn(||
                            env.run_round_robin_with_rewards()
                        );

                        for agent in agents{
                            s.spawn(move ||{
                                let mut guard = agent.as_ref().lock().unwrap();
                                guard.run().unwrap();
                            });
                            //let mut guard = agent.as_ref().lock().unwrap();

                        }
                    });


                },
                criterion::BatchSize::PerIteration
            )
        });
    }

}

pub fn bench_demo_game_mpsc_hashmap(c: &mut Criterion){


    let player_nums = [2,4,6,8,10];

    let mut group = c.benchmark_group("Benchmark hashmap/mpsc communication");

    for player_number_setup in player_nums{
        let parameter_string = format!("with {} players", player_number_setup);
        group.bench_function(BenchmarkId::new("mpsc", parameter_string),
             |b|{
                 b.iter_batched(
                     ||{
                         let bandits = vec![(3.0, 5.0), (10.0, 11.5), (3.0, 6.0), (2.0, 7.0)];
                         let number_of_bandits = bandits.len();

                         let mut agents = Vec::new();
                         let mut env_comms = HashMap::new();
                         let random_policy = RandomPolicy::<DemoScheme, DemoInfoSet>::new();
                         for id in 0..player_number_setup{
                             let (comm_env_temp, comm_agent) = StdEnvironmentEndpoint::new_pair();

                             env_comms.insert(id, comm_env_temp);
                             let info_set = DemoInfoSet::new(id, number_of_bandits);

                             agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, comm_agent, random_policy.clone()))));


                         }
                         let player_set = (0..player_number_setup).collect();
                         let state = DemoState::new_with_players(bandits, 10, &player_set);
                         let env = HashMapEnvironment::new(state, env_comms);
                         (env, agents)
                     },
                     |routine_input|{
                         let mut env = routine_input.0;
                         let agents = routine_input.1;

                         std::thread::scope(|s|{
                             s.spawn(||
                                 env.run_round_robin_with_rewards()
                             );

                             for agent in agents{
                                 s.spawn(move ||{
                                     let mut guard = agent.as_ref().lock().unwrap();
                                     guard.run().unwrap();
                                 });
                                 //let mut guard = agent.as_ref().lock().unwrap();

                             }
                         });


                     },
                     criterion::BatchSize::SmallInput
                 )
             });
    }

}

pub fn bench_demo_game_single_mpsc(c: &mut Criterion){


    let player_nums = [2,4,6,8,10];

    let mut group = c.benchmark_group("Benchmark true mpsc communication");

    for player_number_setup in player_nums{
        let parameter_string = format!("with {} players", player_number_setup);
        group.bench_function(BenchmarkId::new("true mpsc", parameter_string),
             |b|{
                 b.iter_batched(
                     ||{
                         let bandits = vec![(3.0, 5.0), (10.0, 11.5), (3.0, 6.0), (2.0, 7.0)];
                         let number_of_bandits = bandits.len();

                         let mut agents = Vec::new();
                         let random_policy = RandomPolicy::<DemoScheme, DemoInfoSet>::new();
                         let mut env_adapter = EnvironmentMpscPort::new();

                         for id in 0..player_number_setup{
                             //let (comm_env_temp, comm_agent) = StdEnvironmentEndpoint::new_pair();
                             let comm_agent = env_adapter.register_agent(id).unwrap();
                             let info_set = DemoInfoSet::new(id, number_of_bandits);

                             agents.push(Arc::new(Mutex::new(AgentGen::new(info_set, comm_agent, random_policy.clone()))));


                         }
                         let player_set = (0..player_number_setup).collect();
                         let state = DemoState::new_with_players(bandits, 10, &player_set);
                         let env = BasicEnvironment::new(state, env_adapter);
                         (env, agents)
                     },
                     |routine_input|{
                         let mut env = routine_input.0;
                         let agents = routine_input.1;

                         std::thread::scope(|s|{
                             s.spawn(||
                                 env.run_with_scores()
                             );

                             for agent in agents{
                                 s.spawn(move ||{
                                     let mut guard = agent.as_ref().lock().unwrap();
                                     guard.run().unwrap();
                                 });
                                 //let mut guard = agent.as_ref().lock().unwrap();

                             }
                         });


                     },
                     criterion::BatchSize::SmallInput
                 )
             });
    }

}

criterion_group!(multi_user_tcp, bench_demo_game_tcp_speedy_hashmap);
criterion_group!(multi_user_mpsc, bench_demo_game_mpsc_hashmap, bench_demo_game_single_mpsc );
criterion_main!(multi_user_tcp);