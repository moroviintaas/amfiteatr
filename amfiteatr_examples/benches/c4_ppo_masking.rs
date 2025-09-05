use std::collections::HashMap;
use criterion::{criterion_group, criterion_main, AxisScale, Criterion, PlotConfiguration};
use amfiteatr_core::agent::{AutomaticAgent, ReseedAgent};
use amfiteatr_core::comm::StdEnvironmentEndpoint;
use amfiteatr_core::env::{HashMapEnvironment, ReseedEnvironment, RoundRobinPenalisingUniversalEnvironment};
use amfiteatr_examples::connect_four::agent::ConnectFourInfoSet;
use amfiteatr_examples::connect_four::common::{ConnectFourScheme, ConnectFourPlayer};
use amfiteatr_examples::connect_four::env::ConnectFourRustEnvState;
use amfiteatr_examples::connect_four::model::{build_ppo_policy_masking, Agent};
use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_rl::tch::Device;

type Environment<S> = HashMapEnvironment<ConnectFourScheme, S, StdEnvironmentEndpoint<ConnectFourScheme>>;

const LAYER_SIZES:[i64;5] = [64,256,1024i64,4096,16384];

const MAX_GAME_STEPS: usize = 128;
const MAX_GAMES: usize = 512;

fn benchmarked_run(net_size: &[i64]){
    let learning_rate = 1e-04;
    let (c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
    let (c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();

    let mut hm = HashMap::new();
    hm.insert(ConnectFourPlayer::One, c_env1);
    hm.insert(ConnectFourPlayer::Two, c_env2);

    let mut config_ppo = ConfigPPO::default();
    config_ppo.gae_lambda = Some(0.9);
    config_ppo.update_epochs = 4;
    config_ppo.mini_batch_size = 16;

    let mut env = Environment::new(ConnectFourRustEnvState::default(), hm, );
    let agent_policy_0 = build_ppo_policy_masking(net_size, Device::Cpu, config_ppo, learning_rate).unwrap();
    let agent_policy_1 = build_ppo_policy_masking(net_size, Device::Cpu, config_ppo, learning_rate).unwrap();

    let mut agent_0 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0);
    let mut agent_1 = Agent::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1);

    let mut remaining_steps = MAX_GAME_STEPS;

    for _g in 0..MAX_GAMES{
        env.reseed(()).unwrap();
        agent_0.reseed(()).unwrap();
        agent_1.reseed(()).unwrap();

        if remaining_steps > 0{
            std::thread::scope(|s|{
                s.spawn(||{
                    env.run_round_robin_with_rewards_penalise_truncating(
                        |_,_| -10.0, Some(remaining_steps)
                    ).unwrap();
                });
                s.spawn(||{
                    agent_0.run().unwrap()
                });
                s.spawn(||{
                    agent_1.run().unwrap()
                });
            })

        }
        else{
            break;
        }
        remaining_steps = remaining_steps.saturating_sub(env.completed_steps() as usize);
    }
}
fn benchmark_ppo_with_masking(c: &mut Criterion) {
    let plot_config = PlotConfiguration::default()
        .summary_scale(AxisScale::Linear);

    let mut group = c.benchmark_group("PPO on layer sizes");
    group.plot_config(plot_config);
    for ns in LAYER_SIZES.into_iter(){
        group.bench_function(format!("Layer size of {}", ns), |b| {
            b.iter(|| {
                benchmarked_run(&[ns]);
            })


        });
    }
    group.finish()
}


criterion_group!(benches, benchmark_ppo_with_masking);
criterion_main!(benches);
