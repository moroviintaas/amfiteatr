use crate::cart_pole::model::CartPoleModelOptions;
use crate::cart_pole::env_wrapped::PythonGymnasiumWrapCartPole;
use std::ops::Div;
use std::ops::Add;
use std::fs::File;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use clap::Parser;
use log::{info, LevelFilter};
use tboard::EventWriter;
use tboard::tensorboard::WorkerShutdownMode::Default;
use amfiteatr_rl::policy::{ConfigPPO, LearnSummary};
use amfiteatr_core::agent::{AutomaticAgent, MultiEpisodeAutoAgent, PolicyAgent, ReseedAgent, TracingAgentGen};
use amfiteatr_core::comm::{AgentMpscAdapter, EnvironmentMpscPort};
use amfiteatr_core::env::{AutoEnvironment, AutoEnvironmentWithScores, BasicEnvironment, ReseedEnvironment, SequentialGameState, StatefulEnvironment};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::PolicyDiscretePPO;
use amfiteatr_rl::tch;
use amfiteatr_rl::tch::nn::{AdamW, VarStore};
use amfiteatr_rl::torch_net::{Layer, NeuralNetActorCritic, VariableStorage};
use crate::cart_pole::agent::CartPoleObservationEncoding;
use crate::cart_pole::common::{CartPoleActionEncoding, CartPoleObservation, CartPoleScheme, SINGLE_PLAYER_ID};
use crate::cart_pole::env::CartPoleEnvStateRust;
use amfiteatr_rl::tch::nn::OptimizerConfig;
use crate::connect_four::common::{ConnectFourPlayer, ConnectFourScheme, ErrorRL};
use amfiteatr_core::env::GameStateWithPayoffs;
use amfiteatr_rl::policy::LearningNetworkPolicyGeneric;
use amfiteatr_rl::torch_net::build_network_model_ac_discrete;
pub type CartPolePolicy = PolicyDiscretePPO<CartPoleScheme, CartPoleObservation, CartPoleObservationEncoding, CartPoleActionEncoding>;

#[derive(Clone, Debug, Default)]
pub struct EpochSummary {
    pub games_played: f64,
    pub score: f64,
    pub game_steps: f64,
}

impl EpochSummary{
    pub fn describe_as_collected(&self) -> String{
        format!("games played: {}, average game steps: {:.2} | average score: {:.2},",
                self.games_played, self.game_steps, self.score)
    }
}

impl Add<EpochSummary> for EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            score: self.score + rhs.score,

            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}
impl Add<&EpochSummary> for &EpochSummary {
    type Output = EpochSummary;

    fn add(self, rhs: &EpochSummary) -> Self::Output {
        EpochSummary {
            games_played: self.games_played + rhs.games_played,
            score: self.score + rhs.score,

            game_steps: self.game_steps + rhs.game_steps,
        }
    }
}


impl Div<f64> for EpochSummary {
    type Output = EpochSummary;

    fn div(self, rhs: f64) -> Self::Output {
        Self{
            games_played: self.games_played / rhs,
            score: self.score/ rhs,
            game_steps: self.game_steps/ rhs,
        }
    }
}



pub struct CartPoleModelPython{
    env: BasicEnvironment<CartPoleScheme, PythonGymnasiumWrapCartPole, EnvironmentMpscPort<CartPoleScheme>>,
    agent: TracingAgentGen<CartPoleScheme, CartPolePolicy, AgentMpscAdapter<CartPoleScheme>>,
    tboard_writer: Option<tboard::EventWriter<File>>,
}




impl CartPoleModelPython{
    pub fn new_simple(options: &CartPoleModelOptions) -> anyhow::Result<Self>{
        let initial_state = PythonGymnasiumWrapCartPole::new()?;

        let mut env_communicator = EnvironmentMpscPort::new();
        let obs = initial_state.first_observations().unwrap()[0].1.clone();
        let agent_comm = env_communicator.register_agent(SINGLE_PLAYER_ID)?;
        let env = BasicEnvironment::new(initial_state, env_communicator);
        let vs = VarStore::new(tch::Device::Cpu);
        let model = build_network_model_ac_discrete(vec![
            Layer::Linear(64),
            Layer::Tanh,
            Layer::Linear(64),
            Layer::Tanh,
        ],
                                                    vec![4], 2, &vs.root());

        let optimizer = AdamW::default().build(&vs, 0.0003)?;
        let network = NeuralNetActorCritic::new(VariableStorage::Owned(vs), model);

        let mut config_ppo = ConfigPPO::default();
        config_ppo.vf_coef = options.vf_coef;
        config_ppo.ent_coef = options.ent_coef;
        config_ppo.update_epochs = 10;
        config_ppo.mini_batch_size=64;
        let mut policy = CartPolePolicy::new(

            config_ppo,
            network,
            optimizer,
            CartPoleObservationEncoding {},
            CartPoleActionEncoding {}
        );

        if let Some(p) = &options.tboard_agent_writer_path{
            policy.add_tboard_directory(p)?;
        }

        let agent = TracingAgentGen::new(obs, agent_comm, policy);

        let tboard_writer = match &options.tboard_agent_writer_path {
            Some(p) => Some(EventWriter::create(p)?),
            None => None
        };

        Ok(Self{
            env, agent, tboard_writer
        })
    }

    pub fn play_one_game(&mut self, store_episode: bool, truncate_at_step: Option<usize>) -> Result<EpochSummary, AmfiteatrRlError<CartPoleScheme>>{
        let mut summary = EpochSummary::default();
        self.env.reseed(())?;
        self.agent.reseed(())?;

        std::thread::scope(|s|{
            s.spawn(|| {
                let r = self.env.run_with_scores_truncating( truncate_at_step);
                r.inspect_err(|e|{
                    log::error!("Error {e}")
                })

            });
            s.spawn(||{
                self.agent.run().unwrap()
            });
        });

        summary.score = self.env.state().state_payoff_of_player(&SINGLE_PLAYER_ID) as f64;
        summary.games_played = 1.0;
        summary.game_steps = self.env.completed_steps() as f64;

        if store_episode{
            self.agent.store_episode()?;
        }

        Ok(summary)
    }

    pub fn play_epoch(
        &mut self,
        number_of_games: usize,
        summarize: bool,
        training_epoch: bool,
        max_steps: Option<usize>
    ) -> Result<EpochSummary, AmfiteatrRlError<CartPoleScheme>>{

        let mut steps_left = max_steps;
        let mut number_of_games_played = 0;

        self.agent.clear_episodes()?;

        let mut summaries = match summarize{
            true => Vec::with_capacity(number_of_games),
            false => vec![]
        };

        for i in 0..number_of_games{
            let summary = self.play_one_game(training_epoch, steps_left)?;
            if summarize{
                summaries.push(summary);
            }

            number_of_games_played += 1;

            if let Some(step_pool) = steps_left{
                let remaining_steps = step_pool.saturating_sub(self.env.completed_steps() as usize);
                steps_left = Some(remaining_steps);
                log::debug!("Remaining {} steps for epoch", step_pool);
                if remaining_steps == 0{
                    break;
                }
            }
        }

        if summarize{
            let summary_sum: EpochSummary = summaries.iter().fold(EpochSummary::default(), |acc, x| &acc+x);
            let n = number_of_games_played as f64;
            log::debug!("Epoch sum score: {}, games played: {}, number of games: {}", summary_sum.score, summary_sum.games_played, number_of_games_played);
            Ok(EpochSummary {
                games_played: summary_sum.games_played,
                score: summary_sum.score/n,
                game_steps: summary_sum.game_steps / n,
            })
        } else {
            Ok(EpochSummary::default())
        }


    }

    pub fn train_agents_on_experience(&mut self) -> Result<LearnSummary, AmfiteatrRlError<CartPoleScheme>>{
        let t = self.agent.take_episodes();

        //info!("Episodes: {}: [{:?}]", t.len(), t.iter().map(|x| x.number_of_steps()).collect::<Vec<usize>>());
        /*
        print!("Episodes: {}", t.len());
        for i in &t{
            print!("{} ", i.number_of_steps());
        }
        println!();
        */
        let s = self.agent.policy_mut().train(&t[..])?;
        Ok(s)
    }

    pub fn run_session(&mut self, options: CartPoleModelOptions) -> Result<(), AmfiteatrRlError<CartPoleScheme>>{
        info!("Starting session");
        let pre_training_summary = self.play_epoch(options.test_games, true, false, options.limit_steps_in_epochs)?;
        info!("Summary before training: {}", pre_training_summary.describe_as_collected());

        let mut results_learn = Vec::with_capacity(options.epochs);
        let mut results_test = Vec::with_capacity(options.epochs);
        let mut l_summaries = Vec::with_capacity(options.epochs);

        for e in 0..options.epochs{
            let s = self.play_epoch(options.num_episodes, true, true, options.limit_steps_in_epochs)?;
            self.agent.policy_mut().t_write_scalar(e as i64, "train_epoch/score", s.score as f32)?;

            if let Some(ref mut tboard) = self.tboard_writer{
                tboard.write_scalar(e as i64, "train_epoch/number_of_games", s.games_played as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (train)".into(), error: format!("{e}")})?;
                tboard.write_scalar(e as i64, "train_epoch/number_of_steps_in_game", s.game_steps as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (train)".into(), error: format!("{e}")})?;

            }

            log::debug!("{s:?}");
            results_learn.push(s.clone());

            let st = self.train_agents_on_experience()?;

            info!("Summary of games in epoch {}: {}", e+1, s.describe_as_collected());

            info!("Training epoch {}: Critic loss {:.3}; Actor loss {:.3}; entropy loss {:.3}",
                e+1, st.value_loss.unwrap(),
                st.policy_gradient_loss.unwrap(),
                st.entropy_loss.unwrap(),
            );

            if options.test_games > 0{
                let s =self.play_epoch(options.test_games, true, false, options.limit_steps_in_epochs)?;
                results_test.push(s.clone());
                l_summaries.push(s.clone());



                info!("Summary of tests after epoch {}: {}", e+1, s.describe_as_collected());
                if let Some(ref mut tboard) = self.tboard_writer{
                    tboard.write_scalar(e as i64, "test_epoch/number_of_games", s.games_played as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (test)".into(), error: format!("{e}")})?;
                    tboard.write_scalar(e as i64, "test_epoch/number_of_steps_in_game", s.game_steps as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (test)".into(), error: format!("{e}")})?;

                }

            }

        }

        Ok(())


    }
}