use std::collections::HashMap;
use std::fs::File;
use log::{info, warn};
use amfiteatr_core::agent::{ActingAgent, MultiEpisodeAutoAgent, PolicyAgent, ReseedAgent, RewardedAgent, StatefulAgent, TracingAgentGen};
use amfiteatr_core::comm::{StdAgentEndpoint, StdEnvironmentEndpoint};
use amfiteatr_core::domain::{DomainParameters, Renew};
use amfiteatr_core::env::GameStateWithPayoffs;
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::{ConfigA2C, ConfigPPO, LearnSummary, LearningNetworkPolicyGeneric};
use crate::connect_four::agent::ConnectFourInfoSet;
use crate::connect_four::common::{ConnectFourDomain, ConnectFourPlayer, ErrorRL};
use crate::connect_four::model::{build_a2c_policy, build_a2c_policy_masking, build_ppo_policy, build_ppo_policy_masking, C4A2CPolicy, C4A2CPolicyMasking, C4PPOPolicy, C4PPOPolicyMasking, ConnectFourModelRust, EpochSummary};
use crate::connect_four::options::ConnectFourOptions;
use std::io::Write;
use amfiteatr_rl::tch::Device;
use crate::common::ComputeDevice;

pub struct ConnectFourModelRustNoProtocol <
    S: GameStateWithPayoffs<ConnectFourDomain>,
    P: LearningNetworkPolicyGeneric<ConnectFourDomain, Summary=LearnSummary>
> {
    env_state: S,
    agents: HashMap<<ConnectFourDomain as DomainParameters>::AgentId, TracingAgentGen<ConnectFourDomain, P, StdAgentEndpoint<ConnectFourDomain>>>,
    shared_policy: bool,
    tboard_writer: Option<tboard::EventWriter<File>>,
}

impl<
    S:  Default + GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRustNoProtocol<S, C4PPOPolicy>
{
    pub fn new_ppo_no_protocol(
        options: &ConnectFourOptions,
        //mut agent_0_policy: P,
        //mut agent_1_policy: P,
        //shared_policy: bool
    ) -> Self{
        let config_ppo = ConfigPPO {
            gae_lambda: options.gae_lambda,
            update_epochs: options.ppo_update_epochs,
            mini_batch_size: options.mini_batch_size,
            ..Default::default()
        };
        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (_c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (_c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();
        let mut agent_policy_0 = build_ppo_policy(&options.layer_sizes_0[..], device, config_ppo, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_ppo_policy(&options.layer_sizes_1[..], device, config_ppo, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }

        let env_state = S::default();
        let mut agents = HashMap::new();
        agents.insert(
            ConnectFourPlayer::One,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0)
        );
        agents.insert(
            ConnectFourPlayer::Two,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1)
        );

        Self{
            env_state,
            agents,
            shared_policy: false,
            tboard_writer: None
        }
    }
}

impl<
    S:  Default + GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRustNoProtocol<S, C4PPOPolicyMasking>
{
    pub fn new_ppo_masking_no_protocol(
        options: &ConnectFourOptions,
        //mut agent_0_policy: P,
        //mut agent_1_policy: P,
        //shared_policy: bool
    ) -> Self{
        let config_ppo = ConfigPPO {
            gae_lambda: options.gae_lambda,
            update_epochs: options.ppo_update_epochs,
            mini_batch_size: options.mini_batch_size,
            ..Default::default()
        };
        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (_c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (_c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();
        let mut agent_policy_0 = build_ppo_policy_masking(&options.layer_sizes_0[..], device, config_ppo, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_ppo_policy_masking(&options.layer_sizes_1[..], device, config_ppo, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }

        let env_state = S::default();
        let mut agents = HashMap::new();
        agents.insert(
            ConnectFourPlayer::One,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0)
        );
        agents.insert(
            ConnectFourPlayer::Two,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1)
        );

        Self{
            env_state,
            agents,
            shared_policy: false,
            tboard_writer: None
        }
    }
}

impl<
    S:  Default + GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRustNoProtocol<S, C4A2CPolicy>
{
    pub fn new_a2c_no_protocol(
        options: &ConnectFourOptions,
        //mut agent_0_policy: P,
        //mut agent_1_policy: P,
        //shared_policy: bool
    ) -> Self{
        let config_a2c = ConfigA2C { gae_lambda: options.gae_lambda, ..Default::default() };
        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (_c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (_c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();
        let mut agent_policy_0 = build_a2c_policy(&options.layer_sizes_0[..], device, config_a2c, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_a2c_policy(&options.layer_sizes_1[..], device, config_a2c, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }

        let env_state = S::default();
        let mut agents = HashMap::new();
        agents.insert(
            ConnectFourPlayer::One,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0)
        );
        agents.insert(
            ConnectFourPlayer::Two,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1)
        );

        Self{
            env_state,
            agents,
            shared_policy: false,
            tboard_writer: None
        }
    }
}

impl<
    S:  Default + GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
> ConnectFourModelRustNoProtocol<S, C4A2CPolicyMasking>
{
    pub fn new_a2c_masking_no_protocol(
        options: &ConnectFourOptions,
        //mut agent_0_policy: P,
        //mut agent_1_policy: P,
        //shared_policy: bool
    ) -> Self{
        let config_a2c = ConfigA2C { gae_lambda: options.gae_lambda, ..Default::default() };
        let device = match options.device{
            ComputeDevice::Cpu => Device::Cpu,
            ComputeDevice::Cuda => Device::Cuda(0),
        };
        let (_c_env1, c_a1) = StdEnvironmentEndpoint::new_pair();
        let (_c_env2, c_a2) = StdEnvironmentEndpoint::new_pair();
        let mut agent_policy_0 = build_a2c_policy_masking(&options.layer_sizes_0[..], device, config_a2c, options.learning_rate).unwrap();
        let mut agent_policy_1 = build_a2c_policy_masking(&options.layer_sizes_1[..], device, config_a2c, options.learning_rate).unwrap();
        if let Some(t0) = &options.tboard_agent0{
            agent_policy_0.add_tboard_directory(t0).unwrap()
        }
        if let Some(t1) = &options.tboard_agent1{
            agent_policy_1.add_tboard_directory(t1).unwrap()
        }

        let env_state = S::default();
        let mut agents = HashMap::new();
        agents.insert(
            ConnectFourPlayer::One,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::One), c_a1, agent_policy_0)
        );
        agents.insert(
            ConnectFourPlayer::Two,
            TracingAgentGen::new(ConnectFourInfoSet::new(ConnectFourPlayer::Two), c_a2, agent_policy_1)
        );

        Self{
            env_state,
            agents,
            shared_policy: false,
            tboard_writer: None
        }
    }
}

impl<
    S:  Default + GameStateWithPayoffs<ConnectFourDomain> + Clone + Renew<ConnectFourDomain, ()>,
    P: LearningNetworkPolicyGeneric<ConnectFourDomain, InfoSetType=ConnectFourInfoSet, Summary=LearnSummary> + TensorboardSupport<ConnectFourDomain>
> ConnectFourModelRustNoProtocol<S, P>
{




    pub fn play_one_game(&mut self, store_episode: bool, truncate_at_step: Option<usize>) -> Result<EpochSummary, AmfiteatrRlError<ConnectFourDomain>>{
        self.agents.get_mut(&ConnectFourPlayer::One).unwrap().reseed(())?;
        self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().reseed(())?;
        self.env_state.renew_from(())?;

        let mut agent_commited_payoffs = HashMap::new();
        agent_commited_payoffs.insert(ConnectFourPlayer::One, 0.0);
        agent_commited_payoffs.insert(ConnectFourPlayer::Two, 0.0);

        let mut current_step = 0;
        //let mut current_player_id = self.env_state.current_player();
        let mut summary = EpochSummary::default();

        while let Some(current_player) = self.env_state.current_player(){

            let step_reward = self.env_state.state_payoff_of_player(&current_player) - agent_commited_payoffs[&current_player];
            agent_commited_payoffs.insert(current_player,  self.env_state.state_payoff_of_player(&current_player));
            let agent = self.agents.get_mut(&current_player).unwrap();
            agent.current_universal_reward_add(&step_reward);

            let action = agent.select_action()?;
            current_step += 1;

            let r_updates = self.env_state.forward(current_player, action);

            match r_updates{
                Ok(updates) => {
                    for (update_agent_id, update) in updates.into_iter(){
                        self.agents.get_mut(&update_agent_id).unwrap().update(update).map_err(|e|{
                            AmfiteatrRlError::Amfiteatr { source: AmfiteatrError::Game {source: e} }
                        })?;
                    }
                }
                Err(game_error) =>  {
                    summary.invalid_actions[current_player.index()];
                    info!("Game error {game_error}");
                    break;
                }
            }
        }

        for (agent_id, agent) in self.agents.iter_mut(){
            let step_reward = self.env_state.state_payoff_of_player(&agent_id) - agent_commited_payoffs[&agent_id];
            agent.current_universal_reward_add(&step_reward);
            agent.finalize()?;
            agent.store_episode()?;
        }
        summary.games_played = 1.0;
        summary.game_steps = current_step as f64;
        summary.scores = [
            self.env_state.state_payoff_of_player(&ConnectFourPlayer::One) as f64,
            self.env_state.state_payoff_of_player(&ConnectFourPlayer::Two) as f64
        ];

        Ok(summary)


    }

    pub fn play_epoch(
        &mut self,
        number_of_games: usize,
        summarize: bool,
        training_epoch: bool,
        max_steps: Option<usize>
    ) -> Result<EpochSummary, AmfiteatrRlError<ConnectFourDomain>>
    {
        let mut steps_left = max_steps;
        let mut number_of_games_played = 0;

        for agent in self.agents.values_mut() {
            agent.clear_episodes()?;
        }

        let mut vec = match summarize{
            true => Vec::with_capacity(number_of_games),
            false => Vec::with_capacity(0),
        };

        for i in 0..number_of_games{
            let summary = self.play_one_game(training_epoch, steps_left)?;

            if summarize{
                vec.push(summary);
            }
            number_of_games_played = i;
            if let Some(step_pool) = steps_left{

                let remaining_steps = step_pool.saturating_sub(summary.game_steps as usize);
                steps_left = Some(remaining_steps);
                log::debug!("Remaining {} steps for epoch", step_pool);
                if remaining_steps == 0{
                    break;
                }
            }


        }
        if summarize{
            let summary_sum: EpochSummary = vec.iter().fold(EpochSummary::default(), |acc, x| &acc+x);
            let n = number_of_games_played as f64;
            return Ok(EpochSummary {
                games_played: summary_sum.games_played,
                scores: [summary_sum.scores[0]/n, summary_sum.scores[1]/n],
                invalid_actions: [summary_sum.invalid_actions[0]/n, summary_sum.invalid_actions[1]/n],
                game_steps: summary_sum.game_steps / n,
            });
        }
        Ok(EpochSummary::default())
    }

    pub fn train_agents_on_experience(&mut self) -> Result<(LearnSummary,LearnSummary), ErrorRL>{

        let t1 = self.agents.get_mut(&ConnectFourPlayer::One).unwrap().take_episodes();
        let s1 = self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().train(&t1)?;
        let t2 = self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().take_episodes();

        let s2 = self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().train(&t2)?;

        Ok((s1, s2))
    }

    pub fn train_agent1_only(&mut self) -> Result<LearnSummary, ErrorRL>{
        let t1 = self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().take_episodes();
        let s1 = self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().train(&t1)?;
        //let _t2 = self.agent0.take_episodes();
        //self.agent2.policy_mut().train_on_trajectories_env_reward(&t2)?;

        Ok(s1)
    }

    pub fn train_agent0_on_both_experiences(&mut self) -> Result<(LearnSummary,LearnSummary), ErrorRL>{
        let mut t1 = self.agents.get_mut(&ConnectFourPlayer::One).unwrap().take_episodes();
        let mut t2 = self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().take_episodes();
        //let chain = t1.iter().chain(t2);
        t1.append(&mut t2);
        let s1 = self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().train(&t1)?;

        //self.agent2.policy_mut().train_on_trajectories_env_reward(&t2)?;

        Ok((s1.clone(), s1))
    }

    pub fn run_session(&mut self, options: &ConnectFourOptions)
                       -> Result<(), ErrorRL>{
        info!("Starting session");
        let pre_training_summary = self.play_epoch(options.num_test_episodes, true, false, options.limit_steps_in_epochs)?;
        info!("Summary before training: {}", pre_training_summary.describe_as_collected());

        let mut results_learn = Vec::with_capacity(options.epochs);
        let mut results_test = Vec::with_capacity(options.epochs);
        let mut l_summaries_1 = Vec::with_capacity(options.epochs);
        let mut l_summaries_2 = Vec::with_capacity(options.epochs);

        for e in 0..options.epochs{
            let s = self.play_epoch(options.num_episodes, true, true, options.limit_steps_in_epochs)?;

            self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().t_write_scalar(e as i64, "train_epoch/score", s.scores[0] as f32)?;
            self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().t_write_scalar( e as i64, "train_epoch/illegal_moves", s.invalid_actions[0] as f32)?;

            self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().t_write_scalar(e as i64, "train_epoch/score", s.scores[1] as f32)?;
            self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().t_write_scalar(e as i64, "train_epoch/illegal_moves", s.invalid_actions[1] as f32)?;

            if let Some(ref mut tboard) = self.tboard_writer{
                tboard.write_scalar(e as i64, "train_epoch/number_of_games", s.games_played as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (train)".into(), error: format!("{e}")})?;
                tboard.write_scalar(e as i64, "train_epoch/number_of_steps_in_game", s.game_steps as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (train)".into(), error: format!("{e}")})?;

            }
            results_learn.push(s);

            //let (s1,s2) = self.train_agents_on_experience()?;
            let (s1,s2) = match self.shared_policy{
                false => self.train_agents_on_experience(),
                true => self.train_agent0_on_both_experiences()
            }?;


            info!("Summary of games in epoch {}: {}", e+1, s.describe_as_collected());
            info!("Training epoch {}: Critic losses {:.3}, {:.3}; Actor loss {:.3}, {:.3}; entropy loss {:.3}, {:.3}",
                e+1, s1.value_loss.unwrap(), s2.value_loss.unwrap(),
                s1.policy_gradient_loss.unwrap(), s2.policy_gradient_loss.unwrap(),
                s1.entropy_loss.unwrap(), s2.entropy_loss.unwrap()
            );
            if options.num_test_episodes > 0{
                let s =self.play_epoch(options.num_test_episodes, true, false, options.limit_steps_in_epochs)?;
                results_test.push(s);
                l_summaries_1.push(s1);
                l_summaries_2.push(s2);



                info!("Summary of tests after epoch {}: {}", e+1, s.describe_as_collected());
                if let Some(ref mut tboard) = self.tboard_writer{
                    tboard.write_scalar(e as i64, "test_epoch/number_of_games", s.games_played as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (test)".into(), error: format!("{e}")})?;
                    tboard.write_scalar(e as i64, "test_epoch/number_of_steps_in_game", s.game_steps as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (test)".into(), error: format!("{e}")})?;

                }

            }

        }
        if let Some(result_file) = &options.save_path_train_param_summary{
            let yaml = serde_yaml::to_string(&(l_summaries_1, l_summaries_2)).unwrap();
            let mut file = File::create(result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        if let Some(result_file) = &options.save_path_test_epoch{
            let yaml = serde_yaml::to_string(&results_test).unwrap();
            let mut file = File::create(result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        if let Some(result_file) = &options.save_path_train_epoch{
            let yaml = serde_yaml::to_string(&results_learn).unwrap();
            let mut file = File::create(result_file).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
            write!(file, "{}", &yaml).map_err(|e| AmfiteatrError::IO { explanation: format!("{e}") })?;
        }

        for e in 0..options.extended_epochs{
            let s = self.play_epoch(options.num_episodes, true, true, options.limit_steps_in_epochs)?;

            self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().t_write_scalar((options.epochs + e) as i64, "train_epoch/score", s.scores[0] as f32)?;
            self.agents.get_mut(&ConnectFourPlayer::One).unwrap().policy_mut().t_write_scalar((options.epochs + e) as i64, "train_epoch/illegal_moves", s.invalid_actions[0] as f32)?;

            self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().t_write_scalar((options.epochs + e) as i64, "train_epoch/score", s.scores[1] as f32)?;
            self.agents.get_mut(&ConnectFourPlayer::Two).unwrap().policy_mut().t_write_scalar((options.epochs + e) as i64, "train_epoch/illegal_moves", s.invalid_actions[1] as f32)?;

            if let Some(ref mut tboard) = self.tboard_writer{
                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/number_of_games", s.games_played as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (train)".into(), error: format!("{e}")})?;
                tboard.write_scalar((options.epochs + e) as i64, "train_epoch/number_of_steps_in_game", s.game_steps as f32)
                    .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (train)".into(), error: format!("{e}")})?;

            }
            let s1 = self.train_agent1_only()?;
            info!("Summary of games in extended epoch {}: {}", e+1, s.describe_as_collected());
            info!("Training only agent 1 epoch {}: Critic losses {:.3}; Actor loss {:.3}; entropy loss {:.3}",
                e+1, s1.value_loss.unwrap(),
                s1.policy_gradient_loss.unwrap(),
                s1.entropy_loss.unwrap(),
            );
            if options.num_test_episodes > 0 {
                let s =self.play_epoch(options.num_test_episodes, true, false, options.limit_steps_in_epochs)?;
                info!("Summary of tests after extended epoch {}: {}", e+1, s.describe_as_collected());

                if let Some(ref mut tboard) = self.tboard_writer{
                    tboard.write_scalar((options.epochs + e) as i64, "test_epoch/number_of_games", s.games_played as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving games in epoch (test)".into(), error: format!("{e}")})?;
                    tboard.write_scalar((options.epochs + e) as i64, "test_epoch/number_of_steps_in_game", s.game_steps as f32)
                        .map_err(|e| AmfiteatrError::TboardFlattened {context: "Saving average game steps in epoch (test)".into(), error: format!("{e}")})?;

                }

            }

        }


        Ok(())
    }

}