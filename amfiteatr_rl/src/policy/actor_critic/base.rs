use std::cmp::min;
use getset::{Getters, Setters};
use rand::prelude::SliceRandom;
use tch::nn::Optimizer;
use tch::{Kind, TchError, Tensor};
use tch::Kind::Float;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, LearningError, TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::RlPolicyConfigBasic;
use crate::tensor_data::{ActionTensorFormat, ContextEncodeTensor, TensorEncoding};
use crate::torch_net::{ActorCriticOutput, DeviceTransfer, NeuralNet};

/// Configuration structure for A2C
#[derive(Copy, Clone, Debug, Getters, Setters)]
pub struct ConfigA2C{
    pub gamma: f64,
    pub mini_batch_size: Option<usize>,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub gae_lambda: Option<f64>
}

impl RlPolicyConfigBasic for ConfigA2C{
    fn gamma(&self) -> f64 {
        self.gamma
    }

    fn gae_lambda(&self) -> Option<f64> {
        self.gae_lambda
    }
}

impl Default for ConfigA2C {
    fn default() -> Self {
        Self{
            gamma: 0.99,
            mini_batch_size: Some(16),
            ent_coef: 0.01,
            vf_coef: 0.5,
            gae_lambda: None
        }
    }
}

/// Helper trait to build Actor Critic policy.
/// It requires a number of methods, but provides action selection methods.
/// Methods required are also used to automatically implement [`PolicyTrainHelperA2C`]/[`PolicyTrainHelperPPO`](crate::policy::PolicyTrainHelperPPO).
pub trait PolicyHelperA2C<DP: DomainParameters>{

    type InfoSet: InformationSet<DP> + ContextEncodeTensor<Self::InfoSetConversionContext>;
    type InfoSetConversionContext: TensorEncoding;
    type ActionConversionContext: ActionTensorFormat<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>;
    type NetworkOutput: ActorCriticOutput;

    type Config: RlPolicyConfigBasic;

    /// Returns reference to policy config.
    fn config(&self) -> &Self::Config;

    /// Mutable reference to optimizer owned by policy.
    fn optimizer_mut(&mut self) -> &mut Optimizer;

    /// Reference to policy neural-network.
    fn network(&self) ->  &NeuralNet<Self::NetworkOutput>;


    /// Return tensor encoding context for information set
    fn info_set_encoding(&self) -> &Self::InfoSetConversionContext;

    /// Returns tensor index decoding and encoding for action.
    fn action_encoding(&self) -> &Self::ActionConversionContext;

    /// Uses information set (state) and network output to calculate (masked) action distribution(s).
    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput)
                -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

    /// Indicate if action masking is supported.
    fn is_action_masking_supported(&self) -> bool;

    /// Generate action masks for information set.
    /// Let's say that action space is 5 and actions 0,2,3 are illegal now.
    /// The result should be Tensor([false, true, false, false, true])
    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

    /// If policy in mode of exploration i.e. if it is not suppressed to select always the best looking action without random walk
    fn is_exploration_on(&self) -> bool;

    /// Tries converting choice tensor to action - for example for single tensor choice `Tensor([3])` - meaning the aaction  with index `3` is selected, the proper action type is constructed.
    fn try_action_from_choice_tensor(&self,
                                         choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
    ) -> Result<DP::ActionType, AmfiteatrError<DP>>;

    /// This function makes more sense with multi tensor actions.
    /// Suppose you have some action B(7,3), where B, 7 and 3 are parameters from three different parameter distribution.
    /// Action is chosen from five parameters, but when parameter i0 is B parameters i1 and i4 are not used.
    /// Therefore, when constructing ac`tion B we lose information of sampling of parameters i1 and i4.
    /// What;s more they do not had impact on game result, therefore we would like to avoid including them in calculating B(7,3) probability,
    /// and somehow exclude them from impact on entropy.
    /// So in this case we would produce from B(7,3):
    /// 1. Vector of choice tensors `vec![[1], [?], [7], [3], [?]]`
    /// 2. Vector of parameter masks: `vec![[true], [false], [true], [true], [false]`).
    fn vectorize_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<
        (
             <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
             <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType
        ),
        AmfiteatrError<DP>
    >;

    /// Calculate  actions (choices in batch) a triple of
    /// + log probability
    /// + distribution entropy
    /// + critic value
    fn batch_get_logprob_entropy_critic(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
        action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>;


    /// Automatically implemented action selection using required methods.
    fn a2c_select_action(&self, info_set: &Self::InfoSet) -> Result<DP::ActionType, AmfiteatrError<DP>>{
        let state_tensor = info_set.to_tensor(self.info_set_encoding());
        let out = tch::no_grad(|| (self.network().net())(&state_tensor));
        //let actor = out.actor;
        //println!("out: {:?}", out);
        let probs = self.dist(info_set, &out)?;
        //println!("probs: {:?}", probs);
        let choices = match self.is_exploration_on(){
            true => {

                Self::NetworkOutput::perform_choice(&probs, |t| t.f_multinomial(1, true))?
            },

            false => Self::NetworkOutput::perform_choice(&probs, |t| t.f_argmax(None, false)?.f_unsqueeze(-1))?,
        };

        self.try_action_from_choice_tensor( &choices).map_err(|err| {
            #[cfg(feature = "log_error")]
            log::error!("Failed creating action from choices tensor. Error: {}. Tensor: {:?}", err, choices);
            err
        })
    }

    /// Calculates *delta* for GAE advantage.
    fn calculate_delta(index: i64, critic_values: &Tensor, rewards: &Tensor, next_critic_value: &Tensor, gamma: f64, next_nonterminal: f64) -> Result<Tensor, TchError>{
        Ok(rewards.f_get(index)? + (next_critic_value * gamma * next_nonterminal) - critic_values.f_get(index)?)
    }

    /// Calculates gae advantage.
    fn calculate_gae_advantages_and_returns
    <R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>(
        &self,
        trajectory: &AgentTrajectory<DP, Self::InfoSet>,
        critic_t: &Tensor,
        reward_f: &R,
        gae_lambda: f64,

    ) -> Result<(Tensor, Tensor), AmfiteatrError<DP>> {

        //let device = self.network().device();

        let device = critic_t.device();

        if let Some(last_reward) = trajectory.last_view_step()
            .and_then(|ref t| Some(reward_f(t))){


            let mut last_reward_shape = last_reward.size();
            let mut shape = vec![1];
            shape.append(&mut last_reward_shape);

            let rewards = trajectory.iter().map(|step|{
                reward_f(&step)
            }).collect::<Vec<Tensor>>();

            let rewards_t = Tensor::f_vstack(&rewards)
                .map_err(|e| TensorError::from_tch_with_context(e, "Stacking rewards for gae advantage computation".into()))?
                .to_device(device);
            let mut last_gae_lambda = Tensor::from(0.0);
            let advantages_t = Tensor::zeros(critic_t.size(), (Kind::Float, device));
            #[cfg(feature = "log_trace")]
            log::trace!("Crirtic tensor: {critic_t}");
            for index in (0..trajectory.number_of_steps()).rev()
                .map(|i, | i as i64,){
                //chgeck if last step
                let (next_nonterminal, next_value) = match index == trajectory.number_of_steps() as i64 -1{
                    true => (0.0, Tensor::zeros(
                        critic_t.f_get(0).map_err(|e|TensorError::from_tch_with_context(e, "ciritic tensor - get(0)".into()))?
                            .size(), (Kind::Float, device))
                    ),
                    false => (1.0, critic_t.f_get(index+1)
                        .map_err(|e|TensorError::from_tch_with_context(e, format!("ciritic tensor - get({})", index + 1)))?)
                };
                let delta   = rewards_t.get(index) + (next_value.f_mul_scalar(self.config().gamma()).unwrap().f_mul_scalar(next_nonterminal).unwrap()) - critic_t.f_get(index).unwrap();

                //let delta = Self::calculate_delta(index, &rewards_t, &critic_t, &next_value, self.config().gamma(), next_nonterminal)
                    //.map_err(|e| TensorError::from_tch_with_context(e, "Calculating delta dor gae lambda".into()))?;
                last_gae_lambda = delta + ( last_gae_lambda * self.config().gamma() * gae_lambda * next_nonterminal);
                advantages_t.get(index).copy_(&last_gae_lambda);
            }
            let returns_t = advantages_t.f_add(critic_t)
                .map_err(|e| TensorError::from_tch_with_context(e, "Calculating estimated returns from advantages and critic".into()))?;

            #[cfg(feature = "log_trace")]
            log::trace!("Rewards tensor: {rewards_t}");
            #[cfg(feature = "log_trace")]
            log::trace!("Returns tensor: {returns_t}");
            #[cfg(feature = "log_trace")]
            log::trace!("Advantages tensor: {advantages_t}");

            Ok((advantages_t, returns_t))
        } else {
            Err(AmfiteatrError::Learning {
                error: LearningError::EmptyTrajectory
            })
        }


    }

    /// Calculates traditional advantage [R(s,a) + (gamma * reward_sum) - V(s) ]
    fn calculate_advantages_and_returns<R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>(
        &self,
        trajectory: &AgentTrajectory<DP, Self::InfoSet>,
        critic: &Tensor,
        reward_f: &R,

    ) -> Result<(Tensor, Tensor), AmfiteatrError<DP>>{
        let device = self.network().device();

        if let Some(last_reward) = trajectory.last_view_step()
            .and_then(|ref t| Some(reward_f(t))){
            #[cfg(feature = "log_trace")]
            log::trace!("Last reward size: {:?}, last reward: {}", last_reward.size(), last_reward);
            let mut discounted_payoffs = (0..=trajectory.number_of_steps())
                .map(|_|Tensor::zeros(last_reward.size(), (Kind::Float, device)))
                .collect::<Vec<Tensor>>();

            for s in (0..discounted_payoffs.len()-1).rev(){
                //println!("{}", s);
                let this_reward = reward_f(&trajectory.view_step(s).unwrap()).to_device(device);
                let r_s = &this_reward + (&discounted_payoffs[s+1] * self.config().gamma());
                discounted_payoffs[s].copy_(&r_s);
                #[cfg(feature = "log_trace")]
                log::trace!("Calculating discounted payoffs for {} step. This step reward {}, following payoff: {}, result: {}.",
                    s, this_reward, discounted_payoffs[s+1], r_s);
            }
            discounted_payoffs.pop();

            #[cfg(feature = "log_trace")]
            log::trace!("Discounted payoffs for trajectory: {:?}", discounted_payoffs);

            #[cfg(feature = "log_trace")]
            log::trace!("Critic tensor: {}", critic);

            let payoff_tensor = Tensor::f_vstack(&discounted_payoffs[..])
                .map_err(|e| TensorError::Torch{
                    origin: format!("{e}"),
                    context: "Stacking discounted payoffs to tensor".to_string(),
                })?
                .to_device(critic.device());

            #[cfg(feature = "log_trace")]
            log::trace!("Payoff tensor: {}", payoff_tensor);
            let advantage =
            payoff_tensor.f_sub(&critic).map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: "Calculating advantage from critic tensor and discounted payoffs tensor".to_string(),
                }
            }
            )?;
            Ok((advantage, payoff_tensor))

        } else {
            Err(AmfiteatrError::Learning {
                error: LearningError::EmptyTrajectory
            })
        }



    }

}

pub trait PolicyTrainHelperA2C<DP: DomainParameters> : PolicyHelperA2C<DP, Config=ConfigA2C>{

    fn a2c_train_on_trajectories<
        R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>
    (
        &mut self, trajectories: &[AgentTrajectory<DP, Self::InfoSet>],
        reward_f: R
    ) -> Result<(), AmfiteatrError<DP>>{
        let mut rng = rand::rng();

        #[cfg(feature = "log_trace")]
        log::trace!("Starting a2c train session");


        let device = self.network().device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
            acc + x.number_of_steps()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.number_of_steps()
        }).max().unwrap_or(0);

        let step_example = trajectories.iter().find(|&trajectory|{
            trajectory.view_step(0).is_some()
        }).and_then(|trajectory| trajectory.view_step(0))
            .ok_or(AmfiteatrRlError::NoTrainingData)?;

        let sample_info_set = step_example.information_set();

        let sample_info_set_t = sample_info_set.try_to_tensor(self.info_set_encoding())?;
        let sample_net_output = tch::no_grad(|| self.network().net()(&sample_info_set_t));
        let action_params = sample_net_output.param_dimension_size() as usize;


        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);

        let mut action_masks_vec = Self::NetworkOutput::new_batch_with_capacity(action_params ,capacity_estimate);
        let mut multi_action_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut multi_action_cat_mask_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);




        let mut tmp_trajectory_action_tensor_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);

        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut returns_vec = Vec::with_capacity(capacity_estimate);
        //let mut gae_returns_v = Vec::new();
        for t in trajectories {

            t.view_step(0).inspect(|_t|{
                #[cfg(feature = "log_trace")]
                log::trace!("Training neural-network for agent {} (from first trace step entry).", _t.information_set().agent_id());

            });

            if t.last_view_step().is_none(){
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.");
                continue;
            }
            //let final_score_t =   reward_f(&t.last_view_step().unwrap());

            tmp_trajectory_state_tensor_vec.clear();
            Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_tensor_vecs);
            Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_category_mask_vecs);
            tmp_trajectory_reward_vec.clear();
            for step in t.iter(){
                #[cfg(feature = "log_trace")]
                log::trace!("Adding information set tensor to single trajectory vec.",);
                tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(self.info_set_encoding())?);
                #[cfg(feature = "log_trace")]
                log::trace!("Added information set tensor to single trajectory vec.",);
                //let (act_t, cat_mask_t) = step.action().action_index_and_mask_tensor_vecs(&self.action_conversion_context())?;
                let (act_t, cat_mask_t) = self.vectorize_action_and_create_category_mask(step.action())?;
                #[cfg(feature = "log_trace")]
                log::trace!("act_t: {:?}", act_t);
                Self::NetworkOutput::push_to_vec_batch(&mut tmp_trajectory_action_tensor_vecs, act_t);

                #[cfg(feature = "log_trace")]
                log::trace!("tmp_trajectory_action_tensor_vecs: {:?}", tmp_trajectory_action_tensor_vecs);
                Self::NetworkOutput::push_to_vec_batch(&mut tmp_trajectory_action_category_mask_vecs, cat_mask_t);

                tmp_trajectory_reward_vec.push(reward_f(&step));
                if self.is_action_masking_supported(){
                    //action_masks_vec.push(self.generate_action_masks(step.information_set())?);
                    Self::NetworkOutput::push_to_vec_batch(&mut action_masks_vec, self.generate_action_masks(step.information_set())?);
                }
            }
            let information_set_t = Tensor::f_stack(&tmp_trajectory_state_tensor_vec[..],0)
                .map_err(|e|TensorError::Torch {
                    origin: format!("{e}"),
                    context: "Stacking information set - tensors".into(),
                }) ?.to_device(device);
            #[cfg(feature = "log_trace")]
            log::trace!("Tmp infoset shape = {:?}", information_set_t.size());
            let net_out = tch::no_grad(|| (self.network().net())(&information_set_t));
            let critic_t = net_out.critic();
            #[cfg(feature = "log_trace")]
            log::trace!("Tmp values_t shape = {:?}", critic_t.size());
            //let mut returns_t = Tensor::from(0.0);

            //let mut advantages_t = Tensor::from(0.0);

            let (advantages_t, returns_t) = if let Some(gae_lambda) = self.config().gae_lambda{
                tch::no_grad(||self.calculate_gae_advantages_and_returns(t, critic_t, &reward_f, gae_lambda))?
            } else {
                tch::no_grad(||self.calculate_advantages_and_returns(t, critic_t, &reward_f))?
            };


            state_tensor_vec.push(information_set_t);
            advantage_tensor_vec.push(advantages_t);
            returns_vec.push(returns_t);
            Self::NetworkOutput::append_vec_batch(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs );

            Self::NetworkOutput::append_vec_batch(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs );

        }

        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)
            .map_err(|e| TensorError::Torch {
                origin: format!("{e}"),
                context: "Stacking information sets to batch tensor.".into(),
            })?
            .move_to_device(device);


        let action_forward_masks = match self.is_action_masking_supported(){
            true => Some(Self::NetworkOutput::stack_tensor_batch(&action_masks_vec)?.move_to_device(device)),
            false => None
        };
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        //let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec,)?.move_to_device(device);
        let batch_returns_t = Tensor::f_vstack(&returns_vec)
            .map_err(|e| TensorError::from_tch_with_context(e, "Batching return tensors".into()))?
            .move_to_device(device);
        #[cfg(feature = "log_trace")]
        log::trace!("Batch returns shape = {:?}", batch_returns_t.size());

        let batch_actions_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_tensor_vec)?
            .move_to_device(device);
        let batch_action_masks_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_cat_mask_tensor_vec)?
            .move_to_device(device);

        let batch_size = batch_info_sets_t.size()[0];
        let mut indices: Vec<i64> = (0..batch_size).collect();
        indices.shuffle(&mut rng);



        let mini_batch_size = self.config().mini_batch_size.unwrap_or(batch_size as usize);

        for minibatch_start in (0..batch_size).step_by(mini_batch_size){
            let minibatch_end = min(minibatch_start + mini_batch_size as i64, batch_size);
            let minibatch_indices = Tensor::from(&indices[minibatch_start as usize..minibatch_end as usize]).to_device(device);

            let mini_batch_action = Self::NetworkOutput::index_select(&batch_actions_t, &minibatch_indices)
                .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching action".into()))?;
            let mini_batch_action_cat_mask = Self::NetworkOutput::index_select(&batch_action_masks_t, &minibatch_indices)
                .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching action categories masks".into()))?;

            let mini_batch_action_forward_mask = match action_forward_masks{
                None => None,
                Some(ref m) => Some(Self::NetworkOutput::index_select(m, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching action masks".into()))?)
            };

            //let mini_batch_base_logprobs = batch_logprob_t.f_index_select(0, &minibatch_indices)?;
            #[cfg(feature = "log_trace")]
            log::trace!("Selected minibatch logprobs");
            let (log_probs, entropy, critic) = self.batch_get_logprob_entropy_critic(
                &batch_info_sets_t.f_index_select(0, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Calculating log_prob, entropy and critic value for mini-batch".into()))?,
                &mini_batch_action,
                Some(&mini_batch_action_cat_mask),
                mini_batch_action_forward_mask.as_ref(), //to add it some day
            )?;

            let minibatch_returns_t = batch_returns_t.f_index_select(0, &minibatch_indices)
                .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching returns".into()))?;
            //let minibatch_values_t = batch_values_t.f_index_select(0, &minibatch_indices)?;

            //let dist_entropy = categorical_dist_entropy(&probs, &log_probs, Kind::Float).mean(Float);

            let advantages = minibatch_returns_t.f_sub(&critic)
                .map_err(|e| AmfiteatrError::Tensor {error: TensorError::Torch {
                    origin: format!{"{e}"},
                    context: "Calculating A2C advantages.".into(),
                }})?;

            let value_loss = (&advantages * &advantages).mean(Float);
            let action_loss = (-advantages.detach() * log_probs).mean(Float);
            let entropy_loss = entropy.f_mean(Float)
                .map_err(|e| TensorError::from_tch_with_context(e, "Calculating entropy loss".into()))?;
            let loss = action_loss
                .f_add(&(value_loss * self.config().vf_coef))
                .map_err(|e| TensorError::from_tch_with_context(e, "Calculating loss (adding value loss)".into()))?
                .f_sub(&(entropy_loss * self.config().ent_coef))
                .map_err(|e| TensorError::from_tch_with_context(e, "Calculating loss (subtracting entropy loss)".into()))?;

            self.optimizer_mut().zero_grad();

            self.optimizer_mut().backward_step(&loss);

        }

        Ok(())
    }

}

impl<T, DP: DomainParameters> PolicyTrainHelperA2C<DP> for T
    where T: PolicyHelperA2C<DP, Config=ConfigA2C>{}


