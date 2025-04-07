//! Based on [cleanrl PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)

use std::cmp::min;
use getset::{Getters, Setters};
use rand::prelude::SliceRandom;
use tch::{Kind, Tensor};
use tch::Kind::Float;
use tch::nn::Optimizer;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigA2C, PolicyHelperA2C, PolicyTrainHelperA2C, RlPolicyConfigBasic};
use crate::tensor_data::{ActionTensorFormat, ContextEncodeTensor, TensorEncoding};
use crate::torch_net::{ActorCriticOutput, DeviceTransfer, NeuralNet};


/// Configuration structure for PPO Policy
#[derive(Copy, Clone, Debug, Getters, Setters)]
pub struct ConfigPPO {
    pub gamma: f64,
    pub clip_vloss: bool,
    pub clip_coef: f64,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub max_grad_norm: f64,
    pub gae_lambda_old: f64,
    pub gae_lambda: Option<f64>,
    pub mini_batch_size: usize,
    pub tensor_kind: tch::kind::Kind,
    pub update_epochs: usize,
    //pub
}

impl Default for ConfigPPO {
    fn default() -> ConfigPPO {
        Self{
            gamma: 0.99,
            clip_vloss: true,
            clip_coef: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            max_grad_norm: 0.5,
            gae_lambda_old: 0.95,
            gae_lambda: Some(0.95),
            mini_batch_size: 16,
            tensor_kind: tch::kind::Kind::Float,
            update_epochs: 4,
        }
    }
}

impl RlPolicyConfigBasic for ConfigPPO {
    fn gamma(&self) -> f64 {
        self.gamma
    }

    fn gae_lambda(&self) -> Option<f64> {
        self.gae_lambda
    }
}

#[deprecated(since = "0.8.0", note = "Use [PolicyHelperA2C] and [PolicyTrainHelperPPO]")]
/// The sole purpose of this trait is to provide some function dealing with
/// information sets and actions using tensors. It is used to create final implementations.
/// It is made to have single implementation of PPO, regardless if action space is single discrete space or multiple discrete space.
/// And whenever action masking is supported.
/// **You probably don't need to implement this trait if you can use provided implementations.**
/// There are provided generic implementations for:
/// + [`PolicyPpoDiscrete`](crate::policy::ppo::PolicyDiscretePPO);
/// + [`PolicyPpoMultiDiscrete`](crate::policy::ppo::PolicyMultiDiscretePPO);
/// + [`PolicyMaskingPpoDiscrete`](crate::policy::ppo::PolicyMaskingDiscretePPO);
///+ [`PolicyMaskingPpoMultiDiscrete`](crate::policy::ppo::PolicyMaskingMultiDiscretePPO).
pub trait PolicyHelperPPO<DP: DomainParameters>
{
    type InfoSet: InformationSet<DP> + ContextEncodeTensor<Self::InfoSetConversionContext>;
    type InfoSetConversionContext: TensorEncoding;
    type ActionConversionContext: ActionTensorFormat<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>;

    type NetworkOutput: ActorCriticOutput;

    /// Returns reference to policy config.
    fn config(&self) -> &ConfigPPO;

    /// Mutable reference to optimizer owned by policy.
    fn optimizer_mut(&mut self) -> &mut Optimizer;


    /// Reference to policy neural-network.
    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput>;

    /// Return tensor encoding context for information set
    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext;

    /// Returns tensor index decoding and encoding for action.
    fn action_conversion_context(&self) -> &Self::ActionConversionContext;

    /// Uses information set (state) and network output to calculate (masked) action distribution(s).
    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput)
        -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;


    /// Indicate if action masking is supported.
    fn is_action_masking_supported(&self) -> bool;


    /// Generate action masks for information set.
    /// Let's say that action space is 5 and actions 0,2,3 are illegal now.
    /// The result should be Tensor([false, true, false, false, true])
    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

    /// If policy in mode of exploration i.e. if it is not suppressed to select always the best looking action without random walk
    fn ppo_exploration(&self) -> bool;


    /// Tries converting choice tensor to action - for example for single tensor choice `Tensor([3])` - meaning the aaction  with index `3` is selected, the proper action type is constructed.
    fn ppo_try_action_from_choice_tensor(&self,
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
    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType)
        -> Result<(
            <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
            <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType
        ), AmfiteatrError<DP>>;


    /// Calculate  actions (choices in batch) a triple of
    /// + log probability
    /// + distribution entropy
    /// + critic value
    fn ppo_batch_get_logprob_entropy_critic(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
        action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>;

    /// Automatically implemented action selection using required methods.
    fn ppo_select_action(&self, info_set: &Self::InfoSet) -> Result<DP::ActionType, AmfiteatrError<DP>>{
        let state_tensor = info_set.to_tensor(self.info_set_conversion_context());
        let out = tch::no_grad(|| (self.ppo_network().net())(&state_tensor));
        //let actor = out.actor;
        //println!("out: {:?}", out);
        let probs = self.ppo_dist(info_set, &out)?;
        //println!("probs: {:?}", probs);
        let choices = match self.ppo_exploration(){
            true => {

                Self::NetworkOutput::perform_choice(&probs, |t| t.f_multinomial(1, true))?
            },
              //  probs.into_iter().map(|t| t.multinomial(1, true)).collect(),

            false => Self::NetworkOutput::perform_choice(&probs, |t| t.f_argmax(None, false)?.f_unsqueeze(-1))?,
                //probs.into_iter().map(|t| t.argmax(None, false).unsqueeze(-1)).collect()
        };

        self.ppo_try_action_from_choice_tensor( &choices).map_err(|err| {
            #[cfg(feature = "log_error")]
            log::error!("Failed creating action from choices tensor. Error: {}. Tensor: {:?}", err, choices);
            err
        })
    }

    fn ppo_train_on_trajectories<
        R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>
    (
        &mut self, trajectories: &[AgentTrajectory<DP, Self::InfoSet>],
        reward_f: R
    ) -> Result<(), AmfiteatrRlError<DP>>{
        #[cfg(feature = "log_trace")]
        log::trace!("Starting training PPO.");

        let device = self.ppo_network().device();
        let capacity_estimate = AgentTrajectory::sum_trajectories_steps(trajectories);

        //find single sample step

        let step_example = trajectories.iter().find(|&trajectory|{
           trajectory.view_step(0).is_some()
        }).and_then(|trajectory| trajectory.view_step(0))
            .ok_or(AmfiteatrRlError::NoTrainingData)?;

        let sample_info_set = step_example.information_set();

        let sample_info_set_t = sample_info_set.try_to_tensor(self.info_set_conversion_context())?;
        let sample_net_output = tch::no_grad(|| self.ppo_network().net()(&sample_info_set_t));
        let action_params = sample_net_output.param_dimension_size() as usize;
        //let info_set_example = step_example.in





        let mut rng = rand::rng();

        let tmp_capacity_estimate = AgentTrajectory::find_max_trajectory_len(trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        //let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);


        let mut action_masks_vec = Self::NetworkOutput::new_batch_with_capacity(action_params ,capacity_estimate);
        let mut multi_action_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut multi_action_cat_mask_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);


        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut tmp_trajectory_action_tensor_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);

        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut returns_vec = Vec::new();

        #[cfg(feature = "log_debug")]
        log::debug!("Starting operations on trajectories.",);
        for t in trajectories{


            if let Some(_last_step) = t.last_view_step(){

                tmp_trajectory_state_tensor_vec.clear();
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_tensor_vecs);
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_category_mask_vecs);
                //let final_reward_t = reward_f(&last_step).to_device(device);
                tmp_trajectory_reward_vec.clear();
                for step in t.iter(){
                    #[cfg(feature = "log_trace")]
                    log::trace!("Adding information set tensor to single trajectory vec.",);
                    tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(self.info_set_conversion_context())?);
                    #[cfg(feature = "log_trace")]
                    log::trace!("Added information set tensor to single trajectory vec.",);
                    //let (act_t, cat_mask_t) = step.action().action_index_and_mask_tensor_vecs(&self.action_conversion_context())?;
                    let (act_t, cat_mask_t) = self.ppo_vectorise_action_and_create_category_mask(step.action())?;
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


                let information_set_t = Tensor::f_stack(&tmp_trajectory_state_tensor_vec[..],0)?.f_to_device(device)?;
                #[cfg(feature = "log_trace")]
                log::trace!("Tmp infoset shape = {:?}, device: {:?}", information_set_t.size(), information_set_t.device());

                let net_out = tch::no_grad(|| self.ppo_network().net()(&information_set_t));
                let critic_t = net_out.critic();
                #[cfg(feature = "log_trace")]
                log::trace!("Tmp values_t shape = {:?}", critic_t.size());
                let rewards_t = Tensor::f_stack(&tmp_trajectory_reward_vec[..],0)?.f_to_device(device)?;

                let advantages_t = Tensor::zeros(critic_t.size(), (Kind::Float, device));
                //let advantages_t2 = Tensor::zeros(critic_t.size(), (Kind::Float, device));
                //let mut next_is_final = 1f32;
                let mut last_gae_lambda = Tensor::from(0.0).to_device(device);
                //let mut last_gae_lambda2 = Tensor::from(0.0).to_device(device);
                for index in (0..t.number_of_steps()).rev()
                    .map(|i, | i as i64,){
                    //chgeck if last step
                    let (next_nonterminal, next_value) = match index == t.number_of_steps() as i64 -1{
                        true => (0.0, Tensor::zeros(critic_t.f_get(0)?.size(), (Kind::Float, device))),
                        false => (1.0, critic_t.f_get(index+1)?)
                    };
                    let delta   = rewards_t.f_get(index)? + (next_value.f_mul_scalar(self.config().gamma)?.f_mul_scalar(next_nonterminal)?) - critic_t.f_get(index)?;
                    //last_gae_lambda = &delta + (  self.config().gamma * self.config().gae_lambda * next_nonterminal);
                    last_gae_lambda = delta + ( last_gae_lambda * self.config().gamma * self.config().gae_lambda_old * next_nonterminal);
                    advantages_t.f_get(index)?.f_copy_(&last_gae_lambda.detach_copy())?;
                    //advantages_t2.f_get(index)?.f_copy_(&last_gae_lambda2)?;
                }
                returns_vec.push(advantages_t.f_add(critic_t)?);

                state_tensor_vec.push(information_set_t);
                advantage_tensor_vec.push(advantages_t);
                #[cfg(feature = "log_trace")]
                log::trace!("tmp_trajectory_action_tensor_vecs: {:?}", tmp_trajectory_action_tensor_vecs);
                #[cfg(feature = "log_trace")]
                log::trace!("multi_action_tensor_vec: {:?}", multi_action_tensor_vec);
                Self::NetworkOutput::append_vec_batch(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs );

                Self::NetworkOutput::append_vec_batch(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs );



            } else {
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.")
            }


        }
        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)?.move_to_device(device);
        let action_forward_masks = match self.is_action_masking_supported(){
            true => Some(Self::NetworkOutput::stack_tensor_batch(&action_masks_vec)?.move_to_device(device)),
            false => None
        };
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec,)?.move_to_device(device);
        let batch_returns_t = Tensor::f_vstack(&returns_vec)?.move_to_device(device);
        #[cfg(feature = "log_trace")]
        log::trace!("Batch returns shape = {:?}", batch_returns_t.size());

        #[cfg(feature = "log_trace")]
        log::trace!("Batch advantage shape = {:?}", batch_advantage_t.size());
        /*
        let batch_actions_t= multi_action_tensor_vec.iter().map(|cat|{
            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;

        let batch_action_masks_t= multi_action_cat_mask_tensor_vec.iter().map(|cat|{

            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;

         */
        let batch_actions_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_tensor_vec)?
            .move_to_device(device);
        let batch_action_masks_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_cat_mask_tensor_vec)?
            .move_to_device(device);



        let batch_size = batch_info_sets_t.size()[0];
        let mut indices: Vec<i64> = (0..batch_size).collect();

        /*
        let action_forward_masks = match self.is_action_masking_supported(){
            true => self.batch_generate_action_masks()
            false => None,
        }

         */

        #[cfg(feature = "log_trace")]
        log::trace!("batch_actions_t: {:?}", batch_actions_t);

        let (batch_logprob_t, _entropy, batch_values_t) = tch::no_grad(||{
            self.ppo_batch_get_logprob_entropy_critic(
                &batch_info_sets_t,
                &batch_actions_t,
                Some(&batch_action_masks_t),
                action_forward_masks.as_ref(),
            )

        })?;
        for epoch in 0..self.config().update_epochs{
            #[cfg(feature = "log_debug")]
            log::debug!("PPO Update Epoch: {epoch}");

            indices.shuffle(&mut rng);
            //println!("{indices:?}")

            for minibatch_start in (0..batch_size).step_by(self.config().mini_batch_size){
                let minibatch_end = min(minibatch_start + self.config().mini_batch_size as i64, batch_size);
                let minibatch_indices = Tensor::from(&indices[minibatch_start as usize..minibatch_end as usize]).to_device(device);


                let mini_batch_action = Self::NetworkOutput::index_select(&batch_actions_t, &minibatch_indices)?;
                let mini_batch_action_cat_mask = Self::NetworkOutput::index_select(&batch_action_masks_t, &minibatch_indices)?;


                let mini_batch_action_forward_mask = match action_forward_masks{
                    None => None,
                    Some(ref m) => Some(Self::NetworkOutput::index_select(m, &minibatch_indices)?)
                };


                let mini_batch_base_logprobs = batch_logprob_t.f_index_select(0, &minibatch_indices)?;
                #[cfg(feature = "log_trace")]
                log::trace!("Selected minibatch logprobs");
                let (new_logprob, entropy, newvalue) = self.ppo_batch_get_logprob_entropy_critic(
                    &batch_info_sets_t.f_index_select(0, &minibatch_indices)?,
                    &mini_batch_action,
                    Some(&mini_batch_action_cat_mask),
                    mini_batch_action_forward_mask.as_ref(), //to add it some day
                )?;
                #[cfg(feature = "log_debug")]
                log::debug!("Advantages: {:?}", batch_advantage_t.f_index_select(0, &minibatch_indices)?);
                #[cfg(feature = "log_debug")]
                log::debug!("Entropy: {:}", entropy);

                #[cfg(feature = "log_debug")]
                log::debug!("Base logbprob: {:}", mini_batch_base_logprobs);

                #[cfg(feature = "log_debug")]
                log::debug!("New logprob: {:}", new_logprob);

                let logratio = new_logprob.f_sub(&mini_batch_base_logprobs)?;
                let ratio  = logratio.f_exp()?;

                #[cfg(feature = "log_debug")]
                log::debug!("Log ratio: {:}", logratio);

                //Approximate KL

                let (r_old_approx_kl, r_approx_kl) = tch::no_grad(|| {
                    let old_approx_kl = (-&logratio).f_mean(Float);
                    let approx_kl = ((&ratio -1.0) - &logratio).f_mean(Float);
                    //let clip_frac = ((&ratio -1.0).abs().f_is_g)

                    (old_approx_kl, approx_kl)

                });

                let old_approx_kl = r_old_approx_kl?;
                let approx_kl = r_approx_kl?;

                let minibatch_advantages_t = batch_advantage_t.f_index_select(0, &minibatch_indices)?;
                let minibatch_returns_t = batch_returns_t.f_index_select(0, &minibatch_indices)?;
                let minibatch_values_t = batch_values_t.f_index_select(0, &minibatch_indices)?;
                #[cfg(feature = "log_debug")]
                log::debug!("Old Approximate KL: {:}", old_approx_kl);

                #[cfg(feature = "log_debug")]
                log::debug!("Approximate KL: {:}", approx_kl);

                let pg_loss1 = -&minibatch_advantages_t.f_mul(&ratio)?;
                let pg_loss2 = -&minibatch_advantages_t.f_mul(&ratio.f_clamp(1.0 - self.config().clip_coef, 1.0 + self.config().clip_coef)?)?;
                let pg_loss = pg_loss1.f_max_other(&pg_loss2)?.f_mean(self.config().tensor_kind)?;

                #[cfg(feature = "log_debug")]
                log::debug!("PG loss : {}", pg_loss);



                let v_loss = if self.config().clip_vloss{
                    let v_loss_unclipped = (newvalue.f_sub(&minibatch_returns_t)?).f_square()?;
                    let v_clipped =minibatch_values_t.f_add(
                        &newvalue.f_sub(&minibatch_values_t)?
                            .f_clamp(
                                - self.config().clip_coef,
                                self.config().clip_coef
                            )?
                    )?;
                    let v_loss_clipped = (v_clipped.f_sub(&minibatch_returns_t)?).f_square()?;
                    let v_loss_max = v_loss_unclipped.f_max_other(&v_loss_clipped)?;
                    v_loss_max.f_mean(self.config().tensor_kind)? * 0.5
                } else {
                    newvalue.f_sub(&minibatch_returns_t)?.f_square()?.f_mean(self.config().tensor_kind)? *0.5
                };

                let entropy_loss = entropy.f_mean(self.config().tensor_kind)?;
                let loss = pg_loss
                    .f_sub(&(entropy_loss * self.config().ent_coef))?
                    .f_add(&(v_loss * self.config().vf_coef))?;

                self.optimizer_mut().zero_grad();

                self.optimizer_mut().backward_step(&loss);



            }

        }

        Ok(())
    }





}

/// Helper trait to build create training interface for PPO Policy.
/// It provides automatic [`ppo_train_on_trajectories`](PolicyTrainHelperPPO::ppo_train_on_trajectories)
/// implementation for any (policy) type implementing [`PolicyHelperA2C`].
pub trait PolicyTrainHelperPPO<DP: DomainParameters> : PolicyHelperA2C<DP, Config=ConfigPPO>{


    /// Method provided that executes learning step on PPO policy, based on provided [`PolicyHelperA2C`]
    fn ppo_train_on_trajectories<
        R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>
    (
        &mut self, trajectories: &[AgentTrajectory<DP, Self::InfoSet>],
        reward_f: R
    ) -> Result<(), AmfiteatrError<DP>>{

        #[cfg(feature = "log_trace")]
        log::trace!("Starting training PPO.");

        let device = self.network().device();
        let capacity_estimate = AgentTrajectory::sum_trajectories_steps(trajectories);

        //find single sample step

        let step_example = trajectories.iter().find(|&trajectory|{
            trajectory.view_step(0).is_some()
        }).and_then(|trajectory| trajectory.view_step(0))
            .ok_or(AmfiteatrRlError::NoTrainingData)?;

        let sample_info_set = step_example.information_set();

        let sample_info_set_t = sample_info_set.try_to_tensor(self.info_set_encoding())?;
        let sample_net_output = tch::no_grad(|| self.network().net()(&sample_info_set_t));
        let action_params = sample_net_output.param_dimension_size() as usize;

        let mut rng = rand::rng();

        let tmp_capacity_estimate = AgentTrajectory::find_max_trajectory_len(trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        //let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);


        let mut action_masks_vec = Self::NetworkOutput::new_batch_with_capacity(action_params ,capacity_estimate);
        let mut multi_action_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut multi_action_cat_mask_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);


        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut tmp_trajectory_action_tensor_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);

        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut returns_vec = Vec::new();

        #[cfg(feature = "log_debug")]
        log::debug!("Starting operations on trajectories.",);
        for t in trajectories{


            if let Some(_last_step) = t.last_view_step(){

                tmp_trajectory_state_tensor_vec.clear();
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_tensor_vecs);
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_category_mask_vecs);
                //let final_reward_t = reward_f(&last_step).to_device(device);
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


                let information_set_t = Tensor::f_stack(&tmp_trajectory_state_tensor_vec[..], 0)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Stacking information sets tensors (from trajectory tensors to batch tensors).".into()))?
                    .to_device(device);
                #[cfg(feature = "log_trace")]
                log::trace!("State tensor vector: {:?}", tmp_trajectory_state_tensor_vec);

                #[cfg(feature = "log_trace")]
                log::trace!("Tmp infoset shape = {:?}, device: {:?}", information_set_t.size(), information_set_t.device());
                let net_out = tch::no_grad(|| (self.network().net())(&information_set_t));
                let critic_t = net_out.critic();
                #[cfg(feature = "log_trace")]
                log::trace!("Critic: {}", critic_t);

                let (advantages_t, returns_t) = match self.config().gae_lambda {
                    Some(gae_lambda) => tch::no_grad(||self.calculate_gae_advantages_and_returns(t, critic_t, &reward_f, gae_lambda))?,
                    None => tch::no_grad(||self.calculate_advantages_and_returns(t, critic_t, &reward_f))?
                };

                #[cfg(feature = "log_trace")]
                log::trace!("trajectory advantages_t {:}", advantages_t);
                #[cfg(feature = "log_trace")]
                log::trace!("trajectory returns_t {:}", returns_t);

                returns_vec.push(returns_t);

                state_tensor_vec.push(information_set_t);
                advantage_tensor_vec.push(advantages_t);
                #[cfg(feature = "log_trace")]
                log::trace!("tmp_trajectory_action_tensor_vecs: {:?}", tmp_trajectory_action_tensor_vecs);
                //#[cfg(feature = "log_trace")]
                //log::trace!("multi_action_tensor_vec: {:?}", multi_action_tensor_vec);
                Self::NetworkOutput::append_vec_batch(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs );

                Self::NetworkOutput::append_vec_batch(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs );



            } else {
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.")
            }


        }

        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)
            .map_err(|e| TensorError::from_tch_with_context(
                e,
                "Stacking information sets (batch) from trajectory info set tensors.".into()
            ))?.move_to_device(device);
        let action_forward_masks = match self.is_action_masking_supported(){
            true => Some(Self::NetworkOutput::stack_tensor_batch(&action_masks_vec)?.move_to_device(device)),
            false => None
        };
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec,)
            .map_err(|e| TensorError::from_tch_with_context(e, "Stacking advantages to batch".into()))?.move_to_device(device);
        let batch_returns_t = Tensor::f_vstack(&returns_vec)
            .map_err(|e| TensorError::from_tch_with_context(e, "Stacking returns to batch".into()))?.move_to_device(device);
        #[cfg(feature = "log_trace")]
        log::trace!("Batch returns shape = {:?}", batch_returns_t.size());

        #[cfg(feature = "log_trace")]
        log::trace!("Batch advantage shape = {:?}", batch_advantage_t.size());
        let batch_actions_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_tensor_vec)?
            .move_to_device(device);
        let batch_action_masks_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_cat_mask_tensor_vec)?
            .move_to_device(device);



        let batch_size = batch_info_sets_t.size()[0];
        let mut indices: Vec<i64> = (0..batch_size).collect();

        #[cfg(feature = "log_trace")]
        log::trace!("batch_actions_t: {:?}", batch_actions_t);

        let (batch_logprob_t, _entropy, batch_values_t) = tch::no_grad(||{
            self.batch_get_logprob_entropy_critic(
                &batch_info_sets_t,
                &batch_actions_t,
                Some(&batch_action_masks_t),
                action_forward_masks.as_ref(),
            )

        })?;

        for epoch in 0..self.config().update_epochs{
            #[cfg(feature = "log_debug")]
            log::debug!("PPO Update Epoch: {epoch}");

            indices.shuffle(&mut rng);
            //println!("{indices:?}")

            for minibatch_start in (0..batch_size).step_by(self.config().mini_batch_size){
                let minibatch_end = min(minibatch_start + self.config().mini_batch_size as i64, batch_size);
                let minibatch_indices = Tensor::from(&indices[minibatch_start as usize..minibatch_end as usize]).to_device(device);


                let mini_batch_action = Self::NetworkOutput::index_select(&batch_actions_t, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Selecting actions to mini-batch".into()))?;
                let mini_batch_action_cat_mask = Self::NetworkOutput::index_select(&batch_action_masks_t, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Selecting action categories to mini-batch".into()))?;


                let mini_batch_action_forward_mask = match action_forward_masks{
                    None => None,
                    Some(ref m) => Some(Self::NetworkOutput::index_select(m, &minibatch_indices)
                        .map_err(|e| TensorError::from_tch_with_context(e, "Error creating action mask mini-batch".into()))?)
                };


                let mini_batch_base_logprobs = batch_logprob_t.f_index_select(0, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Creating mini-batch pf log probabilities".into()))?;
                #[cfg(feature = "log_trace")]
                log::trace!("Selected minibatch logprobs");
                let (new_logprob, entropy, newvalue) = self.batch_get_logprob_entropy_critic(
                    &batch_info_sets_t.f_index_select(0, &minibatch_indices).map_err(|e| TensorError::from_tch_with_context(e, "Selecting information set mini-batch".into()))?,
                    &mini_batch_action,
                    Some(&mini_batch_action_cat_mask),
                    mini_batch_action_forward_mask.as_ref(),
                )?;
                #[cfg(feature = "log_debug")]
                log::debug!("Advantages: {:?}", batch_advantage_t.f_index_select(0, &minibatch_indices));
                #[cfg(feature = "log_debug")]
                log::debug!("Entropy: {:}", entropy);

                #[cfg(feature = "log_debug")]
                log::debug!("Base logbprob: {:}", mini_batch_base_logprobs);

                #[cfg(feature = "log_debug")]
                log::debug!("New logprob: {:}", new_logprob);

                let logratio = new_logprob - &mini_batch_base_logprobs ;
                let ratio  = logratio.exp();

                #[cfg(feature = "log_debug")]
                log::debug!("Log ratio: {:}", logratio);

                //Approximate KL

                let (r_old_approx_kl, r_approx_kl) = tch::no_grad(|| {
                    let old_approx_kl = (-&logratio).f_mean(Float);
                    let approx_kl = ((&ratio -1.0) - &logratio).f_mean(Float);
                    //let clip_frac = ((&ratio -1.0).abs().f_is_g)

                    (old_approx_kl, approx_kl)

                });

                let old_approx_kl = r_old_approx_kl
                    .map_err(|e| TensorError::from_tch_with_context(e, "Calculating old KL approximation ".into()))?;
                let approx_kl = r_approx_kl
                    .map_err(|e| TensorError::from_tch_with_context(e, "Calculating KL approximation".into()))?;

                #[cfg(feature = "log_trace")]
                log::trace!("Minibatch indices: {:}", minibatch_indices);
                #[cfg(feature = "log_trace")]
                log::trace!("Batch advantage t: {:?}", batch_advantage_t);



                let minibatch_advantages_t = batch_advantage_t.f_index_select(0, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching advantages".into()))?;
                #[cfg(feature = "log_trace")]
                log::trace!("Batch returns: {:?}", batch_returns_t);
                let minibatch_returns_t = batch_returns_t.f_index_select(0, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching returns".into()))?;
                #[cfg(feature = "log_trace")]
                log::trace!("Batch values: {:?}", batch_values_t);
                let minibatch_values_t = batch_values_t.f_index_select(0, &minibatch_indices)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Mini-batching critic values".into()))?;
                #[cfg(feature = "log_debug")]
                log::debug!("Old Approximate KL: {:}", old_approx_kl);

                #[cfg(feature = "log_debug")]
                log::debug!("Approximate KL: {:}", approx_kl);

                let pg_loss1 = -&minibatch_advantages_t.f_mul(&ratio)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Multiplying advantages and ratio (pg_loss 1)".into()))?;
                let pg_loss2 = -&minibatch_advantages_t * (&ratio.f_clamp(1.0 - self.config().clip_coef, 1.0 + self.config().clip_coef)
                    .map_err(|e| TensorError::from_tch_with_context(e, "Clamping ratio (pg_loss 2)".into()))?);
                let pg_loss = pg_loss1.max_other(&pg_loss2).mean(self.config().tensor_kind);

                #[cfg(feature = "log_trace")]
                log::trace!("Minibatch critic: {newvalue}");

                #[cfg(feature = "log_trace")]
                log::trace!("Minibatch returns: {minibatch_returns_t}");

                #[cfg(feature = "log_debug")]
                log::debug!("PG loss : {}", pg_loss);



                let v_loss = if self.config().clip_vloss{
                    let v_loss_unclipped = (&newvalue - &minibatch_returns_t).square();
                    let v_clipped = &minibatch_values_t + (
                        (&newvalue - &minibatch_values_t)
                            .f_clamp(
                                - self.config().clip_coef,
                                self.config().clip_coef
                            ).map_err(|e| TensorError::from_tch_with_context(e, "Clamping vloss".into()))?
                    );
                    let v_loss_clipped = (v_clipped - &minibatch_returns_t).square();
                    let v_loss_max = v_loss_unclipped.max_other(&v_loss_clipped);
                    v_loss_max.mean(tch::Kind::Float) * self.config().vf_coef
                } else {
                    (newvalue -&minibatch_returns_t).square().mean(Float) * self.config().vf_coef
                };

                #[cfg(feature = "log_debug")]
                log::debug!("V loss : {}", v_loss);

                let entropy_loss = entropy.mean(Float);
                let loss = pg_loss
                    - &(entropy_loss * self.config().ent_coef)
                    + &(v_loss * self.config().vf_coef);

                self.optimizer_mut().zero_grad();

                self.optimizer_mut().backward_step(&loss);



            }

        }







        Ok(())
    }
}

impl<T, DP: DomainParameters> PolicyTrainHelperPPO<DP> for T
    where T: PolicyHelperA2C<DP, Config=ConfigPPO>{}


//impl<P:PolicyPPO<DP>, DP: DomainParameters> Policy<DP> for P