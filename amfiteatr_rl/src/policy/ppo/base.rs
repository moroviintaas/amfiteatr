use std::cmp::min;
use getset::{Getters, Setters};
use rand::prelude::SliceRandom;
use tch::{Kind, TchError, Tensor};
use tch::Kind::Float;
use tch::nn::Optimizer;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::error::AmfiteatrRlError;
use crate::policy::{find_max_trajectory_len, sum_trajectories_steps};
use crate::tensor_data::{ActionTensorFormat, ContextTryIntoTensor, ConversionToTensor};
use crate::torch_net::{ActorCriticOutput, NetOutput, NeuralNet};

///! Based on [cleanrl PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
#[derive(Copy, Clone, Debug, Getters, Setters)]
pub struct ConfigPPO {
    pub gamma: f64,
    pub clip_vloss: bool,
    pub clip_coef: f64,
    pub ent_coef: f64,
    pub vf_coef: f64,
    pub max_grad_norm: f64,
    pub gae_lambda: f64,
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
            gae_lambda: 0.95,
            mini_batch_size: 16,
            tensor_kind: tch::kind::Kind::Float,
            update_epochs: 4,
        }
    }
}
pub trait PolicyHelperPPO<DP: DomainParameters>
{
    type InfoSet: InformationSet<DP> + ContextTryIntoTensor<Self::InfoSetConversionContext>;
    type InfoSetConversionContext: ConversionToTensor;
    type ActionConversionContext: ActionTensorFormat<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>;

    type NetworkOutput: ActorCriticOutput;

    fn config(&self) -> &ConfigPPO;

    fn optimizer_mut(&mut self) -> &mut Optimizer;


    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput>;

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext;
    fn action_conversion_context(&self) -> &Self::ActionConversionContext;

    /// Uses information set (state) and network output to calculate (masked) action distribution(s).
    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput)
        -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

    fn is_action_masking_supported(&self) -> bool;

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

    //fn colle
    fn ppo_exploration(&self) -> bool;

    fn ppo_try_action_from_choice_tensor(&self,
        choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
    ) -> Result<DP::ActionType, AmfiteatrError<DP>>;


    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType)
        -> Result<(
            <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
            <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType
        ), AmfiteatrError<DP>>;


    fn ppo_batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
        action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>;
    fn ppo_select_action(&self, info_set: &Self::InfoSet) -> Result<DP::ActionType, AmfiteatrError<DP>>{
        let state_tensor = info_set.to_tensor(self.info_set_conversion_context());
        let out = tch::no_grad(|| (self.ppo_network().net())(&state_tensor));
        //let actor = out.actor;
        //println!("out: {:?}", out);
        let probs = self.ppo_dist(&info_set, &out)?;
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
        #[cfg(feature = "trace")]
        log::trace!("Starting training PPO.");

        let device = self.ppo_network().device();
        let capacity_estimate = sum_trajectories_steps(&trajectories);

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





        let mut rng = rand::thread_rng();

        let tmp_capacity_estimate = find_max_trajectory_len(&trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);

        /*let mut action_masks_vec = (0..self.action_conversion_context().param_dimension_size())
            .map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

         */
        let mut action_masks_vec = Self::NetworkOutput::new_batch_with_capacity(action_params ,capacity_estimate);
        let mut multi_action_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut multi_action_cat_mask_tensor_vec = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);


        // batch dimenstion x param dimension
        /*
        let mut multi_action_tensor_vec: Vec<Vec<Tensor>> = (0..self.action_conversion_context().param_dimension_size())
            .map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut multi_action_cat_mask_tensor_vec: Vec<Vec<Tensor>> = (0..self.action_conversion_context().param_dimension_size())
            .map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();


         */
        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        /*
        let mut tmp_trajectory_action_tensor_vecs: Vec<Vec<Tensor>> = Vec::new();

        let mut tmp_trajectory_action_category_mask_vecs: Vec<Vec<Tensor>> = Vec::new();
        for i in 0.. self.action_conversion_context().param_dimension_size(){
            tmp_trajectory_action_tensor_vecs.push(Vec::new());
            tmp_trajectory_action_category_mask_vecs.push(Vec::new());
        }

         */
        let mut tmp_trajectory_action_tensor_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);

        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut returns_v = Vec::new();

        #[cfg(feature = "log_debug")]
        log::debug!("Starting operations on trajectories.",);
        for t in trajectories{


            if let Some(last_step) = t.last_view_step(){
                let steps_in_trajectory = t.number_of_steps();

                tmp_trajectory_state_tensor_vec.clear();
                /*
                crate::policy::ppo::multi_discrete::vec_2d_clear_second_dim(&mut tmp_trajectory_action_tensor_vecs);
                crate::policy::ppo::multi_discrete::vec_2d_clear_second_dim(&mut tmp_trajectory_action_category_mask_vecs);
                 */
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_tensor_vecs);
                Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_category_mask_vecs);
                let final_reward_t = reward_f(&last_step);
                let critic_shape = tmp_trajectory_reward_vec.clear();
                for step in t.iter(){
                    #[cfg(feature = "log_trace")]
                    log::trace!("Adding information set tensor to single trajectory vec.",);
                    tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(&self.info_set_conversion_context())?);
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
                log::trace!("Tmp infoset shape = {:?}", information_set_t.size());
                let net_out = tch::no_grad(|| (self.ppo_network().net())(&information_set_t));
                let values_t = net_out.critic();
                #[cfg(feature = "log_trace")]
                log::trace!("Tmp values_t shape = {:?}", values_t.size());
                let rewards_t = Tensor::f_stack(&tmp_trajectory_reward_vec[..],0)?.f_to_device(device)?;

                let advantages_t = Tensor::zeros(values_t.size(), (Kind::Float, device));
                //let mut next_is_final = 1f32;
                for index in (0..t.number_of_steps()).rev()
                    .map(|i, | i as i64,){
                    //chgeck if last step
                    let (next_nonterminal, next_value) = match index == t.number_of_steps() as i64 -1{
                        true => (0.0, Tensor::zeros(values_t.f_get(0)?.size(), (Kind::Float, device))),
                        false => (1.0, values_t.f_get(index as i64+1)?)
                    };
                    let delta   = rewards_t.f_get(index)? + (next_value.f_mul_scalar(self.config().gamma)?.f_mul_scalar(next_nonterminal)?) - values_t.f_get(index)?;
                    let last_gae_lambda = delta + (self.config().gamma * self.config().gae_lambda * next_nonterminal);
                    advantages_t.f_get(index)?.f_copy_(&last_gae_lambda)?
                }
                returns_v.push(advantages_t.f_add(&values_t)?);

                state_tensor_vec.push(information_set_t);
                advantage_tensor_vec.push(advantages_t);
                #[cfg(feature = "log_trace")]
                log::trace!("tmp_trajectory_action_tensor_vecs: {:?}", tmp_trajectory_action_tensor_vecs);
                #[cfg(feature = "log_trace")]
                log::trace!("multi_action_tensor_vec: {:?}", multi_action_tensor_vec);
                Self::NetworkOutput::append_vec_batch(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs );

                Self::NetworkOutput::append_vec_batch(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs );

                /*
                crate::policy::ppo::multi_discrete::vec_2d_append_second_dim(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs);
                crate::policy::ppo::multi_discrete::vec_2d_append_second_dim(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs);
                */


            } else {
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.")
            }


        }
        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)?;
        let action_forward_masks = match self.is_action_masking_supported(){
            true => Some(Self::NetworkOutput::stack_tensor_batch(&action_masks_vec)?),
            false => None
        };
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec,)?;
        let batch_returns_t = Tensor::f_vstack(&returns_v)?;
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
        let batch_actions_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_tensor_vec)?;
        let batch_action_masks_t = Self::NetworkOutput::stack_tensor_batch(&multi_action_cat_mask_tensor_vec)?;




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
            self.ppo_batch_get_actor_critic_with_logprob_and_entropy(
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
                let minibatch_indices = Tensor::from(&indices[minibatch_start as usize..minibatch_end as usize]);
                /*
                let mini_batch_action: Vec<Tensor> = batch_actions_t.iter().map(|c|{
                    c.f_index_select(0, &minibatch_indices)
                }).collect::<Result<Vec<_>, TchError>>()?;

                let mini_batch_action_cat_mask: Vec<Tensor> = batch_action_masks_t.iter().map(|c|{
                    c.f_index_select(0, &minibatch_indices)
                }).collect::<Result<Vec<_>, TchError>>()?;

                 */

                let mini_batch_action = Self::NetworkOutput::index_select(&batch_actions_t, &minibatch_indices)?;
                let mini_batch_action_cat_mask = Self::NetworkOutput::index_select(&batch_action_masks_t, &minibatch_indices)?;
                /*
                let mini_batch_action_forward_mask = action_forward_masks.as_ref().and_then(|m|{
                   Self::NetworkOutput::index_select(m, &minibatch_indices)?
                });

                 */

                let mini_batch_action_forward_mask = match action_forward_masks{
                    None => None,
                    Some(ref m) => Some(Self::NetworkOutput::index_select(&m, &minibatch_indices)?)
                };

                let mini_batch_base_logprobs = batch_logprob_t.f_index_select(0, &minibatch_indices)?;

                let (new_logprob, entropy, newvalue) = self.ppo_batch_get_actor_critic_with_logprob_and_entropy(
                    &batch_info_sets_t.f_index_select(0, &minibatch_indices)?,
                    &mini_batch_action,
                    Some(&mini_batch_action_cat_mask),
                    None, //to add it some day
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

//impl<P:PolicyPPO<DP>, DP: DomainParameters> Policy<DP> for P