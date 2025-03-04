use std::cmp::min;
use std::fmt::Debug;
use std::marker::PhantomData;
use getset::{Getters, Setters};
use rand::seq::SliceRandom;
use tch::{kind, Device, Kind, TchError};
use tch::Kind::Float;
use tch::nn::VarStore;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::error::{AmfiteatrRlError, TensorRepresentationError};
use crate::policy::common::{find_max_trajectory_len, sum_trajectories_steps};
use crate::policy::LearningNetworkPolicy;
use crate::tch;
use crate::tch::nn::Optimizer;
use crate::tch::Tensor;
use crate::tensor_data::{ConversionFromMultipleTensors, ConversionFromTensor, ConversionToMultiIndexI64, ConversionToTensor, CtxTryConvertIntoMultiIndexI64, CtxTryFromMultipleTensors, CtxTryIntoTensor};
use crate::torch_net::{A2CNet, MultiDiscreteNet, NeuralNetCriticMultiActor, TensorCriticMultiActor};

///! Based on [cleanrl PPO](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py)
#[derive(Copy, Clone, Debug, Getters, Setters)]
pub struct ConfigPPOMultiDiscrete{
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

impl Default for ConfigPPOMultiDiscrete{
    fn default() -> ConfigPPOMultiDiscrete{
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

pub struct PolicyPPOMultiDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors,
>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext>
{
    config: ConfigPPOMultiDiscrete,
    network: NeuralNetCriticMultiActor,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_conversion_context: InfoSetConversionContext,
    action_build_context: ActionBuildContext,

    exploration: bool,




}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors,
> PolicyPPOMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext>{

    fn batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &[Tensor],
        action_category_mask_batches: Option<&[Tensor]>,
        action_forward_mask_batches: Option<&[Tensor]>,
    ) -> Result<(Tensor, Tensor, Tensor), TensorRepresentationError>{

        let critic_actor= (&self.network.net())(info_set_batch);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<DP>(
            action_param_batches,
            action_category_mask_batches
        ).unwrap();
        let batch_entropy = critic_actor.batch_entropy_masked(
            action_forward_mask_batches,
            action_category_mask_batches

        )?;

        let batch_entropy_avg = batch_entropy.f_sum_dim_intlist(
            Some(1),
            false,
            Kind::Float
        )?.f_div_scalar(batch_entropy.size()[1])?;
        //println!("batch entropy: {}", batch_entropy);
        //println!("batch entropy avg: {}", batch_entropy_avg);

        Ok((batch_logprob, batch_entropy_avg, critic_actor.critic))
    }

}

fn vec_2d_clear_second_dim<T>(v: &mut Vec<Vec<T>>){
    for c in v.iter_mut(){
        c.clear()
    }
}

fn vec_2d_append_second_dim<T>(v: &mut Vec<Vec<T>>, append: &mut Vec<Vec<T>>){
    for (c_append, c_base) in append.iter_mut().zip(v.iter_mut()){
        c_base.append(c_append)
    }
}
fn vec_2d_push_second_dim<T>(v: &mut Vec<Vec<T>>, append: Vec<T>){
    for (c_push, c_base) in append.into_iter().zip(v.iter_mut()){
        c_base.push(c_push)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors,
>
PolicyPPOMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext>
{
    pub fn new(
        config: ConfigPPOMultiDiscrete,
        network: NeuralNetCriticMultiActor,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
        tensor_kind: tch::kind::Kind,

    ) -> Self{
        Self{
            config,
            network,
            optimizer,
            _dp: Default::default(),
            _is: Default::default(),
            info_set_conversion_context,
            action_build_context,
            exploration: true,
        }
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors,
>
Policy<DP> for PolicyPPOMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext, ConvertError: Into<AmfiteatrError<DP>>>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        let state_tensor = state.to_tensor(&self.info_set_conversion_context);
        //log::debug!("Info set tensor kind: {:?}", state_tensor.kind());
        let out = tch::no_grad(|| (self.network.net())(&state_tensor));
        let actor = out.actor;
        let probs = actor.iter()
            .map(|t| t.softmax(-1, self.config.tensor_kind));
        let choices: Vec<Tensor> = match self.exploration{
            true => probs.map(|t| t.multinomial(1, true)).collect(),

            false => probs.map(|t| t.argmax(None, false).unsqueeze(-1)).collect()
        };

        <DP::ActionType>::try_from_tensors(&choices, &self.action_build_context).map_err(|err| {
            #[cfg(feature = "log_error")]
            log::error!("Failed creating action from choices tensor. Error: {}. Tensor: {:?}", err, choices);
            err.into()
        })
        /*
        Ok(<DP::ActionType>::try_from_tensors(&choices, &self.action_build_context)
            .map_or_else(
                |e| {
                    #[cfg(feature = "log_error")]
                    log::error!("Failed creating action from choices tensor. Error: {}. Tensor: {:?}", e, choices);
                    None
                },
                |a| a
            ))

         */

    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors + ConversionToMultiIndexI64,
> LearningNetworkPolicy<DP> for PolicyPPOMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext, ConvertError: Into<AmfiteatrError<DP>>>
    + CtxTryConvertIntoMultiIndexI64<ActionBuildContext>
{
    fn var_store(&self) -> &VarStore {
        &self.network.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.exploration = enabled;
    }



    fn train_on_trajectories<
        R: Fn(&AgentStepView<DP,
            <Self as Policy<DP>>::InfoSetType>) -> Tensor
    >
    (
        &mut self, trajectories: &[AgentTrajectory<DP,
        <Self as Policy<DP>>::InfoSetType>],
        reward_f: R
    ) -> Result<(), AmfiteatrRlError<DP>> {
        #[cfg(feature = "log_debug")]
        log::debug!("Beginning training with {} trajectories.", trajectories.len());

        let device = self.network.device();
        let capacity_estimate = sum_trajectories_steps(&trajectories);

        let mut rng = rand::thread_rng();

        let tmp_capacity_estimate = find_max_trajectory_len(&trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        // batch dimenstion x param dimension
        let mut multi_action_tensor_vec: Vec<Vec<Tensor>> = self.action_build_context.expected_inputs_shape()
            .iter().map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut multi_action_cat_mask_tensor_vec = self.action_build_context.expected_inputs_shape()
            .iter().map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut tmp_discounted_payoff_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_action_tensor_vecs: Vec<Vec<Tensor>> = Vec::new();

        let mut tmp_trajectory_action_category_mask_vecs: Vec<Vec<Tensor>> = Vec::new();
        for i in 0..ActionBuildContext::number_of_params(){
            tmp_trajectory_action_tensor_vecs.push(Vec::new());
            tmp_trajectory_action_category_mask_vecs.push(Vec::new());
        }
        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        let mut returns_v = Vec::new();

        #[cfg(feature = "log_debug")]
        log::debug!("Starting operations on trajectories.",);
        for t in trajectories{


            if let Some(last_step) = t.last_view_step(){
                let steps_in_trajectory = t.number_of_steps();

                //tmp_trajectory_action_category_mask_vecs.clear();
                tmp_trajectory_state_tensor_vec.clear();
                vec_2d_clear_second_dim(&mut tmp_trajectory_action_tensor_vecs);
                vec_2d_clear_second_dim(&mut tmp_trajectory_action_category_mask_vecs);
                //tmp_trajectory_action_category_mask_vecs.clear();
                let final_reward_t = reward_f(&last_step);
                let critic_shape =

                /*
                let reward_shape = final_reward_t.size();
                let state_shape = last_step.information_set()
                    .try_to_tensor(&self.info_set_conversion_context)?.size();



                 */
                tmp_trajectory_reward_vec.clear();
                for step in t.iter(){
                    #[cfg(feature = "log_trace")]
                    log::trace!("Adding information set tensor to single trajectory vec.",);
                    tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(&self.info_set_conversion_context)?);
                    #[cfg(feature = "log_trace")]
                    log::trace!("Added information set tensor to single trajectory vec.",);
                    let (act_t, cat_mask_t) = step.action().action_index_and_mask_tensor_vecs(&self.action_build_context)?;

                    vec_2d_push_second_dim(&mut tmp_trajectory_action_tensor_vecs, act_t);
                    vec_2d_push_second_dim(&mut tmp_trajectory_action_category_mask_vecs, cat_mask_t);
                    //tmp_trajectory_action_tensor_vecs.push(act_t);
                    //tmp_trajectory_action_category_mask_vecs.push(cat_mask_t);
                    tmp_trajectory_reward_vec.push(reward_f(&step))
                }


                let information_set_t = Tensor::f_stack(&tmp_trajectory_state_tensor_vec[..],0)?.f_to_device(device)?;
                #[cfg(feature = "log_trace")]
                log::trace!("Tmp infoset shape = {:?}", information_set_t.size());
                let values_t = tch::no_grad(|| (self.network.net())(&information_set_t)).critic;

                let rewards_t = Tensor::f_stack(&tmp_trajectory_reward_vec[..],0)?.f_to_device(device)?;

                let advantages_t = Tensor::zeros(values_t.size(), (Kind::Float, device));
                //let mut next_is_final = 1f32;
                for index in (0..t.number_of_steps())
                    .map(|i, | i as i64,){
                    //chgeck if last step
                    let (next_nonterminal, next_value) = match index == t.number_of_steps() as i64 -1{
                        true => (0.0, Tensor::zeros(values_t.f_get(0)?.size(), (Kind::Float, device))),
                        false => (1.0, values_t.f_get(index as i64+1)?)
                    };
                    let delta   = rewards_t.f_get(index)? + (next_value.f_mul_scalar(self.config.gamma)?.f_mul_scalar(next_nonterminal)?) - values_t.f_get(index)?;
                    let last_gae_lambda = delta + (self.config.gamma * self.config.gae_lambda * next_nonterminal);
                    advantages_t.f_get(index)?.f_copy_(&last_gae_lambda)?
                }
                returns_v.push(advantages_t.f_add(&values_t)?);

                state_tensor_vec.push(information_set_t);
                advantage_tensor_vec.push(advantages_t);
                #[cfg(feature = "log_trace")]
                log::trace!("tmp_trajectory_action_tensor_vecs[0] = {:?}", tmp_trajectory_action_tensor_vecs[0].len());
                vec_2d_append_second_dim(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs);
                vec_2d_append_second_dim(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs);



            } else {
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.")
            }










        }
        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)?;
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec)?;
        let batch_returns_t = Tensor::f_vstack(&returns_v)?;
        #[cfg(feature = "log_trace")]
        log::trace!("Batch returns shape = {:?}", batch_returns_t.size());
        #[cfg(feature = "log_trace")]
        log::trace!("Batch advantage shape = {:?}", batch_advantage_t.size());
        let batch_actions_t= multi_action_tensor_vec.iter().map(|cat|{
            #[cfg(feature = "log_trace")]
            log::trace!("Cat[0] = {}, size = {:?}", cat[0], cat[0].size());
            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;
        #[cfg(feature = "log_trace")]
        log::trace!("BCat[0] = {}", batch_actions_t[0]);
        let batch_action_masks_t= multi_action_cat_mask_tensor_vec.iter().map(|cat|{

            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;

        println!("info_sets: {:?}", batch_info_sets_t.size());
        println!("advantages: {:?}", batch_advantage_t.size());
        println!("actions: {:?}", batch_actions_t[0].size());

        for (index, cat) in batch_actions_t.iter().enumerate(){
            println!("Category: {index:}, {:?}", cat.size());
        }

        let batch_size = batch_info_sets_t.size()[0];
        let mut indices: Vec<i64> = (0..batch_size).collect();

        /*
        let (batch_logprob_t, _entropy, _batch_value) = self.batch_get_actor_critic_with_logprob_and_entropy(
            &batch_info_sets_t,
            &batch_actions_t,
            Some(&batch_action_masks_t),
            None, //to add it some day
        )?;
        
         */

        let (batch_logprob_t, _entropy, batch_values_t) = tch::no_grad(||{
            self.batch_get_actor_critic_with_logprob_and_entropy(
                &batch_info_sets_t,
                &batch_actions_t,
                Some(&batch_action_masks_t),
                None, //to add it some day
            )

        })?;


        //let mut clip_fracs = vec![];
        for epoch in 0..self.config.update_epochs{
            #[cfg(feature = "log_debug")]
            log::debug!("PPO Update Epoch: {epoch}");

            indices.shuffle(&mut rng);
            //println!("{indices:?}")

            for minibatch_start in (0..batch_size).step_by(self.config.mini_batch_size){
                let minibatch_end = min(minibatch_start + self.config.mini_batch_size as i64, batch_size);
                let minibatch_indices = Tensor::from(&indices[minibatch_start as usize..minibatch_end as usize]);

                println!("batch_actions_t[] size = {:?}", batch_actions_t[0].size());
                let mini_batch_action: Vec<Tensor> = batch_actions_t.iter().map(|c|{
                    c.f_index_select(0, &minibatch_indices)
                }).collect::<Result<Vec<_>, TchError>>()?;

                let mini_batch_action_cat_mask: Vec<Tensor> = batch_action_masks_t.iter().map(|c|{
                    c.f_index_select(0, &minibatch_indices)
                }).collect::<Result<Vec<_>, TchError>>()?;

                let mini_batch_base_logprobs = batch_logprob_t.f_index_select(0, &minibatch_indices)?;

                let (new_logprob, entropy, newvalue) = self.batch_get_actor_critic_with_logprob_and_entropy(
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
                let pg_loss2 = -&minibatch_advantages_t.f_mul(&ratio.f_clamp(1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef)?)?;
                let pg_loss = pg_loss1.f_max_other(&pg_loss2)?.f_mean(Float)?;

                #[cfg(feature = "log_debug")]
                log::debug!("PG loss : {}", pg_loss);



                let v_loss = if self.config.clip_vloss{
                    let v_loss_unclipped = (newvalue.f_sub(&minibatch_returns_t)?).f_square()?;
                    let v_clipped =minibatch_values_t.f_add(
                       &newvalue.f_sub(&minibatch_values_t)?
                           .f_clamp(
                               - self.config.clip_coef,
                               self.config.clip_coef
                           )?
                    )?;
                    let v_loss_clipped = (v_clipped.f_sub(&minibatch_returns_t)?).f_square()?;
                    let v_loss_max = v_loss_unclipped.f_max_other(&v_loss_clipped)?;
                    v_loss_max.f_mean(Float)? * 0.5
                } else {
                    newvalue.f_sub(&minibatch_returns_t)?.f_square()?.f_mean(Float)? *0.5
                };

                let entropy_loss = entropy.f_mean(Float)?;
                let loss = pg_loss
                    .f_sub(&(entropy_loss * self.config.ent_coef))?
                    .f_add(&(v_loss * self.config.vf_coef))?;

                self.optimizer.zero_grad();

                self.optimizer.backward_step(&loss);



            }

        }

        Ok(())





    }
}
