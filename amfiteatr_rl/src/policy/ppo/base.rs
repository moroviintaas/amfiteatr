use tch::{Kind, TchError, Tensor};
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigPPO, find_max_trajectory_len, sum_trajectories_steps};
use crate::tensor_data::{ActionTensorFormat, ContextTryIntoTensor, ConversionToTensor};
use crate::torch_net::{ActorCriticOutput, NeuralNet};

pub trait PolicyPPO<DP: DomainParameters>
{
    type InfoSet: InformationSet<DP> + ContextTryIntoTensor<Self::InfoSetConversionContext>;
    type InfoSetConversionContext: ConversionToTensor;
    type ActionConversionContext: ActionTensorFormat<TensorForm = <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>;

    type NetworkOutput: ActorCriticOutput;

    fn config(&self) -> &ConfigPPO;

    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput>;

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext;
    fn action_conversion_context(&self) -> &Self::ActionConversionContext;

    /// Uses information set (state) and network output to calculate (masked) action distribution(s).
    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput)
        -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>>;

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
        action_param_batches: &Vec<Tensor>,
        action_category_mask_batches: Option<&Vec<Tensor>>,
        action_forward_mask_batches: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>;
    fn ppo_select_action(&self, info_set: &Self::InfoSet) -> Result<DP::ActionType, AmfiteatrError<DP>>{
        let state_tensor = info_set.to_tensor(self.info_set_conversion_context());
        let out = tch::no_grad(|| (self.ppo_network().net())(&state_tensor));
        //let actor = out.actor;


        let probs = self.ppo_dist(&info_set, &out)?;
        let choices = match self.ppo_exploration(){
            true => Self::ActionConversionContext::perform_choice(&probs, |t| t.f_multinomial(1, true))?,
              //  probs.into_iter().map(|t| t.multinomial(1, true)).collect(),

            false => Self::ActionConversionContext::perform_choice(&probs, |t| t.f_argmax(None, false)?.f_unsqueeze(-1))?,
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

        let device = self.ppo_network().device();
        let capacity_estimate = sum_trajectories_steps(&trajectories);

        let mut rng = rand::thread_rng();

        let tmp_capacity_estimate = find_max_trajectory_len(&trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        // batch dimenstion x param dimension
        let mut multi_action_tensor_vec: Vec<Vec<Tensor>> = (0..self.action_conversion_context().param_dimension_size())
            .map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut multi_action_cat_mask_tensor_vec: Vec<Vec<Tensor>> = (0..self.action_conversion_context().param_dimension_size())
            .map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_action_tensor_vecs: Vec<Vec<Tensor>> = Vec::new();

        let mut tmp_trajectory_action_category_mask_vecs: Vec<Vec<Tensor>> = Vec::new();
        for i in 0.. self.action_conversion_context().param_dimension_size(){
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

                tmp_trajectory_state_tensor_vec.clear();
                crate::policy::ppo::multi_discrete::vec_2d_clear_second_dim(&mut tmp_trajectory_action_tensor_vecs);
                crate::policy::ppo::multi_discrete::vec_2d_clear_second_dim(&mut tmp_trajectory_action_category_mask_vecs);
                //tmp_trajectory_action_category_mask_vecs.clear();
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
                    Self::ActionConversionContext::push_to_vec_batch(&mut tmp_trajectory_action_tensor_vecs, act_t);
                    Self::ActionConversionContext::push_to_vec_batch(&mut tmp_trajectory_action_category_mask_vecs, cat_mask_t);
                    //crate::policy::ppo::multi_discrete::vec_2d_push_second_dim(&mut tmp_trajectory_action_tensor_vecs, act_t);
                    //crate::policy::ppo::multi_discrete::vec_2d_push_second_dim(&mut tmp_trajectory_action_category_mask_vecs, cat_mask_t);
                    //tmp_trajectory_action_tensor_vecs.push(act_t);
                    //tmp_trajectory_action_category_mask_vecs.push(cat_mask_t);
                    tmp_trajectory_reward_vec.push(reward_f(&step))
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
                for index in (0..t.number_of_steps())
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
                crate::policy::ppo::multi_discrete::vec_2d_append_second_dim(&mut multi_action_tensor_vec, &mut tmp_trajectory_action_tensor_vecs);
                crate::policy::ppo::multi_discrete::vec_2d_append_second_dim(&mut multi_action_cat_mask_tensor_vec, &mut tmp_trajectory_action_category_mask_vecs);



            } else {
                #[cfg(feature = "log_debug")]
                log::debug!("Slipping empty trajectory.")
            }


        }
        let batch_info_sets_t = Tensor::f_vstack(&state_tensor_vec)?;
        #[cfg(feature = "log_trace")]
        log::trace!("Batch infoset shape = {:?}", batch_info_sets_t.size());
        let batch_advantage_t = Tensor::f_vstack(&advantage_tensor_vec,)?;
        let batch_returns_t = Tensor::f_vstack(&returns_v)?;
        #[cfg(feature = "log_trace")]
        log::trace!("Batch returns shape = {:?}", batch_returns_t.size());

        #[cfg(feature = "log_trace")]
        log::trace!("Batch advantage shape = {:?}", batch_advantage_t.size());
        let batch_actions_t= multi_action_tensor_vec.iter().map(|cat|{
            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;
        let batch_action_masks_t= multi_action_cat_mask_tensor_vec.iter().map(|cat|{

            Tensor::f_stack(cat, 0)
        }).collect::<Result<Vec<_>, TchError>>()?;


        let batch_size = batch_info_sets_t.size()[0];
        let mut indices: Vec<i64> = (0..batch_size).collect();

        let (batch_logprob_t, _entropy, batch_values_t) = tch::no_grad(||{
            self.ppo_batch_get_actor_critic_with_logprob_and_entropy(
                &batch_info_sets_t,
                &batch_actions_t,
                Some(&batch_action_masks_t),
                None, //to add it some day
            )

        })?;
        todo!()


    }





}

//impl<P:PolicyPPO<DP>, DP: DomainParameters> Policy<DP> for P