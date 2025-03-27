use getset::{Getters, Setters};
use tch::nn::Optimizer;
use tch::{kind, Kind, Tensor};
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;
use crate::error::{AmfiteatrRlError, TensorRepresentationError};
use crate::tensor_data::{ActionTensorFormat, ContextEncodeTensor, TensorEncoding};
use crate::torch_net::{ActorCriticOutput, NeuralNet};

/// Configuration structure for A2C
#[derive(Copy, Clone, Debug, Getters, Setters)]
pub struct ConfigA2C{
    pub gamma: f64,
    pub mini_batch_size: Option<usize>,
}

impl Default for ConfigA2C {
    fn default() -> Self {
        Self{
            gamma: 0.99,
            mini_batch_size: Some(16),
        }
    }
}


pub trait PolicyHelperA2C<DP: DomainParameters>{

    type InfoSet: InformationSet<DP> + ContextEncodeTensor<Self::InfoSetConversionContext>;
    type InfoSetConversionContext: TensorEncoding;
    type ActionConversionContext: ActionTensorFormat<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>;
    type NetworkOutput: ActorCriticOutput;

    type Config;


    fn config(&self) -> &Self::Config;

    fn optimizer_mut(&mut self) -> &mut Optimizer;

    fn network(&self) ->  &NeuralNet<Self::NetworkOutput>;

    /// Return tensor encoding context for information set
    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext;

    /// Returns tensor index decoding and encoding for action.
    fn action_conversion_context(&self) -> &Self::ActionConversionContext;

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
        let state_tensor = info_set.to_tensor(self.info_set_conversion_context());
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

}

pub trait PolicyTrainHelperA2C<DP: DomainParameters> : PolicyHelperA2C<DP, Config=ConfigA2C>{

    fn a2c_train_on_trajectories<
        R: Fn(&AgentStepView<DP, Self::InfoSet>) -> Tensor>
    (
        &mut self, trajectories: &[AgentTrajectory<DP, Self::InfoSet>],
        reward_f: R
    ) -> Result<(), AmfiteatrRlError<DP>>{

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

        let sample_info_set_t = sample_info_set.try_to_tensor(self.info_set_conversion_context())?;
        let sample_net_output = tch::no_grad(|| self.network().net()(&sample_info_set_t));
        let action_params = sample_net_output.param_dimension_size() as usize;


        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut advantage_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);

        let mut action_masks_vec = Self::NetworkOutput::new_batch_with_capacity(action_params ,capacity_estimate);



        let mut tmp_trajectory_action_tensor_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Self::NetworkOutput::new_batch_with_capacity(action_params, capacity_estimate);

        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut discounted_payoff_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate+1);

        for t in trajectories {
            let steps_in_trajectory = t.number_of_steps();

            t.view_step(0).inspect(|t|{
                #[cfg(feature = "log_trace")]
                log::trace!("Training neural-network for agent {} (from first trace step entry).", t.information_set().agent_id());

            });

            if t.last_view_step().is_none(){
                continue;
            }
            let final_score_t =   reward_f(&t.last_view_step().unwrap());

            tmp_trajectory_state_tensor_vec.clear();
            Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_tensor_vecs);
            Self::NetworkOutput::clear_batch_dim_in_batch(&mut tmp_trajectory_action_category_mask_vecs);
            tmp_trajectory_reward_vec.clear();
            for step in t.iter(){
                #[cfg(feature = "log_trace")]
                log::trace!("Adding information set tensor to single trajectory vec.",);
                tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(self.info_set_conversion_context())?);
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
            let information_set_t = Tensor::f_stack(&tmp_trajectory_state_tensor_vec[..],0)?.f_to_device(device)?;
            #[cfg(feature = "log_trace")]
            log::trace!("Tmp infoset shape = {:?}", information_set_t.size());
            let net_out = tch::no_grad(|| (self.network().net())(&information_set_t));
            let values_t = net_out.critic();
            #[cfg(feature = "log_trace")]
            log::trace!("Tmp values_t shape = {:?}", values_t.size());
            let rewards_t = Tensor::f_stack(&tmp_trajectory_reward_vec[..],0)?.f_to_device(device)?;

            let advantages_t = Tensor::zeros(values_t.size(), (Kind::Float, device));

            discounted_payoff_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_payoff_tensor_vec.push(Tensor::zeros(final_score_t.size(), (Kind::Float, device)));

            }

            for s in (0..discounted_payoff_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let this_reward = reward_f(&t.view_step(s).unwrap()).to_device(device);
                let r_s = &this_reward + (&discounted_payoff_tensor_vec[s+1] * self.config().gamma);
                discounted_payoff_tensor_vec[s].copy_(&r_s);
                #[cfg(feature = "log_trace")]
                log::trace!("Calculating discounted payoffs for {} step. This step reward {}, following payoff: {}, result: {}.",
                    s, this_reward, discounted_payoff_tensor_vec[s+1], r_s);
            }
            discounted_payoff_tensor_vec.pop();

            state_tensor_vec.push(information_set_t);



            /*

            let steps_in_trajectory = t.number_of_steps();

            let mut state_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                step.information_set().try_to_tensor(&self.info_set_conversion_context())
            }).collect();

            let mut action_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                step.action().try_to_tensor().map(|t| t.to_kind(kind::Kind::Int64))
            }).collect::<Result<Vec<Tensor>, TensorRepresentationError>>()?;

             */
        }

        todo!()
    }

}

impl<T, DP: DomainParameters> PolicyTrainHelperA2C<DP> for T
    where T: PolicyHelperA2C<DP, Config=ConfigA2C>{}


