use std::fmt::Debug;
use std::marker::PhantomData;
use getset::{Getters, Setters};
use tch::{kind, Kind};
use tch::nn::VarStore;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use crate::error::{AmfiteatrRlError, TensorRepresentationError};
use crate::policy::common::{find_max_trajectory_len, sum_trajectories_steps};
use crate::policy::LearningNetworkPolicy;
use crate::tch;
use crate::tch::nn::Optimizer;
use crate::tch::Tensor;
use crate::tensor_data::{ConversionFromMultipleTensors, ConversionFromTensor, ConversionToMultiIndexI64, ConversionToTensor, CtxTryConvertIntoMultiIndexI64, CtxTryFromMultipleTensors, CtxTryIntoTensor};
use crate::torch_net::{A2CNet, MultiDiscreteNet, NeuralNetCriticMultiActor};

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
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType> {
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

        <DP::ActionType>::try_from_tensors(&choices, &self.action_build_context)
            .map_or_else(
                |e| {
                    #[cfg(feature = "log_error")]
                    log::error!("Failed creating action from choices tensor. Error: {}. Tensor: {:?}", e, choices);
                    None
                },
                |a| Some(a)
            )

    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromMultipleTensors + ConversionToMultiIndexI64,
> LearningNetworkPolicy<DP> for PolicyPPOMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: CtxTryFromMultipleTensors<ActionBuildContext>
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

        let device = self.network.device();
        let capacity_estimate = sum_trajectories_steps(&trajectories);

        let tmp_capacity_estimate = find_max_trajectory_len(&trajectories);

        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut multi_action_tensor_vec: Vec<Vec<Tensor>> = self.action_build_context.expected_inputs_shape()
            .iter().map(|_a|Vec::<Tensor>::with_capacity(capacity_estimate)).collect();

        let mut tmp_discounted_payoff_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_state_tensor_vec = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_action_tensor_vecs = Vec::with_capacity(tmp_capacity_estimate);
        let mut tmp_trajectory_action_category_mask_vecs = Vec::with_capacity(tmp_capacity_estimate);
        //let mut tmp_trajectory_reward_vec = Vec::with_capacity(tmp_capacity_estimate);

        for t in trajectories{

            if let Some(_trace_step) = t.view_step(0){
                #[cfg(feature = "log_trace")]
                log::trace!("Training neural-network for agent {} (from first trace step entry).", _trace_step.information_set().agent_id());
            }


            if t.is_empty(){
                //no steps in this trajectory
                continue;
            }
            let steps_in_trajectory = t.number_of_steps();


            //

            tmp_trajectory_action_category_mask_vecs.clear();
            tmp_trajectory_state_tensor_vec.clear();
            tmp_trajectory_action_category_mask_vecs.clear();
            //tmp_trajectory_reward_vec.clear();
            for step in t.iter(){
                tmp_trajectory_state_tensor_vec.push(step.information_set().try_to_tensor(&self.info_set_conversion_context)?);
                let (act_t, cat_mask_t) = step.action().action_index_and_mask_tensor_vecs(&self.action_build_context)?;
                tmp_trajectory_action_tensor_vecs.push(act_t);
                tmp_trajectory_action_category_mask_vecs.push(cat_mask_t);
                //trajectory_reward_vec.push(step.reward().)
            }


            /*
            let state_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                step.information_set().to_tensor(&self.info_set_conversion_context)
            }).collect();
            */
            /*
            let (
                trajectory_state_tensor_vec,
                trajectory_action_tensor_vecs,
                trajectory_action_category_mask_vecs,
                trajectory_reward_vec,
            ): (Vec<Tensor>, Vec<Vec<Tensor>>, Vec<Vec<Tensor>>, Vec<Tensor>) = t.iter().map(|step|{

                let (action_tensor, category_mask_tensor) = step.action().action_index_and_mask_tensor_vecs()?;
                (
                    step.information_set().try_to_tensor(&self.info_set_conversion_context),
                )



            }).collect();



            let trajectory_state_tensor = Tensor::f_stack(state_tensor_vec_t.as_mut_slice(), 0)?
                .f_to_device(self.network.device())?;



             */



            /*
            // HERE
            let mut action_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                step.action().try_to_tensor().map(|t| t.to_kind(kind::Kind::Int64))
            }).collect::<Result<Vec<Tensor>, TensorRepresentationError>>()?;

            //let final_score_t: Tensor =  t.list().last().unwrap().subjective_score_after().to_tensor();
            let final_score_t: Tensor =   reward_f(&t.last_view_step().unwrap());

            discounted_payoff_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_payoff_tensor_vec.push(Tensor::zeros(final_score_t.size(), (self.config.tensor_kind, self.network.device())));
            }
            #[cfg(feature = "log_trace")]
            log::trace!("Discounted_rewards_tensor_vec len before inserting: {}", discounted_payoff_tensor_vec.len());
            //let mut discounted_rewards_tensor_vec: Vec<Tensor> = vec![Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())); steps_in_trajectory+1];
            #[cfg(feature = "log_trace")]
            log::trace!("Reward stream: {:?}", t.iter().map(|x| reward_f(&x)).collect::<Vec<Tensor>>());
            //discounted_payoff_tensor_vec.last_mut().unwrap().copy_(&final_score_t);
            for s in (0..discounted_payoff_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let this_reward = reward_f(&t.view_step(s).unwrap()).to_device(device);
                let r_s = &this_reward + (&discounted_payoff_tensor_vec[s+1] * self.config.gamma);
                discounted_payoff_tensor_vec[s].copy_(&r_s);
                #[cfg(feature = "log_trace")]
                log::trace!("Calculating discounted payoffs for {} step. This step reward {}, following payoff: {}, result: {}.",
                    s, this_reward, discounted_payoff_tensor_vec[s+1], r_s);
            }
            discounted_payoff_tensor_vec.pop();
            #[cfg(feature = "log_trace")]
            log::trace!("Discounted future payoffs tensor: {:?}", discounted_payoff_tensor_vec);
            #[cfg(feature = "log_trace")]
            log::trace!("Discounted rewards_tensor_vec after inserting");

            state_tensor_vec.append(&mut state_tensor_vec_t);
            action_tensor_vec.append(&mut action_tensor_vec_t);
            reward_tensor_vec.append(&mut discounted_payoff_tensor_vec);


             */
        }
        todo!()





    }
}
