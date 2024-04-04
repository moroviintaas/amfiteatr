use std::fmt::Debug;
use std::marker::PhantomData;
use tch::Kind::{Float};
use tch::nn::{Optimizer, VarStore};
use tch::{Kind, kind, Tensor};
use amfiteatr_core::agent::{
    InformationSet,
    Policy,
    EvaluatedInformationSet,
    AgentStepView,
    AgentTrajectory
};
use amfiteatr_core::domain::DomainParameters;

use crate::error::{AmfiteatrRlError, TensorRepresentationError};
use crate::policy::LearningNetworkPolicy;
use crate::tensor_data::{CtxTryIntoTensor, ConversionToTensor, TryIntoTensor, TryFromTensor};
use crate::torch_net::{A2CNet, TensorA2C};
use crate::policy::TrainConfig;

/// Generic implementation of Advantage Actor Critic policy
pub struct ActorCriticPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
   // ActionConversionContext: ConversionFromTensor,
> {
    network: A2CNet,
    #[allow(dead_code)]
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    //state_converter: StateConverter,
    info_set_conversion_context: InfoSetConversionContext,
    //action_conversion_context: ActionConversionContext,
    training_config: TrainConfig,
    exploration: bool
    //action_interpreter: ActInterpreter

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP>  + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    //ActionConversionContext: ConversionFromTensor,
    >
ActorCriticPolicy<
    DP,
    InfoSet,
    InfoSetConversionContext,
    //ActionConversionContext
    > {
    /// ```
    /// use tch::{Device, nn, Tensor};
    /// use tch::nn::{Adam, VarStore};
    /// use amfiteatr_core::demo::{DemoDomain, DemoInfoSet};
    /// use amfiteatr_rl::policy::ActorCriticPolicy;
    /// use amfiteatr_rl::demo::DemoConversionToTensor;
    /// use amfiteatr_rl::torch_net::{A2CNet, TensorA2C};
    /// use amfiteatr_rl::policy::TrainConfig;
    /// let var_store = VarStore::new(Device::Cpu);
    /// let neural_net = A2CNet::new(var_store, |path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 1, 128, Default::default()))
    ///         .add(nn::linear(path / "hidden", 128, 128, Default::default()));
    ///     let actor = nn::linear(path / "al", 128, 2, Default::default());
    ///     let critic = nn::linear(path / "cl", 128, 1, Default::default());
    ///     let device = path.device();
    ///     {move |xs: &Tensor|{
    ///         let xs = xs.to_device(device).apply(&seq);
    ///         TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
    ///     }}
    ///
    /// });
    /// let optimizer = neural_net.build_optimizer(Adam::default(), 0.01).unwrap();
    ///
    /// let policy: ActorCriticPolicy<DemoDomain, DemoInfoSet, DemoConversionToTensor>
    ///     = ActorCriticPolicy::new(neural_net, optimizer, DemoConversionToTensor{}, TrainConfig { gamma: 0.99 });
    /// ```
    pub fn new(network: A2CNet,
               optimizer: Optimizer,
               info_set_conversion_context: InfoSetConversionContext,
               //action_conversion_context: ActionConversionContext,
               training_config: TrainConfig
               //state_converter: StateConverter,
               /*action_interpreter: ActInterpreter*/
    ) -> Self{
        Self{
            network, optimizer,
            //state_converter,
            info_set_conversion_context,
            //action_conversion_context,

            training_config,
            //action_interpreter,
            _dp: Default::default(), _is: Default::default(),
            exploration: true
            }
    }






}

impl<DP: DomainParameters,
    //InfoSet: InformationSet<DP> + Debug,
    //TB: ConvStateToTensor<InfoSet>,
    InfoSet: InformationSet<DP> + Debug + CtxTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    //ActionConversionContext: ConversionFromTensor,
> Policy<DP> for ActorCriticPolicy<
    DP,
    InfoSet,
    //TB,
    InfoSetConversionContext,
    //ActionConversionContext,
>
where <DP as DomainParameters>::ActionType: TryFromTensor{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType> {
        //let state_tensor = self.state_converter.build_tensor(state)
        //    .unwrap_or_else(|_| panic!("Failed converting state to Tensor: {:?}", state));
        //let state_tensor = self.state_converter.make_tensor(state);
        #[cfg(feature = "log_trace")]
        log::trace!("Selecting action");
        let state_tensor = state.to_tensor(&self.info_set_conversion_context);
        let out = tch::no_grad(|| (self.network.net())(&state_tensor));
        let actor = out.actor;
        //somewhen it may be changed with temperature
        let probs = actor.softmax(-1, Float);
        if self.exploration{
            let atensor = probs.multinomial(1, true);
            #[cfg(feature = "log_trace")]
            log::trace!("After selecting action, before converting from tensor to action form");
            //self.action_interpreter.interpret_tensor(&atensor)
            Some(DP::ActionType::try_from_tensor(&atensor)
                .expect("Failed converting tensor to action"))
        } else {
            let atensor = probs.argmax(None, false).unsqueeze(-1);
            Some(DP::ActionType::try_from_tensor(&atensor)
                .expect("Failed converting tensor to action (no explore)"))
        }


    }
}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP>  + Debug + CtxTryIntoTensor<InfoSetWay>,
    InfoSetWay: ConversionToTensor,
    //InfoSet: ScoringInformationSet<DP> + Debug,
    //StateConverter: ConvStateToTensor<InfoSet>>
    > LearningNetworkPolicy<DP> for ActorCriticPolicy<DP, InfoSet, InfoSetWay>
    where <DP as DomainParameters>::ActionType: TryFromTensor + TryIntoTensor,
//<InfoSet as ScoringInformationSet<DP>>::RewardType: FloatTensorReward
{
    type Network = A2CNet;
    type TrainConfig = TrainConfig;


    fn network(&self) -> &A2CNet{
        &self.network
    }


    fn network_mut(&mut self) -> &mut A2CNet{
        &mut self.network
    }

    /// Returns reference to underlying [`VarStore`]
    fn var_store(&self) -> &VarStore{
        self.network.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore{
        self.network.var_store_mut()
    }

    /// For now A2C always explore and switching it off is pointless, in future it will probably
    /// select maximal probability without sampling distribution
    fn switch_explore(&mut self, _enabled: bool) {
        self.exploration = _enabled

    }

    /*
    fn enable_exploration(&mut self, enable: bool) {
        self.exploration = enable
    }

     */


    fn config(&self) -> &Self::TrainConfig {
        &self.training_config
    }

    fn train_on_trajectories<R: Fn(&AgentStepView<DP, InfoSet>) -> Tensor>(

        &mut self,
        trajectories: &[AgentTrajectory<DP, InfoSet>],
        reward_f: R,
        ) -> Result<(), AmfiteatrRlError<DP>>{


        let device = self.network.device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
           acc + x.completed_len()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.completed_len()
        }).max().unwrap_or(0);
        let mut state_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut action_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut discounted_payoff_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        for t in trajectories{

            if let Some(_trace_step) = t.view_step(0){
                #[cfg(feature = "log_trace")]
                log::trace!("Training neural-network for agent {} (from first trace step entry).", _trace_step.info_set().agent_id());
            }


            if t.is_empty(){
                continue;
            }
            let steps_in_trajectory = t.completed_len();

            let mut state_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                //self.state_converter.make_tensor(step.step_state())
                step.info_set().to_tensor(&self.info_set_conversion_context)
            }).collect();

            let mut action_tensor_vec_t: Vec<Tensor> = t.iter().map(|step|{
                step.action().try_to_tensor().map(|t| t.to_kind(kind::Kind::Int64))
            }).collect::<Result<Vec<Tensor>, TensorRepresentationError>>()?;

            //let final_score_t: Tensor =  t.list().last().unwrap().subjective_score_after().to_tensor();
            let final_score_t: Tensor =   reward_f(&t.last_view_step().unwrap());

            discounted_payoff_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_payoff_tensor_vec.push(Tensor::zeros(final_score_t.size(), (Kind::Float, self.network.device())));
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
                let r_s = &this_reward + (&discounted_payoff_tensor_vec[s+1] * self.training_config.gamma);
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

        }
        if state_tensor_vec.is_empty(){
            #[cfg(feature = "log_warn")]
            log::warn!("There were trajectories registered but no steps in any");
            return Err(AmfiteatrRlError::NoTrainingData);
        }
        let states_batch = Tensor::stack(&state_tensor_vec[..], 0).to_device(device);
        let results_batch = Tensor::stack(&reward_tensor_vec[..], 0).to_device(device);
        let action_batch = Tensor::stack(&action_tensor_vec[..], 0).to_device(device);
        #[cfg(feature = "log_debug")]
        log::debug!("Size of states batch: {:?}", states_batch.size());
        #[cfg(feature = "log_debug")]
        log::debug!("Size of result batch: {:?}", results_batch.size());
        #[cfg(feature = "log_debug")]
        log::debug!("Size of action batch: {:?}", action_batch.size());
        #[cfg(feature = "log_trace")]
        log::trace!("State batch: {:?}", states_batch);
        #[cfg(feature = "log_trace")]
        log::trace!("Result batch: {:?}", results_batch);
        #[cfg(feature = "log_trace")]
        log::trace!("Action batch: {:?}", action_batch);
        let TensorA2C{actor, critic} = (self.network.net())(&states_batch);
        let log_probs = actor.log_softmax(-1, Kind::Float);
        let probs = actor.softmax(-1, Float);
        let action_log_probs = {
            let index =  action_batch.to_device(self.network.device());
            //trace!("Index: {}", index);
            log_probs.gather(1, &index, false)
        };

        #[cfg(feature = "log_trace")]
        log::trace!("Action log probs size: {:?}", action_log_probs.size());
        #[cfg(feature = "log_trace")]
        log::trace!("Probs size: {:?}", probs.size());

        let dist_entropy = (-log_probs * probs).sum_dim_intlist(-1, false, Float).mean(Float);
        let advantages = results_batch.to_device(device) - critic;
        let value_loss = (&advantages * &advantages).mean(Float);
        let action_loss = (-advantages.detach() * action_log_probs).mean(Float);
        self.optimizer.zero_grad();
        let loss = value_loss * 0.5 + action_loss - dist_entropy * 0.01;
        self.optimizer.backward_step_clip(&loss, 0.5);

        Ok(())
    }
}
/*
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<InfoSetWay>,
    InfoSetWay: ConversionToTensor,
    //InfoSet: InformationSet<DP> + Debug,
    //TB: ConvStateToTensor<InfoSet>,
    >
SelfExperiencingPolicy<DP> for ActorCriticPolicy<
    DP,
    InfoSet,
    //TB,
    InfoSetWay,
    /*ActInterpreter*/>
where DP::ActionType: From<i64>{
    type PolicyUpdateError = tch::TchError;

    fn select_action_and_collect_experience(&mut self) -> Option<DP::ActionType> {
        todo!()
    }


    fn apply_experience(&mut self) -> Result<(), Self::PolicyUpdateError> {
        todo!()
    }
}


 */

