use std::fmt::Debug;
use std::fs::File;
use std::marker::PhantomData;
use tboard::EventWriter;
use tch::nn::Optimizer;
use tch::Tensor;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::scheme::Scheme;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_core::util::TensorboardSupport;
use crate::error::AmfiteatrRlError;
use crate::policy::{
    ConfigPPO,
    LearnSummary,
    LearningNetworkPolicyGeneric,
    PolicyHelperA2C,
    PolicyTrainHelperPPO
};
use crate::{tensor_data, MaskingInformationSetAction};
use crate::tensor_data::{
    ContextEncodeIndexI64,
    ContextEncodeTensor,
    TensorDecoding,
    TensorIndexI64Encoding,
    TensorEncoding,
    ContextDecodeIndexI64
};
use crate::torch_net::{ActorCriticOutput, NeuralNet, NeuralNetActorCritic, TensorActorCritic};


/// Policy PPO for discrete action space with single distribution using [`tch`] crate for `torch` backed
/// [`Tensors`](tch::Tensor).
pub struct PolicyDiscretePPO<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding,
>{
    config: ConfigPPO,
    network: NeuralNetActorCritic,
    optimizer: Optimizer,
    _dp: PhantomData<S>,
    _is: PhantomData<InfoSet>,
    info_set_encoding: InfoSetConversionContext,
    action_encoding: ActionBuildContext,

    exploration: bool,
    tboard_writer: Option<tboard::EventWriter<File>>,
    global_step: i64,

}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
        + tensor_data::ActionTensorFormat<Tensor>,
> PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>{


    /// ```
    /// use tch::{Device, nn, Tensor};
    /// use tch::nn::{Adam, VarStore};
    /// use amfiteatr_core::demo::{DemoScheme, DemoInfoSet};
    /// use amfiteatr_rl::demo::{DemoActionConversionContext, DemoConversionToTensor};
    /// use amfiteatr_rl::policy::{ConfigPPO, PolicyDiscretePPO};
    /// use amfiteatr_rl::torch_net::{build_network_model_ac_discrete, NeuralNetActorCritic, TensorActorCritic, VariableStorage};
    /// use amfiteatr_rl::torch_net::Layer::Linear;
    /// use tch::nn::OptimizerConfig;
    /// let var_store = VarStore::new(Device::Cpu);
    /// let model = build_network_model_ac_discrete(vec![Linear(128), Linear(128)], vec![1], 2, &var_store.root());
    /// let optimizer = Adam::default().build(&var_store, 0.01).unwrap();
    /// let net = NeuralNetActorCritic::new(VariableStorage::Owned(var_store), model);
    /// let config = ConfigPPO::default();
    /// let demo_info_set_ctx = DemoConversionToTensor::default();
    /// let demo_action_ctx = DemoActionConversionContext{};
    /// let policy: PolicyDiscretePPO<DemoScheme, DemoInfoSet, DemoConversionToTensor, DemoActionConversionContext> =
    ///     PolicyDiscretePPO::new(
    ///     config, net, optimizer, demo_info_set_ctx, demo_action_ctx);
    /// ```
    pub fn new(
        config: ConfigPPO,
        network: NeuralNetActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            config,
            network,
            optimizer,
            _dp: Default::default(),
            _is: Default::default(),
            info_set_encoding: info_set_conversion_context,
            action_encoding: action_build_context,
            exploration: true,
            tboard_writer: None,
            global_step: 0
        }
    }


    /*
    /// Returns three tensors for respectively log probability of each action, entropy of this distribution
    /// and critic value.
    /// Tensor sizes:
    /// 1. BATCH_SIZE x ACTION_SPACE..
    /// 2. BATCH_SIZE x 1
    /// 3. BATCH_SIZE x1
    fn batch_get_logprob_entropy_critic(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &Tensor,
        action_category_mask_batches: Option<&Tensor>,
        action_forward_mask_batches: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>>{

        let critic_actor= self.network.net()(info_set_batch);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<S>(
            action_param_batches,
            action_forward_mask_batches,
            action_category_mask_batches
        )?;
        let batch_entropy = critic_actor.batch_entropy_masked(
            action_forward_mask_batches,
            action_category_mask_batches

        ).map_err(|e|AmfiteatrError::Tensor {
            error: TensorError::Torch {
                context: "batch_get_actor_critic_with_logprob_and_entropy".into(),
                origin: format!("{e}")
            }
        })?;



        Ok((batch_logprob, batch_entropy, critic_actor.critic))
    }
    
     */
    /*

    pub fn var_store(&self) -> &VarStore {
        self.network.var_store()
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

     */
}

impl <
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding,
> TensorboardSupport<S> for PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>{

    /// Creates [`tboard::EventWriter`]. Initially policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    fn add_tboard_directory(&mut self, directory_path: &std::path::Path) -> Result<(), AmfiteatrError<S>> {
        let tboard = EventWriter::create(directory_path).map_err(|e| {
            AmfiteatrError::TboardFlattened { context: "Creating tboard EventWriter".into(), error: format!("{e}") }
        })?;
        self.tboard_writer = Some(tboard);
        Ok(())
    }
    fn t_write_scalar(&mut self, step: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>> {
        match &mut self.tboard_writer{
            None => Ok(false),
            Some(writer) => {
                writer.write_scalar(step, tag, value)
                    .map_err(|e| AmfiteatrError::TboardFlattened {
                        context: "Tboard - writing scalar (PPO)".to_string(),
                        error: e.to_string(),
                    })?;
                Ok(true)
            }
        }
    }
}


impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperA2C<S> for PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType:
        ContextDecodeIndexI64<ActionBuildContext> + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;
    type Config = ConfigPPO;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    fn network(&self) -> &NeuralNet<Self::NetworkOutput> {
        &self.network
    }

    fn tboard_writer(&mut self) -> Option<&mut EventWriter<std::fs::File>> {
        self.tboard_writer.as_mut()
    }

    fn global_learning_step(&self) -> i64 {
        self.global_step
    }

    fn set_global_learning_step(&mut self, step: i64) {
        self.global_step = step;
    }

    fn info_set_encoding(&self) -> &Self::InfoSetConversionContext {
        &self.info_set_encoding
    }

    fn action_encoding(&self) -> &Self::ActionConversionContext {
        &self.action_encoding
    }

    fn dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        Ok(network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "Calculating action distribution (a2c_dist)".into()))?)
    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, _information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }

    fn is_exploration_on(&self) -> bool {
        self.exploration
    }

    fn try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<S::ActionType, AmfiteatrError<S>> {
        let index = choice_tensor.f_int64_value(&[0])
            .map_err(|e| AmfiteatrError::Tensor { error: TensorError::Torch {
                origin: format!("{}", e),
                context: "Converting choice tensor to i64 action index.".to_string() } }
            )?;
        Ok(<S::ActionType>::try_from_index(index, &self.action_encoding)?)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &S::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<S>> {
        let act_i = action.try_to_index(&self.action_encoding)?;


        Ok((Tensor::from(act_i), Tensor::from(true)))
    }

    fn batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>> {
        let critic_actor= self.network.net()(info_set_batch);

        #[cfg(feature = "log_trace")]
        log::trace!("Actor on batch: {}, Critic on batch: {}", critic_actor.actor, critic_actor.critic);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<S>(
            action_param_batches,
            action_forward_mask_batches,
            action_category_mask_batches
        )?;
        let batch_entropy = critic_actor.batch_entropy_masked(
            action_forward_mask_batches,
            action_category_mask_batches

        ).map_err(|e|AmfiteatrError::Tensor {
            error: TensorError::Torch {
                context: "batch_get_actor_critic_with_logprob_and_entropy".into(),
                origin: format!("{e}")
            }
        })?;



        Ok((batch_logprob, batch_entropy, critic_actor.critic))
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<S> for PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <S as Scheme>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.a2c_select_action(state)
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicyGeneric<S> for PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <S as Scheme>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type Summary = LearnSummary;



    fn switch_explore(&mut self, enabled: bool) {
        self.exploration = enabled;
    }

    fn train_generic<
        R: Fn(&AgentStepView<S, <Self as Policy<S>>::InfoSetType>) -> Tensor

    >
    (
        &mut self, trajectories: &[AgentTrajectory<S,
        <Self as Policy<S>>::InfoSetType>],
        reward_f: R
    ) -> Result<Self::Summary, AmfiteatrRlError<S>> {

        Ok(self.ppo_train_on_trajectories(trajectories, reward_f)?)

    }

    fn set_gradient_tracing(&mut self, enabled: bool) {
        self.network.set_gradient_tracing(enabled)

    }


}


/// Policy PPO structure for discrete action space with support of masking.
pub struct PolicyMaskingDiscretePPO<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding,
>{
    /// It wraps around 'normal' PPO policy, but with more traits enforced.
    pub base: PolicyDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>,

}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyMaskingDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>{

    pub fn new(
        config: ConfigPPO,
        network: NeuralNetActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyDiscretePPO::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }
    /*
    /// Creates [`tboard::EventWriter`]. Initialy policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    pub fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<S>> {
        self.base.add_tboard_directory(directory_path)
    }

     */

    /*
    pub fn var_store(&self) -> &VarStore {
        self.base.var_store()
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        self.base.var_store_mut()
    }

     */

}
impl <
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + tensor_data::TensorIndexI64Encoding,
> TensorboardSupport<S> for PolicyMaskingDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext> {

    /// Creates [`tboard::EventWriter`]. Initially policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    fn add_tboard_directory(&mut self, directory_path: &std::path::Path) -> Result<(), AmfiteatrError<S>> {
        self.base.add_tboard_directory(directory_path)
    }
    fn t_write_scalar(&mut self, step: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>> {
        self.base.t_write_scalar(step, tag, value)
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperA2C<S> for PolicyMaskingDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType:
        ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;
    type Config = ConfigPPO;

    fn config(&self) -> &Self::Config {
        self.base.config()
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        self.base.optimizer_mut()
    }

    fn network(&self) -> &NeuralNet<Self::NetworkOutput> {
        self.base.network()
    }

    fn tboard_writer(&mut self) -> Option<&mut EventWriter<std::fs::File>> {
        self.base.tboard_writer()
    }

    fn global_learning_step(&self) -> i64 {
        self.base.global_learning_step()
    }

    fn set_global_learning_step(&mut self, step: i64) {
        self.base.set_global_learning_step(step)
    }

    fn info_set_encoding(&self) -> &Self::InfoSetConversionContext {
        self.base.info_set_encoding()
    }

    fn action_encoding(&self) -> &Self::ActionConversionContext {
        self.base.action_encoding()
    }

    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {

        /*
        let softmax = network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;

        let masks = info_set.try_build_mask(&self.action_encoding())?;


        let product = softmax.f_where_self(&masks, &Tensor::from(0.0))
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax * mask)".into())
            )?;

        #[cfg(feature = "log_debug")]
        {
            if product.nonzero().size()[0] == 0{
                warn!("Distribution of action is all zeros!, policy dist: {softmax}, masks: {masks}.");
            }
        }
        Ok(product)

         */
        let masks = info_set.try_build_mask(self.action_encoding())?;
        let masked_actor = network_output.actor.f_where_self(&masks, &Tensor::from(f32::NEG_INFINITY))
            .map_err(|e| TensorError::from_tch_with_context(e, format!("PPO masking actor network output masks: {masks}, actor: {}", &network_output.actor)))?;
        let softmax = masked_actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;



        /*
        let product = softmax.f_where_self(&masks, &Tensor::from(0.0))
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax * mask)".into())
            )?;


         */
        #[cfg(feature = "log_debug")]
        {
            if softmax.nonzero().size()[0] == 0{
                log::warn!("Distribution of action is all zeros!, policy dist: {softmax}, masks: {masks}.");
            }
        }
        Ok(softmax)


        // let masks = info_set.try_build_mask(&self.action_encoding())?.to_kind(Kind::Float);
        // let logits_masked = masks.f_mul(&network_output.actor).map_err(
        //     |e| TensorError::from_tch_with_context(e, format!("Multiplying actor logits and mask {} x {masks}", &network_output.actor)
        // ))?;
        // let m = (Tensor::ones_like(&masks) - &masks);
        // warn!("m: {m}");
        // let shift = (Tensor::ones_like(&masks) - &masks) * Tensor::from(f32::NEG_INFINITY);//* Tensor::from(f32::NEG_INFINITY).to_device(masks.device());
        //
        // let dist = &logits_masked + &shift;
        // warn!("Distribution of action is all zeros!, policy dist: {}, masks: {masks}, logits_masked: {logits_masked}, shift: {shift}.", network_output.actor);
        // #[cfg(feature = "log_debug")]
        // {
        //     if dist.nonzero().size()[0] == 0{
        //         warn!("Distribution of action is all zeros!, policy dist: {}, masks: {masks}.", network_output.actor);
        //     }
        // }
        // Ok(dist)


        //logits =
    }

    fn is_action_masking_supported(&self) -> bool {
        true
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        information_set.try_build_mask(self.action_encoding())
    }

    fn is_exploration_on(&self) -> bool {
        self.base.is_exploration_on()
    }

    fn try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.base.try_action_from_choice_tensor(choice_tensor)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &S::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<S>> {
        self.base.vectorize_action_and_create_category_mask(action)
    }

    fn batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>> {
        self.base.batch_get_logprob_entropy_critic(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<S> for PolicyMaskingDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <S as Scheme>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.a2c_select_action(state)
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicyGeneric<S> for PolicyMaskingDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <S as Scheme>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>{
    type Summary = LearnSummary;



    fn switch_explore(&mut self, enabled: bool) {
        self.base.switch_explore(enabled)
    }

    fn train_generic<
        R: Fn(&AgentStepView<S, <Self as Policy<S>>::InfoSetType>) -> Tensor
    >(
        &mut self, trajectories: &[AgentTrajectory<S, <Self as Policy<S>>::InfoSetType>],
        reward_f: R
    ) -> Result<Self::Summary, AmfiteatrRlError<S>> {

        Ok(self.ppo_train_on_trajectories(trajectories, reward_f)?)
    }

    fn set_gradient_tracing(&mut self, enabled: bool) {
        self.base.set_gradient_tracing(enabled)

    }

}

