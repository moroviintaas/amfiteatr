use std::fmt::Debug;
use std::marker::PhantomData;
use tch::nn::{Optimizer, VarStore};
use tch::Tensor;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigPPO, LearnSummary, LearningNetworkPolicy, PolicyHelperA2C, PolicyTrainHelperPPO};
use crate::{tensor_data, MaskingInformationSetAction};
use crate::tensor_data::{ContextEncodeIndexI64, ContextEncodeTensor, TensorDecoding, TensorIndexI64Encoding, TensorEncoding, ContextDecodeIndexI64};
use crate::torch_net::{ActorCriticOutput, NeuralNet, NeuralNetActorCritic, TensorActorCritic};


/// Policy PPO for discrete action space with single distribution using [`tch`] crate for `torch` backed
/// [`Tensors`](tch::Tensor).
pub struct PolicyDiscretePPO<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding,
>{
    config: ConfigPPO,
    network: NeuralNetActorCritic,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_encoding: InfoSetConversionContext,
    action_encoding: ActionBuildContext,

    exploration: bool,

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
        + tensor_data::ActionTensorFormat<Tensor>,
> PolicyDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{


    /// ```
    /// use tch::{Device, nn, Tensor};
    /// use tch::nn::{Adam, VarStore};
    /// use amfiteatr_core::demo::{DemoDomain, DemoInfoSet};
    /// use amfiteatr_rl::demo::{DemoActionConversionContext, DemoConversionToTensor};
    /// use amfiteatr_rl::policy::{ConfigPPO, PolicyDiscretePPO};
    /// use amfiteatr_rl::torch_net::{NeuralNetActorCritic, TensorActorCritic};
    /// let var_store = VarStore::new(Device::Cpu);
    /// let net = NeuralNetActorCritic::new(var_store, |path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 1, 128, Default::default()))
    ///         .add(nn::linear(path / "hidden", 128, 128, Default::default()));
    ///     let actor = nn::linear(path / "al", 128, 2, Default::default());
    ///     let critic = nn::linear(path / "cl", 128, 1, Default::default());
    ///     let device = path.device();
    ///     {move |xs: &Tensor|{
    ///         let xs = xs.to_device(device).apply(&seq);
    ///         TensorActorCritic{critic: xs.apply(&critic), actor: xs.apply(&actor)}
    ///     }}
    /// });
    /// let optimizer = net.build_optimizer(Adam::default(), 0.01).unwrap();
    /// let config = ConfigPPO::default();
    /// let demo_info_set_ctx = DemoConversionToTensor::default();
    /// let demo_action_ctx = DemoActionConversionContext{};
    /// let policy: PolicyDiscretePPO<DemoDomain, DemoInfoSet, DemoConversionToTensor, DemoActionConversionContext> =
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
        }
    }

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
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>{

        let critic_actor= self.network.net()(info_set_batch);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<DP>(
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
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperA2C<DP> for PolicyDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType:
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

    fn info_set_encoding(&self) -> &Self::InfoSetConversionContext {
        &self.info_set_encoding
    }

    fn action_encoding(&self) -> &Self::ActionConversionContext {
        &self.action_encoding
    }

    fn dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Ok(network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "Calculating action distribution (a2c_dist)".into()))?)
    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, _information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }

    fn is_exploration_on(&self) -> bool {
        self.exploration
    }

    fn try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        let index = choice_tensor.f_int64_value(&[0])
            .map_err(|e| AmfiteatrError::Tensor { error: TensorError::Torch {
                origin: format!("{}", e),
                context: "Converting choice tensor to i64 action index.".to_string() } }
            )?;
        Ok(<DP::ActionType>::try_from_index(index, &self.action_encoding)?)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        let act_i = action.try_to_index(&self.action_encoding)?;


        Ok((Tensor::from(act_i), Tensor::from(true)))
    }

    fn batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>> {
        let a2c_net = self.network().net()(info_set_batch);

        let (log_prob, entropy) = a2c_net.batch_get_logprob_and_entropy(
            action_param_batches,
            action_category_mask_batches,
            action_forward_mask_batches
        )?;

        Ok((log_prob, entropy, a2c_net.critic))
    }
}
/*
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperPPO<DP> for PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext> + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;

    fn config(&self) -> &ConfigPpo {
        &self.config
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput> {
        &self.network
    }

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext {
        &self.info_set_encoding
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        &self.action_encoding
    }

    fn ppo_dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Ok(network_output.actor.f_softmax(-1, self.config.tensor_kind)
            .map_err(|e| TensorError::from_tch_with_context(e, "Calculating action distribution (ppo_dist)".into()))?)


    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, _information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }

    fn ppo_exploration(&self) -> bool {
        self.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        //Ok(<DP::ActionType>::try_from_tensor(choice_tensor, &self.action_build_context)?)
        let index = choice_tensor.f_int64_value(&[0])
            .map_err(|e| AmfiteatrError::Tensor { error: TensorError::Torch {
                origin: format!("{}", e),
                context: "Converting choice tensor to i64 action index.".to_string() } }
            )?;
        Ok(<DP::ActionType>::try_from_index(index, &self.action_encoding)?)


    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType)
        -> Result<
            (<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
             <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType
            ), AmfiteatrError<DP>> {
        let act_i = action.try_to_index(&self.action_encoding)?;


        Ok((Tensor::from(act_i), Tensor::from(true)))
    }

    fn ppo_batch_get_logprob_entropy_critic(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>
    {

        #[cfg(feature = "log_trace")]
        log::trace!("action_category_mask_batches: {:?}", action_category_mask_batches);
        self.batch_get_logprob_entropy_critic(
            info_set_batch,
            action_param_batches,
            action_category_mask_batches,
            action_forward_mask_batches

        )
    }
}


 */
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<DP> for PolicyDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.a2c_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicy<DP> for PolicyDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type Summary = LearnSummary;

    fn var_store(&self) -> &VarStore {
        self.network.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.exploration = enabled;
    }

    fn train_on_trajectories<
        R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor

    >
    (
        &mut self, trajectories: &[AgentTrajectory<DP,
        <Self as Policy<DP>>::InfoSetType>],
        reward_f: R
    ) -> Result<Self::Summary, AmfiteatrRlError<DP>> {

        Ok(self.ppo_train_on_trajectories(trajectories, reward_f)?)

    }
}

/// Policy PPO structure for discrete action space with support of masking.
pub struct PolicyMaskingDiscretePPO<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding,
>{
    /// It wraps around 'normal' PPO policy, but with more traits enforced.
    pub base: PolicyDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>,

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyMaskingDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{

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
    fn batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &Tensor,
        action_category_mask_batches: Option<&Tensor>,
        action_forward_mask_batches: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>{

        self.base.batch_get_actor_critic_with_logprob_and_entropy(
            info_set_batch,
            action_param_batches,
            action_category_mask_batches,
            action_forward_mask_batches
        )
    }

     */
}
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperA2C<DP> for PolicyMaskingDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType:
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

    fn info_set_encoding(&self) -> &Self::InfoSetConversionContext {
        self.base.info_set_encoding()
    }

    fn action_encoding(&self) -> &Self::ActionConversionContext {
        self.base.action_encoding()
    }

    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        let softmax = network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;

        let masks = info_set.try_build_mask(&self.action_encoding())?;

        let product = softmax.f_mul(&masks)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax * mask)".into())
            )?;
        Ok(product)
    }

    fn is_action_masking_supported(&self) -> bool {
        true
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        information_set.try_build_mask(self.action_encoding())
    }

    fn is_exploration_on(&self) -> bool {
        self.base.is_exploration_on()
    }

    fn try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.base.try_action_from_choice_tensor(choice_tensor)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        self.base.vectorize_action_and_create_category_mask(action)
    }

    fn batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>> {
        self.base.batch_get_logprob_entropy_critic(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}

/*
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperPPO<DP> for PolicyMaskingPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;

    fn config(&self) -> &ConfigPpo {
        self.base.config()
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        self.base.optimizer_mut()
    }

    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput> {
        self.base.ppo_network()
    }

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext {
        self.base.info_set_encoding()
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        self.base.action_encoding()
    }

    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        let softmax = network_output.actor.f_softmax(-1, self.base.config.tensor_kind)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;

        let masks = info_set.try_build_mask(&self.base.action_encoding)?;

        let product = softmax.f_mul(&masks)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax * mask)".into())
        )?;
        Ok(product)

    }

    fn is_action_masking_supported(&self) -> bool {
        true
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {


        information_set.try_build_mask(self.action_conversion_context())
    }

    fn ppo_exploration(&self) -> bool {
        self.base.ppo_exploration()
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.base.ppo_try_action_from_choice_tensor(choice_tensor)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        self.base.ppo_vectorise_action_and_create_category_mask(action)
    }

    fn ppo_batch_get_logprob_entropy_critic(
        &self, info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
        action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>
    {
        self.base.ppo_batch_get_logprob_entropy_critic(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}

 */

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<DP> for PolicyMaskingDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.a2c_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicy<DP> for PolicyMaskingDiscretePPO<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionBuildContext, > + ContextEncodeIndexI64<ActionBuildContext>{
    type Summary = LearnSummary;

    fn var_store(&self) -> &VarStore {
        self.base.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.base.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.base.switch_explore(enabled)
    }

    fn train_on_trajectories<
        R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor
    >(
        &mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>],
        reward_f: R
    ) -> Result<Self::Summary, AmfiteatrRlError<DP>> {

        Ok(self.ppo_train_on_trajectories(trajectories, reward_f)?)
    }
}