use std::fmt::Debug;
use std::marker::PhantomData;
use tch::nn::{Optimizer, VarStore};
use tch::Tensor;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::actor_critic::base::{ConfigA2C, PolicyHelperA2C, PolicyTrainHelperA2C};
use crate::policy::{ConfigPpo, LearningNetworkPolicy, PolicyPpoDiscrete};
use crate::{MaskingInformationSetAction, tensor_data};
use crate::tensor_data::{ContextDecodeIndexI64, ContextEncodeIndexI64, ContextEncodeTensor, TensorDecoding, TensorEncoding, TensorIndexI64Encoding};
use crate::torch_net::{ActorCriticOutput, NeuralNet, NeuralNetActorCritic, TensorActorCritic};

pub struct PolicyA2cDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding,
>{
    config: ConfigA2C,
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
    ActionBuildContext: TensorDecoding,
> PolicyA2cDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{
    pub fn new(
        config: ConfigA2C,
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

    /*
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

     */
}

impl<
DP: DomainParameters,
InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding>,
InfoSetEncoding: TensorEncoding,
ActionEncoding: TensorDecoding + TensorIndexI64Encoding
+ tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperA2C<DP> for PolicyA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
where <DP as DomainParameters>::ActionType:
ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetEncoding;
    type ActionConversionContext = ActionEncoding;
    type NetworkOutput = TensorActorCritic;
    type Config = ConfigA2C;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    fn network(&self) -> &NeuralNet<Self::NetworkOutput> {
        &self.network
    }

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext {
        &self.info_set_encoding
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        &self.action_encoding
    }

    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Ok(network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "Calculating action distribution (a2c_dist)".into()))?)
    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
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

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding>,
    InfoSetEncoding: TensorEncoding,
    ActionEncoding: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<DP> for PolicyA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
    where <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.a2c_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding>,
    InfoSetEncoding: TensorEncoding,
    ActionEncoding: TensorDecoding + TensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicy<DP> for PolicyA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
    where <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    fn var_store(&self) -> &VarStore {
        self.network().var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.exploration = enabled
    }

    fn train_on_trajectories<R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>], reward_f: R) -> Result<(), AmfiteatrRlError<DP>> {
        Ok(self.a2c_train_on_trajectories(trajectories, reward_f)?)
    }
}


/// Policy PPO structure for discrete action space with support of masking.
pub struct PolicyMaskingA2cDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding,
>{
    /// It wraps around 'normal' PPO policy, but with more traits enforced.
    pub base: PolicyA2cDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>,

}

impl <
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: TensorDecoding + TensorIndexI64Encoding,
> PolicyMaskingA2cDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{

    pub fn new(
        config: ConfigA2C,
        network: NeuralNetActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyA2cDiscrete::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }
}

impl <
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding> + MaskingInformationSetAction<DP, ActionEncoding>,
    InfoSetEncoding: TensorEncoding,
    ActionEncoding: TensorDecoding + TensorIndexI64Encoding,
> PolicyHelperA2C<DP> for PolicyMaskingA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
    where <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetEncoding;
    type ActionConversionContext = ActionEncoding;
    type NetworkOutput = TensorActorCritic;
    type Config = ConfigA2C;

    fn config(&self) -> &Self::Config {
        self.base.config()
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        self.base.optimizer_mut()
    }

    fn network(&self) -> &NeuralNet<Self::NetworkOutput> {
        self.base.network()
    }

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext {
        self.base.info_set_conversion_context()
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        self.base.action_conversion_context()
    }

    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        let softmax = network_output.actor.f_softmax(-1, tch::Kind::Float)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;

        let masks = info_set.try_build_mask(&self.action_conversion_context())?;

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

impl <
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding> + MaskingInformationSetAction<DP, ActionEncoding>,
    InfoSetEncoding: TensorEncoding,
    ActionEncoding: TensorDecoding + TensorIndexI64Encoding,
> Policy<DP> for PolicyMaskingA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
    where <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.a2c_select_action(state)
    }
}

impl <
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetEncoding> + MaskingInformationSetAction<DP, ActionEncoding>,
    InfoSetEncoding: TensorEncoding,
    ActionEncoding: TensorDecoding + TensorIndexI64Encoding,
> LearningNetworkPolicy<DP> for PolicyMaskingA2cDiscrete<DP, InfoSet, InfoSetEncoding, ActionEncoding>
    where <DP as DomainParameters>::ActionType:
    ContextDecodeIndexI64<ActionEncoding> + ContextEncodeIndexI64<ActionEncoding>{
    fn var_store(&self) -> &VarStore {
        self.base.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.base.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.base.switch_explore(enabled)
    }

    fn train_on_trajectories<R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>], reward_f: R) -> Result<(), AmfiteatrRlError<DP>> {
        #[cfg(feature = "log_trace")]
        log::trace!("Train on trajectories start.");
        Ok(self.a2c_train_on_trajectories(trajectories, reward_f)?)
    }
}