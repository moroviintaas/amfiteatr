use std::fmt::Debug;
use std::fs::File;
use std::marker::PhantomData;
use tboard::EventWriter;
use tch::nn::{Optimizer, VarStore};
use tch::{TchError, Tensor};
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_core::util::TensorboardSupport;
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigA2C, LearnSummary, LearningNetworkPolicy, PolicyHelperA2C, PolicyTrainHelperA2C};
use crate::{tensor_data, MaskingInformationSetActionMultiParameter};
use crate::tensor_data::{ContextDecodeMultiIndexI64, ContextEncodeMultiIndexI64, ContextEncodeTensor, MultiTensorDecoding, MultiTensorIndexI64Encoding, TensorEncoding};
use crate::torch_net::{ActorCriticOutput, DeviceTransfer, NeuralNet, NeuralNetMultiActorCritic, TensorMultiParamActorCritic};

/// Experimental A2C policy for actions from discrete actions space but sampled from
/// more than one parameter distribution.
/// E.g. Action type is one parameter sampled from one distribution (with space size `N`).
/// Then optionally some additional parameters like e.g. coordinates, tooling (or whatever) are sampled from
/// different distributions.
/// It is an __experimental__ approach disassembling cartesian action space of size `N x P1 x P2 x ... Pk` into
/// `k + 1` distribution of action parameters.
pub struct PolicyMultiDiscreteA2C<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorIndexI64Encoding,
>{
    config: ConfigA2C,
    network: NeuralNetMultiActorCritic,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_encoding: InfoSetConversionContext,
    action_encoding: ActionBuildContext,
    exploration: bool,

    tboard_writer: Option<tboard::EventWriter<File>>,
    global_step: i64,


}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext:  MultiTensorIndexI64Encoding,
> PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType:
ContextDecodeMultiIndexI64<ActionBuildContext>
+ ContextEncodeMultiIndexI64<ActionBuildContext>

{
    pub fn new(
        config: ConfigA2C,
        network: NeuralNetMultiActorCritic,
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

    /// Creates [`tboard::EventWriter`]. Initialy policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    pub fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>>{
        let tboard = EventWriter::create(directory_path).map_err(|e|{
            AmfiteatrError::TboardFlattened {context: "Creating tboard EventWriter".into(), error: format!("{e}")}
        })?;
        self.tboard_writer = Some(tboard);
        Ok(())
    }

    pub fn var_store(&self) -> &VarStore {
        self.network.var_store()
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext:  MultiTensorIndexI64Encoding,
> TensorboardSupport<DP> for PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
{
    /// Creates [`tboard::EventWriter`]. Initially policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        let tboard = EventWriter::create(directory_path).map_err(|e| {
            AmfiteatrError::TboardFlattened { context: "Creating tboard EventWriter".into(), error: format!("{e}") }
        })?;
        self.tboard_writer = Some(tboard);
        Ok(())
    }
    fn t_write_scalar(&mut self, step: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>> {
        match &mut self.tboard_writer{
            None => Ok(false),
            Some(writer) => {
                writer.write_scalar(step, tag, value)
                    .map_err(|e| AmfiteatrError::TboardFlattened {
                        context: "Tboard - writing scalar (PPO)".to_string(),
                        error: "Writer not initialised".to_string(),
                    })?;
                Ok(true)
            }
        }
    }
}
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
PolicyHelperA2C<DP> for PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;
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

    fn dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        network_output.actor.iter().map(|t|

            t.f_softmax(-1, tch::Kind::Float)
        ).collect::<Result<Vec<Tensor>, _>>().map_err(|e|
            TensorError::Torch {
                origin: format!("{e}"),
                context: "Calculating distribution for actions".into()
            }.into())
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
        let choices = choice_tensor.iter().map(|t|
            t.f_int64_value(&[0])
        ).collect::<Result<Vec<_>, TchError>>()
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: format!("Choising action from multiple param tensors: {choice_tensor:?}"),
                }})?;

        Ok(<DP::ActionType>::try_from_indices(&choices[..], &self.action_encoding)?)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        let (act_t, cat_mask_t) = action.action_index_and_mask_tensor_vecs(self.action_encoding())
            .map_err(AmfiteatrError::from)?;

        Ok((act_t, cat_mask_t))
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
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<DP> for PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

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
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicy<DP> for PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
+ ContextEncodeMultiIndexI64<ActionBuildContext>,
{
    type Summary = LearnSummary;


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
    ) -> Result<LearnSummary, AmfiteatrRlError<DP>> {

        Ok(self.a2c_train_on_trajectories(trajectories, reward_f)?)

    }
}

/// Experimental PPO policy for actions from discrete actions space but sampled from
/// more than one parameter distribution with support of masking out illegal actions.
pub struct PolicyMaskingMultiDiscreteA2C<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding,
>{
    pub base: PolicyMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyMaskingMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext> + ContextEncodeMultiIndexI64<ActionBuildContext>{
    pub fn new(
        config: ConfigA2C,
        network: NeuralNetMultiActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyMultiDiscreteA2C::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }

    /// Creates [`tboard::EventWriter`]. Initialy policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    pub fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        self.base.add_tboard_directory(directory_path)
    }

    pub fn var_store(&self) -> &VarStore {
        self.base.var_store()
    }

    pub  fn var_store_mut(&mut self) -> &mut VarStore {
        self.base.var_store_mut()
    }

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> TensorboardSupport<DP> for PolicyMaskingMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
{
    /// Creates [`tboard::EventWriter`]. Initially policy does not use `tensorboard` directory to store epoch
    /// training results (like entropy, policy loss, value loss). However, you cen provide it with directory
    /// to create tensorboard files.
    fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        self.base.add_tboard_directory(directory_path)
    }
    fn t_write_scalar(&mut self, step: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>> {
        self.base.t_write_scalar(step, tag, value)
    }
}
impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyHelperA2C<DP> for PolicyMaskingMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;
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

    fn dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        let masks = info_set.try_build_masks(self.action_encoding())?.move_to_device(network_output.device());
        let masked: Vec<_> = network_output.actor.iter().zip(masks).map(|(t,m)|{
            t.f_softmax(-1, tch::Kind::Float)?.f_mul(
                &m
            )
        }).collect::<Result<Vec<Tensor>, TchError>>()
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: "Calculating action parameter probabilities".into()
                }
            })
            ?;
        Ok(masked)
    }

    fn is_action_masking_supported(&self) -> bool {
        true
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        information_set.try_build_masks(self.action_encoding())
    }

    fn is_exploration_on(&self) -> bool {
        self.base.exploration
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

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<DP> for PolicyMaskingMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{

    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.a2c_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicy<DP> for PolicyMaskingMultiDiscreteA2C<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type Summary = LearnSummary;



    fn switch_explore(&mut self, enabled: bool) {
        self.base.switch_explore(enabled)
    }

    fn train_on_trajectories<R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>], reward_f: R)
        -> Result<Self::Summary, AmfiteatrRlError<DP>> {
        Ok(self.a2c_train_on_trajectories(trajectories, reward_f)?)
    }
}