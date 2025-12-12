
use std::fmt::Debug;
use std::fs::File;
use std::marker::PhantomData;
use tboard::EventWriter;
use tch::{Kind, TchError};
use tch::nn::VarStore;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::scheme::Scheme;
use amfiteatr_core::error::{AmfiteatrError,TensorError};
use amfiteatr_core::util::TensorboardSupport;
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigPPO, LearnSummary, LearningNetworkPolicyGeneric, PolicyHelperA2C, PolicyTrainHelperPPO};
use crate::{tch, MaskingInformationSetActionMultiParameter, tensor_data};
use crate::tch::nn::Optimizer;
use crate::tch::Tensor;
use crate::tensor_data::{
    MultiTensorDecoding,
    MultiTensorIndexI64Encoding,
    TensorEncoding,
    ContextEncodeMultiIndexI64,
    ContextEncodeTensor,
    ContextDecodeMultiIndexI64
};
use crate::torch_net::{
    ActorCriticOutput,
    DeviceTransfer,
    NeuralNet,
    NeuralNetMultiActorCritic,
    TensorMultiParamActorCritic
};




/// Experimental PPO policy for actions from discrete actions space but sampled from
/// more than one parameter distribution.
/// E.g. Action type is one parameter sampled from one distribution (with space size `N`).
/// Then optionally some additional parameters like e.g. coordinates, tooling (or whatever) are sampled from
/// different distributions.
/// It is an __experimental__ approach disassembling cartesian action space of size `N x P1 x P2 x ... Pk` into
/// `k + 1` distribution of action parameters.
pub struct PolicyMultiDiscretePPO<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorIndexI64Encoding,
>
//where <S as DomainParameters>::ActionType: ContextTryFromMultipleTensors<ActionBuildContext>
{
    config: ConfigPPO,
    network: NeuralNetMultiActorCritic,
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
    ActionBuildContext:  MultiTensorIndexI64Encoding,
> PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <S as Scheme>::ActionType:
    ContextDecodeMultiIndexI64<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>

{




    fn batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &Vec<Tensor>,
        action_category_mask_batches: Option<&Vec<Tensor>>,
        action_forward_mask_batches: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>>{

        let critic_actor= self.network.operator()(self.network.var_store(), info_set_batch);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<S>(
            action_param_batches,
            action_forward_mask_batches,
            action_category_mask_batches
        )?;
        let batch_entropy = critic_actor.batch_entropy_masked(
            action_forward_mask_batches,
            action_category_mask_batches

        ).map_err(|e| TensorError::Torch {
            origin: format!("{}", e),
            context: "batch_get_actor_critic_with_logprob_and_entropy (entropy)".into(),
        })?;

        let batch_entropy_avg = batch_entropy.f_sum_dim_intlist(
            Some(1),
            false,
            Kind::Float
        ).and_then(|t| t.f_div_scalar(batch_entropy.size()[1]))
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    context: "Calculating batch entropy avg".into(),
                    origin: format!("{e}")
                }
            }
            )?;
        //println!("batch entropy: {}", batch_entropy);
        //println!("batch entropy avg: {}", batch_entropy_avg);

        Ok((batch_logprob, batch_entropy_avg, critic_actor.critic))
    }

}



impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorIndexI64Encoding,
>
PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <S as Scheme>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
{
    pub fn new(
        config: ConfigPPO,
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
    pub fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<S>>{
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
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<S> for PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <S as Scheme>::ActionType:
    ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

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
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicyGeneric<S> for PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <S as Scheme>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>,
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
}



impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
PolicyHelperA2C<S> for PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType:
        ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;
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
        network_output.actor.iter().map(|t|

            t.f_softmax(-1, self.config().tensor_kind)
        ).collect::<Result<Vec<Tensor>, _>>().map_err(|e|
            TensorError::Torch {
                origin: format!("{e}"),
                context: "Calculating distribution for actions".into()
            }.into())
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
        let choices = choice_tensor.iter().map(|t|
            t.f_int64_value(&[0])
        ).collect::<Result<Vec<_>, TchError>>()
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: format!("Choising action from multiple param tensors: {choice_tensor:?}"),
                }})?;

        Ok(<S::ActionType>::try_from_indices(&choices[..], &self.action_encoding)?)
    }

    fn vectorize_action_and_create_category_mask(&self, action: &S::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<S>> {
        let (act_t, cat_mask_t) = action.action_index_and_mask_tensor_vecs(self.action_encoding())
            .map_err(AmfiteatrError::from)?;

        Ok((act_t, cat_mask_t))
    }

    fn batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>> {
        #[cfg(feature = "log_trace")]
        log::trace!("action_category_mask_batches: {:?}", action_category_mask_batches);
        self.batch_get_actor_critic_with_logprob_and_entropy(
            info_set_batch,
            action_param_batches,
            action_category_mask_batches,
            action_forward_mask_batches

        )
    }
}

/*

impl<
    S: DomainParameters,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
PolicyHelperPPO<S> for PolicyPpoMultiDiscrete<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as DomainParameters>::ActionType:
        ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;

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

    fn ppo_dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
            network_output.actor.iter().map(|t|

                t.f_softmax(-1, self.config().tensor_kind)
            ).collect::<Result<Vec<Tensor>, _>>().map_err(|e|
            TensorError::Torch {
                origin: format!("{e}"),
                context: "Calculating distribution for actions".into()
            }.into())
    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, _information_sets: &Self::InfoSet) -> Result<Vec<Tensor>, AmfiteatrError<S>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }


    fn ppo_exploration(&self) -> bool {
        self.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<S::ActionType, AmfiteatrError<S>> {
        //Ok(<S::ActionType>::try_from_tensors(choice_tensor, &self.action_build_context)?)
        let choices = choice_tensor.iter().map(|t|
            t.f_int64_value(&[0])
        ).collect::<Result<Vec<_>, TchError>>()
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: format!("Choising action from multiple param tensors: {choice_tensor:?}"),
            }})?;

        Ok(<S::ActionType>::try_from_indices(&choices[..], &self.action_encoding)?)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &S::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<S>> {
        let (act_t, cat_mask_t) = action.action_index_and_mask_tensor_vecs(self.action_conversion_context())
            .map_err(AmfiteatrError::from)?;

        Ok((act_t, cat_mask_t))

    }

    fn ppo_batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &Vec<Tensor>, action_category_mask_batches: Option<&Vec<Tensor>>, action_forward_mask_batches: Option<&Vec<Tensor>>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>> {
        #[cfg(feature = "log_trace")]
        log::trace!("action_category_mask_batches: {:?}", action_category_mask_batches);
        self.batch_get_actor_critic_with_logprob_and_entropy(
            info_set_batch,
            action_param_batches,
            action_category_mask_batches,
            action_forward_mask_batches

        )
    }
}

 */
/// Experimental PPO policy for actions from discrete actions space but sampled from
/// more than one parameter distribution with support of masking out illegal actions.
pub struct PolicyMaskingMultiDiscretePPO<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding,
>{
    pub base: PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyMaskingMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where <S as Scheme>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>{
    pub fn new(
        config: ConfigPPO,
        network: NeuralNetMultiActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyMultiDiscretePPO::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }


    pub fn var_store(&self) -> &VarStore {
        self.base.var_store()
    }

    pub fn var_store_mut(&mut self) -> &mut VarStore {
        self.base.var_store_mut()
    }

}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> TensorboardSupport<S> for PolicyMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
{
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
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> TensorboardSupport<S> for PolicyMaskingMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
{
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
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyHelperA2C<S> for PolicyMaskingMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
        + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;
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
        let masks = info_set.try_build_masks(self.action_encoding())?.move_to_device(network_output.device());
        let masked: Vec<_> = network_output.actor.iter().zip(masks).map(|(t,m)|{
            t.f_softmax(-1, tch::Kind::Float)?.f_where_self(&m, &Tensor::from(0.0))
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

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        information_set.try_build_masks(self.action_encoding())
    }

    fn is_exploration_on(&self) -> bool {
        self.base.exploration
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
/*
impl<
    S: DomainParameters,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyHelperPPO<S> for PolicyMaskingPpoMultiDiscrete<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as DomainParameters>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
        + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;

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
        self.base.info_set_conversion_context()
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        self.base.action_conversion_context()
    }

    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        let masks = info_set.try_build_masks(self.action_conversion_context())?.move_to_device(network_output.device());
        let masked: Vec<_> = network_output.actor.iter().zip(masks).map(|(t,m)|{
            t.f_softmax(-1, self.config().tensor_kind)?.f_mul(
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

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<S>> {
        information_set.try_build_masks(self.action_conversion_context())
    }

    fn ppo_exploration(&self) -> bool {
        self.base.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.base.ppo_try_action_from_choice_tensor(choice_tensor)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &S::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<S>> {
        self.base.ppo_vectorise_action_and_create_category_mask(action)
    }

    fn ppo_batch_get_logprob_entropy_critic(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<S>> {
        self.base.ppo_batch_get_logprob_entropy_critic(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}

*/
impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>
        + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<S> for PolicyMaskingMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType:
        ContextDecodeMultiIndexI64<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{

    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.a2c_select_action(state)
    }
}

impl<
    S: Scheme,
    InfoSet: InformationSet<S> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<S, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicyGeneric<S> for PolicyMaskingMultiDiscretePPO<S, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <S as Scheme>::ActionType: ContextDecodeMultiIndexI64<ActionBuildContext>
        + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type Summary = LearnSummary;



    fn switch_explore(&mut self, enabled: bool) {
        self.base.switch_explore(enabled)
    }

    fn train_generic<R: Fn(&AgentStepView<
        S,
        <Self as Policy<S>>::InfoSetType>) -> Tensor
    >(
        &mut self, trajectories: &[AgentTrajectory<S, <Self as Policy<S>>::InfoSetType>],
        reward_f: R
    ) -> Result<Self::Summary, AmfiteatrRlError<S>>

    {

        Ok(self.ppo_train_on_trajectories(trajectories, reward_f)?)
    }
}