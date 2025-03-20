
use std::fmt::Debug;
use std::marker::PhantomData;
use tch::{Kind, TchError};
use tch::nn::VarStore;
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError,TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigPPO, LearningNetworkPolicy, PolicyHelperPPO};
use crate::{tch, MaskingInformationSetActionMultiParameter, tensor_data};
use crate::tch::nn::Optimizer;
use crate::tch::Tensor;
use crate::tensor_data::{
    MultiTensorDecoding,
    MultiTensorIndexI64Encoding,
    TensorEncoding,
    ContextEncodeMultiIndexI64,
    ContextDecodeMultiTensor,
    ContextEncodeTensor,
};
use crate::torch_net::{
    ActorCriticOutput,
    DeviceTransfer,
    NeuralNet,
    NeuralNetMultiActorCritic,
    TensorMultiParamActorCritic
};



pub struct PolicyPpoMultiDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding,
>
//where <DP as DomainParameters>::ActionType: ContextTryFromMultipleTensors<ActionBuildContext>
{
    config: ConfigPPO,
    network: NeuralNetMultiActorCritic,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_conversion_context: InfoSetConversionContext,
    action_build_context: ActionBuildContext,
    exploration: bool,




}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding,
> PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType:
    ContextDecodeMultiTensor<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>

{




    fn batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &Vec<Tensor>,
        action_category_mask_batches: Option<&Vec<Tensor>>,
        action_forward_mask_batches: Option<&Vec<Tensor>>,
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>{

        let critic_actor= (&self.network.net())(info_set_batch);

        let batch_logprob = critic_actor.batch_log_probability_of_action::<DP>(
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

pub(crate) fn vec_2d_clear_second_dim<T>(v: &mut Vec<Vec<T>>){
    for c in v.iter_mut(){
        c.clear()
    }
}

pub(crate) fn vec_2d_append_second_dim<T>(v: &mut Vec<Vec<T>>, append: &mut Vec<Vec<T>>){
    for (c_append, c_base) in append.iter_mut().zip(v.iter_mut()){
        c_base.append(c_append)
    }
}
pub(crate)  fn vec_2d_push_second_dim<T>(v: &mut Vec<Vec<T>>, append: Vec<T>){
    for (c_push, c_base) in append.into_iter().zip(v.iter_mut()){
        c_base.push(c_push)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding,
>
PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: ContextDecodeMultiTensor<ActionBuildContext>
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
            info_set_conversion_context,
            action_build_context,
            exploration: true,
        }
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<DP> for PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
        ContextDecodeMultiTensor<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.ppo_select_action(state)


    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicy<DP> for PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where <DP as DomainParameters>::ActionType: ContextDecodeMultiTensor<ActionBuildContext>
    + ContextEncodeMultiIndexI64<ActionBuildContext>,
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
        R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor

    >
    (
        &mut self, trajectories: &[AgentTrajectory<DP,
        <Self as Policy<DP>>::InfoSetType>],
        reward_f: R
    ) -> Result<(), AmfiteatrRlError<DP>> {

        self.ppo_train_on_trajectories(trajectories, reward_f)

    }
}





impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> ,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
PolicyHelperPPO<DP> for PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType:
        ContextDecodeMultiTensor<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;

    fn config(&self) -> &ConfigPPO {
        &self.config
    }

    fn optimizer_mut(&mut self) -> &mut Optimizer {
        &mut self.optimizer
    }

    fn ppo_network(&self) -> &NeuralNet<Self::NetworkOutput> {
        &self.network
    }

    fn info_set_conversion_context(&self) -> &Self::InfoSetConversionContext {
        &self.info_set_conversion_context
    }

    fn action_conversion_context(&self) -> &Self::ActionConversionContext {
        &self.action_build_context
    }

    fn ppo_dist(&self, _info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
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

    fn generate_action_masks(&self, _information_sets: &Self::InfoSet) -> Result<Vec<Tensor>, AmfiteatrError<DP>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }


    fn ppo_exploration(&self) -> bool {
        self.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        Ok(<DP::ActionType>::try_from_tensors(choice_tensor, &self.action_build_context)?)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        let (act_t, cat_mask_t) = action.action_index_and_mask_tensor_vecs(&self.action_conversion_context())
            .map_err(|e| AmfiteatrError::from(e))?;

        Ok((act_t, cat_mask_t))

    }

    fn ppo_batch_get_actor_critic_with_logprob_and_entropy(&self, info_set_batch: &Tensor, action_param_batches: &Vec<Tensor>, action_category_mask_batches: Option<&Vec<Tensor>>, action_forward_mask_batches: Option<&Vec<Tensor>>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>> {
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

pub struct PolicyMaskingPpoMultiDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding,
>{
    pub base: PolicyPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext> + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyMaskingPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where <DP as DomainParameters>::ActionType: ContextDecodeMultiTensor<ActionBuildContext>{
    pub fn new(
        config: ConfigPPO,
        network: NeuralNetMultiActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyPpoMultiDiscrete::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> PolicyHelperPPO<DP> for PolicyMaskingPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType: ContextDecodeMultiTensor<ActionBuildContext>
        + ContextEncodeMultiIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorMultiParamActorCritic;

    fn config(&self) -> &ConfigPPO {
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

    fn ppo_dist(&self, info_set: &Self::InfoSet, network_output: &Self::NetworkOutput) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
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

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        information_set.try_build_masks(self.action_conversion_context())
    }

    fn ppo_exploration(&self) -> bool {
        self.base.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.base.ppo_try_action_from_choice_tensor(choice_tensor)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType) -> Result<(<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType), AmfiteatrError<DP>> {
        self.base.ppo_vectorise_action_and_create_category_mask(action)
    }

    fn ppo_batch_get_actor_critic_with_logprob_and_entropy(&self, info_set_batch: &Tensor, action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>, action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>> {
        self.base.ppo_batch_get_actor_critic_with_logprob_and_entropy(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
        + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding + tensor_data::ActionTensorFormat<Vec<Tensor>>,
>
Policy<DP> for PolicyMaskingPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType:
        ContextDecodeMultiTensor<ActionBuildContext, > + ContextEncodeMultiIndexI64<ActionBuildContext>,

{

    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.ppo_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextEncodeTensor<InfoSetConversionContext>
    + MaskingInformationSetActionMultiParameter<DP, ActionBuildContext>,
    InfoSetConversionContext: TensorEncoding,
    ActionBuildContext: MultiTensorDecoding + MultiTensorIndexI64Encoding
    + tensor_data::ActionTensorFormat<Vec<Tensor>>,
> LearningNetworkPolicy<DP> for PolicyMaskingPpoMultiDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
    where
        <DP as DomainParameters>::ActionType: ContextDecodeMultiTensor<ActionBuildContext>
        + ContextEncodeMultiIndexI64<ActionBuildContext>
{
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
        self.ppo_train_on_trajectories(trajectories, reward_f)
    }
}