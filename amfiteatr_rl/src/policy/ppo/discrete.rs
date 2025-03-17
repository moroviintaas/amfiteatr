use std::fmt::Debug;
use std::marker::PhantomData;
use tch::nn::{Optimizer, VarStore};
use tch::{Kind, Tensor};
use amfiteatr_core::agent::{AgentStepView, AgentTrajectory, InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use crate::error::AmfiteatrRlError;
use crate::policy::{ConfigPPO, LearningNetworkPolicy, PolicyHelperPPO};
use crate::{tensor_data, MaskingInformationSetAction};
use crate::tensor_data::{ActionTensorFormat, ContextTryFromTensor, ContextTryIntoIndexI64, ContextTryIntoTensor, ConversionFromMultipleTensors, ConversionFromTensor, ConversionToIndexI64, ConversionToTensor};
use crate::torch_net::{ActorCriticOutput, NeuralNet, NeuralNetActorCritic, TensorActorCritic};

pub struct PolicyPpoDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromTensor,
>{
    config: ConfigPPO,
    network: NeuralNetActorCritic,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_conversion_context: InfoSetConversionContext,
    action_build_context: ActionBuildContext,

    exploration: bool,

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
        + tensor_data::ActionTensorFormat<Tensor>,
> PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{

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
            info_set_conversion_context,
            action_build_context,
            exploration: true,
        }
    }
    fn batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &Tensor,
        action_category_mask_batches: Option<&Tensor>,
        action_forward_mask_batches: Option<&Tensor>,
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
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperPPO<DP> for PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;

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
        Ok(network_output.actor.f_softmax(-1, self.config.tensor_kind)
            .map_err(|e| TensorError::from_tch_with_context(e, "Calculating action distribution (ppo_dist)".into()))?)


    }

    fn is_action_masking_supported(&self) -> bool {
        false
    }

    fn generate_action_masks(&self, information_set: &Self::InfoSet) -> Result<<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, AmfiteatrError<DP>> {
        Err(AmfiteatrError::Custom("Action masking is not supported.".into()))
    }

    fn ppo_exploration(&self) -> bool {
        self.exploration
    }

    fn ppo_try_action_from_choice_tensor(&self, choice_tensor: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        Ok(<DP::ActionType>::try_from_tensor(choice_tensor, &self.action_build_context)?)
    }

    fn ppo_vectorise_action_and_create_category_mask(&self, action: &DP::ActionType)
        -> Result<
            (<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
             <Self::NetworkOutput as ActorCriticOutput>::ActionTensorType
            ), AmfiteatrError<DP>> {
        let act_i = action.try_to_index(&self.action_build_context)?;


        Ok((Tensor::from(act_i), Tensor::from(true)))
    }

    fn ppo_batch_get_actor_critic_with_logprob_and_entropy(
        &self,
        info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType, action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>
    {

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

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<DP> for PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.ppo_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext>,
    InfoSetConversionContext: ConversionToTensor,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicy<DP> for PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>
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

pub struct PolicyMaskingPpoDiscrete<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: ConversionToTensor ,
    ActionBuildContext: ConversionFromTensor  + ConversionToIndexI64,
>{
    pub base: PolicyPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>,

}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: ConversionToTensor ,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyMaskingPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>{

    pub fn new(
        config: ConfigPPO,
        network: NeuralNetActorCritic,
        optimizer: Optimizer,
        info_set_conversion_context: InfoSetConversionContext,
        action_build_context: ActionBuildContext,
    ) -> Self{
        Self{
            base: PolicyPpoDiscrete::new(config, network, optimizer, info_set_conversion_context, action_build_context),
        }
    }
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
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: ConversionToTensor ,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> PolicyHelperPPO<DP> for PolicyMaskingPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>
{
    type InfoSet = InfoSet;
    type InfoSetConversionContext = InfoSetConversionContext;
    type ActionConversionContext = ActionBuildContext;
    type NetworkOutput = TensorActorCritic;

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
        let softmax = network_output.actor.f_softmax(-1, self.base.config.tensor_kind)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax)".into()))?;

        let masks = info_set.try_build_mask(&self.base.action_build_context)?;

        let product = softmax.f_mul(&masks)
            .map_err(|e| TensorError::from_tch_with_context(e, "PPO distribution (softmax * mask)".into())
        )?;
        Ok(product)

        /*
        Ok(
            network_output.actor.f_softmax(-1, self.base.config.tensor_kind)?.f_mul(
                &info_set.try_build_mask(&self.base.action_build_context)?
            )?

        )

         */
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

    fn ppo_batch_get_actor_critic_with_logprob_and_entropy(
        &self, info_set_batch: &Tensor,
        action_param_batches: &<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType,
        action_category_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>,
        action_forward_mask_batches: Option<&<Self::NetworkOutput as ActorCriticOutput>::ActionTensorType>
    ) -> Result<(Tensor, Tensor, Tensor), AmfiteatrError<DP>>
    {
        self.base.ppo_batch_get_actor_critic_with_logprob_and_entropy(info_set_batch, action_param_batches, action_category_mask_batches, action_forward_mask_batches)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: ConversionToTensor ,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> Policy<DP> for PolicyMaskingPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>
{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.ppo_select_action(state)
    }
}

impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ContextTryIntoTensor<InfoSetConversionContext> + MaskingInformationSetAction<DP, ActionBuildContext>,
    InfoSetConversionContext: ConversionToTensor ,
    ActionBuildContext: ConversionFromTensor + ConversionToIndexI64
    + tensor_data::ActionTensorFormat<Tensor>,
> LearningNetworkPolicy<DP> for PolicyMaskingPpoDiscrete<DP, InfoSet, InfoSetConversionContext, ActionBuildContext>
where
    <DP as DomainParameters>::ActionType:
    ContextTryFromTensor<ActionBuildContext, > + ContextTryIntoIndexI64<ActionBuildContext>{
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