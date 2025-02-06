use std::fmt::Debug;
use std::marker::PhantomData;
use getset::{Getters, Setters};
use amfiteatr_core::agent::{InformationSet, Policy};
use amfiteatr_core::domain::DomainParameters;
use crate::tch;
use crate::tch::nn::Optimizer;
use crate::tch::Tensor;
use crate::tensor_data::{ConversionFromMultipleTensors, ConversionFromTensor, ConversionToTensor, CtxTryFromMultipleTensors, CtxTryIntoTensor};
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
    tensor_kind: tch::kind::Kind,
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
            tensor_kind,
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
            .map(|t| t.softmax(-1, self.tensor_kind));
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

