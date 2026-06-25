use std::sync::{Arc, Mutex};
use amfiteatr_proc_macro::mcp_policy;
use amfiteatr_rl::error::AmfiteatrRlError;
use amfiteatr_rl::policy::{ConfigA2C, ConfigPPO, PolicyDiscreteA2C, PolicyDiscretePPO, PolicyMaskingDiscreteA2C, PolicyMaskingDiscretePPO};
use amfiteatr_rl::tch::Device;
use amfiteatr_rl::tch::nn::{Adam, VarStore};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{build_network_model_ac_discrete, Layer, NeuralNet, VariableStorage};
use crate::connect_four::agent::{ConnectFourActionTensorRepresentation, ConnectFourTensorReprD1};
use crate::connect_four::common::ConnectFourScheme;
use crate::connect_four::agent::ConnectFourInfoSet;
use amfiteatr_rl::tch::nn::OptimizerConfig;

use rmcp::ErrorData as McpError;
use rmcp::model::{
    GetPromptRequestParams,
    GetPromptResult,
    PaginatedRequestParams,
    ListPromptsResult,
    Meta,
};
use rmcp::service::RequestContext;
use rmcp::RoleServer;
pub type C4A2CPolicy = PolicyDiscreteA2C<ConnectFourScheme, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4A2CPolicyMasking = PolicyMaskingDiscreteA2C<ConnectFourScheme, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4PPOPolicy = PolicyDiscretePPO<ConnectFourScheme, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4PPOPolicyShared = Arc<std::sync::Mutex<C4PPOPolicy>>;
pub type C4PPOPolicyMasking = PolicyMaskingDiscretePPO<ConnectFourScheme, ConnectFourInfoSet, ConnectFourTensorReprD1, ConnectFourActionTensorRepresentation>;
#[allow(dead_code)]
pub type C4PPOPolicyMaskingShared = Arc<std::sync::Mutex<C4PPOPolicyMasking>>;

pub fn build_ppo_policy(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicy, AmfiteatrRlError<ConnectFourScheme>>{
    Ok(build_ppo_policy_masking(layer_sizes, device, config, learning_rate)?.base)

}
#[allow(dead_code)]
pub fn build_ppo_masking_policy_shared(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicyMaskingShared, AmfiteatrRlError<ConnectFourScheme>>{
    Ok(Arc::new(Mutex::new(build_ppo_policy_masking(layer_sizes, device, config, learning_rate)?)))

}
#[allow(dead_code)]
pub fn build_ppo_policy_shared(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicyShared, AmfiteatrRlError<ConnectFourScheme>>{
    Ok(Arc::new(Mutex::new(build_ppo_policy(layer_sizes, device, config, learning_rate)?)))
}


pub fn build_a2c_policy(layer_sizes: &[i64], device: Device, config: ConfigA2C, learning_rate: f64) -> Result<C4A2CPolicy, AmfiteatrRlError<ConnectFourScheme>>{
    let var_store = VarStore::new(device);
    let input_shape = ConnectFourTensorReprD1{}.desired_shape();
    //let hidden_layers = layer_sizes.to_vec();

    let mut layers = Vec::new();
    for l in layer_sizes{
        layers.push(Layer::Linear(*l));
        layers.push(Layer::Tanh);
    }

    let model = build_network_model_ac_discrete(layers, input_shape.to_vec(), 7, &var_store.root());
    //let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, learning_rate)?;

    let net = NeuralNet::new(VariableStorage::Owned(var_store), model);

    Ok(PolicyDiscreteA2C::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{}
    )
    )
}
#[allow(dead_code)]
pub fn build_a2c_policy_masking(layer_sizes: &[i64], device: Device, config: ConfigA2C, learning_rate: f64) -> Result<C4A2CPolicyMasking, AmfiteatrRlError<ConnectFourScheme>>{
    let var_store = VarStore::new(device);

    let input_shape = ConnectFourTensorReprD1{}.desired_shape();
    //let hidden_layers = layer_sizes.to_vec();

    let mut layers = Vec::new();
    for l in layer_sizes{
        layers.push(Layer::Linear(*l));
        layers.push(Layer::Tanh);
    }

    let model = build_network_model_ac_discrete(layers, input_shape.to_vec(), 7, &var_store.root());
    //let net = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, learning_rate)?;

    let net = NeuralNet::new(VariableStorage::Owned(var_store), model);

    Ok(PolicyMaskingDiscreteA2C::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{}
    )
    )
}
pub fn build_ppo_policy_masking(layer_sizes: &[i64], device: Device, config: ConfigPPO, learning_rate: f64) -> Result<C4PPOPolicyMasking, AmfiteatrRlError<ConnectFourScheme>>{
    let var_store = VarStore::new(device);
    let input_shape = ConnectFourTensorReprD1{}.desired_shape();
    //let hidden_layers = layer_sizes.to_vec();

    let mut layers = Vec::new();
    for l in layer_sizes{
        layers.push(Layer::Linear(*l));
        layers.push(Layer::Tanh);
    }



    let model = build_network_model_ac_discrete(layers, input_shape.to_vec(), 7, &var_store.root());
    let optimiser = Adam::default().build(&var_store, learning_rate)?;

    let net = NeuralNet::new(VariableStorage::Owned(var_store), model);

    Ok(PolicyMaskingDiscretePPO::new(
        config,
        net,
        optimiser,
        ConnectFourTensorReprD1{},
        ConnectFourActionTensorRepresentation{})
    )
}


#[mcp_policy(target = std::sync::Arc<std::sync::Mutex<C4PPOPolicyMasking>>, scheme = ConnectFourScheme, seed_type = () )]
pub struct McpPolicyPPOConnectFour;