use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use log::debug;
use amfiteatr_classic::agent::{LocalHistoryConversionToTensor, LocalHistoryInfoSetNumbered};
use amfiteatr_classic::ClassicActionTensorRepresentation;
use amfiteatr_classic::scheme::AgentNum;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_core::util::TensorboardSupport;
use amfiteatr_rl::policy::{ConfigPPO, PolicyDiscretePPO};
use amfiteatr_rl::tch;
use amfiteatr_rl::tch::{nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, OptimizerConfig, VarStore};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{build_network_model_ac, A2CNet, Layer, NeuralNet, TensorActorCritic, VariableStorage};
use crate::common::ComputeDevice;
use crate::replicators::aliases::ReplPPO;
use crate::replicators::error::ReplError;
use crate::replicators::model::{ReplScheme, ReplicatorNetworkPolicy};
use crate::replicators::options::ReplicatorOptions;


pub fn create_ppo_policy(layer_sizes: Vec<i64>, var_store: VarStore, options: &ReplicatorOptions, agent_num: AgentNum)
    -> Result<ReplPPO, AmfiteatrError<ReplScheme>>{
    //todo this should be macro probably one day
    let info_set_repr = LocalHistoryConversionToTensor::new(options.number_of_rounds);
    //let input_shape = info_set_repr.desired_shape_flatten();
    let input_shape = info_set_repr.desired_shape().to_vec();
    //let hidden_layers = layer_sizes;
    let output_shape = 2i64;
    let config = ConfigPPO{
        vf_coef: options.value_loss_coef,
        ent_coef: options.entropy_coefficient,
        ..Default::default()
    };

    let mut layers = Vec::new();
    for l in layer_sizes{
        layers.push(Layer::Linear(l));
        layers.push(Layer::Tanh);
    }
    /*
    let mut config = ConfigPPO::default();
    config.vf_coef = options.value_loss_coef;
    config.ent_coef = options.entropy_coefficient;

     */
    /*

    let network_pattern = NeuralNetTemplate::new(|path| {
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(path / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(path / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(path / "al", input_shape, output_shape, Default::default()),
                 nn::linear(path / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(path / "al", ld, output_shape, Default::default()),
                 nn::linear(path / "cl", ld, 1, Default::default()))
            }
        };
        let device = path.device();
        {move |xs: &Tensor|{
            if seq.is_empty(){
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            } else {
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }
        }}
    });


     */

    /*
    let operator = Box::new(move | tensor: &Tensor|{
        let mut seq = nn::seq();
        let mut last_dim = None;
        if !hidden_layers.is_empty(){
            let mut ld = hidden_layers[0];

            last_dim = Some(ld);
            seq = seq.add(nn::linear(var_store.root() / "INPUT", input_shape, ld, Default::default()));

            for (i, ld_new) in hidden_layers.iter().enumerate().skip(1){
                seq = seq.add(nn::linear(var_store.root() / &format!("h_{:}", i+1), ld, *ld_new, Default::default()))
                    .add_fn(|xs| xs.tanh());

                ld = *ld_new;
                last_dim = Some(ld);
            }
        }
        let (actor, critic) = match last_dim{
            None => {
                (nn::linear(var_store.root() / "al", input_shape, output_shape, Default::default()),
                 nn::linear(var_store.root() / "cl", input_shape, 1, Default::default()))
            }
            Some(ld) => {
                (nn::linear(var_store.root() / "al", ld, output_shape, Default::default()),
                 nn::linear(var_store.root() / "cl", ld, 1, Default::default()))
            }
        };
        let device = var_store.device();
            if seq.is_empty(){
                TensorActorCritic {critic: tensor.apply(&critic), actor: tensor.apply(&actor)}
            } else {
                let xs = tensor.to_device(device).apply(&seq);
                TensorActorCritic {critic: xs.apply(&critic), actor: xs.apply(&actor)}
            }

    });

     */
    //let network = network_pattern.get_net_closure();
    let model = build_network_model_ac(layers, input_shape, 1, &var_store.root());
    let optimiser = Adam::default().build(&var_store, options.learning_rate)
        .map_err(|origin|AmfiteatrError::Tensor {error: TensorError::Torch {origin: format!("{origin}"), context: "Creating optimser".to_string() } })?;
    //let net = A2CNet::new_concept_3(var_store, |_| operator);

    let net = NeuralNet::new(VariableStorage::Owned(var_store), model);

    let mut policy = ReplPPO::new(config, net, optimiser, info_set_repr, ClassicActionTensorRepresentation{});
    if let Some(tboard_path_base) = &options.agent_tboard{

        //if let Some(t)
        //let path = PathBuf::from(format!("{}/{}", tboard_path_base, agent_num));
        let path: PathBuf = [tboard_path_base.as_ref(), std::path::Path::new(&format!("{}", agent_num))].iter().collect();
        debug!("Registering tboard for agent {agent_num} with path {:?}", path);
        policy.add_tboard_directory(path.as_path())?;
    }
    Ok(policy)


}

pub trait LearningPolicyBuilder{
    type ReplPolicy: ReplicatorNetworkPolicy;

    fn build(self) -> Result<Self::ReplPolicy, ReplError>;

}

pub struct ReplPolicyBuilderPPO<'a>{

    pub options: &'a ReplicatorOptions,
    pub agent_id: u32
}

impl LearningPolicyBuilder for ReplPolicyBuilderPPO<'_>{
    type ReplPolicy = PolicyDiscretePPO<ReplScheme, LocalHistoryInfoSetNumbered, LocalHistoryConversionToTensor, ClassicActionTensorRepresentation>;

    fn build(self) -> Result<Self::ReplPolicy, ReplError> {
        let var_store = VarStore::new(match self.options.device{
            ComputeDevice:: Cpu => tch::Device::Cpu,
            ComputeDevice::Cuda => tch::Device::cuda_if_available()

        });

        Ok(create_ppo_policy((&self.options.layer_sizes[..]).to_vec(), var_store, self.options, self.agent_id)?)


    }
}