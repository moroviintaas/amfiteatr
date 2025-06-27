use amfiteatr_classic::agent::{LocalHistoryConversionToTensor, LocalHistoryInfoSet, LocalHistoryInfoSetNumbered};
use amfiteatr_classic::ClassicActionTensorRepresentation;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_rl::policy::{ConfigPPO, PolicyDiscretePPO};
use amfiteatr_rl::tch;
use amfiteatr_rl::tch::{nn, Tensor};
use amfiteatr_rl::tch::nn::{Adam, OptimizerConfig, Path, VarStore};
use amfiteatr_rl::tensor_data::TensorEncoding;
use amfiteatr_rl::torch_net::{A2CNet, ActorCriticOutput, NeuralNetTemplate, TensorActorCritic};
use crate::common::ComputeDevice;
use crate::replicators::aliases::ReplPPO;
use crate::replicators::error::ReplError;
use crate::replicators::model::{ReplDomain, ReplicatorNetworkPolicy};
use crate::replicators::options::ReplicatorOptions;


pub fn create_ppo_policy(layer_sizes: &[i64], var_store: VarStore, options: &ReplicatorOptions, _agent_num: i32)
    -> Result<ReplPPO, AmfiteatrError<ReplDomain>>{
    //todo this should be macro probably one day
    let info_set_repr = LocalHistoryConversionToTensor::new(options.number_of_rounds);
    let input_shape = info_set_repr.desired_shape_flatten();
    let hidden_layers = layer_sizes;
    let output_shape = 2i64;
    let config = ConfigPPO::default();
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

    let network = network_pattern.get_net_closure();
    let optimiser = Adam::default().build(&var_store, options.learning_rate)
        .map_err(|origin|AmfiteatrError::Tensor {error: TensorError::Torch {origin: format!("{origin}"), context: "Creating optimser".to_string() } })?;
    let net = A2CNet::new(var_store, network);

    Ok(ReplPPO::new(config, net, optimiser, info_set_repr, ClassicActionTensorRepresentation{}))


}

pub trait LearningPolicyBuilder{
    type ReplPolicy: ReplicatorNetworkPolicy;

    fn build(self) -> Result<Self::ReplPolicy, ReplError>;

}

pub struct ReplPolicyBuilderPPO<'a>{

    pub options: &'a ReplicatorOptions,
    pub agent_id: u32
}

impl<'a> LearningPolicyBuilder for ReplPolicyBuilderPPO<'a>{
    type ReplPolicy = PolicyDiscretePPO<ReplDomain, LocalHistoryInfoSetNumbered, LocalHistoryConversionToTensor, ClassicActionTensorRepresentation>;

    fn build(self) -> Result<Self::ReplPolicy, ReplError> {
        let var_store = VarStore::new(match self.options.device{
            ComputeDevice:: Cpu => tch::Device::Cpu,
            ComputeDevice::Cuda => tch::Device::cuda_if_available()

        });

        Ok(create_ppo_policy(&self.options.layer_sizes[..], var_store, self.options, 0)?)


    }
}