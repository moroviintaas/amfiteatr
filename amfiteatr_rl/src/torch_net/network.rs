use std::marker::PhantomData;
use tch::{nn, Device, TchError, Tensor};
use tch::nn::{Optimizer, OptimizerConfig, Path,  VarStore};
use crate::torch_net::{
    MultiDiscreteTensor,
    NetOutput,
    TensorActorCritic,
    TensorMultiParamActorCritic
};
use serde::{Serialize, Deserialize};
use std::default::Default;
use std::error::Error;
use std::sync::{Arc, Mutex};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::scheme::Scheme;
// /// Structure wrapping [`VarStore`] and network closure used to build neural network based function.
// /// Examples in [`tch`](https://github.com/LaurentMazare/tch-rs) show how neural networks are used.

/*
pub struct NeuralNet<Output: NetOutput>{
    //net: NetworkModel<Output>
    //net: Box<dyn Fn(&Tensor) -> Output + Send >,
    operator: Box<dyn Fn(&VarStore, &Tensor) -> Output + Send>,
    var_store: VarStore,
    //_input: PhantomData<Input>,
}
*/

/// Struct holding exclusively owned [`VarStore`] or shared with [`Arc`]`<`[`Mutex`]`<`[`VarStore`]`>>`.
/// /// Examples in [`tch`](https://github.com/LaurentMazare/tch-rs) show how neural networks are used.
pub enum VariableStorage{
    Owned(VarStore),
    Shared(Arc<Mutex<VarStore>>)
}
/// Struct for storing [`NetworkModel`] (shape of network) and related data like [`VarStore`] or [`Device`].
pub struct NeuralNet<Output: NetOutput>{
    net: NetworkModel<Output>,
    device: Device,

    //device: tch::Device,
    var_store: VariableStorage,
}




/// [`NeuralNet`] with single `Tensor` as output.
pub type NeuralNet1 = NeuralNet<Tensor>;
/// [`NeuralNet`] with tuple `(Tensor, Tensor)` as output.
pub type NeuralNet2 = NeuralNet< (Tensor, Tensor)>;
/// [`NeuralNet`] with [`TensorActorCritic`] as output.
pub type A2CNet = NeuralNet< TensorActorCritic>;
/// [`NeuralNet`] with single `Tensor` as output. Same as [`NeuralNet1`].
pub type QValueNet = NeuralNet< Tensor>;

/// [`NeuralNet`] with [`TensorActorCritic`] as output, just like [`A2CNet`].
pub type NeuralNetActorCritic= NeuralNet< TensorActorCritic>;
/// /// [`NeuralNet`] with [`TensorMultiParamActorCritic`] as output (for multiple actor params).
pub type NeuralNetMultiActorCritic = NeuralNet< TensorMultiParamActorCritic>;

impl NeuralNetMultiActorCritic{


}
/// [`NeuralNet`] with multiple tensor output (first version used for actor critic networks, now they have dedicated structure of [`TensorActorCritic`].
pub type MultiDiscreteNet = NeuralNet< MultiDiscreteTensor>;

/// To construct network you need `VarStore` and function (closure) taking `nn::Path` as argument
/// and constructs function (closure) which applies network model tp `Tensor` producing `NetOutput`,
/// in following example `NetOutput` of `(Tensor, Tensor)` is used for purpose of actor-critic method.
/// # Example:
/// ```
/// use tch::{Device, nn, Tensor};
/// use tch::nn::{Adam, VarStore};
/// use amfiteatr_rl::torch_net::{build_network_model_ac_discrete, A2CNet, NetworkModel, NeuralNet2, TensorActorCritic, VariableStorage};
/// use amfiteatr_rl::torch_net::Layer::Linear;
/// let device = Device::cuda_if_available();
/// let var_store = VarStore::new(device);
/// let number_of_actions = 33_i64;
/// let model = build_network_model_ac_discrete(vec![Linear(128), Linear(128)], vec![16], 128, &var_store.root());
///
/// let neural_net = A2CNet::new(VariableStorage::Owned(var_store), model);
///
/// let optimizer = neural_net.build_optimizer(Adam::default(), 0.01);
/// ```
impl< Output: NetOutput> NeuralNet< Output>{




    pub fn new(variable_store: VariableStorage, model: NetworkModel<Output>) -> Self{
        let device = match &variable_store{
            VariableStorage::Shared( s) => {
                let vs = s.as_ref().lock().unwrap();
                vs.device()
            },
            VariableStorage::Owned(vs) => vs.device(),


        };
        Self{var_store: variable_store, net:model, device}
    }

    pub fn set_gradient_tracing(&mut self, set_tracing: bool){
        match &mut self.var_store{
            VariableStorage::Owned(v) => match set_tracing{
                true => v.unfreeze(),
                false => v.freeze()
            }
            VariableStorage::Shared(s) => {
                if let Ok(mut vs) = s.as_ref().lock(){
                    match set_tracing{
                        true => vs.unfreeze(),
                        false => vs.freeze()
                    }
                }
            }
        }
    }


    /*
    pub fn var_store_mut<S: Scheme>(&mut self) -> Result<&mut VarStore, AmfiteatrError<S>>{
        match &mut self.var_store{
            VariableStorage::Owned(vs) => Ok(vs),
            VariableStorage::Shared(s) => s.get_mut().map_err(|e|{
                AmfiteatrError::Lock { description: format!("{e}"), object: "VarStore".to_string() }
            })
        }
    }

     */

    pub fn on_internal_var_store_mut<S: Scheme, O, E, F: Fn(&mut VarStore) -> Result<O, E>>(&mut self, f: F) -> Result<O, AmfiteatrError<S>>
    where AmfiteatrError<S>: From<E>{
        match &mut self.var_store {
            VariableStorage::Owned(vs) => Ok(f(vs)?),
            VariableStorage::Shared(avs) => {
                let rvs = avs.as_ref().try_lock();
                match rvs{
                    Err(e) => Err(AmfiteatrError::Lock {description: format!("{e}"), object: "VarStore".to_string()}),
                    Ok(mut vs) => Ok(f(&mut vs)?)
                }
            }
        }
    }








    /// Build optimiser for network, given `OptimizerConfig`. Uses [`VarStore`] stored in [`NeuralNet`] struct;
    pub fn build_optimizer<OptC: OptimizerConfig>
        (&self, optimiser_config: OptC, learning_rate: f64) -> Result<Optimizer, TchError>{

        match &self.var_store{
            VariableStorage::Owned(vs) => optimiser_config.build(vs, learning_rate),

            VariableStorage::Shared(vsa) => {
                let vs = vsa.lock().unwrap();
                optimiser_config.build(&vs, learning_rate)
            }
        }
        //optimiser_config.build(&self.var_store, learning_rate)
    }

    /*

    /// Returns reference to internal network offering `Tensor -> Output` application.
    /// # Example:
    /// ```
    /// use tch::{Device, Kind, nn, Tensor};
    /// use tch::nn::VarStore;
    /// use amfiteatr_rl::torch_net::NeuralNet;
    /// let device = Device::cuda_if_available();
    /// let var_store = VarStore::new(device);
    /// let neural_net = NeuralNet::new(var_store, Box::new(|vs: &VarStore, tensor: &Tensor|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(vs.root() / "input", 32, 4, Default::default()));
    ///     tensor.apply(&seq)
    ///
    /// }));
    /// let input_tensor = Tensor::zeros(32, (Kind::Float, device));
    /// let output_tensor = (neural_net.operator())(neural_net.var_store(),  &input_tensor);
    /// assert_eq!(output_tensor.size(), vec![4]);
    /// ```

    pub fn operator(&self) -> &(dyn Fn(&VarStore, &Tensor) -> Output + Send){&self.operator}
    */

    pub fn net(&self) -> &NetworkModel<Output>{
        &self.net
    }


    pub fn device(&self) -> Device{
        self.device
    }

    pub fn variable_store(&self) -> &VariableStorage{
        &self.var_store
    }

}


pub type NetworkModel< Output> = Box<dyn Fn(&Tensor) -> Output + Send>;






/// This is wrapper for closure representing neural network. In [`NeuralNet`]
/// network was paired with local [`VarStore`], so when you want to construct two identically
/// structured neural networks you have to clone defined closure and use these cloned closures to
/// construct two networks. This helper structure allows declaring closure and then using it to
/// build many neural networks
#[deprecated(since = "0.13.0", note = "Network shape is now maintained as [`NetworkModel`] which can be
 constructed in dynamic way. For example Actor Critic network can be built with function [`build_network_model_ac`]")]
pub struct NeuralNetTemplate<
    Output: NetOutput,
    N: 'static + Send + Fn(&Tensor) -> Output,
    F: Fn(&Path) -> N + Clone>{

    _output: PhantomData<Output>,
    _net_closure: PhantomData<N>,
    net_closure: F,


}



#[allow(deprecated)]
impl<
    O: NetOutput,
    N: 'static + Send + Fn(&Tensor) -> O,
    F: Fn(&Path) -> N + Clone>
NeuralNetTemplate<O, N, F>{


    pub fn new(net_closure: F) -> Self{
        Self{
            #[allow(deprecated)]
            _output: PhantomData,
            #[allow(deprecated)]
            _net_closure: PhantomData,
            #[allow(deprecated)]
            net_closure
        }
    }

    //pub fn with_layers(layers: &[Layer]) ->




    /// Use template to  work with network shapes (it allows compiler to derive `tensor` type and you do not have
    /// to do this yourself).
    /// ```
    /// use tch::{Device, nn, Tensor};
    /// use tch::nn::VarStore;
    /// use amfiteatr_rl::torch_net::{NeuralNet, NeuralNetTemplate, VariableStorage};
    ///
    /// let nc = NeuralNetTemplate::new(|path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 32, 4, Default::default()));
    ///     move |tensor: &Tensor| {tensor.apply(&seq)}
    /// });
    /// let closure = nc.get_net_closure();
    ///
    /// let (vs1, vs2) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
    /// let model = Box::new(closure(&vs1.root()));
    ///
    /// let net1 = NeuralNet::new(VariableStorage::Owned(vs1), model);
    /// // let net2 = NeuralNet::new(vs2, closure);
    ///
    ///
    /// ```

    pub fn get_net_closure(&self) -> F{
        #[allow(deprecated)]
        self.net_closure.clone()
    }


}



/// This is enumeration type to help constructing sequential neural network models.
/// The idea is to describe hidden network layers as a sequence.
/// The [`Sequential`](tch::nn::Sequential)  has disadvantage because it's implementation of [`forward`](tch::nn::Module)
/// returns single tensor which may be problem in multi output networks e.g. Actor-Critic.
/// The list will be expanded in the future.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer{
    /// ReLu layer
    Relu,
    /// Hyperbolic tangens layer
    Tanh,
    /// Fully connected linear layer with *N* nodes
    Linear(i64),
}



/// If you want to create network model for actor critic that has input shape 5x4 (to be flattened),
/// and hidden layers: Linear(32), ReLu, Linear(32), Relu and output in 4 possible actions you would do something like that:
///
/// ```
///
/// use tch::Device;
/// use tch::nn::VarStore;
/// use amfiteatr_rl::torch_net::{build_network_model_ac_discrete, Layer, NeuralNet};
/// let var_store = VarStore::new(Device::Cpu);
///
/// let input_shape = vec![5,4];
/// let actor_shape = 4;
/// let layers = vec![Layer::Linear(32), Layer::Relu, Layer::Linear(32)];
///
/// let model = build_network_model_ac_discrete(layers, input_shape, actor_shape, &var_store.root());
/// // Then we could build network that is used in policy:
/// //let net = NeuralNet::new(var_store, model);
///
/// ```
pub fn build_network_model_ac_discrete(layers: Vec<Layer>, input_shape:  Vec<i64>, actor_shape: i64, path: &nn::Path)
                                       -> NetworkModel< TensorActorCritic>{

        let mut seq = nn::seq();
        let mut current_dim = input_shape[0];
        let mut next_dim = input_shape[0];
        let layer = layers[0].clone();
        seq = match layer {
            Layer::Relu => { seq.add_fn(|x| x.relu()) },
            Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
            Layer::Linear(output) => {
                next_dim = output;
                seq.add(
                    nn::linear(path / "0", current_dim, output, Default::default())
                )
            }
        };
        current_dim = next_dim;
        for (i, new_layer) in layers.iter().enumerate().skip(1) {
            seq = match new_layer {
                Layer::Relu => { seq.add_fn(|x| x.relu()) },
                Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
                Layer::Linear(output) => {
                    next_dim = *output;
                    seq.add(
                        nn::linear(path / &format!("lin_{}", i), current_dim, *output, Default::default()))
                }
            };
            current_dim = next_dim;
        }
        #[cfg(feature = "log_trace")]
        log::trace!("Building actor critic network with actor shape: {actor_shape}, critic shape: 1");
        let (actor, critic) = (
            nn::linear(path/ "actor", current_dim, actor_shape, Default::default()),
            nn::linear(path / "critic", current_dim, 1, Default::default())
        );



        let device = path.device();
    Box::new(move |tensor: &Tensor| {
        let xs = tensor.to_device(device).apply(&seq);

        TensorActorCritic {
            critic: xs.apply(&critic),
            actor: xs.apply(&actor),
        }
    })
}

/// Similar to [`build_network_model_ac_discrete`] but for multi actor networks:
/// ```
///
/// use tch::Device;
/// use tch::nn::VarStore;
/// use amfiteatr_rl::torch_net::{build_network_model_ac_multidiscrete, Layer, NeuralNet};
/// let var_store = VarStore::new(Device::Cpu);
///
/// let input_shape = vec![5,4];
/// let actor_shapes = vec![4, 6, 2];
/// let layers = vec![Layer::Linear(32), Layer::Relu, Layer::Linear(32)];
///
/// let model = build_network_model_ac_multidiscrete(layers, input_shape, actor_shapes, &var_store.root());
/// // This produces network that accepts 5x4 tensor, then applies layers Linear(32), Relu, and Linear(32).
/// // Then there are 3 parallel (not sequential layers) to create 3 actor distributions of respectively 4, 6 and 2 classes.
/// // The critic output is traditionally single (float) value.
/// // Then we could build network that is used in policy:
/// //let net = NeuralNet::new(var_store, model);
///
/// ```

pub fn build_network_model_ac_multidiscrete(layers: Vec<Layer>, input_shape:  Vec<i64>, actor_shapes: Vec<i64>, path: &nn::Path)
                                            -> NetworkModel<TensorMultiParamActorCritic>{

    let mut seq = nn::seq();
    let mut current_dim = input_shape[0];
    let mut next_dim = input_shape[0];
    let layer = layers[0].clone();
    seq = match layer {
        Layer::Relu => { seq.add_fn(|x| x.relu()) },
        Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
        Layer::Linear(output) => {
            next_dim = output;
            seq.add(
                nn::linear(path / "0", current_dim, output, Default::default())
            )
        }
    };
    current_dim = next_dim;
    for (i, new_layer) in layers.iter().enumerate().skip(1) {
        seq = match new_layer {
            Layer::Relu => { seq.add_fn(|x| x.relu()) },
            Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
            Layer::Linear(output) => {
                next_dim = *output;
                seq.add(
                    nn::linear(path / &format!("lin_{}", i), current_dim, *output, Default::default()))
            }
        };
        current_dim = next_dim;
    }
    let (actor_params, critic) = (
        actor_shapes.iter().enumerate().map(|(n,a)|
            nn::linear(path/ &format!("actor_{n}"), current_dim, *a, Default::default())
        ).collect::<Vec<_>>(),


        nn::linear(path / "critic", current_dim, 1, Default::default())
    );



    let device = path.device();
    Box::new(move |tensor: &Tensor| {
        let xs = tensor.to_device(device).apply(&seq);

        TensorMultiParamActorCritic {
            critic: xs.apply(&critic),
            actor: actor_params.iter().map(|a| xs.apply(a)).collect(),
        }
    })
}





