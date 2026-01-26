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
use std::sync::{Arc, Mutex};

/// Structure wrapping [`VarStore`] and network closure used to build neural network based function.
/// Examples in [`tch`](https://github.com/LaurentMazare/tch-rs) show how neural networks are used.
/// This structure shortens some steps of setting and provides some helpful functions - especially
/// [`build_optimiser`](NeuralNet::build_optimizer).

/*
pub struct NeuralNet<Output: NetOutput>{
    //net: NetworkModel<Output>
    //net: Box<dyn Fn(&Tensor) -> Output + Send >,
    operator: Box<dyn Fn(&VarStore, &Tensor) -> Output + Send>,
    var_store: VarStore,
    //_input: PhantomData<Input>,
}
*/

pub enum VariableStorage{
    Owned(VarStore),
    Shared(Arc<Mutex<VarStore>>)
}
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

pub type NeuralNetActorCritic= NeuralNet< TensorActorCritic>;
pub type NeuralNetMultiActorCritic = NeuralNet< TensorMultiParamActorCritic>;

impl NeuralNetMultiActorCritic{


}

pub type MultiDiscreteNet = NeuralNet< MultiDiscreteTensor>;

/// To construct network you need `VarStore` and function (closure) taking `nn::Path` as argument
/// and constructs function (closure) which applies network model tp `Tensor` producing `NetOutput`,
/// in following example `NetOutput` of `(Tensor, Tensor)` is used for purpose of actor-critic method.
/// # Example:
/// ```
/// use tch::{Device, nn, Tensor};
/// use tch::nn::{Adam, VarStore};
/// use amfiteatr_rl::torch_net::{A2CNet, NeuralNet2, TensorActorCritic};
/// let device = Device::cuda_if_available();
/// let var_store = VarStore::new(device);
/// let number_of_actions = 33_i64;
/// let neural_net = A2CNet::new_concept_1(var_store, Box::new(
///     move |vs: &VarStore, tensor: &Tensor| {
///         let seq = nn::seq()
///         .add(nn::linear(vs.root() / "input", 16, 128, Default::default()))
///         .add(nn::linear(vs.root() / "hidden", 128, 128, Default::default()));
///     let actor = nn::linear(vs.root() / "al", 128, number_of_actions, Default::default());
///     let critic = nn::linear(vs.root() / "cl", 128, 1, Default::default());
///     let device = vs.device();
///     let xs = tensor.to_device(device).apply(&seq);
///         //(xs.apply(&critic), xs.apply(&actor))
///     TensorActorCritic{critic: xs.apply(&critic), actor: xs.apply(&actor)}
///
///     }
/// ));
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
    /// Build optimiser for network, given `OptimizerConfig`. Uses [`VarStore`] stored in [`NeuralNet`] struct;
    pub fn build_optimizer<OptC: OptimizerConfig>
        (&self, optimiser_config: OptC, learning_rate: f64) -> Result<Optimizer, TchError>{

        optimiser_config.build(&self.var_store, learning_rate)
    }


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


pub type NetworkModel< Output: NetOutput> = Box<dyn Fn(&Tensor) -> Output + Send>;






/// This is wrapper for closure representing neural network. In [`NeuralNet`]
/// network is pinned with local [`VarStore`], so when you want to construct two identically
/// structured neural networks you have to clone defined closure and use these cloned closures to
/// construct two networks. This helper structure allows declaring closure and then using it to
/// build many neural networks
#[deprecated(since = "0.13.0", note = "Network model is now maintained as [`NetworkModel`] which can be
 constructed in dynamic way. For example Actor Critic network can be built with function [`build_network_model_ac`]")]
pub struct NeuralNetTemplate<
    Output: NetOutput,
    N: 'static + Send + Fn(&Tensor) -> Output,
    F: Fn(&Path) -> N + Clone>{

    _output: PhantomData<Output>,
    _net_closure: PhantomData<N>,
    net_closure: F,


}




impl<
    O: NetOutput,
    N: 'static + Send + Fn(&Tensor) -> O,
    F: Fn(&Path) -> N + Clone>
NeuralNetTemplate<O, N, F>{


    pub fn new(net_closure: F) -> Self{
        Self{
            _output: PhantomData,
            _net_closure: PhantomData,
            net_closure
        }
    }

    //pub fn with_layers(layers: &[Layer]) ->


    /// When you  do something like this it will fail to compile
    /// ```compile_fail
    /// use tch::{Device, nn, Tensor};
    /// use tch::nn::VarStore;
    /// use amfiteatr_rl::torch_net::{NeuralNet, NeuralNetTemplate};
    /// let closure = |path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 32, 4, Default::default()));
    ///     move |tensor| {tensor.apply(&seq)}
    /// };
    /// let (vs1, vs2) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
    /// let net1 = NeuralNet::new(vs1, closure.clone());
    /// let net2 = NeuralNet::new(vs2, closure);
    ///
    ///
    /// ```
    /// Use template to make it work (it allows compiler to derive `tensor` type and you do not have
    /// to do this yourself).
    /// ```
    /// use tch::{Device, nn};
    /// use tch::nn::VarStore;
    /// use amfiteatr_rl::torch_net::{NeuralNet, NeuralNetTemplate};
    ///
    /// let nc = NeuralNetTemplate::new(|path|{
    ///     let seq = nn::seq()
    ///         .add(nn::linear(path / "input", 32, 4, Default::default()));
    ///     move |tensor| {tensor.apply(&seq)}
    /// });
    /// let closure = nc.get_net_closure();
    ///
    ///
    /// let (vs1, vs2) = (VarStore::new(Device::Cpu), VarStore::new(Device::Cpu));
    ///
    /// // let net1 = NeuralNet::new(vs1, closure);
    /// // let net2 = NeuralNet::new(vs2, closure);
    ///
    ///
    /// ```
    pub fn get_net_closure(&self) -> F{
        self.net_closure.clone()
    }


}



//To be added in 0.13, I hope
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer{
    //Reshape(Vec<i64>),
    Relu,

    Tanh,
    Linear(i64),
}

/*
pub struct NeuralNetTemplateDiscreteAC{
    //func: Box<dyn Fn(Tensor) -> TensorActorCritic>
    layers: Vec<Layer>,
    input_shape: Vec<i64>,
    actor_shape: i64,



}


impl NeuralNetActorCritic{

    pub fn new_from_layers(vs: VarStore, layers: &[Layer], input_shape: &[i64], actor_shape: i64) -> Self{
        let mut seq = nn::seq();
        let mut current_dim = input_shape[0];
        let mut next_dim = input_shape[0];
        let mut last_dim = None;
        if !layers.is_empty() {
            let mut layer = layers[0].clone();
            last_dim = Some(layer.clone());
            seq = match layer {
                Layer::Relu => { seq.add_fn(|x| x.relu()) },
                Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
                Layer::Linear(output) => {
                    next_dim = output;
                    seq.add(
                        nn::linear(vs.root() / "0", current_dim, output, Default::default())
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
                            nn::linear(vs.root() / &format!("lin_{}", i), current_dim, *output, Default::default()))
                    }
                };
                current_dim = next_dim;
            }
            let (actor, critic) = (
                nn::linear(vs.root() / "actor", current_dim, actor_shape, Default::default()),
                nn::linear(vs.root() / "critic", current_dim, 1, Default::default())
            );
            //{
            let p = vs.root();
            let device = vs.device();
            let operation = Box::new(
                |xs: &Tensor| { ;
                    let xs = xs.to_device(device).apply(&seq);
                    TensorActorCritic {
                        critic: xs.apply(&critic),
                        actor: xs.apply(&actor),
                    }
                }
                //}
            );
            Self{var_store: vs, net: operation }


        } else{
            panic!("Empty layer sequence")
        }
    }
}

*/

/// If you want to create network model for actor critic that has input shape 5x4 (to be flattened),
/// and hidden layers: Linear(32), ReLu, Linear(32), Relu and output in 4 possible actions you would do something like that:
///
/// ```
///
/// use tch::Device;
/// use tch::nn::VarStore;
/// use amfiteatr_rl::torch_net::{build_network_model_ac, Layer, NeuralNet};
/// let var_store = VarStore::new(Device::Cpu);
///
/// let input_shape = vec![5,4];
/// let actor_shape = 4;
/// let layers = vec![Layer::Linear(32), Layer::Relu, Layer::Linear(32)];
///
/// let model = build_network_model_ac(layers, input_shape, actor_shape, &var_store.root());
/// // Then we could build network that is used in policy:
/// //let net = NeuralNet::new(var_store, model);
///
/// ```
pub fn build_network_model_ac(layers: Vec<Layer>, input_shape:  Vec<i64>, actor_shape: i64, path: &nn::Path)
    -> NetworkModel< TensorActorCritic>{

    //Box::new(move |tensor: &Tensor| {
        let mut seq = nn::seq();
        let mut current_dim = input_shape[0];
        let mut next_dim = input_shape[0];
        //let mut last_dim = None;
        let layer = layers[0].clone();
        //last_dim = Some(layer.clone());
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
/*
pub fn build_network_operator_ac(layers: Vec<Layer>, input_shape:  Vec<i64>, actor_shape: i64)
                                 ->  Box<dyn Fn(&VarStore, &Tensor) -> TensorActorCritic + Send>
{



    if !layers.is_empty() {
        Box::new(move |vs, tensor|{
            let mut seq = nn::seq();
            let mut current_dim = input_shape[0];
            let mut next_dim = input_shape[0];
            //let mut last_dim = None;
            let layer = layers[0].clone();
            //last_dim = Some(layer.clone());
            seq = match layer {
                Layer::Relu => { seq.add_fn(|x| x.relu()) },
                Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
                Layer::Linear(output) => {
                    next_dim = output;
                    seq.add(
                        nn::linear(vs.root() / "0", current_dim, output, Default::default())
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
                            nn::linear(vs.root() / &format!("lin_{}", i), current_dim, *output, Default::default()))
                    }
                };
                current_dim = next_dim;
            }
            let (actor, critic) = (
                nn::linear(vs.root() / "actor", current_dim, actor_shape, Default::default()),
                nn::linear(vs.root() / "critic", current_dim, 1, Default::default())
            );



            let device = vs.device();
            let xs = tensor.to_device(device).apply(&seq);
            TensorActorCritic {
                critic: xs.apply(&critic),
                actor: xs.apply(&actor),
            }


        })



    } else{
        panic!("Empty layer sequence")
    }

}


 */
