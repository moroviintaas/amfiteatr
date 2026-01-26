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
pub struct NeuralNet<Output: NetOutput>{
    net: NetworkModel<Output>,

    //device: tch::Device,
    var_store: VarStore,
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


    /*
    pub fn new(var_store: VarStore, operator: Box<dyn Fn(&VarStore, &Tensor) -> Output + Send>) -> Self{
        NeuralNet { operator, var_store }
    }

     */
    //_input: Default::default()



    /*
    pub fn new( net_model: NetworkModel<Output>, var_store: VarStore) -> Self{
        NeuralNet {
            net: net_model

        }
    }

     */

    /*
    pub fn new_concept_1(ref_var_store: &VarStore, net_model: impl Fn(&tch::nn::Path) -> NetworkModel<Output>, ) -> Self{
        NeuralNet {
            net: net_model(&ref_var_store.root()),
            device: ref_var_store.device()

        }
    }

    pub fn new_concept_2(net_model: NetworkModel<Output>, device: &tch::Device) -> Self{
        NeuralNet{
            net: net_model,
            device: *device
        }
    }

     */

    /*
    pub fn new_concept_3(var_store: VarStore, net_model: impl Fn(&tch::nn::Path) -> NetworkModel< Output>) -> Self{
        let net = net_model(&var_store.root());
        Self{
            net, var_store
        }
    }

     */

    pub fn new_concept_4(var_store: VarStore, model: NetworkModel<Output>) -> Self{
        Self{var_store, net:model}
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
        self.var_store.device()
    }
    /*

    pub fn device(&self) -> Device{
        self.var_store.device()
    }

    pub fn var_store(&self) -> &VarStore{
        &self.var_store
    }
    pub fn var_store_mut(&mut self) -> &mut VarStore{
        &mut self.var_store
    }

     */
}


pub type NetworkModel< Output: NetOutput> = Box<dyn Fn(&Tensor) -> Output + Send>;






/// This is wrapper for closure representing neural network. In [`NeuralNet`]
/// network is pinned with local [`VarStore`], so when you want to construct two identically
/// structured neural networks you have to clone defined closure and use these cloned closures to
/// construct two networks. This helper structure allows declaring closure and then using it to
/// build many neural networks
pub struct NeuralNetTemplate<
    Output: NetOutput,
    N: 'static + Send + Fn(&Tensor) -> Output,
    F: Fn(&Path) -> N + Clone>{

    _output: PhantomData<Output>,
    _net_closure: PhantomData<N>,
    net_closure: F,


}



#[deprecated(since = "0.13.0", note = "Please build `NeuralNetTemplate` with function `new` using operator Box.\
The whole structure does not make sense anymore since NuralNet use `Box<dyn Fn(&VarStore, &Tensor) -> Output>` \
and not `Box<dyn Fn(&Tensor) -> Output + Send >` which led to problems with constructing dynamic network shape without templating them in closures.")]
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

/*
pub fn create_net_model_discrete_ac<'a>(vs: &'a VarStore, layers: &[Layer], input_shape: &[i64], actor_shape: i64)
    ->  Box<dyn Fn(&Tensor) -> TensorActorCritic + Send + 'a>
{

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
        Box::new(
            move |xs: &Tensor| {
                let device = vs.device();
                let xs = xs.to_device(device).apply(&seq);
                TensorActorCritic {
                    critic: xs.apply(&critic),
                    actor: xs.apply(&actor),
                }
            }
            //}
        )


    } else{
        panic!("Empty layer sequence")
    }

}

 */

/*
pub fn create_net_template_discrete_ac<'a>(layers: &[Layer], input_shape: &[i64], actor_shape: i64)
                                        ->  Box<dyn Fn(nn::Path) -> Box<dyn Fn(&Tensor) -> TensorActorCritic + 'a + Send>>
{



    if !layers.is_empty() {

        //{
        Box::new(|p: nn::Path|{
            let mut next_dim = input_shape[0];
            let mut last_dim = None;
            let mut seq = nn::seq();
            let mut current_dim = input_shape[0];
            let mut layer = layers[0].clone();
            last_dim = Some(layer.clone());
            seq = match layer {
                Layer::Relu => { seq.add_fn(|x| x.relu()) },
                Layer::Tanh => { seq.add_fn(|x| x.tanh()) },
                Layer::Linear(output) => {
                    next_dim = output;
                    seq.add(
                        nn::linear(p.clone() / "0", current_dim, output, Default::default())
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
                            nn::linear(p.clone() / &format!("lin_{}", i), current_dim, *output, Default::default()))
                    }
                };
                current_dim = next_dim;
            }
            let (actor, critic) = (
                nn::linear(p.clone() / "actor", current_dim, actor_shape, Default::default()),
                nn::linear(p.clone() / "critic", current_dim, 1, Default::default())
            );
            Box::new(
                move |xs: &Tensor| {
                    let device = p.device();
                    let xs = xs.to_device(device).apply(&seq);
                    TensorActorCritic {
                        critic: xs.apply(&critic),
                        actor: xs.apply(&actor),
                    }
                }
                //}
            )
        })



    } else{
        panic!("Empty layer sequence")
    }

}



 */
/*
impl NeuralNetTemplateDiscreteAC {
    pub fn new(layer_description: Vec<Layer>, input_shape: Vec<i64>, actor_shape: i64) -> Self{
        Self{
            layers: layer_description, input_shape,
            actor_shape,
        }
    }

    //pub fn get_closure<N: 'static + Send + Fn(&Tensor) -> TensorActorCritic>(&self)
    //    -> Box<dyn Fn(&tch::nn::Path) -> N>{

    pub fn get_model(&self, path: &nn::Path)
        ->  Box<dyn Fn(&Tensor) -> TensorActorCritic+ '_>
    {

        //! TODO non 1-d input (require flatten before linear
        //let network_pattern = Box::new(|path: &tch::nn::Path|{
            //Box::new(|t: tch::Tensor|{




            let mut seq = nn::seq();
            let mut current_dim = self.input_shape[0];
            let mut next_dim = self.input_shape[0];
            let mut last_dim = None;
            if !self.layers.is_empty() {
                let mut layer = self.layers[0];
                last_dim = Some(layer);
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
                for (i, new_layer) in self.layers.iter().enumerate().skip(1) {
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
                    nn::linear(path / "actor", current_dim, self.actor_shape, Default::default()),
                    nn::linear(path / "critic", current_dim, 1, Default::default())
                );
                //{
                Box::new(
                move |xs: &Tensor| {
                    let device = path.device();
                    let xs = xs.to_device(device).apply(&seq);
                    TensorActorCritic {
                        critic: xs.apply(&critic),
                        actor: xs.apply(&actor),
                    }
                }
                //}
            )


        } else{
                panic!("Empty layer sequence")
            }

        //});

        //})

        //network_pattern
    }
}
*/
