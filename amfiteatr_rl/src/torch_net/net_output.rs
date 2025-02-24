use tch::{Device, Kind, TchError, Tensor};
use tch::Kind::Float;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::ConvertError;
use amfiteatr_core::reexport::nom::Parser;
use crate::error::AmfiteatrRlError;
use crate::error::AmfiteatrRlError::ZeroBatchSize;


/// Marker trait describing output format for neural network. For example Actor-Critic methods output
/// two Tensors (one for Action distribution and other to evaluate current state (information set).
pub trait NetOutput{}

/// Struct to aggregate both actor and critic output tensors from network.
pub struct TensorA2C{
    pub critic: Tensor,
    pub actor: Tensor
}

/// Struct to aggregate output for actor-critic networks with multi parameter actor
pub struct TensorCriticMultiActor{
    pub critic: Tensor,
    pub actor: Vec<Tensor>
}

impl TensorCriticMultiActor{

    /// # Input
    /// `forward_masks: Option<&[Tensor]>`, where `Tensor` has shape `[BATCH_SIZE, CATEGORY_SIZE`] (works for single param sample probability), and `Vec::len` is the number of categories.
    /// `reverse_masks: Option<&[Tensor]>` where `Tensor` has shape `[BATCH_SIZE]`, (enables/disables whole category), and `Vec::len` is the number of categories.
    /// # Output
    ///  `Result<Tensor, TchError>`
    ///  Tensor ha size BATCH_SIZE * NUMBER_OF_CATEGORIES
    /// # Example
    /// Assume we have 2 game steps by actor critic
    /// We have then 2 critic values (Tensor 2x1)
    /// We have 3 parameters of action, of 4,3 and 2 respectively.
    /// Then first actor is shape 2x4.
    /// The second actor is in shape 2x3.
    /// The third us 2x2.
    ///
    /// Now masks:
    /// Assume we took actions:
    /// 1. [0,2,0] (Index 0 on first param, index 2 on second, index 0 in third]
    /// 2. [2,-,1] (Index 2 on first action, second param is unknown since it not a param wof action type 0, index 1 on third
    ///
    /// ```
    ///
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_rl::torch_net::TensorCriticMultiActor;
    /// let critic = Tensor::from_slice(&[0.7, 0.21]); //first critic calculation, second criric calculation
    /// let actor = vec![
    ///     {
    ///         let state1_dist0 = Tensor::from_slice(&[-3.0, -4.0, -1.0, -1.0]); //parameter 1 on state1
    ///         let state2_dist0 = Tensor::from_slice(&[-5.0, -2.0, -4.0, -1.0]); //parameter 1 on state2
    ///         Tensor::vstack(&[state1_dist0, state2_dist0])
    ///     },   //first parameter
    ///     {
    ///         let state1_dist1 = Tensor::from_slice(&[-3.0, -4.0, -1.0]); //parameter 2 on state1
    ///         let state2_dist1 = Tensor::from_slice(&[-5.0, -2.0, -4.0]); //parameter 2 on state2
    ///         Tensor::vstack(&[state1_dist1, state2_dist1])
    ///     },
    ///     {
    ///         let state1_dist2 = Tensor::from_slice(&[-3.0, -4.0]); //parameter 3 on state1
    ///         let state2_dist2 = Tensor::from_slice(&[-5.0, -2.0]); //parameter 3 on state2
    ///         Tensor::vstack(&[state1_dist2, state2_dist2])
    ///     },
    /// ];
    /// let mca = TensorCriticMultiActor{critic,actor,};
    /// let reverse_masks = vec![
    ///     Tensor::from_slice(&[true, true]),
    ///     Tensor::from_slice(&[true, false]),
    ///     Tensor::from_slice(&[true, true]),
    ///
    /// ];
    /// let forward_masks = vec![
    ///     {
    ///         let state1_dist0 = Tensor::from_slice(&[true, false, true, false]);
    ///         let state2_dist0 = Tensor::ones(&[4], (Kind::Bool, Device::Cpu));
    ///
    ///         Tensor::vstack(&[state1_dist0, state2_dist0])
    ///     },
    ///     {
    ///         let state1_dist1 = Tensor::ones(&[3], (Kind::Bool, Device::Cpu));
    ///         let state2_dist1 = Tensor::from_slice(&[false, false, true]);
    ///         Tensor::vstack(&[state1_dist1, state2_dist1])
    ///     },
    ///     {
    ///         let state1_dist2 = Tensor::from_slice(&[false, true]);
    ///         let state2_dist2 = Tensor::ones(&[2], (Kind::Bool, Device::Cpu));
    ///         Tensor::vstack(&[state1_dist2, state2_dist2])
    ///     },
    ///
    /// ];
    /// let entropy_unmasked = mca.batch_entropy_masked(None, None).unwrap();
    /// let entropy_masked_actions = mca.batch_entropy_masked(Some(&forward_masks), None).unwrap();
    /// let entropy_masked_reverse = mca.batch_entropy_masked(None, Some(&reverse_masks)).unwrap();
    /// let entropy_masked = mca.batch_entropy_masked(Some(&forward_masks), Some(&reverse_masks)).unwrap();
    /// assert_eq!(entropy_unmasked.size2().unwrap(), (2,3));
    ///
    /// ```
    pub fn batch_entropy_masked(&self, forward_masks: Option<&[Tensor]>, reverse_masks: Option<&[Tensor]>) -> Result<Tensor, TchError>{
        
        /* first assume that we have criric -> [c0, c1, c2, ...] (1 dim) and 
            actor is vec![Tensor(a11, a12, ...), Tensor(a21, a22, ...), ...]
            many steps
            
        
         */
        /*
        let p_log_p: Vec<Tensor> = self.actor.iter().map(|ac|{
            Tensor::einsum("ij, jk -> ik", &[ac.f_softmax(-1, Float)?, ac.f_log_softmax(-1, Float)?], None);
        }).collect();


         */
        let mut p_log_p = Vec::new();
        for a in self.actor.iter(){
            p_log_p.push(Tensor::f_einsum("ij, ij -> ij", &[a.f_softmax(-1, Float)?, a.f_log_softmax(-1, Float)?], None::<i64>)?)
        }

        let mut elements = p_log_p;
        if let Some(fm) = forward_masks{
            for (i,  e) in elements.iter_mut().enumerate(){
                *e = e.f_where_self(&fm[i], &Tensor::from(0.0))?
            }
        }
        let elems1:Vec<Tensor> = match reverse_masks{
            Some(mut rm) => {
                let mut v = Vec::new();
                for (i,e) in elements.into_iter().enumerate(){
                    let t = [&e, &rm[i]];
                    v.push(Tensor::f_einsum("ij,i->ij", &t, Option::<i64>::None)?);
                }
                v
                 /*
                 elements.into_iter().enumerate().map(|(i, e)|{

                     let t = vec![e, rm[i]];
                     Tensor::f_einsum("ij,i->ij", &t, Option::<i64>::None)
                 }).collect()?

                  */
            },
            None => elements
        };
        /*
        if let Some(rm) = reverse_masks{
            for (i,  e) in elements.iter_mut().enumerate(){

                *e = Tensor::f_einsum("ij,i->ij", &[ e, &rm[i]], Option::<i64>::None)?
            }
        }

         */

        let mut sums = Vec::new();
        for t in elems1.into_iter(){
            sums.push(t.f_sum_dim_intlist(
                Some(-1), false, Kind::Float
            )?)
        }
        /*
        let sums: Vec<Tensor> = elems1.iter().map(|e|
            e.f_sum_dim_intlist(
                Some(-1), false, Kind::Float
            )
        ).collect();

         */

        let t_cats = Tensor::f_vstack(&sums)?;
        Ok(t_cats.transpose(0,1) * Tensor::from(-1.0))
        /*
        let sum = t_cats.f_sum_dim_intlist(
            Some(0),
           false,
            Kind::Float
        )?;
        Ok(sum * Tensor::from(-1.0 * sums.len() as f32))

         */


    }

    /// # Input
    /// `param_indices`: `Vec<Tensor>`, where `Tensor` is in shape `[BATCH_SIZE]` and `Vec::len` is the number of categories.
    /// `param_masks`: `Option<Vec<Tensor>>`, where `Tensor` is in shape `[BATCH_SIZE]` and `Vec::len` is the number of categories.
    /// # Output
    ///  `Result<Tensor, TchError>`
    ///  Tensor ha size BATCH_SIZE * NUMBER_OF_CATEGORIES
    /// # Example
    /// Assume we have 2 game steps by actor critic
    /// We have then 2 critic values (Tensor 2x1)
    /// We have 3 parameters of action, of 4,3 and 2 respectively.
    /// Then first actor is shape 2x4.
    /// The second actor is in shape 2x3.
    /// The third us 2x2.
    ///
    /// Now masks:
    /// Assume we took actions:
    /// 1. `[0,2,0]` (Index 0 on first param, index 2 on second, index 0 in third]
    /// 2. `[2,-,1]` (Index 2 on first action, second param is unknown since it not a param wof action type 0, index 1 on third
    ///
    /// Now we ask what was a log probability of these taking action `[0,2,0]` and `[2,-,1]` in two respective batch steps.
    /// So in first step, the probability is `p(A) = p0(0) * p1(2) * p2(0)`.
    /// (We need to sample first param with p0(0) probability, second with p1(2) and third with `p2(0)`
    /// and log probability is `lp(A) = lp0(0) + lp1(2) + lp2(0)`.
    ///
    /// Now what if some param is not known - like in second batch step.
    /// In action sampling the parameter was sampled it was 0,1 or 2.
    /// However it was not used in action construction due to some game logic.
    /// > E.g, when actor decides to shoot he chooses direction and distance (balistic), but when he chooses
    /// to look (scan) he just need to select direction.
    /// > The action created is Look(Direction), we lose information about distance param, as it is not used.
    /// When we calculate probability it is now `p(B) = p0(2) * p1(Omega) * p2(1)`.
    /// And `p1(Omega) = 1.0`.
    /// In log prob it is `lp(B) = lp0(2) + lp1(Omega) + lp2(1)`. While `lp1(Omega) = 0`.
    ///
    ///
    ///
    /// ```
    ///
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_core::demo::DemoDomain;
    /// use amfiteatr_rl::torch_net::TensorCriticMultiActor;
    /// let critic = Tensor::from_slice(&[0.7, 0.21]); //first critic calculation, second criric calculation
    /// let actor = vec![
    ///     {
    ///         let state1_dist0 = Tensor::from_slice(&[-3.0, -4.0, -1.0, -1.0]); //parameter 1 on state1
    ///         let state2_dist0 = Tensor::from_slice(&[-5.0, -2.0, -4.0, -1.0]); //parameter 1 on state2
    ///         Tensor::vstack(&[state1_dist0, state2_dist0])
    ///     },   //first parameter
    ///     {
    ///         let state1_dist1 = Tensor::from_slice(&[-3.0, -4.0, -1.0]); //parameter 2 on state1
    ///         let state2_dist1 = Tensor::from_slice(&[-5.0, -2.0, -4.0]); //parameter 2 on state2
    ///         Tensor::vstack(&[state1_dist1, state2_dist1])
    ///     },
    ///     {
    ///         let state1_dist2 = Tensor::from_slice(&[-3.0, -4.0]); //parameter 3 on state1
    ///         let state2_dist2 = Tensor::from_slice(&[-5.0, -2.0]); //parameter 3 on state2
    ///         Tensor::vstack(&[state1_dist2, state2_dist2])
    ///     },
    /// ];
    /// let mca = TensorCriticMultiActor{critic,actor,};
    /// let reverse_masks = vec![
    ///     Tensor::from_slice(&[true, true]),
    ///     Tensor::from_slice(&[true, false]),
    ///     Tensor::from_slice(&[true, true]),
    ///
    /// ];
    /// let action_params_selected_tensors = vec![
    ///     Tensor::from_slice(&[0i64, 2i64]),
    ///     Tensor::from_slice(&[2i64, 1i64]),
    ///     Tensor::from_slice(&[0i64, 1i64]),
    /// /*
    ///     {
    ///         let s1 = Tensor::from(0i64);
    ///         let s2 = Tensor::from(2i64);
    ///         Tensor::vstack(&[s1, s2])
    ///     },
    ///     {
    ///         let s1 = Tensor::from(2i64);
    ///         let s2 = Tensor::from(1i64);
    ///         //Tensor::from_slice(&[2i64, 1])
    ///         Tensor::vstack(&[s1, s2])
    ///     },
    ///     {
    ///         let s1 = Tensor::from(0i64);
    ///         let s2 = Tensor::from(1i64);
    ///         //Tensor::from_slice(&[0i64, 1])
    ///         Tensor::vstack(&[s1, s2])
    ///     },
    /// */
    ///
    /// ];
    /// let probs_unmasked = mca.batch_log_probability_of_action::<DemoDomain>(&action_params_selected_tensors, None).unwrap();
    /// let probs_masked= mca.batch_log_probability_of_action::<DemoDomain>(&action_params_selected_tensors, Some(&reverse_masks)).unwrap();
    /// assert_eq!(&probs_unmasked.size(), &[2]);
    /// let unmasked: Vec<f32> = probs_unmasked.try_into().unwrap();
    /// let masked: Vec<f32> = probs_masked.try_into().unwrap();
    /// assert!(masked[1] > unmasked[1]);
    /// ```
    pub fn batch_log_probability_of_action<DP: DomainParameters>(&self, param_indices: &[Tensor], param_masks: Option<&[Tensor]>)
        -> Result<Tensor, AmfiteatrRlError<DP>>{


        if param_indices.len() != self.actor.len(){
            return Err(AmfiteatrRlError::MismatchedLengthsOfData {
                shape1: param_indices.len(),
                shape2: self.actor.len(),
                context: "batch_log_probability_of_action: number of actor parameters is different than provided param indices".into()
            })
        }

        let probs: Vec<Tensor> = self.actor.iter().enumerate().map(|(i, a)|{

                a.f_log_softmax(-1, Kind::Float)?
                .f_gather(1, &param_indices[i].f_unsqueeze(1)?, false)?
                .f_flatten(0, -1)
        }).collect::<Result<Vec<Tensor>, TchError>>()?;

        let log_probs_vec = match param_masks{
            None => {
                probs
            }
            Some(masks) => {
                let masked_param_log_probs = probs.iter().zip(masks.iter())
                    .map(|(logp, mask)|{
                        let masked_log_probs = Tensor::f_einsum("i,i -> i", &[logp, mask], Option::<i64>::None);
                        masked_log_probs
                    }).collect::<Result<Vec<Tensor>, TchError>>()?;
                masked_param_log_probs
            }
        };

        let batch_size = log_probs_vec.get(0)
            .ok_or_else(|| ZeroBatchSize {
                context: "batch_log_probability_of_action".into()
            })
            .and_then(|t| t.size1()
                .map_err(|tch| AmfiteatrRlError::Torch {
                    source: tch,
                    context: "batch_log_probability_of_action".into()}))?;


        let mut sum = Tensor::f_zeros(batch_size, (Kind::Float, self.critic.device()))
            .map_err(|tch| AmfiteatrRlError::Torch{
                source: tch,
                context: "batch_log_probability_of_action (zero tensor)".into()
            })?;


        for t in log_probs_vec{
            sum = sum.f_add(&t)?;
        }




        Ok(sum)

    }
}

pub type MultiDiscreteTensor = Vec<Tensor>;

impl NetOutput for MultiDiscreteTensor{}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}
impl NetOutput for TensorA2C{}

impl NetOutput for TensorCriticMultiActor{}


/// Converts tensor of shape (1,) and type i64 to i64. Technically it will work
/// with  shape (n,), but it will take the very first element. It is used to when we have single
/// value in Tensor that we want numeric.
///
/// Used usually when converting discrete distribution to action index.
/// Consider distribution of 4 actions: `[0.3, 0.5, 0.1, 0.1]`
/// 1. First we sample one number of `(0,1,2,3)` with probabilities above.
/// Let's say we sampled `1` (here it has 1/2 chance to be so);
/// 2. At this moment we have Tensor of shape `(1,)` of type `i64`. But we want just number `i64`.
/// 3. So we need to do this conversion;
///
/// # Example:
/// ```
/// use tch::Kind::{Double, Float};
/// use tch::Tensor;
/// use amfiteatr_rl::torch_net::index_tensor_to_i64;
/// let t = Tensor::from_slice(&[0.3f64, 0.5, 0.1, 0.1]);
/// let index_tensor = t.multinomial(1, true).softmax(-1, Double);
/// assert_eq!(index_tensor.size(), vec![1]);
/// let index = index_tensor_to_i64(&index_tensor, "context message if error").unwrap();
/// assert!(index >=0 && index <= 3);
/// ```
#[inline]
pub fn index_tensor_to_i64(tensor: &Tensor, additional_context: &str) -> Result<i64, ConvertError>{
    /*
    let v: Vec<i64> = match Vec::try_from(tensor){
        Ok(v) => v,
        Err(_) => {
            return Err(ConvertError::ActionDeserialize(format!("From tensor {} in context \"{}\"", tensor, additional_context)))
        }
    };
    Ok(v[0])

     */

    tensor.f_int64_value(&[0]).map_err(|e|{
        ConvertError::ConvertFromTensor(
            format!("From tensor {} in context \"{}\". The error itself: {}",
                    tensor, additional_context, e))
    })
}