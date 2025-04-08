use std::fmt::Debug;
use tch::{Device, Kind, TchError, Tensor};
use tch::Kind::Float;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::{AmfiteatrError, ConvertError, DataError, TensorError};
use amfiteatr_core::error::DataError::LengthMismatch;
use crate::error::AmfiteatrRlError;
use crate::error::AmfiteatrRlError::ZeroBatchSize;


/// Marker trait describing output format for neural network. For example Actor-Critic methods output
/// two Tensors (one for Action distribution and other to evaluate current state (information set).
pub trait NetOutput{}


pub trait DeviceTransfer{

    fn move_to_device(self, device: Device) -> Self;
}

impl DeviceTransfer for Tensor{
    fn move_to_device(self, device: Device) -> Self {
        self.to_device(device)
    }
}

impl DeviceTransfer for Vec<Tensor>{
    fn move_to_device(self, device: Device) -> Self {
        self.into_iter().map(|x| x.move_to_device(device)).collect()
    }
}

impl DeviceTransfer for Vec<Vec<Tensor>>{
    fn move_to_device(self, device: Device) -> Self {
        let mut result = Vec::with_capacity(self.len());

        for v in self.into_iter(){
            result.push(v.into_iter().map(|x| x.to_device(device)).collect());
        }

        result
    }
}

/// Trait for data types returned by Actor-Critic networks
pub trait ActorCriticOutput : NetOutput + Debug{

    /// Type of actor data. Typically for discrete action space with one category it is `Tensor`
    /// For discrete actions with multiple parameters it will be `Vec<Tensor>`
    type ActionTensorType: Debug + DeviceTransfer;
    /// Vec type to construct Tensor batch, it will be typically `Vec<Tensor>` and `Vec<Vec<Tensor>>`,
    /// however this is not just `Vec<Self:TensorForm>` because on `Vec<Vec<_>>` it mess with dimensions.
    /// Outer is Param dimenstion Inner is Batch dimension. We later create param batches from continuous slice.
    type ActionBatchTensorType: Debug + DeviceTransfer;

    fn device(&self) -> Device;

   // fn move_to_device(&self, tensor: Self::ActionTensorType) -> Self::ActionTensorType;
   // fn move_batch_to_device(&self, tensor: Self::ActionBatchTensorType) -> Self::ActionBatchTensorType;

    /// Returns batched entropy of distribution.
    /// Use when this output is batched - every actor tensor has BATCH_DIMENSION (dimension 0).
    /// This outputs [`Tensor`] that for every member of batch returns entropy of categorical distribution build on actor.
    fn batch_entropy_masked(&self, action_masks: Option<&Self::ActionTensorType>, parameter_masks: Option<&Self::ActionTensorType>)
                            -> Result<Tensor, TchError>;

    /// Returns batched log probability of action.
    /// /// Use when this output is batched - every actor tensor has BATCH_DIMENSION (dimension 0).
    /// This outputs [`Tensor`] that for every member of batch returns log probability of action that is pointed by index..
    fn batch_log_probability_of_action<DP: DomainParameters>(&self, param_indices: &Self::ActionTensorType, action_masks: Option<&Self::ActionTensorType>, parameter_masks: Option<&Self::ActionTensorType>)
                                                             -> Result<Tensor, AmfiteatrError<DP>>;

    fn critic(&self) -> &Tensor;


    /// Pushes to every category vector new entry
    /// For example look at [`TensorMultiParamActorCritic::push_to_vec_batch`].
    fn push_to_vec_batch(vec_batch: &mut Self::ActionBatchTensorType, data: Self::ActionTensorType);

    /// This is analogous to [`Vec::append`] in the logic of [`ush_to_vec_batch`].
    fn append_vec_batch(dst: &mut Self::ActionBatchTensorType, source: &mut Self::ActionBatchTensorType);
    /*{

        for (c_src, c_dst) in source.iter_mut().zip(dst.iter_mut()){
            c_dst.append(c_src)
        }
    }*/
    /// Clears batch dimension.
    /// Refer to example in [`TensorMultiParamActorCritic::clear_batch_dim_in_batch`].
    fn clear_batch_dim_in_batch(vector: &mut Self::ActionBatchTensorType);
    /*{
        for v in vector.iter_mut(){
            v.clear()
        }
    }

     */
    fn param_dimension_size(&self) -> i64;
    fn stack_tensor_batch(batch: &Self::ActionBatchTensorType) -> Result<Self::ActionTensorType, ConvertError>;

    fn new_batch_with_capacity(number_of_params: usize, capacity: usize) -> Self::ActionBatchTensorType;

    fn perform_choice(dist: &Self::ActionTensorType,
                      apply: impl Fn(&Tensor) -> Result<Tensor, TchError> )
                      -> Result<Self::ActionTensorType, TensorError>;

    fn index_select(data: &Self::ActionTensorType, indices: &Tensor) -> Result<Self::ActionTensorType, TchError>;

    fn batch_get_logprob_and_entropy<DP: DomainParameters>(
        &self,
        action_param_batches: &Self::ActionTensorType,
        action_category_mask_batches: Option<&Self::ActionTensorType>,
        action_forward_mask_batches: Option<&Self::ActionTensorType>,
    ) -> Result<(Tensor, Tensor), AmfiteatrError<DP>>;
}

/// Struct to aggregate both actor and critic output tensors from network.
#[derive(Debug)]
pub struct TensorActorCritic {
    pub critic: Tensor,
    pub actor: Tensor
}

impl ActorCriticOutput for TensorActorCritic{
    type ActionTensorType = Tensor;
    type ActionBatchTensorType = Vec<Tensor>;

    fn device(&self) -> Device {
        self.critic.device()
    }

    /*
    fn move_to_device(&self, tensor: Self::ActionTensorType) -> Self::ActionTensorType {
        tensor.to_device(self.device())
    }

    fn move_batch_to_device(&self, tensor: Self::ActionBatchTensorType) -> Self::ActionBatchTensorType {
        tensor.into_iter().map(|x| x.to_device(self.device())).collect()
    }

     */


    fn batch_entropy_masked(&self, forward_masks: Option<&Self::ActionTensorType>, reverse_masks: Option<&Self::ActionTensorType>) -> Result<Tensor, TchError> {

        let p_log_p = Tensor::f_einsum("ij, ij -> ij", &[self.actor.f_softmax(-1, Float)?, self.actor.f_log_softmax(-1, Float)?], None::<i64>)?;
        let mut element = p_log_p;
        if let Some(fm) = forward_masks{
            element = element.f_where_self(&fm.ne(0.0), &Tensor::from(0.0))?
        }
        let elem = match reverse_masks{
            Some(rev_mask) => {
                let t = [&element, &rev_mask.to_device(self.device())];

                Tensor::f_einsum("ij, ij -> ij", &t, None::<i64>)?
            },
            None => element
        };

        let sum = elem.f_sum_dim_intlist(Some(-1), false, Kind::Float)?;

        Ok(sum * tch::Tensor::from(-1.0))
    }

    fn batch_log_probability_of_action<DP: DomainParameters>(&self, param_indices: &Self::ActionTensorType, action_masks: Option<&Self::ActionTensorType>, _param_masks: Option<&Self::ActionTensorType>) -> Result<Tensor, AmfiteatrError<DP>> {

        let log_probs = match action_masks{
            None => self.actor.f_log_softmax(-1, Float).map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch { origin: format!("{e}"), context: "batch_log_probability_of_action".to_string() },
            }),
            Some(mask) => {

                self.actor.f_log_softmax(-1, Float)
                    .and_then(|t| t.f_where_self(&mask.ne(0.0), &Tensor::from(1.0)))
                .map_err(|e| TensorError::Torch {
                    origin: format!("{e}"),
                    context: "batch_log_probability_of_action".into()
                }.into())

            }
        }?;

        let choice = log_probs.f_gather(1, param_indices, false)
            .and_then(|t| t.f_flatten(0, -1))
            .map_err(|e| AmfiteatrError::Tensor {
                error: TensorError::Torch {
                    origin: format!("{e}"),
                    context: "batch_log_probability_of_action (gathering)".into()
                }
            })?;

        match choice.size().first(){
            None => Err(ZeroBatchSize {
                context: format!("Batch log probability error. Indexes tensor: {param_indices}, choice tensor: {choice}")
            }.into()),
            Some(i) if i <= &0 => Err(ZeroBatchSize {
                context: format!("Batch log probability error. Indexes tensor: {param_indices}, choice tensor: {choice}")
            }.into()),
            Some(_) => Ok(choice)
        }

    }

    fn critic(&self) -> &Tensor {
        &self.critic
    }

    fn push_to_vec_batch(vec_batch: &mut Self::ActionBatchTensorType, data: Self::ActionTensorType) {
        vec_batch.push(data);
    }

    fn append_vec_batch(dst: &mut Self::ActionBatchTensorType, source: &mut Self::ActionBatchTensorType) {
        dst.append(source);
    }

    fn clear_batch_dim_in_batch(vector: &mut Self::ActionBatchTensorType) {

        vector.clear()
    }

    fn param_dimension_size(&self) -> i64 {
        1
    }

    fn stack_tensor_batch(batch: &Self::ActionBatchTensorType) -> Result<Self::ActionTensorType, ConvertError> {
        Tensor::f_vstack(batch)
            .map_err(|e| ConvertError::TorchStr {origin: format!("{e}")})
    }

    fn new_batch_with_capacity(_number_of_params: usize, capacity: usize) -> Self::ActionBatchTensorType {
        Vec::with_capacity(capacity)
    }

    fn perform_choice(dist: &Self::ActionTensorType, apply: impl Fn(&Tensor) -> Result<Tensor, TchError>) -> Result<Self::ActionTensorType, TensorError> {
        apply(dist).map_err(|e| TensorError::Torch { origin: format!("{e}"), context: "Performing action choice".into()})
    }

    fn index_select(data: &Self::ActionTensorType, indices: &Tensor) -> Result<Self::ActionTensorType, TchError> {
        data.f_index_select(0, indices)
    }

    fn batch_get_logprob_and_entropy<DP: DomainParameters>(
        &self,
        action_param_batches: &Self::ActionTensorType,
        action_category_mask_batches: Option<&Self::ActionTensorType>,
        action_forward_mask_batches: Option<&Self::ActionTensorType>
    ) -> Result<(Tensor, Tensor), AmfiteatrError<DP>> {
        //let critic_actor= .net()(info_set_batch);

        let batch_logprob = self.batch_log_probability_of_action::<DP>(
            action_param_batches,
            action_forward_mask_batches,
            action_category_mask_batches
        )?;
        let batch_entropy = self.batch_entropy_masked(
            action_forward_mask_batches,
            action_category_mask_batches

        ).map_err(|e|AmfiteatrError::Tensor {
            error: TensorError::Torch {
                context: "batch_get_actor_critic_with_logprob_and_entropy".into(),
                origin: format!("{e}")
            }
        })?;



        Ok((batch_logprob, batch_entropy))
    }
}

/// Struct to aggregate output for actor-critic networks with multi parameter actor
#[derive(Debug)]
pub struct TensorMultiParamActorCritic {
    pub critic: Tensor,
    pub actor: Vec<Tensor>
}


impl ActorCriticOutput for TensorMultiParamActorCritic {
    type ActionTensorType = Vec<Tensor>;
    type ActionBatchTensorType = Vec<Vec<Tensor>>;

    fn device(&self) -> Device {
        self.critic.device()
    }

    /*
    fn move_to_device(&self, tensor: Self::ActionTensorType) -> Self::ActionTensorType {
        tensor.into_iter().map(|x| x.to_device(self.device())).collect()
    }

    fn move_batch_to_device(&self, tensor: Self::ActionBatchTensorType) -> Self::ActionBatchTensorType {
        let mut result = Vec::with_capacity(tensor.len());

        for v in tensor.into_iter(){
            result.push(v.into_iter().map(|x| x.to_device(self.device())).collect());
        }

        result
    }

     */

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
    /// 1. `[0,2,0]` (Index 0 on first param, index 2 on second, index 0 in third]
    /// 2. `[2,-,1]` (Index 2 on first action, second param is unknown since it not a param wof action type 0, index 1 on third
    ///
    /// ```
    ///
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_rl::torch_net::{ActorCriticOutput, TensorMultiParamActorCritic};
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
    /// let mca = TensorMultiParamActorCritic{critic,actor,};
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
    fn batch_entropy_masked(&self, forward_masks: Option<&Self::ActionTensorType>, reverse_masks: Option<&Self::ActionTensorType>) -> Result<Tensor, TchError> {
        let mut p_log_p = Vec::new();
        for a in self.actor.iter(){


            p_log_p.push(Tensor::f_einsum("ij, ij -> ij",
                                          &[a.f_softmax(-1, Float)?, a.f_log_softmax(-1, Float)?],
                                          None::<i64>)?)
        }

        let mut elements = p_log_p;
        if let Some(fm) = forward_masks{
            for (i,  e) in elements.iter_mut().enumerate(){
                *e = e.f_where_self(&fm[i].to_device(self.device()).ne(0.0), &Tensor::from(0.0))?
            }
        }
        let elems1:Vec<Tensor> = match reverse_masks{
            Some(rm) => {
                let mut v = Vec::new();
                for (i,e) in elements.into_iter().enumerate(){
                    let t = [&e, &rm[i].to_device(self.device())];

                    v.push(Tensor::f_einsum("ij,i->ij", &t, Option::<i64>::None)?);
                }
                v

            },
            None => elements
        };


        let mut sums = Vec::new();
        for t in elems1.into_iter(){
            sums.push(t.f_sum_dim_intlist(
                Some(-1), false, Kind::Float
            )?)
        }


        let t_cats = Tensor::f_vstack(&sums)?;
        Ok(t_cats.transpose(0,1) * tch::Tensor::from(-1.0))
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
    /// However, it was not used in action construction due to some game logic.
    /// > E.g, when actor decides to shoot he chooses direction and distance (balistic), but when he chooses
    /// > to look (scan) he just needs to select direction.
    /// > The action created is Look(Direction), we lose information about distance param, as it is not used.
    /// > When we calculate probability it is now `p(B) = p0(2) * p1(Omega) * p2(1)`.
    /// > And `p1(Omega) = 1.0`.
    /// > In log prob it is `lp(B) = lp0(2) + lp1(Omega) + lp2(1)`. While `lp1(Omega) = 0`.
    /// ```
    ///
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_core::demo::DemoDomain;
    /// use amfiteatr_rl::torch_net::{ActorCriticOutput, TensorMultiParamActorCritic};
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
    /// let mca = TensorMultiParamActorCritic{critic,actor,};
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
    /// let probs_unmasked = mca.batch_log_probability_of_action::<DemoDomain>(&action_params_selected_tensors, None, None).unwrap();
    /// let probs_masked= mca.batch_log_probability_of_action::<DemoDomain>(&action_params_selected_tensors, None, Some(&reverse_masks)).unwrap();
    /// assert_eq!(&probs_unmasked.size(), &[2]);
    /// let unmasked: Vec<f32> = probs_unmasked.try_into().unwrap();
    /// let masked: Vec<f32> = probs_masked.try_into().unwrap();
    /// assert!(masked[1] > unmasked[1]);
    /// ```
    fn batch_log_probability_of_action<DP: DomainParameters>(&self, param_indices: &Self::ActionTensorType, action_masks: Option<&Self::ActionTensorType> , param_masks: Option<&Self::ActionTensorType>) -> Result<Tensor, AmfiteatrError<DP>> {
        if param_indices.len() != self.actor.len(){
            return Err(DataError::LengthMismatch {
                left: param_indices.len(),
                right: self.actor.len(),
                context: "batch_log_probability_of_action: number of actor parameters is different than provided param indices".into()
            }.into())
        }



        let choices: Vec<Tensor> = match action_masks{
            None => {
                self.actor.iter().enumerate().map(|(i, a)|{
                    a.f_log_softmax(-1, Kind::Float)?

                        .f_gather(1, &param_indices[i].to_device(self.device()).f_unsqueeze(1)?, false)?
                        .f_flatten(0, -1)
                }).collect::<Result<Vec<Tensor>, TchError>>().map_err(|e|{
                    TensorError::Torch { origin: format!("Torch error while calculating log probabilities of parameters. {}", e),
                        context: "Calculating batch log-probability of action (not masked)".into()}
                })?
            },
            Some(masks) => {

                if masks.len() != self.actor.len(){
                    return Err(LengthMismatch {
                        left: self.actor.len(),
                        right: masks.len(),
                        context: "Number of parameter tensors in actor and masks".to_string(),
                    }.into())
                }

                self.actor.iter().zip(masks).enumerate().map(|(i, (a, m))|{
                    //let Tensor::f_einsum("i,i->i", &[a,m], None::<i64>)
                    let log_softmax = a.f_log_softmax(-1, Kind::Float)?;
                    let masked = log_softmax.f_where_self(&m.ne(0.0), &Tensor::from(1.0))?;
                    masked.f_gather(1, &param_indices[i].to_device(self.device()).f_unsqueeze(1)?, false)?
                        .f_flatten(0, -1)

                }).collect::<Result<Vec<Tensor>, TchError>>().map_err(|e|{
                    TensorError::Torch { origin: format!("{}", e),
                        context: "Calculating batch log-probability of action (masked)".into()}
                })?
            }
        };


        #[cfg(feature = "log_trace")]
        for (i, t) in choices.iter().enumerate(){
            log::trace!("Category: {i:?} choice log probability tensor: {t}");
        }


        //let log_probs_vec = choices;

        let log_probs_vec = match param_masks{
            None => {
                choices
            }
            Some(masks) => {

                choices.iter().zip(masks.iter())
                    .map(|(logp, mask)|{
                        // if parameter is masked it means it was not used to create action therefore
                        // we set log probability to 0 (probability =1) so it does not change entropy
                        #[cfg(feature = "log_trace")]
                        log::trace!("Category:  mask: {mask}",);
                        Tensor::f_einsum("i,i -> i", &[logp, &mask.to_device(self.device())], Option::<i64>::None)
                    }).collect::<Result<Vec<Tensor>, TchError>>().map_err(|e|{
                    TensorError::Torch { origin: format!("{}", e),
                        context: "Calculating batch log-probability of action - during log probs of masked".into() }
                })?
            }
        };




        #[cfg(feature = "log_trace")]
        for (i, t) in log_probs_vec.iter().enumerate(){

            log::trace!("Category: {i:?} masked choice log probability tensor: {t}");
        }





        let batch_size = log_probs_vec.first()
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
            sum = sum.f_add(&t).map_err(|e|{
                TensorError::Torch {
                    origin: format!("{}", e),
                context:  "Calculating batch log-probability of action - during summing log probs".into() }
            })?;
        }

        #[cfg(feature = "log_trace")]
        log::trace!("Log probability sum = {sum}");




        Ok(sum)
    }

    fn critic(&self) -> &Tensor {
        &self.critic
    }



    /// ```
    /// // let's say we have two category action, currently three entries in batch
    /// use tch::Tensor;
    /// use amfiteatr_rl::torch_net::{ActorCriticOutput, TensorMultiParamActorCritic};
    /// let mut vec_batch = vec![ //category
    ///     vec![ // category 0, batch dimension
    ///         Tensor::from(1i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(0i64),
    ///     ]   ,
    ///     vec![ // category 1, batch dimension
    ///         Tensor::from(7i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(4i64),
    ///     ]   ,
    /// ];
    /// let push = vec![Tensor::from(3i64), Tensor::from(2i64)];
    ///
    /// let pushed = TensorMultiParamActorCritic::push_to_vec_batch(&mut vec_batch, push);
    ///
    /// let mut expected = vec![ //category
    ///     vec![ // category 0, batch dimension
    ///         Tensor::from(1i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(3i64),
    ///     ]   ,
    ///     vec![ // category 1, batch dimension
    ///         Tensor::from(7i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(4i64),
    ///         Tensor::from(2i64)
    ///         ]
    ///  ];
    /// assert_eq!(&expected, &vec_batch);
    /// ```
    fn push_to_vec_batch(vec_batch: &mut Self::ActionBatchTensorType, data: Self::ActionTensorType) {
        for (b_param, d_param) in vec_batch.iter_mut().zip(data.into_iter()){
            b_param.push(d_param);
        }
    }

    fn append_vec_batch(dst: &mut Self::ActionBatchTensorType, source: &mut Self::ActionBatchTensorType) {
        for (b_param, d_param) in dst.iter_mut().zip(source.iter_mut()){
            b_param.append(d_param);
        }
    }

    /// ```
    /// // let's say we have two category action, currently three entries in batch
    /// use tch::Tensor;
    /// use amfiteatr_rl::torch_net::{ActorCriticOutput, TensorMultiParamActorCritic};
    /// let mut vec_batch = vec![ //category
    ///     vec![ // category 0, batch dimension
    ///         Tensor::from(1i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(0i64),
    ///     ]   ,
    ///     vec![ // category 1, batch dimension
    ///         Tensor::from(7i64),
    ///         Tensor::from(0i64),
    ///         Tensor::from(4i64),
    ///     ]   ,
    /// ];
    ///
    /// TensorMultiParamActorCritic::clear_batch_dim_in_batch(&mut vec_batch);
    ///
    /// let mut expected = vec![ //category
    ///     Vec::<Tensor>::new(),
    ///     Vec::<Tensor>::new()
    ///  ];
    /// assert_eq!(&expected, &vec_batch);
    /// ```
    fn clear_batch_dim_in_batch(vector: &mut Self::ActionBatchTensorType) {
        for v in vector.iter_mut(){
            v.clear()
        }
    }

    fn param_dimension_size(&self) -> i64 {
        self.actor.len() as i64
    }

    fn stack_tensor_batch(batch: &Self::ActionBatchTensorType) -> Result<Self::ActionTensorType, ConvertError> {
        batch.iter().map(|param|{
            Tensor::f_vstack(param)
                .and_then(|stacked| stacked.f_squeeze())
                .map_err(|e| ConvertError::TorchStr { origin: format!("{e}") })
        }).collect::<Result<Vec<Tensor>, _>>()
    }

    fn new_batch_with_capacity(number_of_params: usize, capacity: usize) -> Self::ActionBatchTensorType {
        (0..number_of_params).map(|_i| Vec::<Tensor>::with_capacity(capacity)).collect()
    }

    fn perform_choice(dist: &Self::ActionTensorType, apply: impl Fn(&Tensor) -> Result<Tensor, TchError>) -> Result<Self::ActionTensorType, TensorError> {
        dist.iter().map(|d|{
            let is_all_zeros = d.eq_tensor(&Tensor::zeros_like(d)).all().int64_value(&[]);
            if is_all_zeros == 0{
                #[cfg(feature = "log_trace")]
                log::trace!("Performing choice on distribution: {d}");
                apply(d)


                
            } else {
                let t = Tensor::ones_like(d);
                apply(&t.softmax(-1, d.kind()))
            }

        }
            .map_err(|e| TensorError::Torch {
                origin: format!("{e}"),
                context: format!("Performing choice of parameter based on tensor {d}")})).collect::<Result<Vec<Tensor>,_>>()
    }

    fn index_select(data: &Self::ActionTensorType, indices: &Tensor) -> Result<Self::ActionTensorType, TchError>{
        data.iter().map(|c|{
            c.f_index_select(0, indices)
        }).collect::<Result<Vec<_>, TchError>>()
    }

    fn batch_get_logprob_and_entropy<DP: DomainParameters>(&self, action_param_batches: &Self::ActionTensorType, action_category_mask_batches: Option<&Self::ActionTensorType>, action_forward_mask_batches: Option<&Self::ActionTensorType>) -> Result<(Tensor, Tensor), AmfiteatrError<DP>> {
        let batch_logprob = self.batch_log_probability_of_action::<DP>(
            action_param_batches,
            action_forward_mask_batches,
            action_category_mask_batches
        )?;
        let batch_entropy = self.batch_entropy_masked(
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

        Ok((batch_logprob, batch_entropy_avg))
    }
}
/*
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

            },
            None => elements
        };


        let mut sums = Vec::new();
        for t in elems1.into_iter(){
            sums.push(t.f_sum_dim_intlist(
                Some(-1), false, Kind::Float
            )?)
        }


        let t_cats = Tensor::f_vstack(&sums)?;
        Ok(t_cats.transpose(0,1) * Tensor::from(-1.0))



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
        -> Result<Tensor, AmfiteatrError<DP>>{



        if param_indices.len() != self.actor.len(){
            return Err(DataError::LengthMismatch {
                left: param_indices.len(),
                right: self.actor.len(),
                context: "batch_log_probability_of_action: number of actor parameters is different than provided param indices".into()
            }.into())
        }

        let probs: Vec<Tensor> = self.actor.iter().enumerate().map(|(i, a)|{
            #[cfg(feature = "log_trace")]
            log::trace!("Sizes of actor category batch {:?} and param_indices{:?}", a.size(), param_indices[i].size());
                a.f_log_softmax(-1, Kind::Float)?
                .f_gather(1, &param_indices[i].f_unsqueeze(1)?, false)?
                .f_flatten(0, -1)
        }).collect::<Result<Vec<Tensor>, TchError>>().map_err(|e|{
            TensorError::Torch {context: format!("Torch error while calculating log probabilities of parameters{}", e)}
        })?;

        let log_probs_vec = match param_masks{
            None => {
                probs
            }
            Some(masks) => {
                let masked_param_log_probs = probs.iter().zip(masks.iter())
                    .map(|(logp, mask)|{
                        let masked_log_probs = Tensor::f_einsum("i,i -> i", &[logp, mask], Option::<i64>::None);
                        masked_log_probs
                    }).collect::<Result<Vec<Tensor>, TchError>>().map_err(|e|{
                    TensorError::Torch {context: format!("Torch error while masking unused parameters {}", e)}
                })?;
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
            sum = sum.f_add(&t).map_err(|e|{
                TensorError::Torch {context: format!("Torch error while summing log probabilities in categories {}", e)}
            })?;
        }




        Ok(sum)

    }
}

 */

pub type MultiDiscreteTensor = Vec<Tensor>;

impl NetOutput for MultiDiscreteTensor{}

impl NetOutput for Tensor{}
impl NetOutput for (Tensor, Tensor){}
impl NetOutput for TensorActorCritic {}

impl NetOutput for TensorMultiParamActorCritic {}


/// Converts tensor of shape (1,) and type i64 to i64. Technically it will work
/// with  shape (n,), but it will take the very first element. It is used to when we have single
/// value in Tensor that we want numeric.
///
/// Used usually when converting discrete distribution to action index.
/// Consider distribution of 4 actions: `[0.3, 0.5, 0.1, 0.1]`
/// 1. First we sample one number of `(0,1,2,3)` with probabilities above.
///     Let's say we sampled `1` (here it has 1/2 chance to be so);
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
pub fn index_tensor_to_i64(tensor: &Tensor, _additional_context: &str) -> Result<i64, ConvertError>{


    tensor.f_int64_value(&[0]).map_err(|e|{
        ConvertError::ConvertFromTensor{
            origin: e.to_string(),
            context: "Converting index tensor to i64".to_string(),
        }
    })
}