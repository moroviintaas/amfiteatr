use std::marker::PhantomData;
use tch::{Kind, Tensor};
use tch::Kind::Int64;
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_core::scheme::Scheme;
use crate::torch_net::{ActorCriticOutput, TensorActorCritic};

pub trait ActorCriticReplayBuffer<S: Scheme>{
    /// Mainly do define actor shape (it can be single `Tensor` for one parameter actions like [`TensorActorCritic`](TensorActorCritic),
    /// or `Vec<Tensor> for multi parameter like [`TensorMultiParamActorCritic`](TensorMultiParamActorCritic))
    type ActionData;

    fn push_tensors(
        &mut self,
        info_set: &Tensor,
        action: &Self::ActionData,
        action_mask: Option<&Self::ActionData>,
        category_mask: &Self::ActionData,
        advantage: &Tensor,
        value: &Tensor,
    )
        -> Result<(), AmfiteatrError<S>>;

    /*
    fn push(
        &mut self,
        info_set: &Tensor,
        action: &Tensor,
        action_mask: Option<&Tensor>,
        advantage: &Tensor,
        reward: &Tensor,)
        -> Result<(), AmfiteatrError<S>>;
    */
    /*
    fn push_more(
        &mut self,
        info_sets: &Tensor,
        actions: &Tensor,
        action_masks: Option<&Tensor>,
        advantages: &Tensor,
        rewards: &Tensor,)
        -> Result<(), AmfiteatrError<S>>;

     */

    fn info_set_buffer(&self) -> Tensor;
    fn action_buffer(&self) -> Self::ActionData;

    fn action_mask_buffer(&self) -> Option<Self::ActionData>;

    fn category_mask_buffer(&self) -> Self::ActionData;

    fn advantage_buffer(&self) -> Tensor;

    fn return_payoff_buffer(&self) -> Tensor;

    fn capacity(&self) -> usize;

    fn size(&self) -> usize;
}

impl<S: Scheme, T: ActorCriticReplayBuffer<S>> ActorCriticReplayBuffer<S> for Box<T>{
    type ActionData = T::ActionData;


    /*
    /// ```
    /// use tch::Tensor;
    /// use tch::Kind;
    /// use tch::Device;
    /// use amfiteatr_core::demo::DemoScheme;
    /// use amfiteatr_rl::policy::{ActorCriticReplayBuffer, CyclicReplayBufferActorCritic};
    ///
    /// let mut buffer = CyclicReplayBufferActorCritic::<DemoScheme>::new(4, &[2, 4], &[6], Some(&[6]), Kind::Float, Device::Cpu).unwrap();
    ///
    /// let info_set = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, -1.0]).reshape(&[2,4]);
    /// let action = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, 6.0]);
    /// let action_mask = Tensor::from_slice(&[true, true, true, false, true, false,]);
    ///
    /// let advantage = Tensor::from_slice(&[2.0]);
    /// let reward = Tensor::from_slice(&[1.0]);
    /// buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
    ///
    /// assert_eq!(buffer.info_set_buffer().size(), &[1,2,4]);
    /// buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
    /// assert_eq!(buffer.info_set_buffer().size(), &[2,2,4]);
    /// buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
    /// assert_eq!(buffer.info_set_buffer().size(), &[3,2,4]);
    /// buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
    /// assert_eq!(buffer.info_set_buffer().size(), &[4,2,4]);
    /// buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
    /// assert_eq!(buffer.info_set_buffer().size(), &[4,2,4]);
    /// ```

    fn push(&mut self, info_set: &Tensor, action: &Tensor, action_mask: Option<&Tensor>, advantage: &Tensor, reward: &Tensor) -> Result<(), AmfiteatrError<S>> {
        self.as_mut().push(info_set, action, action_mask, advantage, reward)
    }

     */

    fn push_tensors(&mut self, info_set: &Tensor, action: &Self::ActionData, action_mask: Option<&T::ActionData>, category_mask: &T::ActionData,  advantage: &Tensor, reward: &Tensor) -> Result<(), AmfiteatrError<S>>
    {
        self.as_mut().push_tensors(info_set, action, action_mask, category_mask, advantage, reward)
    }
    fn info_set_buffer(&self) -> Tensor {
        self.as_ref().info_set_buffer()
    }

    fn action_buffer(&self) -> Self::ActionData {
        self.as_ref().action_buffer()
    }

    fn action_mask_buffer(&self) -> Option<Self::ActionData> {
        self.as_ref().action_mask_buffer()
    }

    fn category_mask_buffer(&self) -> Self::ActionData {
        self.as_ref().category_mask_buffer()
    }

    fn advantage_buffer(&self) -> Tensor {
        self.as_ref().advantage_buffer()
    }

    fn return_payoff_buffer(&self) -> Tensor {
        self.as_ref().return_payoff_buffer()
    }

    fn capacity(&self) -> usize {
        self.as_ref().capacity()
    }

    fn size(&self) -> usize {
        self.as_ref().size()
    }
}

pub struct CyclicReplayBufferActorCritic<S: Scheme>{

    capacity: i64,
    size: i64,
    position: i64,

    info_set_buffer: Tensor,
    action_buffer: Tensor,
    action_mask_buffer: Option<Tensor>,
    category_mask_buffer: Tensor,
    advantages_buffer: Tensor,
    return_payoff_buffer: Tensor,
    _scheme: PhantomData<S>,
}

impl<S: Scheme> ActorCriticReplayBuffer<S> for CyclicReplayBufferActorCritic<S>{
    type ActionData = Tensor;

    /// ```
    ///
    /// // Let's say we have information set in form of Tensor 2 x 4:
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_core::demo::DemoScheme;
    /// use amfiteatr_rl::policy::{ActorCriticReplayBuffer, CyclicReplayBufferActorCritic};
    /// let info_set_1 = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, -1.0]).reshape(&[2,4]);
    /// let info_set_2 = Tensor::from_slice(&[1.0f32, -1.0, 3.0, 2.0, 3.0, 1.0, 1.0, 0.0]).reshape(&[2,4]);
    /// // Now let's stack them to one tensor:
    /// let info_sets = Tensor::stack(&[&info_set_1, &info_set_2],0);
    /// assert_eq!(&info_sets.size(), &[2,2,4]);
    /// // Now simlarly with action logits:
    /// //let actions_1 = Tensor::from_slice(&[1.0f32, 2.0]);
    /// //let actions_2 = Tensor::from_slice(&[1.5f32, 1.0]);
    /// //let actions_logits = Tensor::stack(&[&actions_1, &actions_2], 0);
    /// let actions_selected = Tensor::from_slice(&[1, 4]).unsqueeze(1);
    /// // and masks:
    /// let masks_1 = Tensor::from_slice(&[true, true]);
    /// let masks_2 = Tensor::from_slice(&[false, true]);
    /// let masks = Tensor::stack(&[&masks_1, &masks_2], 0);
    ///
    /// let category_mask_1 = Tensor::from_slice(&[true]);
    /// let category_mask_2 = Tensor::from_slice(&[true]);
    /// let categories = Tensor::stack(&[&category_mask_1, &category_mask_2], 0);
    ///
    /// // now advantages and rewards:
    /// let advantage_1 = Tensor::from_slice(&[2.0f32]);
    /// let advantage_2 = Tensor::from_slice(&[-1.0f32]);
    /// let advantages = Tensor::stack(&[&advantage_1, &advantage_2], 0);
    /// assert_eq!(&[2,1], &advantages.size()[..]);
    /// let reward_1 = Tensor::from_slice(&[3.0f32]);
    /// let reward_2 = Tensor::from_slice(&[0.0f32]);
    /// let rewards = Tensor::stack(&[&reward_1, &reward_2], 0);
    /// // Note that every  batch hash [0] dim the same length (2)
    /// //  - this corresponds that these represent two transitions.
    /// // Now let's say we want replay buffer for 3:
    /// let mut replay_buffer = CyclicReplayBufferActorCritic::<DemoScheme>::new(
    ///     3, //capacity
    ///     &[2, 4], // shape of basic information set
    ///     2, // shape of action categories
    ///     Kind::Float,
    ///     Device::Cpu,
    ///     true,
    /// ).unwrap();
    /// assert_eq!(0, replay_buffer.position());
    /// assert_eq!(0, replay_buffer.size());
    /// replay_buffer.push_tensors(&info_sets, &actions_selected, Some(&masks), &categories, &advantages, &rewards).unwrap();
    /// // Now let's check if size is 2 and position is 2:
    /// assert_eq!(2, replay_buffer.position());
    /// assert_eq!(2, replay_buffer.size());
    ///
    /// replay_buffer.push_tensors(&info_sets, &actions_selected, Some(&masks), &categories, &advantages, &rewards).unwrap();
    /// assert_eq!(1, replay_buffer.position());
    /// assert_eq!(3, replay_buffer.size());
    /// // Now we expect that first tensor in stack is the third in replay buffer ...
    /// assert_eq!(&info_sets.slice(0, 0, 1, 1), &replay_buffer.info_set_buffer().slice(0, 2, None, 1));
    /// // and the second is again first in cyclic buffer...
    /// assert_eq!(&info_sets.slice(0, 1, None, 1), &replay_buffer.info_set_buffer().slice(0, 0, 1, 1));
    /// // Now let's have more transitions than the buffer can load:
    ///
    /// let info_sets = Tensor::stack(&[&info_set_1, &info_set_2, &info_set_1, &info_set_1,],0);
    /// let actions_logits = Tensor::from_slice(&[1, 2, 0, 1]).unsqueeze(1);
    /// let advantages = Tensor::stack(&[&advantage_1, &advantage_2, &advantage_1, &advantage_1], 0);
    /// let masks = Tensor::stack(&[&masks_1, &masks_2, &masks_1, &masks_1, ], 0);
    /// let rewards = Tensor::stack(&[&reward_1, &reward_2,&reward_1, &reward_1], 0);
    /// let categories = Tensor::from_slice(&[true; 4]).unsqueeze(1);
    /// replay_buffer.push_tensors(&info_sets, &actions_logits, Some(&masks), &categories, &advantages, &rewards).unwrap();
    /// assert_eq!(0, replay_buffer.position());
    /// assert_eq!(3, replay_buffer.size());
    /// ```
    fn push_tensors(&mut self, info_set: &Tensor, action: &Tensor, action_mask: Option<&Tensor>, category_mask: &Tensor, advantage: &Tensor, reward: &Tensor) -> Result<(), AmfiteatrError<S>> {

        let positions_added = info_set.size()[0];

        if action.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Action batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", action.size()[0])
            }});
        }

        if let (Some(action_mask_t), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {
            if action_mask_t.size()[0] != positions_added{
                return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                    context: format!("ActionMask batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", action_mask_t.size()[0])
                }});
            }

        }

        if advantage.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Advantage batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", advantage.size()[0])
            }});
        }

        if category_mask.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Category mask batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", category_mask.size()[0])
            }});
        }

        if reward.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Reward batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", advantage.size()[0])
            }});
        }


        if positions_added <= self.capacity {
            //let len_exceeding = ((self.size + positions_added) as usize).saturating_sub(self.capacity as usize) as i64;
            //let len_added_at_end = (positions_added as usize).saturating_sub(len_exceeding as usize) as i64;

            let size_proposition = self.position + positions_added;
            if size_proposition <= self.capacity{
                // We can push everything at the end

                self.info_set_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&info_set).expect(&format!("Info set: {} and {}", self.info_set_buffer.slice(0, self.position, self.position + positions_added, 1),  info_set));

                #[cfg(feature = "log_trace")]
                log::trace!("Pushing action slice: {} on buffer: {}", action, self.action_buffer);
                self.action_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&action).expect(&format!("Action: {} and {}", self.action_buffer.slice(0, self.position, self.position + positions_added, 1),  action));
                if let (Some(action_mask), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {
                    action_mask_buffer.slice(0, self.position, self.position + positions_added, 1)
                        .f_copy_(&action_mask).expect(&format!("Mask: {} and {}", action_mask_buffer.slice(0, self.position, self.position + positions_added, 1),  action_mask));;
                }

                self.advantages_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&advantage).expect(&format!("Advantages: {} and {}", self.advantages_buffer.slice(0, self.position, self.position + positions_added, 1),  advantage));


                self.return_payoff_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&reward).expect(&format!("Reward: {} and {}", self.return_payoff_buffer.slice(0, self.position, self.position + positions_added, 1), reward));

                if self.size < self.capacity{
                    self.size += positions_added
                }
                self.position += positions_added;

            } else {
                let added_at_end = self.capacity - self.position;

                let added_at_begin = positions_added - added_at_end;

                self.info_set_buffer.slice(0, self.position, None, 1)
                    .f_copy_(&info_set.slice(0, 0, added_at_end, 1))
                    .expect(&format!("(Rolling over) Info set: {} and {}", self.info_set_buffer.slice(0, self.position, None, 1),  info_set));
                self.info_set_buffer.slice(0, 0, added_at_begin, 1)
                    .f_copy_(&info_set.slice(0, added_at_end, None, 1))
                    .expect(&format!("(Rolling over) Info set: {} and {}", self.info_set_buffer.slice(0, 0, added_at_begin, 1),  info_set));


                self.action_buffer.slice(0, self.position, None, 1)
                    .f_copy_(&action.slice(0, 0, added_at_end, 1))
                    .expect(&format!("(Rolling over) Action: {} and {}", self.action_buffer.slice(0, self.position, None, 1),  action));
                self.action_buffer.slice(0, 0, added_at_begin, 1)
                    .f_copy_(&action.slice(0, added_at_end, None, 1))
                    .expect(&format!("(Rolling over) Action: {} and {}", self.action_buffer.slice(0, 0, added_at_begin, 1),  action));


                if let (Some(action_mask), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {
                    action_mask_buffer.slice(0, self.position, None, 1)
                        .f_copy_(&action_mask.slice(0, 0, added_at_end, 1))
                        .expect(&format!("(Rolling over) action_mask: {} and {}", action_mask_buffer.slice(0, self.position, None, 1),  action_mask));
                    action_mask_buffer.slice(0, 0, added_at_begin, 1)
                        .f_copy_(&action_mask.slice(0, added_at_end, None, 1))
                        .expect(&format!("(Rolling over) action_mask: {} and {}", action_mask_buffer.slice(0, 0, added_at_begin, 1),  action_mask));


                }

                self.advantages_buffer.slice(0, self.position, None, 1)
                    .f_copy_(&advantage.slice(0, 0, added_at_end, 1))
                    .expect(&format!("(Rolling over) advantages_buffer: {} and {}", self.advantages_buffer.slice(0, self.position, None, 1),  advantage));
                self.advantages_buffer.slice(0, 0, added_at_begin, 1)
                    .f_copy_(&advantage.slice(0, added_at_end, None, 1))
                    .expect(&format!("(Rolling over) advantages_buffer: {} and {}", self.advantages_buffer.slice(0, 0, added_at_begin, 1),  advantage));


                self.position = added_at_begin;
                self.size = self.capacity;
            }
            Ok(())


        } else {
            // the replay buffer is lesser than transitions added, simply store last transitions

            self.info_set_buffer.copy_(&info_set.slice(0, positions_added - self.capacity, None, 1));
            self.action_buffer.copy_(&action.slice(0, positions_added - self.capacity, None, 1));
            self.advantages_buffer.copy_(&advantage.slice(0, positions_added - self.capacity, None, 1));
            self.return_payoff_buffer.copy_(&reward.slice(0, positions_added - self.capacity, None, 1));

            if let (Some(action_mask), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {
                action_mask_buffer.copy_(&action_mask.slice(0, positions_added - self.capacity, None, 1));
            }

            self.position = 0;
            self.size = self.capacity;

            Ok(())
        }

    }

    /*
    fn push(
        &mut self,
        info_set: &Tensor,
        action: &Tensor,
        action_mask: Option<&Tensor>,
        advantage: &Tensor,
        reward: &Tensor,
    ) -> Result<(), AmfiteatrError<S>>{

        if self.position >= self.capacity{
            self.position = 0;
        }
        self.info_set_buffer.slice(0, self.position, None, 1).copy_(&info_set.slice(-1, 0, None, 1));

        self.action_buffer.slice(0, self.position, None, 1).copy_(&action.slice(-1, 0, None, 1));
        self.action_buffer.slice(0, self.position, None, 1).copy_(&action.slice(-1, 0, None, 1));
        self.action_mask_buffer.as_mut().and_then(|buffer| action_mask.map(|mask|{
            buffer.slice(0, self.position, None, 1).copy_(&mask.slice(-1, 0, None, 1));
        }));
        self.advantages_buffer.slice(0, self.position, None, 1).copy_(&advantage.slice(-1, 0, None, 1));
        self.returns_buffer.slice(0, self.position, None, 1).copy_(&reward.slice(-1, 0, None, 1));

        self.position += 1;
        if self.size < self.capacity{
            self.size += 1;
        }

        Ok(())
    }

     */

    /*
    fn push_more(&mut self, info_sets: &Tensor, actions: &Tensor, action_masks: Option<&Tensor>, advantages: &Tensor, rewards: &Tensor) -> Result<(), AmfiteatrError<S>> {



        if info_sets.size()[0] != actions.size()[0]
            || info_sets.size()[0] != advantages.size()[0]
            || info_sets.size()[0] != rewards.size()[0]{
            return Err(AmfiteatrError::Tensor {error: TensorError::BadTensorLength {
                context: "Pushing longer tensors to buffer, \
                but tensors has different length on dimension 0".to_string()
            }})
        }
        if let Some(masks) = action_masks{
            if masks.size()[0] != info_sets.size()[0]{
                return Err(AmfiteatrError::Tensor {error: TensorError::BadTensorLength {
                    context: "Pushing longer tensors to buffer, \
                but tensors has different length on dimension 0".to_string()
                }})
            }
        }

        Ok(())
    }

     */

    fn info_set_buffer(&self) -> Tensor {
        self.info_set_buffer.slice(0, 0, self.size, 1)
    }

    fn action_buffer(&self) -> Tensor {
        self.action_buffer.slice(0, 0, self.size, 1)
    }

    fn action_mask_buffer(&self) -> Option<Tensor> {
        self.action_mask_buffer.as_ref().and_then(|m| Some(m.slice(0, 0, self.size, 1)))

    }

    fn category_mask_buffer(&self) -> Self::ActionData {
        self.category_mask_buffer.slice(0,0,self.size, 1)
    }

    fn advantage_buffer(&self) -> Tensor {
        self.advantages_buffer.slice(0, 0, self.size, 1)
    }

    fn return_payoff_buffer(&self) -> Tensor {
        self.return_payoff_buffer.slice(0, 0, self.size, 1)
    }

    fn capacity(&self) -> usize {
        self.capacity as usize
    }

    fn size(&self) -> usize {
        self.size as usize
    }
}

impl<S: Scheme> CyclicReplayBufferActorCritic<S> {

    pub fn new(
        capacity: usize,
        info_set_shape: &[i64],
        action_shape: i64,

        //action_mask_shape: Option<&[i64]>,
        kind: tch::Kind,
        device: tch::Device,
        support_masks: bool,

    ) -> Result<CyclicReplayBufferActorCritic<S>, AmfiteatrError<S>> {

        let info_set_shape = [&[capacity as i64], info_set_shape].concat();
        let action_tensor_shape = [capacity as i64, 1];
        /*let action_mask_shape = action_shape.and_then(|m| {
            Some([&[capacity as i64], m].concat())
        });

         */
        let action_mask_shape = match support_masks{
            true => Some([capacity as i64, action_shape]),
            false => None,
        };



        let info_set_buffer = Tensor::zeros(&info_set_shape, (kind, device));
        let action_buffer = Tensor::zeros(&action_tensor_shape, (Int64, device));
        //let action_mask_buffer = Tensor::zeros(&action_mask_shape, (kind, device));
        let advantages_buffer = Tensor::zeros(&[capacity as i64, 1], (kind, device));
        let returns_buffer = Tensor::zeros(&[capacity as i64, 1], (kind, device));

        let action_mask_buffer =action_mask_shape.and_then(|m| Some(Tensor::ones(m, (Kind::Bool ,device))));
        let category_mask_buffer = Tensor::ones(&[capacity as i64, 1], (Kind::Bool, device));

        Ok(Self{capacity: capacity as i64,
            size: 0,
            position: 0,
            info_set_buffer, action_buffer, action_mask_buffer, advantages_buffer,
            category_mask_buffer,
            return_payoff_buffer: returns_buffer,
            _scheme: Default::default(),
        })





    }

    pub fn position(&self) -> usize {
        self.position as usize
    }

    pub fn size(&self) -> usize {
        self.size as usize
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }


    /*
    pub fn push(
        &mut self,
        info_set: &Tensor,
        action: &Tensor,
        action_mask: Option<&Tensor>,
        advantage: &Tensor,
        reward: &Tensor,
    ) -> Result<(), AmfiteatrError<S>>{

        if self.position >= self.capacity{
            self.position = 0;
        }
        self.info_set_buffer.slice(0, self.position, None, 1).copy_(&info_set.slice(-1, 0, None, 1));

        self.action_buffer.slice(0, self.position, None, 1).copy_(&action.slice(-1, 0, None, 1));
        self.action_buffer.slice(0, self.position, None, 1).copy_(&action.slice(-1, 0, None, 1));
        self.action_mask_buffer.as_mut().and_then(|buffer| action_mask.map(|mask|{
            buffer.slice(0, self.position, None, 1).copy_(&mask.slice(-1, 0, None, 1));
        }));
        self.advantages_buffer.slice(0, self.position, None, 1).copy_(&advantage.slice(-1, 0, None, 1));
        self.returns_buffer.slice(0, self.position, None, 1).copy_(&reward.slice(-1, 0, None, 1));

        self.position += 1;
        if self.size < self.capacity{
            self.size += 1;
        }

        Ok(())
    }

    pub fn info_set_buffer(&self) -> Tensor{
        self.info_set_buffer.slice(0, 0, self.size, 1)

    }

     */
}

/*
#[cfg(test)]
mod tests{
    use crate::policy::replay::actor_critic_buffer::ActorCriticReplayBuffer;
use tch::{Device, Kind, Tensor};
    use amfiteatr_core::demo::DemoScheme;
    use crate::policy::CyclicReplayBufferActorCritic;

    #[test]
    fn test_push(){

        use amfiteatr_core::demo::DemoScheme;

        let mut buffer = CyclicReplayBufferActorCritic::<DemoScheme>::new(4, &[2, 4], &[6], Some(&[6]), Kind::Float, Device::Cpu).unwrap();

        let info_set = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, -1.0]).reshape(&[2,4]);
        let action = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, 6.0]);
        let action_mask = Tensor::from_slice(&[true, true, true, false, true, false,]);

        let advantage = Tensor::from_slice(&[2.0]);
        let reward = Tensor::from_slice(&[1.0]);
        buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();

        assert_eq!(buffer.info_set_buffer().size(), &[1,2,4]);
        buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
        assert_eq!(buffer.info_set_buffer().size(), &[2,2,4]);
        buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
        assert_eq!(buffer.info_set_buffer().size(), &[3,2,4]);
        buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
        assert_eq!(buffer.info_set_buffer().size(), &[4,2,4]);
        buffer.push(&info_set, &action_mask, Some(&action_mask), &advantage, &reward).unwrap();
        assert_eq!(buffer.info_set_buffer().size(), &[4,2,4]);
    }
}

 */

pub struct CyclicReplayBufferMultiActorCritic<S: Scheme>{

    capacity: i64,
    size: i64,
    position: i64,

    info_set_buffer: Tensor,
    action_buffer: Vec<Tensor>,
    action_mask_buffer: Option<Vec<Tensor>>,
    category_mask_buffer: Vec<Tensor>,
    advantages_buffer: Tensor,
    return_payoff_buffer: Tensor,
    _scheme: PhantomData<S>,
}


impl<S: Scheme> CyclicReplayBufferMultiActorCritic<S> {
    pub fn new(
        capacity: usize,
        info_set_shape: &[i64],
        action_categories_shapes: &[i64],
        //action_mask_shape: Option<&[i64]>,
        //number_of_action_categories: usize,
        kind: tch::Kind,
        device: tch::Device,

    ) -> Result<CyclicReplayBufferMultiActorCritic<S>, AmfiteatrError<S>> {

        let info_set_shape = [&[capacity as i64], info_set_shape].concat();
        let action_shape = [capacity as i64, 1];


        let info_set_buffer = Tensor::zeros(&info_set_shape, (kind, device));
        let action_buffer = (0..action_categories_shapes.len()).map(|s|{
            Tensor::zeros(&[capacity as i64, 1], (Int64, device))
        }).collect();


        //let action_mask_buffer = Tensor::zeros(&action_mask_shape, (kind, device));
        let advantages_buffer = Tensor::zeros(&[capacity as i64, 1], (kind, device));
        let returns_buffer = Tensor::zeros(&[capacity as i64, 1], (kind, device));

        let action_mask_buffer = Some(action_categories_shapes.iter().map(|s| Tensor::ones(
            &[capacity as i64, *s], (Kind::Bool, device))).collect());
        let category_mask_buffer = (0..action_categories_shapes.len()).map(|_|{
            Tensor::ones(&[capacity as i64], (Kind::Bool, device))
        }).collect();

        Ok(Self{capacity: capacity as i64,
            size: 0,
            position: 0,
            info_set_buffer, action_buffer, action_mask_buffer, advantages_buffer,
            category_mask_buffer,
            return_payoff_buffer: returns_buffer,
            _scheme: Default::default(),
        })

    }

    pub fn position(&self) -> usize {
        self.position as usize
    }

    pub fn size(&self) -> usize {
        self.size as usize
    }

    pub fn capacity(&self) -> usize {
        self.capacity as usize
    }
}
impl<S: Scheme> ActorCriticReplayBuffer<S> for CyclicReplayBufferMultiActorCritic<S>{
    type ActionData = Vec<Tensor>;

    /// ```
    ///
    /// // Let's say we have information set in form of Tensor 2 x 4:
    /// use tch::{Device, Kind, Tensor};
    /// use amfiteatr_core::demo::DemoScheme;
    /// use amfiteatr_rl::policy::{ActorCriticReplayBuffer, CyclicReplayBufferMultiActorCritic};
    /// let info_set_1 = Tensor::from_slice(&[1.0f32, 2.0, 0.0, 0.0, 3.0, 2.0, 4.0, -1.0]).reshape(&[2,4]);
    /// let info_set_2 = Tensor::from_slice(&[1.0f32, -1.0, 3.0, 2.0, 3.0, 1.0, 1.0, 0.0]).reshape(&[2,4]);
    /// // Now let's stack them to one tensor:
    /// let info_sets = Tensor::stack(&[&info_set_1, &info_set_2],0);
    /// assert_eq!(&info_sets.size(), &[2,2,4]);
    /// // Now simlarly with action logits:
    /// // First category actions 2 possiblities in cat 1 and 3 in cat 2
    /// //
    /// let actions_cat_1 = Tensor::from_slice(&[1, 0]).unsqueeze(1); //we want format [transition_dim x 1 (singe action param) ]
    /// let actions_cat_2 = Tensor::from_slice(&[0, 2]).unsqueeze(1);
    /// //let actions_logits = Tensor::stack(&[&actions_cat_1, &actions_cat_2], 0);
    /// let actions = vec![actions_cat_1, actions_cat_2];
    /// // and masks:
    /// // first category 2 possibilities
    /// //                                           t=0: a=0   a=1 | t=1: a=0   a=1
    /// let masks_cat_1 = Tensor::from_slice2(&[[        true, true],    [true, false]]);
    /// //  second category, 3 possibilities          t=0: a=0   a=1   a=2 | t=1:  a=0   a=1    a=2
    /// let masks_cat_2 = Tensor::from_slice2(&[[        true, true, false],    [true, false, true]]);
    ///
    /// let masks = vec![masks_cat_1, masks_cat_2];
    ///
    /// let category_mask_1 = Tensor::from_slice(&[true, true]);
    /// let category_mask_2 = Tensor::from_slice(&[true, false]);
    /// //let categories = Tensor::stack(&[&category_mask_1, &category_mask_2], 0);
    /// let categories = vec![category_mask_1, category_mask_2];
    /// // now advantages and rewards:
    /// let advantage_1 = Tensor::from_slice(&[2.0f32]);
    /// let advantage_2 = Tensor::from_slice(&[-1.0f32]);
    /// let advantages = Tensor::stack(&[&advantage_1, &advantage_2], 0);
    /// assert_eq!(&[2,1], &advantages.size()[..]);
    /// let reward_1 = Tensor::from_slice(&[3.0f32]);
    /// let reward_2 = Tensor::from_slice(&[0.0f32]);
    /// let rewards = Tensor::stack(&[&reward_1, &reward_2], 0);
    /// // Note that every  batch hash [0] dim the same length (2)
    /// //  - this corresponds that these represent two transitions.
    /// // Now let's say we want replay buffer for 3:
    /// let mut replay_buffer = CyclicReplayBufferMultiActorCritic::<DemoScheme>::new(
    ///     3, //capacity
    ///     &[2, 4], // shape of basic information set
    ///     &[2, 3], // shape of action categories - two categories first with 2 possible values, second with 3
    ///     //Some(&[2]), // shape of action categories
    ///     Kind::Float,
    ///     Device::Cpu
    /// ).unwrap();
    /// assert_eq!(0, replay_buffer.position());
    /// assert_eq!(0, replay_buffer.size());
    /// replay_buffer.push_tensors(&info_sets, &actions, Some(&masks), &categories, &advantages, &rewards).unwrap();
    /// // Now let's check if size is 2 and position is 2:
    /// assert_eq!(2, replay_buffer.position());
    /// assert_eq!(2, replay_buffer.size());
    ///
    /// replay_buffer.push_tensors(&info_sets, &actions, Some(&masks), &categories, &advantages, &rewards).unwrap();
    /// assert_eq!(1, replay_buffer.position());
    /// assert_eq!(3, replay_buffer.size());
    /// // Now we expect that first tensor in stack is the third in replay buffer ...
    /// assert_eq!(&info_sets.slice(0, 0, 1, 1), &replay_buffer.info_set_buffer().slice(0, 2, None, 1));
    /// // and the second is again first in cyclic buffer...
    /// assert_eq!(&info_sets.slice(0, 1, None, 1), &replay_buffer.info_set_buffer().slice(0, 0, 1, 1));
    /// // Now let's have more transitions than the buffer can load:
    ///
    /// let info_sets = Tensor::stack(&[&info_set_1, &info_set_2, &info_set_1, &info_set_1,],0);
    /// let actions_selected = vec![Tensor::from_slice(&[1, 0, 0, 0]).unsqueeze(1), Tensor::from_slice(&[0, 2, 0, 2]).unsqueeze(1)];
    /// let category_mask = vec![Tensor::from_slice(&[true, true, true, true]), Tensor::from_slice(&[true, true, false, false])];
    /// let advantages = Tensor::stack(&[&advantage_1, &advantage_2, &advantage_1, &advantage_1], 0);
    ///
    /// // So ... 4 transitions (tensor dimension 0), action space is 2 in first cat and 3 in second cat (dimension 2)
    /// // then vector dimension is category dimension
    /// let masks_cat_1 = Tensor::from_slice2(&[[        true, true],    [true, false], [true, false], [true, true]]);
    /// //  second category, 3 possibilities          t=0: a=0   a=1   a=2 | t=1:  a=0   a=1    a=2
    /// let masks_cat_2 = Tensor::from_slice2(&[[        true, true, false],    [true, false, true], [true, false, true], [true, true, true]]);
    ///
    /// let masks = vec![masks_cat_1, masks_cat_2];
    /// let rewards = Tensor::stack(&[&reward_1, &reward_2,&reward_1, &reward_1], 0);
    ///
    /// replay_buffer.push_tensors(&info_sets, &actions_selected, Some(&masks), &category_mask, &advantages, &rewards).unwrap();
    /// assert_eq!(0, replay_buffer.position());
    /// assert_eq!(3, replay_buffer.size());
    /// ```
    fn push_tensors(&mut self, info_set: &Tensor, action: &Self::ActionData, action_mask: Option<&Self::ActionData>, category_mask: &Self::ActionData, advantage: &Tensor, reward: &Tensor) -> Result<(), AmfiteatrError<S>> {

        let positions_added = info_set.size()[0];


        for (param, tensor) in action.iter().enumerate(){
            if tensor.size()[0] != positions_added{
                return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                    context: format!("Action batch tensor has bad 0 dim (action number) for action parameter {param} : {}, expected: {positions_added}", tensor.size()[0])
                }});
            }
        }


        if let (Some(action_mask_vt), Some(_action_mask_buffer)) = (&action_mask, self.action_mask_buffer.as_mut()) {

            for (param, tensor) in action_mask_vt.iter().enumerate(){
                if tensor.size()[0] != positions_added{
                    return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                        context: format!("ActionMask batch tensor has bad 0 dim for action category {param}: {}, expected: {positions_added}", tensor.size()[0])
                    }});
                }
            }

        }

        if advantage.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Advantage batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", advantage.size()[0])
            }});
        }

        if reward.size()[0] != positions_added{
            return Err(AmfiteatrError::Tensor{ error: TensorError::BadTensorLength {
                context: format!("Reward batch tensor has bad 0 dim (action number): {}, expected: {positions_added}", advantage.size()[0])
            }});
        }


        if positions_added <= self.capacity {
            //let len_exceeding = ((self.size + positions_added) as usize).saturating_sub(self.capacity as usize) as i64;
            //let len_added_at_end = (positions_added as usize).saturating_sub(len_exceeding as usize) as i64;

            let size_proposition = self.position + positions_added;
            if size_proposition <= self.capacity{
                // We can push everything at the end

                self.info_set_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&info_set).expect(&format!("Info set: {} and {}", self.info_set_buffer.slice(0, self.position, self.position + positions_added, 1),  info_set));

                //#[cfg(feature = "log_trace")]
                //log::trace!("Pushing action slice: {} on buffer: {}", action, self.action_buffer);


                for (param, (param_tensor, buffer)) in action.iter().zip(self.action_buffer.iter_mut()).enumerate(){
                    buffer.slice(0, self.position, self.position + positions_added, 1)
                        .f_copy_(&param_tensor).expect(&format!("Error copying action param ({param}): {} and {}", buffer.slice(0, self.position, self.position + positions_added, 1),  param_tensor));

                }

                if let (Some(action_masks_buffer), Some(action_masks)) = (&mut self.action_mask_buffer, &action_mask){
                    for (param, (param_tensor, buffer)) in action_masks.iter().zip(action_masks_buffer.iter_mut()).enumerate(){
                        buffer.slice(0, self.position, self.position + positions_added, 1)
                            .f_copy_(&param_tensor).expect(&format!("Error copying action mask param ({param}): {} and {}", buffer.slice(0, self.position, self.position + positions_added, 1),  param_tensor));

                    }
                }

                for (param, (param_tensor, buffer)) in category_mask.iter().zip(self.category_mask_buffer.iter_mut()).enumerate(){
                    buffer.slice(0, self.position, self.position + positions_added, 1)
                        .f_copy_(&param_tensor).expect(&format!("Error copying category mask ({param}): {} and {}", buffer.slice(0, self.position, self.position + positions_added, 1),  param_tensor));

                }





                self.advantages_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&advantage).expect(&format!("Advantages: {} and {}", self.advantages_buffer.slice(0, self.position, self.position + positions_added, 1),  advantage));


                self.return_payoff_buffer.slice(0, self.position, self.position + positions_added, 1)
                    .f_copy_(&reward).expect(&format!("Reward: {} and {}", self.return_payoff_buffer.slice(0, self.position, self.position + positions_added, 1), reward));

                if self.size < self.capacity{
                    self.size += positions_added
                }
                self.position += positions_added;

            } else {
                let added_at_end = self.capacity - self.position;

                let added_at_begin = positions_added - added_at_end;

                self.info_set_buffer.slice(0, self.position, None, 1)
                    .f_copy_(&info_set.slice(0, 0, added_at_end, 1))
                    .expect(&format!("(Rolling over) Info set: {} and {}", self.info_set_buffer.slice(0, self.position, None, 1),  info_set));
                self.info_set_buffer.slice(0, 0, added_at_begin, 1)
                    .f_copy_(&info_set.slice(0, added_at_end, None, 1))
                    .expect(&format!("(Rolling over) Info set: {} and {}", self.info_set_buffer.slice(0, 0, added_at_begin, 1),  info_set));



                for (param, (param_tensor, buffer)) in action.iter().zip(self.action_buffer.iter_mut()).enumerate(){
                    buffer.slice(0, self.position, None, 1)
                        .f_copy_(&param_tensor.slice(0, 0, added_at_end, 1)).expect(&format!("Error (at end of the buffer) copying action param ({param}): {} and {}", buffer.slice(0, self.position, None, 1),  param_tensor.slice(0, 0, added_at_begin, 1)));
                    buffer.slice(0, 0, added_at_begin, 1)
                        .f_copy_(&param_tensor.slice(0, added_at_end, None, 1)).expect(&format!("Error (at beginning of the buffer) copying action param ({param}): {} and {}", buffer.slice(0, 0, added_at_end, 1),  param_tensor.slice(0, added_at_end, None, 1)));

                }


                if let (Some(action_mask), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {

                    for (param, (param_tensor, buffer)) in action_mask.iter().zip(action_mask_buffer.iter_mut()).enumerate(){
                        buffer.slice(0, self.position, None, 1)
                            .f_copy_(&param_tensor.slice(0, 0, added_at_end, 1)).expect(&format!("Error (at end of the buffer) copying action mask (param {param}): {} and {}", buffer.slice(0, self.position, None, 1),  param_tensor.slice(0, 0, added_at_begin, 1)));
                        buffer.slice(0, 0, added_at_begin, 1)
                            .f_copy_(&param_tensor.slice(0, added_at_end, None, 1)).expect(&format!("Error (at beginning of the buffer) copying action mask (param {param}): {} and {}", buffer.slice(0, 0, added_at_end, 1),  param_tensor.slice(0, added_at_end, None, 1)));

                    }
                    /*
                    action_mask_buffer.slice(0, self.position, None, 1)
                        .f_copy_(&action_mask.slice(0, 0, added_at_end, 1))
                        .expect(&format!("(Rolling over) action_mask: {} and {}", action_mask_buffer.slice(0, self.position, None, 1),  action_mask));
                    action_mask_buffer.slice(0, 0, added_at_begin, 1)
                        .f_copy_(&action_mask.slice(0, added_at_end, None, 1))
                        .expect(&format!("(Rolling over) action_mask: {} and {}", action_mask_buffer.slice(0, 0, added_at_begin, 1),  action_mask));
                    */

                }

                for (param, (param_tensor, buffer)) in category_mask.iter().zip(self.category_mask_buffer.iter_mut()).enumerate(){
                    buffer.slice(0, self.position, None, 1)
                        .f_copy_(&param_tensor.slice(0, 0, added_at_end, 1)).expect(&format!("Error (at end of the buffer) copying category mask param ({param}): {} and {}", buffer.slice(0, self.position, None, 1),  param_tensor.slice(0, 0, added_at_begin, 1)));
                    buffer.slice(0, 0, added_at_begin, 1)
                        .f_copy_(&param_tensor.slice(0, added_at_end, None, 1)).expect(&format!("Error (at beginning of the buffer) copying category mask param ({param}): {} and {}", buffer.slice(0, 0, added_at_end, 1),  param_tensor.slice(0, added_at_end, None, 1)));

                }

                self.advantages_buffer.slice(0, self.position, None, 1)
                    .f_copy_(&advantage.slice(0, 0, added_at_end, 1))
                    .expect(&format!("(Rolling over) advantages_buffer: {} and {}", self.advantages_buffer.slice(0, self.position, None, 1),  advantage));
                self.advantages_buffer.slice(0, 0, added_at_begin, 1)
                    .f_copy_(&advantage.slice(0, added_at_end, None, 1))
                    .expect(&format!("(Rolling over) advantages_buffer: {} and {}", self.advantages_buffer.slice(0, 0, added_at_begin, 1),  advantage));


                self.position = added_at_begin;
                self.size = self.capacity;
            }
            Ok(())


        } else {
            // the replay buffer is lesser than transitions added, simply store last transitions

            self.info_set_buffer.copy_(&info_set.slice(0, positions_added - self.capacity, None, 1));
            //self.action_buffer.copy_(&action.slice(0, positions_added - self.capacity, None, 1));

            for (param, (param_tensor, buffer)) in action.iter().zip(self.action_buffer.iter_mut()).enumerate(){
                buffer
                    .f_copy_(&param_tensor.slice(0, positions_added - self.capacity, None, 1)).expect(&format!("Error copying action param ({param}): {} and {}", buffer,  param_tensor.slice(0, positions_added - self.capacity, None, 1)));

            }

            self.advantages_buffer.copy_(&advantage.slice(0, positions_added - self.capacity, None, 1));
            self.return_payoff_buffer.copy_(&reward.slice(0, positions_added - self.capacity, None, 1));

            if let (Some(action_mask), Some(action_mask_buffer)) = (action_mask, self.action_mask_buffer.as_mut()) {
                //action_mask_buffer.copy_(&action_mask.slice(0, positions_added - self.capacity, None, 1));
                for (param, (param_tensor, buffer)) in action_mask.iter().zip(action_mask_buffer.iter_mut()).enumerate(){
                    buffer.slice(0, 0, None, 1)
                        .f_copy_(&param_tensor.slice(0, positions_added - self.capacity, None, 1)).expect(&format!("Error copying action mask (param {param}): {} and {}", buffer.slice(0, self.position, self.position + positions_added, 1),  param_tensor));

                }
            }

            for (param, (param_tensor, buffer)) in category_mask.iter().zip(self.category_mask_buffer.iter_mut()).enumerate(){
                buffer.slice(0, 0, None, 1)
                    .f_copy_(&param_tensor.slice(0, positions_added - self.capacity, None, 1)).expect(&format!("Error copying category mask (param {param}): {} and {}", buffer.slice(0, self.position, self.position + positions_added, 1),  param_tensor));

            }

            self.position = 0;
            self.size = self.capacity;

            Ok(())
        }

    }



    fn info_set_buffer(&self) -> Tensor {
        self.info_set_buffer.slice(0, 0, self.size, 1)
    }

    fn action_buffer(&self) -> Self::ActionData {
        //self.action_buffer.slice(0, 0, self.size, 1)
        self.action_buffer.iter().map(|t|{
            t.slice(0,0, self.size, 1)
        }).collect()
    }

    fn action_mask_buffer(&self) -> Option<Self::ActionData> {
        self.action_mask_buffer.as_ref().and_then(|vm|{
            Some(vm.iter().map(|t|{
                t.slice(0,0, self.size, 1)
            }).collect())
        })

    }

    fn category_mask_buffer(&self) -> Self::ActionData {
        self.category_mask_buffer.iter().map(|t|{
            t.slice(0,0,self.size, 1)
        }).collect()
    }

    fn advantage_buffer(&self) -> Tensor {
        self.advantages_buffer.slice(0, 0, self.size, 1)
    }

    fn return_payoff_buffer(&self) -> Tensor {
        self.return_payoff_buffer.slice(0, 0, self.size, 1)
    }

    fn capacity(&self) -> usize {
        self.capacity as usize
    }

    fn size(&self) -> usize {
        self.size as usize
    }
}