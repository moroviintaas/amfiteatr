use std::default::Default;
use std::marker::PhantomData;
use tch::{Kind, Tensor};
use amfiteatr_core::error::{AmfiteatrError, TensorError};
use amfiteatr_core::scheme::Scheme;
use crate::torch_net::ActorCriticOutput;

pub trait ActorCriticReplayBuffer<S: Scheme>{
    // /// Mainly do define actor shape (it can be single tensor for one parameter actions like [`TensorActorCritic`](TensorActorCritic),
    // /// or multi parameter like [`TensorMultiParamActorCritic`](TensorMultiParamActorCritic)
    //type ActorCriticType: ActorCriticOutput;

    fn push(
        &mut self,
        info_set: &Tensor,
        action: &Tensor,
        action_mask: Option<&Tensor>,
        advantage: &Tensor,
        reward: &Tensor,)
        -> Result<(), AmfiteatrError<S>>;

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
    fn action_buffer(&self) -> Tensor;

    fn action_mask_buffer(&self) -> Option<Tensor>;

    fn advantage_buffer(&self) -> Tensor;

    fn reward_buffer(&self) -> Tensor;

    fn capacity(&self) -> usize;

    fn size(&self) -> usize;
}

impl<S: Scheme, T: ActorCriticReplayBuffer<S>> ActorCriticReplayBuffer<S> for Box<T>{
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

    fn info_set_buffer(&self) -> Tensor {
        self.as_ref().info_set_buffer()
    }

    fn action_buffer(&self) -> Tensor {
        self.as_ref().action_buffer()
    }

    fn action_mask_buffer(&self) -> Option<Tensor> {
        self.as_ref().action_mask_buffer()
    }

    fn advantage_buffer(&self) -> Tensor {
        self.as_ref().advantage_buffer()
    }

    fn reward_buffer(&self) -> Tensor {
        self.as_ref().reward_buffer()
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
    advantages_buffer: Tensor,
    returns_buffer: Tensor,
    _scheme: PhantomData<S>,
}

impl<S: Scheme> ActorCriticReplayBuffer<S> for CyclicReplayBufferActorCritic<S>{
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

    fn advantage_buffer(&self) -> Tensor {
        self.advantages_buffer.slice(0, 0, self.size, 1)
    }

    fn reward_buffer(&self) -> Tensor {
        self.returns_buffer.slice(0, 0, self.size, 1)
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
        action_shape: &[i64],
        action_mask_shape: Option<&[i64]>,
        kind: tch::Kind,
        device: tch::Device,

    ) -> Result<CyclicReplayBufferActorCritic<S>, AmfiteatrError<S>> {

        let info_set_shape = [&[capacity as i64], info_set_shape].concat();
        let action_shape = [&[capacity as i64], action_shape].concat();
        let action_mask_shape = action_mask_shape.and_then(|m| {
            Some([&[capacity as i64], m].concat())
        });


        let info_set_buffer = Tensor::zeros(&info_set_shape, (kind, device));
        let action_buffer = Tensor::zeros(&action_shape, (kind, device));
        //let action_mask_buffer = Tensor::zeros(&action_mask_shape, (kind, device));
        let advantages_buffer = Tensor::zeros(&[capacity as i64], (kind, device));
        let returns_buffer = Tensor::zeros(&[capacity as i64], (kind, device));

        let action_mask_buffer =action_mask_shape.and_then(|m| Some(Tensor::ones(m, (Kind::Bool ,device))));

        Ok(Self{capacity: capacity as i64,
            size: 0,
            position: 0,
            info_set_buffer, action_buffer, action_mask_buffer, advantages_buffer, returns_buffer,
            _scheme: Default::default(),
        })





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