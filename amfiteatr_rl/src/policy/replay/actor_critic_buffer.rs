use std::default::Default;
use std::marker::PhantomData;
use tch::{Kind, Tensor};
use amfiteatr_core::error::AmfiteatrError;
use amfiteatr_core::scheme::Scheme;

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
}

#[cfg(test)]
mod tests{
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