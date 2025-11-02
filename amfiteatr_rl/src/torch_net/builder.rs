
use tch::nn::{Path, Sequential};
use tch::{nn, Tensor};
use crate::torch_net::{ActorCriticOutput, TensorActorCritic};
use std::default::Default;

/*
pub struct SequentialBuilder<F: Fn(&Path) -> Sequential> {

    sequential_fn: F
}

impl<F: Fn(&Path) -> Sequential> SequentialBuilder<F>{

    pub fn new(f: F) -> Self{
        Self{sequential_fn: f}
    }
    pub fn build(&self, path: &Path) -> Sequential{
        (self.sequential_fn)(path)
    }
}
*/

/*

pub(crate) fn build_network_discrete_ac<'p: 't, 't>(seq: Sequential, hidden_output_len: i64, action_space: i64)
    -> Box<dyn Fn(&'p tch::nn::Path) -> Box<dyn Fn(&'t Tensor) -> TensorActorCritic>>{

    Box::new(|path|
        Box::new(move |tensor|{
            let h = tensor.apply(&seq);
            TensorActorCritic{
                critic: h.apply(&nn::linear(path / "critic", hidden_output_len, 1, Default::default())),
                actor: h.apply(&nn::linear(path/"actor", hidden_output_len, action_space, Default::default()))
            }
        })

    )
}

 */


