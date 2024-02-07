use std::fmt::Debug;
use std::marker::PhantomData;

use log::{debug, trace};
use rand::distributions::uniform::{UniformFloat, UniformSampler};
use tch::Kind::Float;
use tch::nn::{Optimizer, VarStore};
use tch::{Kind, Reduction, Tensor};
use amfiteatr_core::agent::{AgentTraceStep, Trajectory, InformationSet, Policy, PresentPossibleActions, EvaluatedInformationSet};
use amfiteatr_core::domain::DomainParameters;


use crate::error::AmfiRLError;
use crate::tensor_data::{ConvertToTensor, ConversionToTensor};
use crate::torch_net::NeuralNet1;
use rand::thread_rng;
use crate::policy::LearningNetworkPolicy;
pub use crate::policy::TrainConfig;

/// Enum used to select best action
#[derive(Debug, Copy, Clone)]
pub enum QSelector{
    /// Always select action with maximal Q-value (do not explore)
    Max,
    //MultinomialLinear,
    /// Treat Q-function as logit distribution and sample action
    MultinomialLogits,
    /// With probability epsilon explore (using multinomial distribution on softmaxed Q-values), otherwise select max
    EpsilonGreedy(f64),
}


impl QSelector{

    pub fn select_q_value_index(&self, q_vals: &Tensor, exploring_enabled: bool) -> Option<usize>{
        if ! exploring_enabled{
            let rv = f32::try_from(q_vals.argmax(None, false));
            return rv.ok().and_then(|i| Some(i as usize))
        }
        match self{
            Self::Max => {
                //println!("{:?}", q_vals.size());
                //println!("{:?}", q_vals.argmax(None, false));
                let rv = f32::try_from(q_vals.argmax(None, false));
                //println!("{:?}", rv);
                //rv.map(|v| v.first()).ok().and_then(|i| Some(i as usize))
                //rv.ok().and_then(|v|v.first().and_then(|i| Some(*i as usize)))
                rv.ok().and_then(|i| Some(i as usize))

            },
            Self::MultinomialLogits => {
                let probs = q_vals.softmax(-1, Float);
                let index_t = probs.multinomial(1, false);
                let rv =  Vec::<f32>::try_from(index_t);
                //rv.map(|v|v.first()).ok().and_then(|i| Some(i as usize))
                rv.ok().and_then(|v|v.first().and_then(|i| Some(*i as usize)))
            }
            Self::EpsilonGreedy(epsilon) =>{
                let mut rng = thread_rng();
                let n: f64 = UniformFloat::<f64>::sample_single(0.0, 1.0, &mut rng);
                //println!("n: {n:}, epsilon: {epsilon:}");
                if n < *epsilon{
                    let probs = q_vals.softmax(-1, Float);
                    let index_t = probs.multinomial(1, false);
                    let rv =  Vec::<f32>::try_from(index_t);
                    //rv.map(|v|v.first()).ok().and_then(|i| Some(i as usize))
                    rv.ok().and_then(|v|v.first().and_then(|i| Some(*i as usize)))
                } else {
                    let rv = f32::try_from(q_vals.argmax(None, false));
                    rv.ok().and_then(|i| Some(i as usize))
                }

            }
        }
    }

    /*
    fn logits<R: ProportionalReward<f32>>(values: &[R]) -> Vec<f32>
    where for<'a> &'a R: Sub<&'a R, Output = R>
    {
        let sum = values.iter().fold(R::neutral(), |acc, x|{
              acc + x
        });
        values.iter().map(|v|{
            v.proportion(&(&sum - v)).ln()
        }).collect()


    }*/

}
/// Generic implementation of Advantage Q-function policy
pub struct QLearningPolicy<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<IS2T>,
    IS2T: ConversionToTensor,
    A2T: ConversionToTensor,

>
{

    network: NeuralNet1,
    optimizer: Optimizer,
    _dp: PhantomData<DP>,
    _is: PhantomData<InfoSet>,
    info_set_way: IS2T,
    action_way: A2T,
    q_selector: QSelector,
    training_config: TrainConfig,
    explore_enabled: bool
}

impl
<
    DP: DomainParameters,
    InfoSet: EvaluatedInformationSet<DP> + Debug + ConvertToTensor<IS2T> + PresentPossibleActions<DP>,
    IS2T: ConversionToTensor,
    A2T: ConversionToTensor
> QLearningPolicy<DP, InfoSet, IS2T, A2T>
where <<InfoSet as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::Item: ConvertToTensor<A2T>, {

    pub fn new(
        network: NeuralNet1,
        optimizer: Optimizer,
        info_set_way: IS2T,
        action_way: A2T,
        q_selector: QSelector,
        training_config: TrainConfig) -> Self{
        Self{
            network,
            optimizer,
            info_set_way,
            action_way,
            q_selector,
            training_config,
            _dp: Default::default(), _is: Default::default(),
            explore_enabled: true}
    }

}



impl
<
    DP: DomainParameters,
    InfoSet: EvaluatedInformationSet<DP> + Debug + ConvertToTensor<IS2T> + PresentPossibleActions<DP>,
    IS2T: ConversionToTensor,
    A2T: ConversionToTensor
> LearningNetworkPolicy<DP> for QLearningPolicy<DP, InfoSet, IS2T, A2T>
where <<InfoSet as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::Item: ConvertToTensor<A2T>,
//<DP as DomainParameters>::UniversalReward: FloatTensorReward,
<DP as DomainParameters>::ActionType: ConvertToTensor<A2T> {
    type Network = NeuralNet1;
    type TrainConfig = TrainConfig;

    fn network(&self) -> &Self::Network {
        &self.network
    }

    fn network_mut(&mut self) -> &mut Self::Network {
        &mut self.network
    }

    fn var_store(&self) -> &VarStore {
        self.network.var_store()
    }

    fn var_store_mut(&mut self) -> &mut VarStore {
        self.network.var_store_mut()
    }

    fn switch_explore(&mut self, enabled: bool) {
        self.explore_enabled = enabled;
    }


    fn config(&self) -> &Self::TrainConfig {
        &self.training_config
    }

    fn train_on_trajectories<
        R: Fn(&AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(
        &mut self,
        trajectories: &[Trajectory<DP, <Self as Policy<DP>>::InfoSetType>],
        reward_f: R)
        -> Result<(), AmfiRLError<DP>> {


        let _device = self.network.device();
        let capacity_estimate = trajectories.iter().fold(0, |acc, x|{
           acc + x.list().len()
        });
        let tmp_capacity_estimate = trajectories.iter().map(|x|{
            x.list().len()
        }).max().unwrap_or(0);
        let batch_size_estimate = trajectories.iter().map(|x|{
            x.list().len()
        }).sum();
        let mut qval_tensor_vec_t = Vec::with_capacity(tmp_capacity_estimate);
        let mut qval_tensor_vec = Vec::with_capacity(batch_size_estimate);
        let mut state_action_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut reward_tensor_vec = Vec::<Tensor>::with_capacity(capacity_estimate);
        let mut discounted_rewards_tensor_vec: Vec<Tensor> = Vec::with_capacity(tmp_capacity_estimate);
        for t in trajectories{


            if t.list().is_empty(){
                continue;
            }
            let steps_in_trajectory = t.list().len();

            let mut state_action_q_tensor_vec_t: Vec<Tensor>  = t.list().iter().map(|step|{
                let s = step.step_info_set().to_tensor(&self.info_set_way);
                let a = step.taken_action().to_tensor(&self.action_way);
                let t = Tensor::cat(&[s,a], 0);
                let q = (self.network.net())(&t);
                qval_tensor_vec_t.push(q);
                t

            }).collect();



            let final_score_t: Tensor =  reward_f(t.list().last().unwrap());
            debug!("Final score tensor shape: {:?}", final_score_t.size());
            discounted_rewards_tensor_vec.clear();
            for _ in 0..=steps_in_trajectory{
                discounted_rewards_tensor_vec.push(Tensor::zeros(final_score_t.size(), (Kind::Float, self.network.device())));
            }
            debug!("Discounted_rewards_tensor_vec len before inserting: {}", discounted_rewards_tensor_vec.len());
            //let mut discounted_rewards_tensor_vec: Vec<Tensor> = vec![Tensor::zeros(DP::UniversalReward::total_size(), (Kind::Float, self.network.device())); steps_in_trajectory+1];
            //discounted_rewards_tensor_vec.last_mut().unwrap().copy_(&final_score_t);
            trace!("Reward stream: {:?}", t.list().iter().map(|x| reward_f(x)).collect::<Vec<Tensor>>());
            for s in (0..discounted_rewards_tensor_vec.len()-1).rev(){
                //println!("{}", s);
                let r_s = reward_f(&t[s]).to_device(self.network.device()) + (&discounted_rewards_tensor_vec[s+1] * self.training_config.gamma);
                discounted_rewards_tensor_vec[s].copy_(&r_s);
            }
            discounted_rewards_tensor_vec.pop();
            trace!("Discounted future payoffs tensor: {:?}", discounted_rewards_tensor_vec);
            debug!("Discounted rewards_tensor_vec after inserting");

            state_action_tensor_vec.append(&mut state_action_q_tensor_vec_t);
            reward_tensor_vec.append(&mut discounted_rewards_tensor_vec);
            qval_tensor_vec.append(&mut qval_tensor_vec_t);

        }
        let _state_action_batch = Tensor::stack(&state_action_tensor_vec[..], 0);
        let results_batch = Tensor::stack(&reward_tensor_vec[..], 0);
        let q_batch = Tensor::stack(&qval_tensor_vec[..], 0);
        debug!("Result batch shape: {:?}", results_batch.size());
        debug!("Q result batch shape: {:?}", q_batch.size());

        //let diff = &results_batch - q_batch;
        //let loss = (&diff * &diff).mean(Float);
        let loss = q_batch.mse_loss(&results_batch, Reduction::Mean);
        self.optimizer.backward_step_clip(&loss, 0.5);
        Ok(())
    }
}


impl<
    DP: DomainParameters,
    InfoSet: InformationSet<DP> + Debug + ConvertToTensor<IS2T> + PresentPossibleActions<DP>,
    IS2T: ConversionToTensor,
    A2T: ConversionToTensor
> Policy<DP> for QLearningPolicy<DP, InfoSet, IS2T, A2T>
where <<InfoSet as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::Item: ConvertToTensor<A2T>{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType> {

        let mut actions = Vec::new();
        let q_predictions : Vec<_>/*<Tensor>*/ = state.available_actions().into_iter().map(|a|{
            let action_tensor = a.to_tensor(&self.action_way);
            let input_tensor = Tensor::cat(&[state.to_tensor(&self.info_set_way), action_tensor], 0);
            let q_val = (&self.network.net())(&input_tensor).narrow(0,0,1);
            actions.push(a);
            q_val
        }).collect();
        let q_pred = Tensor::cat(&q_predictions[..], 0);

        let index = self.q_selector.select_q_value_index(&q_pred, self.explore_enabled);

        index.and_then(|i| actions.get(i)).cloned()

    }
}