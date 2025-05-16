use std::sync::{Arc, Mutex};
use tch::Tensor;
use amfiteatr_core::agent::{
    Policy,
    AgentTrajectory,
    InformationSet,
    AgentStepView
};

use amfiteatr_core::domain::DomainParameters;
use serde::{Deserialize, Serialize};
use amfiteatr_core::error::AmfiteatrError;
use crate::error::AmfiteatrRlError;
use crate::tensor_data::FloatTensorReward;


/// Trait for types that provide discount factor (for example training configs)
pub trait DiscountFactor {
    fn discount_factor(&self) -> f64;
}


/// Basic struct for data summarizing learning procedure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LearnSummary{
    pub value_loss: Option<f64>,
    pub policy_gradient_loss: Option<f64>,
    pub entropy_loss: Option<f64>,
    pub approx_kl: Option<f64>,
    //pub average_payoff

}

/// Trait representing policy that uses neural network to select action and can be trained.
pub trait LearningNetworkPolicy<DP: DomainParameters> : Policy<DP>
where <Self as Policy<DP>>::InfoSetType: InformationSet<DP>
{
    type Summary: Send;

    /*
    /// Returns reference to underlying [`VarStore`]
    fn var_store(&self) -> &VarStore;
    /// Returns mutable reference to underlying [`VarStore`]
    fn var_store_mut(&mut self) -> &mut VarStore;

     */

    /// Switch exploring on and off
    fn switch_explore(&mut self, enabled: bool);





    /*
    /// If supported enable or disable exploration (with disabled exploration policy is expected to always select
    /// action that seems the best).
    fn enable_exploration(&mut self, enable: bool);


     */
    ///// Returns reference to current config of policy
    //fn config(&self) -> &Self::TrainConfig;
    /// This is generic training function. Generic type `R` must produce reward tensor that
    /// agent got in this step. In traditional RL model it will be vectorised reward calculated
    /// by environment. This is in fact implemented by [`train_on_trajectories_env_reward`](LearningNetworkPolicy::train_on_trajectories_env_reward).
    fn train_on_trajectories<R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(
        &mut self,
        trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>],
        reward_f: R,
    ) -> Result<Self::Summary, AmfiteatrRlError<DP>>;

    /// Training implementation using environment distributed reward
    fn train_on_trajectories_env_reward(&mut self,
                                        trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>]) -> Result<Self::Summary, AmfiteatrRlError<DP>>
    where <DP as DomainParameters>::UniversalReward: FloatTensorReward{

        self.train_on_trajectories(trajectories,  |step| step.reward().to_tensor())
    }



}


impl<DP: DomainParameters, T: LearningNetworkPolicy<DP>> LearningNetworkPolicy<DP> for Arc<Mutex<T>>{
    type Summary = T::Summary;

    fn switch_explore(&mut self, enabled: bool) {
        match self.lock(){
            Ok(mut internal_policy) => {
                internal_policy.switch_explore(enabled)
            }
            Err(e) => {
                panic!("Failed switching explre in policy due to Mutex error {e}")
            }
        }
    }

    fn train_on_trajectories<R: Fn(&AgentStepView<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(&mut self, trajectories: &[AgentTrajectory<DP, <Self as Policy<DP>>::InfoSetType>], reward_f: R) -> Result<Self::Summary, AmfiteatrRlError<DP>> {
        match self.as_ref().lock(){
            Ok(mut internal_policy) => {
                internal_policy.train_on_trajectories(trajectories, reward_f)
            }
            Err(e) => Err(AmfiteatrRlError::Amfiteatr{source: AmfiteatrError::Lock { description: e.to_string(), object: "Learning policy".to_string() }})
        }
    }
}