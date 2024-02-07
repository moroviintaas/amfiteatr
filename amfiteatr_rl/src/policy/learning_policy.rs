use tch::nn::VarStore;
use tch::Tensor;
use amfiteatr_core::agent::{AgentTraceStep, Trajectory, Policy, EvaluatedInformationSet};

use amfiteatr_core::domain::DomainParameters;
use crate::error::AmfiRLError;
use crate::tensor_data::FloatTensorReward;


/// Trait for types that provide discount factor (for example training configs)
pub trait DiscountFactor {
    fn discount_factor(&self) -> f64;
}
/// Trait representing policy that uses neural network to select action and can be trained.
pub trait LearningNetworkPolicy<DP: DomainParameters> : Policy<DP>
where <Self as Policy<DP>>::InfoSetType: EvaluatedInformationSet<DP>
{
    type Network;
    type TrainConfig;

    /// Returns reference to underlying neural network
    fn network(&self) -> &Self::Network;
    /// Returns mutable reference to underlying neural network
    fn network_mut(&mut self) -> &mut Self::Network;
    /// Returns reference to underlying [`VarStore`]
    fn var_store(&self) -> &VarStore;
    /// Returns mutable reference to underlying [`VarStore`]
    fn var_store_mut(&mut self) -> &mut VarStore;

    /// Switch exploring on and off
    fn switch_explore(&mut self, enabled: bool);

    /// Returns reference to current config of policy
    fn config(&self) -> &Self::TrainConfig;
    /// This is generic training function. Generic type `R` must produce reward tensor that
    /// agent got in this step. In traditional RL model it will be vectorised reward calculated
    /// by environment. This is in fact implemented by [`train_on_trajectories_env_reward`](LearningNetworkPolicy::train_on_trajectories_env_reward).
    fn train_on_trajectories<R: Fn(&AgentTraceStep<DP, <Self as Policy<DP>>::InfoSetType>) -> Tensor>(
        &mut self,
        trajectories: &[Trajectory<DP, <Self as Policy<DP>>::InfoSetType>],
        reward_f: R,
    ) -> Result<(), AmfiRLError<DP>>;

    /// Training implementation using environment distributed reward
    fn train_on_trajectories_env_reward(&mut self,
                                        trajectories: &[Trajectory<DP, <Self as Policy<DP>>::InfoSetType>]) -> Result<(), AmfiRLError<DP>>
    where <DP as DomainParameters>::UniversalReward: FloatTensorReward{

        self.train_on_trajectories(trajectories,  |step| step.step_universal_reward().to_tensor())
    }

    /// Training implementation using self assessment calculated based on information set
    fn train_on_trajectories_self_assessed(&mut self,
                                           trajectories: &[Trajectory<DP, <Self as Policy<DP>>::InfoSetType>],
                                              ) -> Result<(), AmfiRLError<DP>>
    where <<Self as Policy<DP>>::InfoSetType as EvaluatedInformationSet<DP>>::RewardType: FloatTensorReward{

        self.train_on_trajectories(trajectories,  |step| step.step_subjective_reward().to_tensor())
    }

}