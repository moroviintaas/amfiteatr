use tch::Tensor;
use amfiteatr_core::agent::{AgentTrajectory, InformationSet};
use amfiteatr_core::domain::DomainParameters;

#[inline]
pub fn sum_trajectories_steps<DP: DomainParameters, S: InformationSet<DP>>
(
    trajectories: &[AgentTrajectory<DP, S>]
) -> usize{
    trajectories.iter().fold(0, |acc, x|{
        acc + x.number_of_steps()
    })
}

#[inline]
pub fn find_max_trajectory_len<DP: DomainParameters, S: InformationSet<DP>>
(
trajectories: &[AgentTrajectory<DP, S>]
) -> usize{
    trajectories.iter().map(|x|{
        x.number_of_steps()
    }).max().unwrap_or(0)
}

#[inline]
pub fn categorical_dist_entropy(probabilities: &Tensor, log_probabilities: &Tensor,  kind: tch::Kind) -> Tensor {
    (-log_probabilities * probabilities).sum_dim_intlist(-1, false, kind)
}