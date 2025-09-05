use thiserror::Error;
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;


/// Error dealing with game trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum TrajectoryError<DP: Scheme>{
    #[error("Agent {} tried registering step after closing trajectory", .0)]
    UpdateOnFinishedAgentTrajectory(DP::AgentId),
    #[error("Agent {} tried finishing step after closing trajectory", .0)]
    FinishingOnFinishedAgentTrajectory(DP::AgentId),
    #[error("Agent {} tried finishing step after without having played action in step", .0)]
    TiedStepRecordWithNoAction(DP::AgentId),
    #[error("Update on finished game trajectory: .0")]
    UpdateOnFinishedGameTrajectory{
        description: String
    },
    #[error("Finishing on finished game trajectory: .0")]
    FinishingOnFinishedGameTrajectory{
        description: String
    },
}

impl<DP: Scheme> From<TrajectoryError<DP>> for AmfiteatrError<DP>{
    fn from(value: TrajectoryError<DP>) -> Self {
        Self::Trajectory {source: value}
    }
}