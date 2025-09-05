use thiserror::Error;
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;


/// Error dealing with game trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum TrajectoryError<S: Scheme>{
    #[error("Agent {} tried registering step after closing trajectory", .0)]
    UpdateOnFinishedAgentTrajectory(S::AgentId),
    #[error("Agent {} tried finishing step after closing trajectory", .0)]
    FinishingOnFinishedAgentTrajectory(S::AgentId),
    #[error("Agent {} tried finishing step after without having played action in step", .0)]
    TiedStepRecordWithNoAction(S::AgentId),
    #[error("Update on finished game trajectory: .0")]
    UpdateOnFinishedGameTrajectory{
        description: String
    },
    #[error("Finishing on finished game trajectory: .0")]
    FinishingOnFinishedGameTrajectory{
        description: String
    },
}

impl<S: Scheme> From<TrajectoryError<S>> for AmfiteatrError<S>{
    fn from(value: TrajectoryError<S>) -> Self {
        Self::Trajectory {source: value}
    }
}