
use crate::agent::{AgentTrajectory,
                   InformationSet,
                   MultiEpisodeAutoAgent};
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;


/// Agent that collects game trajectory, which contains recorded information sets
/// in the moment of making decisions and collected rewards on the way to the end game.
pub trait TracingAgent<DP: DomainParameters, S: InformationSet<DP>>{
    /// Resets recorded trajectory
    fn reset_trajectory(&mut self);
    /// Moves out recorded trajectory leaving new initialized in place
    fn take_trajectory(&mut self) -> AgentTrajectory<DP, S>;
    //fn set_new_state(&mut self);
    /// Returns reference to held trajectory.
    fn game_trajectory(&self) -> &AgentTrajectory<DP, S>;
    /// Adds new record to stored trajectory, information set before taking action, and
    /// rewards in which resulted performed action.
    fn commit_trace(&mut self) -> Result<(), AmfiteatrError<DP>>;

    fn finalize_trajectory(&mut self) -> Result<(), AmfiteatrError<DP>>;


}


/// Trait for moving out trajectories of many games from agent.
pub trait MultiEpisodeTracingAgent<DP: DomainParameters, S: InformationSet<DP>, Seed>:
    TracingAgent<DP, S> + MultiEpisodeAutoAgent<DP, Seed>{


    fn take_episodes(&mut self) -> Vec<AgentTrajectory<DP, S>>;


}

