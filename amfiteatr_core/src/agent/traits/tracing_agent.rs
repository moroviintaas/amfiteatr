
use crate::agent::{AgentTrajectory,
                   InformationSet,
                   MultiEpisodeAutoAgent};
use crate::scheme::Scheme;


/// Agent that collects game trajectory, which contains recorded information sets
/// in the moment of making decisions and collected rewards on the way to the end game.
pub trait TracingAgent<DP: Scheme, IS: InformationSet<DP>>{
    /// Resets recorded trajectory
    fn reset_trajectory(&mut self);
    /// Moves out recorded trajectory leaving new initialized in place
    fn take_trajectory(&mut self) -> AgentTrajectory<DP, IS>;
    //fn set_new_state(&mut self);
    /// Returns reference to held trajectory.
    fn trajectory(&self) -> &AgentTrajectory<DP, IS>;



}


/// Trait for moving out trajectories of many games from agent.
pub trait MultiEpisodeTracingAgent<DP: Scheme, IS: InformationSet<DP>, Seed>:
    TracingAgent<DP, IS> + MultiEpisodeAutoAgent<DP, Seed>{


    fn take_episodes(&mut self) -> Vec<AgentTrajectory<DP, IS>>;



}

