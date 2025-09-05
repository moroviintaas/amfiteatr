
use crate::agent::{AgentTrajectory,
                   InformationSet,
                   MultiEpisodeAutoAgent};
use crate::scheme::Scheme;


/// Agent that collects game trajectory, which contains recorded information sets
/// in the moment of making decisions and collected rewards on the way to the end game.
pub trait TracingAgent<S: Scheme, IS: InformationSet<S>>{
    /// Resets recorded trajectory
    fn reset_trajectory(&mut self);
    /// Moves out recorded trajectory leaving new initialized in place
    fn take_trajectory(&mut self) -> AgentTrajectory<S, IS>;
    //fn set_new_state(&mut self);
    /// Returns reference to held trajectory.
    fn trajectory(&self) -> &AgentTrajectory<S, IS>;



}


/// Trait for moving out trajectories of many games from agent.
pub trait MultiEpisodeTracingAgent<S: Scheme, IS: InformationSet<S>, Seed>:
    TracingAgent<S, IS> + MultiEpisodeAutoAgent<S, Seed>{


    fn take_episodes(&mut self) -> Vec<AgentTrajectory<S, IS>>;



}

