
use crate::agent::{AgentTrajectory,
                   InformationSet,
                   MultiEpisodeAutoAgent};
use crate::domain::DomainParameters;


/// Agent that collects game trajectory, which contains recorded information sets
/// in the moment of making decisions and collected rewards on the way to the end game.
pub trait TracingAgent<DP: DomainParameters, S: InformationSet<DP>>{
    /// Resets recorded trajectory
    fn reset_trajectory(&mut self);
    /// Moves out recorded trajectory leaving new initialized in place
    fn take_trajectory(&mut self) -> AgentTrajectory<DP, S>;
    //fn set_new_state(&mut self);
    /// Returns reference to held trajectory.
    fn trajectory(&self) -> &AgentTrajectory<DP, S>;



}


/// Trait for moving out trajectories of many games from agent.
pub trait MultiEpisodeTracingAgent<DP: DomainParameters, S: InformationSet<DP>, Seed>:
    TracingAgent<DP, S> + MultiEpisodeAutoAgent<DP, Seed>{


    fn take_episodes(&mut self) -> Vec<AgentTrajectory<DP, S>>;


}

