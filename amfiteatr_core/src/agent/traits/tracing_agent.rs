
use crate::agent::{EpisodeMemoryAgent, EvaluatedInformationSet, Trajectory};
use crate::domain::DomainParameters;


/// Agent that collects game trajectory, which contains recorded information sets
/// in the moment of making decisions and collected rewards on the way to the end game.
pub trait TracingAgent<DP: DomainParameters, S: EvaluatedInformationSet<DP>>{
    /// Resets recorded trajectory
    fn reset_trajectory(&mut self);
    /// Moves out recorded trajectory leaving new initialized in place
    fn take_trajectory(&mut self) -> Trajectory<DP, S>;
    //fn set_new_state(&mut self);
    /// Returns reference to held trajectory.
    fn game_trajectory(&self) -> &Trajectory<DP, S>;
    /// Adds new record to stored trajectory, information set before taking action, and
    /// rewards in which resulted performed action.
    fn commit_trace(&mut self);


}


/// Trait for moving out trajectories of many games from agent. _Warning:_ It is probable that this trait will be
/// somehow merged with [`EpisodeMemoryAgent`]
pub trait MultiEpisodeTracingAgent<DP: DomainParameters, S: EvaluatedInformationSet<DP>>:
    TracingAgent<DP, S> + EpisodeMemoryAgent{

    /// Takes all stored trajectories leaving empty list in place.
    fn take_episodes(&mut self) -> Vec<Trajectory<DP, S>>;

}

