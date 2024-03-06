use crate::agent::{AutomaticAgent, AutomaticAgentRewarded, EvaluatedInformationSet, ReseedAgent};
use crate::domain::DomainParameters;
use crate::env::Trajectory;
use crate::error::AmfiteatrError;


/// Trait representing agent that plays in several episodes and can store information
/// about previous episodes.
pub trait EpisodeMemoryAgent{

    /// This method is meant to move agent's current episode information to
    /// historical episode storage.
    fn store_episode(&mut self);
    /// This method is used to clean record of historical episodes
    fn clear_episodes(&mut self);
}
/// This trait is combined trait of [`EpisodeMemoryAgent`](crate::agent::EpisodeMemoryAgent)
/// and [`AutomaticAgent`](AutomaticAgent). This trait is automatically implemented if these two
/// are implemented. If you are interested in running agent that collects environment's
/// rewards please use rather [`MultiEpisodeAutoAgentRewarded`](MultiEpisodeAutoAgentRewarded).
pub trait EpisodeMemoryAutoAgent<DP: DomainParameters, Seed>:
    EpisodeMemoryAgent
    + AutomaticAgent<DP>
    + ReseedAgent<DP, Seed>{
    /// This method runs single episode, firstly uses agents [`reseed()`](ReseedAgent)
    /// to prepare new state. Secondly it runs normal episode and finally stores episode in
    /// episode archive using [`store_episode`](ReseedAgent) method.
    fn run_episode(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>> {
        self.reseed(seed)?;
        self.run()?;
        self.store_episode();
        Ok(())
    }
}
impl <
    DP: DomainParameters, Seed,
    T: EpisodeMemoryAgent + AutomaticAgent<DP> + ReseedAgent<DP, Seed>>
EpisodeMemoryAutoAgent<DP, Seed> for T
{

}

/// This trait is combined trait of [`EpisodeMemoryAgent`](crate::agent::EpisodeMemoryAgent)
/// and [`AutomaticAgentRewarded`](AutomaticAgentRewarded). This trait is automatically implemented if these two
/// are implemented. If you are not interested in running agent that collects environment's
/// rewards you can use [`EpisodeMemoryAgent`](EpisodeMemoryAgent).

pub trait MultiEpisodeAutoAgentRewarded<DP: DomainParameters, Seed>:
    EpisodeMemoryAgent
    + AutomaticAgentRewarded<DP>
    + ReseedAgent<DP, Seed>{

    /// This method runs single episode, firstly uses agents [`reseed()`](ReseedAgent)
    /// to prepare new state. Secondly it runs normal episode with reward collection and finally stores episode in
    /// episode archive using [`store_episode`](ReseedAgent) method.
    fn run_episode_rewarded(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>> {
        self.reseed(seed)?;
        self.run_rewarded()?;
        self.store_episode();
        Ok(())
    }
}

impl <
    DP: DomainParameters, Seed,
    T: EpisodeMemoryAgent + AutomaticAgentRewarded<DP> + ReseedAgent<DP, Seed>>
MultiEpisodeAutoAgentRewarded<DP, Seed> for T{

}
