use crate::agent::{AutomaticAgent, AutomaticAgentRewarded, ReseedAgent};
use crate::domain::DomainParameters;
use crate::error::AmfiError;


/// Trait representing agent that plays in several episodes and can store information
/// about previous episodes.
pub trait EpisodeMemoryAgent<DP: DomainParameters, Seed>: ReseedAgent<DP, Seed> {

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
pub trait EpisodeMemoryAutoAgent<DP: DomainParameters, Seed>: EpisodeMemoryAgent<DP, Seed> + AutomaticAgent<DP>{
    /// This method runs single episode, firstly uses agents [`reseed()`](ReseedAgent)
    /// to prepare new state. Secondly it runs normal episode and finally stores episode in
    /// episode archive using [`store_episode`](ReseedAgent) method.
    fn run_episode(&mut self, seed: Seed) -> Result<(), AmfiError<DP>> {
        self.reseed(seed);
        self.run()?;
        self.store_episode();
        Ok(())
    }
}
impl <DP: DomainParameters, Seed, T: EpisodeMemoryAgent<DP, Seed> + AutomaticAgent<DP>> EpisodeMemoryAutoAgent<DP, Seed> for T{

}

/// This trait is combined trait of [`EpisodeMemoryAgent`](crate::agent::EpisodeMemoryAgent)
/// and [`AutomaticAgentRewarded`](AutomaticAgentRewarded). This trait is automatically implemented if these two
/// are implemented. If you are not interested in running agent that collects environment's
/// rewards you can use [`EpisodeMemoryAgent`](EpisodeMemoryAgent).

pub trait MultiEpisodeAutoAgentRewarded<DP: DomainParameters, Seed>: EpisodeMemoryAgent<DP, Seed> + AutomaticAgentRewarded<DP>{

    /// This method runs single episode, firstly uses agents [`reseed()`](ReseedAgent)
    /// to prepare new state. Secondly it runs normal episode with reward collection and finally stores episode in
    /// episode archive using [`store_episode`](ReseedAgent) method.
    fn run_episode_rewarded(&mut self, seed: Seed) -> Result<(), AmfiError<DP>> {
        self.reseed(seed);
        self.run_rewarded()?;
        self.store_episode();
        Ok(())
    }
}

impl <DP: DomainParameters, Seed, T: EpisodeMemoryAgent<DP, Seed> + AutomaticAgentRewarded<DP>> MultiEpisodeAutoAgentRewarded<DP, Seed> for T{

}

/*

impl<DP: DomainParameters,  Seed,  T: MultiEpisodeAgent<DP, Seed>> MultiEpisodeAgent<DP, Seed> for Arc<Mutex<T>>{
    fn store_episodes(&mut self) {
        //let mut guard = self.get_mut().unwrap();
        let mut g = self.lock().unwrap();

        g.store_episodes()
    }

    fn clear_episodes(&mut self) {
        let mut guard = self.lock().unwrap();
        guard.clear_episodes()
    }
}


 */