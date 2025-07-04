use crate::agent::{AutomaticAgent, ReseedAgent};
use crate::domain::DomainParameters;

use crate::error::AmfiteatrError;




/// Trait for agents repeating episodes with collecting rewards
pub trait MultiEpisodeAutoAgent<DP: DomainParameters, Seed>:
    ReseedAgent<DP, Seed> + AutomaticAgent<DP>{

    /// Things to be done between reseeding information set and playing an episode.
    /// For example sampling policy version from mixed policy.
    fn initialize_episode(&mut self)-> Result<(), AmfiteatrError<DP>> ;
    /// This method is meant to move agent's current episode information to
    /// historical episode storage. If agent does not store history, leave it not operating.
    fn store_episode(&mut self)-> Result<(), AmfiteatrError<DP>> ;
    /// This method is used to clean record of historical episodes.
    /// If agent does not store history, leave it not operating.
    fn clear_episodes(&mut self)-> Result<(), AmfiteatrError<DP>> ;
    /// Takes all stored trajectories leaving empty list in place.
    /// This method runs single episode, firstly uses agents [`reseed()`](ReseedAgent)
    /// to prepare new state. Secondly it runs normal episode with reward collection and finally stores episode in
    /// episode archive using [`store_episode`](ReseedAgent) method.
    fn run_episode(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>> {
        self.reseed(seed)?;
        self.initialize_episode()?;
        self.run()?;
        self.store_episode()?;
        Ok(())
    }
}

