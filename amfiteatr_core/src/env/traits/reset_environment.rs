use crate::env::StatefulEnvironment;
use crate::domain::DomainParameters;


/// Environment with ability to be reset wit new state.

pub trait ReinitEnvironment<DP: DomainParameters>: StatefulEnvironment<DP>{
    /// Reinitialisation should set new state (at the beginning of new game episode)
    /// and it should clear every data from previous episode (optionally it can
    /// store previous episodes information on some archive storage).
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State);

}


/// Environment to be reset with some seed.
/// The purpose of this trait is to reinitialize environment and agents based
/// on single data object.
/// Primary use case is when there is a need of modelling multiple episodes of the game,
/// each parametrised with some random data.
/// Then it may be convenient to sample game parameters, and based on this sample
/// create initial state of game and set of information sets for agents.
/// For example in some card game the seed would be the set of initially distributed cards.
/// Derived state is complete information about this distribution and players'
/// information sets are derived as partial information from this sample.
/// __Note__ that this only make sense when agents are trusted as with the seed
/// they can receive complete information about the game.
pub trait ReseedEnvironment<DP: DomainParameters, Seed>: StatefulEnvironment<DP>
{
    /// This method must do reinitialize environment i.e. set new game state.
    /// New game state should be derived from seed.
    fn reseed(&mut self, seed: Seed);
}