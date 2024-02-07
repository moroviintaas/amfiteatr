use crate::agent::{StatefulAgent};
use crate::domain::DomainParameters;

/// Trait for agent that can reset their attributes to some default values
/// while setting new info set. Typically to be used in situations
/// when game is to be relaunched from beginning (optionally with new start point)
pub trait ReinitAgent<DP: DomainParameters>: StatefulAgent<DP>{

    fn reinit(&mut self, new_info_set: <Self as StatefulAgent<DP>>::InfoSetType);
}


/// This trait is for agents able to be reinited for new simulation. Type `Seed`
/// is the source on which the information set must be rebuilt.
/// Conceptually seeding should not change agent's id, but do things like reset observations,
/// clear history saved in information set or init information set with some data.
///
/// __For example__ imagine game when every player starts the game with some set of resources e.g cards.
/// It is convenient to sample some initial game state and send it to agents as a seed (like exact card distribution) and then they
/// construct the information set based on it. However in some games this full information on the start is
/// violation of rules and do it only if you trust your agent to populate their information set with needed data
/// and ignore data they should not know. You gain simpler reinitialization of simulation, but
/// potentially violate information model. Alternatively initial state may be sent to agents during first steps
/// of the game, however it complicates Update structure.
pub trait ReseedAgent<DP: DomainParameters, Seed>: StatefulAgent<DP>
//where <Self as StatefulAgent<DP>>::InfoSetType: ConstructedInfoSet<DP, Seed>{
{
    fn reseed(&mut self, seed: Seed);
}

