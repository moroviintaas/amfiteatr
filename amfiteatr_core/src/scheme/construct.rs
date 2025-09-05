use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

/// Trait for objects that can be renewed using some data.
/// For example agents can be renewed with new state for new game episode without changing
/// things that do not need to be changed (like communication interface or trajectory archive).
pub trait Renew<DP: Scheme, S>{

    fn renew_from(&mut self, base: S) -> Result<(), AmfiteatrError<DP>>;
}


/// Trait for renewing some struct (usually state of game).
/// Lets say you want to renew game state with some seed, but based of this reseeding you want
/// to have generated seeds for information sets.
/// Maybe you want to shuffle cards and distribute them among the players as a game preparation,
/// normally players could receive their cards early in the game - but only their hands. What if
/// you would like to give them unfair advantage and show other players' cards before the game. W
/// Fair game protocol does not have [`UpdateType`](crate::scheme::Scheme::UpdateType) to do such
/// nasty thing.
pub trait RenewWithEffect<DP: Scheme, S>{

    type Effect;
    fn renew_with_effect_from(&mut self, base: S) -> Result<Self::Effect, AmfiteatrError<DP>>;

}