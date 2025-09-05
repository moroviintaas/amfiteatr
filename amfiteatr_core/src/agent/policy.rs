use std::marker::PhantomData;
use std::sync::{Arc, Mutex, RwLock};
use rand::seq::IteratorRandom;
use crate::agent::info_set::InformationSet;
use crate::agent::{PresentPossibleActions};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

/// Trait meant for structures working as action selectors. Policy based on information set
/// must select one action if possible.
pub trait Policy<S: Scheme>: Send{
    /// Information set which for which this policy is meant to work.
    type InfoSetType: InformationSet<S>;

    /// Selects action based on information set.
    /// If at least one action is possible result should be `Ok(action)` otherwise `Err`.
    /// In special case where no action is possible or from the agent's point of view game is finished
    /// suggested error is [`AmfiteatrError::NoActionAvailable`](AmfiteatrError::NoActionAvailable).
    ///
    /// Migration from previous version: use `ok_or`
    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>>;

    /// This method is meant to be called at the beginning of the episode.
    /// It is not crucial for game protocol, yet you can use it to set something in policy for the episode.
    /// For example select one variant of mixed policy or entity from genetic population.
    fn call_on_episode_start(&mut self) -> Result<(), AmfiteatrError<S>>{
        #[cfg(feature = "log_trace")]
        log::trace!("Initial setting of policy.");
        Ok(())
    }

    /// This method is meant to be called at the beginning of the episode.
    /// It is not crucial for game protocol, yet you can use it to set something in policy after the episode.
    /// For example note something about policy and performance of choice at the beginning of the episode.
    #[allow(unused_variables)]
    fn call_on_episode_finish(&mut self,
                              final_env_reward: S::UniversalReward,
    ) -> Result<(), AmfiteatrError<S>>
    {
        #[cfg(feature = "log_trace")]
        log::trace!("End of episode setting of policy.");
        Ok(())
    }

    /// This method is meant to be called at the beginning of the episode.
    /// It is not crucial for game protocol, yet you can use it to set something in policy between epochs
    /// For example clean episode notation.
    fn call_between_epochs(&mut self)  -> Result<(), AmfiteatrError<S>>{
        #[cfg(feature = "log_trace")]
        log::trace!("Between epochs setting of policy.");
        Ok(())
    }


}

impl<S: Scheme, P: Policy<S>> Policy<S> for Arc<Mutex<P>>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.as_ref().lock(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}

impl<S: Scheme, P: Policy<S>> Policy<S> for Mutex<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.lock(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}
impl<S: Scheme, P: Policy<S>> Policy<S> for RwLock<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {

        match self.read(){
            Ok(internal_policy) => {
                internal_policy.select_action(state)
            }
            Err(e) => Err(AmfiteatrError::Lock { description: e.to_string(), object: "Policy (select_action)".to_string() })
        }
    }
}


/// Generic random policy - selects action at random based on iterator of possible actions
/// provided by [`InformationSet`](crate::agent::InformationSet).
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct RandomPolicy<S: Scheme, State: InformationSet<S>>{
    state: PhantomData<State>,
    _spec: PhantomData<S>
}


impl<S: Scheme, InfoSet: InformationSet<S>> RandomPolicy<S, InfoSet>{
    pub fn new() -> Self{
        Self{state: PhantomData, _spec: PhantomData}
    }
}



impl<S: Scheme, InfoSet: PresentPossibleActions<S>> Policy<S> for RandomPolicy<S, InfoSet>
where <<InfoSet as PresentPossibleActions<S>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        let mut rng = rand::rng();
        state.available_actions().into_iter().choose(&mut rng).ok_or_else(|| AmfiteatrError::NoActionAvailable {
            context: "Random policy".into()})
    }
}

impl<S: Scheme, P: Policy<S>> Policy<S> for Box<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<S::ActionType, AmfiteatrError<S>> {
        self.as_ref().select_action(state)
    }
}





