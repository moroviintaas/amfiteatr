use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use rand::seq::IteratorRandom;
use crate::agent::info_set::InformationSet;
use crate::agent::PresentPossibleActions;
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;

/// Trait meant for structures working as action selectors. Policy based on information set
/// must select one action if possible.
pub trait Policy<DP: DomainParameters>: Send{
    /// Information set which for which this policy is meant to work.
    type InfoSetType: InformationSet<DP>;

    /// Selects action based on information set.
    /// If at least one action is possible result should be `Ok(action)` otherwise `Err`.
    /// In special case where no action is possible or from the agent's point of view game is finished
    /// suggested error is [`AmfiteatrError::NoActionAvailable`](AmfiteatrError::NoActionAvailable).
    ///
    /// Migration from previous version: use `ok_or`
    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>>;
}

impl<DP: DomainParameters, P: Policy<DP>> Policy<DP> for Arc<Mutex<P>>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {

        match self.as_ref().lock(){
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
pub struct RandomPolicy<DP: DomainParameters, State: InformationSet<DP>>{
    state: PhantomData<State>,
    _spec: PhantomData<DP>
}


impl<DP: DomainParameters, InfoSet: InformationSet<DP>> RandomPolicy<DP, InfoSet>{
    pub fn new() -> Self{
        Self{state: PhantomData, _spec: PhantomData}
    }
}



impl<DP: DomainParameters, InfoSet: PresentPossibleActions<DP>> Policy<DP> for RandomPolicy<DP, InfoSet>
where <<InfoSet as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        let mut rng = rand::rng();
        state.available_actions().into_iter().choose(&mut rng).ok_or_else(|| AmfiteatrError::NoActionAvailable {
            context: "Random policy".into()})
    }
}

impl<DP: DomainParameters, P: Policy<DP>> Policy<DP> for Box<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<DP::ActionType, AmfiteatrError<DP>> {
        self.as_ref().select_action(state)
    }
}





