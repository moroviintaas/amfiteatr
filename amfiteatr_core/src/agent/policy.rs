use std::marker::PhantomData;
use rand::seq::IteratorRandom;
use crate::agent::info_set::InformationSet;
use crate::agent::PresentPossibleActions;
use crate::domain::DomainParameters;


/// Trait meant for structures working as action selectors. Policy based on information set
/// must select one action if possible.
pub trait Policy<DP: DomainParameters>: Send{
    /// Information set which for which this policy is meant to work.
    type InfoSetType: InformationSet<DP>;

    /// Selects action based on information set.
    /// If at least one action is possible result should be `Some()` otherwise `None`.
    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType>;
}


/// Generic random policy - selects action at random based on iterator of possible actions
/// provided by [`InformationSet`](crate::agent::InformationSet).
#[derive(Debug, Copy, Clone, Default)]
pub struct RandomPolicy<DP: DomainParameters, State: InformationSet<DP>>{
    state: PhantomData<State>,
    _spec: PhantomData<DP>
}
impl<DP: DomainParameters, InfoSet: InformationSet<DP>> RandomPolicy<DP, InfoSet>{
    pub fn new() -> Self{
        Self{state: PhantomData::default(), _spec: PhantomData::default()}
    }
}

impl<DP: DomainParameters, InfoSet: PresentPossibleActions<DP>> Policy<DP> for RandomPolicy<DP, InfoSet>
where <<InfoSet as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type InfoSetType = InfoSet;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType> {
        let mut rng = rand::thread_rng();
        state.available_actions().into_iter().choose(&mut rng)
    }
}

impl<DP: DomainParameters, P: Policy<DP>> Policy<DP> for Box<P>{
    type InfoSetType = P::InfoSetType;

    fn select_action(&self, state: &Self::InfoSetType) -> Option<DP::ActionType> {
        self.as_ref().select_action(state)
    }
}
