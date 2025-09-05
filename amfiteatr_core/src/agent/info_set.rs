use std::fmt::Debug;
use crate::scheme::{Scheme, Reward};

/// Represents agent's point of view on game state.
/// > Formally _information set_ is subset of game _states_ that are indistinguishable
/// > from the point of view of this agent.
/// > Most common case is when agent does not know some detail of the game state.
/// > Game states with identical observable details but differing in unobservable (not known) detail
/// > form single _information set_.
pub trait InformationSet<S: Scheme>: Send + Debug{


    fn agent_id(&self) -> &S::AgentId;
    fn is_action_valid(&self, action: &S::ActionType) -> bool;
    fn update(&mut self, update: S::UpdateType) -> Result<(), S::GameErrorType>;
}
/// Trait for information sets that can provide iterator (Vec or some other kind of list) of
/// actions that are possible in this state of game.
pub trait PresentPossibleActions<S: Scheme>: InformationSet<S>{
    /// Structure that can be transformed into iterator of actions.
    type ActionIteratorType: IntoIterator<Item = S::ActionType>;
    /// Construct and return iterator of possible actions.
    fn available_actions(&self) -> Self::ActionIteratorType;
}

impl<S: Scheme, T: InformationSet<S>> InformationSet<S> for Box<T>{
    fn agent_id(&self) -> &S::AgentId {
        self.as_ref().agent_id()
    }


    fn is_action_valid(&self, action: &S::ActionType) -> bool {
        self.as_ref().is_action_valid(action)
    }

    fn update(&mut self, update: S::UpdateType) -> Result<(), S::GameErrorType> {
        self.as_mut().update(update)
    }
}

impl<S: Scheme, T: PresentPossibleActions<S>> PresentPossibleActions<S> for Box<T>{
    type ActionIteratorType = T::ActionIteratorType;

    fn available_actions(&self) -> Self::ActionIteratorType {
        self.as_ref().available_actions()
    }
}

/// Information Set that can produce score based on it's state.
/// This reward can be in different type that defined in [`DomainParameters`](Scheme).
/// > It can represent different kind of reward than defined in protocol parameters.
/// > Primary use case is to allow agent interpret its situation, for example instead of
/// > one numeric value as reward agent may be interested in some vector of numeric values representing
/// > his multi-criterion view on game's result.
pub trait EvaluatedInformationSet<S: Scheme, R: Reward>: InformationSet<S>{
    fn current_assessment(&self) -> R;
    fn penalty_for_illegal(&self) -> R;
}

impl<T: EvaluatedInformationSet<S, R>, S: Scheme, R: Reward> EvaluatedInformationSet<S, R> for Box<T> {

    fn current_assessment(&self) -> R {
        self.as_ref().current_assessment()
    }

    fn penalty_for_illegal(&self) -> R {
        self.as_ref().penalty_for_illegal()
    }
}


/// Information set that can be constructed using certain (generic type) value to construct new
/// information set instance.
/// > This is meant to be implemented for every information set
/// > used in game by any agent and for state of environment.
/// > Implementing construction based on common seed allows to reload all info sets and states.
pub trait ConstructedInfoSet<S: Scheme, B>: InformationSet<S> + From<B> {}
impl<S: Scheme, B, T: InformationSet<S> + From<B>> ConstructedInfoSet<S, B> for T{}

//impl<S: DomainParameters, B, T: ConstructedInfoSet<S, B>> ConstructedInfoSet<S, B> for Box<T>{}