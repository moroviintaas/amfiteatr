use std::fmt::Debug;
use crate::domain::{DomainParameters, Reward};

/// Represents agent's point of view on game state.
/// > Formally _information set_ is subset of game _states_ that are indistinguishable
/// from the point of view of this agent.
/// Most common case is when agent does not know some detail of the game state.
/// Game states with identical observable details but differing in unobservable (not known) detail
/// form single _information set_.
pub trait InformationSet<DP: DomainParameters>: Send + Debug{


    fn agent_id(&self) -> &DP::AgentId;
    fn is_action_valid(&self, action: &DP::ActionType) -> bool;
    fn update(&mut self, update: DP::UpdateType) -> Result<(), DP::GameErrorType>;
}
/// Trait for information sets that can provide iterator (Vec or some other kind of list) of
/// actions that are possible in this state of game.
pub trait PresentPossibleActions<DP: DomainParameters>: InformationSet<DP>{
    /// Structure that can be transformed into iterator of actions.
    type ActionIteratorType: IntoIterator<Item = DP::ActionType>;
    /// Construct and return iterator of possible actions.
    fn available_actions(&self) -> Self::ActionIteratorType;
}

impl<DP: DomainParameters, T: InformationSet<DP>> InformationSet<DP> for Box<T>{
    fn agent_id(&self) -> &DP::AgentId {
        self.as_ref().agent_id()
    }


    fn is_action_valid(&self, action: &DP::ActionType) -> bool {
        self.as_ref().is_action_valid(action)
    }

    fn update(&mut self, update: DP::UpdateType) -> Result<(), DP::GameErrorType> {
        self.as_mut().update(update)
    }
}

impl<DP: DomainParameters, T: PresentPossibleActions<DP>> PresentPossibleActions<DP> for Box<T>{
    type ActionIteratorType = T::ActionIteratorType;

    fn available_actions(&self) -> Self::ActionIteratorType {
        self.as_ref().available_actions()
    }
}

/// Information Set that can produce score based on it's state.
/// This reward can be in different type that defined in [`DomainParameters`](DomainParameters).
/// > It can represent different kind of reward than defined in protocol parameters.
/// Primary use case is to allow agent interpret it's situation, for example instead of
/// one numeric value as reward agent may be interested in some vector of numeric values representing
/// his multi-criterion view on game's result.
pub trait EvaluatedInformationSet<DP: DomainParameters>: InformationSet<DP>{
    type RewardType: Reward;
    fn current_subjective_score(&self) -> Self::RewardType;
    fn penalty_for_illegal(&self) -> Self::RewardType;
}

impl<T: EvaluatedInformationSet<Spec>, Spec: DomainParameters> EvaluatedInformationSet<Spec> for Box<T> {
    type RewardType = T::RewardType;

    fn current_subjective_score(&self) -> Self::RewardType {
        self.as_ref().current_subjective_score()
    }

    fn penalty_for_illegal(&self) -> T::RewardType {
        self.as_ref().penalty_for_illegal()
    }
}


/// Information set that can be constructed using certain (generic type) value to construct new
/// information set instance.
/// > This is meant to be implemented for every information set
/// used in game by any agent and for state of environment.
/// Implementing construction based on common seed allows to reload all info sets and states.
pub trait ConstructedInfoSet<DP: DomainParameters, B>: InformationSet<DP> + From<B> {}
impl<DP: DomainParameters, B, T: InformationSet<DP> + From<B>> ConstructedInfoSet<DP, B> for T{}

//impl<DP: DomainParameters, B, T: ConstructedInfoSet<DP, B>> ConstructedInfoSet<DP, B> for Box<T>{}