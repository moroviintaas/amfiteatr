use std::fmt::{Debug};
use crate::domain::action::Action;
use crate::agent::AgentIdentifier;
use crate::domain::Reward;
use crate::error::{InternalGameError};
//use crate::state::StateUpdate;

/// Trait locking game domain parameters, to ensure environment and agents can communicate
pub trait DomainParameters: Clone + Debug + Send + Sync + 'static{
    type ActionType: Action;
    type GameErrorType: InternalGameError<Self> ;
    type UpdateType: Debug + Send + Clone ;
    type AgentId: AgentIdentifier;
    type UniversalReward: Reward;
}