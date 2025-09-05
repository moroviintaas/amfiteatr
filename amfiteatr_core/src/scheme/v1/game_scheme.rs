use std::fmt::{Debug};
use crate::scheme::action::Action;
use crate::agent::AgentIdentifier;
use crate::scheme::Reward;
use crate::error::{InternalGameError};
//use crate::state::StateUpdate;

/// Trait locking game scheme parameters, to ensure environment and agents can communicate
pub trait Scheme: Clone + Debug + Send + Sync + 'static{
    type ActionType: Action;
    type GameErrorType: InternalGameError<Self> ;
    type UpdateType: Debug + Send + Clone ;
    type AgentId: AgentIdentifier;
    type UniversalReward: Reward;
}