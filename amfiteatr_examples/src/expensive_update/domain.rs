use amfiteatr_core::demo::{DemoAction, DemoError};
use amfiteatr_core::scheme::Scheme;


#[derive(Debug, Clone)]
pub struct ExpensiveUpdateDomain{}

pub type UpdateCost = u64;



impl Scheme for ExpensiveUpdateDomain {
    type ActionType = DemoAction;
    type GameErrorType = DemoError;
    type UpdateType = UpdateCost;
    type AgentId = u64;
    type UniversalReward = f64;
}