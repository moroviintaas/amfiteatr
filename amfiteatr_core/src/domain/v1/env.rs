use crate::agent::AgentActionPair;
use crate::error::AmfiError;
use crate::domain::v1::domain_parameters::DomainParameters;

/// Message sent by environment to agent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum EnvironmentMessage<DP: DomainParameters>{
    YourMove,
    MoveRefused,
    GameFinished,
    GameFinishedWithIllegalAction(DP::AgentId),
    Kill,
    UpdateState(DP::UpdateType),
    ActionNotify(AgentActionPair<DP::AgentId, DP::ActionType>),
    RewardFragment(DP::UniversalReward),
    ErrorNotify(AmfiError<DP>),

}