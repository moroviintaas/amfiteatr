use crate::agent::AgentActionPair;
use crate::error::AmfiteatrError;
use crate::domain::v1::domain_parameters::DomainParameters;

/// Message sent by environment to agent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnvironmentMessage<DP: DomainParameters>{
    YourMove,
    MoveRefused,
    GameFinished,
    GameTruncated,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::AgentId: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::AgentId: serde::Deserialize<'de>")))]
    GameFinishedWithIllegalAction(DP::AgentId),
    Kill,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::UpdateType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::UpdateType: serde::Deserialize<'de>")))]
    UpdateState(DP::UpdateType),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::AgentId: serde::Serialize, DP::ActionType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::AgentId: serde::Deserialize<'de>, DP::ActionType: serde::Deserialize<'de>")))]
    ActionNotify(AgentActionPair<DP::AgentId, DP::ActionType>),
    RewardFragment(DP::UniversalReward),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "AmfiteatrError<DP>: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "AmfiteatrError<DP>: serde::Deserialize<'de>")))]
    ErrorNotify(AmfiteatrError<DP>),

}