use crate::agent::AgentActionPair;
use crate::error::AmfiteatrError;
use crate::scheme::v1::game_scheme::Scheme;

/// Message sent by environment to agent
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EnvironmentMessage<S: Scheme>{
    YourMove,
    MoveRefused,
    GameFinished,
    GameTruncated,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S::AgentId: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S::AgentId: serde::Deserialize<'de>")))]
    GameFinishedWithIllegalAction(S::AgentId),
    Kill,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S::UpdateType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S::UpdateType: serde::Deserialize<'de>")))]
    UpdateState(S::UpdateType),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S::AgentId: serde::Serialize, S::ActionType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S::AgentId: serde::Deserialize<'de>, S::ActionType: serde::Deserialize<'de>")))]
    ActionNotify(AgentActionPair<S::AgentId, S::ActionType>),
    RewardFragment(S::UniversalReward),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "AmfiteatrError<S>: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "AmfiteatrError<S>: serde::Deserialize<'de>")))]
    ErrorNotify(AmfiteatrError<S>),

}