
use crate::error::AmfiteatrError;
use crate::scheme::v1::game_scheme::Scheme;
/// Message sent by agent to environment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AgentMessage<S: Scheme>{
    #[cfg_attr(feature = "serde", serde(bound(serialize = "S::ActionType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "S::ActionType: serde::Deserialize<'de>")))]
    TakeAction(S::ActionType),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "AmfiteatrError<S>: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "AmfiteatrError<S>: serde::Deserialize<'de>")))]
    NotifyError(AmfiteatrError<S>),
    Quit,

}