
use crate::error::AmfiteatrError;
use crate::domain::v1::domain_parameters::DomainParameters;
/// Message sent by agent to environment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AgentMessage<DP: DomainParameters>{
    #[cfg_attr(feature = "serde", serde(bound(serialize = "DP::ActionType: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "DP::ActionType: serde::Deserialize<'de>")))]
    TakeAction(DP::ActionType),
    #[cfg_attr(feature = "serde", serde(bound(serialize = "AmfiteatrError<DP>: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "AmfiteatrError<DP>: serde::Deserialize<'de>")))]
    NotifyError(AmfiteatrError<DP>),
    Quit,

}