
use crate::error::AmfiError;
use crate::domain::v1::domain_parameters::DomainParameters;
/// Message sent by agent to environment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum AgentMessage<DP: DomainParameters>{
    TakeAction(DP::ActionType),
    NotifyError(AmfiError<DP>),
    Quit,

}