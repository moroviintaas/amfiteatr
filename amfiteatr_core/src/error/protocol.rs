use thiserror::Error;
use crate::error::AmfiteatrError;
use crate::domain::DomainParameters;

/// Error for capturing misbehavior in game protocol.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum ProtocolError<DP: DomainParameters>{
    #[error("lost contact with {:}", .0)]
    BrokenComm(DP::AgentId),
    #[error("agent {:} attempted to move on turn of {:}", .0, .1)]
    ViolatedOrder(DP::AgentId, DP::AgentId),
    #[error("agent {:} called to move, however called states that {:} should move this time", .0, .1)]
    OrderDesync(DP::AgentId, DP::AgentId),
    #[error("agent {:} received kill", .0)]
    ReceivedKill(DP::AgentId),
    #[error("agent {:} has no possible action", .0)]
    NoPossibleAction(DP::AgentId),
    #[error("agent {} has exited the game", .0)]
    PlayerExited(DP::AgentId),
    #[error("Player {} refused to select action (command Quit)", .0)]
    PlayerSelectedNoneAction(DP::AgentId),

}

impl<DP: DomainParameters> From<ProtocolError<DP>> for AmfiteatrError<DP>{
    fn from(value: ProtocolError<DP>) -> Self {
        Self::Protocol{source: value}
    }
}