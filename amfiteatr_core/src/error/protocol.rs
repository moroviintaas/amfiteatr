use thiserror::Error;
use crate::error::AmfiteatrError;
use crate::scheme::Scheme;

/// Error for capturing misbehavior in game protocol.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum ProtocolError<S: Scheme>{
    #[error("lost contact with {:}", .0)]
    BrokenComm(S::AgentId),
    #[error("agent {:} attempted to move on turn of {:}", .0, .1)]
    ViolatedOrder(S::AgentId, S::AgentId),
    #[error("agent {:} called to move, however called states that {:} should move this time", .0, .1)]
    OrderDesync(S::AgentId, S::AgentId),
    #[error("agent {:} received kill", .0)]
    ReceivedKill(S::AgentId),
    #[error("agent {:} has no possible action", .0)]
    NoPossibleAction(S::AgentId),
    #[error("agent {} has exited the game", .0)]
    PlayerExited(S::AgentId),
    #[error("Player {} refused to select action (command Quit)", .0)]
    PlayerSelectedNoneAction(S::AgentId),

}

impl<S: Scheme> From<ProtocolError<S>> for AmfiteatrError<S>{
    fn from(value: ProtocolError<S>) -> Self {
        Self::Protocol{source: value}
    }
}