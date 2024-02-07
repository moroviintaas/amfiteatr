use std::sync::mpsc::{RecvError, SendError, TryRecvError, TrySendError};
use thiserror::Error;

use crate::error::AmfiError;
use crate::domain::DomainParameters;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum CommunicationError<DP: DomainParameters>{
    #[error("Send Error to {0}, text: {1}")]
    SendError(DP::AgentId, String),
    #[error("Send Error, text: {0}")]
    SendErrorUnspecified(String),
    #[error("Broadcast Send Error (on {0})")]
    BroadcastSendError(DP::AgentId),
    #[error("Broadcast Send Error")]
    BroadcastSendErrorUnspecified,
    #[error("Recv Error from {0}, text: {1}")]
    RecvError(DP::AgentId, String),
    #[error("Recv Error, text: {0}")]
    RecvErrorUnspecified(String),
    #[error("TryRecv Error (empty) from {0}")]
    RecvEmptyBufferError(DP::AgentId),
    #[error("TryRecv Error (empty")]
    RecvEmptyBufferErrorUnspecified,
    #[error("TryRecv Error (disconnected")]
    RecvPeerDisconnectedErrorUnspecified,
    #[error("TryRecv Error (disconnected) from {0}")]
    RecvPeerDisconnectedError(DP::AgentId),
    #[error("Serialize Error, text: {0}")]
    SerializeError(String),
    #[error("Deserialize Error, text: {0}")]
    DeserializeError(String),
    #[error("No such connection")]
    NoSuchConnection,
    #[error("Connection to agent {0} not found")]
    ConnectionToAgentNotFound(DP::AgentId),
    #[error("Duplicateed Agent: {0}")]
    DuplicatedAgent(DP::AgentId),


}

impl<Spec: DomainParameters> CommunicationError<Spec>{

    pub fn specify_id(self, id: Spec::AgentId) -> Self{
        match self{
            CommunicationError::SendErrorUnspecified(s) => Self::SendError(id, s),
            CommunicationError::BroadcastSendErrorUnspecified => Self::BroadcastSendError(id),
            CommunicationError::RecvErrorUnspecified(s) => Self::RecvError(id, s),
            CommunicationError::RecvEmptyBufferErrorUnspecified => Self::RecvEmptyBufferError(id),
            CommunicationError::RecvPeerDisconnectedErrorUnspecified => Self::RecvPeerDisconnectedError(id),
            any => any
        }
    }
}
/*
impl Display for CommError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}*/

impl<Spec: DomainParameters> From<RecvError> for CommunicationError<Spec>{
    fn from(e: RecvError) -> Self {
        Self::RecvErrorUnspecified(format!("{e:}"))
    }
}
impl<Spec: DomainParameters, T> From<SendError<T>> for CommunicationError<Spec>{
    fn from(e: SendError<T>) -> Self {
        Self::SendErrorUnspecified(format!("{e:}"))
    }
}
impl<Spec: DomainParameters> From<TryRecvError> for CommunicationError<Spec>{
    fn from(e: TryRecvError) -> Self {
        match e{
            TryRecvError::Empty => Self::RecvEmptyBufferErrorUnspecified,
            TryRecvError::Disconnected => Self::RecvPeerDisconnectedErrorUnspecified
        }
    }
}

impl<Spec: DomainParameters, T> From<TrySendError<T>> for CommunicationError<Spec>{
    fn from(e: TrySendError<T>) -> Self {
        Self::SendErrorUnspecified(format!("{e:}"))
    }
}

impl <Spec: DomainParameters> From<CommunicationError<Spec>> for AmfiError<Spec>{
    fn from(value: CommunicationError<Spec>) -> Self {
        Self::Communication(value)
    }
}