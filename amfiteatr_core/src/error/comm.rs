use std::sync::mpsc::{RecvError, SendError, TryRecvError, TrySendError};
use thiserror::Error;

use crate::error::AmfiteatrError;
use crate::scheme::Scheme;


/// Error during communication.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
pub enum CommunicationError<S: Scheme>{
    #[error("Send Error to {0}, text: {1}")]
    SendError(S::AgentId, String),
    #[error("Send Error, text: {0}")]
    SendErrorUnspecified(String),
    #[error("Broadcast Send Error (on {0})")]
    BroadcastSendError(S::AgentId),
    #[error("Broadcast Send Error")]
    BroadcastSendErrorUnspecified,
    #[error("Recv Error from {0}, text: {1}")]
    RecvError(S::AgentId, String),
    #[error("Recv Error, text: {0}")]
    RecvErrorUnspecified(String),
    #[error("TryRecv Error (empty) from {0}")]
    RecvEmptyBufferError(S::AgentId),
    #[error("TryRecv Error (empty")]
    RecvEmptyBufferErrorUnspecified,
    #[error("TryRecv Error (disconnected")]
    RecvPeerDisconnectedErrorUnspecified,
    #[error("TryRecv Error (disconnected) from {0}")]
    RecvPeerDisconnectedError(S::AgentId),
    #[error("Serialize Error, text: {0}")]
    SerializeError(String),
    #[error("Deserialize Error, text: {0}")]
    DeserializeError(String),
    #[error("No such` connection")]
    NoSuchConnection{ connection: String},
    #[error("Connection to agent {0} not found")]
    ConnectionToAgentNotFound(S::AgentId),
    #[error("Duplicated Agent: {0}")]
    DuplicatedAgent(S::AgentId),
    #[error("Connection initialization error for agent: {description:}")]
    ConnectionInitialization{
        description: String
    }


}

impl<Spec: Scheme> CommunicationError<Spec>{

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

impl<Spec: Scheme> From<RecvError> for CommunicationError<Spec>{
    fn from(e: RecvError) -> Self {
        Self::RecvErrorUnspecified(format!("{e:}"))
    }
}
impl<Spec: Scheme, T> From<SendError<T>> for CommunicationError<Spec>{
    fn from(e: SendError<T>) -> Self {
        Self::SendErrorUnspecified(format!("{e:}"))
    }
}
impl<Spec: Scheme> From<TryRecvError> for CommunicationError<Spec>{
    fn from(e: TryRecvError) -> Self {
        match e{
            TryRecvError::Empty => Self::RecvEmptyBufferErrorUnspecified,
            TryRecvError::Disconnected => Self::RecvPeerDisconnectedErrorUnspecified
        }
    }
}

impl<Spec: Scheme, T> From<TrySendError<T>> for CommunicationError<Spec>{
    fn from(e: TrySendError<T>) -> Self {
        Self::SendErrorUnspecified(format!("{e:}"))
    }
}

impl <Spec: Scheme> From<CommunicationError<Spec>> for AmfiteatrError<Spec>{
    fn from(value: CommunicationError<Spec>) -> Self {
        Self::Communication {source: value}
    }
}