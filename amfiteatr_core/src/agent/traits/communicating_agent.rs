use crate::scheme::{AgentMessage, EnvironmentMessage, Scheme};
use crate::error::{AmfiteatrError, CommunicationError};

/// Trait for agents able to communicate with environment.
/// This trait is meant to work synchronously.
///
pub trait CommunicatingAgent<S: Scheme>{
    /*
    /// An error which is returned in case of communication failure.
    /// In this crate usually [`CommError`](crate::error::CommunicationError) is used in
    /// this context.
    type CommunicationError: Error;


     */
    /// Send message to environment.
    /// Whether block on it is not specified however it is recommended that
    /// message sent should be surely delivered.
    fn send(&mut self, message: AgentMessage<S>) -> Result<(), CommunicationError<S>>;
    /// Wait for message from environment - this should block to the moment of
    /// receiving message
    fn recv(&mut self) -> Result<EnvironmentMessage<S>, CommunicationError<S>>;

    fn and_send_error(&mut self, error: AmfiteatrError<S>) -> AmfiteatrError<S>{
        let _ = self.send(AgentMessage::NotifyError(error.clone()));
        error
    }
}