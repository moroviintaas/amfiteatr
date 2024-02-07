use std::error::Error;
use std::fmt::Debug;

use crate::{
    domain::{
        EnvironmentMessage,
        DomainParameters,
        AgentMessage
    },
    error::CommunicationError
};
/// Trait for structures using to communicate in synchronous mode between two objects.
pub trait BidirectionalEndpoint {
    /// The type that is sent via this endpoint.
    /// In scope of this crate, for environment it will be usually
    /// [`EnvMessage`](crate::domain::EnvironmentMessage) or [`AgentMessage`](crate::domain::AgentMessage)
    type OutwardType: Debug;
    /// The type that is received via this endpoint.
    /// In scope of this crate, for environment it will be usually
    /// [`EnvMessage`](crate::domain::EnvironmentMessage) or [`AgentMessage`](crate::domain::AgentMessage)
    type InwardType: Debug;
    /// The error type that can be caused during communication.
    /// In scope of this crate, for environment it will be usually
    /// [`CommunicationError`](crate::error::CommunicationError)
    type Error: Debug + Error;

    /// Method used to send message. Message can be queued on the side of receiver.
    /// Sender should not block waiting for receiver to consume message.
    fn send(&mut self, message: Self::OutwardType) -> Result<(), Self::Error>;
    /// Method used to receive message. This method should block waiting for message to come.
    fn receive_blocking(&mut self) -> Result<Self::InwardType, Self::Error>;
    /// Method used to receive message. This method should not block.
    fn receive_non_blocking(&mut self) -> Result<Option<Self::InwardType>, Self::Error>;
}

impl<T: ?Sized> BidirectionalEndpoint for Box<T>
where T: BidirectionalEndpoint {
    type OutwardType = T::OutwardType;
    type InwardType = T::InwardType;

    type Error = T::Error;

    fn send(&mut self, message: Self::OutwardType) -> Result<(), Self::Error> {
        self.as_mut().send(message)
    }

    fn receive_blocking(&mut self) -> Result<Self::InwardType, Self::Error> {
        self.as_mut().receive_blocking()
    }

    fn receive_non_blocking(&mut self) -> Result<Option<Self::InwardType>, Self::Error> {
        self.as_mut().receive_non_blocking()
    }
}

/// Help trait for [`BidirectionalEndpoint`](BidirectionalEndpoint) fitted to work on environment
/// side.
pub trait EnvironmentEndpoint<DP: DomainParameters>:
BidirectionalEndpoint<
    OutwardType = EnvironmentMessage<DP>,
    InwardType = AgentMessage<DP>,
    Error = CommunicationError<DP>>{}

impl<DP: DomainParameters, T> EnvironmentEndpoint<DP> for T
where T: BidirectionalEndpoint<
    OutwardType = EnvironmentMessage<DP>,
    InwardType = AgentMessage<DP>,
    Error = CommunicationError<DP>>{}

/// Help trait for [`BidirectionalEndpoint`](BidirectionalEndpoint) fitted to work on agent
/// side
pub trait AgentEndpoint<DP: DomainParameters>: BidirectionalEndpoint<
    OutwardType = AgentMessage<DP>,
    InwardType = EnvironmentMessage<DP>,
    Error = CommunicationError<DP>>{}

impl<DP: DomainParameters, T> AgentEndpoint<DP> for T
where T: BidirectionalEndpoint<
    OutwardType = AgentMessage<DP>,
    InwardType = EnvironmentMessage<DP>,
    Error = CommunicationError<DP>>{}

/// This trait is to be implemented by structs that are meant to be single
/// communication endpoint for environment.
///
pub trait EnvironmentAdapter<DP: DomainParameters>{

    /// Method for sending message to agent
    fn send(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>)
    -> Result<(), CommunicationError<DP>>;
    /// Method for receiving - it blocks waiting for incoming message.
    /// This method does not specify agent from whom message is expected.
    /// Instead it looks for first incoming message (or first queued) and then
    /// outputs tuple of agent id and received message wrapped in `Result`.
    fn receive_blocking(&mut self) -> Result<(DP::AgentId, AgentMessage<DP>), CommunicationError<DP>>;
    /// Method for receiving - it does not block.
    /// If there is no queued input it returns Ok(None).
    /// This method does not specify agent from whom message is expected.
    /// Instead it looks for first incoming message (or first queued) and then
    /// outputs tuple of agent id and received message wrapped in `Result`.
    fn receive_non_blocking(&mut self) -> Result<Option<(DP::AgentId, AgentMessage<DP>)>, CommunicationError<DP>>;

    /// Diagnostic message to state if agent is available - if it can be addressed
    /// to send message.
    fn is_agent_connected(&self, agent_id: &DP::AgentId) -> bool;
}


/// This trait is to be implemented by structs to be used as agent communication endpoint.
/// The difference between this trait and [`AgentEndpoint`](AgentEndpoint) is it's purpose
/// to use paired with [`EnvironmentAdapter`](EnvironmentAdapter) instead
/// [`EnvironmentEndpoint`](EnvironmentEndpoint).
/// Interface looks similar, however there is difference in implementation.
/// Corresponding [`EnvironmentAdapter`](EnvironmentAdapter)  is connected
/// to many agents so this implementation must in some way communicate agent id
/// for environment communication adapter to output message tupled with agent id.
/// For example implementing this trait with [`mpsc`](std::sync::mpsc) channel
/// is of tuple type `(AgentId, AgentMessage)` instead of bare `AgentMessage`.
/// Different communication channels may avoid this problem.
/// __TL;DR__:
/// `AgentAdapter` is paired with `EnvironmentAdapter`
/// while `AgentEndpoint` is paired with `EnvironmentEndpoint`
/// `AgentMessage` over channel.
pub trait AgentAdapter<DP: DomainParameters>{
    fn send(&mut self, message: AgentMessage<DP>) -> Result<(), CommunicationError<DP>>;
    fn receive(&mut self) -> Result<EnvironmentMessage<DP>, CommunicationError<DP>>;
}

/// EnvironmentAdapter extension to clone message and send to every connected agent.
pub trait BroadcastingEnvironmentAdapter<DP: DomainParameters>: EnvironmentAdapter<DP>{
    fn send_all(&mut self, message: EnvironmentMessage<DP>) ->  Result<(), CommunicationError<DP>>;
}