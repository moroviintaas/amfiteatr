use std::error::Error;

use crate::{scheme::{AgentMessage, EnvironmentMessage, Scheme}, error::CommunicationError};

pub(crate) type AgentStampedMessage<S> = (<S as Scheme>::AgentId, AgentMessage<S>);
/// Environment communicating with agent using 1-1 endpoints.
/// This mean environment can receive from specified endpoint
/// (from certain agent)
pub trait CommunicatingEndpointEnvironment<S: Scheme>{
    //type Outward;
    //type Inward;
    type CommunicationError: Error;

    fn send_to(&mut self, agent_id: &S::AgentId, message: EnvironmentMessage<S>) -> Result<(), Self::CommunicationError>;
    fn blocking_receive_from(&mut self, agent_id: &S::AgentId) -> Result<AgentMessage<S>, Self::CommunicationError>;

    fn nonblocking_receive_from(&mut self, agent_id: &S::AgentId) -> Result<Option<AgentMessage<S>>, Self::CommunicationError>;


}

/// Broadcasting extension to trait [`CommunicatingEndpointEnvironment`](CommunicatingEndpointEnvironment)
pub trait BroadcastingEndpointEnvironment<Spec: Scheme>: CommunicatingEndpointEnvironment<Spec>{

    fn send_to_all(&mut self, message: EnvironmentMessage<Spec>) -> Result<(), Self::CommunicationError>;

}
/// Environment communicating with multi-agent (1-N) communication port.
/// Receiving does not require to specify agent and first queued will be
/// returned.
pub trait CommunicatingEnvironmentSingleQueue<S: Scheme>{
    
    fn send(&mut self, agent_id: &S::AgentId,  message: EnvironmentMessage<S>)
        -> Result<(), CommunicationError<S>>;
    fn blocking_receive(&mut self)
                        -> Result<(S::AgentId, AgentMessage<S>), CommunicationError<S>>;
    fn nonblocking_receive(&mut self)
                           -> Result<Option<AgentStampedMessage<S>>, CommunicationError<S>>;
        
}
/// Broadcasting extension to trait [`CommunicatingAdapterEnvironment`](CommunicatingEnvironmentSingleQueue)
pub trait BroadcastingEnvironmentSingleQueue<S: Scheme>{
    fn send_all(&mut self, message: EnvironmentMessage<S>) -> Result<(), CommunicationError<S>>;
}