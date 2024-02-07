use std::error::Error;

use crate::{domain::{AgentMessage, EnvironmentMessage, DomainParameters}, error::CommunicationError};


/// Environment communicating with agent using endpoints.
/// This mean environment can receive from specified endpoint
/// (from certain agent)
pub trait CommunicatingEndpointEnvironment<DP: DomainParameters>{
    //type Outward;
    //type Inward;
    type CommunicationError: Error;

    fn send_to(&mut self, agent_id: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), Self::CommunicationError>;
    fn blocking_receive_from(&mut self, agent_id: &DP::AgentId) -> Result<AgentMessage<DP>, Self::CommunicationError>;

    fn nonblocking_receive_from(&mut self, agent_id: &DP::AgentId) -> Result<Option<AgentMessage<DP>>, Self::CommunicationError>;


}

/// Broadcasting extension to trait [`CommunicatingEndpointEnvironment`](CommunicatingEndpointEnvironment)
pub trait BroadcastingEndpointEnvironment<Spec: DomainParameters>: CommunicatingEndpointEnvironment<Spec>{

    fn send_to_all(&mut self, message: EnvironmentMessage<Spec>) -> Result<(), Self::CommunicationError>;

}
/// Environment communicating with multi-agent adapter.
/// Receiving does not require to specify agent and first queued will be
/// returned.
pub trait CommunicatingAdapterEnvironment<DP: DomainParameters>{
    
    fn send(&mut self, agent_id: &DP::AgentId,  message: EnvironmentMessage<DP>)
        -> Result<(), CommunicationError<DP>>;
    fn blocking_receive(&mut self)
                        -> Result<(DP::AgentId, AgentMessage<DP>), CommunicationError<DP>>;
    fn nonblocking_receive(&mut self)
                           -> Result<Option<(DP::AgentId, AgentMessage<DP>)>, CommunicationError<DP>>;
        
}
/// Broadcasting extension to trait [`CommunicatingAdapterEnvironment`](CommunicatingAdapterEnvironment)
pub trait BroadConnectedEnvironment<DP: DomainParameters>{
    fn send_all(&mut self, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>;
}