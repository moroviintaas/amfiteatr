

use crate::{comm::BidirectionalEndpoint};
use crate::env::StatefulEnvironment;

use crate::domain::{DomainParameters};
use crate::error::WorldError;
/// Interface for building environment that can be dynamically extended to work with more agent.
pub trait EnvironmentBuilderTrait<DP: DomainParameters, Env: StatefulEnvironment<DP>>: Default{

    //type Environment: EnvironmentRR<Spec = Self::ProtocolSpec>;
    type Comm: BidirectionalEndpoint;

    fn build(self) -> Result<Env, WorldError<DP>>;
    fn add_comm(self, agent_id: &DP::AgentId, comm: Self::Comm) -> Result<Self, WorldError<DP>>;
    fn with_state(self, state: Env::State) -> Result<Self, WorldError<DP>>;

}

