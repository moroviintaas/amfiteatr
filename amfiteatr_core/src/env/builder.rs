

use crate::{comm::BidirectionalEndpoint};
use crate::env::StatefulEnvironment;

use crate::scheme::{Scheme};
use crate::error::ModelError;
/// Interface for building environment that can be dynamically extended to work with more agent.
pub trait EnvironmentBuilderTrait<DP: Scheme, Env: StatefulEnvironment<DP>>: Default{

    //type Environment: EnvironmentRR<Spec = Self::ProtocolSpec>;
    type Comm: BidirectionalEndpoint;

    fn build(self) -> Result<Env, ModelError<DP>>;
    fn add_comm(self, agent_id: &DP::AgentId, comm: Self::Comm) -> Result<Self, ModelError<DP>>;
    fn with_state(self, state: Env::State) -> Result<Self, ModelError<DP>>;

}

