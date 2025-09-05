

use crate::{comm::BidirectionalEndpoint};
use crate::env::StatefulEnvironment;

use crate::scheme::{Scheme};
use crate::error::ModelError;
/// Interface for building environment that can be dynamically extended to work with more agent.
pub trait EnvironmentBuilderTrait<S: Scheme, Env: StatefulEnvironment<S>>: Default{

    //type Environment: EnvironmentRR<Spec = Self::ProtocolSpec>;
    type Comm: BidirectionalEndpoint;

    fn build(self) -> Result<Env, ModelError<S>>;
    fn add_comm(self, agent_id: &S::AgentId, comm: Self::Comm) -> Result<Self, ModelError<S>>;
    fn with_state(self, state: Env::State) -> Result<Self, ModelError<S>>;

}

