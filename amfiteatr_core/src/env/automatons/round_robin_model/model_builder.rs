
use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use crate::agent::{AgentGen, IdAgent, Policy, PresentPossibleActions, EvaluatedInformationSet, AutomaticAgentRewarded};
use crate::env::{EnvironmentBuilderTrait, EnvironmentStateUniScore};
use crate::env::automatons::rr::RoundRobinModel;
use crate::comm::{EnvironmentEndpoint, StdEnvironmentEndpoint};
use crate::env::generic::{HashMapEnvironmentBuilder};

use crate::domain::{DomainParameters};
use crate::error::WorldError;

/// __(Experimental)__ Builder for round robin model [`RoundRobinModel`]
pub struct RoundRobinModelBuilder<
    DP: DomainParameters,
    EnvState: EnvironmentStateUniScore<DP>,
    Comm: EnvironmentEndpoint<DP> >{
    env_builder: HashMapEnvironmentBuilder<DP, EnvState,  Comm>,
    local_agents: HashMap<DP::AgentId, Arc<Mutex<dyn AutomaticAgentRewarded<DP> + Send>>>,

}


impl<
    DP: DomainParameters,
    EnvState: EnvironmentStateUniScore<DP>>
RoundRobinModelBuilder<DP, EnvState,  StdEnvironmentEndpoint<DP>>
{
    pub fn with_local_generic_agent<P: Policy<DP> + 'static>(
        mut self,
        _id: DP::AgentId,
        initial_state: <P as Policy<DP>>::InfoSetType,
        policy: P)
        -> Result<Self, WorldError<DP>>
        where <P as Policy<DP>>::InfoSetType: EvaluatedInformationSet<DP> + PresentPossibleActions<DP>{

        let (comm_env, comm_agent) = StdEnvironmentEndpoint::new_pair();
        let agent = AgentGen::new( initial_state, comm_agent, policy);
        self.env_builder = self.env_builder.add_comm(&agent.id(), comm_env)?;
        self.local_agents.insert(agent.id().clone(), Arc::new(Mutex::new(agent)));
        Ok(self)

    }
}


#[allow(clippy::borrowed_box)]
impl<
    DP: DomainParameters,
    EnvState: EnvironmentStateUniScore<DP>,
    Comm: EnvironmentEndpoint<DP>>
RoundRobinModelBuilder<DP, EnvState,  Comm>{
    pub fn new() -> Self{
        Self{ env_builder: HashMapEnvironmentBuilder::new(), local_agents:HashMap::new() }
    }
    
    pub fn with_env_state(mut self, environment_state: EnvState)
        -> Result<Self, WorldError<DP>>{
        self.env_builder = self.env_builder.with_state(environment_state)?;
        Ok(self)
    }

    pub fn get_agent(&self, s: &DP::AgentId) -> Option<&Arc<Mutex<dyn AutomaticAgentRewarded<DP> + Send>>>{
        self.local_agents.get(s)


    }

    pub fn add_local_agent(mut self,
                           agent: Arc<Mutex<dyn AutomaticAgentRewarded<DP>+ Send>>,
                           env_comm: Comm)
                           -> Result<Self, WorldError<DP>>{

        let agent_guard = agent.as_ref().lock().unwrap();
        let id = agent_guard.id().clone();
        std::mem::drop(agent_guard);
        self.env_builder = self.env_builder.add_comm(&id, env_comm)?;
        self.local_agents.insert(id.clone(), agent);

        Ok(self)
    }



    pub fn with_remote_agent(mut self, agent_id: DP::AgentId,
                             env_comm: Comm) -> Result<Self, WorldError<DP>>{

        if self.local_agents.contains_key(&agent_id){
            self.local_agents.remove(&agent_id);
        }
        //self.comm_endpoints.insert(agent_id, env_comm);
        self.env_builder = self.env_builder.add_comm(&agent_id, env_comm)?;
        Ok(self)
    }

    pub fn build(self) -> Result<RoundRobinModel<DP, EnvState, Comm>, WorldError<DP>>{
        Ok(RoundRobinModel::new(self.env_builder.build()?, self.local_agents))
    }




}

impl<Spec: DomainParameters, EnvState: EnvironmentStateUniScore<Spec>,
 Comm: EnvironmentEndpoint<Spec>> Default for RoundRobinModelBuilder<Spec, EnvState, Comm> {
    fn default() -> Self {
        Self::new()
    }
}
