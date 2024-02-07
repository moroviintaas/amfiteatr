use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use std::thread;
use log::{debug, error};
use crate::domain::{DomainParameters};
use crate::agent::AutomaticAgentRewarded;
use crate::env::{EnvironmentStateUniScore};
use crate::env::automatons::rr::RoundRobinUniversalEnvironment;
use crate::comm::EnvironmentEndpoint;
use crate::env::generic::{HashMapEnvironment};
use crate::error::{AmfiError, WorldError};


/// __(Experimental)__ implementation of joint environment and local agents combined in one struct.
pub struct RoundRobinModel<
    DP: DomainParameters + 'static,
    EnvState: EnvironmentStateUniScore<DP>,
    Comm: EnvironmentEndpoint<DP>>{
    environment: HashMapEnvironment<DP, EnvState,  Comm>,
    local_agents: HashMap<DP::AgentId, Arc<Mutex<dyn AutomaticAgentRewarded<DP> + Send>>>,
}

impl<
    DP: DomainParameters + 'static,
    EnvState: EnvironmentStateUniScore<DP>,
    Comm: EnvironmentEndpoint<DP>>
RoundRobinModel<DP, EnvState, Comm>{
    pub fn new(environment: HashMapEnvironment<DP, EnvState, Comm>, local_agents: HashMap<DP::AgentId, Arc<Mutex<dyn AutomaticAgentRewarded<DP>  + Send >>>) -> Self{
        Self{environment, local_agents}
    }




    /// Run environment and aggregated local agents. Environment runs in mode sending rewards and
    /// agents run in mode collecting rewards
    pub fn play(&mut self) -> Result<(), AmfiError<DP>>{

        thread::scope(|s|{
            let mut handlers = HashMap::new();
            for (id, agent) in self.local_agents.iter(){
                let arc_agent = agent.clone();


                let handler = s.spawn( move ||{
                    debug!("Spawning thread for agent {}", id);
                    let mut guard = arc_agent.lock().or_else(|_|Err(WorldError::<DP>::AgentMutexLock)).unwrap();
                    let id = guard.id().clone();
                    guard.run().map_err(|e|{
                        error!("Agent {id:} encountered error: {e:}")
                    }).unwrap();
                });
                handlers.insert(id, handler);
            }
            self.environment.run_round_robin_with_rewards().map_err(|e|{
                error!("Environment run error: {e:}");
                e
            }).unwrap();

        });

        Ok(())

    }

    pub fn env(&self) -> &HashMapEnvironment<DP, EnvState,  Comm>{
        &self.environment
    }
    pub fn local_agents(&self) -> &HashMap<DP::AgentId, Arc<Mutex<dyn AutomaticAgentRewarded<DP> + Send>>>{
        &self.local_agents
    }
}
