
use std::collections::{HashMap};
use std::sync::{Arc, Mutex};
use std::thread;
use log::{debug, error};
use crate::domain::{DomainParameters};
use crate::agent::{AutomaticAgent, ReinitAgent, StatefulAgent};
use crate::env::*;
use crate::error::{AmfiError, CommunicationError, WorldError};

pub struct GenericModel<
    DP: DomainParameters + 'static,
    Env: EnvironmentWithAgents<DP>
        + BroadcastingEnv<DP>
        + CommunicatingEnv<DP, CommunicationError=CommunicationError<DP>>,
    A: AutomaticAgent<DP> + Send

>{
    environment: Env,
    local_agents: HashMap<DP::AgentId, Arc<Mutex<Box<A>>>>,
}

impl<
    DP: DomainParameters + 'static,
    Env: EnvironmentWithAgents<DP>
        + BroadcastingEnv<DP>
        + CommunicatingEnv<DP, CommunicationError=CommunicationError<DP>>,
    A: AutomaticAgent<DP> + Send
>GenericModel<DP, Env, A>{


    pub fn new(environment: Env, local_agents: HashMap<DP::AgentId, Arc<Mutex<Box<A>>>>) -> Self{
        Self{environment, local_agents}
    }




    pub fn play<F: Fn(&mut Env) -> Result<(), AmfiError<DP>>>(&mut self, environment_run: F) -> Result<(), AmfiError<DP>>{

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
            //self.environment.run_round_robin_uni_rewards().map_err(|e|{
            environment_run(&mut self.environment).map_err(|e|{
                error!("Environment run error: {e:}");
                e
            }).unwrap();

        });

        Ok(())

    }
    pub fn play_rr_uni_reward(&mut self) -> Result<(), AmfiError<DP>>
    where Env: RoundRobinUniversalEnvironment<DP>{
        self.play(| env| env.run_round_robin_uni_rewards())
    }

    pub fn reinit_agent(&mut self, agent: &DP::AgentId, new_info_set: <A as StatefulAgent<DP>>::InfoSetType)
    where A: StatefulAgent<DP> + ReinitAgent<DP>{
        if let Some(n) = self.local_agents.get(agent){
            let mut guard  = n.lock().unwrap();
            guard.as_mut().reinit(new_info_set)
        }
    }

    pub fn reinit_environment(&mut self, new_state: <Env as StatefulEnvironment<DP>>::State)
    where Env: ReinitEnvironment<DP> + StatefulEnvironment<DP>{
        self.environment.reinit(new_state)
    }



    pub fn env(&self) -> &Env{
        &self.environment
    }
    pub fn local_agents(&self) -> &HashMap<DP::AgentId, Arc<Mutex<Box<A>>>>{
        &self.local_agents
    }
}
