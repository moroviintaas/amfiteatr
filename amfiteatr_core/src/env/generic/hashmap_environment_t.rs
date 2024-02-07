use std::collections::HashMap;
use crate::comm::EnvironmentEndpoint;
use crate::env::{
    BroadcastingEndpointEnvironment,
    CommunicatingEndpointEnvironment,
    EnvironmentStateSequential,
    EnvironmentStateUniScore,
    EnvironmentWithAgents,
    EnvironmentTraceStep,
    ScoreEnvironment,
    StatefulEnvironment,
    ReinitEnvironment,
    EnvironmentTrajectory,
    TracingEnvironment
};
use crate::env::generic::{HashMapEnvironment};
use crate::error::CommunicationError;
use crate::domain::{AgentMessage, DomainParameters, EnvironmentMessage};

/// Implementation of environment using [`HashMap`](std::collections::HashMap) to store
/// individual [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)'s to communicate with
/// agents. This implementation provides tracing.
/// If you don't need tracing consider using analogous implementation of
/// [`HashMapEnvironment`](crate::env::HashMapEnvironment).
pub struct TracingHashMapEnvironment<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>{

    base_environment: HashMapEnvironment<DP, S,C>,
    history: EnvironmentTrajectory<DP, S>
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    Comm: EnvironmentEndpoint<DP>> TracingHashMapEnvironment<DP, S, Comm>{

    pub fn new(
        game_state: S,
        comm_endpoints: HashMap<DP::AgentId, Comm>) -> Self{

        /*
        let k:Vec<DP::AgentId> = comm_endpoints.keys().copied().collect();
        debug!("Creating environment with:{k:?}");

        let penalties: HashMap<DP::AgentId, DP::UniversalReward> = comm_endpoints.keys()
            .map(|agent| (*agent, DP::UniversalReward::neutral()))
            .collect();

         */

        let base_environment = HashMapEnvironment::new(game_state, comm_endpoints);


        Self{base_environment, history: Default::default() }
    }

    

}



impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP> + Clone,
    C: EnvironmentEndpoint<DP>>
StatefulEnvironment<DP> for TracingHashMapEnvironment<DP, S,C>{

    type State = S;
    //type Updates = <Vec<(DP::AgentId, DP::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        self.base_environment.state()
    }

    fn process_action(&mut self, agent: &DP::AgentId, action: &DP::ActionType)
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, DP::GameErrorType> {

        let state_clone = self.state().clone();
        /*
        match self.action_processor.process_action(
            &mut self.game_state, agent, action){
            Ok(updates) => {
                self.history.push(
                    HistoryEntry::new(state_clone, *agent, action.clone(), true));
                Ok(updates.into_iter())
            }
            Err(err) => {
                self.history.push(
                    HistoryEntry::new(state_clone, *agent, action.clone(), false));
                Err(err)
            }
        }

         */
        match self.base_environment.process_action(agent, action){
            Ok(updates) => {
                if self.base_environment.state().is_finished(){
                    self.history.finalize(self.base_environment.state().clone());
                }
                self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), true));
                Ok(updates)
            }
            Err(e) => {
                if self.base_environment.state().is_finished(){
                    self.history.finalize(self.base_environment.state().clone());
                }
                self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), false));
                Err(e)
            }
        }
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateUniScore<DP> + Clone,
    C: EnvironmentEndpoint<DP> >
ScoreEnvironment<DP> for TracingHashMapEnvironment<DP, S, C>{
    fn process_action_penalise_illegal(
        &mut self, agent: &DP::AgentId, action: &DP::ActionType, penalty_reward: DP::UniversalReward)
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, DP::GameErrorType> {

        let state_clone = self.state().clone();
        match self.base_environment.process_action_penalise_illegal(agent, action, penalty_reward){
            Ok(updates) => {
                if self.base_environment.state().is_finished(){
                    self.history.finalize(self.base_environment.state().clone());
                }
                self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), true));
                Ok(updates)
            }
            Err(e) => {
                if self.base_environment.state().is_finished(){
                    self.history.finalize(self.base_environment.state().clone());
                }
                self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), false));
                Err(e)
            }
        }

    }

    fn actual_state_score_of_player(&self, agent: &DP::AgentId) -> DP::UniversalReward {
        self.base_environment.actual_state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &DP::AgentId) -> DP::UniversalReward {
        self.base_environment.actual_penalty_score_of_player(agent)
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
CommunicatingEndpointEnvironment<DP> for TracingHashMapEnvironment<DP, S, C>{
    type CommunicationError = CommunicationError<DP>;

    fn send_to(&mut self, agent_id: &DP::AgentId, message: EnvironmentMessage<DP>)
        -> Result<(), Self::CommunicationError> {

        self.base_environment.send_to(agent_id, message)
    }

    fn blocking_receive_from(&mut self, agent_id: &DP::AgentId)
                             -> Result<AgentMessage<DP>, Self::CommunicationError> {

        self.base_environment.blocking_receive_from(agent_id)
    }

    fn nonblocking_receive_from(&mut self, agent_id: &DP::AgentId)
                                -> Result<Option<AgentMessage<DP>>, Self::CommunicationError> {

        self.base_environment.nonblocking_receive_from(agent_id)
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
BroadcastingEndpointEnvironment<DP> for TracingHashMapEnvironment<DP, S, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<DP>) -> Result<(), Self::CommunicationError> {
        self.base_environment.send_to_all(message)
    }
}

impl<'a, DP: DomainParameters + 'a,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
 EnvironmentWithAgents<DP> for TracingHashMapEnvironment<DP, S, C>{
    type PlayerIterator = Vec<DP::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.base_environment.players()
    }
}


impl<'a, DP: DomainParameters + 'a,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
TracingEnvironment<DP, S> for TracingHashMapEnvironment<DP, S, C>{
    fn trajectory(&self) -> &EnvironmentTrajectory<DP, S> {
        &self.history
    }
}

impl<
DP: DomainParameters,
    S: EnvironmentStateSequential<DP> + Clone,
    C: EnvironmentEndpoint<DP>>
ReinitEnvironment<DP> for TracingHashMapEnvironment<DP, S, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.base_environment.reinit(initial_state);
        self.history.clear();
    }
}



