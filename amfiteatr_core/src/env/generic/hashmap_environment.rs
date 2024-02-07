use std::collections::{HashMap};

use log::debug;
use crate::env::{BroadcastingEndpointEnvironment, CommunicatingEndpointEnvironment, EnvironmentBuilderTrait, EnvironmentStateSequential, EnvironmentStateUniScore, EnvironmentWithAgents, ReinitEnvironment, ScoreEnvironment, StatefulEnvironment};
use crate::{comm::EnvironmentEndpoint};
use crate::error::{CommunicationError, WorldError};
use crate::domain::{AgentMessage, DomainParameters, EnvironmentMessage, Reward};




/// Implementation of environment using [`HashMap`](std::collections::HashMap) to store
/// individual [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)'s to communicate with
/// agents. This implementation does not provide tracing.
/// If you need tracing refer to analogous implementation of
/// [`TracingHashMapEnvironment`](crate::env::TracingHashMapEnvironment).
pub struct HashMapEnvironment<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>{

    comm_endpoints: HashMap<DP::AgentId, C>,
    penalties: HashMap<DP::AgentId, DP::UniversalReward>,
    game_state: S,
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
HashMapEnvironment<DP, S, C>{

    pub fn new(
        game_state: S,
        comm_endpoints:  HashMap<DP::AgentId, C>) -> Self{

        let k:Vec<DP::AgentId> = comm_endpoints.keys().cloned().collect();
        debug!("Creating environment with:{k:?}");

        let penalties: HashMap<DP::AgentId, DP::UniversalReward> = comm_endpoints.keys()
            .map(|agent| (agent.clone(), DP::UniversalReward::neutral()))
            .collect();

        Self{comm_endpoints, game_state, penalties}
    }

    pub fn replace_state(&mut self, state: S){
        self.game_state = state
    }
}


impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
StatefulEnvironment<DP> for HashMapEnvironment<DP, S, C>{

    type State = S;
    //type Updates = <Vec<(DP::AgentId, DP::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &DP::AgentId, action: &DP::ActionType) 
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, DP::GameErrorType> {
        //let updates = self.action_processor.process_action(&mut self.game_state, agent, action)?;
        self.game_state.forward(agent.clone(), action.clone())
        //Ok(updates)

    }




}

impl<
    DP: DomainParameters,
    S: EnvironmentStateUniScore<DP>,
    C: EnvironmentEndpoint<DP> >
ScoreEnvironment<DP> for HashMapEnvironment<DP, S, C>{

    fn process_action_penalise_illegal(
        &mut self,
        agent: &DP::AgentId,
        action: &DP::ActionType,
        penalty_reward: DP::UniversalReward)
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, DP::GameErrorType> {


        self.game_state.forward(agent.clone(), action.clone()).map_err(|e|{
            self.penalties.insert(agent.clone(), penalty_reward + &self.penalties[agent]);
            e
        })
    }

    fn actual_state_score_of_player(
        &self, agent: &DP::AgentId) -> DP::UniversalReward {

        self.game_state.state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player
    (&self, agent: &DP::AgentId) -> DP::UniversalReward {

        self.penalties.get(agent).unwrap_or(&DP::UniversalReward::neutral()).to_owned()
    }
}



impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
CommunicatingEndpointEnvironment<DP> for HashMapEnvironment<DP, S, C> {
    type CommunicationError = CommunicationError<DP>;

    fn send_to(&mut self, agent_id: &DP::AgentId, message: EnvironmentMessage<DP>)
        -> Result<(), Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection)
            .map(|v| v.send(message))?
    }

    fn blocking_receive_from(&mut self, agent_id: &DP::AgentId)
                             -> Result<AgentMessage<DP>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection)
            .map(|v| v.receive_blocking())?
    }

    fn nonblocking_receive_from(&mut self, agent_id: &DP::AgentId)
                                -> Result<Option<AgentMessage<DP>>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection)
            .map(|v| v.receive_non_blocking())?
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
BroadcastingEndpointEnvironment<DP> for HashMapEnvironment<DP, S, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<DP>) -> Result<(), Self::CommunicationError> {
        let mut result:Option<Self::CommunicationError> = None;

        for comm in self.comm_endpoints.values_mut(){
            if let Err(sending_err) = comm.send(message.clone()){
                result = Some(sending_err)
            }
        }

        match result{
            Some(e) => Err(e),
            None => Ok(())
        }
    }
}

impl<'a, DP: DomainParameters + 'a,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
 EnvironmentWithAgents<DP> for HashMapEnvironment<DP, S, C>{
    type PlayerIterator = Vec<DP::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.comm_endpoints.keys().cloned().collect()
    }


}

/// __(Experimental)__ builder for [`HashMapEnvironment`]
pub struct HashMapEnvironmentBuilder<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP> >{
    state_opt: Option<S>,
    comm_endpoints: HashMap<DP::AgentId,  C>,


}

impl <DP: DomainParameters, S: EnvironmentStateSequential<DP>, C: EnvironmentEndpoint<DP>>
HashMapEnvironmentBuilder<DP, S, C>{


    pub fn new() -> Self{
        Self{comm_endpoints: HashMap::new(),  state_opt: None}
    }


}


impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
Default for HashMapEnvironmentBuilder<DP, S, C> {

    fn default() -> Self {
        Self{
            state_opt: None,
            comm_endpoints: HashMap::new(),
        }
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
EnvironmentBuilderTrait<DP, HashMapEnvironment<DP, S, C>> for HashMapEnvironmentBuilder<DP, S, C>{
    type Comm = C;

    fn build(self) -> Result<HashMapEnvironment<DP, S, C>, WorldError<DP>>{


        Ok(HashMapEnvironment::new(
            self.state_opt.ok_or(WorldError::MissingState)?,
            self.comm_endpoints))

    }

    fn add_comm(mut self, agent_id: &DP::AgentId, comm: C) -> Result<Self, WorldError<DP>>{

        let _ = &mut self.comm_endpoints.insert(agent_id.clone(), comm);
        Ok(self)
    }

    fn with_state(mut self, state: S) -> Result<Self, WorldError<DP>>{
        self.state_opt = Some(state);
        Ok(self)
    }
}

impl<
DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    C: EnvironmentEndpoint<DP>>
ReinitEnvironment<DP> for HashMapEnvironment<DP, S, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.game_state = initial_state;
        for vals in self.penalties.values_mut(){
            *vals = DP::UniversalReward::neutral();
        }
    }
}
