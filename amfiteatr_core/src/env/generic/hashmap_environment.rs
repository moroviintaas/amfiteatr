use std::collections::{HashMap};
use crate::env::{
    BroadcastingEndpointEnvironment,
    CommunicatingEndpointEnvironment, ReseedEnvironmentWithObservation, EnvironmentBuilderTrait,
    SequentialGameState, GameStateWithPayoffs, EnvironmentWithAgents, ReinitEnvironment,
    ReseedEnvironment, ScoreEnvironment, StatefulEnvironment, ListPlayers, AutoEnvironment,
    RoundRobinEnvironment,
    RoundRobinUniversalEnvironment,
    AutoEnvironmentWithScores};
use crate::{comm::EnvironmentEndpoint};

use crate::error::{AmfiteatrError, CommunicationError, ModelError};
use crate::domain::{AgentMessage, DomainParameters, EnvironmentMessage, Renew, RenewWithEffect, Reward};




/// Implementation of environment using [`HashMap`](std::collections::HashMap) to store
/// individual [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)'s to communicate with
/// agents. This implementation does not provide tracing.
/// If you need tracing refer to analogous implementation of
/// [`TracingHashMapEnvironment`](crate::env::TracingHashMapEnvironment).
pub struct HashMapEnvironment<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>{

    comm_endpoints: HashMap<DP::AgentId, C>,
    penalties: HashMap<DP::AgentId, DP::UniversalReward>,
    game_state: S,
    game_steps: u64,
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
HashMapEnvironment<DP, S, C>{

    pub fn new(
        game_state: S,
        comm_endpoints:  HashMap<DP::AgentId, C>) -> Self{

        #[cfg(feature = "log_debug")]
        let k:Vec<DP::AgentId> = comm_endpoints.keys().cloned().collect();
        #[cfg(feature = "log_debug")]
        log::debug!("Creating environment with:{k:?}");

        let penalties: HashMap<DP::AgentId, DP::UniversalReward> = comm_endpoints.keys()
            .map(|agent| (agent.clone(), DP::UniversalReward::neutral()))
            .collect();

        Self{comm_endpoints, game_state, penalties, game_steps: 0}
    }

    pub fn replace_state(&mut self, state: S){
        self.game_state = state
    }

    pub fn comms(&self) -> &HashMap<DP::AgentId, C> {
        &self.comm_endpoints
    }
    pub fn comms_mut(&mut self) -> &mut HashMap<DP::AgentId, C> {
        &mut self.comm_endpoints
    }

    pub fn completed_steps(&self) -> u64{
        self.game_steps
    }
}


impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
StatefulEnvironment<DP> for HashMapEnvironment<DP, S, C>{

    type State = S;
    //type Updates = <Vec<(DP::AgentId, DP::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &DP::AgentId, action: &DP::ActionType) 
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {
        self.game_steps += 1;
        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e| AmfiteatrError::Game{source: e})


    }




}

impl<
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP>,
    C: EnvironmentEndpoint<DP> >
ScoreEnvironment<DP> for HashMapEnvironment<DP, S, C>{

    fn process_action_penalise_illegal(
        &mut self,
        agent: &DP::AgentId,
        action: &DP::ActionType,
        penalty_reward: DP::UniversalReward)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {
        self.game_steps += 1;

        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e|{
                self.penalties.insert(agent.clone(), penalty_reward + &self.penalties[agent]);
                AmfiteatrError::Game{source: e}
            })

    }

    fn actual_state_score_of_player(
        &self, agent: &DP::AgentId) -> DP::UniversalReward {

        self.game_state.state_payoff_of_player(agent)
    }

    fn actual_penalty_score_of_player
    (&self, agent: &DP::AgentId) -> DP::UniversalReward {

        self.penalties.get(agent).unwrap_or(&DP::UniversalReward::neutral()).to_owned()
    }
}



impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
CommunicatingEndpointEnvironment<DP> for HashMapEnvironment<DP, S, C> {
    type CommunicationError = CommunicationError<DP>;

    fn send_to(&mut self, agent_id: &DP::AgentId, message: EnvironmentMessage<DP>)
        -> Result<(), Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.send(message))?
    }

    fn blocking_receive_from(&mut self, agent_id: &DP::AgentId)
                             -> Result<AgentMessage<DP>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.receive_blocking())?
    }

    fn nonblocking_receive_from(&mut self, agent_id: &DP::AgentId)
                                -> Result<Option<AgentMessage<DP>>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.receive_non_blocking())?
    }
}

impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
BroadcastingEndpointEnvironment<DP> for HashMapEnvironment<DP, S, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<DP>) -> Result<(), Self::CommunicationError> {
        let mut result:Option<Self::CommunicationError> = None;

        for (_id, comm) in self.comm_endpoints.iter_mut(){
            #[cfg(feature = "log_trace")]
            log::trace!("While broadcasting. Sending to {_id} message: {message:?}");
            if let Err(sending_err) = comm.send(message.clone()){
                #[cfg(feature = "log_warn")]
                log::warn!("While broadcasting. Error sending to {_id}: {sending_err}");
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
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
 EnvironmentWithAgents<DP> for HashMapEnvironment<DP, S, C>{
    type PlayerIterator = Vec<DP::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.comm_endpoints.keys().cloned().collect()
    }


}

impl<DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
ListPlayers<DP> for HashMapEnvironment<DP, S, C>{
    type IterType = <Vec<DP::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.comm_endpoints.keys().cloned().collect::<Vec<DP::AgentId>>().into_iter()
    }


}




/// __(Experimental)__ builder for [`HashMapEnvironment`]
pub struct HashMapEnvironmentBuilder<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP> >{
    state_opt: Option<S>,
    comm_endpoints: HashMap<DP::AgentId,  C>,


}

impl <DP: DomainParameters, S: SequentialGameState<DP>, C: EnvironmentEndpoint<DP>>
HashMapEnvironmentBuilder<DP, S, C>{


    pub fn new() -> Self{
        Self{comm_endpoints: HashMap::new(),  state_opt: None}
    }


}


impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
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
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
EnvironmentBuilderTrait<DP, HashMapEnvironment<DP, S, C>> for HashMapEnvironmentBuilder<DP, S, C>{
    type Comm = C;

    fn build(self) -> Result<HashMapEnvironment<DP, S, C>, ModelError<DP>>{


        Ok(HashMapEnvironment::new(
            self.state_opt.ok_or(ModelError::MissingState)?,
            self.comm_endpoints))

    }

    fn add_comm(mut self, agent_id: &DP::AgentId, comm: C) -> Result<Self, ModelError<DP>>{

        let _ = &mut self.comm_endpoints.insert(agent_id.clone(), comm);
        Ok(self)
    }

    fn with_state(mut self, state: S) -> Result<Self, ModelError<DP>>{
        self.state_opt = Some(state);
        Ok(self)
    }
}

impl<
DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
ReinitEnvironment<DP> for HashMapEnvironment<DP, S, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {

        self.game_state = initial_state;
        self.game_steps = 0;
        for vals in self.penalties.values_mut(){
            *vals = DP::UniversalReward::neutral();
        }
    }
}
impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone,
    CP: EnvironmentEndpoint<DP>,
    Seed
> ReseedEnvironment<DP, Seed> for HashMapEnvironment<DP, S, CP>
    where <Self as StatefulEnvironment<DP>>::State: Renew<DP, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>>{
        self.game_steps = 0;
        self.game_state.renew_from(seed)
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone + RenewWithEffect<DP, Seed>,
    CP: EnvironmentEndpoint<DP>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<DP, Seed> for HashMapEnvironment<DP, S, CP>
    where <Self as StatefulEnvironment<DP>>::State: RenewWithEffect<DP, Seed>,
          <<Self as StatefulEnvironment<DP>>::State as RenewWithEffect<DP, Seed>>::Effect:
          IntoIterator<Item=(DP::AgentId, AgentSeed)>{
    //type Observation = <<Self as StatefulEnvironment<DP>>::State as RenewWithSideEffect<DP, Seed>>::SideEffect;
    type Observation = AgentSeed;
    type InitialObservations = HashMap<DP::AgentId, Self::Observation>;

    fn reseed_with_observation(&mut self, seed: Seed) -> Result<Self::InitialObservations, AmfiteatrError<DP>>{
        self.game_steps = 0;
        self.game_state.renew_with_effect_from(seed)
            .map(|agent_observation_iter|
                agent_observation_iter.into_iter().collect())
    }
}

impl<DP: DomainParameters,
    S: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
AutoEnvironment<DP> for HashMapEnvironment<DP, S, C>
where Self: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
+ StatefulEnvironment<DP>
+ EnvironmentWithAgents<DP>
+ BroadcastingEndpointEnvironment<DP>, DP: DomainParameters {
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<usize, AmfiteatrError<DP>> {
        self.run_round_robin_truncating(truncate_steps)
    }
}

impl<DP: DomainParameters,
    S: SequentialGameState<DP> ,
    C: EnvironmentEndpoint<DP>>
AutoEnvironmentWithScores<DP> for HashMapEnvironment<DP, S, C>
    where HashMapEnvironment<DP, S, C>:
    CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
    + ScoreEnvironment<DP>
    + EnvironmentWithAgents<DP>
    + BroadcastingEndpointEnvironment<DP>
{
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<usize, AmfiteatrError<DP>> {
        self.run_round_robin_with_rewards()
    }
}