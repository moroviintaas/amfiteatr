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
use crate::scheme::{AgentMessage, Scheme, EnvironmentMessage, Renew, RenewWithEffect, Reward};




/// Implementation of environment using [`HashMap`](std::collections::HashMap) to store
/// individual [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)'s to communicate with
/// agents. This implementation does not provide tracing.
/// If you need tracing refer to analogous implementation of
/// [`TracingHashMapEnvironment`](crate::env::TracingHashMapEnvironment).
pub struct HashMapEnvironment<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>{

    comm_endpoints: HashMap<S::AgentId, C>,
    penalties: HashMap<S::AgentId, S::UniversalReward>,
    game_state: ST,
    game_steps: u64,
    game_violator: Option<S::AgentId>,
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
HashMapEnvironment<S, ST, C>{

    pub fn new(
        game_state: ST,
        comm_endpoints:  HashMap<S::AgentId, C>) -> Self{

        #[cfg(feature = "log_debug")]
        let k:Vec<S::AgentId> = comm_endpoints.keys().cloned().collect();
        #[cfg(feature = "log_debug")]
        log::debug!("Creating environment with:{k:?}");

        let penalties: HashMap<S::AgentId, S::UniversalReward> = comm_endpoints.keys()
            .map(|agent| (agent.clone(), S::UniversalReward::neutral()))
            .collect();

        Self{comm_endpoints, game_state, penalties, game_steps: 0, game_violator: None}
    }

    pub fn replace_state(&mut self, state: ST){
        self.game_state = state
    }

    pub fn comms(&self) -> &HashMap<S::AgentId, C> {
        &self.comm_endpoints
    }
    pub fn comms_mut(&mut self) -> &mut HashMap<S::AgentId, C> {
        &mut self.comm_endpoints
    }

    pub fn completed_steps(&self) -> u64{
        self.game_steps
    }
}


impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
StatefulEnvironment<S> for HashMapEnvironment<S, ST, C>{

    type State = ST;
    //type Updates = <Vec<(S::AgentId, S::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &S::AgentId, action: &S::ActionType)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {
        self.game_steps += 1;
        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e| AmfiteatrError::Game{source: e})


    }

    fn game_violator(&self) -> Option<&S::AgentId> {
        self.game_violator.as_ref()
    }

    fn set_game_violator(&mut self, game_violator: Option<S::AgentId>) {
        self.game_violator = game_violator;
    }
}

impl<
    S: Scheme,
    ST: GameStateWithPayoffs<S>,
    C: EnvironmentEndpoint<S> >
ScoreEnvironment<S> for HashMapEnvironment<S, ST, C>{

    fn process_action_penalise_illegal(
        &mut self,
        agent: &S::AgentId,
        action: &S::ActionType,
        penalty_reward: S::UniversalReward)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {
        self.game_steps += 1;

        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e|{
                self.penalties.insert(agent.clone(), penalty_reward + &self.penalties[agent]);
                AmfiteatrError::Game{source: e}
            })

    }

    fn actual_state_score_of_player(
        &self, agent: &S::AgentId) -> S::UniversalReward {

        self.game_state.state_payoff_of_player(agent)
    }

    fn actual_penalty_score_of_player
    (&self, agent: &S::AgentId) -> S::UniversalReward {

        self.penalties.get(agent).unwrap_or(&S::UniversalReward::neutral()).to_owned()
    }
}



impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
CommunicatingEndpointEnvironment<S> for HashMapEnvironment<S, ST, C> {
    type CommunicationError = CommunicationError<S>;

    fn send_to(&mut self, agent_id: &S::AgentId, message: EnvironmentMessage<S>)
        -> Result<(), Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.send(message))?
    }

    fn blocking_receive_from(&mut self, agent_id: &S::AgentId)
                             -> Result<AgentMessage<S>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.receive_blocking())?
    }

    fn nonblocking_receive_from(&mut self, agent_id: &S::AgentId)
                                -> Result<Option<AgentMessage<S>>, Self::CommunicationError> {

        self.comm_endpoints.get_mut(agent_id).ok_or(CommunicationError::NoSuchConnection{
            connection: format!("To agent {agent_id:}")
        })
            .map(|v| v.receive_non_blocking())?
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
BroadcastingEndpointEnvironment<S> for HashMapEnvironment<S, ST, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<S>) -> Result<(), Self::CommunicationError> {
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

impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
 EnvironmentWithAgents<S> for HashMapEnvironment<S, ST, C>{
    type PlayerIterator = Vec<S::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.comm_endpoints.keys().cloned().collect()
    }




}

impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
ListPlayers<S> for HashMapEnvironment<S, ST, C>{
    type IterType = <Vec<S::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.comm_endpoints.keys().cloned().collect::<Vec<S::AgentId>>().into_iter()
    }


}




/// __(Experimental)__ builder for [`HashMapEnvironment`]
pub struct HashMapEnvironmentBuilder<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S> >{
    state_opt: Option<ST>,
    comm_endpoints: HashMap<S::AgentId,  C>,


}

impl <S: Scheme, ST: SequentialGameState<S>, C: EnvironmentEndpoint<S>>
HashMapEnvironmentBuilder<S, ST, C>{


    pub fn new() -> Self{
        Self{comm_endpoints: HashMap::new(),  state_opt: None}
    }


}


impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
Default for HashMapEnvironmentBuilder<S, ST, C> {

    fn default() -> Self {
        Self{
            state_opt: None,
            comm_endpoints: HashMap::new(),
        }
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
EnvironmentBuilderTrait<S, HashMapEnvironment<S, ST, C>> for HashMapEnvironmentBuilder<S, ST, C>{
    type Comm = C;

    fn build(self) -> Result<HashMapEnvironment<S, ST, C>, ModelError<S>>{


        Ok(HashMapEnvironment::new(
            self.state_opt.ok_or(ModelError::MissingState)?,
            self.comm_endpoints))

    }

    fn add_comm(mut self, agent_id: &S::AgentId, comm: C) -> Result<Self, ModelError<S>>{

        let _ = &mut self.comm_endpoints.insert(agent_id.clone(), comm);
        Ok(self)
    }

    fn with_state(mut self, state: ST) -> Result<Self, ModelError<S>>{
        self.state_opt = Some(state);
        Ok(self)
    }
}

impl<
S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
ReinitEnvironment<S> for HashMapEnvironment<S, ST, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<S>>::State) {

        self.game_state = initial_state;
        self.game_steps = 0;
        for vals in self.penalties.values_mut(){
            *vals = S::UniversalReward::neutral();
        }
    }
}
impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    CP: EnvironmentEndpoint<S>,
    Seed
> ReseedEnvironment<S, Seed> for HashMapEnvironment<S, ST, CP>
    where <Self as StatefulEnvironment<S>>::State: Renew<S, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<S>>{
        self.set_game_violator(None);
        self.game_steps = 0;
        self.game_state.renew_from(seed)
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone + RenewWithEffect<S, Seed>,
    CP: EnvironmentEndpoint<S>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<S, Seed> for HashMapEnvironment<S, ST, CP>
    where <Self as StatefulEnvironment<S>>::State: RenewWithEffect<S, Seed>,
          <<Self as StatefulEnvironment<S>>::State as RenewWithEffect<S, Seed>>::Effect:
          IntoIterator<Item=(S::AgentId, AgentSeed)>{
    //type Observation = <<Self as StatefulEnvironment<S>>::State as RenewWithSideEffect<S, Seed>>::SideEffect;
    type Observation = AgentSeed;
    type InitialObservations = HashMap<S::AgentId, Self::Observation>;

    fn reseed_with_observation(&mut self, seed: Seed) -> Result<Self::InitialObservations, AmfiteatrError<S>>{
        self.set_game_violator(None);
        self.game_steps = 0;
        self.game_state.renew_with_effect_from(seed)
            .map(|agent_observation_iter|
                agent_observation_iter.into_iter().collect())
    }
}

impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
AutoEnvironment<S> for HashMapEnvironment<S, ST, C>
where Self: CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
+ StatefulEnvironment<S>
+ EnvironmentWithAgents<S>
+ BroadcastingEndpointEnvironment<S>, S: Scheme
{
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.run_round_robin_no_rewards_truncating(truncate_steps)
    }
}

impl<S: Scheme,
    ST: SequentialGameState<S> ,
    C: EnvironmentEndpoint<S>>
AutoEnvironmentWithScores<S> for HashMapEnvironment<S, ST, C>
    where HashMapEnvironment<S, ST, C>:
    CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
    + ScoreEnvironment<S>
    + EnvironmentWithAgents<S>
    + BroadcastingEndpointEnvironment<S>
{
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.run_round_robin_with_rewards_truncating(truncate_steps)
    }
}