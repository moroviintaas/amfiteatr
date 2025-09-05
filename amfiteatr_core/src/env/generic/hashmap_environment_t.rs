use std::collections::HashMap;
use crate::comm::{EnvironmentEndpoint};
use crate::env::{BroadcastingEndpointEnvironment, CommunicatingEndpointEnvironment, SequentialGameState, GameStateWithPayoffs, EnvironmentWithAgents, ScoreEnvironment, StatefulEnvironment, ReinitEnvironment, TracingEnvironment, ReseedEnvironment, ReseedEnvironmentWithObservation, GameTrajectory, AutoEnvironment, RoundRobinEnvironment, AutoEnvironmentWithScores, RoundRobinUniversalEnvironment};
use crate::env::generic::{HashMapEnvironment};
use crate::error::{AmfiteatrError, CommunicationError};
use crate::scheme::{AgentMessage, Scheme, EnvironmentMessage, Renew, RenewWithEffect};

/// Implementation of environment using [`HashMap`](std::collections::HashMap) to store
/// individual [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)'s to communicate with
/// agents. This implementation provides tracing.
/// If you don't need tracing consider using analogous implementation of
/// [`HashMapEnvironment`](crate::env::HashMapEnvironment).
pub struct TracingHashMapEnvironment<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>{

    base_environment: HashMapEnvironment<S, ST,C>,
    history: GameTrajectory<S, ST>
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    Comm: EnvironmentEndpoint<S>> TracingHashMapEnvironment<S, ST, Comm>{

    pub fn new(
        game_state: ST,
        comm_endpoints: HashMap<S::AgentId, Comm>) -> Self{

        /*
        let k:Vec<S::AgentId> = comm_endpoints.keys().copied().collect();
        debug!("Creating environment with:{k:?}");

        let penalties: HashMap<S::AgentId, S::UniversalReward> = comm_endpoints.keys()
            .map(|agent| (*agent, S::UniversalReward::neutral()))
            .collect();

         */

        let base_environment = HashMapEnvironment::new(game_state, comm_endpoints);


        Self{base_environment, history: Default::default() }
    }
    pub fn completed_steps(&self) -> u64{
        self.history.number_of_steps() as u64
    }

    

}



impl<
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    C: EnvironmentEndpoint<S>>
StatefulEnvironment<S> for TracingHashMapEnvironment<S, ST,C>{

    type State = ST;
    //type Updates = <Vec<(S::AgentId, S::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        self.base_environment.state()
    }

    fn process_action(&mut self, agent: &S::AgentId, action: &S::ActionType)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {

        let state_clone = self.state().clone();

        match self.base_environment.process_action(agent, action){
            Ok(updates) => {

                //self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), true));
                self.history.register_step_point(state_clone, agent.clone(), action.clone(), true)?;
                if self.base_environment.state().is_finished(){
                    //self.history.finalize(self.base_environment.state().clone());
                    self.history.finish(self.base_environment.state().clone())?;
                }
                Ok(updates)
            }
            Err(e) => {

                //self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), false));
                if self.base_environment.state().is_finished(){
                    //self.history.finalize(self.base_environment.state().clone());
                    self.history.finish(self.base_environment.state().clone())?;
                }
                self.history.register_step_point(state_clone.clone(), agent.clone(), action.clone(), false)?;
                Err(e)
            }
        }
    }
}

impl<
    S: Scheme,
    ST: GameStateWithPayoffs<S> + Clone,
    C: EnvironmentEndpoint<S> >
ScoreEnvironment<S> for TracingHashMapEnvironment<S, ST, C>{
    fn process_action_penalise_illegal(
        &mut self, agent: &S::AgentId, action: &S::ActionType, penalty_reward: S::UniversalReward)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {

        let state_clone = self.state().clone();
        match self.base_environment.process_action_penalise_illegal(agent, action, penalty_reward){
            Ok(updates) => {


                self.history.register_step_point(state_clone, agent.clone(), action.clone(), true)?;
                if self.base_environment.state().is_finished(){
                    self.history.finish(self.base_environment.state().clone())?
                }
                Ok(updates)
            }
            Err(e) => {

                self.history.register_step_point(state_clone, agent.clone(), action.clone(), false)?;
                if self.base_environment.state().is_finished(){
                    //self.history.finalize(self.base_environment.state().clone());
                    self.history.finish(self.base_environment.state().clone())?
                }
                Err(e)
            }
        }

    }

    fn actual_state_score_of_player(&self, agent: &S::AgentId) -> S::UniversalReward {
        self.base_environment.actual_state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &S::AgentId) -> S::UniversalReward {
        self.base_environment.actual_penalty_score_of_player(agent)
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
CommunicatingEndpointEnvironment<S> for TracingHashMapEnvironment<S, ST, C>{
    type CommunicationError = CommunicationError<S>;

    fn send_to(&mut self, agent_id: &S::AgentId, message: EnvironmentMessage<S>)
        -> Result<(), Self::CommunicationError> {

        self.base_environment.send_to(agent_id, message)
    }

    fn blocking_receive_from(&mut self, agent_id: &S::AgentId)
                             -> Result<AgentMessage<S>, Self::CommunicationError> {

        self.base_environment.blocking_receive_from(agent_id)
    }

    fn nonblocking_receive_from(&mut self, agent_id: &S::AgentId)
                                -> Result<Option<AgentMessage<S>>, Self::CommunicationError> {

        self.base_environment.nonblocking_receive_from(agent_id)
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
BroadcastingEndpointEnvironment<S> for TracingHashMapEnvironment<S, ST, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<S>) -> Result<(), Self::CommunicationError> {
        self.base_environment.send_to_all(message)
    }
}

impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
 EnvironmentWithAgents<S> for TracingHashMapEnvironment<S, ST, C>{
    type PlayerIterator = Vec<S::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.base_environment.players()
    }
}


impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
TracingEnvironment<S, ST> for TracingHashMapEnvironment<S, ST, C>{
    fn trajectory(&self) -> &GameTrajectory<S, ST> {
        &self.history
    }
}

impl<
S: Scheme,
    ST: SequentialGameState<S> + Clone,
    C: EnvironmentEndpoint<S>>
ReinitEnvironment<S> for TracingHashMapEnvironment<S, ST, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<S>>::State) {
        self.base_environment.reinit(initial_state);
        self.history.clear();
    }
}


impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    CP: EnvironmentEndpoint<S>,
    Seed
> ReseedEnvironment<S, Seed> for TracingHashMapEnvironment<S, ST, CP>
    where <Self as StatefulEnvironment<S>>::State: Renew<S, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<S>>{

        self.history.clear();
        self.base_environment.reseed(seed)
    }
}
impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone + RenewWithEffect<S, Seed>,
    CP: EnvironmentEndpoint<S>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<S, Seed> for TracingHashMapEnvironment<S, ST, CP>
    where <Self as StatefulEnvironment<S>>::State: RenewWithEffect<S, Seed>,
          <<Self as StatefulEnvironment<S>>::State as RenewWithEffect<S, Seed>>::Effect:
          IntoIterator<Item=(S::AgentId, AgentSeed)>{
    //type Observation = <<Self as StatefulEnvironment<S>>::State as RenewWithSideEffect<S, Seed>>::SideEffect;
    type Observation = AgentSeed;
    type InitialObservations = HashMap<S::AgentId, Self::Observation>;

    fn reseed_with_observation(&mut self, seed: Seed) -> Result<Self::InitialObservations, AmfiteatrError<S>>{
        self.base_environment.reseed_with_observation(seed)
    }
}


impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
AutoEnvironment<S> for TracingHashMapEnvironment<S, ST, C>
where  HashMapEnvironment<S, ST, C>: AutoEnvironment<S>{
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.base_environment.run_round_robin_no_rewards_truncating(truncate_steps)
    }
}

impl<S: Scheme,
    ST: SequentialGameState<S>,
    C: EnvironmentEndpoint<S>>
AutoEnvironmentWithScores<S> for TracingHashMapEnvironment<S, ST, C>
    where HashMapEnvironment<S, ST, C>:AutoEnvironmentWithScores<S> + ScoreEnvironment<S>,
Self: ScoreEnvironment<S>

{
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.base_environment.run_round_robin_with_rewards_truncating(truncate_steps)
    }
}