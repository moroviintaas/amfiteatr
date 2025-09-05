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
    DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>{

    base_environment: HashMapEnvironment<DP, ST,C>,
    history: GameTrajectory<DP, ST>
}

impl<
    DP: Scheme,
    ST: SequentialGameState<DP>,
    Comm: EnvironmentEndpoint<DP>> TracingHashMapEnvironment<DP, ST, Comm>{

    pub fn new(
        game_state: ST,
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
    pub fn completed_steps(&self) -> u64{
        self.history.number_of_steps() as u64
    }

    

}



impl<
    DP: Scheme,
    ST: SequentialGameState<DP> + Clone,
    C: EnvironmentEndpoint<DP>>
StatefulEnvironment<DP> for TracingHashMapEnvironment<DP, ST,C>{

    type State = ST;
    //type Updates = <Vec<(DP::AgentId, DP::UpdateType)> as IntoIterator>::IntoIter;

    fn state(&self) -> &Self::State {
        self.base_environment.state()
    }

    fn process_action(&mut self, agent: &DP::AgentId, action: &DP::ActionType)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {

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
    DP: Scheme,
    ST: GameStateWithPayoffs<DP> + Clone,
    C: EnvironmentEndpoint<DP> >
ScoreEnvironment<DP> for TracingHashMapEnvironment<DP, ST, C>{
    fn process_action_penalise_illegal(
        &mut self, agent: &DP::AgentId, action: &DP::ActionType, penalty_reward: DP::UniversalReward)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {

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

    fn actual_state_score_of_player(&self, agent: &DP::AgentId) -> DP::UniversalReward {
        self.base_environment.actual_state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &DP::AgentId) -> DP::UniversalReward {
        self.base_environment.actual_penalty_score_of_player(agent)
    }
}

impl<
    DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
CommunicatingEndpointEnvironment<DP> for TracingHashMapEnvironment<DP, ST, C>{
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
    DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
BroadcastingEndpointEnvironment<DP> for TracingHashMapEnvironment<DP, ST, C>{
    fn send_to_all(&mut self, message: EnvironmentMessage<DP>) -> Result<(), Self::CommunicationError> {
        self.base_environment.send_to_all(message)
    }
}

impl<DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
 EnvironmentWithAgents<DP> for TracingHashMapEnvironment<DP, ST, C>{
    type PlayerIterator = Vec<DP::AgentId>;

    fn players(&self) -> Self::PlayerIterator {
        self.base_environment.players()
    }
}


impl<DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
TracingEnvironment<DP, ST> for TracingHashMapEnvironment<DP, ST, C>{
    fn trajectory(&self) -> &GameTrajectory<DP, ST> {
        &self.history
    }
}

impl<
DP: Scheme,
    ST: SequentialGameState<DP> + Clone,
    C: EnvironmentEndpoint<DP>>
ReinitEnvironment<DP> for TracingHashMapEnvironment<DP, ST, C>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.base_environment.reinit(initial_state);
        self.history.clear();
    }
}


impl <
    DP: Scheme,
    ST: SequentialGameState<DP> + Clone,
    CP: EnvironmentEndpoint<DP>,
    Seed
> ReseedEnvironment<DP, Seed> for TracingHashMapEnvironment<DP, ST, CP>
    where <Self as StatefulEnvironment<DP>>::State: Renew<DP, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>>{

        self.history.clear();
        self.base_environment.reseed(seed)
    }
}
impl <
    DP: Scheme,
    ST: SequentialGameState<DP> + Clone + RenewWithEffect<DP, Seed>,
    CP: EnvironmentEndpoint<DP>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<DP, Seed> for TracingHashMapEnvironment<DP, ST, CP>
    where <Self as StatefulEnvironment<DP>>::State: RenewWithEffect<DP, Seed>,
          <<Self as StatefulEnvironment<DP>>::State as RenewWithEffect<DP, Seed>>::Effect:
          IntoIterator<Item=(DP::AgentId, AgentSeed)>{
    //type Observation = <<Self as StatefulEnvironment<DP>>::State as RenewWithSideEffect<DP, Seed>>::SideEffect;
    type Observation = AgentSeed;
    type InitialObservations = HashMap<DP::AgentId, Self::Observation>;

    fn reseed_with_observation(&mut self, seed: Seed) -> Result<Self::InitialObservations, AmfiteatrError<DP>>{
        self.base_environment.reseed_with_observation(seed)
    }
}


impl<DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
AutoEnvironment<DP> for TracingHashMapEnvironment<DP, ST, C>
where  HashMapEnvironment<DP, ST, C>: AutoEnvironment<DP>{
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>> {
        self.base_environment.run_round_robin_no_rewards_truncating(truncate_steps)
    }
}

impl<DP: Scheme,
    ST: SequentialGameState<DP>,
    C: EnvironmentEndpoint<DP>>
AutoEnvironmentWithScores<DP> for TracingHashMapEnvironment<DP, ST, C>
    where HashMapEnvironment<DP, ST, C>:AutoEnvironmentWithScores<DP> + ScoreEnvironment<DP>,
Self: ScoreEnvironment<DP>

{
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>> {
        self.base_environment.run_round_robin_with_rewards_truncating(truncate_steps)
    }
}