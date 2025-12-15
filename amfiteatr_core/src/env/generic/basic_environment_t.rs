use std::collections::HashMap;

use crate::{
    env::*,
    scheme::*,
    comm::{EnvironmentAdapter, BroadcastingEnvironmentAdapter}
};
use crate::scheme::Renew;
use crate::env::generic::BasicEnvironment;
use crate::error::AmfiteatrError;


/// This is generic implementation of environment using single endpoint construction
/// ([`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)).
/// This environment does provide game tracing.
/// If you don't want tracing consider using [`BasicEnvironment`](crate::env::BasicEnvironment).
#[derive(Debug)]
pub struct TracingBasicEnvironment<S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S>>{

    base_environment: BasicEnvironment<S, ST, CP>,
    history: GameTrajectory<S, ST>
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S>
> TracingBasicEnvironment<S, ST, CP>{

    pub fn new(game_state: ST, adapter: CP) -> Self{
        Self{
            base_environment: BasicEnvironment::new(game_state, adapter),
            history: Default::default(),
        }
    }

    pub fn insert_penalty_template(&mut self, penalties:  HashMap<S::AgentId, S::UniversalReward>){

        self.base_environment.insert_illegal_reward_template(penalties)

    }
    pub fn set_penalty_template(&mut self, agent: S::AgentId, penalty: S::UniversalReward){
        self.base_environment.set_illegal_reward_template(agent, penalty)
    }
    pub fn completed_steps(&self) -> u64{
        self.history.number_of_steps() as u64
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S> + ListPlayers<S>
> ListPlayers<S> for TracingBasicEnvironment<S, ST, CP>{
    type IterType = <Vec<S::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.base_environment.players()
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>  + Clone,
    CP: EnvironmentAdapter<S>
> StatefulEnvironment<S> for TracingBasicEnvironment<S, ST, CP>{
    type State = ST;

    fn state(&self) -> &Self::State {
        self.base_environment.state()
    }

    fn process_action(&mut self, agent: &<S as Scheme>::AgentId, action: &<S as Scheme>::ActionType)
                      -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {
        let state_clone = self.state().clone();

        match self.base_environment.process_action(agent, action){

            Ok(updates) => {

                //self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), true));
                self.history.register_step_point(state_clone, agent.clone(), action.clone(), true)?;
                if self.base_environment.state().is_finished(){
                    self.history.finish(self.base_environment.state().clone())?;
                }
                Ok(updates)
            }
            Err(e) => {

                self.history.register_step_point(state_clone, agent.clone(), action.clone(), false)?;
                if self.base_environment.state().is_finished(){
                    self.history.finish(self.base_environment.state().clone())?;
                }
                //self.history.push_trace_step(EnvironmentTraceStep::new(state_clone, agent.clone(), action.clone(), false));
                Err(e)
            }
        }
    }

    fn game_violator(&self) -> Option<&S::AgentId> {
        self.base_environment.game_violator()
    }

    fn set_game_violator(&mut self, game_violator: Option<S::AgentId>) {
        self.base_environment.set_game_violator(game_violator)
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    CP: BroadcastingEnvironmentAdapter<S>,
    Seed
> ReseedEnvironment<S, Seed> for TracingBasicEnvironment<S, ST, CP>
where <Self as StatefulEnvironment<S>>::State: Renew<S, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<S>>{

        self.history.clear();
        self.base_environment.reseed(seed)
    }
}
impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone + RenewWithEffect<S, Seed>,
    CP: BroadcastingEnvironmentAdapter<S>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<S, Seed> for TracingBasicEnvironment<S, ST, CP>
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

impl <
    S: Scheme,
    ST: GameStateWithPayoffs<S> + Clone,
    CP: EnvironmentAdapter<S>
> ScoreEnvironment<S> for TracingBasicEnvironment<S, ST, CP>{
    fn process_action_penalise_illegal(
        &mut self,
        agent: &<S as Scheme>::AgentId,
        action: &<S as Scheme>::ActionType,
        penalty_reward: <S as Scheme>::UniversalReward)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {

        let state_clone = self.state().clone();
        match self.base_environment.process_action_penalise_illegal(agent, action, penalty_reward){
            Ok(updates) => {

                self.history.register_step_point(state_clone, agent.clone(), action.clone(), true)?;
                if self.base_environment.state().is_finished(){
                    self.history.finish(self.base_environment.state().clone())?;
                }
                Ok(updates)
            }
            Err(e) => {

                self.history.register_step_point(state_clone, agent.clone(), action.clone(), false)?;
                if self.base_environment.state().is_finished(){
                    self.history.finish(self.base_environment.state().clone())?;
                }
                Err(e)
            }
        }
    }

    fn actual_state_score_of_player(&self, agent: &<S as Scheme>::AgentId) -> <S as Scheme>::UniversalReward {
        self.base_environment.actual_state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &<S as Scheme>::AgentId) -> <S as Scheme>::UniversalReward {
        self.base_environment.actual_penalty_score_of_player(agent)
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> CommunicatingEnvironmentSingleQueue<S> for TracingBasicEnvironment<S, ST, CP>{
    fn send(&mut self, agent_id: &<S as Scheme>::AgentId, message: crate::scheme::EnvironmentMessage<S>)
            -> Result<(), crate::error::CommunicationError<S>> {
        self.base_environment.send(agent_id, message)
    }

    fn blocking_receive(&mut self)
                        -> Result<(<S as Scheme>::AgentId, crate::scheme::AgentMessage<S>), crate::error::CommunicationError<S>> {
        self.base_environment.blocking_receive()
    }

    fn nonblocking_receive(&mut self)
                           -> Result<Option<(<S as Scheme>::AgentId, crate::scheme::AgentMessage<S>)>, crate::error::CommunicationError<S>> {
        self.base_environment.nonblocking_receive()
    }
}


impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> BroadcastingEnvironmentSingleQueue<S> for TracingBasicEnvironment<S, ST, CP>{


    fn send_all(&mut self, message: crate::scheme::EnvironmentMessage<S>) -> Result<(), crate::error::CommunicationError<S>> {
        self.base_environment.send_all(message)
    }
}




impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    CP: BroadcastingEnvironmentAdapter<S>
> ReinitEnvironment<S> for TracingBasicEnvironment<S, ST, CP>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<S>>::State) {
        self.base_environment.reinit(initial_state);
        self.history.clear()
    }
}





impl<S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S>>
TracingEnvironment<S, ST> for TracingBasicEnvironment<S, ST, CP>{
    fn trajectory(&self) -> &GameTrajectory<S, ST> {
        &self.history
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> AutoEnvironment<S> for TracingBasicEnvironment<S, ST, CP>{

    #[inline]
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.base_environment.run_truncating(truncate_steps)
    }
}


impl <
    S: Scheme,
    ST: GameStateWithPayoffs<S>,
    CP: EnvironmentAdapter<S> + ListPlayers<S> + BroadcastingEnvironmentAdapter<S>
> AutoEnvironmentWithScores<S> for TracingBasicEnvironment<S, ST, CP>{
    #[inline]
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.base_environment.run_with_scores_truncating(truncate_steps)
    }
}

impl <
    S: Scheme,
    ST: GameStateWithPayoffs<S> + SequentialGameState<S> + Clone,
    CP: EnvironmentAdapter<S> + ListPlayers<S> + BroadcastingEnvironmentAdapter<S>
> AutoEnvironmentWithScoresAndPenalties<S> for TracingBasicEnvironment<S, ST, CP>
where {
    #[inline]
    fn run_with_scores_and_penalties_truncating<P: Fn(&<Self as StatefulEnvironment<S>>::State, &S::AgentId) -> S::UniversalReward>
    (&mut self, penalty: P, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        self.base_environment.run_with_scores_and_penalties_truncating(penalty, truncate_steps)
    }
}