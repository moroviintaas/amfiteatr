use std::collections::HashMap;

use crate::{
    env::*,
    domain::*,
    comm::{EnvironmentAdapter, BroadcastingEnvironmentAdapter}
};
use crate::domain::Renew;
use crate::env::generic::BasicEnvironment;
use crate::error::AmfiteatrError;


/// This is generic implementation of environment using single endpoint construction
/// ([`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)).
/// This environment does provide game tracing.
/// If you don't want tracing consider using [`BasicEnvironment`](crate::env::BasicEnvironment).
#[derive(Debug)]
pub struct TracingBasicEnvironment<DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP>>{

    base_environment: BasicEnvironment<DP, S, CP>,
    history: GameTrajectory<DP, S>
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP>
> TracingBasicEnvironment<DP, S, CP>{

    pub fn new(game_state: S, adapter: CP) -> Self{
        Self{
            base_environment: BasicEnvironment::new(game_state, adapter),
            history: Default::default(),
        }
    }

    pub fn insert_penalty_template(&mut self, penalties:  HashMap<DP::AgentId, DP::UniversalReward>){

        self.base_environment.insert_illegal_reward_template(penalties)

    }
    pub fn set_penalty_template(&mut self, agent: DP::AgentId, penalty: DP::UniversalReward){
        self.base_environment.set_illegal_reward_template(agent, penalty)
    }
    pub fn completed_steps(&self) -> u64{
        self.history.number_of_steps() as u64
    }
}

impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP>
> ListPlayers<DP> for TracingBasicEnvironment<DP, S, CP>{
    type IterType = <Vec<DP::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.base_environment.players()
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>  + Clone,
    CP: EnvironmentAdapter<DP>
> StatefulEnvironment<DP> for TracingBasicEnvironment<DP, S, CP>{
    type State = S;

    fn state(&self) -> &Self::State {
        self.base_environment.state()
    }

    fn process_action(&mut self, agent: &<DP as DomainParameters>::AgentId, action: &<DP as DomainParameters>::ActionType)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {
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
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone,
    CP: BroadcastingEnvironmentAdapter<DP>,
    Seed
> ReseedEnvironment<DP, Seed> for TracingBasicEnvironment<DP, S, CP>
where <Self as StatefulEnvironment<DP>>::State: Renew<DP, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>>{

        self.history.clear();
        self.base_environment.reseed(seed)
    }
}
impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone + RenewWithEffect<DP, Seed>,
    CP: BroadcastingEnvironmentAdapter<DP>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<DP, Seed> for TracingBasicEnvironment<DP, S, CP>
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

impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP> + Clone,
    CP: EnvironmentAdapter<DP>
> ScoreEnvironment<DP> for TracingBasicEnvironment<DP, S, CP>{
    fn process_action_penalise_illegal(
        &mut self,
        agent: &<DP as DomainParameters>::AgentId,
        action: &<DP as DomainParameters>::ActionType,
        penalty_reward: <DP as DomainParameters>::UniversalReward)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {

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

    fn actual_state_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.base_environment.actual_state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.base_environment.actual_penalty_score_of_player(agent)
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> CommunicatingEnvironmentSingleQueue<DP> for TracingBasicEnvironment<DP, S, CP>{
    fn send(&mut self, agent_id: &<DP as DomainParameters>::AgentId,  message: crate::domain::EnvironmentMessage<DP>)
        -> Result<(), crate::error::CommunicationError<DP>> {
        self.base_environment.send(agent_id, message)
    }

    fn blocking_receive(&mut self)
                        -> Result<(<DP as DomainParameters>::AgentId, crate::domain::AgentMessage<DP>), crate::error::CommunicationError<DP>> {
        self.base_environment.blocking_receive()
    }

    fn nonblocking_receive(&mut self)
                           -> Result<Option<(<DP as DomainParameters>::AgentId, crate::domain::AgentMessage<DP>)>, crate::error::CommunicationError<DP>> {
        self.base_environment.nonblocking_receive()
    }
}


impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> BroadcastingEnvironmentSingleQueue<DP> for TracingBasicEnvironment<DP, S, CP>{


    fn send_all(&mut self, message: crate::domain::EnvironmentMessage<DP>) -> Result<(), crate::error::CommunicationError<DP>> {
        self.base_environment.send_all(message)
    }
}




impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone,
    CP: BroadcastingEnvironmentAdapter<DP>
> ReinitEnvironment<DP> for TracingBasicEnvironment<DP, S, CP>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.base_environment.reinit(initial_state);
        self.history.clear()
    }
}





impl<DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP>>
TracingEnvironment<DP, S> for TracingBasicEnvironment<DP, S, CP>{
    fn trajectory(&self) -> &GameTrajectory<DP, S> {
        &self.history
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> AutoEnvironment<DP> for TracingBasicEnvironment<DP, S, CP>{

    #[inline]
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>> {
        self.base_environment.run_truncating(truncate_steps)
    }
}


impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP> + BroadcastingEnvironmentAdapter<DP>
> AutoEnvironmentWithScores<DP> for TracingBasicEnvironment<DP, S, CP>{
    #[inline]
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>> {
        self.base_environment.run_with_scores_truncating(truncate_steps)
    }
}

impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP> + SequentialGameState<DP> + Clone,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP> + BroadcastingEnvironmentAdapter<DP>
> AutoEnvironmentWithScoresAndPenalties<DP> for TracingBasicEnvironment<DP, S, CP>
where {
    #[inline]
    fn run_with_scores_and_penalties_truncating<P: Fn(&<Self as StatefulEnvironment<DP>>::State, &DP::AgentId) -> DP::UniversalReward>
    (&mut self, penalty: P, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<DP>> {
        self.base_environment.run_with_scores_and_penalties_truncating(penalty, truncate_steps)
    }
}