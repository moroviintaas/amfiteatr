use std::collections::HashMap;

use crate::{
    env::*, 
    domain::{DomainParameters, Reward}, 
    comm::{EnvironmentAdapter, BroadcastingEnvironmentAdapter}
};
use crate::env::ListPlayers;
use crate::domain::Renew;


/// This is generic implementation of environment using single endpoint construction
/// ([`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)).
/// This environment does not provide game tracing.
/// If you want tracing please refer to [`TracingEnvironment`](crate::env::TracingBasicEnvironment).
#[derive(Debug, Clone)]
pub struct BasicEnvironment<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: EnvironmentAdapter<DP>
>{
    adapter: CP,
    game_state: S,
    penalties: HashMap<DP::AgentId, DP::UniversalReward>
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: EnvironmentAdapter<DP>
> BasicEnvironment<DP, S, CP>{

    pub fn new(game_state: S, adapter: CP) -> Self{
        Self{game_state, adapter, penalties: HashMap::new()}
    }

    pub fn inert_penalty_template(&mut self, penalties:  HashMap<DP::AgentId, DP::UniversalReward>){

        self.penalties = penalties;

    }
    pub fn set_penalty_template(&mut self, agent: DP::AgentId, penalty: DP::UniversalReward){
        self.penalties.insert(agent, penalty);
    }
}

impl<
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP>
> ListPlayers<DP> for BasicEnvironment<DP, S, CP>{
    type IterType = <Vec<DP::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.adapter.players().collect::<Vec<DP::AgentId>>().into_iter()
    }
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: EnvironmentAdapter<DP>
> StatefulEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    type State = S;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &<DP as DomainParameters>::AgentId, action: &<DP as DomainParameters>::ActionType) 
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, <DP as DomainParameters>::GameErrorType> {
        self.game_state.forward(agent.clone(), action.clone())
    }
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP> + Clone,
    CP: BroadcastingEnvironmentAdapter<DP>,
    Seed
> ReseedEnvironment<DP, Seed> for BasicEnvironment<DP, S, CP>
where <Self as StatefulEnvironment<DP>>::State: Renew<Seed>{
    fn reseed(&mut self, seed: Seed) {
        self.game_state.renew_from(seed);
    }
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateUniScore<DP>,
    CP: EnvironmentAdapter<DP>
> ScoreEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn process_action_penalise_illegal(
        &mut self,
        agent: &<DP as DomainParameters>::AgentId,
        action: &<DP as DomainParameters>::ActionType,
        penalty_reward: <DP as DomainParameters>::UniversalReward)
        -> Result<<Self::State as EnvironmentStateSequential<DP>>::Updates, <DP as DomainParameters>::GameErrorType> {
        
            self.game_state.forward(agent.clone(), action.clone()).map_err(|e|{
                let actual_penalty = self.penalties.remove(agent).unwrap_or(<DP::UniversalReward as Reward>::neutral());

                self.penalties.insert(agent.clone(), penalty_reward + &actual_penalty);
                e
            })
    }

    fn actual_state_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.game_state.state_score_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.penalties.get(agent).unwrap_or(&DP::UniversalReward::neutral()).to_owned()
    }
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> CommunicatingAdapterEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn send(&mut self, agent_id: &<DP as DomainParameters>::AgentId,  message: crate::domain::EnvironmentMessage<DP>)
        -> Result<(), crate::error::CommunicationError<DP>> {
        self.adapter.send( agent_id, message)
    }

    fn blocking_receive(&mut self)
                        -> Result<(<DP as DomainParameters>::AgentId, crate::domain::AgentMessage<DP>), crate::error::CommunicationError<DP>> {
        self.adapter.receive_blocking()
    }

    fn nonblocking_receive(&mut self)
                           -> Result<Option<(<DP as DomainParameters>::AgentId, crate::domain::AgentMessage<DP>)>, crate::error::CommunicationError<DP>> {
        self.adapter.receive_non_blocking()
    }
}


impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> BroadConnectedEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    

    fn send_all(&mut self, message: crate::domain::EnvironmentMessage<DP>) -> Result<(), crate::error::CommunicationError<DP>> {
        self.adapter.send_all(message)
    }
}

impl <
    DP: DomainParameters,
    S: EnvironmentStateSequential<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> ReinitEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.game_state = initial_state;
        for vals in self.penalties.values_mut(){
            *vals = DP::UniversalReward::neutral();
        }
    }
}



