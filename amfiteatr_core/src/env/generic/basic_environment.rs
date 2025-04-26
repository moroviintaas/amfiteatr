use std::collections::HashMap;

use crate::{
    env::*, 
    domain::{DomainParameters, Reward}, 
    comm::{EnvironmentAdapter, BroadcastingEnvironmentAdapter}
};
use crate::env::ListPlayers;
use crate::domain::{AgentMessage, EnvironmentMessage, Renew, RenewWithEffect};
use crate::error::{AmfiteatrError, CommunicationError};
use crate::error::ProtocolError::PlayerExited;


/// This is generic implementation of environment using single endpoint construction
/// ([`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)).
/// This environment does not provide game tracing.
/// If you want tracing please refer to [`TracingEnvironment`](crate::env::TracingBasicEnvironment).
#[derive(Debug, Clone)]
pub struct BasicEnvironment<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP>
>{
    adapter: CP,
    game_state: S,
    penalties: HashMap<DP::AgentId, DP::UniversalReward>,
    game_steps: u64,
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP>
> BasicEnvironment<DP, S, CP>{

    pub fn new(game_state: S, adapter: CP) -> Self{
        Self{game_state, adapter, penalties: HashMap::new(), game_steps: 0}
    }

    pub fn insert_illegal_reward_template(&mut self, penalties:  HashMap<DP::AgentId, DP::UniversalReward>){

        self.penalties = penalties;

    }
    pub fn set_illegal_reward_template(&mut self, agent: DP::AgentId, penalty: DP::UniversalReward){
        self.penalties.insert(agent, penalty);
    }
    pub fn completed_steps(&self) -> u64{
        self.game_steps
    }
}

impl<
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP>
> ListPlayers<DP> for BasicEnvironment<DP, S, CP>{
    type IterType = <Vec<DP::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.adapter.players().collect::<Vec<DP::AgentId>>().into_iter()
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    OneComm: EnvironmentAdapter<DP>
> StatefulEnvironment<DP> for BasicEnvironment<DP, S, OneComm>{
    type State = S;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &<DP as DomainParameters>::AgentId, action: &<DP as DomainParameters>::ActionType) 
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {
        self.game_steps += 1;
        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e|{
                AmfiteatrError::Game{source: e}
            })

    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone,
    CP: BroadcastingEnvironmentAdapter<DP>,
    Seed
> ReseedEnvironment<DP, Seed> for BasicEnvironment<DP, S, CP>
where <Self as StatefulEnvironment<DP>>::State: Renew<DP, Seed>{
    fn reseed(&mut self, seed: Seed) -> Result<(), AmfiteatrError<DP>>{
        self.game_steps = 0;
        self.game_state.renew_from(seed)
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP> + Clone + RenewWithEffect<DP, Seed>,
    CP: BroadcastingEnvironmentAdapter<DP>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<DP, Seed> for BasicEnvironment<DP, S, CP>
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

impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP>,
    CP: EnvironmentAdapter<DP>
> ScoreEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn process_action_penalise_illegal(
        &mut self,
        agent: &<DP as DomainParameters>::AgentId,
        action: &<DP as DomainParameters>::ActionType,
        penalty_reward: <DP as DomainParameters>::UniversalReward)
        -> Result<<Self::State as SequentialGameState<DP>>::Updates, AmfiteatrError<DP>> {
            self.game_steps +=1;
        
            self.game_state.forward(agent.clone(), action.clone()).map_err(|e|{
                let actual_penalty = self.penalties.remove(agent).unwrap_or(<DP::UniversalReward as Reward>::neutral());

                self.penalties.insert(agent.clone(), penalty_reward + &actual_penalty);
                AmfiteatrError::Game{source: e}
            })

    }

    fn actual_state_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.game_state.state_payoff_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &<DP as DomainParameters>::AgentId) -> <DP as DomainParameters>::UniversalReward {
        self.penalties.get(agent).unwrap_or(&DP::UniversalReward::neutral()).to_owned()
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> CommunicatingEnvironmentSingleQueue<DP> for BasicEnvironment<DP, S, CP>{
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
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> BroadcastingEnvironmentSingleQueue<DP> for BasicEnvironment<DP, S, CP>{
    

    fn send_all(&mut self, message: crate::domain::EnvironmentMessage<DP>) -> Result<(), crate::error::CommunicationError<DP>> {
        self.adapter.send_all(message)
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> ReinitEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<DP>>::State) {
        self.game_steps = 0;
        self.game_state = initial_state;
        for vals in self.penalties.values_mut(){
            *vals = DP::UniversalReward::neutral();
        }
    }
}

impl <
    DP: DomainParameters,
    S: SequentialGameState<DP>,
    CP: BroadcastingEnvironmentAdapter<DP>
> AutoEnvironment<DP> for BasicEnvironment<DP, S, CP>{
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>> {

        let first_player = match self.current_player(){
            None => {
                #[cfg(feature = "log_warn")]
                log::warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        #[cfg(feature = "log_debug")]
        log::debug!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            match self.blocking_receive(){
                Ok((player, message)) => {
                    match message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){
                                Ok(updates) => {
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e| {
                                                let _ = self.send_all(EnvironmentMessage::ErrorNotify(e.clone().into()));
                                            })?;

                                    }

                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(e);
                                }
                            }
                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        },
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game.", player);
                            self.notify_error(
                                AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    }
                }
                Err(e) => match e{

                    CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                    CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                        //debug!("Empty channel");
                    },
                    err => {
                        #[cfg(feature = "log_error")]
                        log::error!("Failed trying to receive message");
                        self.send_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                        return Err(AmfiteatrError::Communication{source: err});
                    }


                }
            }

        }
    }
}

impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP> + BroadcastingEnvironmentAdapter<DP>
> AutoEnvironmentWithScores<DP> for BasicEnvironment<DP, S, CP>{
    fn run_with_scores(&mut self) -> Result<(), AmfiteatrError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players()
            .map(|id|{
                (id, DP::UniversalReward::neutral())
            }).collect();
        let first_player = match self.current_player(){
            None => {
                #[cfg(feature = "log_warn")]
                log::warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        #[cfg(feature = "log_debug")]
        log::debug!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            match self.blocking_receive(){
                Ok((player, message)) => {
                    match message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){
                                Ok(updates) => {
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e| {
                                                let _ = self.send_all(EnvironmentMessage::ErrorNotify(e.clone().into()));
                                            })?;

                                    }
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        self.send(player, EnvironmentMessage::RewardFragment(reward))?;
                                    }

                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(e);
                                }
                            }
                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        },
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game.", player);
                            self.notify_error(AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    }
                }
                Err(e) => match e{

                    CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                    CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                        //debug!("Empty channel");
                    },
                    err => {
                        #[cfg(feature = "log_error")]
                        log::error!("Failed trying to receive message");
                        self.send_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                        return Err(AmfiteatrError::Communication{source: err});
                    }


                }
            }

        }
    }
}

impl <
    DP: DomainParameters,
    S: GameStateWithPayoffs<DP>,
    CP: EnvironmentAdapter<DP> + ListPlayers<DP> + BroadcastingEnvironmentAdapter<DP>
> AutoEnvironmentWithScoresAndPenalties<DP> for BasicEnvironment<DP, S, CP> {
    fn run_with_scores_and_penalties<P: Fn(&<Self as StatefulEnvironment<DP>>::State,&DP::AgentId)
        -> DP::UniversalReward>(&mut self, penalty: P) -> Result<(), AmfiteatrError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players()
            .map(|id|{
                (id, DP::UniversalReward::neutral())
            }).collect();
        let first_player = match self.current_player(){
            None => {
                #[cfg(feature = "log_warn")]
                log::warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        #[cfg(feature = "log_debug")]
        log::debug!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            match self.blocking_receive(){
                Ok((player, message)) => {
                    match message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){
                                Ok(updates) => {
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e| {
                                                let _ = self.send_all(EnvironmentMessage::ErrorNotify(e.clone().into()));
                                            })?;

                                    }
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        self.send(player, EnvironmentMessage::RewardFragment(reward))?;
                                    }

                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Player {player:} performed illegal action: {action:}");
                                    let _ = self.send(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send(&player, EnvironmentMessage::RewardFragment(penalty(self.state(), &player)));
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        let _ = self.send(player, EnvironmentMessage::RewardFragment(reward));
                                    }
                                    let _ = self.send_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(e);
                                }
                            }
                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        },
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game.", player);
                            self.notify_error(AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    }
                }
                Err(e) => match e{

                    CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                    CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                        //debug!("Empty channel");
                    },
                    err => {
                        #[cfg(feature = "log_error")]
                        log::error!("Failed trying to receive message");
                        self.send_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                        return Err(AmfiteatrError::Communication{source: err});
                    }


                }
            }

        }
    }
}