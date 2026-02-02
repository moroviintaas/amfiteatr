use std::collections::HashMap;

use crate::{
    env::*,
    scheme::{Scheme, Reward},
    comm::{EnvironmentAdapter, BroadcastingEnvironmentAdapter}
};
use crate::env::ListPlayers;
use crate::scheme::{AgentMessage, EnvironmentMessage, Renew, RenewWithEffect};
use crate::scheme::EnvironmentMessage::ErrorNotify;
use crate::error::{AmfiteatrError, CommunicationError};
use crate::error::ProtocolError::PlayerExited;


/// This is generic implementation of environment using single endpoint construction
/// ([`EnvironmentAdapter`](crate::comm::EnvironmentAdapter)).
/// This environment does not provide game tracing.
/// If you want tracing please refer to [`TracingEnvironment`](crate::env::TracingBasicEnvironment).
#[derive(Debug, Clone)]
pub struct BasicEnvironment<
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S>
>{
    adapter: CP,
    game_state: ST,
    penalties: HashMap<S::AgentId, S::UniversalReward>,
    game_steps: u64,
    game_violators: Option<S::AgentId>,
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S>
> BasicEnvironment<S, ST, CP>{

    pub fn new(game_state: ST, adapter: CP) -> Self{
        Self{game_state, adapter, penalties: HashMap::new(), game_steps: 0, game_violators: None}
    }

    pub fn insert_illegal_reward_template(&mut self, penalties:  HashMap<S::AgentId, S::UniversalReward>){

        self.penalties = penalties;

    }
    pub fn set_illegal_reward_template(&mut self, agent: S::AgentId, penalty: S::UniversalReward){
        self.penalties.insert(agent, penalty);
    }
    pub fn completed_steps(&self) -> u64{
        self.game_steps
    }
}

impl<
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: EnvironmentAdapter<S> + ListPlayers<S>
> ListPlayers<S> for BasicEnvironment<S, ST, CP>{
    type IterType = <Vec<S::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.adapter.players().collect::<Vec<S::AgentId>>().into_iter()
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    OneComm: EnvironmentAdapter<S>
> StatefulEnvironment<S> for BasicEnvironment<S, ST, OneComm>{
    type State = ST;

    fn state(&self) -> &Self::State {
        &self.game_state
    }

    fn process_action(&mut self, agent: &<S as Scheme>::AgentId, action: &<S as Scheme>::ActionType)
                      -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {
        self.game_steps += 1;
        self.game_state.forward(agent.clone(), action.clone())
            .map_err(|e|{
                AmfiteatrError::Game{source: e}
            })

    }

    fn game_violator(&self) -> Option<&S::AgentId> {
        self.game_violators.as_ref()
    }

    fn set_game_violator(&mut self, game_violator: Option<S::AgentId>) {
        self.game_violators = game_violator;
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S> + Clone,
    CP: BroadcastingEnvironmentAdapter<S>,
    Seed
> ReseedEnvironment<S, Seed> for BasicEnvironment<S, ST, CP>
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
    CP: BroadcastingEnvironmentAdapter<S>,
    Seed,
    AgentSeed
> ReseedEnvironmentWithObservation<S, Seed> for BasicEnvironment<S, ST, CP>
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

impl <
    S: Scheme,
    ST: GameStateWithPayoffs<S>,
    CP: EnvironmentAdapter<S>
> ScoreEnvironment<S> for BasicEnvironment<S, ST, CP>{
    fn process_action_penalise_illegal(
        &mut self,
        agent: &<S as Scheme>::AgentId,
        action: &<S as Scheme>::ActionType,
        penalty_reward: <S as Scheme>::UniversalReward)
        -> Result<<Self::State as SequentialGameState<S>>::Updates, AmfiteatrError<S>> {
            self.game_steps +=1;
        
            self.game_state.forward(agent.clone(), action.clone()).map_err(|e|{
                let actual_penalty = self.penalties.remove(agent).unwrap_or(<S::UniversalReward as Reward>::neutral());

                self.penalties.insert(agent.clone(), penalty_reward + &actual_penalty);
                AmfiteatrError::Game{source: e}
            })

    }

    fn actual_state_score_of_player(&self, agent: &<S as Scheme>::AgentId) -> <S as Scheme>::UniversalReward {
        self.game_state.state_payoff_of_player(agent)
    }

    fn actual_penalty_score_of_player(&self, agent: &<S as Scheme>::AgentId) -> <S as Scheme>::UniversalReward {
        self.penalties.get(agent).unwrap_or(&S::UniversalReward::neutral()).to_owned()
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> CommunicatingEnvironmentSingleQueue<S> for BasicEnvironment<S, ST, CP>{
    fn send(&mut self, agent_id: &<S as Scheme>::AgentId, message: crate::scheme::EnvironmentMessage<S>)
            -> Result<(), crate::error::CommunicationError<S>> {
        self.adapter.send( agent_id, message)
    }

    fn blocking_receive(&mut self)
                        -> Result<(<S as Scheme>::AgentId, crate::scheme::AgentMessage<S>), crate::error::CommunicationError<S>> {
        self.adapter.receive_blocking()
    }

    fn nonblocking_receive(&mut self)
                           -> Result<Option<(<S as Scheme>::AgentId, crate::scheme::AgentMessage<S>)>, crate::error::CommunicationError<S>> {
        self.adapter.receive_non_blocking()
    }
}


impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> BroadcastingEnvironmentSingleQueue<S> for BasicEnvironment<S, ST, CP>{
    

    fn send_all(&mut self, message: crate::scheme::EnvironmentMessage<S>) -> Result<(), crate::error::CommunicationError<S>> {
        self.adapter.send_all(message)
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> ReinitEnvironment<S> for BasicEnvironment<S, ST, CP>{
    fn reinit(&mut self, initial_state: <Self as StatefulEnvironment<S>>::State) {
        self.game_steps = 0;
        self.game_state = initial_state;
        for vals in self.penalties.values_mut(){
            *vals = S::UniversalReward::neutral();
        }
    }
}

impl <
    S: Scheme,
    ST: SequentialGameState<S>,
    CP: BroadcastingEnvironmentAdapter<S>
> AutoEnvironment<S> for BasicEnvironment<S, ST, CP>{
    fn run_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {

        let mut current_step = 0;
        if let Some(initial_updates) = self.state().first_observations(){
            for (ag, update) in initial_updates{
                self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                    .inspect_err(|e| {
                        let _ = self.send_all(ErrorNotify(e.clone().into()));
                    })?;
            }
        }
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
                                    current_step += 1;
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
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }

                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }
                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
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
    S: Scheme,
    ST: GameStateWithPayoffs<S>,
    CP: EnvironmentAdapter<S> + ListPlayers<S> + BroadcastingEnvironmentAdapter<S>
> AutoEnvironmentWithScores<S> for BasicEnvironment<S, ST, CP>{
    fn run_with_scores_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        let mut current_step = 0;
        if let Some(initial_updates) = self.state().first_observations(){
            for (ag, update) in initial_updates{
                self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                    .inspect_err(|e| {
                        let _ = self.send_all(ErrorNotify(e.clone().into()));
                    })?;
            }
        }
        let mut actual_universal_scores: HashMap<S::AgentId, S::UniversalReward> = self.players()
            .map(|id|{
                (id, S::UniversalReward::neutral())
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
                                    current_step += 1;
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
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }

                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }
                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
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
    S: Scheme,
    ST: GameStateWithPayoffs<S>,
    CP: EnvironmentAdapter<S> + ListPlayers<S> + BroadcastingEnvironmentAdapter<S>
> AutoEnvironmentWithScoresAndPenalties<S> for BasicEnvironment<S, ST, CP> {
    fn run_with_scores_and_penalties_truncating<P: Fn(&<Self as StatefulEnvironment<S>>::State,&S::AgentId)
        -> S::UniversalReward>(&mut self, penalty: P, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {

        let mut current_step = 0;
        if let Some(initial_updates) = self.state().first_observations(){
            for (ag, update) in initial_updates{
                self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                    .inspect_err(|e| {
                        let _ = self.send_all(ErrorNotify(e.clone().into()));
                    })?;
            }
        }
        let mut actual_universal_scores: HashMap<S::AgentId, S::UniversalReward> = self.players()
            .map(|id|{
                (id, S::UniversalReward::neutral())
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
                                    current_step += 1;
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
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }

                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }
                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_all(EnvironmentMessage::ErrorNotify(er.clone().into()));
                                        er

                                    })?;
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