use std::collections::HashMap;
use crate::env::{BroadcastingEndpointEnvironment, CommunicatingEndpointEnvironment, SequentialGameState, EnvironmentWithAgents, ScoreEnvironment, StatefulEnvironment};
use crate::error::{CommunicationError, AmfiteatrError};
use crate::error::ProtocolError::PlayerExited;
use crate::scheme::{AgentMessage, EnvironmentMessage, Scheme, Reward};
use crate::scheme::EnvironmentMessage::ErrorNotify;


/// Interface for environment using round robin strategy for listening to agents' messages.
pub trait RoundRobinEnvironment<S: Scheme>{
    /// Runs environment listening to agents in RoundRobin order.
    ///
    /// Argument `truncate_steps` is the number of steps after which the game is truncated.
    /// Set to `None` to disable.
    fn run_round_robin_no_rewards_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>>;
    fn run_round_robin_no_rewards(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_round_robin_no_rewards_truncating(None)
    }

}
/// Similar interface to [`RoundRobinEnvironment`], but it must ensure agents receive rewards.
pub trait RoundRobinUniversalEnvironment<S: Scheme> : RoundRobinEnvironment<S>{
    /// Runs environment listening to agents in RoundRobin order.
    ///
    /// Argument `truncate_steps` is the number of steps after which the game is truncated.
    /// Set to `None` to disable.
    fn run_round_robin_with_rewards_truncating(&mut self, truncate_steps: Option<usize>)
        -> Result<(), AmfiteatrError<S>>;

    /// Alias to [`run_round_robin_with_rewards_truncating`](RoundRobinUniversalEnvironment::run_round_robin_with_rewards_truncating)
    fn run_round_robin_truncating(&mut self, truncate_steps: Option<usize>)
                                               -> Result<(), AmfiteatrError<S>>{
        self.run_round_robin_with_rewards_truncating(truncate_steps)
    }
    /// Runs environment without truncation, default implementation calls
    /// [`run_round_robin_with_rewards_truncating(None)`](RoundRobinUniversalEnvironment::run_round_robin_with_rewards_truncating)
    /// Exactly the same default behaviour as [`run_round_robin`](RoundRobinUniversalEnvironment::run_round_robin) just more explicit in naming.
    fn run_round_robin_with_rewards(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_round_robin_with_rewards_truncating(None)
    }
    /// Runs environment without truncation, default implementation calls
    /// [`run_round_robin_with_rewards_truncating(None)`](RoundRobinUniversalEnvironment::run_round_robin_with_rewards_truncating)
    fn run_round_robin(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_round_robin_with_rewards_truncating(None)
    }
}
/// Similar interface to [`RoundRobinEnvironment`] and [`RoundRobinUniversalEnvironment`],
/// in addition to rewards based on current game state, penalties for illegal actions are sent.
/// This is __experimental__ interface.
pub trait RoundRobinPenalisingUniversalEnvironment<
    S: Scheme,
    P: Fn(&<Self as StatefulEnvironment<S>>::State, &S::AgentId) -> S::UniversalReward
>: RoundRobinUniversalEnvironment<S> + StatefulEnvironment<S>{
    /// Runs environment listening to agents in RoundRobin order.
    ///
    /// Argument `truncate_steps` is the number of steps after which the game is truncated.
    /// Set to `None` to disable.
    fn run_round_robin_with_rewards_penalise_truncating(&mut self, penalty_f: P, truncate_steps: Option<usize>)
        -> Result<(), AmfiteatrError<S>>;
    fn run_round_robin_with_rewards_penalise(&mut self, penalty_f: P) -> Result<(), AmfiteatrError<S>>{
        self.run_round_robin_with_rewards_penalise_truncating(penalty_f, None)
    }
}



pub(crate) trait EnvironmentRRInternal<S: Scheme>{
    fn notify_error(&mut self, error: AmfiteatrError<S>) -> Result<(), CommunicationError<S>>;
    fn send_message(&mut self, agent: &S::AgentId, message: EnvironmentMessage<S>) -> Result<(), CommunicationError<S>>;

}

impl<Env, S: Scheme> EnvironmentRRInternal<S> for Env
where Env: CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
 + StatefulEnvironment<S>
 + EnvironmentWithAgents<S>
 + BroadcastingEndpointEnvironment<S>,

S: Scheme
{
    fn notify_error(&mut self, error: AmfiteatrError<S>) -> Result<(), CommunicationError<S>> {
        self.send_to_all(ErrorNotify(error))
    }

    fn send_message(&mut self, agent: &S::AgentId, message: EnvironmentMessage<S>) -> Result<(), CommunicationError<S>>{
        self.send_to(agent, message)
            .inspect_err(|e| {
                self.notify_error(e.clone().into())
                    .unwrap_or_else(|_| panic!("Failed broadcasting error message {}", e));
            })
    }



}

/*
Generic implementations of RoundRobinEnvironment, RoundRobinUniversalEnvironment and
RoundRobinPenalisingUniversalEnvironment are very similar, thus in future it will be rewritten
as some macro.

 */


impl<Env, S: Scheme> RoundRobinEnvironment<S> for Env
where Env: CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
 + StatefulEnvironment<S>
 + EnvironmentWithAgents<S>
 + BroadcastingEndpointEnvironment<S>, S: Scheme
{
    fn run_round_robin_no_rewards_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        let mut current_step = 0;

        if let Some(initial_updates) = self.state().first_observations(){
            for (ag, update) in initial_updates{
                self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                    .inspect_err(|e| {
                        let _ = self.send_to_all(ErrorNotify(e.clone().into()));
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
        #[cfg(feature = "log_info")]
        log::info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                match self.nonblocking_receive_from(&player){
                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){

                                Ok(updates) => {
                                    current_step += 1;
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e| {
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                            })?;

                                    }

                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }

                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }

                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_to_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er

                                    })?;
                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_to_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game (via Quit signal).", player);
                            self.notify_error(AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            #[cfg(feature = "log_error")]
                            log::error!("Failed trying to receive from {} with {err}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiteatrError::Communication{source: err});
                        }


                    }
                }
            }
        }
    }
}


impl<Env, S: Scheme> RoundRobinUniversalEnvironment<S> for Env
where Env: CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
 + ScoreEnvironment<S>
 + EnvironmentWithAgents<S>
 + BroadcastingEndpointEnvironment<S>, S: Scheme
{
    fn run_round_robin_with_rewards_truncating(&mut self, truncate_steps: Option<usize>) -> Result<(), AmfiteatrError<S>> {
        let mut current_step = 0;
        let mut actual_universal_scores: HashMap<S::AgentId, S::UniversalReward> = self.players().into_iter()
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
        #[cfg(feature = "log_info")]
        log::info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                //trace!("Listening to messages from {}", player);
                match self.nonblocking_receive_from(&player){

                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){
                                Ok(updates) => {
                                    current_step += 1;
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e|{
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                            })?;
                                    }
                                    #[cfg(feature = "log_debug")]
                                    log::debug!("Preparing rewards, previous scores: {:?}", actual_universal_scores);
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        self.send_to(player, EnvironmentMessage::RewardFragment(reward))?;
                                    }

                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);

                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }



                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }

                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_to_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er
                                    })?;
                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_to_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game (via Quit signal).", player);
                            self.notify_error(AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            #[cfg(feature = "log_error")]
                            log::error!("Failed trying to receive from {} with {err}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiteatrError::Communication{source: err});
                        }


                    }
                }
            }
        }
    }
}

impl<Env, S: Scheme, P> RoundRobinPenalisingUniversalEnvironment<S, P> for Env
where Env: CommunicatingEndpointEnvironment<S, CommunicationError=CommunicationError<S>>
 + ScoreEnvironment<S>
 + EnvironmentWithAgents<S>
 + BroadcastingEndpointEnvironment<S>, S: Scheme,
P: Fn(&<Self as StatefulEnvironment<S>>::State, &S::AgentId) -> S::UniversalReward{
    fn run_round_robin_with_rewards_penalise_truncating(&mut self, penalty_fn: P, truncate_steps: Option<usize>)
        -> Result<(), AmfiteatrError<S>>
    {
        let mut current_step = 0;
        let mut actual_universal_scores: HashMap<S::AgentId, S::UniversalReward> = self.players().into_iter()
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
        #[cfg(feature = "log_info")]
        log::info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                match self.nonblocking_receive_from(&player){
                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            #[cfg(feature = "log_info")]
                            log::info!("Player {} performs action: {:#}", &player, &action);
                            let potential_penalty = penalty_fn(self.state(), &player);
                            match self.process_action_penalise_illegal(&player, &action, potential_penalty){
                                Ok(updates) => {
                                    current_step += 1;
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .inspect_err(|e|{
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                            })?;
                                    }

                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        self.send_to(player, EnvironmentMessage::RewardFragment(reward))?;
                                    }
                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Player {player:} performed illegal action: {action:}, detailed error: {e}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_to(&player, EnvironmentMessage::RewardFragment(penalty_fn(self.state(), &player)));
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        let _ = self.send_to(player, EnvironmentMessage::RewardFragment(reward));
                                    }
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    self.set_game_violator(Some(player));
                                    return Err(e);
                                }
                            }

                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }

                            if let Some(truncation_limit) = truncate_steps
                                && current_step >= truncation_limit{
                                    #[cfg(feature = "log_info")]
                                    log::info!("Game reached truncation boundary");
                                    self.send_to_all(EnvironmentMessage::GameTruncated)?;
                                    return Ok(());
                                }

                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er
                                    })?;
                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} informed about error: {}", player, &e);
                            let _r = self.notify_error(e.clone());
                            let _r = self.send_to_all(EnvironmentMessage::GameFinished);
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            #[cfg(feature = "log_error")]
                            log::error!("Player {} exited game (via Quit signal).", player);
                            self.notify_error(AmfiteatrError::Protocol{source: PlayerExited(player.clone())})?;
                            return Err(AmfiteatrError::Protocol{source: PlayerExited(player)})
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            #[cfg(feature = "log_error")]
                            log::error!("Failed trying to receive from {} with {err}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiteatrError::Communication{source: err});
                        }


                    }
                }
            }
        }
    }
}

