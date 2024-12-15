use std::collections::HashMap;
use crate::env::{BroadcastingEndpointEnvironment, CommunicatingEndpointEnvironment, SequentialGameState, EnvironmentWithAgents, ScoreEnvironment, StatefulEnvironment};
use crate::error::{CommunicationError, AmfiteatrError};
use crate::error::ProtocolError::PlayerExited;
use crate::domain::{AgentMessage, EnvironmentMessage, DomainParameters, Reward};
use crate::domain::EnvironmentMessage::ErrorNotify;


/// Interface for environment using round robin strategy for listening to agents' messages.
pub trait RoundRobinEnvironment<DP: DomainParameters>{
    fn run_round_robin(&mut self) -> Result<(), AmfiteatrError<DP>>;
}
/// Similar interface to [`RoundRobinEnvironment`], but it must ensure agents receive rewards.
pub trait RoundRobinUniversalEnvironment<DP: DomainParameters> : RoundRobinEnvironment<DP>{
    fn run_round_robin_with_rewards(&mut self) -> Result<(), AmfiteatrError<DP>>;
}
/// Similar interface to [`RoundRobinEnvironment`] and [`RoundRobinUniversalEnvironment`],
/// in addition to rewards based on current game state, penalties for illegal actions are sent.
/// This is __experimental__ interface.
pub trait RoundRobinPenalisingUniversalEnvironment<DP: DomainParameters>: RoundRobinUniversalEnvironment<DP>{
    fn run_round_robin_with_rewards_penalise(&mut self, penalty: DP::UniversalReward) -> Result<(), AmfiteatrError<DP>>;
}



pub(crate) trait EnvironmentRRInternal<DP: DomainParameters>{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>>;
    fn send_message(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>;


    //fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiteatrError<DP>>;

}

impl<'a, Env, DP: DomainParameters + 'a> EnvironmentRRInternal<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + StatefulEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>,

DP: DomainParameters
{
    fn notify_error(&mut self, error: AmfiteatrError<DP>) -> Result<(), CommunicationError<DP>> {
        self.send_to_all(ErrorNotify(error))
    }

    fn send_message(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>{
        self.send_to(agent, message)
            .map_err(|e| {
                self.notify_error(e.clone().into())
                    .unwrap_or_else(|_| panic!("Failed broadcasting error message {}", &e));
                e
            })
    }

    // fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiteatrError<DP>> {
    //     match self.process_action(&player, action){
    //         Ok(iter) => {
    //             //let mut n=0;
    //             for (ag, update) in iter{
    //                 //debug!("{}", n);
    //                 //n+= 1;
    //                 //self.send_message(&ag, EnvMessage::ActionNotify(AgentActionPair::new(player.clone(), action.clone())))?;
    //                 self.send_message(&ag, EnvironmentMessage::UpdateState(update))?;
    //             }
    //             Ok(())
    //         }
    //         Err(e) => {Err(e)}
    //     }
    // }


}


impl<'a, Env, DP: DomainParameters + 'a> RoundRobinEnvironment<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + StatefulEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>, DP: DomainParameters {
    fn run_round_robin(&mut self) -> Result<(), AmfiteatrError<DP>> {
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
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .map_err(|e| {
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                                e
                                            })?;

                                    }
                                }
                                Err(e) => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(e);
                                }
                            }
                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er

                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

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

impl<'a, Env, DP: DomainParameters + 'a> RoundRobinUniversalEnvironment<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + ScoreEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>, DP: DomainParameters {
    fn run_round_robin_with_rewards(&mut self) -> Result<(), AmfiteatrError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players().into_iter()
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
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .map_err(|e|{
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                                e
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
                                    return Err(e);
                                }
                            }


                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er
                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

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

impl<'a, Env, DP: DomainParameters + 'a> RoundRobinPenalisingUniversalEnvironment<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + ScoreEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>, DP: DomainParameters{
    fn run_round_robin_with_rewards_penalise(&mut self, penalty: DP::UniversalReward) -> Result<(), AmfiteatrError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players().into_iter()
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
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .map_err(|e|{
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                                e
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
                                    let _ = self.send_to(&player, EnvironmentMessage::RewardFragment(penalty));
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        let _ = self.send_to(player, EnvironmentMessage::RewardFragment(reward));
                                    }
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(e);
                                }
                            }
                            if let Some(next_player) = self.current_player(){
                                self.send_message(&next_player, EnvironmentMessage::YourMove)
                                    .map_err(|e| {
                                        let er = e.specify_id(next_player);
                                        let _ = self.send_to_all(ErrorNotify(er.clone().into()));
                                        er
                                    })?;
                            }
                            if self.state().is_finished(){
                                #[cfg(feature = "log_info")]
                                log::info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

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

