use std::collections::HashMap;
use log::{debug, error, info, warn};
use crate::env::{BroadcastingEndpointEnvironment, CommunicatingEndpointEnvironment, EnvironmentStateSequential, EnvironmentWithAgents, ScoreEnvironment, StatefulEnvironment};
use crate::error::{CommunicationError, AmfiError};
use crate::error::ProtocolError::PlayerExited;
use crate::domain::{AgentMessage, EnvironmentMessage, DomainParameters, Reward};
use crate::domain::EnvironmentMessage::ErrorNotify;
use crate::error::AmfiError::GameA;


/// Interface for environment using round robin strategy for listening to agents' messages.
pub trait RoundRobinEnvironment<DP: DomainParameters>{
    fn run_round_robin(&mut self) -> Result<(), AmfiError<DP>>;
}
/// Similar interface to [`RoundRobinEnvironment`], but it must ensure agents receive rewards.
pub trait RoundRobinUniversalEnvironment<DP: DomainParameters> : RoundRobinEnvironment<DP>{
    fn run_round_robin_with_rewards(&mut self) -> Result<(), AmfiError<DP>>;
}
/// Similar interface to [`RoundRobinEnvironment`] and [`RoundRobinUniversalEnvironment`],
/// in addition to rewards based on current game state, penalties for illegal actions are sent.
/// This is __experimental__ interface.
pub trait RoundRobinPenalisingUniversalEnvironment<DP: DomainParameters>: RoundRobinUniversalEnvironment<DP>{
    fn run_round_robin_with_rewards_penalise(&mut self, penalty: DP::UniversalReward) -> Result<(), AmfiError<DP>>;
}



pub(crate) trait EnvironmentRRInternal<DP: DomainParameters>{
    fn notify_error(&mut self, error: AmfiError<DP>) -> Result<(), CommunicationError<DP>>;
    fn send_message(&mut self, agent: &DP::AgentId, message: EnvironmentMessage<DP>) -> Result<(), CommunicationError<DP>>;
    fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiError<DP>>;

    //fn broadcast_message(&mut self ,message: EnvMessage<Spec>) -> Result<(), CommError>;
}

impl<'a, Env, DP: DomainParameters + 'a> EnvironmentRRInternal<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + StatefulEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>,

DP: DomainParameters
{
    fn notify_error(&mut self, error: AmfiError<DP>) -> Result<(), CommunicationError<DP>> {
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

    fn process_action_and_inform(&mut self, player: DP::AgentId, action: &DP::ActionType) -> Result<(), AmfiError<DP>> {
        match self.process_action(&player, action){
            Ok(iter) => {
                //let mut n=0;
                for (ag, update) in iter{
                    //debug!("{}", n);
                    //n+= 1;
                    //self.send_message(&ag, EnvMessage::ActionNotify(AgentActionPair::new(player.clone(), action.clone())))?;
                    self.send_message(&ag, EnvironmentMessage::UpdateState(update))?;
                }
                Ok(())
            }
            Err(e) => {Err(AmfiError::Game(e))}
        }
    }


}


impl<'a, Env, DP: DomainParameters + 'a> RoundRobinEnvironment<DP> for Env
where Env: CommunicatingEndpointEnvironment<DP, CommunicationError=CommunicationError<DP>>
 + StatefulEnvironment<DP> + 'a
 + EnvironmentWithAgents<DP>
 + BroadcastingEndpointEnvironment<DP>, DP: DomainParameters {
    fn run_round_robin(&mut self) -> Result<(), AmfiError<DP>> {
        let first_player = match self.current_player(){
            None => {
                warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                match self.nonblocking_receive_from(&player){
                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            info!("Player {} performs action: {:#}", &player, &action);

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
                                    error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(GameA(e, player));
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
                                info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            error!("Player {} informed about error: {}", player, &e);
                            self.notify_error(e.clone())?;
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            error!("Player {} exited game.", player);
                            self.notify_error(AmfiError::Protocol(PlayerExited(player.clone())))?;
                            return Err(AmfiError::Protocol(PlayerExited(player)))
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            error!("Failed trying to receive from {}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiError::Communication(err));
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
    fn run_round_robin_with_rewards(&mut self) -> Result<(), AmfiError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players().into_iter()
            .map(|id|{
                (id, DP::UniversalReward::neutral())
            }).collect();
        let first_player = match self.current_player(){
            None => {
                warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                match self.nonblocking_receive_from(&player){
                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            info!("Player {} performs action: {:#}", &player, &action);

                            match self.process_action(&player, &action){
                                Ok(updates) => {
                                    for (ag, update) in updates{
                                        self.send_message(&ag, EnvironmentMessage::UpdateState(update))
                                            .map_err(|e|{
                                                let _ = self.send_to_all(ErrorNotify(e.clone().into()));
                                                e
                                            })?;
                                    }
                                    debug!("Preparing rewards, previous scores: {:?}", actual_universal_scores);
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        self.send_to(player, EnvironmentMessage::RewardFragment(reward))?;
                                    }
                                }
                                Err(e) => {
                                    error!("Action was refused or caused error in updating state: {e:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);

                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(GameA(e, player));
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
                                info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            error!("Player {} informed about error: {}", player, &e);
                            self.notify_error(e.clone())?;
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            error!("Player {} exited game.", player);
                            self.notify_error(AmfiError::Protocol(PlayerExited(player.clone())))?;
                            return Err(AmfiError::Protocol(PlayerExited(player)))
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            error!("Failed trying to receive from {}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiError::Communication(err));
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
    fn run_round_robin_with_rewards_penalise(&mut self, penalty: DP::UniversalReward) -> Result<(), AmfiError<DP>> {
        let mut actual_universal_scores: HashMap<DP::AgentId, DP::UniversalReward> = self.players().into_iter()
            .map(|id|{
                (id, DP::UniversalReward::neutral())
            }).collect();
        let first_player = match self.current_player(){
            None => {
                warn!("No first player, stopping environment.");
                return Ok(())
            }
            Some(n) => n
        };
        info!("Sending YourMove signal to first agent: {:?}", &first_player);
        self.send_to(&first_player, EnvironmentMessage::YourMove).map_err(|e|e.specify_id(first_player))?;
        loop{
            for player in self.players(){
                match self.nonblocking_receive_from(&player){
                    Ok(Some(agent_message)) => match agent_message{
                        AgentMessage::TakeAction(action) => {
                            info!("Player {} performs action: {:#}", &player, &action);
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
                                    error!("Player {player:} performed illegal action: {action:}");
                                    let _ = self.send_to(&player, EnvironmentMessage::MoveRefused);
                                    let _ = self.send_to(&player, EnvironmentMessage::RewardFragment(penalty));
                                    for (player, score) in actual_universal_scores.iter_mut(){

                                        let reward = self.actual_score_of_player(player) - score.clone();
                                        *score = self.actual_score_of_player(player);
                                        let _ = self.send_to(player, EnvironmentMessage::RewardFragment(reward));
                                    }
                                    let _ = self.send_to_all(EnvironmentMessage::GameFinishedWithIllegalAction(player.clone()));
                                    return Err(GameA(e, player));
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
                                info!("Game reached finished state");
                                self.send_to_all(EnvironmentMessage::GameFinished)?;
                                return Ok(());

                            }


                        }
                        AgentMessage::NotifyError(e) => {
                            error!("Player {} informed about error: {}", player, &e);
                            self.notify_error(e.clone())?;
                            return Err(e);
                        }
                        AgentMessage::Quit => {
                            error!("Player {} exited game.", player);
                            self.notify_error(AmfiError::Protocol(PlayerExited(player.clone())))?;
                            return Err(AmfiError::Protocol(PlayerExited(player)))
                        }
                    },
                    Ok(None) => {},
                    Err(e) => match e{

                        CommunicationError::RecvEmptyBufferError(_) | CommunicationError::RecvPeerDisconnectedError(_) |
                        CommunicationError::RecvEmptyBufferErrorUnspecified | CommunicationError::RecvPeerDisconnectedErrorUnspecified => {
                            //debug!("Empty channel");
                        },
                        err => {
                            error!("Failed trying to receive from {}", player);
                            self.send_to_all(EnvironmentMessage::ErrorNotify(err.clone().into()))?;
                            return Err(AmfiError::Communication(err));
                        }


                    }
                }
            }
        }
    }
}

