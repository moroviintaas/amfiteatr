
use crate::agent::{
    CommunicatingAgent,
    ActingAgent,
    StatefulAgent,
    PolicyAgent,
    RewardedAgent,
    IdAgent,
    InformationSet};
use crate::error::{CommunicationError, AmfiteatrError};
use crate::error::ProtocolError::{NoPossibleAction, ReceivedKill};
use crate::error::AmfiteatrError::Protocol;
use crate::domain::{AgentMessage, EnvironmentMessage, DomainParameters};



/// Trait for agents that perform their interactions with environment automatically,
/// without waiting for interrupting interaction from anyone but environment.
pub trait AutomaticAgentRewarded<DP: DomainParameters>:RewardedAgent<DP> + IdAgent<DP>{
    /// Runs agent beginning in it's current state (information set)
    /// and returns when game is finished.
    fn run_rewarded(&mut self) -> Result<(), AmfiteatrError<DP>>;
}



impl<Agnt, DP> AutomaticAgentRewarded<DP> for Agnt
where Agnt: StatefulAgent<DP> + ActingAgent<DP>
    + CommunicatingAgent<DP, CommunicationError=CommunicationError<DP>>
    + PolicyAgent<DP>
    + RewardedAgent<DP>,
    //+ SelfEvaluatingAgent<DP>,
      DP: DomainParameters,
    <Agnt as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>
{
    fn run_rewarded(&mut self) -> Result<(), AmfiteatrError<DP>>
    {
        #[cfg(feature = "log_info")]
        log::info!("Agent {} starts", self.id());
        //let mut current_score = Spec::UniversalReward::default();
        loop{
            match self.recv(){
                Ok(message) => match message{
                    EnvironmentMessage::YourMove => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received 'YourMove' signal.", self.id());
                        //debug!("Agent's {:?} possible actions: {}]", self.id(), self.info_set().available_actions().into_iter()
                        //    .fold(String::from("["), |a, b| a + &format!("{b:#}") + ", ").trim_end());
                        match self.take_action(){
                            Ok(act_opt) => match act_opt{
                                None => {
                                    #[cfg(feature = "log_error")]
                                    log::error!("Agent {} has no possible action", self.id());
                                    self.send(AgentMessage::NotifyError(NoPossibleAction(self.id().clone()).into()))?;
                                }

                                Some(a) => {
                                    #[cfg(feature = "log_debug")]
                                    log::debug!("Agent {} selects action {:#}", self.id(), &a);
                                    self.send(AgentMessage::TakeAction(a))?;
                                }
                            }
                            Err(e) => {
                                self.and_send_error(e);
                            }
                        }
                    }
                    EnvironmentMessage::MoveRefused => {
                        let _ = self.react_refused_action().map_err(|e|self.and_send_error(e));
                    }
                    EnvironmentMessage::GameFinished => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {} received information that game is finished.", self.id());
                        self.finalize().map_err(|e| self.and_send_error(e))?;
                        return Ok(())

                    }
                    EnvironmentMessage::GameFinishedWithIllegalAction(_id)=> {
                        #[cfg(feature = "log_warn")]
                        log::warn!("Agent {} received information that game is finished with agent {_id:} performing illegal action.", self.id());
                        self.finalize().map_err(|e| self.and_send_error(e))?;
                        return Ok(())

                    }
                    EnvironmentMessage::Kill => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {:?} received kill signal.", self.id());
                        return Err(Protocol{source: ReceivedKill(self.id().clone())})
                    }
                    EnvironmentMessage::UpdateState(su) => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received state update {:?}", self.id(), &su);
                        match self.update(su){
                            Ok(_) => {
                                #[cfg(feature = "log_debug")]
                                log::debug!("Agent {:?}: successful state update", self.id());
                            }
                            Err(err) => {
                                #[cfg(feature = "log_error")]
                                log::error!("Agent {:?} error on updating state: {}", self.id(), &err);
                                self.send(AgentMessage::NotifyError(AmfiteatrError::Game{source: err.clone()}))?;
                                return Err(AmfiteatrError::Game{source: err});
                            }
                        }
                    }
                    EnvironmentMessage::ActionNotify(_a) => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received information that agent {} took action {:#}", self.id(), _a.agent(), _a.action());
                    }
                    EnvironmentMessage::ErrorNotify(_e) => {
                        #[cfg(feature = "log_error")]
                        log::error!("Agent {} received error notification {}", self.id(), &_e)
                    }
                    EnvironmentMessage::RewardFragment(r) =>{
                        //current_score = current_score + r;
                        //self.set_current_universal_reward(current_score.clone());
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received reward fragment: {:?}", self.id(), r);
                        self.current_universal_reward_add(&r);
                    }
                }
                Err(e) => return Err(e.into())
            }
        }
    }
}
