
use crate::agent::{
    CommunicatingAgent,
    ActingAgent,
    StatefulAgent,
    PolicyAgent,
    RewardedAgent,
    IdAgent,
};
use crate::error::{AmfiteatrError};
use crate::error::ProtocolError::{NoPossibleAction, ReceivedKill};
use crate::error::AmfiteatrError::Protocol;
use crate::domain::{AgentMessage, EnvironmentMessage, DomainParameters};

/// Helping trait for almost automatic agent.
/// It is used as frame implementation of automatic agent.
pub trait ProcedureAgent<DP: DomainParameters>: RewardedAgent<DP> + IdAgent<DP>
+ CommunicatingAgent<DP> + ActingAgent<DP> + StatefulAgent<DP>{

    /// Runs automatic agent provided with function selecting action.
    /// This is meant to be backend implementation for [`AutomaticAgent::run`] and [`CliAgent`](crate::agent::manual_control::CliAgent).
    fn run_protocol<
        P: Fn(&mut Self) -> Result<Option<DP::ActionType>, AmfiteatrError<DP>>,
    >(&mut self, action_selector: P) -> Result<(), AmfiteatrError<DP>>{
        #[cfg(feature = "log_info")]
        log::info!("Agent {} starts", self.id());
        loop{
            match self.recv(){
                Ok(message) => match message{
                    EnvironmentMessage::YourMove => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received 'YourMove' signal.", self.id());
                        match action_selector(self){
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

impl<A, DP: DomainParameters> ProcedureAgent<DP> for A
where A: RewardedAgent<DP> + IdAgent<DP>
+ CommunicatingAgent<DP> + ActingAgent<DP> + StatefulAgent<DP>
{

}

/// Trait for agents that perform their interactions with environment automatically,
/// without waiting for interrupting interaction from anyone but environment.
pub trait AutomaticAgent<DP: DomainParameters>{



    /// Runs agent beginning in its current state (information set)
    /// and returns when game is finished.
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>;
}

impl<A, DP: DomainParameters> AutomaticAgent<DP> for A
where A: ProcedureAgent<DP> + PolicyAgent<DP>{
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>{
        self.run_protocol(|s| s.select_action())
    }

}


/*
impl<Agnt, DP> AutomaticAgent<DP> for Agnt
where Agnt: StatefulAgent<DP> + ActingAgent<DP>
    + CommunicatingAgent<DP>
    + PolicyAgent<DP>
    + RewardedAgent<DP>,
    //+ SelfEvaluatingAgent<DP>,
      DP: DomainParameters,
    <Agnt as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>
{


    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>
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
                        match self.select_action(){
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


 */