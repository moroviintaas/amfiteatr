
use crate::agent::{
    CommunicatingAgent,
    ActingAgent,
    StatefulAgent,
    PolicyAgent,
    RewardedAgent,
    IdAgent,
};
use crate::error::{AmfiteatrError};
use crate::error::ProtocolError::ReceivedKill;
use crate::error::AmfiteatrError::Protocol;
use crate::scheme::{AgentMessage, EnvironmentMessage, Scheme};

/// Helping trait for almost automatic agent.
/// It is used as frame implementation of automatic agent.
pub trait ProcedureAgent<S: Scheme>: RewardedAgent<S> + IdAgent<S>
+ CommunicatingAgent<S> + ActingAgent<S> + StatefulAgent<S>{

    /// Runs automatic agent provided with function selecting action.
    /// This is meant to be backend implementation for [`AutomaticAgent::run`] and [`CliAgent`](crate::agent::manual_control::CliAgent).
    fn run_protocol<
        P: Fn(&mut Self) -> Result<S::ActionType, AmfiteatrError<S>>,
    >(&mut self, action_selector: P) -> Result<(), AmfiteatrError<S>>{
        #[cfg(feature = "log_info")]
        log::info!("Agent {} starts", self.id());

        loop{
            match self.recv(){
                Ok(message) => match message{
                    EnvironmentMessage::YourMove => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received 'YourMove' signal.", self.id());
                        match action_selector(self){
                            Ok(act) => {
                                #[cfg(feature = "log_debug")]
                                log::debug!("Agent {} selects action {:#}", self.id(), &act);
                                self.send(AgentMessage::TakeAction(act))?;
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
                    EnvironmentMessage::GameTruncated => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {} received information that game is truncated.", self.id());
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

impl<A, S: Scheme> ProcedureAgent<S> for A
where A: RewardedAgent<S> + IdAgent<S>
+ CommunicatingAgent<S> + ActingAgent<S> + StatefulAgent<S>
{

}

/// Trait for agents that perform their interactions with environment automatically,
/// without waiting for interrupting interaction from anyone but environment.
pub trait AutomaticAgent<S: Scheme>{



    /// Runs agent beginning in its current state (information set)
    /// and returns when game is finished.
    fn run(&mut self) -> Result<(), AmfiteatrError<S>>;
}

impl<A, S: Scheme> AutomaticAgent<S> for A
where A: ProcedureAgent<S> + PolicyAgent<S>{
    fn run(&mut self) -> Result<(), AmfiteatrError<S>>{
        self.run_protocol(|s| s.select_action())
    }

}

