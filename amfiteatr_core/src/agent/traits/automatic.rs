
use crate::agent::{
    CommunicatingAgent,
    ActingAgent,
    StatefulAgent,
    PolicyAgent,
    RewardedAgent,
    EvaluatedInformationSet,
    IdAgent,
    InformationSet};
use crate::error::{CommunicationError, AmfiteatrError};
use crate::error::ProtocolError::{NoPossibleAction, ReceivedKill};
use crate::error::AmfiteatrError::Protocol;
use crate::domain::{AgentMessage, EnvironmentMessage, DomainParameters};

/// Trait for agents that perform their interactions with environment automatically,
/// without waiting for interrupting interaction from anyone but environment.
/// This trait describes behaviour of agent that is not necessary interested in
/// collecting rewards from environment.
/// Implementations are perfectly fine to skip messages about rewards coming
/// from environment. As trait suited for running game regarding collected rewards
/// refer to [`AutomaticAgentRewarded`](AutomaticAgentRewarded)
pub trait AutomaticAgent<DP: DomainParameters>: IdAgent<DP>{
    /// Runs agent beginning in it's current state (information set)
    /// and returns when game is finished.
    /// > __Note__ It is not specified how agent should react when encountering error.
    /// > One conception is to inform environment about error, which then should broadcast
    /// > error message to every agent and end game.
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>>;
}

/// Trait for agents that perform their interactions with environment automatically,
/// without waiting for interrupting interaction from anyone but environment.
/// Difference between [`AutomaticAgent`](AutomaticAgent) is that
/// this method should collect rewards and somehow store rewards sent by environment.
pub trait AutomaticAgentRewarded<DP: DomainParameters>: AutomaticAgent<DP> + RewardedAgent<DP>{
    /// Runs agent beginning in it's current state (information set)
    /// and returns when game is finished.
    fn run_rewarded(&mut self) -> Result<(), AmfiteatrError<DP>>;
}

/*
/// Combination of traits [`AutomaticAgentRewarded`](AutomaticAgentRewarded),
/// [`SelfEvaluatingAgent`](SelfEvaluatingAgent).
pub trait AutomaticAgentRE<DP: DomainParameters>: AutomaticAgentRewarded<DP> + SelfEvaluatingAgent<DP>{

}



impl<DP: DomainParameters, T: AutomaticAgentRewarded<DP> + SelfEvaluatingAgent<DP>> AutomaticAgentRE<DP> for T{}
*/
/*
/// [`AutomaticAgent`](AutomaticAgent) that is also a [`TracingAgent`](crate::agent::TracingAgent) using
/// .
pub trait AutomaticAgentWithStdTrace<DP: DomainParameters, IS: EvaluatedInformationSet<DP>>:
    AutomaticAgentRewarded<DP>
    + SelfEvaluatingAgent<DP, Assessment = IS::RewardType>
    + TracingAgent<DP, IS>{}

impl<
    DP: DomainParameters,
    IS: EvaluatedInformationSet<DP>,
    T: AutomaticAgentRewarded<DP>
        + TracingAgent<DP, IS>
        + SelfEvaluatingAgent<DP, Assessment = IS::RewardType>> AutomaticAgentWithStdTrace<DP, IS> for T{

}
*/

/// Generic implementation of AutomaticAgent - probably will be done via macro
/// in the future to avoid conflicts with custom implementations.
impl<A, DP> AutomaticAgent<DP> for A
where A: StatefulAgent<DP> + ActingAgent<DP>
    + CommunicatingAgent<DP, CommunicationError=CommunicationError<DP>>
    + PolicyAgent<DP>,
    //+ SelfEvaluatingAgent<DP>,
      DP: DomainParameters,
      <A as StatefulAgent<DP>>::InfoSetType: InformationSet<DP>
{
    fn run(&mut self) -> Result<(), AmfiteatrError<DP>> {
        #[cfg(feature = "log_info")]
        log::info!("Agent {} starts", self.id());
        //let mut current_score = Spec::UniversalReward::default();
        loop{
            match self.recv(){
                Ok(message) => match message{
                    EnvironmentMessage::YourMove => {
                        #[cfg(feature = "log_trace")]
                        log::trace!("Agent {} received 'YourMove' signal.", self.id());
                        //current_score = Default::default();

                        //debug!("Agent's {:?} possible actions: {:?}", self.id(), Vec::from_iter(self.state().available_actions().into_iter()));
                        //trace!("Agent's {} possible actions: {}]", self.id(), self.info_set().available_actions().into_iter()
                        //    .fold(String::from("["), |a, b| a + &format!("{b:#}") + ", ").trim_end());
                        //match self.policy_select_action(){
                        match self.take_action(){
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
                    }
                    EnvironmentMessage::MoveRefused => {
                        self.react_refused_action()
                        //self.add_explicit_assessment(&self.penalty_for_illegal_action())
                            /*&<Self as InternalRewardedAgent<DP>>::InternalReward
                            ::penalty_for_illegal())

                             */
                    }
                    EnvironmentMessage::GameFinished => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {} received information that game is finished.", self.id());
                        self.finalize();
                        return Ok(())

                    }
                    EnvironmentMessage::GameFinishedWithIllegalAction(_id) => {
                        #[cfg(feature = "log_warn")]
                        log::warn!("Agent {} received information that game is finished with agent {_id:} performing illegal action.", self.id());
                        self.finalize();
                        return Ok(())

                    }
                    EnvironmentMessage::Kill => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {:?} received kill signal.", self.id());
                        return Err(Protocol(ReceivedKill(self.id().clone())))
                    }
                    EnvironmentMessage::UpdateState(su) => {
                        #[cfg(feature = "log_trace")]
                        log::trace!("Agent {} received state update {:?}", self.id(), &su);
                        match self.update(su){
                            Ok(_) => {
                                #[cfg(feature = "log_trace")]
                                log::trace!("Agent {:?}: successful state update", self.id());
                            }
                            Err(err) => {
                                #[cfg(feature = "log_error")]
                                log::error!("Agent {:?} error on updating state: {}", self.id(), &err);
                                self.send(AgentMessage::NotifyError(AmfiteatrError::GameA(err.clone(), self.id().clone())))?;
                                return Err(AmfiteatrError::GameA(err.clone(), self.id().clone()));
                            }
                        }
                    }
                    EnvironmentMessage::ActionNotify(_a) => {
                        #[cfg(feature = "log_trace")]
                        log::trace!("Agent {} received information that agent {} took action {:#}", self.id(), _a.agent(), _a.action());
                    }
                    EnvironmentMessage::ErrorNotify(_e) => {
                        #[cfg(feature = "log_error")]
                        log::error!("Agent {} received error notification {}", self.id(), &_e)
                    }
                    EnvironmentMessage::RewardFragment(_r) =>{
                    }
                }
                Err(e) => return Err(e.into())
            }
        }
    }
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
                            None => {
                                #[cfg(feature = "log_error")]
                                log::error!("Agent {} has no possible action", self.id());
                                self.send(AgentMessage::NotifyError(NoPossibleAction(self.id().clone()).into()))?;
                            }

                            Some(a) => {
                                #[cfg(feature = "log_info")]
                                log::info!("Agent {} selects action {:#}", self.id(), &a);
                                self.send(AgentMessage::TakeAction(a))?;
                            }
                        }
                    }
                    EnvironmentMessage::MoveRefused => {
                        self.react_refused_action()
                        //self.add_explicit_assessment(&self.penalty_for_illegal_action())
                        /*(
                            &<<Self as StatefulAgent<DP>>::InfoSetType as ScoringInformationSet<DP>>
                            ::penalty_for_illegal())

                         */
                    }
                    EnvironmentMessage::GameFinished => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {} received information that game is finished.", self.id());
                        self.finalize();
                        return Ok(())

                    }
                    EnvironmentMessage::GameFinishedWithIllegalAction(_id)=> {
                        #[cfg(feature = "log_warn")]
                        log::warn!("Agent {} received information that game is finished with agent {_id:} performing illegal action.", self.id());
                        self.finalize();
                        return Ok(())

                    }
                    EnvironmentMessage::Kill => {
                        #[cfg(feature = "log_info")]
                        log::info!("Agent {:?} received kill signal.", self.id());
                        return Err(Protocol(ReceivedKill(self.id().clone())))
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
                                self.send(AgentMessage::NotifyError(AmfiteatrError::Game(err.clone())))?;
                                return Err(AmfiteatrError::Game(err));
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
/*
impl<DP: DomainParameters, A: AutomaticAgent<DP>> AutomaticAgent<DP> for Mutex<A>{
    fn run(&mut self) -> Result<(), AmfiError<DP>> {
        let mut guard = self.lock().or_else(|_|Err(WorldError::<DP>::AgentMutexLock))?;
        let id = guard.id().clone();
        guard.run().map_err(|e|{
            error!("Agent {id:} encountered error: {e:}");
            e
        })
    }
}
*/
/*
impl<DP: DomainParameters, A: AutomaticAgent<DP>> AutomaticAgent<DP> for Box<A>{
    fn run(&mut self) -> Result<(), AmfiError<DP>> {
        self.run()
    }
}

 */