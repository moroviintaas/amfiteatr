use std::fmt::Display;
use std::io;
use std::io::{BufRead};
use nom::{IResult, Parser};
use crate::agent::{ActingAgent, AutomaticAgent, CommunicatingAgent, IdAgent, InformationSet, PolicyAgent, ProcedureAgent, RewardedAgent, StatefulAgent};
use crate::agent::manual_control::{AssistingPolicy, NomParsed, TurnCommand};
use crate::domain::{AgentMessage, DomainParameters, EnvironmentMessage};
use crate::error::{AmfiteatrError, CommunicationError, ProtocolError};
use crate::error::AmfiteatrError::Protocol;
use crate::error::ProtocolError::{NoPossibleAction, ReceivedKill};

pub trait CliAgent<DP: DomainParameters>{

    fn run_interactive(&mut self) -> Result<(), AmfiteatrError<DP>>;
}

impl<A, DP> CliAgent<DP> for A
where
    A: StatefulAgent<DP> + ActingAgent<DP>
    + CommunicatingAgent<DP>
    + PolicyAgent<DP>
    + RewardedAgent<DP>,
    DP: DomainParameters,
    <A as StatefulAgent<DP>>::InfoSetType: InformationSet<DP> + Display,
    <Self as PolicyAgent<DP>>::Policy: AssistingPolicy<DP>,
    <<Self as PolicyAgent<DP>>::Policy as AssistingPolicy<DP>>::Question: for<'a> NomParsed<&'a str>,
    //TopCommand<DP, <<Self as PolicyAgent<DP>>::Policy>>: for<'a> NomParsed<'str>,
    DP::ActionType: for<'a> NomParsed<&'a str>{
    fn run_interactive(&mut self) -> Result<(), AmfiteatrError<DP>> {


        /*ProcedureAgent::run_protocol(self, |agent|{
            let mut buffer = String::new();
            let stdin = io::stdin();
            #[cfg(feature = "log_debug")]
            log::debug!("Agent {} received 'YourMove' signal.", self.id());
            loop{
                let mut handle = stdin.lock();

                println!("Agent: {} >", self.id());
                buffer.clear();

                handle.read_line(&mut buffer).map_err(|e|AmfiteatrError::IO {explanation: format!("{e}")})?;

                match TurnCommand::<DP, <Self as PolicyAgent<DP>>::Policy>::nom_parse(&buffer[..]){
                    Ok((_rest, command)) => match command{
                        TurnCommand::Quit => {
                            return Err(ProtocolError::PlayerSelectedNoneAction(self.id().clone()).into())
                        }
                        TurnCommand::Play(_) => {}
                        TurnCommand::Show => {}
                        TurnCommand::AskPolicy(_) => {}
                    }
                    Err(e) => {
                        println!("Failed parsing input: {buffer} with error: {e:}" )
                    }
                }

            }
        })

         */

        let mut buffer = String::new();
        let stdin = io::stdin();
        #[cfg(feature = "log_info")]
        log::info!("Agent {} starts", self.id());
        //let mut current_score = Spec::UniversalReward::default();
        loop{
            match self.recv(){
                Ok(message) => match message{
                    EnvironmentMessage::YourMove => {
                        #[cfg(feature = "log_debug")]
                        log::debug!("Agent {} received 'YourMove' signal.", self.id());


                        loop{
                            let mut handle = stdin.lock();

                            println!("Agent: {} >", self.id());
                            buffer.clear();

                            handle.read_line(&mut buffer).map_err(|e|AmfiteatrError::IO {explanation: format!("{e}")})?;

                            match TurnCommand::<DP, <Self as PolicyAgent<DP>>::Policy>::nom_parse(&buffer[..]){
                                Ok((_rest, command)) => match command{
                                    TurnCommand::Quit => {
                                        self.send(AgentMessage::Quit).unwrap();
                                        //return Ok(())
                                    }
                                    TurnCommand::Play(n) => {
                                        self.send(AgentMessage::TakeAction(n)).unwrap();
                                    }
                                    TurnCommand::Show => {
                                        println!("{:#}", self.info_set());
                                    }
                                    TurnCommand::AskPolicy(p) => {
                                        self.policy().assist(p).unwrap();
                                    }
                                }
                                Err(e) => {
                                    println!("Failed parsing input: {buffer} with error: {e:}" )
                                }
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