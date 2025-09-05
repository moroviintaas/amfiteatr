use std::fmt::Display;
use std::io;
use std::io::{BufRead};
use crate::agent::{
    ActingAgent,
    CommunicatingAgent,
    IdAgent,
    InformationSet,
    PolicyAgent,
    ProcedureAgent,
    ReseedAgent,
    RewardedAgent,
    StatefulAgent
};
use crate::agent::manual_control::{AssistingPolicy, TurnCommand};
use crate::scheme::{AgentMessage, Scheme};
use crate::error::{AmfiteatrError};
use crate::util::{StrParsed};


/// Trait for agents that uses human policy.
pub trait CliAgent<S: Scheme>{

    /// Selecting action interactively. Basically this function should somehow prompt player
    /// and parse his input into action.
    fn interactive_action_select(&mut self) -> Result<S::ActionType, AmfiteatrError<S>>;

    /// Runs agent's side protocol with interactive action selection.
    fn run_interactive(&mut self) -> Result<(), AmfiteatrError<S>>;
}

impl<A, S> CliAgent<S> for A
where
    A: StatefulAgent<S> + ActingAgent<S>
    + CommunicatingAgent<S>
    + PolicyAgent<S>
    + RewardedAgent<S>,
    S: Scheme,
    <A as StatefulAgent<S>>::InfoSetType: InformationSet<S> + Display,
    <Self as PolicyAgent<S>>::Policy: AssistingPolicy<S>,
    <<Self as PolicyAgent<S>>::Policy as AssistingPolicy<S>>::Question: StrParsed,
    //TopCommand<S, <<Self as PolicyAgent<S>>::Policy>>: for<'a> NomParsed<'str>,
    S::ActionType: for<'a> StrParsed{

    fn interactive_action_select(&mut self) -> Result<S::ActionType, AmfiteatrError<S>>{
        let mut buffer = String::new();
        let stdin = io::stdin();
        #[cfg(feature = "log_debug")]
        log::debug!("Agent {} received 'YourMove' signal.", self.id().clone());
        loop{
            let mut handle = stdin.lock();

            println!("Agent: {} >", self.id());
            buffer.clear();

            handle.read_line(&mut buffer).map_err(|e|AmfiteatrError::IO {explanation: format!("{e}")})?;

            match TurnCommand::<S, <Self as PolicyAgent<S>>::Policy>::parse_from_str(&buffer[..]){
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
    fn run_interactive(&mut self) -> Result<(), AmfiteatrError<S>> {


        ProcedureAgent::run_protocol(self, |agent| {
            agent.interactive_action_select()
        })




    }
}

/// Dummy trait to link [`CliAgent`] and [`ReseedAgent`](ReseedAgent)
pub trait MultiEpisodeCliAgent<S: Scheme, Seed>: ReseedAgent<S, Seed> + CliAgent<S>{

}
impl<S: Scheme, Seed, A: ReseedAgent<S, Seed> + CliAgent<S>> MultiEpisodeCliAgent<S, Seed> for A{

}

