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
use crate::domain::{AgentMessage, DomainParameters};
use crate::error::{AmfiteatrError};
use crate::util::{StrParsed};


/// Trait for agents that uses human policy.
pub trait CliAgent<DP: DomainParameters>{

    /// Selecting action interactively. Basically this function should somehow prompt player
    /// and parse his input into action.
    fn interactive_action_select(&mut self) -> Result<DP::ActionType, AmfiteatrError<DP>>;

    /// Runs agent's side protocol with interactive action selection.
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
    <<Self as PolicyAgent<DP>>::Policy as AssistingPolicy<DP>>::Question: StrParsed,
    //TopCommand<DP, <<Self as PolicyAgent<DP>>::Policy>>: for<'a> NomParsed<'str>,
    DP::ActionType: for<'a> StrParsed{

    fn interactive_action_select(&mut self) -> Result<DP::ActionType, AmfiteatrError<DP>>{
        let mut buffer = String::new();
        let stdin = io::stdin();
        #[cfg(feature = "log_debug")]
        log::debug!("Agent {} received 'YourMove' signal.", self.id().clone());
        loop{
            let mut handle = stdin.lock();

            println!("Agent: {} >", self.id());
            buffer.clear();

            handle.read_line(&mut buffer).map_err(|e|AmfiteatrError::IO {explanation: format!("{e}")})?;

            match TurnCommand::<DP, <Self as PolicyAgent<DP>>::Policy>::parse_from_str(&buffer[..]){
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
    fn run_interactive(&mut self) -> Result<(), AmfiteatrError<DP>> {


        ProcedureAgent::run_protocol(self, |agent| {
            agent.interactive_action_select()
        })




    }
}

/// Dummy trait to link [`CliAgent`] and [`ReseedAgent`](ReseedAgent)
pub trait MultiEpisodeCliAgent<DP: DomainParameters, Seed>: ReseedAgent<DP, Seed> + CliAgent<DP>{

}
impl<DP: DomainParameters, Seed, A: ReseedAgent<DP, Seed> + CliAgent<DP>> MultiEpisodeCliAgent<DP, Seed> for A{

}

