use nom::branch::alt;
use nom::bytes::complete::{tag};
use nom::character::complete::space1;
use nom::IResult;
use nom::sequence::{pair};
use crate::agent::{Policy, PresentPossibleActions, RandomPolicy};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;
use crate::util::{StrParsed};
use nom::Parser;

/// Experimental policy trait that is meant to be used as agent interactive policies.
/// Planned use case is as follows:
/// Agent (probably [`CliAgent`](crate::agent::manual_control::CliAgent) executes game protocol, but asks player what action is to be chosen.
/// Action can be invoked by player inputing some command, however player may prompt for some information
/// about information set or ask for hint (e.g some Q-table).
/// Example question implementation is [`TurnCommand`].
/// *Implementation of human controlled playing interface is not a priority now, however this is current idea of how it would look like.*
pub trait AssistingPolicy<DP: Scheme>: Policy<DP>{

    type Question;

    fn assist(&self, question: Self::Question) -> Result<String, AmfiteatrError<DP>>;

}



impl<DP: Scheme, IS: PresentPossibleActions<DP>> AssistingPolicy<DP> for RandomPolicy<DP, IS>
    where <<IS as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type Question = ();

    fn assist(&self, _: Self::Question) -> Result<String, AmfiteatrError<DP>> {
        Ok("I am random policy, I do things at random. I cannot assist you with hints".into())
    }
}
//pub enum PolicyCommand


/// Example question structure for implementation. Here player would have following commands to use:
/// + `quit` - for exiting the game (sending signal [`Quit`](crate::scheme::AgentMessage::Quit);
/// + `play SOMETHING` - play action that is parsed from 'SOMETHING' `str`;
/// + `show` - display information set
/// + `ask` QUESTION - special command to be parsed - depending on Policy it might be something like "ask action top 10" which would list at most ten best looking actions with their Q-functions.
///
/// *Implementation of human controlled playing interface is not a priority now, however this is current idea of how it would look like.*
#[derive(Clone, Debug, PartialEq)]
pub enum TurnCommand<DP: Scheme, P: AssistingPolicy<DP>>{
    Quit,
    Play(DP::ActionType),
    Show,
    AskPolicy(P::Question)

}


impl<DP: Scheme, P: AssistingPolicy<DP>> StrParsed for TurnCommand<DP, P>
    where DP::ActionType: StrParsed,
    <P as AssistingPolicy<DP>>::Question: StrParsed{
    fn parse_from_str(input: &str) -> IResult<&str, Self> {

        if let Ok((action_str, (_, _))) = pair(
            alt((tag("do"), tag("action"), tag("play"), tag::<&str, &str, nom::error::Error<&str>>("a"))),
            space1
        ).parse(input){
            <DP::ActionType as StrParsed>::parse_from_str(action_str)
                .map(|(rest, action)| (rest, Self::Play(action)))
        } else if let Ok((question_str, _)) = pair(
            alt((tag("hint"), tag("policy"), tag("Kowalski analysis"), tag("analysis"), tag("p"), tag::<&str, &str, nom::error::Error<&str>>("Kowalski"))),
            space1
        ).parse(input){
            <P::Question as StrParsed>::parse_from_str(question_str)
                .map(|(rest, question)| (rest, Self::AskPolicy(question)))
        }

        else if let Ok((rest, _)) = pair(
            alt((tag("show"), tag("state"), tag("information"), tag::<&str, &str, nom::error::Error<&str>>("info"))),
            space1
        ).parse(input){
            Ok((rest, Self::Show))
        }

        else if let Ok((rest, _)) = pair(
            alt((tag("exit"), tag::<&str, &str, nom::error::Error<&str>>("quit"))),
            space1
        ).parse(input){
            Ok((rest, Self::Quit))
        }
        else {
            IResult::Err(nom::Err::Failure(nom::error::Error::new(input, nom::error::ErrorKind::Tag)))
        }



    }






}

#[cfg(test)]
mod tests{
    use crate::agent::manual_control::{TurnCommand};
    use crate::agent::RandomPolicy;
    use crate::demo::{DemoAction, DemoDomain, DemoInfoSet};
    use crate::util::StrParsed;

    type DemoTopCommand = TurnCommand<DemoDomain, RandomPolicy<DemoDomain, DemoInfoSet>>;
    #[test]
    fn parse_interactive_command(){


        let mut tc = DemoTopCommand::parse_from_str("quit  dasd").unwrap().1;
        assert_eq!(tc, TurnCommand::Quit);
        tc = TurnCommand::parse_from_str("play     2").unwrap().1;
        assert_eq!(tc, TurnCommand::Play(DemoAction(2)));
        assert!(DemoTopCommand::parse_from_str("play  900").is_err());

        tc = TurnCommand::parse_from_str("hint ddd").unwrap().1;
        assert_eq!(tc, TurnCommand::AskPolicy(()));
        tc = TurnCommand::parse_from_str("info dsaaww").unwrap().1;
        assert_eq!(tc, TurnCommand::Show);

    }
}
