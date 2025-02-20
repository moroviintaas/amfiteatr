use nom::branch::alt;
use nom::bytes::complete::{tag};
use nom::character::complete::space1;
use nom::IResult;
use nom::sequence::{pair};
use crate::agent::{Policy, PresentPossibleActions, RandomPolicy};
use crate::domain::DomainParameters;
use crate::error::AmfiteatrError;
use crate::util::{StrParsed};
use nom::Parser;


pub trait AssistingPolicy<DP: DomainParameters>: Policy<DP>{

    type Question;

    fn assist(&self, question: Self::Question) -> Result<String, AmfiteatrError<DP>>;

}



impl<DP: DomainParameters, IS: PresentPossibleActions<DP>> AssistingPolicy<DP> for RandomPolicy<DP, IS>
    where <<IS as PresentPossibleActions<DP>>::ActionIteratorType as IntoIterator>::IntoIter : ExactSizeIterator{
    type Question = ();

    fn assist(&self, _: Self::Question) -> Result<String, AmfiteatrError<DP>> {
        Ok("I am random policy, I do things at random. I cannot assist you with hints".into())
    }
}
//pub enum PolicyCommand

#[derive(Clone, Debug, PartialEq)]
pub enum TurnCommand<DP: DomainParameters, P: AssistingPolicy<DP>>{
    Quit,
    Play(DP::ActionType),
    Show,
    AskPolicy(P::Question)

}


impl<DP: DomainParameters, P: AssistingPolicy<DP>> StrParsed for TurnCommand<DP, P>
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
