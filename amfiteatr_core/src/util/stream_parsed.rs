use nom::character::complete::space0;
use nom::IResult;



/// Trait for data that can be constructed with [`nom`] parser.
/// It is designed to support  hierarchical construction, typically for action.
/// # Example 1: Simple 2-level action construction
/// ```
/// use amfiteatr_core::util::StreamParsed;
/// use amfiteatr_proc_macro::StreamParsed;
/// #[derive(StreamParsed, PartialEq, Debug)]
/// pub enum Direction{
///     #[keywords("up", "w")]
///     Up,
///     #[keywords("down", "s", )]
///     Down,
///     #[keywords("right", "d")]
///     Right,
///     #[keywords("left", "a")]
///     Left
/// }
/// #[derive(StreamParsed, PartialEq, Debug)]
/// pub enum Action<T: for<'a> amfiteatr_core::util::StreamParsed<&'a str>>{
///     #[keywords("move", "m")]
///     Move(Direction, T),
///     #[keywords("look", "l", )]
///     Look(Direction),
///     #[keywords("wait", "w")]
///     Wait(T),
/// }
/// let a = Action::nom_parse("m w 72 h");
/// let d = Direction::nom_parse("w 72 h");
///  assert_eq!(d, Ok(("72 h", Direction::Up)));
/// assert_eq!(a, Ok(("h", Action::Move(Direction::Up, 72u8))));
/// ```
pub trait StreamParsed<I>: Sized{

    fn parse_from_stream(input: I) -> IResult<I, Self>;
}

impl<I> StreamParsed<I>  for (){
    fn parse_from_stream(input: I) -> IResult<I, Self> {
        Ok((input, ()))
    }
}

impl StreamParsed<&str> for u8{
    fn parse_from_stream(input: &str) -> IResult<&str, Self> {
        nom::sequence::terminated(nom::character::complete::u8, space0)(input)
    }
}
#[cfg(test)]
mod tests{
    use crate::util::StreamParsed;
    use crate::macros;
    use crate::util;

}