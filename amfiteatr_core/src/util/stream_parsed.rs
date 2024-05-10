use nom::character::complete::space0;
use nom::IResult;
use crate as amfiteatr_core;

/// Trait for data that can be constructed with [`nom`] parser.
/// It is designed to support  hierarchical construction, typically for action.
/// # Example 1: Simple 2-level action construction
/// ```
/// use amfiteatr_core::util::{StrParsed};
/// use amfiteatr_proc_macro::{StrParsed};
/// #[derive( StrParsed, PartialEq, Debug)]
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
/// #[derive(StrParsed, PartialEq, Debug)]
/// pub enum Action<T: for<'a> amfiteatr_core::util::StrParsed + StrParsed>{
///     #[keywords("move", "m")]
///     Move(Direction, T),
///     #[keywords("look", "l", )]
///     Look(Direction),
///     #[keywords("wait", "w")]
///     Wait(T),
/// }
/// let a = Action::parse_from_str("m w 72 h");
/// let a2 = Action::parse_from_str("m w 72 h");
/// let d = Direction::parse_from_str("w 72 h");
/// assert_eq!(d, Ok(("72 h", Direction::Up)));
/// assert_eq!(a, Ok(("h", Action::Move(Direction::Up, 72u8))));
/// assert_eq!(a2, Ok(("h", Action::Move(Direction::Up, 72u8))));
/// ```
pub trait StrParsed: Sized{
    fn parse_from_str(input: &str) -> IResult<&str, Self>;
}


impl StrParsed for u8{
    fn parse_from_str(input: &str) -> IResult<&str, Self> {
        nom::sequence::terminated(nom::character::complete::u8, space0)(input)
    }
}

impl StrParsed for (){
    fn parse_from_str(input: &str) -> IResult<&str, Self> {
        Ok((input, ()))
    }
}
/*
/// Trait for data that can be constructed with [`nom`] parser.
/// It is designed to support  hierarchical construction, typically for action.
/// # Example 1: Simple 2-level action construction
/// ```
/// use nom::InputTake;
/// use amfiteatr_core::util::{TokenParsed};
/// use amfiteatr_proc_macro::{TokenParsed};
///
///
/// pub enum AToken{
///     Up,
///     Down,
///     Left,
///     Right,
///     Move,
///     Look,
///     Wait,
///     U8(u8)
/// }
/// pub struct ATokenSlice<'a>(pub &'a [AToken]);
/// impl InputTake for ATokenSlice{
///     fn take(&self, count: usize) -> Self {
///         Self(&self.0[0..count])
///     }
///
///     fn take_split(&self, count: usize) -> (Self, Self) {
///         todo!()
///     }
///
/// }
///
/// #[derive( TokenParsed, PartialEq, Debug)]
/// #[token_type(AToken)]
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
/// #[derive(TokenParsed, PartialEq, Debug)]
/// #[token_type(AToken)]
/// pub enum Action<T>{
///     #[token(AToken::Move)]
///     Move(Direction, T),
///     #[token(AToken::Look)]
///     Look(Direction),
///     #[token(AToken::Wait)]
///     Wait(T),
/// }
/// let tokens_move = vec![AToken::Move, AToken::Left, AToken::U8(5)];
/// let action = Action::parse_from_tokens(tokens_move).unwrap();
/// ```


*/

pub struct TokensBorrowed<'a, T>(pub &'a [T]);

impl<'a ,T, Idx> std::ops::Index<Idx> for TokensBorrowed<'a, T>
where Idx: std::slice::SliceIndex<[T]>{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

pub trait TokenParsed<T>: Sized{

    fn parse_from_tokens(input: T) -> IResult<T, Self>;
}

impl<I> TokenParsed<I>  for (){
    fn parse_from_tokens(input: I) -> IResult<I, Self> {
        Ok((input, ()))
    }
}

impl TokenParsed<&str> for u8{
    fn parse_from_tokens(input: &str) -> IResult<&str, Self> {
        nom::sequence::terminated(nom::character::complete::u8, space0)(input)
    }
}


#[cfg(test)]
mod tests{
    //use crate::util::TokenParsed;
    use crate::macros;
    use crate::util;

}