use nom::character::complete::space0;
use nom::error::ErrorKind;
use nom::IResult;
use amfiteatr_proc_macro::{TokenParsed, TokenVariant};
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






pub trait PrimitiveMarker<Pt>{

    fn primitive(&self) -> Option<Pt>;
}

impl<'a, T: PrimitiveMarker<Pt>, Pt> TokenParsed<TokensBorrowed<'a, T>> for Pt{
    fn parse_from_tokens(input: TokensBorrowed<'a, T>) -> IResult<TokensBorrowed<'a, T>, Self> {

        if input.is_empty(){
            return Err(nom::Err::Failure(nom::error::Error{input, code: ErrorKind::Eof}))
        }
        let token = &input[0];
        match token.primitive(){
            None => Err(nom::Err::Error(nom::error::Error{input, code: ErrorKind::Tag})),
            Some(t) => {
                let rest = TokensBorrowed(&input.0[1..]);
                Ok((rest, t))
            }
        }

    }
}


#[derive(Clone, Debug)]
pub struct TokensBorrowed<'a, T>(pub &'a [T]);
impl<'a, T> From<&'a [T]> for TokensBorrowed<'a, T>{
    fn from(value: &'a [T]) -> Self {
        Self(value)
    }
}

impl<'a, T> TokensBorrowed<'a, T>{
    pub fn len(&self) -> usize{
        self.0.len()
    }
    pub fn is_empty(&self) -> bool{
        self.len() == 0
    }
}
impl<'a ,T, Idx> std::ops::Index<Idx> for TokensBorrowed<'a, T>
where Idx: std::slice::SliceIndex<[T]>{
    type Output = Idx::Output;

    fn index(&self, index: Idx) -> &Self::Output {
        &self.0[index]
    }
}

/// Trait for data that can be constructed with [`nom`] parser.
/// It is designed to support  hierarchical construction, typically for action.
/// # Example 1: Simple 2-level action construction
/// ```
/// use std::fmt::Alignment::Left;
/// use nom::InputTake;
/// use amfiteatr_core::util::{PrimitiveMarker, TokenParsed, TokensBorrowed};
/// use amfiteatr_proc_macro::{TokenParsed, TokenVariant};
///
///
/// #[derive(TokenVariant, PartialEq, Debug)]
/// pub enum AToken{
///     Up,
///     Down,
///     Left,
///     Right,
///     Move,
///     Look,
///     Wait,
///     #[primitive]
///     U8(u8),
///     #[primitive]
///     F32(f32),
/// }
///
/// #[derive( TokenParsed, PartialEq, Debug)]
/// #[token_type(AToken)]
/// pub enum Direction{
///     #[token(Up)]
///     Up,
///     #[token(Down)]
///     Down,
///     #[token(Right)]
///     Right,
///     #[token(Left)]
///     Left
/// }
///
///
/// #[derive(TokenParsed, PartialEq, Debug)]
/// #[token_type(AToken)]
/// pub enum Action<T>
///     where AToken: PrimitiveMarker<T>{
///     #[token(Move)]
///     Move(Direction, T),
///     #[token(Look)]
///     Look(Direction),
///     #[token(Wait)]
///     Wait(u8),
/// }
///
/// let tokens_move = vec![AToken::Move, AToken::Left, AToken::U8(5u8), AToken::Look, AToken::Down ];///
/// let borrowed = TokensBorrowed(&tokens_move[..]);
/// let (rest, action) = Action::<u8>::parse_from_tokens(borrowed).unwrap();
/// assert_eq!(action, Action::Move(Direction::Left, 5u8));
/// assert_eq!(rest.0, &tokens_move[3..]);
/// let (rest, action2) = Action::<u8>::parse_from_tokens(rest).unwrap();
/// assert_eq!(action2, Action::Look(Direction::Down));
///
///     //let  action: Action<u8> = Action::parse_from_tokens(borrowed).unwrap().1;
/// ```
pub trait TokenParsed<T>: Sized{

    fn parse_from_tokens(input: T) -> IResult<T, Self>;
}

/*
impl<I> TokenParsed<I>  for (){
    fn parse_from_tokens(input: I) -> IResult<I, Self> {
        Ok((input, ()))
    }
}

 */

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