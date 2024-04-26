use nom::IResult;

pub trait NomParsed<I>: Sized{

    fn nom_parse(input: I) -> IResult<I, Self>;
}

impl<I> NomParsed<I>  for (){
    fn nom_parse(input: I) -> IResult<I, Self> {
        Ok((input, ()))
    }
}