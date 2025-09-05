use amfiteatr_core::agent::Policy;
use amfiteatr_core::scheme::Scheme;
use amfiteatr_core::error::AmfiteatrError;


pub trait PolicySpecimen<S: Scheme, M>: Policy<S> {



    fn cross(&self, other: &Self) -> Self;

    fn mutate(&mut self, mutagen: M) -> Result<(), AmfiteatrError<S>>;

}

