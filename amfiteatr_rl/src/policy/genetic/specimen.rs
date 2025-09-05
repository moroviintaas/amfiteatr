use amfiteatr_core::agent::Policy;
use amfiteatr_core::scheme::Scheme;
use amfiteatr_core::error::AmfiteatrError;


pub trait PolicySpecimen<DP: Scheme, M>: Policy<DP> {



    fn cross(&self, other: &Self) -> Self;

    fn mutate(&mut self, mutagen: M) -> Result<(), AmfiteatrError<DP>>;

}

