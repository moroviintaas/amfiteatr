use rand::distr::Distribution;
use amfiteatr_core::agent::Policy;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;


pub trait PolicySpecimen<DP: DomainParameters, M>: Policy<DP> {



    fn cross(&self, other: &Self) -> Self;

    fn mutate(&mut self, mutagen: M) -> Result<(), AmfiteatrError<DP>>;

}

