
use amfiteatr_core::agent::Policy;
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::error::AmfiteatrError;


pub trait PolicySpecimen<DP: DomainParameters>: Policy<DP>{

    type Mutagen;


    fn cross(&self, other: &Self) -> Self;

    fn mutate(&mut self, mutagen: &Self::Mutagen) -> Result<(), AmfiteatrError<DP>>;

}

