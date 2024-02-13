use crate::domain::DomainParameters;
use crate::error::AmfiError;

/// Trait for objects that can be renewed using some data.
/// For example agents can be renewed with new state for new game episode without changing
/// things that do not need to be changed (like communication interface or trajectory archive).
pub trait Renew<DP: DomainParameters, S>{

    fn renew_from(&mut self, base: S) -> Result<(), AmfiError<DP>>;
}

pub trait RenewWithSideEffect<DP: DomainParameters, S>{

    type SideEffect;
    fn renew_with_side_effect_from(&mut self, base: S) -> Result<Self::SideEffect, AmfiError<DP>>;

}