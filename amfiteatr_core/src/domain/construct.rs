


/// Trait for objects that can be renewed using some data.
/// For example agents can be renewed with new state for new game episode without changing
/// things that do not need to be changed (like communication interface or trajectory archive).
pub trait Renew<S>{

    fn renew_from(&mut self, base: S);



}
