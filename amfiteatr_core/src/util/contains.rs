use crate::agent::Policy;
use crate::scheme::Scheme;

pub trait MaybeContainsOne<T>{

    fn get(&self) -> Option<&T>;
    fn get_mut(&mut self) -> Option<&mut T>;
}