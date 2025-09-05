use std::error::Error;
use std::fmt::Debug;
use crate::scheme::Scheme;

pub trait InternalGameError<Spec: Scheme>: Error + Clone + Debug + Send{

}


impl<T: Error + Clone + Debug + Send, S: Scheme> InternalGameError<S> for T{

}
/*
impl<Internal, Spec: ProtocolSpecification> From<Internal> for TurError<Spec>{
    fn from(value: Internal) -> Self {
        Self::GameError(value)
    }
}*/