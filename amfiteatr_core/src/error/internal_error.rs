use std::error::Error;
use std::fmt::Debug;
use crate::domain::DomainParameters;

pub trait InternalGameError<Spec: DomainParameters>: Error + Clone + Debug + Send{

}


impl<T: Error + Clone + Debug + Send, DP:DomainParameters> InternalGameError<DP> for T{

}
/*
impl<Internal, Spec: ProtocolSpecification> From<Internal> for TurError<Spec>{
    fn from(value: Internal) -> Self {
        Self::GameError(value)
    }
}*/