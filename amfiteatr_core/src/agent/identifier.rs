use std::fmt::{Debug, Display};
use std::hash::Hash;

/// Marker trait for values identifying agent. This simplifies all required traits.
pub trait AgentIdentifier: Debug + Send + Sync + Clone + Hash + Display + PartialEq + Eq + 'static{

}

macro_rules! impl_agent_id_std {
    ($($x: ty), +) => {
        $(
          impl AgentIdentifier for $x{}

        )*

    }
}

impl_agent_id_std!(u8, u16, u32, u64, u128);