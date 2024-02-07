use std::fmt::{Debug, Display};

/// This trait does not anything particular, however it marks type that it
/// can be used in domain between [_environment_](crate::env) and [_agent_](crate::agent).
pub trait Action: Debug + Send + Clone + Display{}