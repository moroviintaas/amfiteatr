mod communicating_agent;
mod stateful_agent;
mod automatic;
mod rewarded_agent;
mod tracing_agent;
mod policy_agent;
mod reset_agent;
mod id_agent;
mod multi_episode_agent;
mod model;

pub use communicating_agent::*;
pub use stateful_agent::*;
pub use automatic::*;
pub use rewarded_agent::*;
pub use tracing_agent::*;
pub use reset_agent::*;
pub use policy_agent::*;
pub use id_agent::*;
pub use multi_episode_agent::*;
pub use model::*;


//#[cfg(feature = "manual_control")]
/// Experimental module for supporting human-controlled policies (Human playing interface).
pub mod manual_control;
//pub use manual_control::MultiEpisodeCliAgent;
