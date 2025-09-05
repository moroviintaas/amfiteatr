use std::fmt::{Display, Formatter};
use crate::agent::AgentIdentifier;
use crate::scheme::Action;
//use crate::state::StateUpdate;

/// Structure to represent relation between agent and action.
/// This is just named tuple (pair in this case).
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "speedy", derive(speedy::Writable, speedy::Readable))]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AgentActionPair<Id: AgentIdentifier, A: Action>{
    #[cfg_attr(feature = "serde", serde(bound(serialize = "A: serde::Serialize", deserialize = "A: serde::Deserialize<'de>")))]
    //#[cfg_attr(feature = "serde", serde(bound(deserialize = "A: Deserialize")))]
    pub action: A,
    #[cfg_attr(feature = "serde", serde(bound(serialize = "Id: serde::Serialize")))]
    #[cfg_attr(feature = "serde", serde(bound(deserialize = "Id: serde::Deserialize<'de>")))]
    pub agent: Id
}

impl<Agt: AgentIdentifier, Act: Action> AgentActionPair<Agt, Act>{
    pub fn new(agent_id: Agt, action: Act) -> Self { Self{action, agent: agent_id}}

    pub fn action(&self) -> &Act { &self.action}
    pub fn agent(&self) -> &Agt {&self.agent}
}

impl<Agt: AgentIdentifier, Act: Action> Display for AgentActionPair<Agt, Act> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Update [agent: {} performed action {}", self.agent, self.action)
    }
}

/*
impl<Agt: AgentIdentifier, Act: Action> StateUpdate for AgentActionPair<Agt, Act>{

}

 */