use crate::scheme::Scheme;



/// Provide identification tag for agent.
pub trait IdAgent<S: Scheme>{
    fn id(&self) -> &<S as Scheme>::AgentId;
}