use crate::scheme::Scheme;



/// Provide identification tag for agent.
pub trait IdAgent<DP: Scheme>{
    fn id(&self) -> &<DP as Scheme>::AgentId;
}