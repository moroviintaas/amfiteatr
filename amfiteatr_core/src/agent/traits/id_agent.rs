use crate::domain::DomainParameters;



/// Provide identification tag for agent.
pub trait IdAgent<DP: DomainParameters>{
    fn id(&self) -> &<DP as DomainParameters>::AgentId;
}