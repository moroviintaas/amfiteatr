use serde::{Serialize};
use amfiteatr_core::domain::DomainParameters;

#[derive(Serialize, Clone, Debug)]
pub struct PayoffSeries<DP: DomainParameters>
where <DP as DomainParameters>::AgentId: Serialize,
    <DP as DomainParameters>::UniversalReward: Serialize,
{
    pub id: DP::AgentId,
    pub payoffs: Vec<f32>,

}

#[derive(Serialize, Clone, Debug)]
pub struct PayoffGroupSeries{
    pub id: String,
    pub payoffs: Vec<f32>,

}

#[derive(Serialize,  Clone, Debug, Default)]
pub struct MultiAgentPayoffSeries<DP: DomainParameters>
where <DP as DomainParameters>::AgentId: Serialize,
    <DP as DomainParameters>::UniversalReward: Serialize,{
    pub agent_series: Vec<PayoffSeries<DP>>
}