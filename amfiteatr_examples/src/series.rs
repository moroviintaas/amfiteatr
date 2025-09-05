use serde::{Serialize};
use amfiteatr_core::scheme::Scheme;

#[derive(Serialize, Clone, Debug)]
pub struct PayoffSeries<DP: Scheme>
where <DP as Scheme>::AgentId: Serialize,
    <DP as Scheme>::UniversalReward: Serialize,
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
pub struct MultiAgentPayoffSeries<DP: Scheme>
where <DP as Scheme>::AgentId: Serialize,
    <DP as Scheme>::UniversalReward: Serialize,{
    pub agent_series: Vec<PayoffSeries<DP>>
}