use serde::{Serialize};
use amfiteatr_core::scheme::Scheme;

#[derive(Serialize, Clone, Debug)]
pub struct PayoffSeries<S: Scheme>
where <S as Scheme>::AgentId: Serialize,
    <S as Scheme>::UniversalReward: Serialize,
{
    pub id: S::AgentId,
    pub payoffs: Vec<f32>,

}

#[derive(Serialize, Clone, Debug)]
pub struct PayoffGroupSeries{
    pub id: String,
    pub payoffs: Vec<f32>,

}

#[derive(Serialize,  Clone, Debug, Default)]
pub struct MultiAgentPayoffSeries<S: Scheme>
where <S as Scheme>::AgentId: Serialize,
    <S as Scheme>::UniversalReward: Serialize,{
    pub agent_series: Vec<PayoffSeries<S>>
}