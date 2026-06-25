use std::marker::PhantomData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::agent::InformationSet;
use crate::scheme::Scheme;


/// Structure dedicated for `mcp` requests that expect single agent id as argument.
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqArgAgentId<Sc: Scheme >
where Sc::AgentId: JsonSchema{
    pub agent_id: Sc::AgentId
}


/// Structure dedicated for `mcp` requests that expect single agent id along with action as argument.
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqArgAgentAction<Sc: Scheme >
where Sc::AgentId: JsonSchema,
    Sc::ActionType: JsonSchema,
{
    pub agent_id: Sc::AgentId,
    pub action: Sc::ActionType,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqReward<Sc: Scheme >
    where Sc::UniversalReward: JsonSchema,
{
    pub reward: Sc::UniversalReward,
}

#[derive(Clone, JsonSchema, Serialize, Deserialize)]
pub struct McpReqSelectAction<SC: Scheme, IS: InformationSet<SC>>
    where
        IS:  JsonSchema,
        //SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    #[cfg_attr(feature = "serde", serde(bound(serialize = "IS: serde::Serialize", deserialize = "IS: serde::Deserialize<'de>")))]
    pub information_set: IS,
    _sc: PhantomData<SC>,
}