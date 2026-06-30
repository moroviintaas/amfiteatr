use std::marker::PhantomData;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use crate::agent::InformationSet;
use crate::scheme::Scheme;


/// Structure dedicated for `mcp` requests that expect single agent id as argument.
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqArgAgentId<Sc: Scheme >
where Sc::AgentId: JsonSchema{
    /// Id of the agent for whom operation is to be made
    pub agent_id: Sc::AgentId
}


/// Structure dedicated for `mcp` requests that expect single agent id along with action as argument.
#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqArgAgentAction<Sc: Scheme >
where Sc::AgentId: JsonSchema,
    Sc::ActionType: JsonSchema,
{
    /// Id of the agent whose information set is updated
    pub agent_id: Sc::AgentId,
    /// Game action to be included as argument (for example to be performed in environment)
    pub action: Sc::ActionType,
}

#[derive(JsonSchema, Serialize, Deserialize)]
pub struct McpReqReward<Sc: Scheme >
    where Sc::UniversalReward: JsonSchema,
{
    /// Noted reward
    pub reward: Sc::UniversalReward,
}

#[derive(Clone, JsonSchema, Serialize, Deserialize)]
pub struct McpReqSelectAction<SC: Scheme, IS: InformationSet<SC>>
    where
        IS:  JsonSchema,
        //SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    /// Representation of information set (game state viewed from perspective of a player).
    #[cfg_attr(feature = "serde", serde(bound(serialize = "IS: serde::Serialize", deserialize = "IS: serde::Deserialize<'de>")))]
    pub information_set: IS,
    _sc: PhantomData<SC>,
}

#[derive(Clone, JsonSchema, Serialize, Deserialize)]
pub struct McpReqUpdateInformationSet<SC: Scheme>
where
    SC::AgentId: JsonSchema,
    SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    /// Id of the agent whose information set is updated
    #[cfg_attr(feature = "serde", serde(bound(serialize = "SC::AgentId: serde::Serialize", deserialize = "SC::AgentId: serde::Deserialize<'de>")))]
    pub agent_id: SC::AgentId,
    /// The list of updates to apply - in chronological order.
    //#[cfg_attr(feature = "serde", serde(bound(serialize = "SC::UpdateType: serde::Serialize", deserialize = "SC::UpdateType: serde::Deserialize<'de>")))]
    pub updates: Vec<SC::UpdateType>,
}