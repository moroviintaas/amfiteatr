use crate::scheme::Scheme;


/// Structure dedicated for `mcp` requests that expect single agent id as argument.
pub struct McpReqArgAgentId<Sc: Scheme>{
    pub agent_id: Sc::AgentId
}


/// Structure dedicated for `mcp` requests that expect single agent id along with action as argument.
pub struct McpReqArgAgentAction<Sc: Scheme>{
    pub agent_id: Sc::AgentId,
    pub action: Sc::ActionType,
}