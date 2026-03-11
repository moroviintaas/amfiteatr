use rmcp::model::ErrorCode;
use rmcp::model::Content;
use rmcp::model::{CallToolResult, ServerInfo};
use crate::scheme::Renew;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use crate::env::SequentialGameState;
use crate::scheme::Scheme;
use rmcp::{
    handler::server::{
    router::{prompt::PromptRouter, tool::ToolRouter}
    },
    schemars::JsonSchema, tool, tool_router, ErrorData, ServerHandler,
    model::ServerCapabilities,
};
use rmcp::handler::server::router::tool::IntoToolRoute;
use rmcp::handler::server::wrapper::Parameters;
use tokio::sync::Mutex;


#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct McpRequestPerformAction<SC: Scheme>
where SC::ActionType:  JsonSchema + Serialize,
      for<'a> SC::AgentId: JsonSchema + Serialize,
{
    agent_id: SC::AgentId,
    action: SC::ActionType,
}


struct McpEnvironmentInternal<SC: Scheme, ST: SequentialGameState<SC> + 'static, Seed: 'static>
where SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
    SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    pub(crate) game_state: ST,
    pub(crate) penalties: HashMap<SC::AgentId, SC::UniversalReward>,
    pub(crate) game_steps: u64,
    pub(crate) game_violators: Option<SC::AgentId>,
    pub(crate) _seed: PhantomData<Seed>,
}



impl<SC: Scheme, ST: SequentialGameState<SC> + 'static, Seed: 'static> McpEnvironmentInternal<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    pub fn new(game_state: ST) -> Self {
        Self{game_state, game_steps: 0, penalties: HashMap::new(), game_violators: None, _seed: Default::default()}
    }
}
pub struct McpEnvironment<SC: Scheme, ST: SequentialGameState<SC> + 'static, Seed: 'static>
where SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
    SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
    Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
      SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
{
    game_name: String,
    tool_router: ToolRouter<Self>,
    internal: Arc<Mutex<McpEnvironmentInternal<SC, ST, Seed>>>
}

#[tool_router]
impl<SC: Scheme, ST: SequentialGameState<SC> + 'static, Seed: 'static> McpEnvironment<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        ST: Renew<SC, Seed>

{


    pub fn new(game_state: ST) -> Self {
        let game_name = game_state.game_name();
        Self{
            internal: Arc::new(Mutex::new(McpEnvironmentInternal::new(game_state))),
            tool_router: Self::tool_router(),
            game_name,
        }
    }


    #[tool(description = "Reset environment")]
    async fn reset(&self, Parameters(seed): Parameters<Seed>) -> Result<(), ErrorData>
    {

        let mut env = self.internal.lock().await;
        env.game_violators = None;
        env.game_steps = 0;
        env.game_state.renew_from(seed).map_err(|e| ErrorData{
            code: ErrorCode::INTERNAL_ERROR,
            message: format!("Failed to renew game : {:?}", e).into(),
            data: None
        })
    }
    

}



impl<SC: Scheme, ST: SequentialGameState<SC> + 'static, Seed: 'static> ServerHandler for McpEnvironment<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        ST: Renew<SC, Seed>

{
    fn get_info(&self) -> ServerInfo {

        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions(format!("A game environment for game {} (controls game flow)", self.game_name))
    }
}