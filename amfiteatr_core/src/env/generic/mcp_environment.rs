use std::any::Any;
use rmcp::model::{ErrorCode, PromptMessage, PromptMessageRole};
use rmcp::model::Content;
use rmcp::model::{CallToolResult, ServerInfo};
use crate::scheme::Renew;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::Arc;
use nom::Parser;
use serde::{Deserialize, Serialize};
use crate::env::{GameStateWithPayoffs, ScoreEnvironment, SequentialGameState};
use crate::scheme::Scheme;
use rmcp::{handler::server::{
    router::{prompt::PromptRouter, tool::ToolRouter}
}, schemars::JsonSchema, tool, ErrorData, ServerHandler, model::ServerCapabilities};
use rmcp::handler::server::router::tool::IntoToolRoute;
use rmcp::handler::server::wrapper::Parameters;
use tokio::sync::Mutex;
use crate::error::AmfiteatrError;
use rmcp::{
    service::RequestContext,
    RoleServer,
    tool_handler,
    tool_router,
    task_handler,
    prompt_handler,
    model::{
        Meta,
        InitializeRequestParams,
        InitializeResult,
        Resource,
        RawResource,
        AnnotateAble,
        Implementation,
        ProtocolVersion,
        GetPromptRequestParams,
        GetPromptResult,
        PaginatedRequestParams,
        ListPromptsResult,

    },
    ServiceError,
    ErrorData as McpError,
};
use rmcp::task_manager::{OperationProcessor, OperationResultTransport};

struct ToolCallOperationResult {
    id: String,
    result: Result<CallToolResult, McpError>,
}

impl OperationResultTransport for ToolCallOperationResult {
    fn operation_id(&self) -> &String {
        &self.id
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct McpRequestPerformAction<SC: Scheme>
where SC::ActionType:  JsonSchema + Serialize,
      for<'a> SC::AgentId: JsonSchema + Serialize,
{
    pub agent_id: SC::AgentId,
    pub action: SC::ActionType,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct McpRequestForAgent<SC: Scheme>
    where SC::ActionType:  JsonSchema + Serialize,
          for<'a> SC::AgentId: JsonSchema + Serialize,
{
    pub agent_id: SC::AgentId,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct McpActionResponse<SC: Scheme>
where SC::AgentId: JsonSchema + Serialize + for<'a> Deserialize<'a>,{
    pub game_finished: bool,
    pub next_player: Option<SC::AgentId>,
}



/*
#[derive(Debug, serde::Serialize, serde::Deserialize, JsonSchema)]
pub struct McpResponsePerformAction<SC: Scheme, ST: SequentialGameState<SC, Updates: serde::Serialize + serde::Deserialize + JsonSchema>>
    where
          for<'a> SC::AgentId: JsonSchema + Serialize,
{
    agent_id: SC::AgentId,
    action: SC::ActionType,
}

 */


#[derive(Clone)]
struct McpEnvironmentInternal<SC: Scheme  + Send + 'static,
    ST: SequentialGameState<SC> + GameStateWithPayoffs<SC> + Send  +  'static + Renew<SC, Seed>,
    Seed: 'static + Send >
where SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
      SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema
{
    pub(crate) game_state: ST,
    pub(crate) penalties: HashMap<SC::AgentId, SC::UniversalReward>,
    pub(crate) game_steps: u64,
    pub(crate) game_violator: Option<SC::AgentId>,
    pub(crate) _seed: PhantomData<Seed>,

}



impl<SC: Scheme + Serialize + for<'a> Deserialize<'a> + JsonSchema + Send  + 'static,
    ST:  SequentialGameState<SC>  + GameStateWithPayoffs<SC>  + Send  + 'static + Renew<SC, Seed>,
    Seed: 'static + Send >
McpEnvironmentInternal<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema
{
    pub fn new(game_state: ST) -> Self {
        Self{game_state, game_steps: 0, penalties: HashMap::new(), game_violator: None, _seed: Default::default()}
    }
}
#[derive(Clone)]
pub struct McpCoreSequentialEnvironment<
    SC: Scheme + Serialize + for<'a> Deserialize<'a> + JsonSchema + Send  + 'static,
    ST: SequentialGameState<SC>  + GameStateWithPayoffs<SC> + Send  + 'static  + Renew<SC, Seed>,
    Seed: 'static + Send
>
where SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
      SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
      SC::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema
{
    game_name: String,
    //tool_router: ToolRouter<McpEnvironment<SC, ST, Seed>>,
    //prompt_router: PromptRouter<McpEnvironment<SC, ST, Seed>>,
    internal: Arc<Mutex<McpEnvironmentInternal<SC, ST, Seed>>>,
    update_queues: Arc<Mutex<HashMap<SC::AgentId, Vec<SC::UpdateType>>>>,
    //processor: Arc<Mutex<OperationProcessor>>,
}

//#[tool_router]
impl<
    SC: Scheme + Serialize + for<'a> Deserialize<'a> + JsonSchema + Send  + 'static,
    ST: SequentialGameState<SC>  + GameStateWithPayoffs<SC> + Send  + 'static + Renew<SC, Seed>,
    Seed: 'static + Send
>
McpCoreSequentialEnvironment<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema

{


    pub fn new(game_state: ST) -> Self {
        let game_name = game_state.game_name();
        Self{
            internal: Arc::new(Mutex::new(McpEnvironmentInternal::new(game_state))),
            //tool_router: Self::tool_router(),//ToolRouter::new(),
            //prompt_router: Self::prompt_router(),
            game_name,
            update_queues: Arc::new(Mutex::new(HashMap::new())),
            //processor: Arc::new(Mutex::new(OperationProcessor::new())),
        }
    }


    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }


    pub fn game_name(&self) -> &str{
        &self.game_name[..]
    }
    fn clear_observations(
        &self,
        store: &mut tokio::sync::MutexGuard<'_, HashMap<SC::AgentId, Vec<SC::UpdateType>>>
    ){
        for obs in store.values_mut(){
            obs.clear()
        }
    }

    fn store_observation(&self,
                         store: &mut tokio::sync::MutexGuard<'_, HashMap<SC::AgentId, Vec<SC::UpdateType>>>,
                         agent_id: &SC::AgentId,
                         observation: SC::UpdateType
    ){

        if let Some(update_list) = store.get_mut(agent_id){
            update_list.push(observation);
        } else {
            store.insert(agent_id.clone(), vec![observation]);
        }

    }

    fn take_observation(
        &self,
        store: &mut tokio::sync::MutexGuard<'_, HashMap<SC::AgentId, Vec<SC::UpdateType>>>,
        agent_id: &SC::AgentId,

    ) -> Vec<SC::UpdateType>{

        let r = store.remove(agent_id);
        store.insert(agent_id.clone(), vec![]);
        r.unwrap_or(Vec::new())
    }







    //#[tool(description = "Reset environment")]
    pub async fn reset(&self, Parameters(seed): Parameters<Seed>) -> Result<CallToolResult, ErrorData>
    {

        let mut env = self.internal.lock().await;

        env.game_violator = None;
        env.game_steps = 0;
        /*
        for obs in observations.values_mut(){
            obs.clear()
        }
        */


        let r = env.game_state.renew_from(seed).map_or_else(
            |e| Err(ErrorData{
                code: ErrorCode::INTERNAL_ERROR,
                message: format!("Failed to renew game : {:?}", e).into(),
                data: None
            }),
            |_|{
                Ok(CallToolResult::success(vec![]))
            }
        );
        let first_obs = env.game_state.first_observations();
        let mut observations = self.update_queues.lock().await;
        self.clear_observations(&mut observations);
        if let Some(first_obs) = first_obs{
            for (agent, obs) in first_obs.into_iter(){
                self.store_observation(&mut observations, &agent, obs)
            }
        }


        r

    }



    //#[tool(description = "Get updates for selected agent")]
    pub async fn get_updates(&self, Parameters(McpRequestForAgent{agent_id}): Parameters<McpRequestForAgent<SC>>) -> Result<CallToolResult, ErrorData>
    {
        let mut observations = self.update_queues.lock().await;
        let updates = self.take_observation(&mut observations, &agent_id);
        Ok(CallToolResult::success(vec![Content::json(updates)?]))

    }


    //#[tool(description = "Get score of specific agent")]
    pub async fn get_score(&self, Parameters(McpRequestForAgent{agent_id}): Parameters<McpRequestForAgent<SC>>) -> Result<CallToolResult, ErrorData> {
        let env = self.internal.lock().await;
        Ok(CallToolResult::success(vec![Content::json(env.game_state.state_payoff_of_player(&agent_id))?]))
    }

    //#[tool(description = "Process action on environment and produce update messages")]
    pub async fn process_action(&self, Parameters(request): Parameters<McpRequestPerformAction<SC>>) -> Result<CallToolResult, ErrorData>
    {
        let agent = request.agent_id;
        let action = request.action;

        let mut env = self.internal.lock().await;
        let mut observations = self.update_queues.lock().await;

        env.game_steps += 1;
        let r_updates = env.game_state.forward(agent, action);
        match r_updates{
            Err(e) => Ok(CallToolResult::error(vec![Content::text(format!("{}",
                   AmfiteatrError::Game::<SC> {source: e})
            )])),
            Ok(updates) => {


                for (agent, update) in updates.into_iter(){
                    self.store_observation(&mut observations, &agent, update);
                }
                let response = McpActionResponse::<SC>{
                    game_finished: env.game_state.is_finished(),
                    next_player: env.game_state.current_player(),
                };
                Ok(CallToolResult::success(vec![Content::json(response)?]))
            }

        }



    }

    pub async fn player_step_prompt(
        &self,
        //Parameters(args): Parameters<CounterAnalysisArgs>,
        _ctx: RequestContext<RoleServer>,
    ) -> Result<GetPromptResult, McpError> {
        let mut env = self.internal.lock().await;
        let current_player = env.game_state.current_player();
        let messages = match current_player{
            None => vec![
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "It seems that no player can play now. Usually this means that the game is finished. \
                    Collect the scores of players, and if you wish reset environment and players' information sets for new game".to_string()
                ),
            ],
            Some(player) => vec![
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    format!("It seems that now the player {player} should play")
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    format!("I want to make action as player {player}. What should I do?")
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    format!("Firstly, collect situation updates for player {player}, e.g. by asking environment for updates for that player.")
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "I have collected updates for that player, what should I do with them?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Now you should apply them to update the information set  responsible for maintaining that player's knowledge if situation.\
                    This may be external program or avaiable MCP server".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "The information set is updated, what now?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Using this information set get description of current situation viewed by player".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    format!("I have now a description of current situation viewed by player {player}. What now?")
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Make a decision what action to do. Maybe you can use MCP server that serves policy which can suggest you what action you should play.".to_string()
                ),


            ]
        };

        Ok(GetPromptResult::new(messages).with_description("Player step workflow."))


    }

    pub async fn get_current_player(&self) -> Result<CallToolResult, ErrorData>{
        let mut env = self.internal.lock().await;
        let current_player = env.game_state.current_player();
        Ok(CallToolResult::success(vec![Content::json(current_player)?]))
    }



    

}

/*

//#[tool_handler(meta = Meta(rmcp::object!({"tool_meta_key": "tool_meta_value"})))]
//#[prompt_handler(meta = Meta(rmcp::object!({"router_meta_key": "router_meta_value"})))]
//#[task_handler]
impl<
    SC: Scheme + Serialize + for<'a> Deserialize<'a> + JsonSchema + Send + 'static,
    ST: SequentialGameState<SC>  + GameStateWithPayoffs<SC> + Send  + 'static + Renew<SC, Seed>,
    Seed: 'static + Send
> ServerHandler for McpEnvironment<SC, ST, Seed>
where   SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Send,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UniversalReward: Serialize + for<'a> Deserialize<'a> + JsonSchema

{

    fn get_info(&self) -> ServerInfo {

        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .build()
        )
        .with_server_info(Implementation::from_build_env())
        .with_protocol_version(ProtocolVersion::V_2024_11_05)
        .with_instructions(format!("A game environment for game {} (controls game flow)", self.game_name))
    }




    async fn initialize(
        &self,
        _request: InitializeRequestParams,
        context: RequestContext<RoleServer>,
    ) -> Result<InitializeResult, McpError> {
        if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
            let initialize_headers = &http_request_part.headers;
            let initialize_uri = &http_request_part.uri;
            tracing::info!(?initialize_headers, %initialize_uri, "initialize from http server");
        }
        Ok(self.get_info())
    }




}

 */