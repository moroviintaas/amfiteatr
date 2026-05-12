use std::sync::Arc;
use rmcp::{
    ErrorData as McpError, RoleServer, ServerHandler,
    handler::server::{
        router::{prompt::PromptRouter, tool::ToolRouter},
        wrapper::Parameters,
    },
    model::*,
    prompt, prompt_handler, prompt_router, schemars,
    service::RequestContext,
    task_handler,
    task_manager::OperationProcessor,
    tool, tool_handler, tool_router,
};
use amfiteatr_core::env::{McpCoreSequentialEnvironment, McpRequestPerformAction};
use crate::connect_four::common::ConnectFourScheme;
use crate::connect_four::env::ConnectFourRustEnvState;
use serde_json::json;
use tokio::sync::Mutex;
use amfiteatr_core::env::McpRequestForAgent;

#[derive(Clone)]
pub struct McpEnvironmentConnectFour{
    core: McpCoreSequentialEnvironment<ConnectFourScheme, ConnectFourRustEnvState, ()>,
    #[allow(dead_code)]
    tool_router: ToolRouter<McpEnvironmentConnectFour>,
    #[allow(dead_code)]
    prompt_router: PromptRouter<McpEnvironmentConnectFour>,
    processor: Arc<Mutex<OperationProcessor>>,

}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct AgentRequest{
    agent_id: u32
}

#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ExamplePromptArgs {
    /// A message to put in the prompt
    pub message: String,
}


#[derive(Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct CounterAnalysisArgs {
    /// The target value you're trying to reach
    pub goal: i32,
    /// Preferred strategy: 'fast' or 'careful'
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strategy: Option<String>,
}



#[prompt_router]
impl McpEnvironmentConnectFour{
    #[prompt(
        name = "player_step_prompt",
        meta = Meta(rmcp::object!({"meta_key": "meta_value"}))
    )]
    async fn player_step_prompt(
        &self,
        _ctx: RequestContext<RoleServer>,
    ) ->  Result<GetPromptResult, McpError> {
        self.core.player_step_prompt(_ctx).await
    }





}
#[tool_router]
impl McpEnvironmentConnectFour{
    pub fn new() -> Self{
        Self{
            core: McpCoreSequentialEnvironment::new(ConnectFourRustEnvState::default()),
            #[allow(dead_code)]
            tool_router: Self::tool_router(),
            processor: Arc::new(Mutex::new(OperationProcessor::new())),
            prompt_router: Self::prompt_router(),
        }
    }

    fn _create_resource_text(&self, uri: &str, name: &str) -> Resource {
        RawResource::new(uri, name.to_string()).no_annotation()
    }


    #[tool(description = "Reset environment - set game in initial state")]
    async fn reset(&self) -> Result<CallToolResult, ErrorData>
    {

        self.core.reset(Parameters(())).await
    }



    /*
    #[tool(description = "Repeat what you say")]
    fn echo(&self, Parameters(object): Parameters<JsonObject>) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(
            serde_json::Value::Object(object).to_string(),
        )]))
    }

     */




    #[tool(description = "Get updates for selected agent")]
    async fn get_updates(&self, Parameters(McpRequestForAgent{agent_id}): Parameters<McpRequestForAgent<ConnectFourScheme>>) -> Result<CallToolResult, ErrorData>
    {

        //let s = agent_id_json.to_string();
        //let c4p: ConnectFourPlayer = serde_json::from_str(&s[..])?;
        //self.core.get_updates(Parameters(player)).await
        self.core.get_updates(Parameters(McpRequestForAgent{agent_id})).await
        //self.core.get_updates(Parameters(c4p)).await

    }

    #[tool(description = "Get score of specific agent")]
    async fn get_score(&self, Parameters(McpRequestForAgent{agent_id}): Parameters<McpRequestForAgent<ConnectFourScheme>>)  -> Result<CallToolResult, ErrorData> {

        /*
        let player = match agent_id  & 0x01{

            1 => ConnectFourPlayer::Two,
            _ => ConnectFourPlayer::One,
        };

         */
        self.core.get_score(Parameters(McpRequestForAgent{agent_id})).await
    }

    #[tool(description = "Process action on environment and produce update messages")]
    async fn process_action(&self, Parameters(McpRequestPerformAction{action, agent_id}): Parameters<McpRequestPerformAction<ConnectFourScheme>>) -> Result<CallToolResult, ErrorData>
    {
        self.core.process_action(Parameters(McpRequestPerformAction{action, agent_id})).await

    }

    /*
    #[tool(description = "Get updates for selected agent")]
    async fn get_updates(&self, Parameters(agent_id): Parameters<ConnectFourPlayer>) -> Result<CallToolResult, ErrorData>
    {
        self.core.get_updates(Parameters(agent_id)).await

    }






     */

}

#[tool_handler(meta = Meta(rmcp::object!({"tool_meta_key": "tool_meta_value"})))]
#[prompt_handler(meta = Meta(rmcp::object!({"router_meta_key": "router_meta_value"})))]
#[task_handler]
impl ServerHandler for McpEnvironmentConnectFour

{

    fn get_info(&self) -> ServerInfo {

        ServerInfo::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_prompts()
                .enable_tasks()
                .build()
        )
            .with_server_info(Implementation::from_build_env())
            .with_protocol_version(ProtocolVersion::V_2024_11_05)
            .with_instructions(format!("A game environment for game {} (controls game flow)", self.core.game_name()))
    }


    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, McpError> {
        Ok(ListResourcesResult {
            resources: vec![
                self._create_resource_text("str:////Users/to/some/path/", "cwd"),
                self._create_resource_text("memo://insights", "memo-name"),
            ],
            next_cursor: None,
            meta: None,
        })
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResult, McpError> {
        let uri = &request.uri;
        match uri.as_str() {
            "str:////Users/to/some/path/" => {
                let cwd = "/Users/to/some/path/";
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    cwd,
                    uri.clone(),
                )]))
            }
            "memo://insights" => {
                let memo = "Business Intelligence Memo\n\nAnalysis has revealed 5 key insights ...";
                Ok(ReadResourceResult::new(vec![ResourceContents::text(
                    memo,
                    uri.clone(),
                )]))
            }
            _ => Err(McpError::resource_not_found(
                "resource_not_found",
                Some(json!({
                    "uri": uri
                })),
            )),
        }
    }



    async fn list_resource_templates(
        &self,
        _request: Option<PaginatedRequestParams>,
        _: RequestContext<RoleServer>,
    ) -> Result<ListResourceTemplatesResult, McpError> {
        Ok(ListResourceTemplatesResult {
            next_cursor: None,
            resource_templates: Vec::new(),
            meta: None,
        })
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



#[cfg(test)]
mod tests {
    use rmcp::{ClientHandler, ServiceExt};
    use rmcp::model::{CallToolRequestParams, ClientRequest, Request, ServerResult};


    #[derive(Default, Clone)]
    struct TestClient;

    impl ClientHandler for TestClient {}


    #[tokio::test]
    async fn test_client_enqueues_long_task() -> anyhow::Result<()> {
        // let counter = McpEnvironmentConnectFour::new();
        //let processor = counter.processor.clone();
        let client = TestClient::default();

        let (_server_transport, client_transport) = tokio::io::duplex(4096);
        /*
        let server_handle = tokio::spawn(async move {
            let service = counter.serve(server_transport).await?;
            service.waiting().await?;
            anyhow::Ok(())
        });

         */

        let client_service = client.serve(client_transport).await?;
        let mut task_meta = serde_json::Map::new();
        task_meta.insert(
            "source".into(),
            serde_json::Value::String("integration-test".into()),
        );
        let params = CallToolRequestParams::new("reset");
            //.with_task(task_meta);//.with_arguments(json!(r#" "1": ()"#));//.with_task(task_meta);//.with_arguments(json!(()));


        let response = client_service
            .send_request(ClientRequest::CallToolRequest(Request::new(params.clone())))
            .await?;






        let ServerResult::CallToolResult(info) = response else {
            panic!("expected task creation result, got {response:?}");
        };

        assert!(!info.is_error.unwrap());


        /*
        let ServerResult::CreateTaskResult(info) = response else {
            panic!("expected task creation result, got {response:?}");
        };
        let task = info.task;
        assert_eq!(task.status, TaskStatus::Working);


         */

        /*
        // task list should show the task
        let tasks = client_service
            .send_request(ClientRequest::ListTasksRequest(
                RequestOptionalParam::default(),
            ))
            .await
            .unwrap();
        let ServerResult::ListTasksResult(listed) = tasks else {
            panic!("expected list tasks result, got {tasks:?}");
        };
        assert_eq!(listed.tasks[0].task_id, task.task_id);
        tokio::time::sleep(Duration::from_millis(50)).await;
        let running = processor.lock().await.running_task_count();
        assert_eq!(running, 1);

        client_service.cancel().await?;
        let _ = server_handle.await;

         */
        Ok(())


    }
}
