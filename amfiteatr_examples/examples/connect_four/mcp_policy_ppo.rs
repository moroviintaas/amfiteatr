use std::sync::{Arc, Mutex};
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    {self},
};
use amfiteatr_rl::policy::ConfigPPO;
use amfiteatr_rl::tch;

const BIND_ADDRESS: &str = "127.0.0.1:7002";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug".to_string().into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    let ct = tokio_util::sync::CancellationToken::new();

    let policy = Arc::new(Mutex::new(amfiteatr_examples::connect_four::policy::build_ppo_policy_masking(&[64,64], tch::Device::Cpu, ConfigPPO::default(), 5e-4)?));
    //let policy =amfiteatr_examples::connect_four::policy::build_ppo_policy_masking(&[64,64], tch::Device::Cpu, ConfigPPO::default(), 5e-4)?;

    let service = StreamableHttpService::new(
        move || Ok(amfiteatr_examples::connect_four::policy::McpPolicyPPOConnectFour
            ::mcp_new(policy.clone(), "PPO policy for \"Connect Four game\"".into(), String::new())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
    );

    let router = axum::Router::new().nest_service("/mcp", service);

    let tcp_listener = tokio::net::TcpListener::bind(BIND_ADDRESS).await?;
    let _ = axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.unwrap();
            ct.cancel();
        })
        .await;
    Ok(())
}