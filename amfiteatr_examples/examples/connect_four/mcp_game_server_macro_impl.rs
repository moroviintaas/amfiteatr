use std::path::PathBuf;
use clap::Parser;
use rmcp::transport::streamable_http_server::{
    StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
};
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    {self},
};
use amfiteatr_examples::connect_four::env::{ConnectFourRustEnvState, McpConnectFourRustEnvState};



#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct McpEnvOptions{
    #[arg(short = 'p', long = "port", default_value = "7701")]
    pub port: u16,
}

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
    let args = McpEnvOptions::try_parse()?;

    let service = StreamableHttpService::new(
        || Ok(McpConnectFourRustEnvState::mcp_new(ConnectFourRustEnvState::new())),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default().with_cancellation_token(ct.child_token()),
    );

    let router = axum::Router::new().nest_service("/mcp", service);

    let tcp_listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", args.port)).await?;
    let _ = axum::serve(tcp_listener, router)
        .with_graceful_shutdown(async move {
            tokio::signal::ctrl_c().await.unwrap();
            ct.cancel();
        })
        .await;
    Ok(())
}