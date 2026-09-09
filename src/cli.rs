//! The command line shared by both binaries. A binary passes the engine
//! factory selector it was built with.

use clap::{Parser, Subcommand};
use hologram_live::app::AppState;
use hologram_live::config::AppConfig;
use hologram_live::error::LiveError;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "freeinference", about = "Free verified AI inference.")]
pub struct Cli {
    /// Configuration file; defaults to hologram-live's `live.toml` location.
    #[arg(long, global = true, env = "HOLOGRAM_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Serve the OpenAI compatible endpoint on the configured address.
    Serve {
        /// Listen address, for example 127.0.0.1:11435.
        #[arg(long)]
        listen: Option<String>,
    },
    /// Verify a receipt by exact replay against the running daemon.
    Verify {
        /// Receipt κ, as returned in the x-hologram-receipt header.
        kappa: String,
        /// Daemon base URL.
        #[arg(long, default_value = "http://127.0.0.1:11435")]
        endpoint: String,
    },
}

/// Runs the CLI with `select_engine` deciding how `inference.engine` is
/// built. Exits the process with the conventional code on failure.
pub async fn run(
    select_engine: fn(
        &hologram_live::config::InferenceConfig,
    ) -> Option<hologram_live::inference::EngineFactory>,
) {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Command::Serve { listen: None });
    if let Err(error) = execute(cli.config, command, select_engine).await {
        eprintln!("freeinference: {}: {error}", error.code());
        std::process::exit(match error {
            LiveError::Config(_) | LiveError::Protocol(_) => 2,
            _ => 1,
        });
    }
}

async fn execute(
    config_path: Option<PathBuf>,
    command: Command,
    select_engine: fn(
        &hologram_live::config::InferenceConfig,
    ) -> Option<hologram_live::inference::EngineFactory>,
) -> Result<(), LiveError> {
    match command {
        Command::Verify { kappa, endpoint } => {
            let url = format!(
                "{}/v1/receipts/{kappa}/verify",
                endpoint.trim_end_matches('/')
            );
            let response = reqwest::Client::new()
                .post(&url)
                .send()
                .await
                .map_err(|error| LiveError::Transport(error.to_string()))?;
            let status = response.status();
            let verdict: serde_json::Value = response
                .json()
                .await
                .map_err(|error| LiveError::Protocol(error.to_string()))?;
            if status.is_success() && verdict["verified"].as_bool() == Some(true) {
                println!("confirmed: {kappa} replays byte for byte on this machine");
                Ok(())
            } else if let Some(byte) = verdict["replay"]["first_divergence_byte"].as_u64() {
                println!("refuted: {kappa} diverges at byte {byte}");
                std::process::exit(1)
            } else {
                let reason = verdict["reason"]
                    .as_str()
                    .or_else(|| verdict["error"]["message"].as_str())
                    .unwrap_or("not verified");
                println!("not verified: {reason}");
                std::process::exit(1)
            }
        }
        Command::Serve { listen } => {
            let (mut config, _) = AppConfig::load(config_path.as_deref())?;
            if let Some(listen) = listen {
                config.server.listen = listen;
            }
            crate::configure(&mut config);
            config.validate()?;
            let tracing = hologram_live::observability::init(&config.tracing, &config.telemetry)?;
            let listen = config.server.listen.clone();
            let engine = select_engine(&config.inference);
            let state =
                AppState::build_with(config, tracing, crate::extra_modules(), engine).await?;
            hologram_live::server::serve_with_ready(state, move || {
                println!("freeinference: serving http://{listen}/v1");
                Ok(())
            })
            .await
        }
    }
}
