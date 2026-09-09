#![forbid(unsafe_code)]

use clap::{Parser, Subcommand};
use hologram_live::app::AppState;
use hologram_live::config::AppConfig;
use hologram_live::error::LiveError;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "freeinference", about = "Free verified AI inference.")]
struct Cli {
    /// Configuration file; defaults to hologram-live's `live.toml` location.
    #[arg(long, global = true, env = "HOLOGRAM_CONFIG")]
    config: Option<PathBuf>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the OpenAI compatible endpoint on the configured address.
    Serve {
        /// Listen address, for example 127.0.0.1:11435.
        #[arg(long)]
        listen: Option<String>,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let command = cli.command.unwrap_or(Command::Serve { listen: None });
    if let Err(error) = run(cli.config, command).await {
        eprintln!("freeinference: {}: {error}", error.code());
        std::process::exit(match error {
            LiveError::Config(_) | LiveError::Protocol(_) => 2,
            _ => 1,
        });
    }
}

async fn run(config_path: Option<PathBuf>, command: Command) -> Result<(), LiveError> {
    match command {
        Command::Serve { listen } => {
            let (mut config, _) = AppConfig::load(config_path.as_deref())?;
            if let Some(listen) = listen {
                config.server.listen = listen;
            }
            freeinference::configure(&mut config);
            config.validate()?;
            let tracing = hologram_live::observability::init(&config.tracing, &config.telemetry)?;
            let listen = config.server.listen.clone();
            let state =
                AppState::build_with_modules(config, tracing, freeinference::extra_modules())
                    .await?;
            hologram_live::server::serve_with_ready(state, move || {
                println!("freeinference: serving http://{listen}/v1");
                Ok(())
            })
            .await
        }
    }
}
