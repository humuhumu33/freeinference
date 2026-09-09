#![forbid(unsafe_code)]

//! freeinference with the deterministic engine compiled in.

mod holo;

use hologram_live::config::InferenceConfig;
use hologram_live::inference::{EngineFactory, InferenceEngine};
use hologram_live::models::ModelCatalog;
use std::sync::Arc;

fn factory_for(config: &InferenceConfig) -> Option<EngineFactory> {
    if config.engine != freeinference::modules::engine::ENGINE_NAME {
        return None;
    }
    Some(Box::new(
        |config: &InferenceConfig,
         catalog: Arc<ModelCatalog>|
         -> hologram_live::error::Result<Arc<dyn InferenceEngine>> {
            Ok(Arc::new(holo::HoloEngine::new(config, catalog)?))
        },
    ))
}

#[tokio::main]
async fn main() {
    freeinference::cli::run(factory_for).await;
}
