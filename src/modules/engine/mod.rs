//! Deterministic engine: hologram-ai in process, every weight κ addressed.
//!
//! Not a `LiveModule`: the engine is supplied to hologram-live through its
//! engine factory seam. Selected when `inference.engine = "holo"`.

pub mod holo;

use hologram_live::config::InferenceConfig;
use hologram_live::inference::{EngineFactory, InferenceEngine};
use hologram_live::models::ModelCatalog;
use std::sync::Arc;

/// Engine name in `inference.engine`.
pub const ENGINE_NAME: &str = "holo";

/// The exact hologram-ai revision compiled into this binary. Kept in one
/// place next to the Cargo pin; the two must move together.
pub const HOLOGRAM_AI_REV: &str = "50ebbfe3dd9b558cf238f3f52784233e7fe1c254";

/// Build identity of the deterministic engine: the address of the pinned
/// hologram-ai revision and the active quantization tier.
pub fn build_kappa() -> String {
    let tier = std::env::var("HOLO_QUANT_TIER").unwrap_or_else(|_| "int8".to_owned());
    crate::receipt::kappa_of(format!("hologram-ai/{HOLOGRAM_AI_REV}/{tier}").as_bytes())
}

/// Returns the factory hologram-live calls when the configured engine is
/// ours, and `None` otherwise so hologram-live's own engines apply.
pub fn factory_for(config: &InferenceConfig) -> Option<EngineFactory> {
    if config.engine != ENGINE_NAME {
        return None;
    }
    Some(Box::new(
        |config: &InferenceConfig, catalog: Arc<ModelCatalog>| {
            Ok(Arc::new(holo::HoloEngine::new(config, catalog)?) as Arc<dyn InferenceEngine>)
        },
    ))
}
