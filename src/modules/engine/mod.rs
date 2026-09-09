//! Deterministic engine identity. The engine itself lives in the separate
//! `engine/` package (`freeinference-holo`), because its substrate is pinned
//! to a revision that is private today; keeping it out of this manifest
//! keeps this crate buildable by anyone. This module holds what both
//! binaries share: the engine name and its build identity.

use hologram_live::config::InferenceConfig;
use hologram_live::inference::{EngineFactory, InferenceEngine};
use hologram_live::models::ModelCatalog;
use std::sync::Arc;

/// Engine name in `inference.engine`.
pub const ENGINE_NAME: &str = "holo";

/// The exact hologram-ai revision the engine package compiles in. Kept
/// here next to `build_kappa` and mirrored in `engine/Cargo.toml`; the two
/// must move together.
pub const HOLOGRAM_AI_REV: &str = "50ebbfe3dd9b558cf238f3f52784233e7fe1c254";

/// Build identity of the deterministic engine: the address of the pinned
/// hologram-ai revision and the active quantization tier.
pub fn build_kappa() -> String {
    let tier = std::env::var("HOLO_QUANT_TIER").unwrap_or_else(|_| "int8".to_owned());
    crate::receipt::kappa_of(format!("hologram-ai/{HOLOGRAM_AI_REV}/{tier}").as_bytes())
}

/// The factory a binary built without the engine package supplies: it
/// refuses `inference.engine = "holo"` with the reason, never silently
/// falls back to another engine.
pub fn refusing_factory(config: &InferenceConfig) -> Option<EngineFactory> {
    if config.engine != ENGINE_NAME {
        return None;
    }
    Some(Box::new(
        |_: &InferenceConfig,
         _: Arc<ModelCatalog>|
         -> hologram_live::error::Result<Arc<dyn InferenceEngine>> {
            Err(hologram_live::error::LiveError::Config(
                "inference.engine = \"holo\" needs the freeinference-holo binary (built from engine/)".to_owned(),
            ))
        },
    ))
}
