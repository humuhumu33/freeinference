//! Free verified AI inference.
//!
//! One capability is one module. Each module lives in its own directory
//! under `modules/`, owns one config section, one conformance suite and one
//! feature file, and is registered in exactly one place: [`extra_modules`].
//! Removing a capability means deleting its directory and one line here.

#![forbid(unsafe_code)]

pub mod cli;
pub mod modules;
pub mod receipt;

use hologram_live::config::AppConfig;
use hologram_live::module::LiveModule;
use std::sync::Arc;

/// Built-in hologram-live module this crate replaces, because it owns the
/// same OpenAI routes. The two cannot be enabled together: the module
/// registry merges routers and a duplicate path panics at boot (RC-04).
const REPLACED: &[&str] = &["dev.hologram.live.openai-compat"];

/// Every module this crate contributes, in registration order.
pub fn extra_modules() -> Vec<Arc<dyn LiveModule>> {
    vec![
        Arc::new(modules::openai::OpenAiModule),
        Arc::new(modules::receipts::ReceiptsModule),
        Arc::new(modules::console::ConsoleModule),
    ]
}

/// Adjusts a hologram-live configuration so this crate's modules are enabled
/// and the built-ins they replace are not.
pub fn configure(config: &mut AppConfig) {
    config
        .modules
        .enabled
        .retain(|id| !REPLACED.contains(&id.as_str()));
    for module in extra_modules() {
        let id = module.descriptor().id.to_owned();
        if !config.modules.enabled.contains(&id) {
            config.modules.enabled.push(id);
        }
    }
}
