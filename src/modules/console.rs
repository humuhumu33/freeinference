//! Local console: a Dashboard and a Playground served by the daemon itself.
//!
//! Static pages, no build step, no framework. They call the same local
//! surfaces any client uses: `/v1/models`, `/v1/chat/completions`,
//! `/api/v1/objects`, `/healthz`. Served from the daemon so there is no
//! CORS, no key, and no network. Removing this capability means deleting
//! this file, its `console/` directory, and one registration line.

use axum::http::header;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use hologram_live::app::AppState;
use hologram_live::module::{LiveModule, ModuleDescriptor};

pub const MODULE_ID: &str = "ai.freeinference.console";

static DESCRIPTOR: ModuleDescriptor = ModuleDescriptor {
    id: MODULE_ID,
    name: "Local console",
    version: env!("CARGO_PKG_VERSION"),
    dependencies: &[super::openai::MODULE_ID, "dev.hologram.live.kappa-registry"],
    operations: &[],
};

const DASHBOARD: &str = include_str!("console/dashboard.html");
const PLAYGROUND: &str = include_str!("console/playground.html");
const STYLES: &str = include_str!("console/console.css");

pub struct ConsoleModule;

impl LiveModule for ConsoleModule {
    fn descriptor(&self) -> &'static ModuleDescriptor {
        &DESCRIPTOR
    }

    fn router(&self) -> Router<AppState> {
        Router::new()
            .route("/dashboard", get(|| async { Html(DASHBOARD) }))
            .route("/playground", get(|| async { Html(PLAYGROUND) }))
            .route("/console.css", get(styles))
    }
}

async fn styles() -> Response {
    ([(header::CONTENT_TYPE, "text/css; charset=utf-8")], STYLES).into_response()
}
