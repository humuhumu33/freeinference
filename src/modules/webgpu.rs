//! The engine: Hologram Q's WebGPU ternary engine, served by the daemon and
//! run in the browser.
//!
//! A pinned snapshot of `hologram-apps/apps/q` (MIT), the closure of
//! `core/q-brain-fast.mjs`: loader, engine, per block verification, receipt
//! sealing, and the wasm tokenizer. Served under `/q/` so the Playground can
//! run natively ternary κ objects such as HOLOGRAMTECH/q-bitnet-2b entirely in
//! the browser, streaming blocks from Hugging Face and re-deriving each
//! block's κ before use. The snapshot's own hash list is `q/SNAPSHOT.txt`.
//!
//! Compute lives in the browser, the store lives in the daemon. The daemon
//! itself executes nothing: its `InferenceEngine` refuses and says where to
//! go. What the browser seals comes back through `/v1/webgpu/seal` as three
//! content addressed objects, the Q receipt, the answer bytes, and a memo,
//! so a repeated prompt from the Playground or from any OpenAI client is
//! served from the receipt with no execution, on this machine or on any
//! machine the objects are carried to.

use crate::modules::openai::{find_memo, render_prompt};
use crate::receipt::{
    did_holo, kappa_of, Memo, ANSWER_KIND, ANSWER_MEDIA_TYPE, MEMO_IRI, MEMO_KIND, MEMO_MEDIA_TYPE,
};
use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hologram_live::app::AppState;
use hologram_live::config::InferenceConfig;
use hologram_live::error::{LiveError, Result};
use hologram_live::inference::{
    Completion, CompletionRequest, EngineFactory, InferenceEngine, StreamKind,
};
use hologram_live::models::{ModelCatalog, ModelInfo};
use hologram_live::module::{LiveModule, ModuleDescriptor};
use serde_json::{json, Value};
use std::sync::Arc;

pub const MODULE_ID: &str = "ai.freeinference.webgpu";

/// Engine name in `inference.engine`.
pub const ENGINE_NAME: &str = "webgpu";

/// Kind of the receipt the browser engine seals (PROV-O, `did:holo`).
pub const Q_RECEIPT_KIND: &str = "q-receipt";
pub const Q_RECEIPT_MEDIA_TYPE: &str = "application/vnd.hologram.q-receipt+json";

/// Models the browser engine offers, as they appear in `/v1/models`. The
/// Playground shows them only when the browser has WebGPU.
pub const MODELS: &[(&str, &str)] = &[
    ("webgpu:BitNet", "BitNet 2B 4T, ternary, 0.69 GB"),
    ("webgpu:Qwen2.5-Coder", "Qwen2.5 Coder 7B, q3f, 3.4 GB"),
];

static DESCRIPTOR: ModuleDescriptor = ModuleDescriptor {
    id: MODULE_ID,
    name: "WebGPU engine (Hologram Q snapshot)",
    version: env!("CARGO_PKG_VERSION"),
    dependencies: &[super::console::MODULE_ID],
    operations: &[],
};

/// The snapshot, embedded. Paths are as the modules import each other.
static FILES: &[(&str, &[u8])] = &[
    ("SNAPSHOT.txt", include_bytes!("webgpu/q/SNAPSHOT.txt")),
    ("LICENSE", include_bytes!("webgpu/q/LICENSE")),
    (
        "core/q-brain-fast.mjs",
        include_bytes!("webgpu/q/core/q-brain-fast.mjs"),
    ),
    ("core/loader.js", include_bytes!("webgpu/q/core/loader.js")),
    ("core/engine.js", include_bytes!("webgpu/q/core/engine.js")),
    (
        "core/q-self.mjs",
        include_bytes!("webgpu/q/core/q-self.mjs"),
    ),
    (
        "core/semantic.js",
        include_bytes!("webgpu/q/core/semantic.js"),
    ),
    (
        "core/holo-stream-load.mjs",
        include_bytes!("webgpu/q/core/holo-stream-load.mjs"),
    ),
    (
        "core/kv-commons.mjs",
        include_bytes!("webgpu/q/core/kv-commons.mjs"),
    ),
    (
        "core/parts-cache.mjs",
        include_bytes!("webgpu/q/core/parts-cache.mjs"),
    ),
    ("core/kappa.js", include_bytes!("webgpu/q/core/kappa.js")),
    ("qvac-gpu.js", include_bytes!("webgpu/q/qvac-gpu.js")),
    (
        "qvac-ingest.mjs",
        include_bytes!("webgpu/q/qvac-ingest.mjs"),
    ),
    ("qvac-kdisk.mjs", include_bytes!("webgpu/q/qvac-kdisk.mjs")),
    ("qvac-2bit.mjs", include_bytes!("webgpu/q/qvac-2bit.mjs")),
    (
        "holo-load2bit.mjs",
        include_bytes!("webgpu/q/holo-load2bit.mjs"),
    ),
    (
        "holo-load-delta.mjs",
        include_bytes!("webgpu/q/holo-load-delta.mjs"),
    ),
    ("holo-delta.mjs", include_bytes!("webgpu/q/holo-delta.mjs")),
    (
        "holo-model-frame.mjs",
        include_bytes!("webgpu/q/holo-model-frame.mjs"),
    ),
    (
        "forge/gguf-forge-iq-dequant.mjs",
        include_bytes!("webgpu/q/forge/gguf-forge-iq-dequant.mjs"),
    ),
    (
        "forge/gguf-forge-iq-grids.mjs",
        include_bytes!("webgpu/q/forge/gguf-forge-iq-grids.mjs"),
    ),
    (
        "pkg/holospaces_web.js",
        include_bytes!("webgpu/q/pkg/holospaces_web.js"),
    ),
    (
        "pkg/holospaces_web_bg.wasm",
        include_bytes!("webgpu/q/pkg/holospaces_web_bg.wasm"),
    ),
];

pub struct WebGpuModule;

impl LiveModule for WebGpuModule {
    fn descriptor(&self) -> &'static ModuleDescriptor {
        &DESCRIPTOR
    }

    fn router(&self) -> Router<AppState> {
        Router::new()
            .route("/q/{*path}", get(serve))
            .route("/v1/webgpu/seal", post(seal))
            .route("/v1/webgpu/lookup", post(lookup))
    }
}

/// Where compute is. An OpenAI client that asks this daemon to execute gets
/// this message; a repeated prompt never reaches it, the memo answers first.
pub const WHERE_COMPUTE_IS: &str = "compute runs in the browser: open /playground on a WebGPU device and send the prompt there; answers sealed there are served here from their receipts";

/// The daemon's engine: it executes nothing and says so.
pub struct WebGpuEngine;

#[async_trait::async_trait]
impl InferenceEngine for WebGpuEngine {
    fn name(&self) -> &'static str {
        ENGINE_NAME
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<Completion> {
        Err(LiveError::Capability(WHERE_COMPUTE_IS.to_owned()))
    }

    async fn verify_answer(&self, _kappa: &str) -> Result<String> {
        Err(LiveError::Capability(
            "re-derive in the Playground on a WebGPU device".to_owned(),
        ))
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        Ok(Vec::new())
    }

    fn stream_kind(&self) -> StreamKind {
        StreamKind::Buffered
    }
}

/// Engine selection for the binary. `webgpu` is the engine; `holo`, the
/// removed CPU engine, is refused by name so an old configuration cannot
/// fall through to the echo engine unnoticed.
pub fn select_engine(config: &InferenceConfig) -> Option<EngineFactory> {
    match config.engine.as_str() {
        ENGINE_NAME => Some(Box::new(
            |_: &InferenceConfig, _: Arc<ModelCatalog>| -> Result<Arc<dyn InferenceEngine>> {
                Ok(Arc::new(WebGpuEngine))
            },
        )),
        "holo" => Some(Box::new(
            |_: &InferenceConfig, _: Arc<ModelCatalog>| -> Result<Arc<dyn InferenceEngine>> {
                Err(LiveError::Config(
                    "inference.engine = \"holo\" was removed; set inference.engine = \"webgpu\""
                        .to_owned(),
                ))
            },
        )),
        _ => None,
    }
}

fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next() {
        Some("js") | Some("mjs") => "text/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        _ => "text/plain; charset=utf-8",
    }
}

async fn serve(Path(path): Path<String>) -> Response {
    let Some((_, bytes)) = FILES.iter().find(|(name, _)| *name == path) else {
        return (StatusCode::NOT_FOUND, "not in the snapshot").into_response();
    };
    let mut response = Response::new(Body::from(*bytes));
    let headers = response.headers_mut();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(content_type(&path)),
    );
    // Immutable by construction: the snapshot changes only with the binary.
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=31536000, immutable"),
    );
    response
}

struct Failure(StatusCode, String);

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": { "message": self.1 } }))).into_response()
    }
}

impl From<LiveError> for Failure {
    fn from(error: LiveError) -> Self {
        Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
    }
}

/// The three κ inputs a memo is keyed by, computed from a request the same
/// way the OpenAI route computes them, so the two paths share memos.
struct Key {
    prompt: String,
    prompt_kappa: String,
    params_kappa: String,
}

fn key_of(raw: &Value) -> std::result::Result<Key, Failure> {
    let messages = raw["messages"].as_array().cloned().unwrap_or_default();
    if messages.is_empty() {
        return Err(Failure(
            StatusCode::BAD_REQUEST,
            "messages must not be empty".to_owned(),
        ));
    }
    let prompt = render_prompt(&messages);
    let max_tokens = raw["max_tokens"]
        .as_u64()
        .and_then(|value| u32::try_from(value).ok());
    let temperature = raw["temperature"].as_f64().map(|value| value as f32);
    let seed = raw["seed"].as_u64();
    let params = json!({ "max_tokens": max_tokens, "temperature": temperature, "seed": seed });
    Ok(Key {
        prompt_kappa: kappa_of(prompt.as_bytes()),
        params_kappa: kappa_of(params.to_string().as_bytes()),
        prompt,
    })
}

/// `POST /v1/webgpu/seal`: the browser hands over what it sealed. Body:
/// `{ "receipt": <Q receipt with id, body, text, turnIds, outIds, params>,
///   "model": "webgpu:BitNet", "messages": [...], "max_tokens", "temperature", "seed" }`.
/// The receipt's integrity is checked before anything is stored: its
/// `did:holo` must re-derive from its body.
async fn seal(
    State(state): State<AppState>,
    Json(raw): Json<Value>,
) -> std::result::Result<Json<Value>, Failure> {
    let receipt = raw["receipt"].clone();
    let id = receipt["id"].as_str().unwrap_or("").to_owned();
    if id.is_empty() || did_holo(&receipt["body"]) != id {
        return Err(Failure(
            StatusCode::BAD_REQUEST,
            "receipt id does not re-derive from its body".to_owned(),
        ));
    }
    let text = receipt["text"].as_str().unwrap_or("").to_owned();
    if text.is_empty() {
        return Err(Failure(
            StatusCode::BAD_REQUEST,
            "receipt carries no answer text".to_owned(),
        ));
    }
    let key = key_of(&raw)?;
    let used = &receipt["body"]["prov:used"];
    let mut model: Vec<String> = Vec::new();
    for candidate in [used["holo:model"].as_str(), raw["model"].as_str()]
        .into_iter()
        .flatten()
    {
        if !model.contains(&candidate.to_owned()) {
            model.push(candidate.to_owned());
        }
    }
    let engine_kappa = used["holo:engine"].as_str().unwrap_or("").to_owned();
    let output_kappa = kappa_of(text.as_bytes());
    let registry = state.registry().clone();
    let stored = tokio::task::spawn_blocking(move || -> Result<Value> {
        let receipt_bytes = serde_json::to_vec(&receipt)
            .map_err(|error| LiveError::Protocol(format!("receipt encode: {error}")))?;
        let stored = registry.put_object(
            Q_RECEIPT_KIND.to_owned(),
            Q_RECEIPT_MEDIA_TYPE.to_owned(),
            None,
            &receipt_bytes,
        )?;
        registry.put_object(
            ANSWER_KIND.to_owned(),
            ANSWER_MEDIA_TYPE.to_owned(),
            None,
            text.as_bytes(),
        )?;
        let memo = Memo {
            iri: MEMO_IRI.to_owned(),
            model,
            engine_kappa,
            prompt_kappa: key.prompt_kappa,
            params_kappa: key.params_kappa,
            output_kappa: output_kappa.clone(),
            receipt: stored.id.clone(),
        };
        let memo_bytes = serde_json::to_vec(&memo)
            .map_err(|error| LiveError::Protocol(format!("memo encode: {error}")))?;
        let memo = registry.put_object(
            MEMO_KIND.to_owned(),
            MEMO_MEDIA_TYPE.to_owned(),
            None,
            &memo_bytes,
        )?;
        Ok(json!({ "receipt": stored.id, "answer": output_kappa, "memo": memo.id }))
    })
    .await
    .map_err(|error| Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;
    Ok(Json(stored))
}

/// `POST /v1/webgpu/lookup`: the same body shape as a chat completion
/// request. Answers `{ "hit": false }` or the stored answer with its receipt,
/// so the Playground can serve a repeated prompt without touching the GPU.
async fn lookup(
    State(state): State<AppState>,
    Json(raw): Json<Value>,
) -> std::result::Result<Json<Value>, Failure> {
    let key = key_of(&raw)?;
    let candidates = vec![raw["model"].as_str().unwrap_or("").to_owned()];
    let hit = find_memo(&state, &candidates, &key.prompt_kappa, &key.params_kappa)
        .await
        .map_err(|error| Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))?;
    Ok(Json(match hit {
        None => json!({ "hit": false, "prompt_kappa": key.prompt_kappa }),
        Some(hit) => json!({
            "hit": true,
            "text": hit.text,
            "receipt": hit.receipt_id,
            "memo": hit.memo_id,
            "fingerprint": hit.fingerprint,
            "prompt_kappa": key.prompt_kappa,
            "prompt": key.prompt,
        }),
    }))
}
