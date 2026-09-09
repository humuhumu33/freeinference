//! OpenAI compatible surface with receipts.
//!
//! Replaces hologram-live's built-in `openai-compat` module. Requests are
//! validated against the published OpenAI schema through `async-openai`'s
//! types. Responses are the standard chat completion object. Two fields the
//! spec always had are made true: `system_fingerprint` is the resolved model
//! κ joined with the engine κ, and the `x-hologram-receipt` header names a
//! stored, signed receipt. When the model does not resolve in the local
//! catalog, neither is present, because neither would be true.

use crate::receipt::{kappa_of, Bound, ReceiptSigner, RECEIPT_KIND, RECEIPT_MEDIA_TYPE};
use async_openai::types::chat::CreateChatCompletionRequest;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hologram_live::app::AppState;
use hologram_live::error::LiveError;
use hologram_live::inference::CompletionRequest;
use hologram_live::module::{LiveModule, ModuleContext, ModuleDescriptor, ModuleStartFuture};
use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;

pub const MODULE_ID: &str = "ai.freeinference.openai";
pub const RECEIPT_HEADER: &str = "x-hologram-receipt";
pub const STREAM_HEADER: &str = "x-hologram-stream";

static DESCRIPTOR: ModuleDescriptor = ModuleDescriptor {
    id: MODULE_ID,
    name: "OpenAI compatible API with receipts",
    version: env!("CARGO_PKG_VERSION"),
    dependencies: &[
        "dev.hologram.live.inference",
        "dev.hologram.live.kappa-registry",
    ],
    operations: &[],
};

static SIGNER: OnceLock<ReceiptSigner> = OnceLock::new();

pub struct OpenAiModule;

impl LiveModule for OpenAiModule {
    fn descriptor(&self) -> &'static ModuleDescriptor {
        &DESCRIPTOR
    }

    fn router(&self) -> Router<AppState> {
        Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/models", get(list_models))
    }

    fn openapi(&self) -> utoipa::openapi::OpenApi {
        <ApiDoc as utoipa::OpenApi>::openapi()
    }

    fn start<'a>(&'a self, context: &'a ModuleContext) -> ModuleStartFuture<'a> {
        let path = context.data_dir().join("receipts").join("signing.key");
        Box::pin(async move {
            let signer = ReceiptSigner::load_or_create(&path)
                .map_err(|error| LiveError::Config(format!("receipt signing key: {error}")))?;
            let _ = SIGNER.set(signer);
            Ok(())
        })
    }
}

#[derive(utoipa::OpenApi)]
#[openapi(paths(chat_completions, list_models))]
struct ApiDoc;

/// Error envelope in OpenAI's shape.
pub struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl From<LiveError> for ApiError {
    fn from(error: LiveError) -> Self {
        Self::server(error.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "message": self.message, "type": self.kind, "param": null, "code": null } });
        (self.status, Json(body)).into_response()
    }
}

#[utoipa::path(
    post,
    path = "/v1/chat/completions",
    responses(
        (status = 200, description = "A chat completion object. `system_fingerprint` is the resolved model κ, a semicolon, and the engine κ, present only when the model resolves in the local catalog. The `x-hologram-receipt` header names the stored signed receipt under the same condition."),
        (status = 400, description = "Invalid request")
    )
)]
async fn chat_completions(
    State(state): State<AppState>,
    Json(raw): Json<Value>,
) -> Result<Response, ApiError> {
    // Validate against the published request schema, then read the fields
    // we use from the JSON so this module does not track every field name.
    let _typed: CreateChatCompletionRequest = serde_json::from_value(raw.clone())
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    let messages = raw["messages"].as_array().cloned().unwrap_or_default();
    if messages.is_empty() {
        return Err(ApiError::invalid("messages must not be empty"));
    }
    let requested = raw["model"].as_str().unwrap_or("").trim().to_owned();
    let stream = raw["stream"].as_bool().unwrap_or(false);
    let engine = state.chat().engine().clone();
    let default_model = state.config().inference.default_model.clone();
    let model_name = if requested.is_empty() {
        default_model
    } else {
        requested
    };

    // Resolve the model in the local catalog. A hit yields the model κ, the
    // BLAKE3 id of the imported artifact record. A miss is not an error for
    // the echo engine, which accepts any name; it only means no fingerprint.
    let catalog = state.models().clone();
    let lookup = model_name.clone();
    let resolved = tokio::task::spawn_blocking(move || catalog.resolve(&lookup))
        .await
        .map_err(|error| ApiError::server(format!("join model lookup: {error}")))?
        .ok();

    let prompt = render_prompt(&messages);
    let max_tokens = raw["max_completion_tokens"]
        .as_u64()
        .or_else(|| raw["max_tokens"].as_u64())
        .and_then(|value| u32::try_from(value).ok());
    let temperature = raw["temperature"].as_f64().map(|value| value as f32);
    let seed = raw["seed"].as_u64();
    let completion = engine
        .complete(CompletionRequest {
            prompt: prompt.clone(),
            model: Some(model_name.clone()),
            max_tokens,
            temperature,
            seed,
            session_key: None,
        })
        .await?;

    let created = unix_seconds();
    let mut fingerprint = None;
    let mut receipt_id = None;
    // The model κ is the engine's root κ when the engine addresses its
    // weights (a manifest over config, tokenizer, every tensor and every
    // derived artifact); otherwise the catalog's record id.
    let model_kappa = completion
        .model_kappa
        .clone()
        .or_else(|| resolved.as_ref().map(|info| info.id.clone()));
    if let Some(model_kappa) = model_kappa {
        let engine_kappa = engine_kappa(engine.name());
        let params = json!({ "max_tokens": max_tokens, "temperature": temperature, "seed": seed });
        let bound = Bound {
            model_kappa: model_kappa.clone(),
            engine_kappa: engine_kappa.clone(),
            prompt_kappa: kappa_of(prompt.as_bytes()),
            params_kappa: kappa_of(params.to_string().as_bytes()),
            output_kappa: kappa_of(completion.text.as_bytes()),
            answer_kappa: completion.answer_kappa.clone().unwrap_or_default(),
        };
        let signer = SIGNER
            .get()
            .ok_or_else(|| ApiError::server("receipt signer not started"))?;
        let receipt = signer.seal(bound, created * 1000);
        let bytes =
            serde_json::to_vec(&receipt).map_err(|error| ApiError::server(error.to_string()))?;
        let registry = state.registry().clone();
        let stored = tokio::task::spawn_blocking(move || {
            registry.put_object(
                RECEIPT_KIND.to_owned(),
                RECEIPT_MEDIA_TYPE.to_owned(),
                None,
                &bytes,
            )
        })
        .await
        .map_err(|error| ApiError::server(format!("join receipt store: {error}")))??;
        fingerprint = Some(format!("{model_kappa};{engine_kappa}"));
        receipt_id = Some(stored.id);
    }

    let usage = completion.usage.map(|usage| {
        json!({
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total(),
        })
    });
    let id = completion_id(created, &completion.text);
    let body = if stream {
        let chunk = |delta: Value, finish: Value| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model_name,
                "system_fingerprint": fingerprint,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        };
        let frames = vec![
            chunk(json!({ "role": "assistant", "content": "" }), Value::Null),
            chunk(json!({ "content": completion.text }), Value::Null),
            chunk(json!({}), json!("stop")),
        ];
        let events = tokio_stream::iter(frames)
            .map(|frame| {
                Ok::<Event, std::convert::Infallible>(Event::default().data(frame.to_string()))
            })
            .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
        let mut response = Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response();
        set_headers(&mut response, receipt_id.as_deref(), "emulated");
        return Ok(response);
    } else {
        json!({
            "id": id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "system_fingerprint": fingerprint,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": completion.text, "refusal": null },
                "logprobs": null,
                "finish_reason": "stop",
            }],
            "usage": usage,
        })
    };
    let mut response = Json(body).into_response();
    set_headers(
        &mut response,
        receipt_id.as_deref(),
        engine.stream_kind().header_value(),
    );
    Ok(response)
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, description = "Models in the local catalog")))]
async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = state.models().clone();
    let models = tokio::task::spawn_blocking(move || catalog.list())
        .await
        .map_err(|error| ApiError::server(format!("join model listing: {error}")))??;
    let data: Vec<Value> = models
        .into_iter()
        .map(|model| {
            json!({ "id": model.name, "object": "model", "created": model.created_at_millis / 1000, "owned_by": "local" })
        })
        .collect();
    Ok(Json(json!({ "object": "list", "data": data })))
}

fn set_headers(response: &mut Response, receipt: Option<&str>, stream: &'static str) {
    let headers = response.headers_mut();
    headers.insert(STREAM_HEADER, HeaderValue::from_static(stream));
    if let Some(id) = receipt {
        if let Ok(value) = HeaderValue::from_str(id) {
            headers.insert(RECEIPT_HEADER, value);
        }
    }
}

/// The engine κ names the engine this daemon runs. Until the deterministic
/// engine exposes a build identity, this is the address of its name and this
/// crate's version, which is exactly what it claims to be and nothing more.
pub fn engine_kappa(engine_name: &str) -> String {
    kappa_of(
        format!(
            "hologram-live/{engine_name}/freeinference/{}",
            env!("CARGO_PKG_VERSION")
        )
        .as_bytes(),
    )
}

/// Same `role: content` transcript shape hologram-live's chat module uses.
/// Array content keeps only text parts.
fn render_prompt(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|message| {
            let role = message["role"].as_str().unwrap_or("user");
            let content = match &message["content"] {
                Value::String(text) => text.clone(),
                Value::Array(parts) => parts
                    .iter()
                    .filter_map(|part| part["text"].as_str())
                    .collect::<Vec<_>>()
                    .join("\n"),
                _ => String::new(),
            };
            format!("{role}: {content}")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn completion_id(created: u64, text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&created.to_le_bytes());
    hasher.update(text.as_bytes());
    let hex = hasher.finalize().to_hex();
    format!("chatcmpl-{}", &hex[..24])
}
