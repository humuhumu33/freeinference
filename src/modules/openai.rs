//! OpenAI compatible surface with receipts.
//!
//! Replaces hologram-live's built-in `openai-compat` module. Requests are
//! validated against the published OpenAI schema through `async-openai`'s
//! types. Responses are the standard chat completion object. Two fields the
//! spec always had are made true: `system_fingerprint` is the resolved model
//! κ joined with the engine κ, and the `x-hologram-receipt` header names a
//! stored, signed receipt. When the model does not resolve in the local
//! catalog, neither is present, because neither would be true.

use crate::receipt::{
    did_holo, kappa_of, Bound, Memo, Receipt, ReceiptSigner, ANSWER_KIND, ANSWER_MEDIA_TYPE,
    MEMO_IRI, MEMO_KIND, MEMO_MEDIA_TYPE, RECEIPT_KIND, RECEIPT_MEDIA_TYPE,
};
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
use hologram_live::protocol::ObjectMetadata;
use serde_json::{json, Value};
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio_stream::StreamExt;

pub const MODULE_ID: &str = "ai.freeinference.openai";
pub const RECEIPT_HEADER: &str = "x-hologram-receipt";
pub const STREAM_HEADER: &str = "x-hologram-stream";
/// Present only when the answer was served from a stored receipt and its
/// answer bytes, with no engine run; the value is the memo object's κ.
pub const REUSE_HEADER: &str = "x-hologram-reuse";

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
    let params = json!({ "max_tokens": max_tokens, "temperature": temperature, "seed": seed });
    let prompt_kappa = kappa_of(prompt.as_bytes());
    let params_kappa = kappa_of(params.to_string().as_bytes());
    let created = unix_seconds();

    // Reuse before execution. A sealed answer to the same model, prompt and
    // parameters is served from the store, whichever machine sealed it. The
    // model may be named by its κ, which is how an answer sealed elsewhere is
    // asked for here without the model being present.
    let mut candidates = vec![model_name.clone()];
    if let Some(info) = resolved.as_ref() {
        candidates.push(info.id.clone());
    }
    if let Some(hit) = find_memo(&state, &candidates, &prompt_kappa, &params_kappa).await? {
        return Ok(respond(Reply {
            stream,
            model_name,
            created,
            text: hit.text,
            fingerprint: Some(hit.fingerprint),
            receipt_id: Some(hit.receipt_id),
            usage: None,
            stream_kind: "memo",
            reuse: Some(hit.memo_id),
        }));
    }

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
        let bound = Bound {
            model_kappa: model_kappa.clone(),
            engine_kappa: engine_kappa.clone(),
            prompt_kappa: prompt_kappa.clone(),
            params_kappa: params_kappa.clone(),
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
        let text = completion.text.clone();
        let mut model = vec![model_kappa.clone()];
        if let Some(info) = resolved.as_ref() {
            if info.id != model_kappa {
                model.push(info.id.clone());
            }
        }
        let memo = Memo {
            iri: MEMO_IRI.to_owned(),
            model,
            engine_kappa: engine_kappa.clone(),
            prompt_kappa,
            params_kappa,
            output_kappa: receipt.bound.output_kappa.clone(),
            receipt: String::new(),
        };
        let stored = tokio::task::spawn_blocking(move || -> Result<ObjectMetadata, LiveError> {
            let stored = registry.put_object(
                RECEIPT_KIND.to_owned(),
                RECEIPT_MEDIA_TYPE.to_owned(),
                None,
                &bytes,
            )?;
            registry.put_object(
                ANSWER_KIND.to_owned(),
                ANSWER_MEDIA_TYPE.to_owned(),
                None,
                text.as_bytes(),
            )?;
            let memo = Memo {
                receipt: stored.id.clone(),
                ..memo
            };
            let memo_bytes = serde_json::to_vec(&memo)
                .map_err(|error| LiveError::Protocol(format!("memo encode: {error}")))?;
            registry.put_object(
                MEMO_KIND.to_owned(),
                MEMO_MEDIA_TYPE.to_owned(),
                None,
                &memo_bytes,
            )?;
            Ok(stored)
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
    Ok(respond(Reply {
        stream,
        model_name,
        created,
        text: completion.text,
        fingerprint,
        receipt_id,
        usage,
        stream_kind: engine.stream_kind().header_value(),
        reuse: None,
    }))
}

/// A stored answer found for a request: its text, the fingerprint the
/// sealing engine wrote, the receipt object that seals it, and the memo that
/// indexed it.
pub struct Hit {
    pub text: String,
    pub fingerprint: String,
    pub receipt_id: String,
    pub memo_id: String,
}

/// Newest memo first whose model, prompt κ and params κ match. A memo is an
/// index, not evidence: the receipt it names must verify (a signed daemon
/// receipt over its canonical bytes, or a Q receipt whose `did:holo`
/// re-derives from its body), its output κ must be the memo's, and the
/// answer bytes must hash to that κ. Anything else is skipped.
pub async fn find_memo(
    state: &AppState,
    candidates: &[String],
    prompt_kappa: &str,
    params_kappa: &str,
) -> Result<Option<Hit>, LiveError> {
    let registry = state.registry().clone();
    let candidates = candidates.to_vec();
    let prompt_kappa = prompt_kappa.to_owned();
    let params_kappa = params_kappa.to_owned();
    tokio::task::spawn_blocking(move || -> Result<Option<Hit>, LiveError> {
        let mut memos = registry.list_objects(Some(MEMO_KIND))?;
        memos.sort_by_key(|meta| std::cmp::Reverse(meta.created_at_millis));
        for meta in memos {
            let Ok(object) = registry.get_object(&meta.id) else {
                continue;
            };
            let Ok(memo) = serde_json::from_slice::<Memo>(&object.bytes) else {
                continue;
            };
            if memo.prompt_kappa != prompt_kappa
                || memo.params_kappa != params_kappa
                || !memo.model.iter().any(|model| candidates.contains(model))
            {
                continue;
            }
            let Ok(sealed) = registry.get_object(&memo.receipt) else {
                continue;
            };
            let Some(fingerprint) =
                receipt_fingerprint(&sealed.metadata.kind, &sealed.bytes, &memo.output_kappa)
            else {
                continue;
            };
            let Ok(answer) = registry.get_object(&memo.output_kappa) else {
                continue;
            };
            if kappa_of(&answer.bytes) != memo.output_kappa {
                continue;
            }
            let Ok(text) = String::from_utf8(answer.bytes) else {
                continue;
            };
            return Ok(Some(Hit {
                text,
                fingerprint,
                receipt_id: memo.receipt.clone(),
                memo_id: meta.id,
            }));
        }
        Ok(None)
    })
    .await
    .map_err(|error| LiveError::Conflict(format!("join memo lookup: {error}")))?
}

/// Checks a stored receipt of either kind and returns the fingerprint it
/// carries, `model κ;engine κ`, or `None` when it does not verify or does
/// not seal the given output κ.
fn receipt_fingerprint(kind: &str, bytes: &[u8], output_kappa: &str) -> Option<String> {
    match kind {
        RECEIPT_KIND => {
            let receipt: Receipt = serde_json::from_slice(bytes).ok()?;
            if receipt.verify().is_err() || receipt.bound.output_kappa != output_kappa {
                return None;
            }
            Some(format!(
                "{};{}",
                receipt.bound.model_kappa, receipt.bound.engine_kappa
            ))
        }
        crate::modules::webgpu::Q_RECEIPT_KIND => {
            let receipt: Value = serde_json::from_slice(bytes).ok()?;
            if receipt["id"].as_str()? != did_holo(&receipt["body"])
                || kappa_of(receipt["text"].as_str()?.as_bytes()) != output_kappa
            {
                return None;
            }
            let used = &receipt["body"]["prov:used"];
            Some(format!(
                "{};{}",
                used["holo:model"].as_str().unwrap_or(""),
                used["holo:engine"].as_str().unwrap_or("")
            ))
        }
        _ => None,
    }
}

struct Reply {
    stream: bool,
    model_name: String,
    created: u64,
    text: String,
    fingerprint: Option<String>,
    receipt_id: Option<String>,
    usage: Option<Value>,
    stream_kind: &'static str,
    reuse: Option<String>,
}

/// One response shape for both paths, executed or reused: a chat completion
/// object, or the same content as emulated server sent events.
fn respond(reply: Reply) -> Response {
    let id = completion_id(reply.created, &reply.text);
    let mut response = if reply.stream {
        let chunk = |delta: Value, finish: Value| {
            json!({
                "id": id, "object": "chat.completion.chunk", "created": reply.created, "model": reply.model_name,
                "system_fingerprint": reply.fingerprint,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
            })
        };
        let frames = vec![
            chunk(json!({ "role": "assistant", "content": "" }), Value::Null),
            chunk(json!({ "content": reply.text }), Value::Null),
            chunk(json!({}), json!("stop")),
        ];
        let events = tokio_stream::iter(frames)
            .map(|frame| {
                Ok::<Event, std::convert::Infallible>(Event::default().data(frame.to_string()))
            })
            .chain(tokio_stream::once(Ok(Event::default().data("[DONE]"))));
        Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": reply.created,
            "model": reply.model_name,
            "system_fingerprint": reply.fingerprint,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": reply.text, "refusal": null },
                "logprobs": null,
                "finish_reason": "stop",
            }],
            "usage": reply.usage,
        }))
        .into_response()
    };
    let kind = if reply.stream && reply.reuse.is_none() {
        "emulated"
    } else {
        reply.stream_kind
    };
    set_headers(&mut response, reply.receipt_id.as_deref(), kind);
    if let Some(memo) = reply.reuse.as_deref() {
        if let Ok(value) = HeaderValue::from_str(memo) {
            response.headers_mut().insert(REUSE_HEADER, value);
        }
    }
    response
}

#[utoipa::path(get, path = "/v1/models", responses((status = 200, description = "Models in the local catalog")))]
async fn list_models(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    let catalog = state.models().clone();
    let models = tokio::task::spawn_blocking(move || catalog.list())
        .await
        .map_err(|error| ApiError::server(format!("join model listing: {error}")))??;
    let mut data: Vec<Value> = models
        .into_iter()
        .map(|model| {
            json!({ "id": model.name, "object": "model", "created": model.created_at_millis / 1000, "owned_by": "local" })
        })
        .collect();
    // The browser engine's models, answered here from receipts and executed
    // in a WebGPU browser. Listed so any OpenAI client can name them.
    for (id, _) in crate::modules::webgpu::MODELS {
        data.push(json!({ "id": id, "object": "model", "created": 0, "owned_by": "browser" }));
    }
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
pub fn render_prompt(messages: &[Value]) -> String {
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
