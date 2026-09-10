//! Receipts: fetch a sealed answer by κ, and verify it by exact replay.
//!
//! `GET /v1/receipts/{kappa}` returns the stored receipt document.
//! `POST /v1/receipts/{kappa}/verify` checks the receipt's own signature and
//! κ, then asks the engine to replay the bound answer record. An engine
//! without a deterministic replay path refuses, and the refusal is the
//! answer: nothing is reported verified that was not replayed.

use crate::receipt::Receipt;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use hologram_live::app::AppState;
use hologram_live::error::LiveError;
use hologram_live::module::{LiveModule, ModuleDescriptor};
use serde_json::{json, Value};

pub const MODULE_ID: &str = "ai.freeinference.receipts";

static DESCRIPTOR: ModuleDescriptor = ModuleDescriptor {
    id: MODULE_ID,
    name: "Receipts",
    version: env!("CARGO_PKG_VERSION"),
    dependencies: &[super::openai::MODULE_ID, "dev.hologram.live.kappa-registry"],
    operations: &[],
};

pub struct ReceiptsModule;

impl LiveModule for ReceiptsModule {
    fn descriptor(&self) -> &'static ModuleDescriptor {
        &DESCRIPTOR
    }

    fn router(&self) -> Router<AppState> {
        Router::new()
            .route("/v1/receipts/{kappa}", get(get_receipt))
            .route("/v1/receipts/{kappa}/verify", post(verify_receipt))
    }
}

struct Failure(StatusCode, String);

impl IntoResponse for Failure {
    fn into_response(self) -> Response {
        (self.0, Json(json!({ "error": { "message": self.1 } }))).into_response()
    }
}

impl From<LiveError> for Failure {
    fn from(error: LiveError) -> Self {
        let status = match &error {
            LiveError::NotFound(_) => StatusCode::NOT_FOUND,
            LiveError::Capability(_) => StatusCode::NOT_IMPLEMENTED,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Failure(status, error.to_string())
    }
}

/// A stored receipt of either kind: the daemon's signed receipt, or the Q
/// receipt the browser engine sealed.
enum Loaded {
    Signed(Box<Receipt>),
    Q(Value),
}

async fn load(state: &AppState, kappa: &str) -> Result<Loaded, Failure> {
    let registry = state.registry().clone();
    let id = kappa.to_owned();
    let object = tokio::task::spawn_blocking(move || registry.get_object(&id))
        .await
        .map_err(|error| Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;
    let parse = |error: serde_json::Error| {
        Failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("receipt parse: {error}"),
        )
    };
    match object.metadata.kind.as_str() {
        crate::receipt::RECEIPT_KIND => Ok(Loaded::Signed(Box::new(
            serde_json::from_slice(&object.bytes).map_err(parse)?,
        ))),
        crate::modules::webgpu::Q_RECEIPT_KIND => Ok(Loaded::Q(
            serde_json::from_slice(&object.bytes).map_err(parse)?,
        )),
        _ => Err(Failure(
            StatusCode::NOT_FOUND,
            format!("{kappa} is not a receipt"),
        )),
    }
}

async fn get_receipt(
    State(state): State<AppState>,
    Path(kappa): Path<String>,
) -> Result<Json<Value>, Failure> {
    let mut value = match load(&state, &kappa).await? {
        Loaded::Signed(receipt) => serde_json::to_value(&receipt)
            .map_err(|e| Failure(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?,
        Loaded::Q(receipt) => receipt,
    };
    value["id"] = json!(kappa);
    Ok(Json(value))
}

/// Verdict shape: `verified` is true only when the engine replayed the bound
/// answer record and the bytes matched. `integrity` reports the receipt's own
/// signature and κ check, which happens first and independently.
async fn verify_receipt(
    State(state): State<AppState>,
    Path(kappa): Path<String>,
) -> Result<Json<Value>, Failure> {
    let receipt = match load(&state, &kappa).await? {
        Loaded::Signed(receipt) => *receipt,
        Loaded::Q(receipt) => {
            // Integrity: the did:holo must re-derive from the body. Replay
            // is the browser's: the Playground re-derives on a WebGPU device.
            let integrity =
                receipt["id"].as_str().unwrap_or("") == crate::receipt::did_holo(&receipt["body"]);
            return Ok(Json(json!({
                "id": kappa, "verified": false, "integrity": integrity,
                "reason": if integrity {
                    "re-derive in the Playground on a WebGPU device"
                } else {
                    "receipt id does not re-derive from its body"
                },
            })));
        }
    };
    if let Err(reason) = receipt.verify() {
        return Ok(Json(json!({
            "id": kappa, "verified": false, "integrity": false, "reason": reason,
        })));
    }
    if receipt.bound.answer_kappa.is_empty() {
        return Ok(Json(json!({
            "id": kappa, "verified": false, "integrity": true,
            "reason": "the engine that produced this answer sealed no replayable record",
        })));
    }
    let engine = state.chat().engine().clone();
    // Integrity holds on any machine; replay needs the engine and the model
    // that sealed the record. A machine that has neither says so, it does
    // not fail.
    let verdict = match engine.verify_answer(&receipt.bound.answer_kappa).await {
        Ok(verdict) => verdict,
        Err(error) => {
            return Ok(Json(json!({
                "id": kappa, "verified": false, "integrity": true,
                "answer_kappa": receipt.bound.answer_kappa,
                "reason": format!("replay is not possible on this machine: {error}"),
            })));
        }
    };
    let verdict: Value = serde_json::from_str(&verdict).unwrap_or(Value::String(verdict));
    let verified = verdict["verified"].as_bool().unwrap_or(false);
    Ok(Json(json!({
        "id": kappa, "verified": verified, "integrity": true,
        "answer_kappa": receipt.bound.answer_kappa, "replay": verdict,
    })))
}
