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

async fn load(state: &AppState, kappa: &str) -> Result<Receipt, Failure> {
    let registry = state.registry().clone();
    let id = kappa.to_owned();
    let object = tokio::task::spawn_blocking(move || registry.get_object(&id))
        .await
        .map_err(|error| Failure(StatusCode::INTERNAL_SERVER_ERROR, error.to_string()))??;
    if object.metadata.kind != crate::receipt::RECEIPT_KIND {
        return Err(Failure(
            StatusCode::NOT_FOUND,
            format!("{kappa} is not a receipt"),
        ));
    }
    serde_json::from_slice(&object.bytes).map_err(|error| {
        Failure(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("receipt parse: {error}"),
        )
    })
}

async fn get_receipt(
    State(state): State<AppState>,
    Path(kappa): Path<String>,
) -> Result<Json<Value>, Failure> {
    let receipt = load(&state, &kappa).await?;
    let mut value = serde_json::to_value(&receipt)
        .map_err(|e| Failure(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
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
    let receipt = load(&state, &kappa).await?;
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
    let verdict = engine.verify_answer(&receipt.bound.answer_kappa).await?;
    let verdict: Value = serde_json::from_str(&verdict).unwrap_or(Value::String(verdict));
    let verified = verdict["verified"].as_bool().unwrap_or(false);
    Ok(Json(json!({
        "id": kappa, "verified": verified, "integrity": true,
        "answer_kappa": receipt.bound.answer_kappa, "replay": verdict,
    })))
}
