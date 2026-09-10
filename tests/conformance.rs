//! One test per registered claim, named `conformance_<id>`. The register is
//! `model/ids.toml`; the scenarios are under `features/suites/`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use freeinference::modules::openai::{engine_kappa, MODULE_ID, RECEIPT_HEADER, REUSE_HEADER};
use freeinference::receipt::{did_holo, kappa_of, Receipt};
use hologram_live::app::AppState;
use hologram_live::config::AppConfig;
use hologram_live::observability::TracingHandle;
use http_body_util::BodyExt;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;
use tempfile::TempDir;
use tower::ServiceExt;

fn repo(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

fn site_check() -> std::process::Output {
    Command::new("python")
        .arg(repo("scripts/check_site.py"))
        .arg(repo("site/index.html"))
        .output()
        .expect("python is available")
}

#[test]
fn conformance_si_01() {
    let output = site_check();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("no dashes"));
}

#[test]
fn conformance_si_02() {
    let output = site_check();
    assert!(String::from_utf8_lossy(&output.stdout).contains("headline exact"));
}

#[test]
fn conformance_si_03() {
    let output = site_check();
    let text = String::from_utf8_lossy(&output.stdout);
    let words: usize = text
        .split_whitespace()
        .find_map(|token| token.parse().ok())
        .expect("word count in output");
    assert!(words < 120, "{words} words");
}

fn tracing() -> TracingHandle {
    static HANDLE: OnceLock<TracingHandle> = OnceLock::new();
    HANDLE
        .get_or_init(|| {
            let config = AppConfig::default();
            hologram_live::observability::init(&config.tracing, &config.telemetry).expect("tracing")
        })
        .clone()
}

struct Daemon {
    state: AppState,
    _dir: TempDir,
}

async fn daemon() -> Daemon {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = AppConfig::default();
    config.paths.config_dir = dir.path().join("config");
    config.paths.data_dir = dir.path().join("data");
    config.paths.state_dir = dir.path().join("state");
    config.paths.cache_dir = dir.path().join("cache");
    freeinference::configure(&mut config);
    // The endpoint mechanics are exercised with hologram-live's echo engine
    // standing in for compute; the real engine is the browser's (WG-04).
    config.inference.engine = "echo".to_owned();
    config.validate().expect("valid config");
    let state = AppState::build_with_modules(config, tracing(), freeinference::extra_modules())
        .await
        .expect("state builds");
    Daemon { state, _dir: dir }
}

/// A daemon exactly as `freeinference serve` builds it: the WebGPU engine.
async fn daemon_as_shipped() -> Daemon {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut config = AppConfig::default();
    config.paths.config_dir = dir.path().join("config");
    config.paths.data_dir = dir.path().join("data");
    config.paths.state_dir = dir.path().join("state");
    config.paths.cache_dir = dir.path().join("cache");
    freeinference::configure(&mut config);
    assert_eq!(config.inference.engine, "webgpu");
    config.validate().expect("valid config");
    let engine = freeinference::modules::webgpu::select_engine(&config.inference);
    let state = AppState::build_with(config, tracing(), freeinference::extra_modules(), engine)
        .await
        .expect("state builds");
    Daemon { state, _dir: dir }
}

async fn post_json(
    daemon: &Daemon,
    path: &str,
    body: Value,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let app = daemon
        .state
        .module_router()
        .with_state(daemon.state.clone());
    let response = app
        .oneshot(
            Request::post(path)
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, value)
}

/// A minimal weightc style artifact directory the catalog will import.
fn fixture_model(root: &Path, name: &str) -> std::path::PathBuf {
    let dir = root.join(format!("{name}.wcpu"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("manifest.json"), br#"{"name":"tiny"}"#).unwrap();
    std::fs::write(dir.join("weights.bin"), [1u8, 2, 3, 4]).unwrap();
    dir
}

async fn chat(daemon: &Daemon, model: &str) -> (StatusCode, axum::http::HeaderMap, Value) {
    chat_prompt(daemon, model, "hello").await
}

async fn chat_prompt(
    daemon: &Daemon,
    model: &str,
    prompt: &str,
) -> (StatusCode, axum::http::HeaderMap, Value) {
    let app = daemon
        .state
        .module_router()
        .with_state(daemon.state.clone());
    let body = json!({ "model": model, "messages": [{ "role": "user", "content": prompt }] });
    let response = app
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, value)
}

#[tokio::test]
async fn conformance_rc_01() {
    let d = daemon().await;
    let model = fixture_model(d._dir.path(), "tiny");
    let info = d.state.models().import(&model).expect("import");
    let (status, _, body) = chat(&d, &info.name).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let expected = format!("{};{}", info.id, engine_kappa(d.state.chat().engine_name()));
    assert_eq!(body["system_fingerprint"].as_str(), Some(expected.as_str()));
    assert!(info.id.starts_with("blake3:"));
}

#[tokio::test]
async fn conformance_rc_02() {
    let d = daemon().await;
    let model = fixture_model(d._dir.path(), "tiny");
    let info = d.state.models().import(&model).expect("import");
    let (_, headers, body) = chat(&d, &info.name).await;
    let id = headers
        .get(RECEIPT_HEADER)
        .expect("receipt header")
        .to_str()
        .unwrap()
        .to_owned();
    let object = d.state.registry().get_object(&id).expect("stored receipt");
    assert_eq!(
        kappa_of(&object.bytes),
        id,
        "object id is the hash of its bytes"
    );
    let receipt: Receipt = serde_json::from_slice(&object.bytes).expect("receipt json");
    receipt.verify().expect("signature and kappa verify");
    assert_eq!(receipt.bound.model_kappa, info.id);
    assert_eq!(
        receipt.bound.output_kappa,
        kappa_of(
            body["choices"][0]["message"]["content"]
                .as_str()
                .unwrap()
                .as_bytes()
        )
    );
}

#[tokio::test]
async fn conformance_rc_03() {
    let d = daemon().await;
    let (status, headers, body) = chat(&d, "not-in-catalog").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["system_fingerprint"].is_null());
    assert!(headers.get(RECEIPT_HEADER).is_none());
}

#[tokio::test]
async fn conformance_rc_04() {
    let d = daemon().await;
    assert!(d
        .state
        .module_info()
        .iter()
        .any(|module| module.id == MODULE_ID));
    let app = d.state.module_router().with_state(d.state.clone());
    let response = app
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

async fn page(daemon: &Daemon, path: &str) -> (StatusCode, String) {
    let app = daemon
        .state
        .module_router()
        .with_state(daemon.state.clone());
    let response = app
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

#[tokio::test]
async fn conformance_co_01() {
    let d = daemon().await;
    let (status, body) = page(&d, "/dashboard").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("Inferences"));
    assert!(body.contains("/api/v1/objects"));
    assert!(body.contains("kind === \"receipt\""));
    let (status, css) = page(&d, "/console.css").await;
    assert_eq!(status, StatusCode::OK);
    assert!(css.contains("--brand: #e93b01"));
}

#[tokio::test]
async fn conformance_co_02() {
    let d = daemon().await;
    let (status, body) = page(&d, "/playground").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("/v1/chat/completions"));
    assert!(body.contains("system_fingerprint"));
    assert!(body.contains("x-hologram-receipt"));
}

#[tokio::test]
async fn conformance_vf_01() {
    let d = daemon().await;
    let model = fixture_model(d._dir.path(), "tiny");
    let info = d.state.models().import(&model).expect("import");
    let (_, headers, _) = chat(&d, &info.name).await;
    let id = headers
        .get(RECEIPT_HEADER)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let (status, body) = page(&d, &format!("/v1/receipts/{id}")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let receipt: Receipt = serde_json::from_str(&body).expect("receipt json");
    receipt.verify().expect("signature and kappa verify");
    assert_eq!(receipt.bound.model_kappa, info.id);
}

#[tokio::test]
async fn conformance_vf_02() {
    let d = daemon().await;
    let model = fixture_model(d._dir.path(), "tiny");
    let info = d.state.models().import(&model).expect("import");
    let (_, headers, _) = chat(&d, &info.name).await;
    let id = headers
        .get(RECEIPT_HEADER)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let app = d.state.module_router().with_state(d.state.clone());
    let response = app
        .oneshot(
            Request::post(format!("/v1/receipts/{id}/verify"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let verdict: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(verdict["integrity"], true);
    assert_eq!(verdict["verified"], false);
    assert!(verdict["reason"]
        .as_str()
        .unwrap()
        .contains("no replayable record"));
}

#[tokio::test]
async fn conformance_wg_01() {
    let d = daemon().await;
    let (status, list) = page(&d, "/q/SNAPSHOT.txt").await;
    assert_eq!(status, StatusCode::OK);
    for name in [
        "core/q-brain-fast.mjs",
        "core/engine.js",
        "holo-load2bit.mjs",
        "qvac-gpu.js",
        "pkg/holospaces_web_bg.wasm",
    ] {
        assert!(list.contains(name), "hash list names {name}");
    }
    let (status, js) = page(&d, "/q/core/q-brain-fast.mjs").await;
    assert_eq!(status, StatusCode::OK);
    assert!(js.contains("createFastQBrain"));
    let app = d.state.module_router().with_state(d.state.clone());
    let response = app
        .oneshot(
            Request::get("/q/pkg/holospaces_web_bg.wasm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get("content-type").unwrap(),
        "application/wasm"
    );
}

#[tokio::test]
async fn conformance_wg_02() {
    let d = daemon().await;
    let (status, body) = page(&d, "/playground").await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("navigator.gpu"));
    assert!(body.contains("/q/core/engine.js"));
    assert!(body.contains("buildReceipt"));
    assert!(body.contains("/v1/webgpu/seal"));
    assert!(body.contains("/v1/webgpu/lookup"));
}

fn count(daemon: &Daemon, kind: &str) -> usize {
    daemon
        .state
        .registry()
        .list_objects(Some(kind))
        .unwrap()
        .len()
}

#[tokio::test]
async fn conformance_rc_05() {
    let d = daemon().await;
    let model = fixture_model(d._dir.path(), "tiny");
    let info = d.state.models().import(&model).expect("import");
    let prompt = "What is the capital of France?";
    let (status, first, body1) = chat_prompt(&d, &info.name, prompt).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        first.get(REUSE_HEADER).is_none(),
        "first answer is executed"
    );
    assert_eq!(count(&d, "receipt"), 1);
    assert_eq!(count(&d, "answer"), 1);
    assert_eq!(count(&d, "memo"), 1);

    let (status, again, body2) = chat_prompt(&d, &info.name, prompt).await;
    assert_eq!(status, StatusCode::OK);
    let memo = again
        .get(REUSE_HEADER)
        .expect("reuse header")
        .to_str()
        .unwrap();
    assert!(memo.starts_with("blake3:"));
    assert_eq!(again.get(RECEIPT_HEADER), first.get(RECEIPT_HEADER));
    assert_eq!(again.get("x-hologram-stream").unwrap(), "memo");
    assert_eq!(
        body2["choices"][0]["message"]["content"],
        body1["choices"][0]["message"]["content"]
    );
    assert_eq!(body2["system_fingerprint"], body1["system_fingerprint"]);
    assert_eq!(count(&d, "receipt"), 1, "no new receipt was sealed");

    // A different prompt is not a hit.
    let (_, other, _) = chat_prompt(&d, &info.name, "And of Germany?").await;
    assert!(other.get(REUSE_HEADER).is_none());
    assert_eq!(count(&d, "receipt"), 2);
}

#[tokio::test]
async fn conformance_rc_06() {
    let a = daemon().await;
    let model = fixture_model(a._dir.path(), "tiny");
    let info = a.state.models().import(&model).expect("import");
    let prompt = "What is the capital of France?";
    let (_, headers, body) = chat_prompt(&a, &info.name, prompt).await;
    let receipt_id = headers
        .get(RECEIPT_HEADER)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    let fingerprint = body["system_fingerprint"].as_str().unwrap().to_owned();
    let model_kappa = fingerprint.split(';').next().unwrap().to_owned();

    // Another machine: a fresh daemon with an empty catalog. Carry the three
    // objects over its public objects API; each lands at the same κ.
    let b = daemon().await;
    assert!(b.state.models().list().unwrap().is_empty());
    for kind in ["memo", "receipt", "answer"] {
        for meta in a.state.registry().list_objects(Some(kind)).unwrap() {
            let object = a.state.registry().get_object(&meta.id).unwrap();
            let app = b.state.module_router().with_state(b.state.clone());
            let response = app
                .oneshot(
                    Request::post("/api/v1/objects")
                        .header("content-type", object.metadata.media_type)
                        .header("x-hologram-kind", kind)
                        .body(Body::from(object.bytes))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::CREATED);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            let stored: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                stored["id"], meta.id,
                "same bytes, same κ on the other machine"
            );
        }
    }

    let (status, again, body2) = chat_prompt(&b, &model_kappa, prompt).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        again.get(REUSE_HEADER).is_some(),
        "served from the carried receipt"
    );
    assert_eq!(
        again.get(RECEIPT_HEADER).unwrap().to_str().unwrap(),
        receipt_id
    );
    assert_eq!(body2["system_fingerprint"], fingerprint);
    assert_eq!(
        body2["choices"][0]["message"]["content"],
        body["choices"][0]["message"]["content"]
    );
    assert_eq!(
        count(&b, "receipt"),
        1,
        "nothing was sealed on the other machine"
    );

    // The carried receipt's integrity verifies there under the sealing key it carries.
    let app = b.state.module_router().with_state(b.state.clone());
    let response = app
        .oneshot(
            Request::post(format!("/v1/receipts/{receipt_id}/verify"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let verdict: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(verdict["integrity"], true);
}

#[tokio::test]
async fn conformance_wg_03() {
    let d = daemon().await;
    // What the browser engine seals: a Q receipt whose did:holo re-derives
    // from its PROV-O body. Shape as core/kappa.js writes it.
    let body = json!({
        "@type": "prov:Activity", "holo:kind": "verifiable-inference",
        "prov:used": { "holo:model": "did:holo:sha256:model", "holo:engine": "did:holo:sha256:engine",
                       "holo:prompt": "did:holo:sha256:prompt", "holo:context": "did:holo:sha256:ctx",
                       "holo:params": { "decode": "greedy-argmax", "maxTokens": 64 } },
        "prov:generated": { "holo:outputTokens": "did:holo:sha256:out", "holo:tokenCount": 4 },
    });
    let id = did_holo(&body);
    let messages = json!([{ "role": "user", "content": "What is the capital of France?" }]);
    let request = json!({ "model": "webgpu:BitNet", "messages": messages, "max_tokens": 64, "temperature": 0.7 });

    // A tampered receipt is refused before anything is stored.
    let mut forged = request.clone();
    forged["receipt"] = json!({ "id": id, "body": { "prov:used": { "holo:model": "did:holo:sha256:other" } }, "text": "Paris." });
    let (status, _, _) = post_json(&d, "/v1/webgpu/seal", forged).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(count(&d, "memo"), 0);

    let mut sealing = request.clone();
    sealing["receipt"] = json!({ "id": id, "body": body, "text": "The capital of France is Paris.", "turnIds": [1, 2], "outIds": [3, 4], "params": { "decode": "greedy-argmax" } });
    let (status, _, stored) = post_json(&d, "/v1/webgpu/seal", sealing).await;
    assert_eq!(status, StatusCode::OK, "{stored}");
    let receipt = stored["receipt"].as_str().unwrap().to_owned();
    assert_eq!(count(&d, "q-receipt"), 1);
    assert_eq!(count(&d, "answer"), 1);
    assert_eq!(count(&d, "memo"), 1);

    // The Playground asks before it generates.
    let (status, _, hit) = post_json(&d, "/v1/webgpu/lookup", request.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(hit["hit"], true);
    assert_eq!(hit["text"], "The capital of France is Paris.");
    assert_eq!(hit["receipt"], receipt);

    // Any OpenAI client gets the same answer from the same receipt.
    let (status, headers, reply) = post_json(&d, "/v1/chat/completions", request.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(headers.get(REUSE_HEADER).is_some());
    assert_eq!(
        headers.get(RECEIPT_HEADER).unwrap().to_str().unwrap(),
        receipt
    );
    assert_eq!(
        reply["choices"][0]["message"]["content"],
        "The capital of France is Paris."
    );
    assert_eq!(
        reply["system_fingerprint"],
        "did:holo:sha256:model;did:holo:sha256:engine"
    );

    // The receipt resolves and its integrity verifies; replay is the browser's.
    let (status, body) = page(&d, &format!("/v1/receipts/{receipt}")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body.contains("verifiable-inference"));
    let (status, _, verdict) =
        post_json(&d, &format!("/v1/receipts/{receipt}/verify"), json!({})).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(verdict["integrity"], true);
    assert_eq!(verdict["verified"], false);
    assert!(verdict["reason"].as_str().unwrap().contains("Playground"));

    // A different prompt is not a hit.
    let mut other = request.clone();
    other["messages"] = json!([{ "role": "user", "content": "And of Germany?" }]);
    let (_, _, miss) = post_json(&d, "/v1/webgpu/lookup", other).await;
    assert_eq!(miss["hit"], false);
}

#[tokio::test]
async fn conformance_wg_04() {
    let d = daemon_as_shipped().await;
    let (status, _, models) = {
        let app = d.state.module_router().with_state(d.state.clone());
        let response = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, (), serde_json::from_slice::<Value>(&bytes).unwrap())
    };
    assert_eq!(status, StatusCode::OK);
    let ids: Vec<&str> = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(ids.contains(&"webgpu:BitNet"), "{ids:?}");

    // Nothing is executed on the daemon: an unanswered prompt is refused
    // with the place compute lives, never echoed or invented.
    let request =
        json!({ "model": "webgpu:BitNet", "messages": [{ "role": "user", "content": "hello" }] });
    let (status, headers, reply) = post_json(&d, "/v1/chat/completions", request).await;
    assert!(!status.is_success(), "{status} {reply}");
    assert!(headers.get(REUSE_HEADER).is_none());
    assert!(
        reply["error"]["message"]
            .as_str()
            .unwrap()
            .contains("/playground"),
        "{reply}"
    );
    assert_eq!(count(&d, "receipt"), 0);
}
