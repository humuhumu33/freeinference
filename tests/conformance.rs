//! One test per registered claim, named `conformance_<id>`. The register is
//! `model/ids.toml`; the scenarios are under `features/suites/`.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use freeinference::modules::openai::{engine_kappa, MODULE_ID, RECEIPT_HEADER};
use freeinference::receipt::{kappa_of, Receipt};
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
    config.validate().expect("valid config");
    let state = AppState::build_with_modules(config, tracing(), freeinference::extra_modules())
        .await
        .expect("state builds");
    Daemon { state, _dir: dir }
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
    let app = daemon
        .state
        .module_router()
        .with_state(daemon.state.clone());
    let body = json!({ "model": model, "messages": [{ "role": "user", "content": "hello" }] });
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
