//! In-process deterministic inference over hologram-ai.
//!
//! Ported from hologram-live-ip `unification/phase-1` (src/inference_holo.rs)
//! into freeinference as the engine module; the only edits are crate paths
//! and the `usage` field upstream added to `Completion`.

// Ported verbatim; style lints stay off so the file diffs cleanly against
// its origin. Behaviour is unchanged.
#![allow(
    dead_code,
    clippy::too_many_arguments,
    clippy::manual_clamp,
    clippy::drop_non_drop
)]
//!
//! `HoloEngine` implements [`InferenceEngine`](hologram_live::inference::InferenceEngine)
//! directly over hologram-ai's staged decode pipeline — the same machinery the
//! hologram-ai web app runs in wasm, here on the native host: the model's
//! safetensors are content-addressed into a κ-store once, stages compile on
//! demand, and each conversation holds a [`DecodeSession`] whose carried K/V
//! makes a warm turn cost only its novel tokens (the session rewinds to the
//! common prefix of the new transcript and its realized history).
//!
//! Threading: the daemon is async, the pipeline is synchronous, `!Send`
//! (Rc-based κ-store sharing), and owns large state — so all model work lives
//! on one dedicated worker thread. Requests cross over a std mpsc channel and
//! answers return on tokio oneshots. The worker serializes generation: one
//! local host, one model; concurrency would only thrash the CPU.
//!
//! Templating: the models used here (SmolLM2, Qwen2.5) speak ChatML. The
//! daemon's transcript rendering (`user: …` lines, `chat::render_transcript`)
//! is parsed back into turns and re-rendered as ChatML. Per-family template
//! selection (read from the archive) is a Phase 3 concern, not wired yet.

use hologram_live::config::InferenceConfig;
use hologram_live::error::{LiveError, Result};
use hologram_live::inference::{Completion, CompletionRequest, InferenceEngine};
use hologram_live::models::{ModelCatalog, ModelInfo};
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::path::{Path, PathBuf};
use std::sync::mpsc as std_mpsc;
use std::sync::Arc;
use std::time::Instant;

use hologram_ai::commands::generate::{
    generate_stream_decode, generate_stream_speculative, GenConfig,
};
use hologram_ai::decode::DecodeSession;
use hologram_ai::engine::decode_bucket_for_turn;
use hologram_ai::materialize::DirKappaStore;
use hologram_ai::speculative::PromptLookupDrafter;
use hologram_ai::staged::{GrowableStagedSession, StagedRunner};
use hologram_ai::{DType, RopeSpec, SessionProvider};
use hologram_ai_tokenizer::{NativeTokenizer, Tokenizer};

const CHATML_STOP: &str = "<|im_end|>";
const CHATML_OPEN: &str = "<|im_start|>";
/// Bound on a single turn when the request does not carry `max_tokens`;
/// hologram-ai's own `None` means "the remaining context", which is far too
/// chatty a default for a chat turn.
const DEFAULT_MAX_TOKENS: usize = 256;
const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_TOP_K: usize = 40;
/// Speculative decode draft cap: the zero-weight prompt-lookup drafter
/// proposes up to this many tokens, verified in ONE `M = 1 + K` pass —
/// byte-identical to single-step decode at the same seed ("a pure speedup
/// whose size is the draft's acceptance rate"; upstream speculative.rs).
/// OPT-IN (HOLO_SPECULATIVE=on). The full ledger (PERF-1000-REPORT.md):
/// with lazy verify construction + engagement floor + acceptance
/// retirement, novel text decodes at parity with plain (27.5 vs 26.0
/// control) — but the per-turn-permanent retirement also forfeits the
/// repetitive-text win (27 vs 36 plain in-state), so there is no robust
/// measured win to justify default-on yet. Next: EMA-based adaptive
/// gating that can re-engage after retirement.
const SPECULATIVE_DRAFT: usize = 12;

fn speculative_enabled() -> bool {
    matches!(
        std::env::var("HOLO_SPECULATIVE").as_deref(),
        Ok("on") | Ok("1")
    )
}
/// Ceiling on decoder layers per stage archive. Native RAM is not the wasm
/// 4 GiB ceiling, so stages can be coarse; 8 keeps even a 1.5B model's
/// largest stage modest. The EFFECTIVE value is chosen per model by
/// [`chunking_layers_per_stage`] so the LM head always vocab-chunks.
const LAYERS_PER_STAGE: u64 = 8;

/// Layers-per-stage that guarantees the LM head chunks. The stage compiler
/// partitions the head into vocab-row chunks of at most one stage's elements
/// (`head_chunk_rows = layer_elems · L / hidden`); a head that fits ONE chunk
/// is emitted whole and unranged, which keeps it OUT of the int8 tier — it
/// then runs as a wide-BF16 matmul on a (near-)serial kernel. Measured on
/// SmolLM2-135M at L=8: rows 49,168 ≥ vocab 49,152 → whole head → ~155% CPU
/// and 5–8 tok/s, while Qwen2.5-0.5B's chunked int8 head ran ~1400% CPU at
/// 36 tok/s. Choosing the largest L ≤ ceiling with `rows < vocab` moves every
/// model's head into the chunked int8 + parallel-GEMV regime.
fn chunking_layers_per_stage(config: &serde_json::Value) -> u64 {
    let get = |k: &str| config.get(k).and_then(|v| v.as_u64());
    let (Some(h), Some(i), Some(vocab)) = (
        get("hidden_size"),
        get("intermediate_size"),
        get("vocab_size"),
    ) else {
        return LAYERS_PER_STAGE;
    };
    let heads = get("num_attention_heads").unwrap_or(1).max(1);
    let head_dim = get("head_dim").unwrap_or(h / heads);
    let kv = get("num_key_value_heads").unwrap_or(heads) * head_dim;
    // Mirrors the stage compiler's `layer_param_elements`:
    // q + o (h·h each) + k + v (kv·h each) + gate/up/down (3·i·h) + 2 norms.
    let layer_elems = 2 * h * h + 2 * kv * h + 3 * i * h + 2 * h;
    if layer_elems == 0 {
        return LAYERS_PER_STAGE;
    }
    // rows(L) = L·layer_elems / h  <  vocab  ⇔  L ≤ (vocab·h − 1) / layer_elems
    let max_chunking = (vocab.saturating_mul(h).saturating_sub(1)) / layer_elems;
    if max_chunking == 0 {
        // Even one layer outweighs the head — chunking is impossible; the
        // head is small relative to a layer and the ceiling is fine.
        return LAYERS_PER_STAGE;
    }
    max_chunking.min(LAYERS_PER_STAGE).max(1)
}
/// Weight-quantization tier for the derived artifact set. int8 is the
/// full-quality default; int4 halves bytes-per-token (the decode is
/// DRAM-bandwidth-bound: measured 26 GB/s of a 62 GB/s ceiling at int8) at
/// a measurable quality cost. Env-selectable while the ladder is measured:
/// HOLO_QUANT_TIER=int4|int8.
fn engine_quant_tier() -> hologram_ai_common::lower::QuantTier {
    match std::env::var("HOLO_QUANT_TIER").as_deref() {
        Ok("int4") => hologram_ai_common::lower::QuantTier::Int4,
        _ => hologram_ai_common::lower::QuantTier::Int8,
    }
}

fn tier_tag(tier: hologram_ai_common::lower::QuantTier) -> &'static str {
    match tier {
        hologram_ai_common::lower::QuantTier::Int4 => "int4",
        _ => "int8",
    }
}

/// Stage-residency floor: the fixed native allowance when the machine's
/// memory cannot be read (the wasm tier lives under 4 GiB and cannot
/// afford even this).
const RESIDENCY_FLOOR_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Stage-residency budget, sized to the machine: 60% of physical RAM,
/// clamped to [8 GiB, 24 GiB]. A fixed 8 GiB starved the 4B model — the
/// decode pipeline (~4.2 GiB int8) plus the chunked-prefill seeder's
/// pipeline overflowed it, so every turn re-streamed both from disk
/// (measured: warm TTFT 11-13 s; seeder disabled: ~2 s). Override with
/// HOLO_RESIDENCY_GB.
fn residency_budget_bytes() -> u64 {
    use std::sync::OnceLock;
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        const GIB: u64 = 1024 * 1024 * 1024;
        if let Ok(gb) = std::env::var("HOLO_RESIDENCY_GB") {
            if let Ok(gb) = gb.parse::<u64>() {
                return gb.max(1) * GIB;
            }
        }
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let total = system.total_memory();
        if total == 0 {
            return RESIDENCY_FLOOR_BYTES;
        }
        // Bound by AVAILABLE memory too: on a busy machine, 60% of total
        // can promise RAM that other programs already hold — and when the
        // disk is too full for the pagefile to grow, that promise aborts
        // the process mid-decode (measured: three OOM aborts at ~0.4-0.8
        // GiB allocations while the disk sat under 2 GiB free).
        let available_cap = system
            .available_memory()
            .saturating_sub(4 * GIB)
            .max(4 * GIB);
        (total * 3 / 5).min(available_cap).clamp(4 * GIB, 24 * GIB)
    })
}
/// Session key used for sessionless (rendered-transcript) calls. One shared
/// session: the decode-plan rewind makes reuse correct for any prompt, and a
/// repeated transcript still lands on its warm prefix.
const ONE_SHOT_KEY: &str = "\u{0}one-shot";

enum WorkerMsg {
    Generate(WorkerRequest),
    Verify {
        kappa: String,
        reply: tokio::sync::oneshot::Sender<Result<String>>,
    },
}

struct WorkerRequest {
    prompt: String,
    /// Catalog model (id or name); `None` = the configured default.
    model: Option<String>,
    session_key: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    reply: tokio::sync::oneshot::Sender<Result<WorkerReply>>,
}

struct WorkerReply {
    text: String,
    model: String,
    /// Root κ of the model that produced this text.
    model_kappa: Option<String>,
    /// κ of the canonical answer record (see `Completion::answer_kappa`).
    answer_kappa: Option<String>,
    /// Decode rate over the generated tokens after the first one.
    tokens_per_second: Option<f64>,
    /// Prefill cost: time from generation start to the first decoded token.
    ttft_millis: Option<u64>,
}

/// In-process hologram-ai engine. See the module docs for the shape.
pub struct HoloEngine {
    tx: std_mpsc::Sender<WorkerMsg>,
    catalog: Arc<ModelCatalog>,
}

impl HoloEngine {
    pub fn new(config: &InferenceConfig, catalog: Arc<ModelCatalog>) -> Result<Self> {
        if config.default_model.trim().is_empty() {
            return Err(LiveError::Config(
                "inference.engine = \"holo\" requires inference.default_model \
                 (a catalog id or name from `hologram models list`)"
                    .to_owned(),
            ));
        }
        let (tx, rx) = std_mpsc::channel::<WorkerMsg>();
        // The staged pipeline is `!Send` (Rc-shared κ-store), so the Worker
        // is BORN on its thread; only Send construction inputs cross over.
        let worker_catalog = catalog.clone();
        let default_model = config.default_model.clone();
        let max_sessions = config.max_resident_sessions.max(1);
        std::thread::Builder::new()
            .name("holo-engine".to_owned())
            .spawn(move || {
                let mut worker = Worker {
                    catalog: worker_catalog,
                    default_model,
                    max_sessions,
                    loaded: HashMap::new(),
                    sessions: HashMap::new(),
                    recency: Vec::new(),
                };
                // W1a pre-warm: touch the expensive bytes BEFORE the first
                // request. Skipped instantly if a request is already queued.
                worker.prewarm_largest(&rx);
                worker.run(rx)
            })
            .map_err(|error| {
                LiveError::Io(format!("failed to spawn holo engine worker: {error}"))
            })?;
        Ok(Self { tx, catalog })
    }
}

#[tonic::async_trait]
impl InferenceEngine for HoloEngine {
    fn name(&self) -> &'static str {
        "holo"
    }

    fn supports_sessions(&self) -> bool {
        true
    }

    async fn complete(&self, request: CompletionRequest) -> Result<Completion> {
        let started = Instant::now();
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WorkerMsg::Generate(WorkerRequest {
                prompt: request.prompt,
                model: request.model,
                session_key: request.session_key,
                max_tokens: request.max_tokens,
                temperature: request.temperature,
                seed: request.seed,
                reply: reply_tx,
            }))
            .map_err(|_| LiveError::Transport("holo engine worker is gone".to_owned()))?;
        let reply = reply_rx
            .await
            .map_err(|_| LiveError::Transport("holo engine dropped the request".to_owned()))??;
        Ok(Completion {
            text: reply.text,
            model: reply.model,
            model_kappa: reply.model_kappa,
            answer_kappa: reply.answer_kappa,
            tokens_per_second: reply.tokens_per_second,
            elapsed_millis: started.elapsed().as_millis() as u64,
            usage: None,
            ttft_millis: reply.ttft_millis,
            // The native staged pipeline executes on the host CPU, on this
            // machine — stated per answer so the UI never has to guess.
            device: Some("cpu".to_owned()),
            locality: Some("local".to_owned()),
        })
    }

    async fn verify_answer(&self, kappa: &str) -> Result<String> {
        let (reply_tx, reply_rx) = tokio::sync::oneshot::channel();
        self.tx
            .send(WorkerMsg::Verify {
                kappa: kappa.to_owned(),
                reply: reply_tx,
            })
            .map_err(|_| LiveError::Transport("holo engine worker is gone".to_owned()))?;
        reply_rx
            .await
            .map_err(|_| LiveError::Transport("holo engine dropped the request".to_owned()))?
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.catalog.list()
    }
}

/// Everything that lives on the dedicated model thread.
struct Worker {
    catalog: Arc<ModelCatalog>,
    default_model: String,
    max_sessions: usize,
    /// Resident models keyed by catalog id. Loading is lazy; models stay
    /// resident once used (native RAM is the budget — no eviction yet).
    loaded: HashMap<String, Loaded>,
    sessions: HashMap<String, SessionState>,
    /// Session keys, least recently used first.
    recency: Vec<String>,
}

struct Loaded {
    growable: GrowableStagedSession,
    rope: RopeSpec,
    context_len: u64,
    /// Approximate resident bytes of ONE materialized pipeline (int8 tier:
    /// ~1 byte per weight element) — the seeder fit-gate's input.
    weights_bytes: u64,
    tokenizer: ChatMlTokenizer,
    /// Catalog display name of the resident model.
    model_name: String,
    /// The unified κ-store directory, for durable answer records.
    store_dir: PathBuf,
    /// Root κ of the model: the content address of a canonical manifest
    /// binding every component κ (config, tokenizer, every tensor, every
    /// int8 artifact). One address names everything this model executes.
    model_kappa: String,
}

/// `Write` sink that timestamps the first decoded delta (TTFT).
struct MeteredSink {
    started: Instant,
    first_write: Option<Instant>,
}

impl MeteredSink {
    fn new() -> Self {
        Self {
            started: Instant::now(),
            first_write: None,
        }
    }

    fn ttft_millis(&self) -> Option<u64> {
        self.first_write
            .map(|t| t.duration_since(self.started).as_millis() as u64)
    }
}

impl std::io::Write for MeteredSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.first_write.is_none() && !buf.is_empty() {
            self.first_write = Some(Instant::now());
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Special-token-aware wrapper: hologram-ai's BPE `encode` treats special
/// tokens as plain text (they sub-tokenize, the model never sees real ChatML
/// control ids and so never emits a real eos). This wrapper splits the input
/// on the special-token strings present in the vocab and splices their ids in
/// directly — the same treatment HF fast tokenizers give `added_tokens`.
struct ChatMlTokenizer {
    inner: NativeTokenizer,
    /// `(content, id)`, longest content first so overlaps match greedily.
    specials: Vec<(String, u32)>,
}

impl ChatMlTokenizer {
    fn new(inner: NativeTokenizer) -> Self {
        let mut specials: Vec<(String, u32)> = [
            CHATML_OPEN,
            CHATML_STOP,
            "<|endoftext|>",
            "<think>",
            "</think>",
        ]
        .iter()
        .filter_map(|s| inner.token_to_id(s).map(|id| ((*s).to_owned(), id)))
        .collect();
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        Self { inner, specials }
    }
}

impl ChatMlTokenizer {
    /// Encode a plain-text segment WITHOUT the tokenizer's automatic bos/eos
    /// framing — a ChatML prompt supplies its own structure, and the inner
    /// `encode` would otherwise prepend bos to every spliced segment.
    fn encode_segment(&self, segment: &str) -> Vec<u32> {
        let mut ids = self.inner.encode(segment);
        if let Some(bos) = self.inner.bos_token_id() {
            if ids.first() == Some(&bos) {
                ids.remove(0);
            }
        }
        let eos = self.inner.eos_token_id();
        if ids.last() == Some(&eos) {
            ids.pop();
        }
        ids
    }
}

impl Tokenizer for ChatMlTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        let mut rest = text;
        'outer: while !rest.is_empty() {
            // The earliest special-token occurrence, longest content first.
            let hit = self
                .specials
                .iter()
                .filter_map(|(s, id)| rest.find(s.as_str()).map(|at| (at, s.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            match hit {
                Some((at, len, id)) => {
                    if at > 0 {
                        ids.extend(self.encode_segment(&rest[..at]));
                    }
                    ids.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    ids.extend(self.encode_segment(rest));
                    break 'outer;
                }
            }
        }
        ids
    }

    fn decode(&self, tokens: &[u32]) -> String {
        self.inner.decode(tokens)
    }

    fn eos_token_id(&self) -> u32 {
        self.inner.eos_token_id()
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id()
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        self.inner.id_to_token(id)
    }
}

struct SessionState {
    decode: Option<ResidentDecode>,
    /// The conversation rendered as ChatML so far, ending after the last
    /// assistant close. The next turn appends to it. Empty for the shared
    /// one-shot session (its prompt arrives fully rendered each call).
    chatml: String,
}

struct ResidentDecode {
    bucket: usize,
    session: DecodeSession<StagedRunner<'static>>,
    /// The `M = 1 + K` verify pipeline for speculative decode, over the same
    /// bucket. Built LAZILY on the drafter's first qualifying proposal
    /// (a resident verify pipeline taxes plain decode ~35-40% by residency
    /// interference alone) and cached here across turns once built.
    verify: Option<StagedRunner<'static>>,
}

impl Worker {
    fn run(mut self, rx: std_mpsc::Receiver<WorkerMsg>) {
        while let Ok(msg) = rx.recv() {
            match msg {
                WorkerMsg::Generate(request) => {
                    let outcome = self.serve(&request);
                    let _ = request.reply.send(outcome);
                }
                WorkerMsg::Verify { kappa, reply } => {
                    let _ = reply.send(self.verify(&kappa));
                }
            }
        }
    }

    /// Verify an answer record by EXACT REPLAY. The verifier reads and
    /// re-hashes the payload (a tampered record file fails the integrity
    /// check before anything executes - the regression the vLLM-era witness
    /// study demands), matches the recorded model root against a loadable
    /// model, re-executes the decode with the record's own prompt and
    /// parameters, and compares the bytes. Temperature-0 decode is
    /// byte-identical across substrates (measured), so a verified record is
    /// a machine-independent fact, not a local one.
    fn verify(&mut self, kappa: &str) -> Result<String> {
        // The unified store lives beside the model artifacts.
        let default = self.catalog.resolve(&self.default_model)?;
        let dir = self.catalog.artifact_dir(&default.id)?;
        let store_dir = dir
            .parent()
            .map(|p| p.join(".kappa-store"))
            .unwrap_or_else(|| dir.join(".kappa-store"));
        let path = store_dir.join(hologram_ai::materialize::kappa_file_name(kappa));
        let bytes = std::fs::read(&path).map_err(|_| {
            LiveError::NotFound(format!("answer record {kappa} is not in the store"))
        })?;
        // INTEGRITY: the payload must reproduce its own address.
        let derived = hologram_ai::materialize::kappa_of(&bytes);
        if derived != kappa {
            return Err(LiveError::Conflict(format!(
                "answer record {kappa} fails integrity: stored bytes hash to {derived} -                  the record has been tampered with"
            )));
        }
        let record: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|error| LiveError::Protocol(format!("answer record parse: {error}")))?;
        if record.get("kind").and_then(|v| v.as_str()) != Some("hologram-live/answer") {
            return Err(LiveError::Protocol(format!(
                "{kappa} is not an answer record"
            )));
        }
        let model_root = record["model_root"]
            .as_str()
            .ok_or_else(|| LiveError::Protocol("record missing model_root".to_owned()))?
            .to_owned();
        let prompt = record["prompt"]
            .as_str()
            .ok_or_else(|| LiveError::Protocol("record missing prompt".to_owned()))?
            .to_owned();
        let expected = record["text"]
            .as_str()
            .ok_or_else(|| LiveError::Protocol("record missing text".to_owned()))?
            .to_owned();
        let temperature = record["temperature"].as_f64().unwrap_or(0.0) as f32;
        let seed = record["seed"].as_u64().unwrap_or(0);
        let max_tokens = record["max_tokens"].as_u64().map(|n| n as usize);

        // Find the catalog model whose root matches the record's. Receipts
        // make each candidate load ~15 ms warm.
        let mut matched: Option<(String, String)> = None;
        for info in self.catalog.list()? {
            let key = self.ensure_loaded(Some(&info.name))?;
            if self.loaded.get(&key).map(|l| l.model_kappa.as_str()) == Some(model_root.as_str()) {
                matched = Some((key, info.name));
                break;
            }
        }
        let Some((model_key, model_name)) = matched else {
            return Err(LiveError::NotFound(format!(
                "no catalog model reproduces root {model_root}"
            )));
        };

        // Replay with the exact recorded parameters (the sampler is a pure
        // function of logits, position, temperature, top_k, seed).
        let cfg = GenConfig {
            max_tokens,
            temperature,
            top_k: Some(DEFAULT_TOP_K),
            stop: vec![CHATML_STOP.to_owned(), CHATML_OPEN.to_owned()],
            eos: None,
            seed,
        };
        let verify_key = format!("{model_key}\x1fverify");
        self.sessions.remove(&verify_key);
        self.session_entry(&verify_key);
        let outcome = self.generate(&model_key, &verify_key, &prompt, &cfg);
        self.sessions.remove(&verify_key);
        self.recency.retain(|k| k != &verify_key);
        let (got, _, _) = outcome?;

        let verified = got == expected;
        let divergence = if verified {
            None
        } else {
            Some(
                got.bytes()
                    .zip(expected.bytes())
                    .position(|(a, b)| a != b)
                    .unwrap_or_else(|| got.len().min(expected.len())),
            )
        };
        Ok(serde_json::json!({
            "verified": verified,
            "answer_kappa": kappa,
            "model_root": model_root,
            "model": model_name,
            "temperature": temperature,
            "expected_bytes": expected.len(),
            "replayed_bytes": got.len(),
            "first_divergence_byte": divergence,
        })
        .to_string())
    }

    /// Cold-start killer: the first decode on a large model pays ~13 s of
    /// first-touch byte streaming (measured, 4B). At startup the worker is
    /// idle, so warm the LARGEST catalog model that fits current available
    /// memory: load by receipt, build the one-shot decode session, and feed
    /// a single token so every stage materializes. Aborts before starting
    /// if a real request is already waiting; a failure only logs — warming
    /// is never load-bearing.
    fn prewarm_largest(&mut self, rx: &std_mpsc::Receiver<WorkerMsg>) {
        let started = Instant::now();
        let mut mem = sysinfo::System::new();
        mem.refresh_memory();
        let available = mem.available_memory();
        let candidate = self
            .catalog
            .list()
            .unwrap_or_default()
            .into_iter()
            // safetensors are bf16 (2 B/weight); the int8 pipeline is ~half.
            .filter(|info| info.size / 2 + 3 * (1u64 << 30) <= available)
            .max_by_key(|info| info.size)
            .map(|info| info.id.clone());
        let Some(model) = candidate else {
            tracing::info!("prewarm skipped: no catalog model fits available memory");
            return;
        };
        if !matches!(rx.try_recv(), Err(std_mpsc::TryRecvError::Empty)) {
            // A request beat us (or the channel closed): the user's turn
            // will pay its own warm exactly as before — never twice.
            return;
        }
        let outcome = (|| -> Result<()> {
            let model_key = self.ensure_loaded(Some(&model))?;
            let key = format!("{model_key}\u{1f}{ONE_SHOT_KEY}");
            self.session_entry(&key);
            let cfg = GenConfig {
                max_tokens: Some(1),
                temperature: 0.0,
                top_k: Some(1),
                stop: vec![],
                eos: None,
                seed: 0,
            };
            self.generate(&model_key, &key, "hi", &cfg).map(|_| ())
        })();
        match outcome {
            Ok(()) => tracing::info!(
                model = %model,
                elapsed_millis = started.elapsed().as_millis() as u64,
                "prewarm complete: first message will start warm"
            ),
            Err(error) => tracing::warn!(%error, "prewarm failed (non-fatal)"),
        }
    }

    /// The assistant-turn preamble for this model. A thinking-family model
    /// (its tokenizer knows `<think>`, e.g. Qwen3) is primed with the
    /// official empty think block so it answers directly — unprimed, greedy
    /// decode mimics turn markers as plain text and pollutes the reply.
    fn assistant_preamble(&self, model_key: &str) -> &'static str {
        let thinking = self
            .loaded
            .get(model_key)
            .is_some_and(|m| m.tokenizer.inner.token_to_id("<think>").is_some());
        if thinking {
            "assistant\n<think>\n\n</think>\n\n"
        } else {
            "assistant\n"
        }
    }

    fn serve(&mut self, request: &WorkerRequest) -> Result<WorkerReply> {
        let model_key = self.ensure_loaded(request.model.as_deref())?;
        let cfg = gen_config(request);
        let preamble = self.assistant_preamble(&model_key);

        let last_prompt: String;
        let (text, ttft_millis, tokens_per_second) = match request.session_key.as_deref() {
            // A conversation turn: the daemon sends only the raw new user
            // content (`supports_sessions`), the session carries the history.
            Some(key) => {
                // A conversation answered by a different model gets its own
                // session; switching models never replays foreign K/V.
                let key = format!("{model_key}{key}");
                let prompt = {
                    let state = self.session_entry(&key);
                    state.chatml.push_str(&format!(
                        "{CHATML_OPEN}user\n{}{CHATML_STOP}\n",
                        request.prompt
                    ));
                    format!("{}{CHATML_OPEN}{preamble}", state.chatml)
                };
                last_prompt = prompt.clone();
                let outcome = self.generate(&model_key, &key, &prompt, &cfg)?;
                let state = self.sessions.get_mut(&key).expect("session exists");
                state.chatml.push_str(&format!(
                    "{CHATML_OPEN}assistant\n{}{CHATML_STOP}\n",
                    outcome.0
                ));
                outcome
            }
            // Sessionless (the OpenAI/Ollama modules render the whole message
            // list into one transcript). The shared session's rewind makes
            // reuse correct; a growing transcript lands on its warm prefix.
            None => {
                let key = format!("{model_key}\x1f{ONE_SHOT_KEY}");
                let prompt = format!(
                    "{}{CHATML_OPEN}{preamble}",
                    transcript_to_chatml(&request.prompt)
                );
                self.session_entry(&key);
                last_prompt = prompt.clone();
                self.generate(&model_key, &key, &prompt, &cfg)?
            }
        };
        // Seal the ANSWER RECORD: one canonical document binding everything
        // that produced this text, content-addressed into the same unified
        // store as the model itself. At temperature 0 the decode is
        // byte-identical across substrates (measured), so this κ is a
        // machine-independent name for the answer.
        let answer_kappa = self.loaded.get(&model_key).and_then(|loaded| {
            let record = serde_json::json!({
                "kind": "hologram-live/answer",
                "version": 1,
                "model_root": loaded.model_kappa,
                "prompt": last_prompt,
                "temperature": cfg.temperature,
                "seed": if cfg.temperature > 0.0 { Some(cfg.seed) } else { None },
                "max_tokens": cfg.max_tokens,
                "text": text,
            });
            let store = DirKappaStore::new(&loaded.store_dir);
            match store.insert(record.to_string().as_bytes()) {
                Ok(kappa) => Some(kappa),
                Err(error) => {
                    tracing::warn!(%error, "answer record insert failed; answer unaddressed");
                    None
                }
            }
        });
        Ok(WorkerReply {
            text,
            model: self
                .loaded
                .get(&model_key)
                .map(|l| l.model_name.clone())
                .unwrap_or_else(|| "holo".to_owned()),
            model_kappa: self.loaded.get(&model_key).map(|l| l.model_kappa.clone()),
            answer_kappa,
            tokens_per_second,
            ttft_millis,
        })
    }

    /// Generate against the keyed session, (re)building its decode session
    /// when absent or when the turn needs a wider bucket. A rebuild loses the
    /// carried K/V — the next turn prefills cold, then carries again.
    /// Returns `(text, ttft_millis, tokens_per_second)`.
    fn generate(
        &mut self,
        model_key: &str,
        key: &str,
        prompt: &str,
        cfg: &GenConfig,
    ) -> Result<(String, Option<u64>, Option<f64>)> {
        let loaded = self.loaded.get_mut(model_key).expect("ensure_loaded ran");
        let prompt_len = loaded.tokenizer.encode(prompt).len().max(1);
        let max_new = cfg.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);
        let want = decode_bucket_for_turn(prompt_len, max_new, loaded.context_len as usize);

        let state = self.sessions.get_mut(key).expect("session_entry ran");
        let needs_build = match &state.decode {
            Some(resident) => resident.bucket < want,
            None => true,
        };
        if needs_build {
            // Drop the outgoing runner BEFORE building the wider one — the
            // staged pipeline's growth-residency law (decode_growth_residency).
            state.decode = None;
            let started = Instant::now();
            let runner = loaded.growable.decode_runner_for(want).map_err(|error| {
                LiveError::Protocol(format!("decode pipeline (bucket {want}): {error:#}"))
            })?;
            let mut session =
                DecodeSession::new(runner, loaded.rope.clone(), loaded.context_len)
                    .map_err(|error| LiveError::Protocol(format!("decode session: {error:#}")))?;
            let bucket = session.geometry().bucket;
            // Chunked prefill (row `chunked-prefill`): a seeder pipeline
            // processes the prompt several positions per pass instead of one —
            // the TTFT lever. Failure falls back to stepping, never an error.
            let seed_chunk = (hologram_ai::engine::geometric_window(1, loaded.context_len as usize)
                as u64)
                .min(bucket as u64);
            let seeder_enabled = !matches!(
                std::env::var("HOLO_PREFILL_SEED").as_deref(),
                Ok("off") | Ok("0")
            );
            // Fit gate: the seeder is a SECOND materialized pipeline. When
            // decode + seeder cannot both stay resident, the seeder turns a
            // one-time cost into a per-turn re-stream of the whole model —
            // strictly worse than stepping the prompt. Gate on the measured
            // weight footprint instead of hoping.
            // Both checks are load-bearing: the budget bounds what the
            // growable may keep resident; AVAILABLE memory bounds what this
            // moment can actually take — the seeder's second pipeline once
            // OOM-aborted the whole server when total-RAM math said yes but
            // 21 GiB of the machine was already spoken for.
            let mut mem = sysinfo::System::new();
            mem.refresh_memory();
            let available = mem.available_memory();
            let seeder_fits = loaded.weights_bytes.saturating_mul(2) + (1 << 30)
                <= residency_budget_bytes()
                && loaded.weights_bytes + 3 * (1u64 << 30) <= available;
            if seeder_enabled && !seeder_fits {
                tracing::info!(
                    weights_bytes = loaded.weights_bytes,
                    budget = residency_budget_bytes(),
                    "prefill seeder skipped: two pipelines exceed the residency budget"
                );
            }
            // Crossover gate: a chunk pass computes seed_chunk positions
            // whatever the prompt holds; below one full chunk, stepping the
            // prompt is measurably cheaper (4B: ~2 s vs ~5 s). The seeder
            // earns its pass only when the prompt fills it.
            let seeder_worthwhile = prompt_len >= seed_chunk as usize;
            if seeder_enabled && seeder_fits && seeder_worthwhile && seed_chunk >= 2 {
                if let Err(error) = loaded
                    .growable
                    .chunk_runner_for(bucket, seed_chunk)
                    .and_then(|seeder| session.set_seeder(seeder))
                {
                    tracing::warn!(%error, "prefill seeder unavailable; stepping instead");
                }
            }
            tracing::info!(
                session = %key.escape_debug(),
                bucket = want,
                elapsed_millis = started.elapsed().as_millis() as u64,
                "holo engine: decode session built"
            );
            state.decode = Some(ResidentDecode {
                bucket: want,
                session,
                verify: None,
            });
        }

        let Loaded {
            growable,
            tokenizer,
            ..
        } = loaded;
        let resident = state.decode.as_mut().expect("built above");
        let mut sink = MeteredSink::new();
        let raw = if speculative_enabled() {
            let bucket = resident.session.geometry().bucket;
            // The cached runner (if any) rides through the closure; if the
            // drafter never qualifies, the closure is never called and the
            // cache is restored untouched below.
            let mut slot = resident.verify.take();
            let mut make_verify = || match slot.take() {
                Some(cached) => Ok(cached),
                None => growable.verify_runner_for(bucket, SPECULATIVE_DRAFT as u64),
            };
            let mut drafter = PromptLookupDrafter;
            let outcome = generate_stream_speculative(
                &mut resident.session,
                &mut make_verify,
                tokenizer,
                prompt,
                cfg,
                &mut drafter,
                SPECULATIVE_DRAFT,
                &mut sink,
            );
            drop(make_verify);
            match outcome {
                Ok((text, returned)) => {
                    resident.verify = returned.or(slot);
                    Ok(text)
                }
                Err(error) => Err(error),
            }
        } else {
            generate_stream_decode(&mut resident.session, tokenizer, prompt, cfg, &mut sink)
        }
        .map_err(|error| LiveError::Protocol(format!("generation failed: {error:#}")))?;
        let gen_millis = sink.started.elapsed().as_millis() as u64;
        let ttft = sink.ttft_millis();
        // Decode rate over the tokens after the first: the first token's cost
        // is the prefill (reported separately as TTFT).
        let generated = tokenizer.encode(&raw).len();
        let tokens_per_second = match (generated, ttft) {
            (n, Some(t)) if n > 1 && gen_millis > t => {
                Some(((n - 1) as f64) / ((gen_millis - t) as f64 / 1000.0))
            }
            _ => None,
        };
        Ok((clean_completion(&raw), ttft, tokens_per_second))
    }

    fn session_entry(&mut self, key: &str) -> &mut SessionState {
        if !self.sessions.contains_key(key) {
            self.evict_to_fit();
            self.sessions.insert(
                key.to_owned(),
                SessionState {
                    decode: None,
                    chatml: String::new(),
                },
            );
        }
        self.touch(key);
        self.sessions.get_mut(key).expect("inserted above")
    }

    /// Resolve the requested model (default when `None`) and load it once:
    /// content-address every safetensors tensor into the artifact's κ-store,
    /// then stand up the staged growable over it. Returns the catalog id the
    /// resident model is keyed by.
    fn ensure_loaded(&mut self, want: Option<&str>) -> Result<String> {
        let name = want
            .filter(|w| !w.trim().is_empty())
            .unwrap_or(&self.default_model);
        let info = self.catalog.resolve(name)?;
        if self.loaded.contains_key(&info.id) {
            return Ok(info.id);
        }
        let dir = self.catalog.artifact_dir(&info.id)?;

        // THE unified substrate: one κ-store for every model and every
        // artifact kind — tensors, int8 tiers, configs, tokenizers, roots.
        // Content addressing dedups shared bytes across models for free.
        let store_dir = dir
            .parent()
            .map(|p| p.join(".kappa-store"))
            .unwrap_or_else(|| dir.join(".kappa-store"));
        std::fs::create_dir_all(&store_dir).map_err(|error| LiveError::io(&store_dir, error))?;
        let store = DirKappaStore::new(&store_dir);

        // config.json: content-address the exact bytes we parse — the parsed
        // view and the κ can never diverge.
        let config_path = dir.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|error| LiveError::io(&config_path, error))?;
        let config_kappa = store
            .insert(config_json.as_bytes())
            .map_err(|error| LiveError::Io(format!("κ-store insert (config): {error:#}")))?;
        let config: serde_json::Value = serde_json::from_str(&config_json)
            .map_err(|error| LiveError::Protocol(format!("config.json: {error}")))?;

        // tokenizer.json: the loader re-reads the file (its eos identity
        // comes from the sibling tokenizer_config.json), so hash before AND
        // after the load and refuse a divergence — no window in which the
        // executed tokenizer differs from the addressed one.
        let tokenizer_path = dir.join("tokenizer.json");
        let tokenizer_bytes = std::fs::read(&tokenizer_path)
            .map_err(|error| LiveError::io(&tokenizer_path, error))?;
        let tokenizer_kappa = store
            .insert(&tokenizer_bytes)
            .map_err(|error| LiveError::Io(format!("κ-store insert (tokenizer): {error:#}")))?;
        let tokenizer = NativeTokenizer::from_tokenizer_json(&tokenizer_path).map_err(|error| {
            LiveError::Protocol(format!(
                "loading tokenizer {}: {error:#}",
                tokenizer_path.display()
            ))
        })?;
        let reread = std::fs::read(&tokenizer_path)
            .map_err(|error| LiveError::io(&tokenizer_path, error))?;
        if reread != tokenizer_bytes {
            return Err(LiveError::Conflict(format!(
                "{} changed while loading; refusing to run a tokenizer that \
                 differs from its content address {tokenizer_kappa}",
                tokenizer_path.display()
            )));
        }
        // tokenizer_config.json carries the eos/bos identity; address it too
        // when present so the root covers everything the tokenizer read.
        let tokenizer_config_kappa = std::fs::read(dir.join("tokenizer_config.json"))
            .ok()
            .map(|bytes| store.insert(&bytes))
            .transpose()
            .map_err(|error| {
                LiveError::Io(format!("κ-store insert (tokenizer_config): {error:#}"))
            })?;

        let rope = hologram_ai_safetensors::parametric::rope_spec_from_config(&config).map_err(
            |error| LiveError::Protocol(format!("rope spec from config.json: {error:#}")),
        )?;

        let started = Instant::now();
        let layers_per_stage_raw = chunking_layers_per_stage(&config);
        let layers_per_stage = NonZeroU64::new(layers_per_stage_raw).expect("nonzero >= 1");
        let tier = engine_quant_tier();

        // Cold-boot fast path (κ-collapse of the ingest): if a prior boot
        // sealed a receipt whose component κs all still exist in the store,
        // re-hashing the safetensors is redundant work — the runtime trust
        // boundary re-verifies every κ at materialization anyway (fail-closed
        // is preserved; a corrupted store entry fails THERE, loudly). The
        // receipt is trusted only up to consistency: the recomputed root κ
        // must reproduce the receipt's root, and config/tokenizer κs (hashed
        // fresh above — they are small) must match.
        let receipt_path = dir.join(".holo-ingest-receipt.json");
        let recalled = load_receipt(
            &receipt_path,
            &dir,
            &store_dir,
            &config_kappa,
            &tokenizer_kappa,
            tokenizer_config_kappa.as_deref(),
            layers_per_stage_raw,
            tier_tag(tier),
        );
        let (keys, kappas, shapes, dtypes, quant, model_kappa) = match recalled {
            Some(receipt) => {
                let quant = receipt.quant_map(tier);
                let root_json = build_root_json(
                    &config_kappa,
                    &tokenizer_kappa,
                    tokenizer_config_kappa.as_deref(),
                    &receipt.keys,
                    &receipt.kappas,
                    &receipt.shapes,
                    &quant,
                );
                let root = store
                    .insert(root_json.as_bytes())
                    .map_err(|error| LiveError::Io(format!("κ-store insert (root): {error:#}")))?;
                if root != receipt.root {
                    return Err(LiveError::Conflict(format!(
                        "ingest receipt root {} does not reproduce from its own components \
                         (got {root}); refusing the fast path — delete {} to re-ingest",
                        receipt.root,
                        receipt_path.display()
                    )));
                }
                tracing::info!(model = %info.name, root = %root, "holo engine: model recalled from receipt (ingest skipped)");
                let dtypes = receipt.parsed_dtypes()?;
                (
                    receipt.keys,
                    receipt.kappas,
                    receipt.shapes,
                    dtypes,
                    quant,
                    root,
                )
            }
            None => {
                tracing::info!(model = %info.name, "holo engine: content-addressing safetensors");
                let manifest = kappa_store_from_safetensors(&dir, &store)?;
                let quant =
                    derive_quant_tier(&config_json, &manifest, &store, layers_per_stage, tier)?;
                let root_json = build_root_json(
                    &config_kappa,
                    &tokenizer_kappa,
                    tokenizer_config_kappa.as_deref(),
                    &manifest.keys,
                    &manifest.kappas,
                    &manifest.shapes,
                    &quant,
                );
                let root = store
                    .insert(root_json.as_bytes())
                    .map_err(|error| LiveError::Io(format!("κ-store insert (root): {error:#}")))?;
                write_receipt(
                    &receipt_path,
                    &dir,
                    layers_per_stage_raw,
                    tier_tag(tier),
                    &config_kappa,
                    &tokenizer_kappa,
                    tokenizer_config_kappa.as_deref(),
                    &manifest,
                    &quant,
                    &root,
                );
                tracing::info!(model = %info.name, root = %root, "holo engine: model root κ sealed");
                (
                    manifest.keys,
                    manifest.kappas,
                    manifest.shapes,
                    manifest.dtypes,
                    quant,
                    root,
                )
            }
        };
        // Approximate one materialized pipeline: ~1 byte per weight element
        // at the int8 tier (rank-1 norms are f32 but negligible).
        let weights_bytes: u64 = shapes
            .iter()
            .map(|shape| shape.iter().product::<u64>())
            .sum();
        let mut growable = GrowableStagedSession::new(
            config_json,
            keys,
            kappas,
            shapes,
            dtypes,
            None, // window ceiling = the model's trained context
            layers_per_stage,
            Box::new(store),
        )
        .map_err(|error| LiveError::Protocol(format!("staged session: {error:#}")))?;
        growable.set_quant_map(quant);
        // Stages whose materialized sessions fit the budget stay resident
        // across tokens — κ-store bandwidth per window, not per token. The
        // default (0) is strict one-stage windowing: ~1 s/token spent
        // re-materializing every step. Native RAM is the budget here.
        growable.set_residency_budget(residency_budget_bytes());
        let context_len = SessionProvider::max_window(&growable) as u64;
        tracing::info!(
            elapsed_millis = started.elapsed().as_millis() as u64,
            context_len,
            "holo engine: model resident"
        );
        let key = info.id.clone();
        self.loaded.insert(
            key.clone(),
            Loaded {
                growable,
                rope,
                context_len,
                weights_bytes,
                tokenizer: ChatMlTokenizer::new(tokenizer),
                model_name: info.name,
                store_dir,
                model_kappa,
            },
        );
        Ok(key)
    }

    fn touch(&mut self, key: &str) {
        self.recency.retain(|k| k != key);
        self.recency.push(key.to_owned());
    }

    fn evict_to_fit(&mut self) {
        while self.sessions.len() >= self.max_sessions && !self.recency.is_empty() {
            let oldest = self.recency.remove(0);
            self.sessions.remove(&oldest);
            tracing::info!(session = %oldest.escape_debug(), "holo engine: evicted least recent session");
        }
    }
}

fn gen_config(request: &WorkerRequest) -> GenConfig {
    GenConfig {
        max_tokens: Some(
            request
                .max_tokens
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MAX_TOKENS),
        ),
        temperature: request.temperature.unwrap_or(DEFAULT_TEMPERATURE),
        top_k: Some(DEFAULT_TOP_K),
        stop: vec![CHATML_STOP.to_owned(), CHATML_OPEN.to_owned()],
        eos: None,
        // hologram-ai's default seed is a fixed constant; identical turns
        // would resample identically. Vary unless the caller pins it.
        seed: request.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1)
        }),
    }
}

/// The canonical model-root manifest JSON — one function so the ingest path
/// and the receipt fast path mint byte-identical roots (κ-stability law).
fn build_root_json(
    config_kappa: &str,
    tokenizer_kappa: &str,
    tokenizer_config_kappa: Option<&str>,
    keys: &[String],
    kappas: &[String],
    shapes: &[Vec<u64>],
    quant: &hologram_ai_common::lower::QuantMap,
) -> String {
    let quant_entries: std::collections::BTreeMap<&String, (&String, u64, u64)> = quant
        .iter()
        .map(|(k, (artifact, out, inf, _tier))| (k, (artifact, *out, *inf)))
        .collect();
    let tensors: Vec<serde_json::Value> = keys
        .iter()
        .zip(kappas)
        .zip(shapes)
        .map(|((name, kappa), shape)| {
            serde_json::json!({ "name": name, "kappa": kappa, "shape": shape })
        })
        .collect();
    serde_json::json!({
        "kind": "hologram-live/model-root",
        "version": 1,
        "config": config_kappa,
        "tokenizer": tokenizer_kappa,
        "tokenizer_config": tokenizer_config_kappa,
        "tensors": tensors,
        "int8_tier": quant_entries
            .iter()
            .map(|(k, (artifact, out, inf))| {
                serde_json::json!({ "wide": k, "artifact": artifact, "out": out, "in": inf })
            })
            .collect::<Vec<_>>(),
    })
    .to_string()
}

/// Ingest receipt: what a completed ingest learned, so the next boot can
/// skip re-hashing unchanged content. Trusted only up to consistency — the
/// root must reproduce from the receipt's own components, and every κ must
/// still exist in the store; the runtime trust boundary re-verifies bytes.
#[derive(serde::Serialize, serde::Deserialize)]
struct IngestReceipt {
    version: u32,
    /// Stage granularity the quant tier was derived under; a different
    /// choice changes the head chunking, so it invalidates the receipt.
    layers_per_stage: u64,
    /// Quant tier the artifacts were derived to ("int8"/"int4"); a different
    /// requested tier invalidates the receipt. Defaulted for older receipts.
    #[serde(default = "default_receipt_tier")]
    tier: String,
    root: String,
    config: String,
    tokenizer: String,
    tokenizer_config: Option<String>,
    safetensors_len: u64,
    safetensors_mtime_millis: u64,
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
    /// DType tags as their Debug names ("F32" / "F16" / "BF16").
    dtypes: Vec<String>,
    quant: Vec<ReceiptQuantEntry>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ReceiptQuantEntry {
    key: String,
    artifact: String,
    out: u64,
    #[serde(rename = "in")]
    inf: u64,
}

fn default_receipt_tier() -> String {
    "int8".to_owned()
}

impl IngestReceipt {
    fn quant_map(
        &self,
        tier: hologram_ai_common::lower::QuantTier,
    ) -> hologram_ai_common::lower::QuantMap {
        self.quant
            .iter()
            .map(|e| (e.key.clone(), (e.artifact.clone(), e.out, e.inf, tier)))
            .collect()
    }

    fn parsed_dtypes(&self) -> Result<Vec<DType>> {
        self.dtypes
            .iter()
            .map(|tag| match tag.as_str() {
                "F32" => Ok(DType::F32),
                "F16" => Ok(DType::F16),
                "BF16" => Ok(DType::BF16),
                other => Err(LiveError::Protocol(format!(
                    "ingest receipt holds unsupported dtype {other:?}"
                ))),
            })
            .collect()
    }
}

fn file_len_mtime(path: &Path) -> Option<(u64, u64)> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_millis() as u64;
    Some((meta.len(), mtime))
}

/// Filesystem name of a κ entry in the store — MUST mirror hologram-ai's
/// `DirKappaStore` naming (κ labels contain `:`, encoded `-` for NTFS).
/// Upstream candidate: a `contains(κ)` on the store would remove this coupling.
fn kappa_exists(store_dir: &Path, kappa: &str) -> bool {
    store_dir
        .join(format!("{}.bin", kappa.replace(':', "-")))
        .is_file()
}

/// The receipt fast path's admission test. `None` means "take the full
/// ingest path" — never an error: a stale or damaged receipt only costs the
/// re-ingest it would have skipped.
fn load_receipt(
    receipt_path: &Path,
    dir: &Path,
    store_dir: &Path,
    config_kappa: &str,
    tokenizer_kappa: &str,
    tokenizer_config_kappa: Option<&str>,
    expected_layers_per_stage: u64,
    expected_tier: &str,
) -> Option<IngestReceipt> {
    let receipt: IngestReceipt = serde_json::from_slice(&std::fs::read(receipt_path).ok()?).ok()?;
    if receipt.version != 2
        || receipt.layers_per_stage != expected_layers_per_stage
        || receipt.tier != expected_tier
        || receipt.config != config_kappa
        || receipt.tokenizer != tokenizer_kappa
        || receipt.tokenizer_config.as_deref() != tokenizer_config_kappa
    {
        return None;
    }
    let (len, mtime) = file_len_mtime(&dir.join("model.safetensors"))?;
    if len != receipt.safetensors_len || mtime != receipt.safetensors_mtime_millis {
        return None;
    }
    let all_present = receipt
        .kappas
        .iter()
        .chain(receipt.quant.iter().map(|e| &e.artifact))
        .all(|kappa| kappa_exists(store_dir, kappa));
    if !all_present {
        return None;
    }
    Some(receipt)
}

fn write_receipt(
    receipt_path: &Path,
    dir: &Path,
    layers_per_stage: u64,
    tier: &str,
    config_kappa: &str,
    tokenizer_kappa: &str,
    tokenizer_config_kappa: Option<&str>,
    manifest: &KappaManifest,
    quant: &hologram_ai_common::lower::QuantMap,
    root: &str,
) {
    let Some((len, mtime)) = file_len_mtime(&dir.join("model.safetensors")) else {
        return;
    };
    let receipt = IngestReceipt {
        version: 2,
        layers_per_stage,
        tier: tier.to_owned(),
        root: root.to_owned(),
        config: config_kappa.to_owned(),
        tokenizer: tokenizer_kappa.to_owned(),
        tokenizer_config: tokenizer_config_kappa.map(str::to_owned),
        safetensors_len: len,
        safetensors_mtime_millis: mtime,
        keys: manifest.keys.clone(),
        kappas: manifest.kappas.clone(),
        shapes: manifest.shapes.clone(),
        dtypes: manifest.dtypes.iter().map(|d| format!("{d:?}")).collect(),
        quant: quant
            .iter()
            .map(|(key, (artifact, out, inf, _tier))| ReceiptQuantEntry {
                key: key.clone(),
                artifact: artifact.clone(),
                out: *out,
                inf: *inf,
            })
            .collect(),
    };
    // Best-effort: a receipt that fails to write only costs the next boot
    // the re-ingest this one performed anyway.
    if let Ok(bytes) = serde_json::to_vec_pretty(&receipt) {
        let _ = std::fs::write(receipt_path, bytes);
    }
}

/// The per-tensor manifest of a safetensors file, content-addressed into a
/// `DirKappaStore` beside the model (`.holo-kappa-store/`, reused across
/// restarts because insertion is content-addressed).
struct KappaManifest {
    keys: Vec<String>,
    kappas: Vec<String>,
    shapes: Vec<Vec<u64>>,
    dtypes: Vec<DType>,
    /// Byte range of each tensor within the safetensors file (data section
    /// absolute), for the int8 tier derivation.
    ranges: Vec<(usize, usize)>,
    /// The whole safetensors file, alive until the quant tier has derived.
    file_bytes: Vec<u8>,
}

fn kappa_store_from_safetensors(dir: &Path, store: &DirKappaStore) -> Result<KappaManifest> {
    let path = dir.join("model.safetensors");
    let bytes = std::fs::read(&path).map_err(|error| LiveError::io(&path, error))?;
    if bytes.len() < 8 {
        return Err(LiveError::Protocol(format!(
            "{} is not a safetensors file (too short)",
            path.display()
        )));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let data_start = 8 + header_len;
    if bytes.len() < data_start {
        return Err(LiveError::Protocol(format!(
            "{}: header length {header_len} exceeds the file",
            path.display()
        )));
    }
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&bytes[8..data_start]).map_err(|error| {
            LiveError::Protocol(format!("{}: safetensors header: {error}", path.display()))
        })?;

    let mut manifest = KappaManifest {
        keys: Vec::new(),
        kappas: Vec::new(),
        shapes: Vec::new(),
        dtypes: Vec::new(),
        ranges: Vec::new(),
        file_bytes: Vec::new(),
    };
    for (name, entry) in &header {
        if name == "__metadata__" {
            continue;
        }
        let field = |key: &str| {
            entry.get(key).ok_or_else(|| {
                LiveError::Protocol(format!("{name}: safetensors entry missing {key}"))
            })
        };
        let dtype = match field("dtype")?.as_str().unwrap_or_default() {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            other => {
                return Err(LiveError::Protocol(format!(
                    "{name}: unsupported safetensors dtype {other:?} (expected F32/F16/BF16)"
                )))
            }
        };
        let shape: Vec<u64> = field("shape")?
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let offsets = field("data_offsets")?
            .as_array()
            .and_then(|a| {
                let start = a.first()?.as_u64()? as usize;
                let end = a.get(1)?.as_u64()? as usize;
                Some((start, end))
            })
            .ok_or_else(|| {
                LiveError::Protocol(format!("{name}: malformed safetensors data_offsets"))
            })?;
        let (start, end) = (data_start + offsets.0, data_start + offsets.1);
        if end > bytes.len() || start > end {
            return Err(LiveError::Protocol(format!(
                "{name}: safetensors data range {start}..{end} exceeds the file"
            )));
        }
        let kappa = store
            .insert(&bytes[start..end])
            .map_err(|error| LiveError::Io(format!("κ-store insert for {name}: {error:#}")))?;
        manifest.keys.push(name.clone());
        manifest.kappas.push(kappa);
        manifest.shapes.push(shape);
        manifest.dtypes.push(dtype);
        manifest.ranges.push((start, end));
    }
    if manifest.keys.is_empty() {
        return Err(LiveError::Protocol(format!(
            "{} holds no tensors",
            path.display()
        )));
    }
    manifest.file_bytes = bytes;
    Ok(manifest)
}

/// Derive the int8 quantized tier — the same tier the hologram-ai web app
/// runs — and return the quant map to install on the staged session: whole
/// projection weights retire to per-channel int8 artifacts, and the LM head's
/// vocab-row chunks crystallize so no whole-panel F32 image thrashs the
/// windows. Artifacts are content-addressed into the same κ-store, so a
/// restart re-derives cheaply into cache hits.
fn derive_quant_tier(
    config_json: &str,
    manifest: &KappaManifest,
    store: &DirKappaStore,
    layers_per_stage: NonZeroU64,
    tier: hologram_ai_common::lower::QuantTier,
) -> Result<hologram_ai_common::lower::QuantMap> {
    use hologram_ai_common::lower::quant_key;

    let index: HashMap<&str, usize> = manifest
        .kappas
        .iter()
        .enumerate()
        .map(|(i, k)| (k.as_str(), i))
        .collect();
    let mut map = hologram_ai_common::lower::QuantMap::new();

    let projections = hologram_ai::staged::quantizable_weights(
        config_json,
        &manifest.keys,
        &manifest.kappas,
        &manifest.shapes,
        &manifest.dtypes,
        None,
        layers_per_stage,
    )
    .map_err(|error| LiveError::Protocol(format!("quantizable weights: {error:#}")))?;
    for wide_kappa in &projections {
        let i = *index.get(wide_kappa.as_str()).ok_or_else(|| {
            LiveError::Protocol(format!("quant target {wide_kappa} is not in the manifest"))
        })?;
        let shape = &manifest.shapes[i];
        if shape.len() != 2 {
            continue; // only 2-D projections quantize
        }
        let (out, inf) = (shape[0], shape[1]);
        let (start, end) = manifest.ranges[i];
        let artifact = hologram_ai::quantized::derive_quantized_artifact_tier(
            &manifest.file_bytes[start..end],
            manifest.dtypes[i],
            tier,
            out,
            inf,
        )
        .map_err(|error| {
            LiveError::Protocol(format!(
                "deriving {} for {}: {error:#}",
                tier_tag(tier),
                manifest.keys[i]
            ))
        })?;
        let artifact_kappa = store
            .insert(&artifact)
            .map_err(|error| LiveError::Io(format!("κ-store insert (int8): {error:#}")))?;
        map.insert(
            quant_key(wide_kappa, None),
            (artifact_kappa, out, inf, tier),
        );
    }

    let chunks = hologram_ai::staged::head_quant_chunks(
        config_json,
        &manifest.keys,
        &manifest.kappas,
        &manifest.shapes,
        &manifest.dtypes,
        None,
        layers_per_stage,
    )
    .map_err(|error| LiveError::Protocol(format!("head quant chunks: {error:#}")))?;
    let n_chunks = chunks.len();
    for chunk in chunks {
        let i = *index.get(chunk.kappa.as_str()).ok_or_else(|| {
            LiveError::Protocol(format!(
                "head chunk κ {} is not in the manifest",
                chunk.kappa
            ))
        })?;
        let (start, _end) = manifest.ranges[i];
        let (s, e) = (
            start + chunk.offset as usize,
            start + (chunk.offset + chunk.len) as usize,
        );
        if e > manifest.file_bytes.len() {
            return Err(LiveError::Protocol(format!(
                "head chunk range of {} exceeds the tensor",
                manifest.keys[i]
            )));
        }
        let artifact = hologram_ai::quantized::derive_quantized_artifact_tier(
            &manifest.file_bytes[s..e],
            manifest.dtypes[i],
            tier,
            chunk.out_features,
            chunk.in_features,
        )
        .map_err(|error| {
            LiveError::Protocol(format!(
                "deriving int8 head chunk of {}: {error:#}",
                manifest.keys[i]
            ))
        })?;
        let artifact_kappa = store
            .insert(&artifact)
            .map_err(|error| LiveError::Io(format!("κ-store insert (int8 head): {error:#}")))?;
        map.insert(
            quant_key(&chunk.kappa, Some((chunk.offset, chunk.len))),
            (artifact_kappa, chunk.out_features, chunk.in_features, tier),
        );
    }
    tracing::info!(
        tier = tier_tag(tier),
        projections = projections.len(),
        head_chunks = n_chunks,
        "holo engine: quant tier derived"
    );
    Ok(map)
}

/// Trim the completion at the first turn boundary the model produced — a real
/// stop token decoded to text, or a mimicked plain-text one.
fn clean_completion(raw: &str) -> String {
    let mut text = raw.trim().to_owned();
    for marker in [CHATML_STOP, CHATML_OPEN, "\nuser:", "\nassistant:"] {
        if let Some(i) = text.find(marker) {
            text.truncate(i);
        }
    }
    text.trim_end().to_owned()
}

/// The daemon's transcript rendering (`chat::render_transcript`, role-prefixed
/// lines) → ChatML turns. Lines without a role prefix continue the current
/// turn (multi-line messages); input with no prefixes at all is one user turn.
fn transcript_to_chatml(transcript: &str) -> String {
    let mut turns: Vec<(&str, String)> = Vec::new();
    for line in transcript.lines() {
        let (role, rest) = if let Some(rest) = line.strip_prefix("user: ") {
            ("user", rest)
        } else if let Some(rest) = line.strip_prefix("assistant: ") {
            ("assistant", rest)
        } else if let Some(rest) = line.strip_prefix("system: ") {
            ("system", rest)
        } else if let Some((_, content)) = turns.last_mut() {
            content.push('\n');
            content.push_str(line);
            continue;
        } else {
            ("user", line)
        };
        turns.push((role, rest.to_owned()));
    }
    if turns.is_empty() {
        turns.push(("user", transcript.to_owned()));
    }
    let mut chatml = String::new();
    for (role, content) in &turns {
        chatml.push_str(&format!("{CHATML_OPEN}{role}\n{content}{CHATML_STOP}\n"));
    }
    chatml
}

#[cfg(test)]
mod tests {
    use super::{clean_completion, transcript_to_chatml};

    #[test]
    fn transcript_renders_as_chatml_turns() {
        let chatml = transcript_to_chatml("user: hi\nassistant: hello\nuser: how are you");
        assert_eq!(
            chatml,
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nhello<|im_end|>\n<|im_start|>user\nhow are you<|im_end|>\n"
        );
    }

    #[test]
    fn unprefixed_lines_continue_the_previous_turn() {
        let chatml = transcript_to_chatml("user: first\nsecond line");
        assert_eq!(chatml, "<|im_start|>user\nfirst\nsecond line<|im_end|>\n");
    }

    #[test]
    fn bare_text_is_one_user_turn() {
        let chatml = transcript_to_chatml("just a prompt");
        assert_eq!(chatml, "<|im_start|>user\njust a prompt<|im_end|>\n");
    }

    /// Ground-truth probe against a real model dir (safetensors + configs).
    /// Run explicitly: HOLO_TEST_MODEL_DIR=<dir> cargo test --release
    /// holo_real_model_probe -- --ignored --nocapture
    #[test]
    #[ignore]
    fn holo_real_model_probe() {
        use super::*;
        let dir =
            PathBuf::from(std::env::var("HOLO_TEST_MODEL_DIR").expect("set HOLO_TEST_MODEL_DIR"));
        let tokenizer =
            NativeTokenizer::from_tokenizer_json(&dir.join("tokenizer.json")).expect("tokenizer");
        let wrapped = ChatMlTokenizer::new(tokenizer);
        println!("specials found: {:?}", wrapped.specials);
        println!("eos_token_id: {}", wrapped.eos_token_id());
        let prompt =
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n";
        let ids = wrapped.encode(prompt);
        println!(
            "prompt ids ({}): {:?}",
            ids.len(),
            &ids[..ids.len().min(20)]
        );

        let config_json = std::fs::read_to_string(dir.join("config.json")).expect("config");
        let config: serde_json::Value = serde_json::from_str(&config_json).expect("json");
        let rope =
            hologram_ai_safetensors::parametric::rope_spec_from_config(&config).expect("rope");
        let store_dir = dir.join(".holo-kappa-store");
        std::fs::create_dir_all(&store_dir).expect("store dir");
        let store = DirKappaStore::new(&store_dir);
        let manifest = kappa_store_from_safetensors(&dir, &store).expect("manifest");
        let lps = NonZeroU64::new(LAYERS_PER_STAGE).expect("nonzero");
        let t0 = Instant::now();
        let quant = derive_quant_tier(
            &config_json,
            &manifest,
            &store,
            lps,
            hologram_ai_common::lower::QuantTier::Int8,
        )
        .expect("int8");
        println!("int8 tier: {} entries in {:?}", quant.len(), t0.elapsed());
        let mut growable = GrowableStagedSession::new(
            config_json,
            manifest.keys,
            manifest.kappas,
            manifest.shapes,
            manifest.dtypes,
            None,
            lps,
            Box::new(store),
        )
        .expect("growable");
        growable.set_quant_map(quant);
        // Stages whose materialized sessions fit the budget stay resident
        // across tokens — κ-store bandwidth per window, not per token. The
        // default (0) is strict one-stage windowing: ~1 s/token spent
        // re-materializing every step. Native RAM is the budget here.
        growable.set_residency_budget(residency_budget_bytes());
        let context_len = SessionProvider::max_window(&growable) as u64;
        let want = decode_bucket_for_turn(ids.len(), 48, context_len as usize);
        let t1 = Instant::now();
        let runner = growable.decode_runner_for(want).expect("runner");
        let mut session = DecodeSession::new(runner, rope, context_len).expect("session");
        println!("decode session (bucket {want}) in {:?}", t1.elapsed());
        let cfg = GenConfig {
            max_tokens: Some(48),
            temperature: 0.0,
            top_k: None,
            stop: vec![CHATML_STOP.to_owned(), CHATML_OPEN.to_owned()],
            eos: None,
            seed: 7,
        };
        let t2 = Instant::now();
        let mut out = Vec::new();
        let raw =
            generate_stream_decode(&mut session, &wrapped, prompt, &cfg, &mut out).expect("gen");
        let dt = t2.elapsed();
        println!(
            "generated {:?} in {:?} ({} realized)",
            raw,
            dt,
            session.realized_len()
        );
    }

    #[test]
    fn completion_is_cut_at_the_first_turn_boundary() {
        assert_eq!(
            clean_completion("Paris.<|im_end|>\n<|im_start|>user\nmore"),
            "Paris."
        );
        assert_eq!(clean_completion("Paris.\nuser: next question"), "Paris.");
        assert_eq!(clean_completion("  Paris.  "), "Paris.");
    }
}
