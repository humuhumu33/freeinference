# VERIFICATION

Every gate is armed by a planted defect that made it fail, then restored. A gate that has never failed proves nothing. Records keep the exact diagnostic where it is stable.

| Gate | Planted defect | Diagnostic | Restored |
| --- | --- | --- | --- |
| Site rules (`scripts/check_site.py`, SI-01) | A hyphen in the footer: `Apache-2.0 or MIT` | `FAIL hyphen minus in visible text: 'Hologram · Apache-2.0 or MIT'`, exit 1 | Footer text returned to `Apache 2.0 or MIT`; gate prints `ok: no dashes, 111 words of prose, headline exact` |
| Receipt integrity (`cargo test`, RC-02 and the receipt unit tests) | Receipt κ computed over the unsigned bytes instead of the sealed bytes: `kappa: kappa_of(&bound.signable())` | `receipt::tests::sealed_receipt_verifies_and_tampering_is_refused` panicked: `genuine receipt verifies: "kappa blake3:90a15fb9… does not match sealed bytes blake3:8b29bde1…"`; the run stopped before the conformance binary | `kappa: kappa_of(&sealed)`; 3 unit tests and 7 conformance tests pass |

## Measured verdicts, WebGPU engine, 2026-09-10

The Playground's WebGPU path runs Hologram Q's ternary engine, served by the daemon under `/q`, on HOLOGRAMTECH/q-bitnet-2b in this browser. The daemon's echo engine was configured; the daemon only stores what the browser sealed. Measured in the desktop app's browser pane on the developer machine.

| Case | What was done | Verdict |
| --- | --- | --- |
| Load | The WebGPU model chosen in the Playground; weights stream from Hugging Face on the first visit and from the browser's own store after that | First visit: 0.69 GB verified block by block. Later visit: resident on the GPU in 6.4 s before the first prompt |
| Cold turn | A prompt with the whole conversation framed and prefilled from scratch | 767 ms in total, first token at 553 ms |
| Warm turn | The next prompt, extending the running token ids of the last committed turn as Q's fast brain does, so the engine reuses the resident KV | 454 ms in total, first token at 353 ms; a four sentence answer streamed at 30.7 tokens per second |
| Same prompt, new session | The first prompt sent again after a page reload | The same receipt κ `d98f4812…`, the same output κ |
| Re-derive | The "re-derive on this GPU" button on a cold answer and on a warm answer | `re-derived: identical` for both |
| Dashboard | The Dashboard after four answers | Four rows marked `Sealed · WebGPU`, each linking to the stored q-receipt |
| Sealed to the daemon | A new prompt in the Playground after the engine became the browser's | Generated in 1007 ms, cold prefill, first token 856 ms; `/v1/webgpu/seal` stored the Q receipt, the answer bytes and a memo |
| Repeated in the Playground | The page reloaded, the same prompt sent again | 82 ms, `served from receipt, no execution`; the GPU was not touched, the model was still loading |
| Repeated from an OpenAI client | The same messages and parameters through `/v1/chat/completions` | 85 ms, `x-hologram-stream: memo`, `x-hologram-reuse`, the Q receipt κ in `x-hologram-receipt`, the fingerprint the browser engine wrote |
| Re-derived from the store | "re-derive on this GPU" on the served answer, the stored Q receipt fetched by κ | `re-derived: identical` |
| Unanswered prompt | A prompt no receipt answers, through `/v1/chat/completions` | Refused: compute runs in the browser, open /playground; nothing echoed, nothing sealed |

## Gates and what they cover

1. `python scripts/model_write.py --check`: every id in `model/ids.toml` has exactly one scenario whose file, level and statement match, and exactly one test named `conformance_<id>`; ledger claims graded `some-true` name an authority, `build` claims do not; `CONFORMANCE.md` equals its regeneration.
2. `python scripts/check_site.py site/index.html`: no hyphen or dash of any kind in visible text, code samples included; under 120 words of prose; headline exact.
3. `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`.
4. `cargo test --locked -- --test-threads=1`: receipt unit tests and the seventeen conformance tests, which build the daemon in a temporary directory, with hologram-live's echo engine standing in for compute where a prompt must be answered and with the shipped WebGPU engine where the refusal is the claim, and drive the router directly.

`just vv` runs all four. CI runs the same commands on ubuntu.

## Removed: the CPU engine, 2026-09-10

The deterministic CPU engine (hologram-ai in the `engine/` package) and its private substrate pin were removed on 2026-09-10; the engine is the browser's from this commit. The two records below were measured on it the day before and the day of removal and are kept as history of what the receipt and reuse design was checked against. They no longer describe the shipped binary.

### History: deterministic engine, 2026-09-09

Daemon on this machine's real configuration: `inference.engine = "holo"`, default model SmolLM2 from the unified κ-store. Client: the official `openai` Python SDK 2.24, temperature 0, seed 7.

| Case | What was done | Verdict |
| --- | --- | --- |
| Genuine | `freeinference verify <receipt κ>` on the receipt returned in `x-hologram-receipt` | `confirmed: … replays byte for byte on this machine`, exit 0, 15.4 s; replay reported 31 expected and 31 replayed bytes, no divergence |
| Determinism | The same prompt, seed and parameters sent twice | Identical output κ, identical answer κ, identical receipt κ |
| Tampered | A copy of the genuine receipt with its bound output κ replaced, stored as a new receipt object | `verified false, integrity false`, reason: signature does not verify over the canonical bytes |
| Missing | `freeinference verify` on a κ that names no object | `not verified`, exit 1 |

The fingerprint on these answers was the model root κ `blake3:e41292a4…`, a manifest over config, tokenizer, every tensor and every derived artifact, joined with the engine build κ, the address of the pinned hologram-ai revision and quantization tier.

### History: reuse and byte identity on the deterministic engine, 2026-09-10

The two claims on the homepage, measured on the deterministic engine (`engine/`, SmolLM2 at the model root κ `blake3:e41292a4…`) and on a second daemon started with fresh directories, an empty catalog and the echo engine, standing in for another machine. Loopback, developer machine, debug build.

| Case | What was done | Verdict |
| --- | --- | --- |
| Executed | `POST /v1/chat/completions` for `smollm2`, max_tokens 32, temperature 0, seed 1 | 22.1 s; `x-hologram-stream: emulated`, receipt `blake3:6be02717…` |
| Repeated | The same request twice more | 91 ms and 84 ms; `x-hologram-stream: memo`, `x-hologram-reuse` names the memo `blake3:f010c232…`, the same receipt κ, the same fingerprint, the same text; no new receipt sealed |
| Carried | The memo, receipt and answer objects fetched from the first daemon's objects API and posted to the second daemon's | Each landed at the same κ on the second daemon |
| Reused elsewhere | The same request on the second daemon, model named by its root κ, no model present | 114 ms; the SmolLM2 text with the original receipt κ and fingerprint; one receipt on that daemon, the carried one; nothing executed |
| Integrity elsewhere | `POST /v1/receipts/{κ}/verify` on the second daemon | Before the fix in this commit: an error, because the record could not be replayed there. After: `integrity true, verified false`, reason: replay is not possible on this machine |
| Replay at home | The same verify on the first daemon | `verified true, integrity true`, 31 of 31 bytes, 15.0 s |
| Model bytes, independent | `scripts/rederive_root.py` cut every tensor out of `model.safetensors` by the header, hashed each, hashed config, tokenizer and every int8 artifact in the store, and rebuilt the root manifest | 272 tensors, 212 artifacts, root κ `e41292a4…` all reproduced; the store's root object is byte identical to the re-derivation |
| Model bytes, second daemon | Only the raw model files copied, no ingest receipt, no store; the second daemon ingested on its own | Identical root κ, identical 272 tensor κs, identical 212 artifact κs |
| Output, second daemon | The same new prompt executed fresh on both daemons | Identical prompt, params, output and answer κs under two different signing keys |
| Tamper | One bit flipped in an artifact in the second daemon's store, daemon restarted | Refused at use in 236 ms; the store purged the object on read; restoring the bytes recovered it |

What this establishes: a repeated prompt costs a store lookup, not an execution, and the answer, its receipt and its index travel as three content addressed objects that any daemon serves and checks for integrity. What it does not establish: replay on a machine without the model. That is by design; verification by replay is the price of a machine that wants proof rather than a signature.

## Not yet armed

Re-derivation of a Q receipt on a second, different GPU has not been measured; today re-derive is checked on the sealing device only.

The Rust CI job has not yet produced a failing run from a planted defect on GitHub itself; the records above are local runs. The PrismPM template with its pinned devcontainer and digest bound action is adopted in a following step and will re-arm every gate under its own record.
