# VERIFICATION

Every gate is armed by a planted defect that made it fail, then restored. A gate that has never failed proves nothing. Records keep the exact diagnostic where it is stable.

| Gate | Planted defect | Diagnostic | Restored |
| --- | --- | --- | --- |
| Site rules (`scripts/check_site.py`, SI-01) | A hyphen in the footer: `Apache-2.0 or MIT` | `FAIL hyphen minus in visible text: 'Hologram · Apache-2.0 or MIT'`, exit 1 | Footer text returned to `Apache 2.0 or MIT`; gate prints `ok: no dashes, 111 words of prose, headline exact` |
| Receipt integrity (`cargo test`, RC-02 and the receipt unit tests) | Receipt κ computed over the unsigned bytes instead of the sealed bytes: `kappa: kappa_of(&bound.signable())` | `receipt::tests::sealed_receipt_verifies_and_tampering_is_refused` panicked: `genuine receipt verifies: "kappa blake3:90a15fb9… does not match sealed bytes blake3:8b29bde1…"`; the run stopped before the conformance binary | `kappa: kappa_of(&sealed)`; 3 unit tests and 7 conformance tests pass |

## Measured verdicts, deterministic engine, 2026-09-09

Daemon on this machine's real configuration: `inference.engine = "holo"`, default model SmolLM2 from the unified κ-store. Client: the official `openai` Python SDK 2.24, temperature 0, seed 7.

| Case | What was done | Verdict |
| --- | --- | --- |
| Genuine | `freeinference verify <receipt κ>` on the receipt returned in `x-hologram-receipt` | `confirmed: … replays byte for byte on this machine`, exit 0, 15.4 s; replay reported 31 expected and 31 replayed bytes, no divergence |
| Determinism | The same prompt, seed and parameters sent twice | Identical output κ, identical answer κ, identical receipt κ |
| Tampered | A copy of the genuine receipt with its bound output κ replaced, stored as a new receipt object | `verified false, integrity false`, reason: signature does not verify over the canonical bytes |
| Missing | `freeinference verify` on a κ that names no object | `not verified`, exit 1 |

The fingerprint on these answers was the model root κ `blake3:e41292a4…`, a manifest over config, tokenizer, every tensor and every derived artifact, joined with the engine build κ, the address of the pinned hologram-ai revision and quantization tier.

## Gates and what they cover

1. `python scripts/model_write.py --check`: every id in `model/ids.toml` has exactly one scenario whose file, level and statement match, and exactly one test named `conformance_<id>`; ledger claims graded `some-true` name an authority, `build` claims do not; `CONFORMANCE.md` equals its regeneration.
2. `python scripts/check_site.py site/index.html`: no hyphen or dash of any kind in visible text, code samples included; under 120 words of prose; headline exact.
3. `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`.
4. `cargo test --locked -- --test-threads=1`: receipt unit tests and the seven conformance tests, which build the daemon in a temporary directory with the echo engine, import a fixture model, and drive the router directly.

`just vv` runs all four. CI runs the same commands on ubuntu.

## Engine build and CI

The engine's substrate, `hologram-archive` and siblings, is pinned by hologram-ai at revision `15d155b9`, the commit that fixes summation order on every target. That commit exists only in the private repository `humuhumu33/hologram` today. Public CI therefore builds with `--no-default-features`, which verifies everything except the engine, and the `engine` job runs the full build only when the repository secret `HOLOGRAM_GIT_TOKEN` grants read access. Until one of three things happens, the engine build is verified on the developer machine only: the substrate commit is upstreamed to `Hologram-Technologies/hologram`, the private repository is made public, or the token is added.

## Not yet armed

The Rust CI job has not yet produced a failing run from a planted defect on GitHub itself; the records above are local runs at the commit that introduced Phase 1. The PrismPM template with its pinned devcontainer and digest bound action is adopted in a following step and will re-arm every gate under its own record.
