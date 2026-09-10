# Free verified AI inference: phased plan, 2026-09-09

Written backwards from the end state. Every phase names what the user experiences, what is reused, what is new, and how PrismPM governs it. Grounded in reads of hologram-live a9f44a5, hologram b6d9a76, PrismPM f57a697, kappa-registry 2af8656, hologram-live-ip unification/phase-1 b8db8e5.

## The end state, from the user's chair

Day 30. A developer with an OpenAI client and an open model in production.

1. **Discover.** On a Hugging Face model page, "Use this model" lists Hologram next to Ollama. The line under it: Free verified AI inference. Same line in LiteLLM's provider list and Open WebUI's connection guide.
2. **Install.** One line. `curl -fsSL https://freeinference.ai/install | sh`, or brew, winget, pip, npm.
3. **Run.** `hologram run qwen/qwen3-8b`. Each file prints its hash and a verifying step. The last line is a base URL.
4. **Change one line.** `base_url` in their existing code. Same SDK, same calls.
5. **See three fields mean something.** `system_fingerprint` is the model κ and engine κ. `cached_tokens` is true. A receipt address rides in a header. All ignorable.
6. **Pay nothing twice.** The second request sharing a prefix skips prefill and says so.
7. **Check when it matters.** `hologram verify <κ>` prints one line: confirmed, or refuted at byte N.
8. **See it in the app.** Activity page with cost, tokens and cache rate. Verification Center with three green checks: model bytes match the published hash, engine is deterministic on this machine, this answer replays.
9. **Work offline, work in a fleet.** After first pull nothing needs the network. Machines that share a model share the trie without configuration.

Every phase below exists because one of those nine lines needs it.

## Architecture and governance, fixed before Phase 0

**Repo.** `github.com/humuhumu33/freeinference`, public, MIT or Apache 2.0. No hyphen in the name because the homepage forbids hyphens and the slug appears on it.

**Modularity rule.** One capability equals one hologram-live module, one config section, one PrismPM suite, one feature file, one directory. The module trait is four methods: descriptor, router, openapi, start. A module declares its dependencies in its descriptor and nothing else may reference it. Removing a module means deleting its directory and its one registration line. Routers merge, so each module owns unique paths, checked by a test that boots all modules and asserts no panic.

**Dependency direction.** The new repo depends on hologram-live by git rev, never forks it. hologram-live's registry pulls modules only from its own builtin list, so the first upstream contribution is a ten-line seam: `ModuleRegistry::build_with(extra)` accepting external modules. Until it merges, a patch section in Cargo.toml carries it. Engine comes from hologram-live-ip's `unification/phase-1` branch behind the existing `InferenceEngine` trait, feature `engine-holo`, never copied. κ primitives come from `uor-hologram` at or after 2bda6a9: `address_bytes`, `derive_label`, the `realization!` macro, `SessionAttestation` as the receipt precedent.

**PrismPM, both tiers, used where each fits.**

- Tier 2, governance, adopted by forking `UOR-Foundation/template`, because its lock pins that origin. Every module has a suite in `model/ids.toml`, one claim per user-feelable fact, one Gherkin scenario byte-equal to the claim, one test named `conformance_<id>`. `CONFORMANCE.md` is generated and never edited. Every gate gets a planted defect and a restore commit in `VERIFICATION.md`. `just vv` is the only green. CI calls the PrismPM action pinned to a full commit SHA with the SDK image pinned by digest. The devcontainer is amd64 Linux, so on Windows the gate runs in WSL. The template's Justfile needs a `model-write` recipe that PrismPM's own Justfile lacks, added on day one.
- Tier 1, generated code, for exactly the pure parts that fit its domain: the receipt's canonical encoding and the verify predicate, both total validators over structs, lists and naturals. Authored once in `src/Receipt.lex.tex`, checked by Lean, emitted as a `no_std` Rust validator and a Core-Wasm guest. The verifier that judges every answer is itself kernel-checked, and the same bytes run in the daemon, the browser and the desktop.

**Honesty grades.** Library facts are `some-true` with an authority. Everything a test proves is `build`. Cross-machine determinism is `build` only where two hosts in CI produced identical bytes, never asserted beyond that. The `open` grade is unreachable in PrismPM's parser today, noted as a known limitation.

**Reuse, by name.** hologram-live: daemon, module registry, axum router, OpenAPI, clap CLI, config overlay, audit log, OTel, BDD harness, install script, Tauri sidecar. hologram: κ, realizations, `KappaStore`, HTTP CAS at `/cas/{κ}`, `.holo` archive. OpenAI types from `async-openai` 0.42 instead of hologram-live's hand-written structs, which lack `system_fingerprint`, tools and content parts. Official `openai` 3.10 Python and 7.12 Node SDKs, `anthropic` 1.4 and 0.124, as conformance clients. `ed25519-dalek` 3.0 for signatures. `huggingface_hub` header contract for the pull path. LiteLLM, Open WebUI and Claude Code driven by their own configs in integration tests.

## Phases, in build order

| Phase | User line served | Reuse | New code | PrismPM suite | Days |
|---|---|---|---|---|---|
| 0 Homepage | 1 | hologram-website stack, React, Vite, Tailwind, Pages | copy and one page | `site`: no hyphen, under 120 words, deploys | 2 |
| 1 Endpoint with receipts | 4, 5 | openai-compat module, `async-openai` types, `SessionAttestation` shape, ed25519-dalek | receipt realization, header insert, fingerprint | `receipts` | 4 |
| 2 Verify | 7 | Tier 1 generated validator, re-derivation in the browser engine | two routes, one CLI subcommand | `verify` | 3 |
| 3 Prefix cache | 6 | `derive_label`, redb store range scans | trie, HIT MISS PARTIAL header, truthful `cached_tokens` | `prefix` | 5 |
| 4 Verified pull | 3, 9 | `update.rs` verify pattern, `holo_fetch` transport hardening, huggingface_hub headers | streaming dual-hash store write, Range, `run` | `pull` | 8 |
| 5 Catalog and second protocol | 1, 2 | `/v1/models` in HF provider format, Ollama's `/v1/messages` precedent | anthropic-compat module, listings | `catalog` | 4 |
| 6 Desktop | 8 | Tauri app, existing view switch, sidecar command pattern | two pages, three commands | `desktop` | 3 |
| 7 Share | 9 | hologram-net `serve_get`, `HttpKappaSync` verify-on-receipt | RemoteRegistryProvider, spot recompute, tenant salt | `share` | 4 |

Thirty-three working days. New code is roughly 3,400 lines of Rust and 300 of TypeScript, all glue around existing parts except the trie and the streaming store write.

**Phase 0, homepage.** Copy the deployed gethologram.ai stack into `site/`. One page. Headline verbatim: Free verified AI inference. Three lines of what, one install line, one code sample with the base URL swap as the only diff, three proofs the reader can act on, one call to action. Under 120 words of prose. A CI step greps the built HTML for the hyphen character and fails on any hit, including inside code blocks and the footer. Gate: the page deploys to Pages and the check passes. Planted defect: a hyphen in the footer, caught, removed.

**Phase 1, endpoint with receipts.** Replace hologram-live's hand-written OpenAI types with `async-openai`'s in a new `openai-compat` module owned by this repo, so `system_fingerprint`, tools and content parts exist. Add a `Receipt` realization in the `SessionAttestation` shape: operands are model κ, engine κ, prompt κ, params κ, output κ; payload is the detached Ed25519 signature; `signable_bytes` re-encodes with an empty payload. On every response set `system_fingerprint` to model κ plus engine κ, insert `x-hologram-receipt` with the receipt κ, store the receipt in the object store. Claims: a stock `openai` client receives a fingerprint that equals the loaded model's κ; the receipt in the header resolves to a stored object; a response with no engine κ available refuses rather than fingerprints. Planted defect: a fingerprint copied from config rather than derived, caught by the equality test.

**Phase 2, verify.** `GET /v1/receipts/{κ}` returns the receipt. `POST /v1/receipts/{κ}/verify` replays the decode with the deterministic engine and compares bytes, returning confirmed or the first divergent byte. `hologram verify <κ>` prints one line. The comparison predicate is the Tier 1 generated validator, so the judgement is kernel-checked. Claims: a tampered receipt is refused before replay because its κ no longer matches; a forged receipt with a valid κ is refuted at byte N; a genuine receipt replays identically on a second CI host. Planted defect: verify that compares lengths only, caught by the forged-receipt scenario.

**Phase 3, prefix cache.** Node address is `derive_label` over the parent address and the next token, rooted at model κ, engine κ and params κ. Nodes live in the redb store, which has no secondary index today, so the trie uses redb's ordered range over the ASCII κ key, a fourth table, the one extension the store needs. On a request the longest matching prefix is looked up, the remainder computed, the new nodes written. `cached_tokens` reports the matched prefix. Header `x-hologram-cache` is HIT, MISS or PARTIAL. Claims: the second request sharing a prefix reports a nonzero cached count and completes faster by a measured margin; a request under a different model never hits; the count never exceeds the prompt length. Planted defect: a cached count that includes the tail, caught by the bound test.

**Phase 4, verified pull.** The verifying proxy from the research, as a `pull` module. `hologram run <model>` resolves the revision to a commit once, reads each file's sha256 from that commit's tree through huggingface_hub's own info endpoint, streams the file to a temp file while hashing sha256 and blake3, refuses on mismatch, renames into the store, records a signed manifest per repo at commit. Legacy `resolve` and info routes serve stock huggingface_hub clients through `HF_ENDPOINT` with the three legacy headers and no Xet headers, mounted outside the auth middleware so a client's `HF_TOKEN` is never checked against the daemon's own token. Range and HEAD are new, about 80 lines. Claims: a corrupted upstream byte is refused with a diff; a later resolve of the same revision to different bytes is refused; a stock `hf` client downloads unchanged; the model loads offline after first pull. Planted defect: a store write that hashes only the first chunk, caught by the corruption scenario.

**Phase 5, catalog and second protocol.** `GET /v1/models` in Hugging Face's provider format with pricing zero and context length. An `anthropic-compat` module serving `/v1/messages` so Claude Code connects with two environment variables. Listings in order: Open WebUI connection guide, LiteLLM providers file PR, Hugging Face local-apps PR with icon and snippet. Continue's hub is gone, so it is dropped. Claims: `anthropic` SDK completes a streamed message; LiteLLM routes to the daemon from its own config; the models list validates against HF's schema. Planted defect: a model entry without context length, caught by schema validation.

**Phase 6, desktop.** Two pages in the existing Tauri app: Activity and Verification Center. The webview's CSP blocks direct HTTP to the daemon, so each page reads through a new sidecar command that calls the CLI, the pattern every existing page uses. Three checks on the Verification Center: model bytes match the manifest, engine determinism self-test passes, last answer replays. Claims: each check maps to one CLI command whose output the test asserts. Planted defect: a check that reports green without running, caught by a fixture that breaks the model file.

**Phase 7, share.** A `RemoteRegistryProvider` behind hologram-live's existing provider trait, fetching receipts and trie nodes over hologram's `/cas/{κ}` with its existing verify-on-receipt. Spot recompute of a random prefix slice before accepting a foreign node. Tenant salt mixed into the trie root so private prefixes stay unlinkable. kappa-registry attaches here over HTTP only, after its own broken manifest is fixed. Claims: a node fetched from a peer that fails spot recompute is refused; two machines converge on the same node address without contact. Planted defect: a peer node accepted on size alone, caught by the forged-node scenario.

## Risks, ranked

1. **The deterministic engine is on a branch, not main.** `unification/phase-1` of hologram-live-ip, 77 commits ahead. Mitigation: pin the branch rev, make Phase 1 the first consumer, and merge the branch to main before Phase 3.
2. **Both stores are whole-buffer.** Neither hologram-live's `ObjectStore` nor hologram's `KappaStore` streams. Mitigation: the streaming write is written once in Phase 4 and is the only truly new storage code.
3. **PrismPM's gate needs Docker and Linux.** Mitigation: WSL on the founder's machine, CI on ubuntu, decided on day one.
4. **Routers merge, collisions panic.** Mitigation: the boot test in Phase 1 that loads every module.
5. **hologram-live ships no LICENSE files despite declaring dual licence.** Mitigation: an upstream PR adding them, before any binary is published.
6. **Determinism across machines is a claim to earn.** Mitigation: two CI hosts, `build` grade only, never stated beyond what CI produced.

## Deliberately excluded

Hosted tier, accounts, keys, billing, chunking, Xet CAS compatibility, peer mesh beyond `/cas`, Filecoin, decentralised compute, semantic caching. Each is a later backend behind a seam this plan leaves open, and none is needed to make the nine lines in the end state true.
