# VERIFICATION

Every gate is armed by a planted defect that made it fail, then restored. A gate that has never failed proves nothing. Records keep the exact diagnostic where it is stable.

| Gate | Planted defect | Diagnostic | Restored |
| --- | --- | --- | --- |
| Site rules (`scripts/check_site.py`, SI-01) | A hyphen in the footer: `Apache-2.0 or MIT` | `FAIL hyphen minus in visible text: 'Hologram · Apache-2.0 or MIT'`, exit 1 | Footer text returned to `Apache 2.0 or MIT`; gate prints `ok: no dashes, 111 words of prose, headline exact` |
| Receipt integrity (`cargo test`, RC-02 and the receipt unit tests) | Receipt κ computed over the unsigned bytes instead of the sealed bytes: `kappa: kappa_of(&bound.signable())` | `receipt::tests::sealed_receipt_verifies_and_tampering_is_refused` panicked: `genuine receipt verifies: "kappa blake3:90a15fb9… does not match sealed bytes blake3:8b29bde1…"`; the run stopped before the conformance binary | `kappa: kappa_of(&sealed)`; 3 unit tests and 7 conformance tests pass |

## Gates and what they cover

1. `python scripts/model_write.py --check`: every id in `model/ids.toml` has exactly one scenario whose file, level and statement match, and exactly one test named `conformance_<id>`; ledger claims graded `some-true` name an authority, `build` claims do not; `CONFORMANCE.md` equals its regeneration.
2. `python scripts/check_site.py site/index.html`: no hyphen or dash of any kind in visible text, code samples included; under 120 words of prose; headline exact.
3. `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`.
4. `cargo test --locked -- --test-threads=1`: receipt unit tests and the seven conformance tests, which build the daemon in a temporary directory with the echo engine, import a fixture model, and drive the router directly.

`just vv` runs all four. CI runs the same commands on ubuntu.

## Not yet armed

The Rust CI job has not yet produced a failing run from a planted defect on GitHub itself; the records above are local runs at the commit that introduced Phase 1. The PrismPM template with its pinned devcontainer and digest bound action is adopted in a following step and will re-arm every gate under its own record.
