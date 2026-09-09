set shell := ["bash", "-c"]

default:
    @just --list

# Regenerate CONFORMANCE.md from model/*.toml.
model-write:
    python scripts/model_write.py

# Check the claim register, the scenario bijection, and the homepage rules.
model-check:
    python scripts/model_write.py --check
    python scripts/check_site.py site/index.html

fmt:
    cargo fmt --all -- --check

clippy:
    cargo clippy --all-targets --locked -- -D warnings

test:
    cargo test --locked -- --test-threads=1

# The only definition of green.
vv: model-check fmt clippy test
    @echo "vv: green"
