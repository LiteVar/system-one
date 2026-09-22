# Contributing to System One

Thank you for contributing. System One is a Rust-first, local-first runtime; keep
the public API model-independent and keep model-specific behavior inside a backend.

## Development setup

Install a stable Rust toolchain (Rust 1.85 or newer), then run:

```sh
cargo test --all-targets --locked
cargo clippy --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
cargo build --release --locked
```

The unit tests do not download model files. To run the optional real-model smoke
test, first cache the model and runtime, then set `SYSTEM_ONE_E2E_CACHE` to that
cache directory:

```sh
cargo run --release -- model pull laya-multilingual
SYSTEM_ONE_E2E_CACHE="$HOME/.cache/system-one" cargo test --test e2e_smoke -- --nocapture
```

Do not commit model weights, runtime archives, cache directories, `.part` files, or
user state/questions. Keep tests deterministic and avoid logging raw input data.

## Pull requests

Before opening a pull request:

1. Add or update tests for behavior changes.
2. Update `README.md` or `docs/` when the CLI, API, cache, or runtime contract changes.
3. Run formatting, Clippy, tests, and a release build locally.
4. Explain platform-specific behavior and any compatibility impact.

Changes that alter tokenization, tensor math, llama.cpp ABI handling, probabilities,
or calibration should include a reference vector or a clear explanation of how parity
was checked.

## Model and runtime changes

Model assets and llama.cpp archives are pinned by revision, size, and SHA-256. Any
pin change must document the upstream source, target platforms, checksum, and the
validation performed against the native inference path.
