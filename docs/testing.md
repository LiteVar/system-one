# Testing and release validation

## Fast local checks

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
cargo package --locked --allow-dirty --list
```

## Native model smoke test

The repository's normal test suite is intentionally network-free. A real-model smoke
test is available when a valid local cache is supplied:

```sh
cargo run --release -- model pull laya-multilingual
SYSTEM_ONE_E2E_CACHE="$HOME/.cache/system-one" cargo test --test e2e_smoke -- --nocapture
```

The test runs the built executable in offline mode and verifies that the native
tokenizer, llama.cpp encoder, decision head, typed answer, probabilities, and token
usage are wired together. It skips cleanly when `SYSTEM_ONE_E2E_CACHE` is not set.

This smoke test is not a substitute for golden reference vectors. Changes to
tokenization, tensor operations, calibration, or the llama.cpp ABI should add or
update independently generated reference cases before release.

## Release checklist

Before tagging a release:

1. Run the fast checks and the native smoke test on at least one CPU platform.
2. Validate the model asset and runtime checksums against the intended upstream sources.
3. Run the HTTP endpoints with a representative choice, score, and noul request.
4. Confirm `--offline` performs no network access when all assets are cached.
5. Review `NOTICE`, model terms, release notes, and supported platform claims.
