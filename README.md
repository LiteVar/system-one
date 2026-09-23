# System One

[English](README.md) | [简体中文](README-zh_CN.md)

System One is an open-source, local-first, cross-platform runtime for System One decision models. It exposes a Jev-compatible `POST /v1/systemone` API and a native CLI without requiring Python, PyTorch, Transformers, Conda, or a virtual environment.

The first backend is Laya Multilingual. The public API is intentionally model-independent; Laya is one implementation of the `DecisionModel` abstraction.

This project is independent and is not affiliated with or endorsed by TypeSafe AI, Jev, or any other upstream project.

## Status

This is the `0.1.1` release. The runtime contains the native tokenizer,
Laya decision head, model cache, Jev-shaped API, CLI, managed llama.cpp provisioning,
and dynamic llama.cpp FFI. The first run downloads the pinned model assets and, when
no compatible library is found, a pinned CPU/Metal/Vulkan llama.cpp runtime for the
current platform. The selected runtime is kept resident for the process lifetime.

The native path has been smoke-tested end to end on macOS arm64. Cross-platform CI
checks source compilation and release builds for the supported targets; hardware
backend and reference-vector validation remain release checklist items.

See [docs/laya-runtime.md](docs/laya-runtime.md) for the exact reference behavior and compatibility notes.

## Quick start

```sh
cargo run -- model pull laya-multilingual
cargo run -- serve
```

The server binds to `127.0.0.1:8080` by default.

If startup assets are unavailable, the server still exposes `/health` and
`/v1/models` with `model: "unavailable"`; inference requests return HTTP 503 with a
diagnostic error. This makes offline/container health checks useful without hiding
the missing model or runtime.

To use an existing llama.cpp build instead:

```sh
system-one --llama-cpp-home /path/to/llama.cpp serve
```

Runtime discovery order is explicit `--llama-cpp-home`, `SYSTEM_ONE_LLAMA_CPP_HOME`,
`SYSTEM_ONE_LLAMA_LIBRARY`, a library on `PATH`, the local runtime cache, then the
pinned managed download. `system-one runtime info` reports the selected library.

An optional TOML configuration file is read from the platform config directory at
`system-one/config.toml`; use `--config PATH` or `SYSTEM_ONE_CONFIG` to select one.
Command-line and environment values override the file:

```toml
cache_dir = "/var/tmp/system-one"
model = "laya-multilingual"
device = "cpu"

[runtime.llama_cpp]
home = "/opt/llama.cpp"
```

```sh
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "model": "laya-multilingual",
    "state": "The user wants to restart nginx",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What should happen next?",
        "criteria": {
          "execute": "Execute the restart",
          "confirm": "Ask for confirmation",
          "reject": "Reject the request"
        }
      }
    }
  }'
```

## Offline mode

```sh
system-one serve --offline
```

Offline mode never attempts to download model files or llama.cpp. Missing assets are reported explicitly.

## Model and runtime cache

The default cache uses the platform cache directory. Override it with
`--cache-dir /path/to/cache` or `SYSTEM_ONE_CACHE`. Model files are stored under a
revisioned directory and are SHA-256 checked before reuse. Downloads use `.part` files,
and invalid or interrupted files are never treated as complete.

The model consists of three independent assets: the Q8_0 encoder GGUF, the Laya
decision head, and `tokenizer.json`. The GGUF is not treated as a complete decision
model. `scripts/download-model.sh` is a convenience wrapper around the built-in model
cache command.

## API shape and deviations

`POST /v1/systemone` accepts `state` as a string, object, or array and maps question
definitions to Laya's `choice`, `score`, and `noul` types. Choice accepts either
`criteria`/`options` as an array or a label-to-description object. Results include
probabilities, normalized entropy confidence for choice/score, Noul probability and
confidence, and the Laya action probability.

`GET /health` reports process liveness separately from model availability. A loaded
model returns `status: "ok"`; an unavailable model returns `status: "degraded"` and
the startup diagnostic. `GET /v1/models` exposes a standard-compatible `models`
list and retains the local runtime diagnostics in the `data` list.

This is a Jev-shaped adapter, not a claim of official Jev protocol certification. The
adapter is isolated from the model backend so additional decision models can be added
without changing the public inference contract.

## Architecture

```text
HTTP / CLI
   -> model-independent decision adapter
   -> Laya backend
      -> Hugging Face Rust tokenizer
      -> llama.cpp encoder FFI -> per-token hidden states
      -> native safetensors decision head
      -> calibration -> typed probabilistic answer
```

The encoder context is serialized behind a mutex because llama.cpp context access is
not assumed to be safe for unrestricted concurrent use; independent head computation
and HTTP request handling remain asynchronous around that critical section.
The current resident context accepts up to 16 questions per request and up to 1024
tokens per question; requests beyond those limits return a structured validation error.

## Supported platforms

The baseline is CPU-capable on macOS arm64/x86_64, Windows x86_64, and Linux x86_64/
arm64. The managed runtime also provides Metal-capable macOS packages and Vulkan
packages where upstream binaries exist. CUDA is available through an explicitly
installed runtime and the pinned Windows CUDA package; CPU remains the fallback.

Build the native executable with:

```sh
cargo build --release
```

## Development and testing

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features --locked -- -D warnings
cargo test --all-targets --locked
cargo build --release --locked
```

The normal test suite is network-free. To opt into the native model smoke test, set
`SYSTEM_ONE_E2E_CACHE` to a cache containing the validated model and llama.cpp
runtime:

```sh
SYSTEM_ONE_E2E_CACHE="$HOME/.cache/system-one" cargo test --test e2e_smoke -- --nocapture
```

See [docs/testing.md](docs/testing.md) for the release validation checklist and
[CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements.

## Releases

Pushing a `v*` tag runs the release workflow and publishes archives for macOS arm64,
macOS x86_64, Linux x86_64, Linux arm64, and Windows x86_64, together with
`SHA256SUMS`. The executable does not bundle model weights; the first run provisions
the model and runtime into the user cache.

## License

The Rust source is licensed under the MIT License and is copyright LiteVar. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Model
files and runtime libraries retain their upstream licenses and are not bundled in
this repository. Project homepage: <https://github.com/LiteVar/system-one>.
