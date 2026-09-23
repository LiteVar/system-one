# Changelog

All notable changes to this project will be documented here.

## [Unreleased]

- Continue native parity testing against upstream Laya reference vectors.

## [0.1.1] - 2026-09-23

- Batch Laya encoder inputs by complete sequence so no `llama_encode` call exceeds the configured micro-batch token limit.
- Preserve question order, token positions, sequence IDs, and per-question embeddings across encoder batches.
- Add regression coverage for multi-question requests exceeding 1024 tokens and requests at the configured maximum limits.

## [0.1.0] - 2026-09-22

- Added the native Rust System One runtime and CLI.
- Added Laya Multilingual tokenizer, Safetensors decision head, and llama.cpp FFI.
- Added model/runtime cache management, offline mode, and checksum validation.
- Added `/health`, `/v1/models`, and `/v1/systemone` HTTP endpoints.
- Added cross-platform CI and managed llama.cpp runtime provisioning.
