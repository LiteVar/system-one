#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cache_dir="${SYSTEM_ONE_CACHE:-${project_dir}/.cache/system-one}"

echo "Using model cache: ${cache_dir}"
cargo run --release --manifest-path "${project_dir}/Cargo.toml" -- \
  --cache-dir "${cache_dir}" model pull laya-multilingual
