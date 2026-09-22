## Summary

<!-- What changed and why? -->

## Validation

- [ ] `cargo fmt --all -- --check`
- [ ] `cargo clippy --all-targets --all-features --locked -- -D warnings`
- [ ] `cargo test --all-targets --locked`
- [ ] `cargo build --release --locked`
- [ ] Native model smoke test, if inference behavior changed

## Compatibility and security

- [ ] Updated README/docs for user-visible changes
- [ ] No model weights, runtime archives, or user input are committed
- [ ] Reviewed cross-platform and llama.cpp ABI impact
