.PHONY: check fmt clippy test audit deny build doc clean

# Run all CI checks locally
check: fmt clippy test audit

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-targets -- -D warnings

# Run test suite
test:
	cargo test

# Security audit
audit:
	cargo audit

# Supply-chain checks
deny:
	cargo deny check

# Build release
build:
	cargo build --release

# Generate documentation
doc:
	cargo doc --no-deps

# Clean build artifacts
clean:
	cargo clean
