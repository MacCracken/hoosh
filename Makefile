.PHONY: check fmt clippy test audit deny vet semver bench bench-history coverage build release doc fuzz msrv clean

# Run all CI checks locally
check: fmt clippy test audit deny

# Format check
fmt:
	cargo fmt --all -- --check

# Lint (zero warnings)
clippy:
	cargo clippy --all-features --all-targets -- -D warnings

# Run test suite
test:
	cargo test --all-features

# Security audit
audit:
	cargo audit

# Supply-chain checks
deny:
	cargo deny check

# Dependency audit chain
vet:
	cargo vet

# SemVer compatibility check (requires cargo-semver-checks)
semver:
	cargo semver-checks check-release

# MSRV check
msrv:
	cargo +1.89 check
	cargo +1.89 test

# Run benchmarks (synthetic)
bench:
	cargo bench --bench routing --bench providers --bench hot_path --bench e2e -- --noplot

# Run benchmarks with CSV history tracking
bench-history:
	./scripts/bench-history.sh

# Coverage report (HTML + lcov)
coverage:
	cargo llvm-cov --all-features --lcov --output-path lcov.info
	cargo llvm-cov --all-features --html

# Fuzz deserialization paths (requires cargo-fuzz, nightly)
fuzz:
	cargo +nightly fuzz run fuzz_inference_request -- -max_total_time=300
	cargo +nightly fuzz run fuzz_message_content -- -max_total_time=300

# Build release
build:
	cargo build --release

# Build and package release artifact
release:
	@VERSION=$$(cat VERSION | tr -d '[:space:]'); \
	cargo build --release; \
	tar czf "hoosh-$${VERSION}-linux-amd64.tar.gz" -C target/release hoosh; \
	sha256sum "hoosh-$${VERSION}-linux-amd64.tar.gz" > "hoosh-$${VERSION}-linux-amd64.tar.gz.sha256"; \
	echo "Packaged hoosh-$${VERSION}-linux-amd64.tar.gz"

# Generate documentation (warnings as errors)
doc:
	RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# Clean build artifacts
clean:
	cargo clean
	rm -f hoosh-*.tar.gz hoosh-*.sha256
