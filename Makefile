.PHONY: check fmt clippy test audit deny vet bench coverage build release doc clean

# Run all CI checks locally
check: fmt clippy test audit deny

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

# Dependency audit chain
vet:
	cargo vet

# Run benchmarks (synthetic + e2e)
bench:
	cargo bench --bench routing --bench providers --bench e2e -- --noplot

# Coverage report
coverage:
	cargo llvm-cov --lcov --output-path lcov.info

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

# Generate documentation
doc:
	cargo doc --no-deps

# Clean build artifacts
clean:
	cargo clean
	rm -f hoosh-*.tar.gz hoosh-*.sha256
