# Contributing to Hoosh

## Project Layout

```
src/
├── main.rs              # CLI entry point (serve, models, infer, health, info)
├── lib.rs               # Public API exports
├── inference/           # Core types: InferenceRequest, InferenceResponse, MessageContent, batch
├── provider/            # LlmProvider trait + 14 backends (Ollama, OpenAI, Anthropic, ...)
│   ├── mod.rs           # Trait, ProviderType, ProviderRegistry
│   ├── openai_compat.rs # Shared OpenAI-compatible streaming base
│   ├── metadata.rs      # Model registry (63 models, tiers, modalities)
│   └── retry.rs         # Jittered exponential backoff
├── server/              # Axum HTTP server, handlers, types
├── router.rs            # Provider selection, load balancing, DLP-aware routing
├── cache/               # Response cache (TTL, stats, semantic, warming)
├── context/             # Token counting, context compaction, prompt compression
├── cost/                # Pricing table, cost tracking, cost optimizer
├── budget/              # Token pool management (reserve/commit/release)
├── dlp/                 # PII scanning, classification, privacy-aware routing
├── audit.rs             # HMAC-SHA256 tamper-proof audit chain
├── health.rs            # Background health checks, failover
├── hardware.rs          # GPU/TPU/NPU detection (ai-hwaccel)
├── middleware/           # Auth (bearer token), rate limiting (sliding window)
├── tools/               # MCP bridge, tool definitions, format conversion
├── config.rs            # hoosh.toml parsing
├── error.rs             # HooshError enum
├── events.rs            # Provider event bus (majra pubsub)
├── metrics.rs           # Prometheus counters
├── queue.rs             # Priority inference queue
├── client.rs            # HooshClient HTTP wrapper
└── tests/               # Integration + conformance test suite
```

## Development Workflow

1. Fork and clone the repository
2. Create a feature branch from `main`
3. Make changes, add tests
4. Run `make check` to verify locally
5. Submit a pull request

## Local CI Checks

```bash
make check         # fmt + clippy + test + audit + deny
make bench         # synthetic benchmarks
make bench-history # benchmarks with CSV history tracking
make coverage      # coverage report (HTML + lcov)
make semver        # semver compatibility check
make fuzz          # fuzz deserialization paths (requires nightly)
make msrv          # MSRV check (1.89)
make doc           # docs with warnings-as-errors
```

## Code Style

- `cargo fmt` — enforced in CI
- `cargo clippy --all-features --all-targets -- -D warnings` — zero warnings
- `#[non_exhaustive]` on all public enums
- `#[must_use]` on all pure functions
- `#[inline]` on hot-path functions
- `write!` over `format!` on hot paths
- Tests for all public APIs
- Doc comments with examples on all public items
- No `unwrap()` or `panic!()` in library code

## Commit Messages

Use imperative mood, present tense:
- `add connection pooling to HooshClient`
- `fix token budget overflow on streaming disconnect`
- `update benchmark results for v1.0.0`

## Pull Request Requirements

- All CI checks pass (fmt, clippy, test, audit, deny, semver)
- New public API items have doc comments with examples
- New features have tests (target 85%+ coverage)
- Performance-sensitive changes include benchmark results
- CHANGELOG.md updated if user-facing

## Running Tests

```bash
cargo test --all-features              # all tests (no providers needed)
cargo test -- --ignored                # live Ollama tests
cargo test -- conformance              # OpenAI API conformance suite
cargo bench --bench routing --bench providers --bench hot_path --bench e2e  # benchmarks
```

## Adding a Provider

1. Create `src/provider/<name>.rs`
2. Implement `LlmProvider` trait (`infer`, `infer_stream`, `list_models`, `health_check`, `embeddings`)
3. Add `ProviderType` variant (with `is_local()` and `supports_streaming()`)
4. Add feature flag in `Cargo.toml`
5. Register in `ProviderRegistry::register_from_route()`
6. Add to `default_base_url()` in `config.rs`
7. Add model entries to `ModelMetadataRegistry::load_defaults()`
8. Add pricing to `cost/mod.rs` `PRICING` table
9. Add mock server tests
10. Update docs

## Updating Benchmarks

After performance-relevant changes:

```bash
./scripts/bench-history.sh   # runs benchmarks, appends CSV, generates markdown
```

Results tracked in `bench-history.csv`. Markdown summary in `benchmarks.md`.

## Versioning

Version is tracked in three places (kept in sync by `scripts/version-bump.sh`):
- `Cargo.toml` — package version
- `VERSION` — plain text version file
- `CHANGELOG.md` — release notes

```bash
./scripts/version-bump.sh 1.2.0
```

## License

By contributing, you agree that your contributions will be licensed under the [GPL-3.0-only](LICENSE) license.
