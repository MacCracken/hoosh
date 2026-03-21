# Contributing to Hoosh

## Development Workflow

1. Fork and clone the repository
2. Create a feature branch from `main`
3. Make changes, add tests
4. Run `make check` to verify locally
5. Submit a pull request

## Local CI Checks

```bash
make check    # fmt + clippy + test + audit + deny
make bench    # run synthetic + e2e benchmarks
make coverage # coverage report
```

## Code Style

- `cargo fmt` — enforced in CI
- `cargo clippy -- -D warnings` — zero warnings policy
- Tests for all public APIs
- Doc comments on all public items

## Commit Messages

Use imperative mood, present tense:
- `add connection pooling to HooshClient`
- `fix token budget overflow on streaming disconnect`
- `update benchmark results for v0.20.4`

## Running Tests

```bash
cargo test                           # all tests (no Ollama needed)
cargo test -- --ignored              # live Ollama tests
cargo bench --bench routing --bench providers  # synthetic benchmarks
cargo bench --bench e2e              # end-to-end (needs Ollama)
```

## Adding a Provider

1. Create `src/provider/<name>.rs`
2. Implement `LlmProvider` trait
3. Add `ProviderType` variant
4. Add feature flag in `Cargo.toml`
5. Register in `ProviderRegistry::register_from_route()`
6. Add to `default_base_url()` in `config.rs`
7. Add tests and update docs

## Updating Benchmarks

After performance-relevant changes, run benchmarks and update docs:

```bash
cargo bench 2>&1 | grep -B1 'time:' | grep -v '^--$' | paste - -
```

Update `docs/development/performance.md` with new median values and the "Last updated" date.
