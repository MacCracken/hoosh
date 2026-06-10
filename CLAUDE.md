# Hoosh — Claude Code Instructions

## Project Identity

**Hoosh** (Persian: intelligence (the word for AI)) — AI inference gateway — multi-provider LLM routing, token budgets, caching, local model serving

- **Type**: Single-binary Cyrius project (server, port 8088)
- **License**: GPL-3.0-only
- **Toolchain**: Cyrius (pinned in `cyrius.cyml`, currently 6.1.27)
- **Version**: SemVer 2.3.0 stable
- **Genesis repo**: [agnosticos](https://github.com/MacCracken/agnosticos)
- **Philosophy**: [AGNOS Philosophy & Intention](https://github.com/MacCracken/agnosticos/blob/main/docs/philosophy.md)
- **Standards**: [First-Party Standards](https://github.com/MacCracken/agnosticos/blob/main/docs/development/applications/first-party-standards.md)
- **Recipes**: [zugot](https://github.com/MacCracken/zugot) — takumi build recipes

## Consumers

All AGNOS apps (LLM inference), daimon (inference routing)

**Note**: 15 LLM providers. Uses murti for local inference. OpenAI-compatible API.

**Data files**: `data/cloud_pricing.json` + `data/models.json` are vendored from
ai-hwaccel (power `/v1/hardware/cost` + `/compatible-models`). Re-sync them when
bumping the ai-hwaccel tag. `models.json` MUST be a **top-level JSON array** —
ai-hwaccel's `load_models` only parses the first object from a `{"models":[…]}`
wrapper (guarded by the `hardware_data_files` test).

## Development Process

### P(-1): Scaffold Hardening (before any new features)

0. Read roadmap, CHANGELOG, and open issues — know what was intended before auditing what was built
1. Test + benchmark sweep of existing code
2. Cleanliness check: `cyrius fmt <file> --check` (all `src/**/*.cyr`, `tests/*.tcyr`, `tests/*.bcyr`), `cyrius lint` (no `warn` lines), `cyrius vet src/main.cyr`, `cyrius deny src/main.cyr`
3. Get baseline benchmarks (`./scripts/bench-history.sh`)
4. Initial refactor + audit (performance, memory, security, edge cases)
5. Cleanliness check — must be clean after audit
6. Additional tests/benchmarks from observations
7. Post-audit benchmarks — prove the wins
8. Repeat audit if heavy
9. Documentation audit — ADRs, source citations, guides, examples (see Documentation Standards in first-party-standards.md)

### Development Loop (continuous)

1. Work phase — new features, roadmap items, bug fixes
2. Cleanliness check: `cyrius fmt <file> --check` (all `src/**/*.cyr`, `tests/*.tcyr`, `tests/*.bcyr`), `cyrius lint` (no `warn` lines), `cyrius vet src/main.cyr`, `cyrius deny src/main.cyr`
3. Test + benchmark additions for new code
4. Run benchmarks (`./scripts/bench-history.sh`)
5. Audit phase — review performance, memory, security, throughput, correctness
6. Cleanliness check — must be clean after audit
7. Deeper tests/benchmarks from audit observations
8. Run benchmarks again — prove the wins
9. If audit heavy → return to step 5
10. Documentation — update CHANGELOG, roadmap, docs, ADRs for design decisions, source citations for algorithms/formulas, update docs/sources.md, guides and examples for new API surface, verify recipe version in zugot
11. Version check — VERSION (source of truth; `cyrius.cyml` tracks it via `${file:VERSION}`), recipe (in zugot) all in sync. Use `./scripts/version-bump.sh <v>`.
12. **Release gate — benchmarks are mandatory** (see Key Principles); CI runs `./scripts/bench-history.sh` and fails the build if the suite does not run. Do not tag a release without a green bench run unless the maintainer explicitly waives it.
13. Return to step 1

### Key Principles

- **Benchmarks are mandatory for every release.** Numbers don't lie; the CSV history (`bench-history.csv`) is the proof. CI runs `./scripts/bench-history.sh` as a hard release gate — a release may ship without a fresh bench run **only** with an explicit maintainer waiver (`CYRIUS_SKIP_BENCH=1` repo var). Never skip benchmarks before claiming a performance change.
- **Tests + benchmarks are the way.** Keep `tests/hoosh.tcyr` green (`cyrius test`) and `tests/hoosh.bcyr` running (`cyrius bench`).
- **Own the stack.** If an AGNOS project wraps an external capability, depend on the AGNOS project (e.g. hardware detection via `ai-hwaccel`, consumed as its `dist/` bundle).
- **No magic.** Every operation is measurable, auditable, traceable.
- **Pin the toolchain in `cyrius.cyml`.** Build/CI use the pinned `cyrius` (`~/.cyrius/versions/<pin>/bin/cyrius`); `cyrius lib sync` + `cyrius deps` vendor stdlib + deps into gitignored `lib/`.
- **Native TLS is the default — build with no TLS flag** (`cyrius build src/main.cyr build/hoosh`). Since cyrius 6.1.21 / sandhi native-default, the native TLS stack is compiled in and active by default, so the gateway never fdlopen-loads libssl/glibc — whose brk-malloc arena + TLS machinery **SIGSEGVs on repeated remote HTTPS requests**. **Do NOT pass `-D CYRIUS_TLS_LIBSSL`** (the libssl-only opt-out) — it routes remote HTTPS back through the crash-prone libssl bridge. `main()` calls `sandhi_tls_use_native()` and prints a startup WARNING if a libssl-only build ever disabled native. The legacy `-D CYRIUS_TLS_NATIVE` flag is deprecated but harmless (native is already the default). CI + release build without any TLS flag.
- **Syscalls via `sys_*` wrappers** (`sys_write`/`sys_read`/`sys_close`/`sys_socket`/`sys_connect`/`sys_exit`) — never raw `syscall(N, …)` numbers or bare `SYS_*` enum members.
- **Single-pass include order** — modules must be `include`d before first use (`include "lib/ai-hwaccel.cyr"` before any module that calls into it).
- **Build strings with `str_builder`**, allocate with `alloc` only when you must — avoid temporary allocations.
- **Vec arena over HashMap** — when indices are known, direct access beats hashing.
- **Do not panic in library code** — return error codes / `Result`; reserve fatal exits for `main`.

## DO NOT
- **Do not commit or push** — the user handles all git operations (commit, push, tag)

- **NEVER use `gh` CLI** — use `curl` to GitHub API only
- Do not add unnecessary dependencies — keep it lean
- Do not panic in library code — return error codes / `Result`
- Do not skip benchmarks before claiming performance improvements, or before a release (CI-enforced gate)
- Do not commit `build/` or `lib/` (gitignored, regenerated by `cyrius lib sync`/`cyrius deps`); **do** commit `cyrius.lock`

## Documentation Structure

```
Root files (required):
  README.md, CHANGELOG.md, CLAUDE.md, CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md, LICENSE

docs/ (required):
  architecture/overview.md — module map, data flow, consumers
  development/roadmap.md — completed, backlog, future, v1.0 criteria

docs/ (when earned):
  adr/ — architectural decision records
  guides/ — usage guides, integration patterns
  examples/ — worked examples
  standards/ — external spec conformance
  compliance/ — regulatory, audit, security compliance
  sources.md — source citations for algorithms/formulas (required for science/math crates)
```
