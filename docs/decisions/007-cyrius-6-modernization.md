# ADR-007: Cyrius 6.0 Toolchain & Scaffolding Modernization

**Status:** Accepted
**Date:** 2026-06-04

## Context

Hoosh was ported from Rust to Cyrius at v2.0.0 but kept the old Cyrius (4.5.0)
scaffolding and several Rust-era leftovers (a `cargo`-based `bench-history.sh`, a
`Cargo.toml`-driven `version-bump.sh`, `tarpaulin` coverage files, criterion
`.gitignore` entries). Meanwhile the sibling Cyrius projects **ai-hwaccel**
(2.3.7, cyrius 6.0.54) and **patra** (1.10.3, cyrius 6.0.3) had moved to the
current 6.0.x conventions. The dependency `ai-hwaccel` had also advanced from
2.0.0 to 2.3.7 and now ships a single-file `dist/ai-hwaccel.cyr` bundle.

Building hoosh under the modern toolchain surfaced real stdlib drift: syscall
constants are no longer global, `sigil`'s `hmac_sign` was removed, `ct_eq` was
renamed, `vec_new` lost its capacity argument, and `toml_get_sections`/`toml_get`
now take **cstr** keys (which silently broke config parsing).

## Decision

### Toolchain & manifest
- Pin **cyrius 6.0.57** and **ai-hwaccel 2.3.7** in a new **`cyrius.cyml`**
  (replacing `cyrius.toml`), with `version = "${file:VERSION}"` so VERSION is the
  single source of truth, plus a `repository` field.
- **Retire `.cyrius-toolchain`** — the pin lives only in `cyrius.cyml`; CI reads
  it via `grep` and installs through the canonical installer.

### Dependency consumption
- Consume ai-hwaccel as its **distlib bundle**:
  `[deps.ai-hwaccel] modules = ["dist/ai-hwaccel.cyr"]`, vendored by
  `cyrius deps` into `lib/ai-hwaccel.cyr` and pulled in with
  `include "lib/ai-hwaccel.cyr"` from `src/main.cyr`. This replaces the old
  per-source-module list. **hoosh itself publishes no bundle** — it is a server
  binary consumed over its HTTP API, so there is no `[lib]`/`cyrius distlib`/
  `dist/`.
- Cyrius does not resolve transitive deps, so the stdlib modules the bundle and
  `sigil` need (`thread`, `thread_local`, `ct`, `keccak`) are listed explicitly
  in `[deps] stdlib`.

### Source idioms (6.0.x)
- Syscalls go through the `sys_*` wrappers (`sys_write`/`sys_read`/`sys_close`/
  `sys_socket`/`sys_connect`/`sys_exit`), never raw `syscall(N, …)` or bare
  `SYS_*` enum members.
- Audit HMAC uses `hmac_sha256(...)` + `hex_encode` (new `_hmac_hex` helper).
- Config and tests pass **cstr** keys to `toml_get*` (no `str_from(...)` wrapper).

### Benchmarks as a release gate
- Benchmarks are **mandatory for every release**. CI runs
  `./scripts/bench-history.sh` and fails if the suite does not run or records no
  data. The only escape is an explicit maintainer waiver
  (`CYRIUS_SKIP_BENCH=1`). See CLAUDE.md → Key Principles / Development Loop.

### CI/release & scripts
- Workflows use the canonical installer (pin from `cyrius.cyml`),
  `cyrius lib sync` + `cyrius deps`, and hard `fmt`/`lint`/`vet` gates; release
  verifies tag == VERSION == `${file:VERSION}` and presence in the changelog.
- `bench-history.sh` parses `cyrius bench` output; `version-bump.sh` drives
  VERSION + CLAUDE.md + CHANGELOG. No `cargo` anywhere.

## Consequences

- **Positive:** hoosh builds/tests (231/231)/benchmarks clean on a current,
  pinned toolchain; config parsing and the audit chain are correct again; the
  build/CI/scripts match the sibling repos, lowering maintenance friction; the
  bench gate makes performance regressions hard to ship unnoticed.
- **Negative / cost:** the cyrius pin must be advanced deliberately (stdlib drift
  recurs at major jumps); `cyrius.lock` is now tracked and must be refreshed when
  the ai-hwaccel tag moves.
- **Deferred:** broader sibling conventions (per-topic test/bench split,
  `docs/development/state.md`, fuzz harnesses, security-pattern CI scan) are
  logged as 2.1.x backlog in the roadmap, not done here.
