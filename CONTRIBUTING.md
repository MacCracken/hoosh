# Contributing to Hoosh

Hoosh is a single-binary **Cyrius** project. Build/CI use the toolchain pinned in
`cyrius.cyml` (`~/.cyrius/versions/<pin>/bin/cyrius`). See
[CLAUDE.md](CLAUDE.md) for the full development process and
[docs/architecture/overview.md](docs/architecture/overview.md) for the module map.

## Project Layout

```
src/
├── main.cyr            CLI (serve/models/health/infer/info/version) + HTTP route dispatch + accept loop
├── lib/*.cyr           29 modules — see docs/architecture/overview.md for the table
└── vendor/             committed distlib bundles (bote-core.cyr, majra.cyr)
tests/
├── hoosh.tcyr          self-contained test suite (cyrius test)
└── hoosh.bcyr          microbenchmark suite (cyrius bench)
scripts/                bench-history.sh, version-bump.sh, sync-bote.sh
docs/                   architecture, decisions (ADRs), development
```

## Development Workflow

1. Branch from `main` (the maintainer handles all commits/pushes/tags).
2. Make changes; add tests to `tests/hoosh.tcyr` and a benchmark to
   `tests/hoosh.bcyr` for new hot-path code.
3. Run the cleanliness gate + tests + benchmarks (below) — all must be green.
4. Update `CHANGELOG.md` (+ roadmap/ADR/docs for design changes).
5. Open a PR.

## Local checks (the CI gate)

```bash
CYR=~/.cyrius/versions/$(grep '^cyrius' cyrius.cyml | grep -oE '[0-9.]+')/bin/cyrius

# Cleanliness — must be clean
$CYR fmt <file> --check          # all src/**/*.cyr, tests/*.tcyr, tests/*.bcyr
$CYR lint src/main.cyr           # no `warn` lines
$CYR vet src/main.cyr            # no UNTRUST outside src/ + cyrius.lock
$CYR deny src/main.cyr           # no policy violations

# Tests + benchmarks (benchmarks are a mandatory release gate)
$CYR test tests/hoosh.tcyr
./scripts/bench-history.sh bench-history.csv benchmarks.md
```

After a toolchain pin bump, wipe `lib/` and re-sync (`cyrius lib sync` +
`cyrius deps`) before trusting a local build — stale `lib/` masks stdlib renames.

## Code Style

Match the surrounding code. Key idioms (full list in CLAUDE.md → Key Principles):

- Build strings with `str_builder`; `alloc` only when necessary (avoid temporaries).
- Syscalls via `sys_*` wrappers — never raw `syscall(N, …)` numbers.
- Single-pass include order — modules `include`d before first use.
- Vec arena over HashMap when indices are known.
- **Do not panic in library code** — return error codes / `Result`; fatal exits
  only in `main`.
- The self-contained `tests/hoosh.tcyr` may only call stdlib + vendored symbols
  (a call to an undefined `src` fn compiles but SIGILLs at runtime).

## Commit Messages

Imperative mood, present tense — e.g. `add connection pooling`, `fix token budget
overflow on streaming disconnect`. (The maintainer makes commits.)

## Adding a Provider

1. Add a `PROV_*` entry + `provider_name` case in `src/lib/types.cyr` (bump
   `PROV_COUNT`).
2. Wire request shaping / response extraction in `src/lib/provider.cyr` (local
   raw-socket vs. remote sandhi path; auth headers for remote).
3. Default base URL + config handling in `src/lib/config.cyr` / `route.cyr`.
4. Add pricing (`src/lib/pricing.cyr`) + model metadata (`src/lib/metadata.cyr`)
   if it should participate in the cost optimizer.
5. Add a mock-backend test to `tests/hoosh.tcyr` and update docs.

## Versioning

`VERSION` is the source of truth (`cyrius.cyml` tracks it via `${file:VERSION}`).
Use `./scripts/version-bump.sh <v>` (updates VERSION, CLAUDE.md, `types.cyr`
`HOOSH_VERSION`, and a CHANGELOG stub). Keep the zugot recipe version in sync.

## License

By contributing, you agree that your contributions will be licensed under the
[GPL-3.0-only](LICENSE) license.
