# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work only. Shipped releases
live in [CHANGELOG.md](../../CHANGELOG.md), one entry each; design decisions live
in [ADRs](../decisions/). Nothing here is a record of what was done — if an item
ships, it moves to the CHANGELOG and leaves this file.

**Current**: v2.5.11. The **rust-old parity closeout arc (v2.5.1–v2.5.11) is
complete** — the port is at behavioral parity with the archived Rust reference and
past it. Evidence: [rust-old-parity-review.md](rust-old-parity-review.md).

---

## Open work

### Memory — per-request arena  *(highest priority)*

hoosh's allocator **never frees**. v2.5.11 removed the dominant per-connection
64 KiB allocation (measured: ~128 MB → 2.0 MB of growth over 2000 requests), but
**~1 KiB per request still accumulates** — response string building
(`str_builder`, `to_cstr`, path parsing). At 100 req/s that is ~360 MB/hour, so it
is still an eventual OOM on a long-lived gateway, just a slower one.

Point fixes will not close this. It needs a **per-request arena with
mark/release**: take a mark when a request starts, release it when the response is
written. `alloc_reset` exists but cannot be called per request as-is — the response
cache, audit chain, cost records, routes and health records all live in the same
arena and must survive the request that created them. The design work is deciding
what is request-scoped versus process-scoped and enforcing that split.

Until then, treat a hoosh instance as needing a periodic restart under sustained
load, and measure with `scripts/` + `/proc/<pid>/status` rather than assuming.

### Observability

- **Nested OTLP spans** — provider-forward / cache / retry child spans under the
  inference span. Extends 2.3.5.
- **OTLP/protobuf** — the standard wire format. *Upstream-gated* on a cyrius
  protobuf lib (proposed:
  `cyrius/docs/development/proposals/2026-06-10-protobuf-lib.md`).

### Embeddings

`POST /v1/embeddings` needs its own pass. Two known defects: the handler passes a
port where a base-url cstr is expected, and Ollama's response is forwarded raw
rather than normalized to the OpenAI `{object:"list", data:[…]}` envelope
(rust-old `ollama.rs:314-340`). Deferred from 2.5.3 because fixing the envelope
alone would be half a job.

### Test-suite structure

- **Mirror drift is unguarded.** `tests/hoosh.tcyr` re-implements the logic it
  tests rather than linking `src/` (`src/main.cyr` is a program, not a library).
  That means src and its mirror can diverge while both stay internally consistent
  and the suite stays green — which has happened twice (v2.5.6 pricing
  local-provider ordering, v2.5.7 audit chain-link verification; in both the
  mirror was right and src was wrong). `scripts/coverage.sh` is a floor against
  *unwatched* code, not against drift. Closing this properly means making `src/`
  linkable by tests, which is a structural change worth designing.
- **Split `tests/hoosh.tcyr` / `hoosh.bcyr` into per-topic units** — only if the
  suite keeps growing. Currently workable as single files.

### MCP tools (szál) — *upstream-gated*

`/v1/tools/list` + `/v1/tools/call` are live, but the registry holds only a
`bote_echo` smoke tool until **szál** (58 built-in MCP tools) ships as a Cyrius
distlib. Register them in `mcp_init` alongside `bote_echo` — no transport changes.
([ADR 005](../decisions/005-mcp-via-bote.md).)

### Upstream-gated (sandhi)

- **Connection pooling** — the high-value case is remote TLS-handshake reuse;
  gated on sandhi keep-alive/pooling. (Local loopback connect ≪ inference latency,
  so the local path has low ROI.)
- **Certificate pinning + mTLS** — pinning/mTLS exist
  (`sandhi_tls_policy_new_pinned`) but the high-level `sandhi_http_post`/`_stream`
  client doesn't thread a TLS policy. Filed upstream
  (`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`).

### Upstream-gated (ai-hwaccel)

**`registry_detect_threaded` corrupts the registry.** Its post-passes are called
with the wrong argument — `detect_interconnects(r, …)`, `detect_storage(r)`,
`detect_environment(r)` pass the registry where a `system_io` is expected. The
layouts overlap but differ (`reg {profiles, warnings, system_io, schema}` vs
`sio {interconnects, storage, environment}`), so storage devices land in
`reg.warnings`, `reg.system_io` is overwritten with a `runtime_env`, and on a box
with NVLink/InfiniBand the interconnects are pushed into **`reg.profiles`, the
device list** — corrupting device counts and VRAM totals on exactly the multi-GPU
machines that need placement.

hoosh switched to the **serial** detector in 2.5.9 (34 ms vs 20 ms — not a trade
worth making). **Revert to threaded once fixed.**

### Upstream-gated (cyrius)

- **`cyrius coverage` reports on the vendored stdlib, not the local repo** — filed
  as `cyrius/docs/development/issues/2026-07-23-hoosh-coverage-reports-stdlib-not-local-repo.md`,
  proposing a local-repo default with `--full` and `--min <pct>`. hoosh gates on
  `scripts/coverage.sh` meanwhile.
- **No `sys_exit_group` wrapper** — `sys_exit` is `SYS_EXIT` (thread exit) despite
  a "terminate process" doc comment, which left hoosh alive after a clean shutdown
  until 2.5.11 added a local `hoosh_exit_process`. Replace when upstream provides
  one.
- **`clock_now_ms()` has no vDSO path** — it is a raw `syscall(228)` measuring
  **1.351 µs**, which dominated every hot path until 2.5.11 worked around it with
  a coarse ticker. A vDSO route would let the workaround be removed.

---

## Deferred (external)

### svara — Speech/Audio (migration pending)

STT (Whisper) and TTS (Piper) are migrating from hoosh to **svara**. Hoosh keeps
the provider interface; svara owns the audio pipeline. `/v1/audio/transcriptions`
and `/v1/audio/speech` will not be ported here.

---

## Non-goals

- **Model training** — hoosh is for inference.
- **Model storage** — hoosh doesn't manage model files.
- **Direct GPU compute** — delegated to backends; ai-hwaccel handles detection.
- **Web UI** — hoosh is an API gateway; a dashboard is separate.
- **Audio pipeline** — speech processing belongs to svara.
- **WASM target** — Cyrius doesn't target WASM.

---

<a id="toolchain"></a>

## Process notes

**Toolchain bumps.** On each pin bump, wipe `lib/` and run a clean
`cyrius lib sync` + `cyrius deps`, then the full CI step order, before trusting a
local build — a stale `lib/` masks stdlib module renames (6.1.27 merged
`bigint`/`toml`/`json` → `bayan`). Compiler strictness also tightens: 6.2.11 turned
duplicate same-scope `var` declarations into a hard error.

**Benchmarks are a release gate.** CI runs `./scripts/bench-history.sh` and fails
the build if the suite does not run. `bench-history.csv` is the record. Two
benchmarks (`estimate_tokens_per_provider`, `pool_available`) sit at single-digit
nanoseconds where 1 ns of timer quantization reads as a >10% swing — check a
flagged result across repeats before treating it as a regression.

**Live verification.** `scripts/bench-live.sh` (opt-in, needs a running gateway and
backend) measures end-to-end; its ~5 ms floor is `curl` process startup, not the
gateway, so use it for relative comparisons only.
