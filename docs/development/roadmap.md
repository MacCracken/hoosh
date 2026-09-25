# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work only. Shipped releases
live in [CHANGELOG.md](../../CHANGELOG.md), one entry each; design decisions live
in [ADRs](../decisions/). Nothing here is a record of what was done — if an item
ships, it moves to the CHANGELOG and leaves this file.

**Current**: v2.7.1. The **rust-old parity closeout arc (v2.5.1–v2.5.11) is
complete** — the port is at behavioral parity with the archived Rust reference and
past it. Evidence: [rust-old-parity-review.md](rust-old-parity-review.md); 2.7.0 and 2.7.1 closed what remained
([rust-old-retirement.md](rust-old-retirement.md)).

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

### Retire `rust-old/` *(after 2.7.1 is tagged)*

2.7.0 and 2.7.1 closed the last parity gaps with the archived Rust tree. Delete `rust-old/` once 2.7.1 is
tagged; [rust-old-retirement.md](rust-old-retirement.md) has the checklist and the list of
deliberate non-ports, so nothing needs the Rust tree afterwards.

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
`bote_echo` smoke tool. szál's Cyrius port is done (2.1.2, 54 tools on the same
bote API), but it has no dist bundle yet, and its own names collide with upstream
majra, bote-core and hoosh (`STEP_*` with different values, `step_result_new` and
`cache_new` with different arities). Filed 2026-09-25 as
`szal/docs/development/issues/2026-09-25-hoosh-consumer-bundle.md`, which asks for
`szal_` renames, a bundle without the vendored libraries, and a
register-into-an-existing-dispatcher entry point. Then vendor it at
`src/vendor/szal-mcp.cyr` and register the tools in `mcp_init` next to `bote_echo`,
with no transport changes. ([ADR 005](../decisions/005-mcp-via-bote.md).)

### Upstream-gated (sandhi)

- **Remote SSE keep-alive** — local streams send `: keep-alive` every 15 s of
  silence (2.7.0); remote streams cannot, because `sandhi_http_stream` gives the
  caller no turn while the upstream is quiet. Filed 2026-09-25 as
  `sandhi/docs/development/issues/2026-09-25-http-stream-no-idle-hook.md`
  (proposes `sandhi_http_options_idle_ms` / `_idle_cb`).
- **Connection pooling** — no longer upstream-gated: sandhi has
  `sandhi_http_options_pool` (policy-bound requests bypass it). The high-value case
  is remote TLS-handshake reuse; adopt it when that cost shows up. (Local loopback
  connect ≪ inference latency, so the local path has low ROI.)

### Hardware detection — threaded detector *(low priority)*

No longer upstream-gated. ai-hwaccel 2.3.25 fixed the post-pass bug that moved
2.5.9 to the serial detector, and 2.6.12 checked the fix on ai-hwaccel 2.4.0. The
serial and threaded registries match except in profile order. hoosh still runs
the serial detector for two reasons:

- **Profile order.** The threaded path lists the sysfs backends (ROCm, Intel NPU,
  AMD XDNA, TPU, …) before the CLI ones (CUDA, Gaudi, Neuron, Vulkan, oneAPI,
  Apple). `POST /v1/hardware/requirement-match` reports the *first* matching
  profile and `/v1/hardware/simulate`'s `remove_count` drops the *first* N
  accelerators, so their answers would change on mixed hosts. An Intel NPU +
  NVIDIA laptop would report `Intel NPU` for `any-accelerator` instead of the GPU.
- **Little to gain.** Serial takes 22.5 ms and threaded 21.2 ms on the dev host
  (medians of 15 alternated rounds). `vulkaninfo` is ~20 ms of either, and threads
  cannot split a single probe.

Revisit only if startup time matters on hosts with several slow probes
(nvidia-smi and vulkaninfo together). The precondition is making those two
consumers order-independent, or sorting profiles into serial order after
detection. `_hw_detect` in `src/lib/hardware.cyr` has the details.

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
