# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work. Shipped releases are
summarized below with per-release detail in [CHANGELOG.md](../../CHANGELOG.md).
Design decisions live in [ADRs](../decisions/).

---

## Shipped

The Cyrius port reached parity with the `rust-old` reference and then extended it.
One line per release; see CHANGELOG for detail.

| Release | Theme |
|---------|-------|
| **2.0.0** | Cyrius port — gateway control plane (routing, budgets, cache, auth, audit, retries, rate-limit, metrics, CLI) |
| **2.1.0 / 2.1.x** | Cyrius 6.0 toolchain modernization; hardware planning endpoints, patra persistence, sakshi logging |
| **2.2.0** | Remote provider transport — sandhi HTTPS, auth headers, Anthropic/Gemini shaping, incremental remote streaming |
| **2.2.1–2.2.4** | Provider correctness (token ratios, lifecycle); DLP; cost optimizer + semantic cache + compression; tool calling (3 families, streaming) |
| **2.3.0** | MCP tool server (`/v1/tools/list` + `/v1/tools/call`, bote) |
| **2.3.1 / 2.3.2 / 2.3.3** | Batch inference — concurrent (`/v1/batch`), then async (job-id/progress/cancel), then concurrent multi-batch + registry eviction ([ADR 009](../decisions/009-concurrent-batch-inference.md)) |
| **2.3.4 / 2.3.5** | Observability — latency histograms, majra event bus + `/v1/events/recent`, traceparent propagation, then OTLP/JSON span export ([ADR 010](../decisions/010-observability.md)) |
| **2.4.0** | Multi-threaded accept loop — unified 7-worker pool serves all traffic concurrently; batch unified onto it ([ADR 011](../decisions/011-multithreaded-accept-loop.md)) |
| **2.4.1** | Hardware planning endpoints — `/v1/hardware/model-format` + `/v1/hardware/requirement-match` |
| **2.4.2** | Threaded hardware detection — parallel CLI probes (`registry_detect_threaded`), unblocked by the 2.4.0 thread-safe foundation |
| **2.4.3** | OTLP remote/`https` export (worker-routed TLS) + scaffolding (state.md, fuzz harnesses, CI security scan) |
| **2.4.4** | New backends — vLLM, TensorRT-LLM, ONNX Runtime (OpenAI-compatible local provider types) |
| **2.4.5** | Hardening review — concurrency sync-pass fixes (cache_stats/tokens_pools), configurable routing strategy + working lowest-latency, dead-code removal ([ADR 011](../decisions/011-multithreaded-accept-loop.md) §2.4.5) |
| **2.4.6** | Toolchain + dependency refresh — Cyrius 6.2.11, ai-hwaccel 2.3.12, bote 2.7.6 (`registry_new` → `tool_registry_new`), majra 2.4.7 |

**Toolchain**: Cyrius pin currently **6.2.11** (bumped per release; clean `lib/`
re-sync each time — see [the bump note](#toolchain)).

---

## v2.4.x — Concurrency & completeness arc

Mirroring the 2.2.x parity arc: **v2.4.0 landed the foundation** (the
multi-threaded accept loop, ✅ shipped); the point releases below work the
remaining completeness + hardening items. Order within the arc is a guide, not a
commitment; items tagged *upstream-gated* wait on a sibling repo.

### v2.4.0 — Multi-threaded accept loop  *(arc foundation)* — ✅ SHIPPED 2026-06-10
A unified pool of 7 banked worker threads serves all traffic concurrently; the
accept loop only accepts + enqueues. Batch items share the same pool (the separate
crypto-lane pool is gone); sync batch work-steals the queue. Synchronization pass
landed (batch-registry lock, `_chat_lock` extended to token/health handlers,
lock-free atomic `_router` swap on reload). Live-verified: 14 concurrent chats in
115 ms (≈6× the serialized path), 200 mixed + reload-under-load survived.
See [ADR 011](../decisions/011-multithreaded-accept-loop.md).
**Follow-on:** the concurrency ceiling is 7 (the sigil bank budget) — a
per-thread-arena allocator + more banks would lift it (deferred). Also now
unblocks threaded hardware detection (below).

### v2.4.x candidates (point releases)

**Hardware planning** — remaining ai-hwaccel surface:
- [x] `POST /v1/hardware/model-format` — detect SafeTensors/GGUF/ONNX/PyTorch
      from raw model bytes (**2.4.1**).
- [x] `POST /v1/hardware/requirement-match` — scheduler requirement matching
      against detected hardware (**2.4.1**).
- [x] Threaded detection at startup (`registry_detect_threaded`) — parallel CLI
      probes; was blocked on the pre-thread-safe allocator, wired in **2.4.2**
      (verified byte-identical to serial).

**New backends** — [x] vLLM (PagedAttention), TensorRT-LLM (NVIDIA), ONNX Runtime
(**2.4.4**) — added as OpenAI-compatible local provider types.

**OTLP follow-ups** (extends 2.3.5):
- [x] **Remote / `https://` collector** — DNS + TLS via sandhi; https POSTs
      worker-routed so a banked worker does the TLS (**2.4.3**).
- [ ] **Nested spans** — provider-forward / cache / retry child spans under the
      inference span.
- [ ] **OTLP/protobuf** — the standard wire format; *upstream-gated* on a cyrius
      protobuf lib (proposed: `cyrius/docs/development/proposals/2026-06-10-protobuf-lib.md`).

**Scaffolding modernization** (sibling-repo conventions):
- [x] `docs/doc-health.md` — doc currency tracker (2026-06-10 doc sweep).
- [x] `docs/development/state.md` — volatile state snapshot, per release (**2.4.3**).
- [x] Fuzz harnesses (`fuzz/*.fcyr`) + a CI fuzz step (**2.4.3**).
- [x] Security-pattern scan in CI (`scripts/security-scan.sh`) (**2.4.3**).
- [ ] Split `tests/hoosh.tcyr`/`hoosh.bcyr` into per-topic units — only if the
      suite keeps growing (currently fine as single files).

**MCP tools (szál)** — *upstream-gated*. `/v1/tools/list` + `/v1/tools/call` are
live, but the registry holds only a `bote_echo` smoke tool until **szál** (58
built-in MCP tools) ships as a Cyrius distlib. Register them in `mcp_init`
alongside `bote_echo` — no transport changes. ([ADR 005](../decisions/005-mcp-via-bote.md).)

### Upstream-gated (sandhi)
- **Connection pooling** — high-value case is remote TLS-handshake reuse; gated on
  sandhi keep-alive/pooling. (Local loopback connect ≪ inference latency, so the
  local path has low ROI.)
- **Certificate pinning + mTLS** — pinning/mTLS exist
  (`sandhi_tls_policy_new_pinned`) but the high-level `sandhi_http_post`/`_stream`
  client doesn't thread a TLS policy. Filed upstream
  (`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`).

<a id="toolchain"></a>
> **Toolchain bumps** (process, not arc work): on each pin bump, wipe `lib/` and
> run a clean `cyrius lib sync` + `cyrius deps`, then the full CI step order,
> before trusting a local build — stale `lib/` masks stdlib module renames (e.g.
> 6.1.27 merged `bigint`/`toml`/`json` → `bayan`). Compiler strictness can also
> tighten: 6.2.11 turned duplicate same-scope `var` declarations into a hard
> error (the 2.4.6 bump renamed three such test vars; block-scoped shadowing
> still compiles).

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
