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

**Toolchain**: Cyrius pin currently **6.1.29** (bumped per release; clean `lib/`
re-sync each time — see [the bump note](#toolchain)).

---

## Next

### v2.4.0 — Multi-threaded accept loop  *(the big throughput item)*
Thread the accept loop so **all** traffic runs concurrently, not just batch
workers. **Unblocked** since 2.3.1 — the cyrius allocator is thread-safe (v6.0.64
CAS spinlock, re-verified by a 4-thread × 200k-alloc stress, 0 corruption).
Requires a **shared-state synchronization pass** across every mutating handler
(cache / budget / audit / rate / cost / metrics). 2.3.x already locked the chat
path (`_chat_lock`) and made metrics atomic; this extends that to the rest, plus
per-worker sigil crypto banks (see [ADR 009 §5](../decisions/009-concurrent-batch-inference.md))
for any handler that hashes/HMACs/TLSes on a worker thread. Also enables
loopback-style batching and lifts the per-request alloc-spinlock contention
ceiling (a per-thread-arena allocator is the deeper follow-up).

### OTLP follow-ups (2.3.x)
- **OTLP/protobuf** — the standard wire format; gated on a cyrius protobuf lib
  (proposed upstream: `cyrius/docs/development/proposals/2026-06-10-protobuf-lib.md`).
  Add a `[[telemetry]] encoding = "protobuf"` once it lands.
- **Remote / `https://` collector** — current exporter is localhost-`http` only;
  remote needs DNS + TLS via sandhi.
- **Nested spans** — provider-forward / cache / retry child spans under the
  inference span.

### Connection pooling for backend sockets
Deferred — low near-term ROI: the local path connects to `127.0.0.1` (loopback
connect ≪ inference latency). The high-value case is **remote TLS-handshake
reuse**, gated on sandhi keep-alive/pooling support. Revisit when sandhi exposes
it.

### Certificate pinning + mTLS (remote hardening)
Deferred — blocked on a sandhi wiring gap: pinning/mTLS exist
(`sandhi_tls_policy_new_pinned`) but the high-level `sandhi_http_post`/`_stream`
client doesn't thread a TLS policy. Filed upstream
(`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`); pick up
hoosh-side once it lands.

### Hardware planning (remaining ai-hwaccel surface)
- `POST /v1/hardware/model-format` — detect SafeTensors/GGUF/ONNX/PyTorch
  (ai-hwaccel `model_format.cyr`).
- `POST /v1/hardware/requirement-match` — scheduler requirement matching
  (ai-hwaccel `requirement.cyr`).
- Threaded detection at startup (`registry_detect_threaded`) — was blocked on the
  single-threaded runtime; revisit with the v2.4.0 accept loop.

### New backends
- vLLM (PagedAttention), TensorRT-LLM (NVIDIA), ONNX Runtime (local ONNX).

### MCP tools (szál)
`/v1/tools/list` + `/v1/tools/call` are live, but the registry holds only a
`bote_echo` smoke tool until **szál** (58 built-in MCP tools) ships as a Cyrius
distlib. Register them in `mcp_init` alongside `bote_echo` — no transport changes.
([ADR 005](../decisions/005-mcp-via-bote.md).)

### Scaffolding modernization (sibling-repo conventions)
- [x] `docs/doc-health.md` — doc currency tracker (this sweep).
- [ ] `docs/development/state.md` — volatile state (version, test/bench counts,
      binary size, recent releases), refreshed each release (patra/cyrius pattern).
- [ ] Fuzz harnesses (`fuzz/*.fcyr`) + a CI fuzz step.
- [ ] Security-pattern scan in CI (raw `execve`, hardcoded `/etc` paths).
- [ ] Split `tests/hoosh.tcyr`/`hoosh.bcyr` into per-topic units — only if the
      suite keeps growing (currently fine as single files).

<a id="toolchain"></a>
> **Toolchain bumps**: on each pin bump, wipe `lib/` and run a clean
> `cyrius lib sync` + `cyrius deps`, then the full CI step order, before trusting
> a local build — stale `lib/` masks stdlib module renames (e.g. 6.1.27 merged
> `bigint`/`toml`/`json` → `bayan`).

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
