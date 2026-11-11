# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work only. Shipped releases
live in [CHANGELOG.md](../../CHANGELOG.md), one entry each; design decisions live
in [ADRs](../decisions/). Nothing here is a record of what was done — if an item
ships, it moves to the CHANGELOG and leaves this file.

**Current**: v2.12.2 (cyrius 6.7.18). `POST /v1/audio/speech` is rendered in hoosh's own process through shabda 3.1.1
and svara 3.6.3 ([ADR 016](../decisions/016-in-process-speech-through-svara.md)); szál's MCP tools are served behind
an allow-list ([ADR 005](../decisions/005-mcp-via-bote.md)'s 2.12.1 update); the command line goes through cmdit 1.5.1
([ADR 015](../decisions/015-command-line-through-cmdit.md)); a packed ternary model is served in process through
tentib 1.5.1 ([ADR 014](../decisions/014-in-process-ternary-inference.md)); and a request's own data is released
when its thread's next request starts ([ADR 013](../decisions/013-request-memory.md)). See the CHANGELOG.
**Next**: 2.13.x, accounting and capacity.

The bands are a plan, not a promise. Every item was checked against the code and its upstream on 2026-11-11.
- **2.13.x – 2.15.x minors** each take one theme: they add features or change answers, compatibly. They go in order,
  and nothing below them holds them up.
- **Patches** fix things without changing what a client sees, and ship on whichever minor is current when ready.
- **2.x.x** holds what waits on an upstream release or on demand, to be banded when it is ready.
- **3.0.0** collects what would break a client or an operator's config. Nothing there ships in a 2.x.

---

## 2.13.x — accounting and capacity

- **Cache tokens.** Anthropic reports `cache_creation_input_tokens` / `cache_read_input_tokens`, OpenAI reports
  `prompt_tokens_details.cached_tokens`, and every provider prices cached input lower. hoosh counts none of them, so
  a cached prompt is costed at the full input rate.
- **`POST /v1/cost/recommend` over the live catalog.**
  - **Today:** `handle_cost_recommend` ranks only exact-model route patterns and skips every pattern that ends in
    `*`. So a gateway whose routes are all `claude-*`-style answers 404 ("no capable exact-model route"), and a
    model a provider lists but no pattern names is never recommended.
  - **Change:** rank each route's catalog entries instead. `metadata_lookup` and `estimate_cost_micro` already read
    their live context window, vision bit and price.
  - agnosai's B33 (choosing a model per tier from the catalog) can use it. It reads `tier`, `kind` and `tools` from
    the catalog since 2.10.2.
- **Configurable worker pool** (`[server] workers`).
  - `WORKER_COUNT = 7` came from sigil's 8 crypto banks. Since sigil 3.13.14 (2.9.1) lanes come from a pool of 64
    per operation ([ADR 011](../decisions/011-multithreaded-accept-loop.md#update-291-the-crypto-bank-constraint-is-gone)).
  - **Change:** make the pool size a config key, bounded well under 64. Benchmark contention on `_chat_lock` before
    raising the default.

---

## 2.14.x — requests for current models

- **Thinking on Gemini and Ollama.** `reasoning_effort` maps to Anthropic's thinking and is forwarded to
  OpenAI-compatible reasoning models. Gemini's `generationConfig.thinkingConfig` and Ollama's `think` are not
  mapped. This is the smallest of the three; start here.
- **Multimodal content parts.** `content` is read as a string, and a plain message passes through verbatim
  (`provider.cyr`). So an OpenAI `image_url` part is never translated to Anthropic `image` or Gemini `inline_data`.
  The catalog already knows each model's vision bit. This is the gate for thoth's F10 (image input).
- **An OpenAI Responses API path.** gpt-6-astra and gpt-6.1-sol accept no function calling on Chat Completions;
  OpenAI serves their tools only through `/v1/responses`. So a tool-using client cannot drive them through hoosh, and
  the catalog says `tools: false` for them. Translate a tool-carrying request for those models into the Responses
  shape and back. This is the largest of the three.
- **Sampling on a tentib route.** `temperature` and `top_p` are ignored there (greedy only). tentib leaves the logits
  in `ki_logits`, so hoosh can sample from them itself.

---

## 2.15.x — observability

- **Nested OTLP spans.** Provider-forward, cache and retry child spans under the inference span, extending 2.3.5.
  Since 2.10.3 the inference span has an id of its own and names the caller's as its parent, and `trace_begin_from`
  starts a context under any given one, so a child span is `trace_begin_from(trace_current_or_new())` around the
  forward, with the old context restored after. A stream exports no span yet, and an embeddings request neither.
- **OTLP/protobuf**, the standard wire format. It is no longer upstream-gated: cyrius ships `lib/protobuf.cyr`
  (`pb_tag`, `pb_write_varint`, …). Encode the existing OTLP/JSON document's fields as
  `ExportTraceServiceRequest`, into the exporter's reused batch buffer (2.11.2).
- **AGNOS in CI.** hoosh serves on AGNOS since 2.10.1, and `agnos/scripts/smoke/spine-turn-smoke.sh` runs a real
  turn through it, but nothing here builds `--agnos`, which is how 2.10.0 and earlier could say they were serving and
  read no request. A build lane is cheap. A boot needs the three sibling binaries the smoke stages.

---

## Patches *(any release; none gates a minor)*

Fixes that change nothing a client sees. Each ships as a patch on whatever minor is current when it is ready, and no
band waits for one. Two of them wait on an upstream fix; the others are hoosh's own and can go at any time.

- **szál 2.3.3 on AGNOS.** szál 2.3.3 builds `dist/szal-mcp.cyr` for AGNOS: `szal_exec` has an AGNOS path, and all 54
  tools stay on both targets (szál's issue was archived 2026-11-11). Vendor it (`scripts/sync-szal.sh 2.3.3`, after
  szál's `consumer-check.sh`), drop the `CYRIUS_TARGET_AGNOS` guards in `src/main.cyr` and `src/lib/mcp.cyr`
  (`mcp_szal_built`), and build `--agnos`.

- **ADR 013 — what is left.**
  - **The semantic index.** An insert keeps a 32-byte entry, the embedding vector and a scope string, and a removal
    frees none of them; a lookup that inserts nothing leaves its query vector and scope string on the heap. It is
    opt-in (`[semantic]`, an embedding model). Recycle the entries and vectors as the cache does (`keep.cyr`), and
    build the lookup's in the request arena.
  - **Paths not measured yet:** the hardware endpoints (ai-hwaccel allocates for each answer), the token-pool
    endpoints and the batch handler's own bodies. Measure each with `scripts/memory-growth.py` (`--get`, or a POST
    mode to add) before converting it.
  - **An MCP tool's own allocations.** Since 2.12.1 bote's bytes for a message are on the request arena, so
    `GET /v1/tools/list` keeps nothing. A szál tool allocates on the heap: `szal_math_eval` keeps about 1.6 KB a call,
    and bote keeps its copy of a call's arguments, since a handler may keep it. Both are upstream: szál would need
    allocator-taking handlers, which bote's handler ABI (`fn(args, claims)`) does not pass.
  - **A remote request's TLS state** (~179 KB each) is sandhi's; see *Upstream-gated*.
- **Layout-dependent string costs: waiting on upstream.** 2.9.11 found why 2.9.10's version string moved three
  bench rows, and made the bench time aligned heap copies (CHANGELOG 2.9.11). The causes are filed upstream
  (2026-11-04):
  - **cyrius,** `docs/development/issues/2026-11-04-pkgver-declared-by-a-comment-mention.md`: the check that keeps
    `CYRIUS_PKG_VERSION` matches the name in comments, so ai-hwaccel's comment declares it in every consumer, as
    the first literal in `.rodata`.
  - **cyrius,** `docs/development/issues/2026-11-04-unaligned-literals-make-strlen-cost-layout-dependent.md`, with
    a repro: literals are packed unaligned, and `strlen` walks byte by byte to an 8-byte boundary (17 against 30 ns
    for one 13-byte string).
  - **ai-hwaccel,** `docs/development/issues/2026-11-04-comment-naming-cyrius-pkg-version-declares-it-in-consumers.md`:
    reword the comment at `src/detect/command.cyr:222`.

  When ai-hwaccel's fix ships, bump its tag here and check the version literal is gone from hoosh's binaries.
- **Connection pooling for remote TLS.**
  - **Measured in 2.9.3:** a fresh HTTPS request takes a median 40 ms, a reused connection 6.7 ms, so −33 ms per
    request at a ~6 ms RTT, and more for distant providers.
  - **Two preconditions before adopting it:**
    1. Pooled connections must outlive the request, so they cannot live in a per-thread arena. They need sandhi to
       allocate them from the pool's own allocator.
    2. sandhi returns a failed send or read on a pooled connection to the caller rather than retrying it on a fresh
       one (`_sandhi_http_exchange_keepalive_a`), so a stale socket would become a retried POST. Keep the pool's
       idle timeout well under the providers' keep-alive, or get sandhi to retry stale connections internally.
  - A policy-bound route (`tls_pinned_certs`, mTLS) bypasses the pool by design.

---

## 2.x.x — waiting on upstream or on demand

Each gets a band when its gate opens.

- **tentib's next releases.** Each waits on tentib (`docs/development/roadmap.md` there), then is a pin bump and a
  change here.
  - **A model's own vocabulary.** A file can declare the byte vocabulary (tentib 1.4.0), but not another one, so
    hoosh still requires V = 256 and reads bytes as tokens. When the artifact can carry a tokenizer (tentib's 1.4.x
    token table), read it and lift the byte rule.
  - **The 2-bit store.** A load of a model of another shape keeps the one it replaces, `tx_serve_alloc_bytes` since
    2.11.2: 190,168 bytes at T = 30, C = 16. tentib 1.5.1's 2-bit store, internal there, would cut the weights to a
    quarter, at 2–3× the decode cost until cyrius has a vector byte shuffle. Adopt it when tentib makes it public, as
    an opt-in for large models.
  - **Model handles** (tentib 2.0.0) end the kept memory, and allow more than one tentib route per process and a
    decode cache per reply.
- **The model-load seam stays in hoosh.** There is no murti port to adopt: the Rust repo was deleted 2026-11-10.
  Per agnosticos `docs/development/planning/murti.md`, the load → place → dispatch → pool seam grows inside hoosh's
  local providers, and the tentib provider is its first piece, the svara provider (2.12.2) its second. It moves to its own repo
  only if a second consumer needs it outside hoosh. Whether ternary models would then move behind that repo is a new
  decision, not ADR 014's.
- **Speech, beyond 2.12.2** (ADR 016).
  - **Recognition** (`POST /v1/audio/transcriptions`). AGNOS has no speech-recognition engine in Cyrius. The
    `whisper` provider type (`PROV_WHISPER`) names a server hoosh could forward to, and no endpoint uses it. It gets
    an item when there is an engine, or when someone asks for the forward.
  - **Other languages.** shabda converts Spanish, German, Hindi, Arabic and Sanskrit by rules alone (their
    dictionaries are empty), and OpenAI's request names no language. A route key (`language`) when shabda has them.
  - **Forwarding speech to a remote provider** (OpenAI's own `tts-1`) is not done: only svara routes serve it. On
    demand.
  - Mixing, playback and capture stay dhvani's.
- **Split `tests/hoosh.tcyr` / `hoosh.bcyr` into per-topic units**, only if the suite keeps growing (2263 assertions
  in one file at 2.11.2).
- **Threaded hardware detector** *(parked)*.
  - **Why hoosh stays serial:** the threaded path lists sysfs backends (ROCm, Intel NPU, AMD XDNA, TPU, …) before
    the CLI ones (CUDA, Gaudi, Neuron, Vulkan, oneAPI, Apple). `requirement-match` reports the *first* matching
    profile, and `simulate`'s `remove_count` drops the *first* N accelerators, so their answers would change on
    mixed hosts.
  - **Little to gain:** serial takes 22.5 ms and threaded 21.2 ms on the dev host, and since ai-hwaccel 2.5.x the
    threaded path runs the vendor tools one at a time.
  - **When to revisit:** only if startup time matters. Make those two consumers order-independent first.
    `_hw_detect` in `src/lib/hardware.cyr` has the details.

---

## 3.0.0 — backlog *(breaking changes, collected)*

What a client or an operator's config relies on and should go, gathered here so one major takes them together. Each
keeps working through 2.x, and the release before 3.0.0 logs a warning where an operator still uses one.

- **`GET /v1/queue/status` answers one count twice:** `queued` (rust-old's key) and `pending` (the port's 2.x key)
  (`wq_status_json`). Keep one.
- **`rate_limit_rpm`**, rust-old's provider key, is accepted beside `rate_limit` (`_config_rate_limit_str`). Drop the
  alias.
- **`hoosh serve <port>`**, the bare positional port of before 2.5.8, beside `--port` (cli.cyr's `_cli_serve`). 2.12.0
  checks it as cmdit checks `-p`; drop it once the flag form has had a release of its own.

Add to this list only what removes or changes something a client or a config can see. Internal renames ship in any
release.

---

## Upstream-gated

- **sandhi — TLS connection state ignores the request's allocator.** `_sandhi_conn_finalize_with_early_data_a` is
  handed `a` but opens TLS with `tls_connect_alloc(fd, …)`, the global-heap wrapper, instead of
  `tls_connect_alloc_in(a, fd, …)`. So each new TLS connection's state, ~179 KB, stays on hoosh's never-freeing
  heap, even under 2.9.3's per-thread arena: ~170–190 KB per remote request.
  - **Measured** with a standalone probe and an arena: plain HTTP 856 B per request, HTTPS 178,768 B.
  - **The fix has to cover the session cache too:** passing `a` must not leave TLS session-cache entries pointing
    into an arena that gets reset.
  - **Filed:** `sandhi/docs/development/issues/2026-11-02-bote-tls-connect-keeps-each-connection-on-the-heap.md`.
    bote 3.6.5 reported it first; hoosh 2.9.3 was added on 2026-11-04 as a second reporter, with the measurements
    above and three constraints on the fix: a per-thread arena, the session cache kept out of it, and pooled
    connections owning their own allocations. hoosh meets the issue's own Critical bar (a server that fetches on
    every request) and has no workaround.
- **shabda and svara — what a speech request keeps.** About 1 KB per input byte (9.8 KB a request at 6 bytes of
  input, 62.6 KB at 60, 408 KB at 400; `scripts/memory-growth.py --speech`). hoosh's plan keeps nothing; the rest is
  inside the libraries. For a 378-byte input:
  - **shabda's conversion, 250,760 bytes.** It allocates on the default heap and has no allocator-taking forms.
    Filed: `shabda/docs/development/issues/2026-11-11-conversion-allocates-on-the-default-heap.md`, asking for
    `shabda_convert_a` / `shabda_convert_with_a`. When it ships, convert into the request arena.
  - **svara's renders, 133,736 bytes:** a 1.9 KB context per request, and a formant filter or target for each phoneme
    (up to about 1 KB for a consonant). Filed:
    `svara/docs/development/issues/2026-11-11-a-server-render-keeps-what-each-phoneme-allocates.md`. It also asks
    for a streaming sequence render, which would let hoosh drop its copy of svara's crossfade and plan
    (`svara_local.cyr`'s `spk_xf_*` and `spk_plan_events_a`).

  Until both ship, a gateway serving speech should rate-limit its svara routes.
- **svara — module scratch makes concurrent renders unsafe.** `_sv_tract_target_scratch` and `_sv_glottal_rd_out` are
  written and read back within one render, so two threads rendering at once sometimes render the wrong samples (2
  to 4 of 4,000 in the repro). hoosh holds `_spk_lock` around every call into svara, one phoneme at a time. Filed:
  `svara/docs/development/issues/2026-11-11-module-scratch-makes-concurrent-renders-unsafe.md`. When svara keeps its
  scratch per object, the lock can narrow to shabda's engine, and renders can run in parallel.
- **cyrius — `cyrius coverage` reports on the vendored stdlib, not the local repo.** Still open:
  `cyrius/docs/development/issues/2026-07-23-hoosh-coverage-reports-stdlib-not-local-repo.md`. hoosh gates on
  `scripts/coverage.sh` meanwhile.
- **cyrius — `clock_now_ms()` has no vDSO path.** It is a raw clock syscall, measured at 1.351 µs. That dominated
  every hot path until 2.5.11 worked around it with a coarse ticker. A vDSO route would let the ticker go.
  `lib/chrono.cyr` at 6.7.18 still has none.

---

## Non-goals

- **Model training** — hoosh is for inference.
- **Model storage** — hoosh doesn't manage model files.
- **Direct GPU compute** — delegated to backends; ai-hwaccel handles detection.
- **Web UI** — hoosh is an API gateway; a dashboard is separate.
- **Audio DSP** — hoosh serves the speech API over the AGNOS libraries (2.12.2); synthesis is svara's, text to
  phonemes shabda's, and mixing, playback and capture dhvani's.
- **WASM target** — Cyrius doesn't target WASM.

---

<a id="toolchain"></a>

## Process notes

**Toolchain bumps.**
- Remove `lib/*.cyr`, but not `lib/unicode/`, which is tracked. Then run a clean `cyrius lib sync` + `cyrius deps`
  and CI's full step order before trusting a local build. A stale `lib/` hides stdlib renames: 6.1.27 merged
  `bigint`/`toml`/`json` into `bayan`.
- When a git dependency's tag changes, regenerate `cyrius.lock` from empty. 6.6.20+ keeps the old tag's `commit`
  line beside the new one.
- Grep the build for `undefined` and `deprecated`. CI fails on neither: an undefined function compiles to a trap.
- New compiler and linter strictness lands with pins: 6.2.11 duplicate `var`s, 6.7.3's lone `!`, 6.7.18's
  write-length lint.

**Benchmarks are a release gate.**
- CI runs `./scripts/bench-history.sh` and fails the build if the suite does not run. `bench-history.csv` is the
  record.
- Two benchmarks (`estimate_tokens_per_provider`, `pool_available`) sit at single-digit nanoseconds, where 1 ns of
  timer quantization reads as a >10% swing.
- For a toolchain A/B, build both bench binaries and alternate them for at least 7 rounds. To attribute a moved row,
  compare symbol maps (`CYRIUS_SYMS=<path>`): unchanged function bytes with a shifted 64-byte offset is placement.

**Memory is measured, not guessed.** `scripts/memory-growth.py` against a `-D HOOSH_ALLOC_PROBE` build reports
`alloc_used()` per request. `--chunks N` makes a stream N tokens long, `--get` measures a listing, `--cache-max`,
`--hits`, `--audit` and `--otlp` exercise the keepers, and `--provider` each provider's shape. Run any new path or
keeper through it before and after (ADR 013).

**Live verification.** `scripts/bench-live.sh` (opt-in, needs a running gateway and
backend) measures end-to-end; its ~5 ms floor is `curl` process startup, not the
gateway, so use it for relative comparisons only.
