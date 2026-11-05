# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic
> API — backends are swappable without consumer changes.

This roadmap is **forward-looking**: open and planned work only. Shipped releases
live in [CHANGELOG.md](../../CHANGELOG.md), one entry each; design decisions live
in [ADRs](../decisions/). Nothing here is a record of what was done — if an item
ships, it moves to the CHANGELOG and leaves this file.

**Current**: v2.9.10 (cyrius 6.7.18). 2.9.3 stopped remote requests leaking megabytes each; 2.9.3–2.9.10 put
sixteen modules under test for real (fifteen replaced mirrors; `events.cyr` had no tests). See the CHANGELOG.
**Next**: the 2.9.x patches and 2.10.0 below.

The bands are a plan, not a promise. Every item was checked against the code and its upstream on 2026-11-04.
- **2.9.x patches** fix things without changing what a client sees.
- **2.10.x and later minors** add features or change answers.

---

## 2.9.x — later patches *(fixes, no API change)*

- **Test-suite mirror drift: done in 2.9.22.** `tests/hoosh.tcyr` used to re-implement much of what it tested,
  and copies drifted from `src/` while both stayed green: v2.5.6 pricing order, v2.5.7 audit chain-link
  verification, then 2.9.3–2.9.22 found more each time a copy was replaced (each is in its CHANGELOG entry, the
  last a `_requires_post` list of 4 paths out of 19 and a `_models_emit` that did not escape ids).
  - **Now:** the test file copies no current `src/` code. Its helpers are fixtures, readers, loopback servers and
    drivers over the real functions. 37 of 43 `src/lib` modules are included for real (`scripts/coverage.sh`
    lists them). The other six, `handlers`, `pool`, `batch`, `catalog_fetch`, `otlp_export` and `config`, hold the
    sockets and main.cyr's globals. Their testable parts were split out instead: `reqparse.cyr`, `cfgparse.cyr`,
    `cli.cyr`, `wq.cyr`, `models.cyr`, `otlp_export.cyr`'s pure half in `otlp.cyr`, and single functions moved into
    `jsonlite`, `provider`, `provider_err`, `router`, `hardware`, `health`, `metrics` and `http_client`.
  - **How it was done** (the bench file followed in 2.9.24): include each module for real; where a helper
    lives in a module that needs the whole server, move it out first. Mutation-check each move: change the real
    function and confirm a test fails (a hang counts; the CI test step has a 5-minute timeout since 2.9.4).
  - `scripts/coverage.sh` is a floor against *unwatched* code, not against drift. It counts a name appearing in
    `tests/`, so deleting a copy can lower it even when the real function now runs. Its header says so since 2.9.5.
- **The bench file times the real code: done in 2.9.24.** Through 2.9.23 `tests/hoosh.bcyr` included no `src/`
  module and 62 of its 65 functions were copies, so no change to `src/` could move a row. It now includes `src/lib`
  as the test file does; its only functions are `main` and `_b_s`. Two rows that timed a priority queue the
  gateway does not have were dropped (23 rows). The A/B is in CHANGELOG 2.9.24: five rows had been timing far less
  work than the gateway does, the largest `cost_record_known_model` (155 ns copy, 3.28 µs real; the next item).
- **Lookups on the request path no longer scan: done in 2.9.25.** 2.9.24's real bench showed `cost_record` scanning
  the 81-row pricing table twice per completion (3.28 µs). Measuring with a live catalog showed the larger cost:
  every lookup scanned every route's model list, 22 µs per `cost_record` (under `_chat_lock`) and 20 µs per
  `metadata_lookup` with three routes of 100 models. Both tables and every published list are now indexed when
  built, and `estimate_cost_micro` looks each source up once (CHANGELOG 2.9.25: 14×, 39× and 44× faster).
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
    1. Pooled connections must outlive the request, so they cannot live in the per-thread request arena 2.9.3 added.
       They need sandhi to allocate them from the pool's own allocator.
    2. sandhi returns a failed send or read on a pooled connection to the caller rather than retrying it on a fresh
       one (`_sandhi_http_exchange_keepalive_a`), so a stale socket would become a retried POST. Keep the pool's
       idle timeout well under the providers' keep-alive, or get sandhi to retry stale connections internally.
  - A policy-bound route (`tls_pinned_certs`, mTLS) bypasses the pool by design.

---

## 2.10.x — memory and accounting

### 2.10.0 — per-request arena *(highest priority)*

hoosh's allocator **never frees**.
- v2.5.11 removed the dominant per-connection 64 KiB allocation: over 2000 requests, growth fell from ~128 MB to
  2.0 MB.
- **~1 KiB per request still accumulates**, from response building (`str_builder`, `to_cstr`, path parsing). At
  100 req/s that is ~360 MB/hour, still an eventual OOM on a long-lived gateway, only a slower one.

Point fixes will not close this. It needs a **per-request arena with mark/release**: take a mark when a request
starts, and release it once the response is written.
- `alloc_reset` cannot run per request as it stands. The response cache, audit chain, cost records, routes and
  health records live in the same arena, and they must outlive the request that created them.
- The design work is deciding what is request-scoped and what is process-scoped, then enforcing that split. Write
  an ADR first.
- Two working precedents: the 2.8.0 catalog fetch's reusable arena, and 2.9.3's per-thread `_remote_arena()`
  (`provider.cyr`). The latter took every remote request's sandhi allocations, megabytes each, off the heap.
- Measure with unique prompts. A repeated prompt is answered from the response cache and never exercises the
  request path.

Until then, a hoosh instance needs a periodic restart under sustained load. Measure with `/proc/<pid>/status` rather
than assuming.

### Then, in any order

- **Cache tokens.** Anthropic reports `cache_creation_input_tokens` / `cache_read_input_tokens`, OpenAI reports
  `prompt_tokens_details.cached_tokens`, and every provider prices cached input lower. hoosh counts none of them, so
  a cached prompt is costed at the full input rate.
- **`POST /v1/cost/recommend` over the live catalog.**
  - **Today:** `handle_cost_recommend` ranks only exact-model route patterns and skips every pattern that ends in
    `*`. So a gateway whose routes are all `claude-*`-style answers 404 ("no capable exact-model route"), and a
    model a provider lists but no pattern names is never recommended.
  - **Change:** rank each route's catalog entries instead. `metadata_lookup` and `estimate_cost_micro` already read
    their live context window, vision bit and price.
  - agnosai's B33 (choosing a model per tier from the catalog) can use it.
- **Configurable worker pool** (`[server] workers`).
  - `WORKER_COUNT = 7` came from sigil's 8 crypto banks. Since sigil 3.13.14 (2.9.1) lanes come from a pool of 64
    per operation ([ADR 011](../decisions/011-multithreaded-accept-loop.md#update-291-the-crypto-bank-constraint-is-gone)).
  - **Change:** make the pool size a config key, bounded well under 64. Benchmark contention on `_chat_lock` before
    raising the default.

---

## 2.11.x — requests for current models

- **Thinking on Gemini and Ollama.** `reasoning_effort` maps to Anthropic's thinking and is forwarded to
  OpenAI-compatible reasoning models. Gemini's `generationConfig.thinkingConfig` and Ollama's `think` are not
  mapped. This is the smallest of the three; start here.
- **Multimodal content parts.** `content` is read as a string, and a plain message passes through verbatim
  (`provider.cyr`). So an OpenAI `image_url` part is never translated to Anthropic `image` or Gemini `inline_data`.
  The catalog already knows each model's vision bit. This is the gate for thoth's F10 (image input).
- **An OpenAI Responses API path.** gpt-6-astra and gpt-6.1-sol accept no function calling on Chat Completions;
  OpenAI serves their tools only through `/v1/responses`. So a tool-using client cannot drive them through hoosh.
  Translate a tool-carrying request for those models into the Responses shape and back. This is the largest of the
  three.

---

## 2.12.x — tools and observability

- **szál's MCP tools.** No longer upstream-gated.
  - **What szál delivered:** szál 2.2.0 (2026-09-25) shipped everything hoosh filed for: `szal_` renames, a consumer
    bundle (`dist/szal-mcp.cyr`) with no vendored libraries inside, and `szal_register_into(dispatcher)`.
  - **What is left:**
    1. **The toolchain.** szál 2.2.0 pins cyrius 6.6.6 and predates bote 3.4.8's registration checks, which bote's
       own audit says szál's schemas pass. Wait for a szál release on 6.7.x, or prove that 2.2.0 compiles clean on
       hoosh's pin.
    2. **An allow-list.** The tool groups include `file`, `git`, `net`, `process` and `system`, so a gateway must not
       expose them by default. Add a per-group allow-list (`[mcp] tools = [...]`, empty by default), and require
       `[[auth]]` for anything beyond pure groups such as `json`, `math` and `hash`.
    3. **The wiring.** Vendor the bundle at `src/vendor/szal-mcp.cyr` and call `szal_register_into` in `mcp_init`.
       The transport does not change ([ADR 005](../decisions/005-mcp-via-bote.md)).
- **Nested OTLP spans.** Provider-forward, cache and retry child spans under the inference span, extending 2.3.5.
- **OTLP/protobuf**, the standard wire format. It is no longer upstream-gated: cyrius ships `lib/protobuf.cyr`
  (`pb_tag`, `pb_write_varint`, …). Encode the existing OTLP/JSON document's fields as
  `ExportTraceServiceRequest`.

---

## Unscheduled *(maintainer's call or demand-gated)*

- **cmdit adoption.** See [Adopting cmdit](#adopting-cmdit-cli-parsing) below. It changes CLI behavior on purpose
  (exit codes, last-wins flags), so it ships in a minor.
- **AGNOS build.** There is no `hoosh_agnos` build in CI or `scripts/`; the one on disk downstream is 2.4.11. thoth's
  v1.0 gate 1, rung 2 (a real turn against the spine on AGNOS) waits on it.
- **Split `tests/hoosh.tcyr` / `hoosh.bcyr` into per-topic units**, only if the suite keeps growing.
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

- **cyrius — `cyrius coverage` reports on the vendored stdlib, not the local repo.** Still open:
  `cyrius/docs/development/issues/2026-07-23-hoosh-coverage-reports-stdlib-not-local-repo.md`. hoosh gates on
  `scripts/coverage.sh` meanwhile.
- **cyrius — `clock_now_ms()` has no vDSO path.** It is a raw clock syscall, measured at 1.351 µs. That dominated
  every hot path until 2.5.11 worked around it with a coarse ticker. A vDSO route would let the ticker go.
  `lib/chrono.cyr` at 6.7.18 still has none.

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

**Live verification.** `scripts/bench-live.sh` (opt-in, needs a running gateway and
backend) measures end-to-end; its ~5 ms floor is `curl` process startup, not the
gateway, so use it for relative comparisons only.

---

## Adopting cmdit (CLI parsing)

A tier-2 ("verb tool") candidate from the 2026-06-25 ecosystem CLI review
([agnosticos `planning/cmdit.md`](https://github.com/MacCracken/agnosticos/blob/main/docs/development/planning/cmdit.md)).
**Demand-gated, not a forced sweep**: the natural moment is the next change to the CLI or
to its test mirror.

**What is hand-rolled** (surveyed 2026-10-14 in `src/main.cyr`, line numbers re-checked 2026-11-04; `lib/` and
`src/vendor/` excluded; no `flags_` / `FLAG_ERR_`; `HOOSH_LOG` (`:1073`) is env, not argv):

- **Three argv re-scanners:** `_flag_val` (`:961`, first-wins), `_flag_set` (`:985`) and
  `_positional` (`:997`). `_positional` assumes every dash token but `--stream` takes a
  value, so an unknown flag is silently ignored and eats the next positional, a dash-led
  prompt is skipped, and `--` counts as a flag.
- **`-c/--config`** is read globally (`:1080`), but the verb is always `argv(1)`
  (`:1141`), so `hoosh -c f serve` fails with `unknown command: -c`.
- **Verbs** (`:1139-1201`):
  - no verb → `serve`; `serve` takes `-p/--port`, `--bind` or a bare positional port, and
    silently ignores a bad port;
  - `models` and `health` take `--server`; `bench <url> [count]` is missing from
    `cmd_help`;
  - `infer` takes `-m`, `--server`, `--stream` and 1–2 positionals;
  - `info`; `help` / `--help` / `-h` work as `argv(1)` only; `version` / `--version`
    print `hoosh v<HOOSH_VERSION>` to stdout.
- **Every verb, help and version included, runs after config parse, `hw_init()` and
  storage open** (`:1100-1135`). So `hoosh --help` exits 1 under a broken config, and
  `hoosh serve --help` starts the gateway.
- **Help and errors:** `cmd_help` (`:824-848`) is a stdout wall with per-verb rows, and
  the header comment (`:4-8`) is stale. An unknown command (`:1203`) prints its text to
  stderr and its newline to stdout, and exits 1.

**Site → cmdit:**

- [ ] **Top level:** `cmdit_str(h, 'c', "config", "hoosh.cyml", …)` + `cmdit_verb` × 8.
  Call `cmdit_dispatch(h)` **before** `load_config`, so `-c` binds on either side of the
  verb and help, version and usage errors exit before config, hardware and storage are
  touched. `cmdit_verb_matched(h) < 0` → `serve`.
- [ ] **Per-verb handles** (`cmdit_parse_verb`), then delete the three scanners:
  - `serve`: `cmdit_int` `-p` + `cmdit_range(…, 1, 65535)`, `--bind`, at most 1
    positional; `cmdit_seen` picks flag, positional or config port;
  - `models` / `health`: `--server`;
  - `infer`: `-m`, `--server`, `cmdit_bool` `--stream`; `_exact` 1 positional if `-m`
    was seen, else 2;
  - `bench`: 1..2 positionals.
- [ ] **Version and help:** `CMDIT_VERSION` keeps the `hoosh v` + `HOOSH_VERSION`
  print, because `cmdit_version` would need a `v`-prefixed string. `CMDIT_HELP` keeps
  `cmd_help`; per-verb `cmdit_help(vh)` is new. Errors → `cmdit_print_error` +
  `CMDIT_EXIT_USAGE`.

**Gaps (none blocking):**

- **Deliberate changes, for the CHANGELOG:**
  - unknown flags, a missing value and a bad port become exit-2 errors
    (`tests/hoosh.tcyr:6653` asserts today's silence on a missing value);
  - repeated flags become last-wins;
  - an unknown command exits 2, not 1;
  - a dash-led prompt needs `--`.
- **Help:** cmdit renders on stderr and has no renderer for `cmd_help`'s per-verb rows,
  so `cmd_help` stays hand-written.
- **`bench`:** cmdit cannot hide a verb, so a registered `bench` shows in the
  unknown-command list and in completions. Accept that, or match it on
  `cmdit_raw_argv` before dispatch.

**cyrius.cyml:** `[deps].stdlib` already lists `args` and `io` (no `flags`). Add the block below. cmdit 1.5.0 pins
cyrius 6.7.18, hoosh's pin; check its 1.3.0 → 1.5.0 changelog against the API names above before adopting.

```
[deps.cmdit]
git = "https://github.com/MacCracken/cmdit.git"
tag = "1.5.0"
modules = ["dist/cmdit.cyr"]
```

**Tests:**

- [ ] **Replace the CLI mirror**, the drift shape *2.9.x — Test-suite mirror drift* names, with
  synthetic-argv cases on the pure `cmdit_dispatch_argv`. The mirror is in
  `tests/hoosh.tcyr`:
  - `:1102-1150` (`_t_flag_val` / `_t_positional`);
  - `:1412-1425` (`_t_infer_args`);
  - `:6633-6676` (`cli_flag_parsing` / `cli_positionals`);
  - `:6950-6966` (`rustold_config_cli` infer).

  Put the handle builder in an includable `src/lib/` module (the `urlparse.cyr` split)
  and name it in the tcyr, so the suite drives the real table and
  `scripts/coverage.sh` still counts it.
- [ ] **Output-identity bar:** the only binary-level CLI check is `release.yml:55-56`.
  Before migrating, capture stdout, stderr and exit code for every verb and error arm
  (at least `README.md:97-113`'s invocations). Diff afterwards: only the deliberate
  changes above may differ.

References: cmdit `README.md` ("Verb dispatch", API reference),
`docs/adr/0002-verb-dispatch.md`; stiva's `src/main.cyr` is the worked verb tool.
