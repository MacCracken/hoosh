# Changelog

All notable changes to hoosh are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/).

## [2.9.11] — 2026-11-04

**Benchmark rows no longer depend on the version string.** 2.9.10's one byte longer version moved three rows by up
to 25% on unchanged code. 2.9.11 finds the mechanism and makes the bench time what the gateway handles: heap
strings, not packed literals. No gateway code changed. The gates are unchanged and clean (1443 assertions, coverage
373/684, 26 of 37 modules included for real).

### Found — how a version string moved the benchmarks

The chain, each link checked on a built binary:
1. **ai-hwaccel is in every hoosh build.** cbt prepends each `[deps]` module to every compile, the bench and the tests
   included; ai-hwaccel's strings are in the bench binary.
2. **One of its comments names `CYRIUS_PKG_VERSION`** (`lib/ai-hwaccel.cyr:2520`). cycc declares that constant only
   when the expanded source mentions it, but its check matches the text anywhere, comments included. So every hoosh
   build gets `var CYRIUS_PKG_VERSION = "<version>";`. With ai-hwaccel left out (a scratch build), the declaration is
   gone, and the bench binaries built at 2.9.9 and 2.9.10 are byte-identical.
3. **That string is the first literal in `.rodata`,** and cyrius packs string literals back to back, unaligned. When
   the version changes length, every literal after it moves by that many bytes.
4. **The stdlib `strlen` walks byte by byte until it reaches an 8-byte boundary,** then reads 8 bytes at a time.
   `streq` calls it twice. The hot keys sat at `% 8 == 0` in the 2.9.9 binary and at `% 8 == 1` in 2.9.10's:
   `"prompt-hash-1"` for `cache_get_hit` and `"http://localhost:11434"` for `cost_record_local_free`. No literal
   moved across a cache line; the cost was the byte-by-byte prologue.

**In the gateway this mostly averages out.** It reads many literals, about one in eight aligned whatever the shift,
and its hot keys (cache keys, configured URLs, model patterns) are heap strings. A benchmark timing one literal sees
the whole swing.

### Changed — the bench times heap copies of its string inputs

`tests/hoosh.bcyr` gains `_b_s(lit)`, an 8-byte-aligned heap copy made outside the timed loop. `alloc` rounds every
allocation to 8 bytes, as for the gateway's own strings. Every string a timed loop reads now goes through it:
route patterns and model names, cache keys, the MCP messages, the batch body, the auth tokens, the cost-record URLs
and models, and the audit fields. The DLP prompt was already a heap copy.

**Proof:** the fixed bench, built at version `2.9.9` and at `2.9.11`, gives the same rows. Over eight alternated
rounds `cache_get_hit` had a median of 120 ns for both builds; `cost_record_local_free` read 109–110 and
`route_matches_model` 32–33 for both.

### Notes

- **A step in `bench-history.csv` at 2.9.11,** from the measurement change, not from code. Compared with 2.9.10:
  - `cache_get_hit` 146 → 121 ns, `cost_record_local_free` 136 → 110 and `route_matches_model` 36 → 32, back where
    2.9.9's lucky layout had them;
  - `auth_verify_token` / `_wrong_late` / `_wrong_early` 105 / 102 / 92 → 80 / 81 / 81;
  - `cost_record_known_model` 161 → 154. Its URL used to sit across a cache line.
- **The auth rows answer a question they could not before.** Those rows exist to show the constant-time compare
  costs the same whether a wrong token differs early or late. In 2.9.10 they read 92 and 102 ns, a 10 ns gap that
  would look like a timing leak. It came from the literals' alignment: it appeared with 2.9.10's layout shift (2.9.9
  read 103 and 102), and with aligned copies both read 81.
- **Upstream, not changed here,** and filed in each repo's `docs/development/issues/` on 2026-11-04:
  - cyrius's `CYRIUS_PKG_VERSION` check matching comments (`2026-11-04-pkgver-declared-by-a-comment-mention.md`);
  - cyrius's unaligned string literals, together with `strlen`'s byte-by-byte prologue
    (`2026-11-04-unaligned-literals-make-strlen-cost-layout-dependent.md`, with a repro);
  - the ai-hwaccel comment that triggers the declaration
    (`2026-11-04-comment-naming-cyrius-pkg-version-declares-it-in-consumers.md`).

  hoosh cannot edit `lib/ai-hwaccel.cyr`: `cyrius deps` re-vendors it, and `deps --verify` checks its hash.

## [2.9.10] — 2026-11-04

**The tests run the real health prober and circuit breaker.** One function moved to make that possible:
`route_is_remote` went from `provider.cyr` to `route.cyr`, unchanged. It was all `health.cyr` needed from
`provider.cyr`. The test file's copies of the health state machine are gone, along with nine older groups that
tested a map-based health design `src/` does not have. **1443 assertions** in 185 groups. All gates are clean:
fmt, lint, vet, deny, coverage (373/684, 54%; **26 of 37** modules included for real), the security scan and the
four fuzz targets.

### Changed — `route_is_remote` lives in `route.cyr`

- **What it is:** a route predicate (does the base URL start with `https`?), defined after `route_health`. Its body
  is unchanged, and `provider.cyr` leaves a one-line note where it was.
- **Comments:** the `# Requires` headers of `health.cyr` and `catalog_fetch.cyr` now say where it comes from.
- **The binary:** the same size as before. Its bytes differ only because a function moved. The bench binary is
  byte-identical to 2.9.9's tree rebuilt at version 2.9.10.

### Changed — `health.cyr` is included for real

`tests/hoosh.tcyr` includes it after `router.cyr`. `health_sweep` reads `main.cyr`'s `_router` global, so the test
file declares that global in `main.cyr`'s place. Gone:
- **The test file's `route_is_remote` copy;** the real one now comes from `route.cyr`.
- **The `health_record` copy,** which returned whether the circuit flipped. The real function signals a flip by
  publishing a health-changed event, and the tests now count those events.
- **The copies of `health_status_str` and the probe path.**
- **Nine groups** (`health_map_empty` through `health_rapid_cycle`, 35 assertions) that never called `src/`. They ran
  their own state machine on a hashmap, modeled on rust-old's provider map. They recorded each failure's own error
  message, where `src/` records "probe failed", and shadowed `UNHEALTHY_THRESHOLD` with a local `3`, so a change to
  the real threshold could not reach them.

### Added — tests for behavior no test had reached

- **`health_circuit_breaker`,** on the real `health_record`:
  - `health_start` with probing off creates the record lock and starts no prober;
  - the circuit trips at the third consecutive failure, and each transition publishes exactly one event, naming the
    route, with `healthy` false or true;
  - a repeat failure or a repeat success publishes nothing;
  - the error is recorded on failure and cleared on recovery;
  - the last check is stamped with the clock.
- **`health_failures_must_be_consecutive`:**
  - two failures, a success, then two more does not trip it;
  - the boundary sits at the real `UNHEALTHY_THRESHOLD`;
  - trip, recover and trip again is two events, with the second run counted from 0;
  - routes are independent, and a route without a health record is left alone.
- **`health_status_str` and `health_probe_path`,** on real routes.
- **`health_probe` over loopback, local routes:** Ollama is probed with `GET /api/tags`, the others with
  `GET /v1/models`. 2xx and 3xx count as up; 404, 503 and no answer count as down.
- **`health_probe` over loopback, remote routes:** a bare TCP connect, with no request sent. During those tests the
  I/O timeout drops to 2 s, so a probe that sent a request and waited fails in seconds rather than hanging. With
  nothing listening, either kind of probe reports down, and so does a route without a base URL.
- **`health_sweep`:**
  - with no router it does nothing;
  - over three sweeps, a dead local backend is depooled, a reachable remote stays healthy, and a disabled route is
    never probed.
- **`route_is_remote`:** a route without a base URL is not remote.

### Mutation-checked

A copy of the tree was changed 17 ways, and every mutant is caught:

| mutant | result |
|---|---|
| the threshold compared with `>` instead of `>=` | 7 fail |
| a success keeping the failure count, or the error | 5 and 1 fail |
| an event on every probe; never an event; the event's `healthy` inverted | 5, 6 and 2 fail |
| the last check not stamped; a success not recovering | 1 and 4 fail |
| a remote probe sending a GET instead of connecting | 2 fail |
| a 3xx counted as down; Ollama probed on `/v1/models` | 1 and 2 fail |
| `health_sweep` probing disabled routes; `disabled` not outranking `unhealthy` | 1 and 2 fail |
| `health_start` creating no lock when probing is off | fails, then crashes |
| `health_start` starting the prober at interval 0 | fails, then the run hangs |
| `route_is_remote` treating `http://` as remote | 6 fail |
| `route_is_remote` without its no-base-URL guard | crashes |

**Two first-round results led to test changes:**
- **The no-base-URL mutant survived:** no test called `route_is_remote` on a route without a base URL, because
  `health_probe` returns before reaching it. The assertion above now does.
- **The remote-GET mutant hung** for the full timeout. The 2 s I/O timeout now turns that into two failures.

**Why the interval-0 mutant hangs:** its prober thread spins forever and keeps the test process alive after the
failure. CI's 5-minute cap on the test step ends it.

### Notes

- The test file is 7,136 lines (was 7,341). Symbol coverage rose from 369 to 373 of 684.
- **Benchmarks:** 22 of the 25 are within 1 ns or 13% of 2.9.9's. Three moved, all from the version string:
  - **The moves:** `cache_get_hit` 119 → 146 ns, `cost_record_local_free` 109 → 136 ns, `route_matches_model`
    32 → 36 ns. Three repeat runs agreed.
  - **The cause:** "2.9.10" is one byte longer than "2.9.9", so everything stored after it moves by a byte.
  - **The proof:** 2.9.9's tree, rebuilt with only its version set to 2.9.10, produces a bench binary byte-identical
    to this one. Alternated over three rounds, the old binary ran 118–128 / 109–111 ns and both 2.9.10 builds
    143–148 / 136–137 ns.
  - **Not recorded:** two earlier runs went alongside another workload's compiles (load 4–7) and were discarded.
- **What's next for mirror drift:** `retry.cyr` now needs only `provider.cyr`, and `provider.cyr` needs only
  `otlp.cyr`'s `otlp_enqueue_inference_span`. `otlp.cyr` sits in a cycle with `pool.cyr`.
- **More groups that test only their own inline code:** the count-only `rate_limit_*` groups and the `registry_*`
  groups exercise a hashmap, not `src/`. The real rate limiter has its own tests. Both are listed in the roadmap.

## [2.9.9] — 2026-11-04

**The tests run the real router and the real hardware module, on ai-hwaccel itself.** Their copies in the test file
had drifted in six places. One src comment was stale and is corrected; no gateway code changed (the binary differs
from 2.9.8's only in the version string). **1448 assertions** (was 1355) in 193 groups. All gates are clean: fmt,
lint, vet, deny, coverage (369/684, 53%; **25 of 37** modules included for real), the security scan and the four
fuzz targets.

### Changed — `router.cyr` and `hardware.cyr` are included for real

`tests/hoosh.tcyr` now includes three more files:
- the vendored ai-hwaccel bundle (`lib/ai-hwaccel.cyr`), first among the non-stdlib includes, as in `src/main.cyr`;
- `src/lib/hardware.cyr`;
- `src/lib/router.cyr`.

**The drifted copies, now gone:**
- **The router** (`router_new`, `router_select`, `router_report_latency` and the `RouterField` enum).
  - **Health:** the copy had no health filter. `src/` skips a depooled route, and serves the unfiltered set when
    every candidate is down.
  - **Unmeasured routes:** the copy ranked a route with no latency recorded as the slowest, so it was never tried.
- **A second lowest-latency copy, `_t_ll_pick`,** ranked that route as 0 ms, so it always won.
  - **What `src/` does:** it counts the route as `LAT_UNMEASURED_MS` (50 ms). It is tried while every measured route
    is slower, and a route that fails (failures record no latency) cannot win every pick.
  - **Effect on the old test:** `_t_ll_pick`'s "an untried route beats one measured at 50 ms" is false against `src/`,
    where the tie keeps priority order.
- **The builder mask:** the copy said ai-hwaccel has 17 detection backends; it has 18 since `agnos-gpu`.
- **The simulator's shard count:** an inline greedy loop, written before the shared-RAM pool existed.
- **ai-hwaccel's estimators:**
  - the placement estimate's copy wrapped negative where ai-hwaccel now saturates;
  - the training estimate's copy charged the optimizer 2x its base, where ai-hwaccel charges 7x (AdamW);
  - the training estimate's copy also multiplied by 2^30 before dividing.

**Also replaced:**
- the copies of the available-VRAM arithmetic, the simulator's totals and `_sim_shared_part`;
- a transcription of ai-hwaccel's `reg_total_accel_memory`;
- the copied caps `SIM_MAX_DEVICE_BYTES` and `HW_MAX_MODEL_PARAMS`.

The tests now build synthetic ai-hwaccel profiles and registries (`_t_prof`, `registry_from_profiles`) and call the
real functions.

**Two groups tested nothing real:** `hardware_unknown_sentinel` and `hardware_simulate_fits` did arithmetic on local
variables. They now drive `hw_available_vram`, the telemetry writer and `hw_simulate_json`.

The handlers' whole-number parser and requirement-match scan stay mirrored, because `handlers.cyr` cannot be
included. They use the real caps now.

### Fixed — a stale comment in `hardware.cyr`

The comment above `HW_MAX_MODEL_PARAMS` described the 2.6.13 estimators: training at 5.5x its base, wrapping near
8.4 × 10^14 parameters, so 8x headroom.
- **Against ai-hwaccel 2.5.6:** placement and the simulator saturate, and training-estimate totals 10.5x its base.
  Its bytes stay exact to about 4.4 × 10^17 parameters, over 4000 times the cap.
- **The cap stays:** it keeps every answer a real byte count.
- **Pinned by tests:** the `hardware_planner_caps` group checks each of these numbers against the real estimators.

### Added — tests for behavior no test had reached

- **`router_health`:**
  - an unhealthy first choice is skipped;
  - with every candidate down, priority order applies and a disabled route stays out;
  - round robin rotates over the healthy routes only.
- **`router_sort`:** `router_new` orders routes by priority and keeps the given order among equals.
- **`router_select_local`** (DLP-confidential traffic): the route is never remote, disabled routes are skipped, it
  answers 0 when no local route serves the model, and round robin works among the local routes.
- **`router_select_hw`:**
  - no hardware detected, or an unknown model size: the plain pick;
  - a model that fits the free VRAM stays local, and 70B on an 8 GiB card goes remote;
  - the VRAM reserve counts: a 7 GiB reserve sends a 1B model remote too;
  - an unhealthy remote, or no remote serving the model, falls back to local;
  - round robin applies among the remotes.
- **`hardware_real`, with nothing detected:** not ready, INT4 suggested, no plan, `{"detected":false}`, `[]`, and a
  placement error.
- **`hardware_real`, on a synthetic host:**
  - device, GPU and memory counts, and free VRAM;
  - the summary's counts, best device and profiles, with an unavailable card listed as such and not counted as
    capacity;
  - telemetry that leaves out the CPU and writes `null` for anything a card did not report;
  - placement that keeps the asked quantization, and lists cloud alternatives only when the model does not fit.
- **`hardware_real`, settings and helpers:**
  - a refresh inside the cache TTL does not re-probe, and no refresh thread starts when the interval is 0;
  - negative `[hardware]` values leave settings alone;
  - the data-file check finds both files from the repo root and counts both missing under another `data_dir`;
  - `_hoosh_parse_model_params` reads sizes out of model names.
- **The simulator:**
  - `remove_count` drains accelerators, never the CPU;
  - the snapshot's own `shards_required` agrees with `_sim_shards` over the shared pool, on the Apple and laptop
    layouts.

### Mutation-checked

A copy of the tree was changed 34 ways, and every mutant is caught:

| mutants | result |
|---|---|
| `router_select` without its health filter, or without the all-down fallback | 3 and 2 fail |
| an unmeasured route at 0 ms; a latency tie going to the later route | 2 and 1 fail |
| `router_new`'s sort unstable among equals; the EMA weights swapped | 2 fail each |
| `router_select_hw` ignoring the fit, using total VRAM, or counting an unhealthy remote | 3, 1 and 1 fail |
| no round robin in the remote or the local pick; `router_select_local` allowing remote routes | 1, 1 and 4 fail |
| free VRAM subtracting -1, not floored at 0, or ignoring the reserve | 3, 2 and 4 fail |
| disabling a backend toggling it; the last backend unreachable by name; a negative reserve applied | 1 fails each |
| telemetry writing -1 or listing the CPU; the summary's `available` inverted | 3, 1 and 2 fail |
| placement never listing cloud alternatives, or overriding the asked quantization | 1 and 2 fail |
| a refresh ignoring the cache TTL | 2 fail |
| the simulator dropping the shared pool, or counting system RAM toward `fits` | 13 and 1 fail |
| shards in registry order; the pool spent per device | 5 and 3 fail |
| `_sim_profiles` removing the CPU or sharing an added card; no clamp in `_sim_shared_part` | 2, 2 and 1 fail |
| the model-size cap raised; the second data file not counted; `data_dir` not applied | 1 fails each |

The TTL mutant first crashed the suite instead of failing an assertion. The test had not created the registry lock
that `hw_init` makes at startup, so the mutant's re-probe reached a lock that did not exist. The test now creates it
as startup does.

### Notes

- The test file is 7,341 lines (was 7,227). Symbol coverage rose from 344 to 369 of 684.
- The bench binary differs from 2.9.8's by one byte, the version string. On a quiet machine all 25 benchmarks are
  within 1 ns or 1% of 2.9.8's. A first run went alongside another workload's compiles (load 6.7) and doubled some
  numbers on identical code, so it was discarded rather than recorded.
- **No remaining module can be included as it stands.** Each one sits in a dependency cycle, so the next step is a
  split (see the roadmap). The smallest: `health.cyr` needs only `route_is_remote` from `provider.cyr`, a route
  predicate that could live in `route.cyr`.

## [2.9.8] — 2026-11-04

**The tests run the real plaintext provider client and the real event bus.** The test file's copy of
`_build_req_header` had drifted: it never wrote the `traceparent` header `src/` sends. The event bus had no tests
at all. No gateway code changed. **1355 assertions** (was 1291) in 188 groups. All gates are clean: fmt, lint, vet,
deny, coverage (344/684, 50%; **23 of 37** modules included for real), the security scan and the four fuzz targets.

### Changed — `http_client.cyr` and `events.cyr` are included for real

`tests/hoosh.tcyr` now includes three more files:
- `src/lib/http_client.cyr`;
- `src/lib/events.cyr`;
- the vendored majra bundle (`src/vendor/majra.cyr`), which `events.cyr` publishes to. It is included at the same
  point as in `src/main.cyr`. As there, it shares one name with `types.cyr`: the enum member `ERR_NONE`, which is 0
  in both.
- **The `_build_req_header` copy is gone.** It dated from before `trace.cyr` was in the test harness; its comment
  still said so. It reserved 64 bytes for the traceparent but never wrote the header. So no test could notice the
  header going missing, or check that the capacity bound counts it.

### Added — tests for behavior no test had reached

- **`http_client`, the header builders.** Both `_build_req_header` and `_build_post_header`:
  - write the Host line, the full JSON media type, `Connection: close` and the body's Content-Length;
  - write the thread's current trace id as `traceparent`, the last header before the blank line.
- **`http_client`, the capacity bound.** A sweep over every cap from 0 to 400 bytes: each builder either refuses
  (-1) or writes inside the buffer.
- **`http_client`, `host_addr`.** No host and `localhost` give loopback; dotted quads are parsed.
- **`http_client`, `stream_skip_headers`.** Over a pipe:
  - it returns the body's offset;
  - a buffer one byte bigger than the header block works, one the header block fills is refused;
  - EOF before the blank line is -1.
- **`http_client`, `http_post_stream` over loopback,** single-threaded. The client's connection waits in the listen
  backlog, so the test can accept it after the client has written.
  - On the wire: the request line, Host, Content-Length, traceparent and body.
  - After the reply: the response headers are skipped and the SSE body stays in the fd to be read.
- **`http_client`, `http_req_local`,** which blocks until it gets the reply, against a one-shot server thread:
  - POST and GET round trips, with the upstream status (503) passed through;
  - a GET sends `Content-Length: 0` and no body;
  - a backend that closes without answering gives 0;
  - with `HTTP_RESP_CAP` lowered to 32: a 31-byte reply is parsed whole, and a 32-byte one is refused as truncated;
  - a refused connection gives 0, or -1 from `http_post_stream`.
- **`events`, before `events_init`:** every builder is a no-op, nothing is published, and `/v1/events/recent` answers
  an empty list. **After it:**
  - a second `events_init` keeps the bus and its count;
  - each builder's JSON is checked, with the model, error and base URL escaped, and health as `true` / `false`;
  - after 200 more events, the ring holds the newest 128, oldest first, and the majra bus has counted all 205.

These are the suite's first sockets and its first thread. The listener binds 127.0.0.1 on an ephemeral port. Its
accept and the server's reads time out after 5 s, so a broken client fails a test instead of hanging the suite (CI's
5-minute cap stays as the backstop). The suite still runs in about 2 s.

### Mutation-checked

A copy of the tree was changed 21 ways. 19 mutants are caught; the 2 that survive are equivalent (see below):

| mutant | result |
|---|---|
| `_build_req_header` not writing the traceparent | 1 fails |
| `_build_post_header` not writing the traceparent | 3 fail |
| either builder's capacity check leaving out the traceparent | 1 fails each (299 overruns in the sweep) |
| Content-Length written as 0 | 1 fails |
| `http_req_local` cap check `>` instead of `>=` | 1 fails |
| `http_req_local` parsing an empty reply | 1 fails |
| `http_req_local` not sending the body | 1 fails |
| `http_post_stream` not sending the body | 2 fail |
| `stream_skip_headers` bound off by one | 1 fails |
| `stream_skip_headers` reading in chunks (consuming the body) | 4 fail |
| the recent-events ring not capped at 128 | 2 fail |
| recent events listed newest first | 3 fail |
| `events_init` not idempotent | fails, then crashes |
| a failed event's model not escaped | 1 fails |
| health `true` / `false` inverted | 2 fail |
| `event_emit` not counting an event | 8 fail |
| `event_emit` not publishing to the bus | 4 fail |
| the completed event dropping its token count | 1 fails |
| `host_addr` without its dotted-quad shortcut | survives |
| `host_addr` without its `localhost` shortcut | survives |

**Why the two survivors are equivalent:** sandhi's resolver returns the same address for both inputs. Removing a
shortcut costs a lookup, not a wrong address, so no test can see the difference.

### Notes

- The test file is 7,227 lines (was 6,912).
- Symbol coverage rose from 332 to 344 of 684.
- The bench binary differs from 2.9.7's by one byte, the version string. The 25 benchmarks ran on identical code.
- With `http_client` and `events` in, `health.cyr` and `retry.cyr` each need only `provider.cyr`.

## [2.9.7] — 2026-11-04

**The tests run the real route records and response cache.** The test file's copies of both had drifted. The route
copy was a 64-byte record, while `src/`'s is 96 bytes and holds four more fields. The cache copy predated TTL and
LRU: it stored values directly, and at capacity it counted an eviction without evicting anything. No gateway code
changed. **1291 assertions** (was 1274). fmt, lint, vet, deny, coverage (332/684, 48%; **21 of 37** modules included
for real), the security scan and the four fuzz targets are clean.

### Changed — `route.cyr` and `cache.cyr` are included for real

`tests/hoosh.tcyr` includes both modules and drops their copies.
- **The route record and its accessors:** the `RouteField` enum, `route_new`, `route_provider` / `_priority` /
  `_enabled` / `_base_url`, `route_set_enabled`, `route_set_patterns` and `route_matches_model`.
  - **Layout:** the copy's record was 64 bytes. `src/`'s is 96, adding the output cap, the health record, the TLS
    policy and the catalog mode (`ROUTE_SIZE`).
  - **Empty patterns:** the copy matched an empty pattern against an empty model name. `src/` skips empty patterns.
- **The cache:** the `CacheField` enum, `cache_new`, `cache_get` and `cache_insert`.
  - **Arguments:** `cache_new` takes a third argument, `ttl_secs`.
  - **Behavior:** the copy stored values with no entry struct, never expired anything, and at capacity bumped the
    eviction counter without evicting. `src/` keeps value, last-access and created stamps per entry, expires on read
    (counting both an eviction and a miss, and dropping the semantic-index entry), and at capacity sweeps expired
    entries before evicting the least recently used.
  - **Ripple:** the six existing callers pass TTL 0. The capacity test now also asserts the cache holds at three
    entries.
- **The parallel TTL-cache mirror is gone.** `_t_tc_*` and its field enum duplicated the real cache so that time could
  be injected. The real cache reads `hoosh_now_ms()`, which returns `logging.cyr`'s coarse-clock value once that clock
  has started. So the tests now set time with `_t_at(ms)` (plus `_t_ins_at` / `_t_get_at` / `_t_sweep_at`) and give
  the clock back with `_t_clock_real()`. Every TTL assertion passed unchanged against the real module: that mirror
  had stayed faithful, while the plain cache copy had not.
- **`cache_clear` existed only in the test file;** `src/` has none. Its assertion is gone. `cache_len` and the
  hit/miss/eviction readers stay, as readers of the real struct.
- **The per-route health record:** `_health_record_new`, the `HEALTH_*` layout and `UNHEALTHY_THRESHOLD` are now
  `route.cyr`'s own. The state machine, `health_record`, lives in `health.cyr`, which cannot be included yet, so it
  stays mirrored. The mirror now works on the real record and threshold.

### Added — tests for behavior no test had reached

- **`route_real`:**
  - a new route is healthy until probed;
  - it asks for a live catalog by default, except OpenRouter and Whisper;
  - it has no output cap, rate limiter or TLS policy until configured, and the cap and limiter setters work;
  - an empty pattern matches nothing, while a prefix pattern still matches.
- **`cache_lru_eviction`:** with nothing expired, a full cache evicts the least recently used entry, and a read
  refreshes recency.
- **The capacity case plain LRU gets wrong:** an entry that has expired but was read after a live one was stored. Only
  the expired sweep keeps the live entry.

### Mutation-checked

A copy of the tree was changed seven ways, and every mutant is caught:

| mutant | result |
|---|---|
| an expired read not counted as a miss | 1 fails |
| TTL measured from last access instead of creation | 5 fail |
| LRU evicting the most recently used entry | 2 fail |
| a full cache skipping the expired sweep | 1 fails |
| an empty route pattern matching again | 1 fails |
| OpenRouter defaulting to a live catalog | 1 fails |
| `UNHEALTHY_THRESHOLD` 3 → 1 in `route.cyr` | 5 fail |

The expired-sweep mutant first survived. In the existing capacity test the expired entry was also the
least-recently-used one, so plain LRU evicted the same entry. The new capacity case above is what catches it.

### Notes

The test file is 160 lines shorter (7,072 → 6,912). Symbol coverage rose from 323 to 332 of 684. `router.cyr` now
needs only `hardware.cyr` to be includable.

### Performance

The bench binary is byte-identical to 2.9.6's apart from the version string. No benched path changed. The
2026-11-04 rows in `bench-history.csv` after 2.9.6's are this release's mandatory run.

## [2.9.6] — 2026-11-04

**The tests run on the real `types.cyr`.** It holds the shared types every other module builds on, and the test
file had kept a copy of nearly all of it: the error, provider and strategy enums, the error and provider helpers,
the sampling renderers and the retry classifier. No gateway code changed. **1274 assertions** (was 1261; six dead
ones removed, 19 added). fmt, lint, vet, deny, coverage (323/684, 47%; **19 of 37** modules included for real), the
security scan and the four fuzz targets are clean.

### Changed — `types.cyr` is included for real

- **Ordering.** `tests/hoosh.tcyr` includes it right after the vendored bote-core, first among hoosh's modules, as
  `src/main.cyr` orders them.
- **Removed:** the test file's own copies of
  - the `HooshErr`, `Provider` and `Strategy` enums;
  - `error_http_status`, `error_code_str`, `provider_is_local` and `provider_name`;
  - `_T_GEN_UNSET`, `_t_sb_add_milli`, `_t_status_retryable` and `_t_default_url`.
- **Kept as thin helpers:** `_t_milli_str` and `_t_sampling_str` now render the real `_sb_add_milli` and
  `_sb_add_sampling` (through `genopts_new`) instead of copying them.
- **What the comparison found**, checked before switching:
  - The `Provider` and `Strategy` enums matched value for value. `HooshErr` matched apart from `ERR_NONE = 0`,
    which the copy lacked.
  - `error_http_status`, `error_code_str`, `provider_name`, `status_retryable` and `_sb_add_milli` were identical.
    `provider_is_local` and the sampling renderer were the same logic written differently.
  - **`provider_default_url`:** the copy covered three providers, by bare number (10, 12, 6). The real one covers
    all 17 by enum name.
  - **`error_is_retryable`:** six assertions tested a function that existed only in the test file, a port of rust-old's
    `HooshError::is_retryable`. `src/` has none; production decides retries from the provider's HTTP status
    (`status_retryable`, tested in `status_retryable`). The assertions are gone. One asserting `ERR_NONE`'s mapping
    takes their place.

### Added — tests for `types.cyr` code no test had reached

`types_real`:
- **Two invariants over all 17 providers:**
  - `provider_parse(provider_name(p)) == p`, so a `[[providers]] type` written the way hoosh prints it always loads;
  - the default base URL is `https://` exactly for the remote providers, and `http://localhost:` for the local ones.
- **`provider_parse`:** names are case-insensitive, the `tensorrt-llm`, `trtllm` and `onnxruntime` aliases work, and
  an unknown name gives -1.
- **`_to_lower`** folds only A–Z.
- **The generation-options accessors** (`genopts_new`, `gen_effort` / `_max_tokens` / `_temp` / `_top_p`,
  `gen_set_effort_name`, `gen_set_usage`) on a real struct, and the documented defaults on a 0 pointer, the batch and
  CLI paths: zeros, and `GEN_UNSET` for the sampling values.

### Mutation-checked

A copy of the tree was changed seven ways, and every mutant is caught:

| mutant | result |
|---|---|
| the `trtllm` alias dropped from `provider_parse` | 1 fails |
| Groq's default URL loses its `/openai` prefix | 2 fail |
| 408 treated as permanent by `status_retryable` | 1 fails |
| Ollama's output cap rendered as `max_tokens`, not `num_predict` | 2 fail |
| vLLM no longer counted as local | 2 fail |
| a timeout mapped to 504 instead of 408 | 1 fails |
| Grok's canonical name changed so it no longer parses back | 1 fails |

### Notes

The test file is 136 lines shorter (7,208 → 7,072). Symbol coverage rose from 312 to 323 of 684.

A dependency scan of the 18 modules still mirrored shows five that need no hoosh module the tests lack:
- `route.cyr` (17 same-name copies) and `cache.cyr` (6) can be compared and swapped as compression and semantic
  were;
- `http_client.cyr` (one collision, plus sockets) needs nothing more;
- `events.cyr` needs the vendored majra bundle, and `hardware.cyr` the ai-hwaccel bundle.

The rest need a split or their dependencies first. The roadmap lists them.

### Performance

The bench binary is byte-identical to 2.9.5's apart from the version string. No benched path changed. The
2026-11-04 rows in `bench-history.csv` after 2.9.5's are this release's mandatory run.

## [2.9.5] — 2026-11-04

**The tests run the real trace and metrics code, and the coverage script describes the suite as it is now.** These
are the next two modules of the roadmap's mirror-drift item. Unlike 2.9.3 and 2.9.4, their copies had not drifted.
The gain is that they can no longer drift, and that real code no test touched before is now tested: the trace-id
generator, the thread-local current id, the request and token counters, and the Prometheus histogram output.
No gateway code changed. **1261 assertions** (was 1243). fmt, lint, vet, deny, coverage (312/684, 45%), the
security scan and the four fuzz targets are clean.

### Changed — `trace.cyr` and `metrics.cyr` are included for real

- **The copies are gone.** `tests/hoosh.tcyr` includes both modules and drops five copies: `_t_trace_extract`,
  `_t_trace_ci_eq` and `_t_trace_is_hex`, plus `_t_lat_bound` and `_t_lat_bucket`. Before deleting them I compared
  each with its source, and all were identical. `metrics.cyr` is included after the test file's Provider enum and
  `provider_name`, which it needs.
- **Latency buckets.** The bucket assertions now record through the real `metrics_record_latency` and read the
  module's own histogram storage (`_t_lat_slot`), rather than a copy of the bucket loop. New assertions cover:
  - the per-provider count and sum;
  - an out-of-range provider, which records nothing;
  - the Prometheus exposition from `metrics_emit_latency`: the `histogram` TYPE line, a bucket's labelled line,
    the cumulative le=100 bucket and `+Inf`.
- **Counters:** `metrics_record` sums requests and prompt and completion tokens; `metrics_reset` zeroes them.
- **Trace ids.**
  - `trace_extract` is the real one.
  - New assertions for `trace_generate`: an id is 55 bytes, with version `00` and flags `01`; two ids differ; a
    generated id passes `trace_extract`'s own W3C validation.
  - `trace_set` / `trace_current_or_new`: the id set is the one read back; when unset, one id is generated and
    then kept until reset.
- **Mutation-checked.** A copy of the tree was changed five ways, and every mutant is caught:

  | mutant | result |
  |---|---|
  | a latency sample exactly on a bucket's bound goes to the next bucket | 4 fail |
  | the histogram stops being cumulative | 1 fails |
  | the traceparent hex check removed | 1 fails |
  | `trace_current_or_new` stops storing the id it generated | 1 fails |
  | the generator writes flags `00` | 1 fails |

### Changed — `scripts/coverage.sh` says what the suite is

Its header still said no test ever executes a line of `src/`. That has not been true since modules began to be
included for real. The header now:
- explains the two halves, modules included for real and modules mirrored;
- states the name-based limitation both ways. A real module's helper that runs only through its caller counts as
  uncovered, and a same-name copy used to count as covering one. That is the 2.9.4 dip from 306 to 301.

The script also prints which `src/lib` modules the suite includes for real. That line is informational; the gate
is unchanged. Today it lists **18 of 37**: auth, catalog, compact, compression, dlp, http_server, jsonlite, logging,
mcp, metadata, metrics, outcome, pricing, provider_err, ratelimit, semantic, trace and urlparse.

### Performance

The bench binary is byte-identical to 2.9.4's apart from the version string. No benched path changed. The
2026-11-04 rows in `bench-history.csv` after 2.9.4's are this release's mandatory run.

## [2.9.4] — 2026-11-04

**The tests run the real response-compression and semantic-cache code.** The test file had kept its own copies of
both modules under the same function names, and the copies had drifted from `src/` in ways that mattered. The tests
were exercising a string scan with a bug that `src/` had already fixed, two parser hangs that `src/` had already
guarded, a return shape `src/` no longer used, and a cache-remove that only worked because the test used integer
keys. The roadmap's mirror-drift item takes its next two modules. No gateway code changed.
**1243 assertions** (was 1226). fmt, lint, vet, deny, coverage (306/684), the security scan and the four fuzz
targets are clean.

### Changed — `compression.cyr` and `semantic.cyr` are included for real

`tests/hoosh.tcyr` includes both modules and drops 18 same-name copies, along with the global defaults they carried.
How the copies had drifted:
- **`_cmp_span_end` / `_cmp_str_field` (compression).** The copies decided a quote was escaped when the byte before
  it was `\`. That misreads a string ending in an escaped backslash (`"C:\\"`), where the scan ran on past the
  object. `src/` skips each escape pair instead. New assertions cover an object whose value ends in `\\`, and the
  field that follows it.
- **`compress_ws_json` / `prune_tool_pairs`.** `src/` returns a `(ptr, len)` pair; the copies returned a C string, an
  API that no longer existed. The `_compress` / `_prune` helpers now read the pair.
- **`semantic_insert` / `semantic_find`.** The copies predated per-model entries. `src/` delegates to
  `semantic_insert_model` / `semantic_find_model`, which store the model with each entry.
- **`_sem_parse_num` / `semantic_parse_embedding`.** The copies lacked the exponent clamp (at most 40 decades) and the
  no-progress guard. Without the guard, a bare `NaN` (Python's `json.dumps` emits one by default) spins and allocates
  until the heap runs out. Without the clamp, an exponent like `1e9223372036854775` loops ~9e15 times. New
  assertions feed both.
- **`semantic_remove`.** The copy (`_t_sem_remove`) compared keys as pointers, and its test used integer keys. That is
  the bug that once left the real expired-entry removal dead. The test now stores string keys and removes one through
  a separately allocated copy of the same text.
- **The semantic store ships disabled.** The copy had defaulted it on, so the tests now call `semantic_configure`
  first, as `[cache]` does.

### Added — tests for code that had none

- **`semantic_model_scope`.** `semantic_insert_model` / `semantic_find_model` had no test. Their own comment records
  the bug they prevent: a `claude-opus-4-1` request served the `gpt-4o-mini` answer. The new group checks that:
  - another model's entry never answers;
  - the same model, compared by text, does;
  - an unscoped query finds only unscoped entries;
  - `max_search` bounds the scan;
  - a disabled store finds nothing.
- **The compression switch.** `compression_configure` / `compression_is_enabled`: off by default, on, and off again.

### Mutation-checked

A copy of the tree was changed seven ways, and every mutant is caught:

| mutant | result |
|---|---|
| `_cmp_span_end` back to the copy's escape check | 1 assertion fails |
| `_cmp_str_field` back to the copy's escape check | 1 fails |
| `semantic_remove` by pointer | 4 fail |
| the `NaN` no-progress guard removed | the suite hangs |
| the exponent clamp removed | the suite hangs |
| an unscoped query matching every entry | 1 fails |
| any model matching any model | 1 fails |

The first version of the scope test let the unscoped mutant through. Its scoped entry shared `key-cat`'s vector, so
the first equal match won either way. The entry now uses a vector only it matches above the threshold.

### Changed — CI fails a hung test suite in minutes

Two of those mutants hang the suite rather than failing an assertion, and a GitHub Actions job waits 6 hours by
default. The *Test suite* step in `ci.yml` gets `timeout-minutes: 5`; the suite takes ~2 s.

### Notes

- The test file is 397 lines shorter (7,611 → 7,214).
- Symbol coverage dipped to 301 when the copies went, then returned to 306. `scripts/coverage.sh` counts a `src/`
  function as covered when its name appears in `tests/`. The deleted copies had carried the names of seven internal
  helpers (`_compress_is_ws`, `_cmp_is_role`, `_cmp_collect_ids`, `_cmp_id_in`, `_sem_find`, `_sem_is_digit`,
  `_sem_parse_num`) that now run for real through their callers. The new tests above reference five functions the
  suite had never named.

### Performance

The bench binary is byte-identical to 2.9.3's apart from the version string. No benched path changed. The
2026-11-04 rows in `bench-history.csv` after 2.9.3's are this release's mandatory run.

## [2.9.3] — 2026-11-04

**Remote requests stop leaking megabytes each, and the tests stop mirroring four modules.** The roadmap's
connection-pooling measurement turned up something much bigger. Every remote request kept its sandhi allocations on
hoosh's never-freeing heap:
- **~2.2 MB resident and ~8.5 MB of address space** per buffered request (non-streaming chat, embeddings, the OTLP
  export);
- **~6.4 MB resident and ~26 MB of address space** per stream.

At one remote stream a second that is ~23 GB of RSS an hour. The roadmap had tracked only the local path's
~1 KiB/request. Both remote paths now use a per-thread scratch arena. A residue of ~170–190 KB per new TLS
connection remains; it is in sandhi, which ignores the arena for TLS state, and is filed in the roadmap.
**1226 assertions** (was 1215). fmt, lint, vet, deny, coverage (306/684, from 299/683), the security scan and the
four fuzz targets are clean.

### Fixed — remote requests no longer accumulate on the heap

- **The cause.** sandhi allocates everything an exchange needs through the allocator it is given: the connection,
  record buffers, and a receive buffer of `max_response_bytes + 1`. That buffer is allocated up front, and hoosh
  sets the cap to 8 MiB (`REMOTE_MAX_BYTES`). hoosh called the default-allocator variants, so all of it went onto the
  global bump heap, which never frees.
- **The fix: `_remote_arena()`** (`provider.cyr`).
  - Each thread that makes remote calls (the pool workers, or main for `hoosh infer`) owns one growable arena, in
    fixed thread-local slot 6.
  - It is reset at the **start** of that thread's next remote call, so it settles at the largest single exchange.
  - That is safe because every caller is done with the previous response by then. Kept bodies are copied out (chat's
    `rb`, the semantic-cache embedding), the error capture copies into its own buffer, a response relayed to the
    client is written before the handler returns, and the stream callback copies text, tool fields and usage out of
    sandhi's event buffers.
  - The stream's status is read before the empty-stream fallback calls `retry_forward`, which resets the arena.
  - Used by: chat (`provider_forward`), `/v1/embeddings`, the semantic-cache embedding, the remote stream
    (`sandhi_http_stream_opts_a`) and the https OTLP export (`sandhi_http_post_a`).
- **Measured live** against a local HTTPS upstream, each request with a unique prompt (a repeated prompt is answered
  from the response cache and never reaches upstream, which hid this in a first run):

  | path | 2.9.2 | 2.9.3 |
  |---|---|---|
  | buffered, 60 requests | RSS 12 → 148 MB; VmSize +256 MB | RSS 12 → 37 MB; VmSize flat |
  | buffered, 200 requests | — | RSS 12 → 64 MB; VmSize flat |
  | stream, 90 requests | RSS 16 → 592 MB; VmSize 276 → 2,373 MB | RSS 16 → 76 MB; VmSize flat |

  The early rise in 2.9.3 is each worker's arena growing once. After it, growth is ~170–190 KB per request.
- **What remains is upstream.** sandhi's arena-aware connect (`_sandhi_conn_finalize_with_early_data_a`) opens TLS
  with `tls_connect_alloc(fd, …)`, the global-heap wrapper, instead of `tls_connect_alloc_in(a, fd, …)`. So each new
  TLS connection's state (~179 KB) still lands on the heap. A standalone probe isolates it: with an arena, plain HTTP
  costs 856 B per request and HTTPS 178,768 B. bote 3.6.5 had already filed it
  (`sandhi/docs/development/issues/2026-11-02-bote-tls-connect-keeps-each-connection-on-the-heap.md`); hoosh's
  measurements and fix constraints were added there as a second reporter.
- **Still correct:**
  - a stream that fails before emitting falls back to the buffered path and reports the provider's 500;
  - a buffered 500 carries the provider's message;
  - remote keep-alive still fires at 15.5 s.

  All 200 buffered and 90 streamed requests returned 200.

### Changed — four modules lose their test mirrors

`tests/hoosh.tcyr` already included `dlp.cyr`, `compact.cyr`, `http_server.cyr` and `pricing.cyr` for real, yet
it still carried copies of their functions. That is the roadmap's mirror-drift item, and two copies had drifted:
- **`req_content_length`.** The real one was hardened later (the header must start a line, the value is capped
  against overflow, and a value with no digits is refused), and the copy never followed. It and `_hdr_ci_eq` were
  dead: the real-function tests had already replaced their callers.
- **The pricing and cost tracker.** A whole legacy copy sat in the tests: its own `PRICE_*` table,
  `pricing_lookup_packed`, `cost_calculate`, and a `cost_tracker_*` keyed by provider **name**. That is the keying
  2.5.6 replaced with (provider, base_url), so `cost_accumulation` was testing the old design. Every cost assertion
  now runs against `estimate_cost_micro`, `cost_record`, `cost_record_count` / `_at`, `cost_total_micro` and
  `cost_reset`. Every expected figure held.
- **DLP.** `_dlp_find_ci` is the real one. A `max()` helper that stood in for `dlp_custom_level`'s fold is replaced by
  tests of the real `dlp_custom_add` / `dlp_custom_level` / `dlp_custom_hit_name`: highest level wins, order does not
  matter, no match gives -1, the first matching rule is named, and an empty pattern is refused.
- **Compaction.** `_cmp_ws` / `_msg_role_is` are the real ones. A duplicate `CHARS_PER_TOKEN_X10` global is gone.
- **Mutation-checked.** A copy of the tree was changed three ways:
  - cost records keyed by provider only fails 3 assertions;
  - DLP keeping the first match instead of the highest level fails 1;
  - a role matcher that skips the name compare **passed**. The closing-quote check alone rejected `"user"` and
    `"systemx"` by length, so a new assertion uses a same-length role (`"tester"`), and that mutant now fails.
- The test file is 212 lines shorter.

### Measured — remote connection pooling (not adopted yet)

- **Method.** A probe made alternating fresh and pooled HTTPS GETs with sandhi's pool against a 2xx CDN endpoint.
  sandhi pools only after a 2xx/3xx response, so a provider's unauthenticated 401 cannot show reuse.
- **Result.** A fresh request takes a median 40.0 ms, a reused one 6.7 ms: **−33 ms per request** at a ~6 ms RTT, and
  more for a distant provider. About 12 ms is the two extra round trips and about 20 ms native TLS handshake CPU.
  curl reached the same server in ~25 ms fresh.
- **Why not yet.** Pooled connections must outlive the request, so they cannot live in the request arena that this
  release introduces. Also, sandhi returns a failed send or read on a pooled connection to the caller rather than
  retrying it on a fresh one (`_sandhi_http_exchange_keepalive_a`), so a stale socket would become a retried POST.
  The roadmap records both preconditions.

### Performance

The bench binary is byte-identical to 2.9.2's apart from the version string. No benched path changed. The
2026-11-04 rows in `bench-history.csv` after 2.9.2's are this release's mandatory run.

## [2.9.2] — 2026-11-04

**Quick wins from the 2026-11-04 roadmap review.** Five fixes, none of which changes an API:
- remote streams send keep-alives;
- an installed gateway finds its hardware data files, and says so when it cannot;
- a refused MCP tool registration is logged;
- the archived Rust tree is removed;
- the last two bare syscalls go through the stdlib.

Same toolchain and dependencies as 2.9.1. **1215 assertions** (was 1205). fmt, lint, vet, deny, coverage (299/683),
the security scan and the four fuzz targets are clean.

### Added — remote SSE keep-alive

- **The problem.** A local stream has sent `: keep-alive` after every 15 s of backend silence since 2.7.0, so a proxy's
  idle timeout (nginx and most load balancers: 60 s) does not cut a stream while a model loads or thinks. A remote
  stream could not, because sandhi gave hoosh no turn while the upstream was quiet.
- **The fix.** sandhi 1.10.5 added the hook hoosh filed for on 2026-09-25 (`sandhi_http_options_idle_ms` /
  `_idle_cb`). `_remote_stream_opts` now sets it, with `SSE_KEEPALIVE_S`, to the new `http_sse_idle_keepalive`
  (`http_server.cyr`). That callback writes the same comment to the client fd in the stream ctx.
- **A client that has gone stops the stream.** The callback returns 0 when the write fails, and sandhi then stops
  the stream. Before, the stream kept its worker until the upstream wrote again or the 300 s `read_ms` passed.
- **Verified live over TLS.** A route pointed at a local HTTPS upstream, trusted through `tls_pinned_certs`, sent
  SSE headers and then stayed silent for 20 s:
  - 2.9.2 sent `: keep-alive` at 15.5 s and the content at 20.5 s;
  - 2.9.0 sent nothing for the full 20 s;
  - with 40 s of silence and a client that left at 17 s, hoosh held no upstream connection by 33 s, and the upstream's
    next write hit a broken pipe.
- Tests: `remote_sse_keepalive`, 5 assertions on a real pipe, covering a live client, a gone client and no ctx.

### Fixed — hardware data files on an installed gateway

- **The problem.** ai-hwaccel reads `data/cloud_pricing.json` (POST `/v1/hardware/cost`) and `data/models.json`
  (POST `/v1/hardware/models`, `/v1/hardware/compatible-models`) at run time. hoosh never told it where they were,
  so they resolved against the working directory or `AI_HWACCEL_DATA_DIR`, which nothing documented. The release
  shipped only the binary. So an installed gateway answered those endpoints with empty lists and logged nothing.
- **`[hardware] data_dir`** (new) is the directory that contains `data/`. It is passed to ai-hwaccel's
  `data_dir_set` on every config load. When unset, ai-hwaccel tries `$AI_HWACCEL_DATA_DIR`, then the working
  directory, as before.
- **A startup WARN.** `hoosh serve` checks both files, and so does each reload (`/v1/admin/reload` and SIGHUP). A
  missing file is named with the path tried, the endpoints it affects, and the two ways to fix it.
- **Reload picks up new data.** A reload re-applies `data_dir` and drops ai-hwaccel's cached pricing rows, so a
  replaced `cloud_pricing.json` is read on the next request. Removing the key falls back as above.
- **The release ships the data.** `release.yml` packages both files as `hoosh-<version>-data.tar.gz` (extracting to
  `data/`) and lists it in `SHA256SUMS`.
- **Verified live:**
  - Started outside a checkout, the gateway logged both WARNs and recommended 0 instances.
  - With `data_dir`, or with the env var, it recommended 5.
  - Reloads removing and restoring the key moved it from 5 to 0 (WARNs) and back to 5.
  - A simulated release, with the binary and the extracted tarball elsewhere, answered both endpoints.
- README (*Hardware data files*) and the sample `hoosh.cyml` document the key and the env var.

### Fixed — a refused MCP tool registration is logged

- **The problem.** `mcp_init` ignored what `dispatcher_register_tool` returned. Since bote 3.4.8, registration checks
  the definition (`registry_check_tool_def`), so a refused tool would just be missing from `tools/list`. bote logs
  most refusals itself, but not a bad name, which `validate_tool_name` refuses before the check.
- **The fix.** The new `mcp_register_tool` returns 1 or 0, and logs every refusal as `mcp: tool registration
  refused: <name> - <bote's reason>`.
- **Tests use the real module.** `tests/hoosh.tcyr` now includes `src/lib/mcp.cyr` instead of mirroring its echo
  handler, and drives `mcp_init` itself. That is one mirror fewer, the roadmap's drift item. New assertions check
  that a bad name and a non-object `inputSchema` are refused (0) and not registered, and that a valid second tool
  registers (1).

### Changed — `rust-old/` removed

The archived Rust implementation (74 files) is deleted, as [rust-old-retirement.md](docs/development/rust-old-retirement.md)
planned once 2.7.1 was tagged. Nothing in the build, CI, scripts, tests, fuzz targets or coverage read it. The tree
stays in git history at tag **2.7.1**, and the `rust-old <file>:<line>` provenance comments in `src/` and this
CHANGELOG refer to it there. README, `docs/architecture/overview.md`, the retirement document and `docs/index.md`
say so in the past tense.

### Changed — no bare syscalls left in `src/`

- **`main.cyr` exits through the stdlib's `sys_exit_group`.** That replaces the local `hoosh_exit_process` that 2.5.11
  added. The roadmap listed the wrapper as missing upstream, but cyrius has shipped it since 6.5.6, with an identical
  body. Verified live: a SIGTERM drains and exits all 12 threads with status 0.
- **`otlp.cyr`'s span timestamps use chrono's `clock_epoch_ns`.** That replaces a local `syscall(228, …)`. On Linux it
  is the same `CLOCK_REALTIME` read, and it also covers macOS, Windows and AGNOS.

### Performance

The bench binary is byte-identical to 2.9.1's apart from the embedded version string. No benched path changed, so
no A/B was run. The 2026-11-04 rows in `bench-history.csv` after 2.9.1's are this release's mandatory run.

## [2.9.1] — 2026-11-04

**Toolchain and dependency refresh to the latest releases.** cyrius 6.7.9 → 6.7.18, `ai-hwaccel` 2.4.2 → 2.5.6,
and the vendored `bote-core` 3.4.10 → 3.8.3 and `majra` 2.12.0 → 2.13.0. All four are their latest releases, and
each pins 6.7.18. Two of the inherited changes alter what clients get back. The hardware endpoints now return
corrected numbers, and a JSON request body with a duplicate key is refused. hoosh's own changes are four:
- moving off bayan's deprecated lookups;
- a one-byte fix to the off-loopback security warning that 6.7.18's linter caught;
- dropping a sigil call that is now a no-op;
- rewriting the crypto-threading comments for sigil's new lane pool.

**1205 assertions** (unchanged). fmt, lint, vet, deny, coverage, the security scan and the four fuzz targets are clean.
The build prints one warning, the large-static-data note, which drops from 455,072 to 136,288 bytes. sigil's static
crypto-buffer warning, present since 2.6.11, is gone.

### Changed — cyrius pin 6.7.9 → **6.7.18**

- **`lib/` re-vendored from an empty tree, and `cyrius.lock` regenerated.** 84 entries, one more than before:
  `lib/math.cyr`, which bayan 1.11.1 requires. There is one commit pin, ai-hwaccel 2.5.6 at `ccdcd9c`, matching the
  upstream tag. `deps --verify` reports 84/0. `cyrius deps` now notes that `lib/hashmap_fast.cyr` is outside the
  declared footprint. `lib sync` has copied that file since 6.6.6 and nothing includes it, so the lock keeps it, and
  CI's `lib sync` → `deps` order reproduces the lock byte for byte.
- **Bundled modules moved:** sigil 3.13.11 → 3.13.15, sandhi 1.10.9 → 1.13.7, bayan 1.5.13 → 1.11.1,
  sakshi 2.5.8 → 2.5.10, patra 1.16.0 → 1.16.6.
- **Binary:** 2,017,384 → 2,305,640 B under `CYRIUS_DCE=1`, and 3,714,664 B without. The growth is mostly the two
  bundles: bote-core grows from 4,074 to 9,170 lines and ai-hwaccel from 7,532 to 10,843.
- **No language rule changed in 6.7.10–6.7.18 for code shaped like hoosh's.** Those releases' notes are about structs,
  vectors, generics, multi-value returns and closures, and hoosh uses none of them. The build was clean, and it
  would have stopped on the "stops compiling" items.

### Changed — sigil crypto lanes (sigil 3.13.14, CYRIUS-2026-0040)

- **Any thread may now do crypto.** sigil no longer pins a "bank" to a thread for life (8 banks: main plus hoosh's 7
  workers). Each entry point that touches module-global crypto state now borrows one of 64 lanes for the operation
  and returns it.
- **The upstream bug this fixes.** Through 3.13.13 the 64th crypto-touching thread in a process wrapped the bank
  counter and shared a bank with another thread, which could accept forged ChaCha20-Poly1305 records and reject
  genuine ECDSA signatures. hoosh never reached 64 crypto threads.
- **`crypto_bank_set` is a no-op, and `_pool_worker` no longer calls it.** `crypto_tls_main_init()` stays: it
  installs main's thread-local block.
- **The pool's design is unchanged.** It still has 7 workers, and HTTPS work from helper threads (OTLP export, the
  catalog refresh) still goes to a worker. Those are now choices rather than requirements, and the ceiling is 64
  concurrent crypto operations, not 8 banks.
- **Comments and docs.** The comments in `pool.cyr`, `batch.cyr`, `health.cyr`, `catalog_fetch.cyr`, `otlp.cyr`,
  `trace.cyr`, `handlers.cyr` and `main.cyr` now describe lanes, and keep one line on how it worked through 2.9.0.
  [ADR 011](docs/decisions/011-multithreaded-accept-loop.md) gains a 2.9.1 update, and ADR 012 a pointer to it.
- **The "slot 0 = sigil's bank" comments were already wrong before this release.** `thread_local_alloc` hands sigil a
  slot from 16 up, and slots 0–15 are the fixed ones (hoosh uses 1–5), so nothing collided.

### Changed — `ai-hwaccel` 2.4.2 → **2.5.6** (`/v1/hardware/*` answers change)

All 45 ai-hwaccel functions hoosh calls keep their signatures. 17 of them changed body, mostly in 2.5.5's audit, which
fixed about 75 defects. What reaches hoosh's endpoints:

- **`/v1/hardware/cost` now reads every pricing row.** Before, the loader read only 1 of the 18: it matched the
  file's `{"instances": [...]}` wrapper as a single object, so every answer came from that one row.
  - Verified live: for a 7B BF16 model, 2.9.0 recommended **`p5.48xlarge` at $98.32/hr**. 2.9.1 lists the five
    cheapest that fit, starting with **`g4dn.xlarge` at $0.53/hr**.
  - `data/cloud_pricing.json` is re-synced from 2.5.6. It has 19 rows: the Azure A100 rows are corrected and an
    MI300X row is added.
  - `data/models.json` holds the same 26 models. It stays a top-level array; 2.5.x's `load_models` accepts that form
    or the wrapper.
- **Model sizes are parsed correctly.** "1.5B" used to plan as 15B, "350M" as 350B and "8x7B" as 87B. Verified live:
  `/v1/hardware/cost` for `"1.5b"` now needs 3.6 GB, where 2.9.0 said 36 GB.
- **Training estimates** (`/v1/hardware/training-estimate`) use mixed-precision AdamW accounting for full fine-tune,
  DPO, RLHF and distillation, about 16 B per parameter. A 7B full fine-tune goes from 52.2 to 117.4 GiB. LoRA, QLoRA
  and prefix are unchanged.
- **Placement and sharding plans** gain `"fits"` (false when nothing holds the model). Tensor parallel now requires
  one kind of device. Pipeline layers are split in proportion to device memory, and multi-device tokens/s now
  depends on model size; before, it read about 100× the single-device figure.
- **Detection** (`/v1/hardware`) fixes several wrong readings:
  - NVIDIA power was 100× too high.
  - AMD bandwidth used the DPM level index as the clock.
  - ROCm's compute capability was the PCI revision byte.
  - Any `/sys/class/accel` node was reported as a Google TPU v4.
  - Intel IPU6 cameras were reported as Graphcore IPUs.

  NUMA and PCIe links are now taken from the right GPU. JSON strings are escaped. A host with any of these devices
  will see a different device list. A tool that ran and failed is listed in `warnings`.
- **`registry_detect_with(mask)` runs no tools when the mask has no tool-based backend**, which matters when
  `[hardware] disabled_backends` leaves only sysfs backends. `profile_from_json_str` refuses a duplicate key and a
  leading-zero number.

### Changed — vendored `bote-core` 3.4.10 → **3.8.3** and `majra` 2.12.0 → **2.13.0**

- **bote: no wire change for hoosh's clients.**
  - A request that does not name the new `2026-07-28` revision in `params._meta` gets the same answer as before. bote
    fuzzed 200,000 legacy messages to byte-identical replies.
  - Verified live against a 2.9.1 gateway: `tools/list`, an echo call, an unknown tool (`-32602`), invalid JSON
    (`-32700`) and `initialize` answer exactly as on 2.9.0, apart from `serverInfo`'s version, now `3.8.3`.
  - All seven functions hoosh calls keep their signatures.
  - The core bundle no longer needs bayan (3.4.13); its `# Requires` block is `hashmap` and `sys`.
  - The rest of 3.4.11–3.8.3 is in parts hoosh does not use: the HTTP/Streamable/WebSocket transports, OAuth
    resource metadata and JWT, cancellation, progress, resource updates, and output and input schemas.
  - `scripts/sync-bote.sh` defaults to 3.8.3.
- **majra:** `pubsub_new`, `pubsub_publish` and `pubsub_total_published` are byte-identical.

### Changed — bayan 1.11.1: stricter JSON, typed lookups

- **A request body with a duplicate key or a malformed number is refused.** Malformed means a leading zero (`01`), no
  digits, or an out-of-range integer. Every handler that parses a body with bayan answers 400 `invalid JSON`, where
  it used to take the first value of a duplicate key. Verified live: `{"model":"7b","model":"70b",…}` on
  `/v1/hardware/cost` was answered as 7b by 2.9.0 and is refused by 2.9.1.
- **Provider model lists get the same treatment.** A duplicate key fails the parse, and that route keeps its last good
  catalog.
- **The bare lookups are deprecated.** 119 call sites move to the typed names:
  - **`handlers.cyr`, 37 calls:** `json_get(pairs, str_from("key"))` → `bayan_json_get_by_cstr(pairs, "key")`. The
    match is identical. Each lookup also stops allocating a `Str` header that hoosh's never-freeing allocator kept
    for the life of the process.
  - **`config.cyr`:** 63 `toml_get` and 19 `toml_get_sections` calls move to their `_by_cstr` names, which is where
    the deprecated names already forwarded.
  - **`tests/hoosh.tcyr`:** the same 14 calls.

### Fixed

- **The off-loopback security warning lost its newline** (`src/main.cyr`). `sys_write(2, "WARNING: bound to a
  non-loopback address with no [[auth]] tokens.\n", 65)` wrote 65 of the literal's 66 bytes, so the warning's second
  line ran onto the first. Now 66. 6.7.18's `cyrius lint` caught it, which made CI's lint gate fail. Verified live
  with `--bind 0.0.0.0`.

### Inherited without a hoosh change

- **Native TLS** (6.7.10 / 6.7.14 / 6.7.18):
  - Stricter with non-conformant peers: an unsolicited extension, an un-offered ALPN, or a byte after a signature
    fails the handshake.
  - The TLS 1.2 client keeps the server's intermediates, so a 1.2 server that sends one now verifies.
  - X.509 nameConstraints are enforced.
  - A peer's KeyUpdate storm is bounded.
- **sandhi 1.13.x:** HTTP/2 flow control and request deadlines, the TLS session cache under a mutex, HPACK limits,
  and an HTTP/1 header holding a NUL is dropped.
- **Not applicable:** `lib/http.cyr`'s new `http_get` deadlines (hoosh's client is its own, over sandhi);
  `args_init`'s negative-errno return (hoosh ignores the return).

### Performance

**No code regression, and the MCP path is faster.** The 2.9.0 (6.7.9) and 2.9.1 (6.7.18) bench binaries ran
alternately for 9 rounds each. The box was busy (load ~3.5), so only rows whose two ranges do not overlap are called:

| benchmark | 2.9.0 | 2.9.1 | Δ | cause |
|---|---:|---:|---:|---|
| `mcp_tools_list` | 4.400 µs | 3.674 µs | **−16.5%** | bote 3.4.11–3.8.3 |
| `mcp_tools_call` | 6.863 µs | 5.839 µs | **−14.9%** | bote 3.4.11–3.8.3 |
| `cost_record_local_free` | 140 ns | 113 ns | −19.3% | placement |
| `auth_verify_wrong_early` | 96 ns | 106 ns | +10.4% | placement |
| `cost_record_known_model` | 165 ns | 171 ns | +3.6% | placement |

The other 20 rows' ranges overlap at this load. Their medians are within ±5%, apart from five that read faster and
are too wide to call:
- `queue_enqueue_dequeue` −32% (2.9.0 range 265–644 ns);
- `pool_reserve_commit` −26% (31–47 ns);
- `cache_insert` −17% (96–107 against 86–121 ns);
- `cache_get_miss` −9.5% (62–63 against 56–77 ns);
- `batch_split_4` −5.2%.

**The placement rows are proven:**
- `ct_eq_bytes`, `ct_eq_bytes_lens`, `strlen`, `streq`, `memeq`, `_map_find` and `map_get` are byte-identical between
  the binaries.
- `_bench_cost_record`, `_bench_price` and `cache_get` differ only in `call` and `movabs` operands.
- What moved is their place in the 64-byte line. `ct_eq_bytes` went from offset 48 to 13, and `_bench_cost_record` /
  `_bench_price` from 51 / 24 to 32 / 5.
- `vec_get`'s cold path changed: its two error branches now call a shared abort helper. It shrinks from 257 to 225 B,
  and its hot path is unchanged.

The 2026-11-04 rows in `bench-history.csv` that follow 2.9.0's are this release's mandatory run.

## [2.9.0] — 2026-11-04

**Toolchain and dependency refresh onto the cyrius 6.7.x language minor.** cyrius 6.6.20 → 6.7.9. The vendored
bundles move to their last releases that pin 6.7.9: `bote-core` 3.3.17 → 3.4.10 and `majra` 2.9.3 → 2.12.0.
`ai-hwaccel` stays at 2.4.2. It has no 6.7.9 release (2.4.2 pins 6.6.20, and 2.5.0 jumps to 6.7.18), and 2.4.2's
bundle compiles clean on 6.7.9. No hoosh source changed. The minor bump is for what `POST /v1/tools/call` now
answers (bote 3.4.5, below). **1205 assertions** (unchanged). fmt, lint, vet, deny, coverage, the security scan and
the four fuzz targets are clean. The build prints only the two upstream warnings 2.8.2 had. The binary is
2,017,384 B (+35,680 B, mostly bote-core's growth from 2,911 to 4,074 lines).

### Changed — cyrius pin 6.6.20 → **6.7.9**

- **`lib/` was re-vendored from an empty tree and `cyrius.lock` regenerated.** It holds the same 83 entries as at
  6.6.20 (the snapshot's file list did not change), with one commit pin (ai-hwaccel 2.4.2). `deps --verify` 83/0.
- **6.7.x is a language minor.** hoosh's sources, tests, fuzz targets and all three bundles were checked against
  each change that can alter a program that already compiled. None applies:
  - **`!` is a real operator since 6.7.3.** Before that, the lexer silently dropped a lone `!`, so `if (!ok)`
    compiled as `if (ok)`. A scan that skips comments and strings finds no lone `!` anywhere in the compile set.
  - **New reserved words**: `trait` (6.7.0), `const` (6.7.2), `true` / `false` (6.7.3), `do` (6.7.5), `dyn`
    (6.7.8). None is used as an identifier.
  - **Unary `-` / `~` on a constant binds tighter than a following literal operator** (6.7.9: `-K + 10` is now 5,
    not 15). Every unary use in the compile set is parenthesised (`~(1 << b)`) or is binary subtraction
    (`0 - 1 - …`).
  - **Typed-pointer, `self`, struct-copy, `f32`, `u128` and slice semantics** (6.7.0–6.7.9). hoosh declares no
    struct, `impl`, narrow typed pointer, `f32` or slice; it addresses memory with `load64` / `store64`.
  - **Stricter checks that can only refuse, never change an answer.** These would have stopped the build, and it is
    clean: arity checks on forward calls, a stray `}` at module scope, a `for` step assigning an undeclared name.
- **Codegen for hoosh's hot paths is byte-identical.** Comparing the two bench binaries function by function (symbol
  maps from `CYRIUS_SYMS`), `ct_eq_bytes`, `ct_eq_bytes_lens`, `strlen`, `streq`, `memeq` and `_map_find` match
  byte for byte. `_bench_cost_record`, `_bench_price`, `vec_get` and `cache_get` differ only in the operands of a
  `call` rel32 or a `movabs` address.
- Bundled modules re-folded at 6.7.x (W2). `lib/vec.cyr` and `lib/hashmap.cyr` changed only in comments.

### Changed — vendored `bote-core` 3.3.17 → **3.4.10**

hoosh calls seven bote-core functions. Every signature is unchanged, and `dispatcher_new`, `tool_registry_new`,
`tool_def_new` and `schema_new` are byte-identical. What changed for hoosh:

- **`POST /v1/tools/call` answers as the MCP and JSON-RPC 2.0 specs require** (bote 3.4.5; `codec_process_message`
  now validates the JSON before dispatching). A client should update:
  - An unknown tool is error **`-32602`** (was `-32601`, which now means only that the *method* does not exist).
  - Arguments that fail the tool's `inputSchema` return a **result** with `isError: true`, whose text names the
    argument (was error `-32602`). A handler that returns nothing also gives an `isError` result (was
    `"result":null`).
  - Invalid JSON is error **`-32700`** with id `null`. Before, it was `-32600`, dispatched anyway, or dropped.
  - An id that is an object, array or boolean, and `params` that is a string or number, are `-32600`.
  - A `cursor` on a list method that bote never issued is `-32602`.
- **Every string bote writes into JSON is fully escaped** (bote 3.4.6). `_json_emit_escaped` now forwards to the new
  `bote_json_escape`. It escapes every control byte below 0x20 (`\b`, `\f`, or `\u00XX`); before, it escaped only
  `"`, `\`, LF, CR and TAB and copied the rest raw. hoosh calls `_json_emit_escaped` for its own JSON too: the
  event-bus payloads (`events.cyr`: model, error text, base URL), the OTLP attributes (`otlp.cyr`) and the
  `infer` request body (`main.cyr`). A provider error or model name carrying a control byte (an ANSI colour code, a
  BEL) used to produce invalid JSON there and now produces valid JSON.
- **Registration checks the tool definition** (bote 3.4.8, `registry_check_tool_def`): name syntax, an `object`
  `inputSchema`, property and `required` entries. The schema is compiled at registration. hoosh's `bote_echo`
  passes (bote's own consumer audit lists hoosh as unaffected). `mcp_init` ignores the return value, so a future
  tool that fails the check would be missing from `tools/list`. The suite's `registry_len == 1` assertion pins
  that today.
- **`tools/call` reads the clock only when something consumes the duration** (bote 3.4.1). `mcp_tools_call` is
  **−27.8%** (below).
- **bote logs through sakshi** (bote 3.4.0, `log.cyr`). Lines go wherever hoosh's sakshi output goes, at hoosh's level
  (`HOOSH_LOG` / `[logging] level`). Refusals are WARN, one line per request is DEBUG, and no secret or tool
  argument is logged. A copied bundle must list `sakshi` in its own `[deps].stdlib`, which hoosh already does. At
  the default level a live run prints nothing new.
- `initialize` reports `serverInfo` version `3.4.10`. `scripts/sync-bote.sh` defaults to 3.4.10.
- **Verified live** against a 2.9.0 gateway: `tools/list` lists `bote_echo`, an echo call returns valid JSON, an
  unknown tool gives `-32602`, a truncated body gives `-32700`, `ping` answers `{}`, and `initialize` reports
  3.4.10.

### Changed — vendored `majra` 2.9.3 → **2.12.0**

`pubsub_new` and `pubsub_total_published` are byte-identical, and `pubsub_publish` changed only in a comment. That
comment records the invariant its unlocked pattern walk relies on under 6.6.20's in-place hashmap rebuild: pubsub
never deletes a key. The 2.10.1–2.12.0 fixes are in parts hoosh does not use: relay refcounts, the sliding-window
limiter, signed envelopes, workflows, barriers, the admin endpoint, RESP and PostgreSQL.

### Performance

**No code regression, and one real win.** The 2.8.2 (6.6.20) and 2.9.0 (6.7.9) bench binaries ran alternately for 7
rounds each (load ~1.1). Medians:

| benchmark | 2.8.2 | 2.9.0 | Δ | cause |
|---|---:|---:|---:|---|
| `mcp_tools_call` | 8.644 µs | 6.241 µs | **−27.8%** | bote 3.4.1: no clock read per call |
| `mcp_tools_list` | 4.058 µs | 3.875 µs | −4.5% | bote 3.4.x |
| `cache_get_hit` | 147 ns | 124 ns | −15.6% | placement (back to its 2.8.1 value) |
| `route_select_20_providers` | 843 ns | 780 ns | −7.5% | placement |
| `route_matches_model` | 36 ns | 33 ns | −8.3% | placement |
| `auth_verify_token` | 86 ns | 104 ns | +20.9% | placement (its 2.8.1 value) |
| `cost_record_local_free` | 111 ns | 135 ns | +21.6% | placement |
| `cache_get_miss` | 56 ns | 62 ns | +10.7% | placement |
| `batch_split_4` | 766 ns | 807 ns | +5.4% | placement |

The other 16 rows are within ±5.1%, or ±1 ns on the rows a few ns long. **The placement rows are proven,
not assumed.** `auth_verify_token` is `ct_eq_bytes_lens` → `ct_eq_bytes` over two `strlen`s. All three are
byte-identical between the binaries. Only `ct_eq_bytes`'s start moved, from offset 50 to 48 in its 64-byte
line. The row has flipped with it, 104 → 86 → 104 ns across 2.8.1 → 2.8.2 → 2.9.0. `cost_record_local_free`'s
two functions keep their sizes and code, and move from offsets 26 / 63 to 51 / 24. `_bench_batch_split` and
`route_matches_model` differ only in `call` displacements, and both start at offset 52 instead of 27. The
2026-11-04 rows in `bench-history.csv` that follow 2.8.2's are this release's mandatory run.

## [2.8.2] — 2026-11-04

**Toolchain and dependency refresh.** cyrius 6.6.6 → 6.6.20. Each dependency moves to its last release that pins
6.6.20: `ai-hwaccel` 2.4.0 → 2.4.2, vendored `bote-core` 3.3.13 → 3.3.17 and `majra` 2.9.1 → 2.9.3. Every newer
release of the three pins cyrius 6.7.x. No gateway behavior changed. Two source edits clear warnings that 6.6.20
added (below). **1205 assertions** (unchanged). fmt, lint, vet and deny are clean, and so are coverage, the security
scan and the four fuzz targets. **The release binary shrinks from 2,931,928 to 1,981,704 B (−32%)**, because
`CYRIUS_DCE=1` now removes dead code instead of NOP-filling it.

### Changed — cyrius pin 6.6.6 → **6.6.20**

- **`lib/` was re-vendored from an empty tree** (`lib sync`, then `deps`). The snapshot gains two leaves that hoosh's
  compile set pulls in: `poison` (6.6.18's `freelist` includes it) and `tls_hostid`. 83 entries are locked, and
  `deps --verify` reports 83/0.
- **Bundled modules moved:** sandhi 1.9.17 → 1.10.8, sigil 3.12.18 → 3.13.10, patra 1.14.3 → 1.15.2,
  bayan 1.5.6 → 1.5.12, sakshi → 2.5.7.
- **The `duplicate fn 'uname_release'` warning is gone.** sigil 3.13.x now takes `uname_release` from `lib/sys.cyr`
  instead of carrying its own copy. `sys` stays ahead of `sigil` in `[deps].stdlib`.
- **The build now prints only two warnings, both upstream and both seen on 6.6.6:** sigil's per-bank static crypto
  buffer (now `lib/sigil.cyr:28046`) and the large-static-data note.
- **DCE now compacts.** On 6.6.6 a `CYRIUS_DCE=1` build was the same size as a plain one (2,931,928 B), and its
  ~1.05 MB of dead code was NOP-filled in place. At 6.6.20 the text segment drops from 2,401,856 to 1,448,176 B. The
  data segment is unchanged. A plain build is 3,099,912 B (+5.7%, from compiler and stdlib growth). This matches
  6.6.20's HEAP-08 fix: the whole-program compaction registry was capped at 49,152 sources, and a build past the
  cap silently gave up on compaction. Now the registry grows instead.
- **The response cache benefits from 6.6.20's hashmap fix.** `lib/hashmap.cyr` now counts the tombstones
  `map_delete` leaves and rebuilds in place. Before, a map kept at a steady size by delete + insert churn slowly
  turned every miss into a scan of the whole table. That is the shape of the response cache at capacity (LRU
  `map_delete`, then `map_set`). Nothing in hoosh keeps a slot pointer across a `map_set`: cache values are separate
  allocations, and eviction walks a `map_keys` snapshot.
- **Not applicable to hoosh:** sandhi 1.10.7's 60-second whole-request budget is applied by `sandhi_server_run`,
  and hoosh runs its own accept loop. hoosh already ignores SIGPIPE at startup (`_sandhi_server_ignore_sigpipe`,
  2.4.13). 6.6.16's `MSG_NOSIGNAL` socket writes add a second guard for the stdlib's own socket writes.

### Changed — `ai-hwaccel` 2.4.0 → **2.4.2**

The bundle's body is byte-identical to 2.4.0's, apart from three `bayan_json_v_obj_get` →
`bayan_json_v_obj_get_by_cstr` renames inside `profile_from_json_str`. It also gains a compile-verified
`# Requires` block. Every leaf that block names is already in hoosh's `[deps].stdlib`, so the include-once
lines change nothing here. `data/cloud_pricing.json` is byte-identical at 2.4.2. `data/models.json` holds the same
26 models (hoosh keeps the top-level-array form). Nothing needed re-syncing. **`cyrius.lock` was regenerated
from empty.** After the tag bump, 6.6.20 kept the 2.4.0 commit pin beside the 2.4.2 one. It fails closed on purpose,
and deleting the lock is the documented way to re-pin. The lock now carries one commit pin, ai-hwaccel 2.4.2 at
`4d689d5`.

### Changed — vendored `bote-core` 3.3.13 → **3.3.17** and `majra` 2.9.1 → **2.9.3**

Both are toolchain releases. Their code is unchanged:

- **bote-core** changes only its version header, the `_bote_server_version()` literal and a new `# Requires`
  block (`hashmap bayan sigil ct`). So `initialize` over `POST /v1/tools/call` now reports `serverInfo` version
  `3.3.17`. `scripts/sync-bote.sh` defaults to 3.3.17.
- **majra** changes comments and adds its own `# Requires` block. `pubsub_new`, `pubsub_publish` and
  `pubsub_total_published` are unchanged.
- Both bundles were fetched from their tags and are byte-identical to the local tag checkouts.

### Fixed — warnings 6.6.20 added

- **24 calls to the deprecated `bayan_json_v_obj_get` in `src/lib/catalog.cyr`** now call
  `bayan_json_v_obj_get_by_cstr`. bayan 1.5.10 deprecated the bare name because it does not say whether the key is
  a C string or a `Str`. Every call site passes a C-string literal, either directly or through `_cat_get_cstr`,
  `_cat_get_int` and `_cat_supported`, whose callers all pass literals. The deprecated name already forwarded to
  `_by_cstr`, so no lookup changes. One line was split to stay under the 120-byte limit.
- **`ERR_TIMEOUT` is renamed `ERR_TIMED_OUT`** in `src/lib/types.cyr` and the test mirror. 6.6.20 warns
  `duplicate symbol 'ERR_TIMEOUT' redefined with conflicting value (last definition wins)`. sakshi's `ErrCode` has
  defined `ERR_TIMEOUT = 5` all along, and hoosh's `HooshErr` defined it as 6. 6.6.6 compiled the same collision
  silently. Nothing in the stdlib reads sakshi's constant, so hoosh's value already won wherever it was used and
  behavior is unchanged. The rename removes the hazard. The only `HooshErr` name still shared is `ERR_NONE`, with
  majra's alias, and both are 0.

### Performance

**No code regression.** The 6.6.6 (2.8.1 tree) and 6.6.20 (this tree) bench binaries ran alternately for 7 rounds
each on one box (load ~1.2). Medians:

| benchmark | 6.6.6 | 6.6.20 | Δ |
|---|---:|---:|---:|
| `auth_verify_token` | 104 ns | 86 ns | −17.3% |
| `cost_record_known_model` | 170 ns | 155 ns | −8.8% |
| `audit_record_sign` | 3.463 µs | 3.242 µs | −6.4% |
| `auth_verify_wrong_late` / `_early` | 103 / 102 ns | 98 / 97 ns | −4.9% |
| `mcp_tools_list` | 4.237 µs | 4.079 µs | −3.7% |
| `route_select_20_providers` | 798 ns | 828 ns | +3.8% |
| `cache_get_hit` | 124 ns | 144 ns | **+16.1%** |

The other 17 rows are within ±3.3%, or ±1 ns on the rows a few ns long.

**`cache_get_hit` moved because of code placement, not codegen.** Every function on its path compiles to the same
size under both toolchains: `cache_get` 578 B, `map_get` 136 B, `_map_lookup` 175 B, `_map_find` 589 B, `hash_str`
259 B, `_hm_mix` 197 B, `_map_key_eq` 162 B, and `streq`, `strlen` and `memeq`. Their source is identical too. What
changed is where they sit in the 64-byte line (symbol maps from `CYRIUS_SYMS`). `_map_find` moved from offset 34 to
offset 5, `cache_get` from 59 to 37, and `map_get` from 0 to 62, so `map_get` now straddles a cache line.
`cache_get_miss` runs the same `map_get` without the key compare and is flat (+1.8%). This row has swung with
layout before: 114 → 140 → 114 ns across three consecutive builds on 2026-08-30 in `bench-history.csv`. The
2026-11-04 row in that file is this release's mandatory `bench-history.sh` run.

## [2.8.1] — 2026-10-09

**A request's `max_tokens` is trimmed to the serving model's own output ceiling.** Providers refuse anything over
it with a 400; they don't trim it. Measured 2026-10-09 against the live Anthropic API:
- Haiku 4.5 answers `max_tokens: 65536 > 64000, which is the maximum allowed number of output tokens`.
- Every current Claude model refuses 131072 (their ceiling is 128,000).

2.8.0's catalog already held each model's ceiling, which the Models API lists and `/v1/models/catalog` publishes as
`max_output_tokens`. But hoosh clamped only to a route's operator-set `max_tokens_limit`. So a client asking for
the most a newer model takes failed on every request to Haiku, Sonnet and Opus 4.5. thoth's `reasoning = "max"`
(65,536) is one such client, and so is its own `[hoosh].max_tokens` maximum (131,072). Found by thoth's repair
batch 12 audit. **1205 assertions** (was 1192).

### Fixed

- **The model's ceiling clamps the request** (`catalog_out_cap`, `catalog_clamp_out`).
  - The ceiling is the operator's `[[models]]` `max_output_tokens` when a block sets one (the operator's word wins,
    as everywhere in the catalog). Otherwise it is what the serving route's provider listed. A model with no known
    ceiling is sent what was asked, as before.
  - `_chat_prep` trims the client's value after the route clamp, so the token-budget reservation counts what is
    actually sent. It logs `chat: max_tokens clamped to the model's output ceiling`.
  - The Anthropic builder applies the same ceiling to its own default (16,384 with thinking, 4,096 without), and
    plans a thinking budget against the trimmed figure. A Claude 4.5 model at effort `max` thinks on 16,384 inside
    a 64,000 ceiling. A `[[models]]` ceiling of 8,192 plans 7,168.
- Verified live through a 2.8.1 gateway: the requests that had returned 400 now return 200.
  - Haiku 4.5 at 65,536, buffered, and its alias at effort `max`, streamed.
  - Sonnet 4.5 at `xhigh` and 65,536, streamed.
  - Opus 5.5 at 131,072.
  - The four were trimmed, and a request under the ceiling was left alone.
- Tests: `_t281_catalog_out_cap` (13 assertions). Replacing the clamp with a pass-through fails three of them.

## [2.8.0] — 2026-10-09

**The model catalog is asked of the providers, and requests take the shape each current model accepts.** Through
2.7.1, `GET /v1/models/catalog` (thoth's model picker) listed a table compiled in mid-2025. Against the live
Anthropic API on 2026-10-09, none of the six Claude ids it offered could be served, and no newer model could
appear without a new hoosh release. Pricing billed Opus 4.5–4.8 at three times their rate, and the Anthropic
request builder sent a thinking shape that Haiku 4.5 and Sonnet 4.6 reject. Streams always ended with
`finish_reason: "stop"` and never sent the usage chunk clients ask for. [ADR 012](docs/decisions/012-live-model-catalog.md)
records the design. Also fixed: since 2.5.5 the health prober had been checking every remote route at
localhost:80. **1192 assertions** (was 959).

### Added — the live model catalog

- **Asked of each route's own provider, with its own key.** Each provider's list endpoint is used:
  - **Anthropic** `/v1/models`: each model's context window, output ceiling and capability tree (which thinking
    modes and which effort levels it accepts).
  - **Gemini** `/v1beta/models`: generateContent models only.
  - **OpenAI-compatible** `/v1/models`: embeddings, speech, image, moderation and realtime models are dropped.
    OpenRouter and Mistral answers also carry context lengths, vision flags and per-model prices.
  - **Ollama** `/api/tags`.
- **When it refreshes.** At startup (enqueued, so startup never waits on a provider), after `/v1/admin/reload`,
  on `POST /v1/models/refresh` (new; inline), and every `[catalog] refresh_secs` (new; six hours by default).
- **Where it runs.** A refresh is a pool job, because remote lists are HTTPS and only the workers own a crypto
  bank. It is single-flight.
- **How it is published.** A fresh snapshot goes out with one pointer store. A route whose fetch fails keeps its
  last good list.
- **Memory.** The fetch and the parse use one reusable arena, reset after every route, so only the kept entries
  reach the never-freeing heap.
- **Per route.** `[[providers]] catalog = "live" | "static"` (new). Live is the default everywhere except
  OpenRouter (a marketplace of several hundred models) and Whisper. Measured: 14 Claude models listed live.
- **Richer catalog entries.** `source`, `routes` (every enabled route that would serve the id, in router order),
  `display_name`, `context_window`, `max_output_tokens`, `capabilities` (vision, thinking modes, effort levels) and
  `pricing` (micro-USD per 1K tokens, only when known). The 2.4.9 fields come first and are unchanged.
  `/v1/models` lists live ids too.
- **Aliases find snapshots.** An alias finds its dated entry: Anthropic lists `claude-haiku-4-5-20251001`, and
  clients send `claude-haiku-4-5`.
- **`[[models]]` operator blocks (new).** They set a model's price (`input_per_mtok` / `output_per_mtok` in USD
  per million, both or neither), `context`, `max_output`, `tier`, `vision`, `reasoning` and `listed`. These win
  over anything a provider or the table says, and are layered over the provider's entry rather than replacing
  it. This is how to price a model released after this hoosh.
- **`/v1/health/providers` reports each route's last catalog answer.** It carries `catalog`, `catalog_status`,
  `catalog_models` and `catalog_checked_ms`. A 401/403 there is the first place a revoked or mistyped key shows:
  the remote health probe is a TCP connect and cannot see a key. Measured: a bad OpenAI key reads
  `catalog_status: 401`.

### Fixed — requests for current models

- **The thinking shape comes from the model's capability tree.** Models that take it get adaptive thinking plus
  effort. Models that take only a budget get `{"type":"enabled","budget_tokens":N}`, inside `max_tokens` with
  1024 left for the answer. The effort level is clamped to the ones the model accepts.
  - Measured on 2026-10-09: Haiku 4.5 rejected the 2.7.1 shape with HTTP 400 ("adaptive thinking is not supported
    on this model"), and Sonnet 4.6 rejected effort `xhigh` ("Supported levels: high, low, max, medium").
  - Both now answer, and a model the catalog has not listed gets what the client asked for, as before.
- **Reasoning text is requested.** Since Opus 4.7, thinking blocks arrive with empty text unless the request sets
  `display: "summarized"`, which hoosh now sends whenever thinking is on. It costs the same, and a client that
  folds `reasoning_content` now has something to fold. Measured: Haiku 4.5 streamed 218 characters of reasoning.
- **`reasoning_effort` knows `xhigh` and `max`** (Claude 4.7 and later) and `minimal` / `none` (OpenAI). 2.7.1
  read any of them as "no thinking".
- **`reasoning_effort` is forwarded on OpenAI-compatible routes**, verbatim, to models whose providers document it:
  OpenAI's o-series, GPT-5 and GPT-6; xAI's Grok 4.x and Grok Build; DeepSeek; gpt-oss on Groq. 2.7.1 never
  forwarded it, so a client's reasoning setting did nothing on o3 or GPT-5. GPT-4o and GPT-4.1 400 on the field,
  so they never get it. `[[models]] reasoning = true|false` overrides.

### Fixed — pricing and metadata

- **Anthropic rows are the current lineup, from the Models API and Anthropic's published rates.** One row per
  family, with Haiku 5.5's long-context tier ($0.10/$0.50, then $0.50/$2.50 past 100K input). Before, the single
  `claude-opus-4` row ($15/$75) priced every Opus 4.x, and Claude 5.x models fell to the $3/$15 provider default.
- **Retired and invalid ids are gone from the table.** `claude-3.5-haiku` (never an id) and the Claude 3.x rows
  are removed. Rows now carry a `listed` bit, so the static catalog never offers a family prefix
  (`claude-opus-4`) as an id.
- **Other providers' rows are refreshed from their official pricing pages, read 2026-10-09.** That covers OpenAI
  GPT-6 / GPT-5.x / GPT-4.x / o-series, Gemini 3.x and 2.5, Grok 4.x, DeepSeek V4, Mistral Large 4 / Medium 3.5 /
  Small 4 / Codestral, and Groq's gpt-oss and Qwen. The published long-context tiers are included: OpenAI past
  272K, Gemini Pro past 200K, xAI at 200K. Shut-down or retiring ids stay for lookups but are no longer offered:
  `gemini-2.0-flash`, `deepseek-chat` / `deepseek-reasoner`, Groq's Llama models, and o1 / o3-mini / o4-mini
  (shutting down 2026-10-23).
- **Context windows come from the provider when it reports one.** Claude 4.6 and later are 1M; the 2.7.1 table had
  200K for every Claude.
- **Gemini thinking tokens are billed as output** (`thoughtsTokenCount`, on both paths). They were not counted.
- **The shipped `hoosh.cyml` OpenAI patterns** were `o1-*` / `o3-*`, which missed the bare `o1`, `o3` and `o4-mini`
  ids. They are now `o1*` / `o3*` / `o4*`.

### Fixed — streams

- **`finish_reason` is the provider's own, mapped** (new `src/lib/outcome.cyr`). `length` when a reply hits
  `max_tokens` or the context window; `tool_calls` whenever calls are carried; `content_filter` for a classifier
  refusal (Claude 5.x answers HTTP 200 with `stop_reason: "refusal"`) or a Gemini safety stop. 2.7.1's streams
  always said `stop`, and its buffered replies said `stop` or `tool_calls`. Measured: `max_tokens: 12` now ends
  `length` on both paths.
- **The usage chunk.** When the client sets `stream_options.include_usage`, a stream ends with OpenAI's usage chunk
  (`choices: []`, then `usage` with hoosh's `cost_micro_usd` and `provider`), built from the provider's own counts:
  - Anthropic: `message_start` / `message_delta`.
  - OpenAI and DeepSeek: hoosh now asks them for usage.
  - Gemini: `usageMetadata`.
  - Ollama: the final line.

  thoth always asked for this chunk and never got one. Measured: Haiku 4.5, 53 + 184 tokens, 973 µ$, which is
  exactly $1/$5 per million.
- **Streams are metered by those counts** (pool commit and cost record), falling back to the admission estimate
  only when the provider reported none. A local stream that never reached its backend is no longer billed. 2.7.1
  billed every stream at the estimate and called `cost_record` without `_chat_lock`.
- **Orphan tool-argument fragments are dropped.** An Anthropic `input_json_delta` for a content block whose
  `tool_use` start was never forwarded is dropped and counted instead of sent without a header. The root cause of
  the headerless calls thoth reported at 0.44.2 was sandhi's SSE parser losing an event at a TCP read boundary.
  That is fixed in sandhi 1.9.15, part of the cyrius 6.6.6 stdlib hoosh pins, and a new test feeds a `tool_use`
  start through `sandhi_sse_parse` split at every byte.
- **`/v1/health/providers` JSON-escapes `base_url`.** A quote in an operator's url made the whole body
  unparseable, so a client read every route's health as unavailable.

### Fixed — the remote health probe connected to localhost:80

`url_host` and `url_port` assumed an `http://` scheme, because the raw-socket local path they were written for is
plaintext. The background prober (2.5.5) reused them for remote routes. For `https://api.anthropic.com` the host
came out as `localhost` and the port as 80, so every remote route's liveness was a TCP connect to localhost:80.

- With nothing listening there (measured 2026-10-09), every remote route went `unhealthy` after three sweeps while
  its requests succeeded. `router_select` fell back to "all candidates unhealthy" on every request, and a client
  painted every cloud model red.
- With something listening on :80, every remote route read `healthy` whatever its real state.

Both functions now read the scheme (`https://` defaults to 443) and live in the new `src/lib/urlparse.cyr`, which
the tests include. Measured after the fix: both remote routes healthy, zero failures over two sweeps.

### Known limitations (named, not fixed here)

- **gpt-6-astra and gpt-6.1-sol take no function calling on Chat Completions.** OpenAI serves tools for them
  through the Responses API only. hoosh speaks Chat Completions to OpenAI, so a tool-using client cannot drive
  them yet. The metadata rows say `tools = 0`.
- **Cache-token pricing is not modelled.** Anthropic's `cache_*` input counts and the providers' cached-input
  rates are absent, so a cached prompt is costed at the full input rate.
- **Gemini and Ollama thinking are not mapped** from `reasoning_effort` (`thinkingConfig`, Ollama's `think`).

### Tests

- **Fourteen new groups, 233 assertions, against the real modules.** That covers the catalog parsers for all four
  families, units, the thinking plan, request options, the snapshot, entry JSON, metadata and pricing, OpenAI
  extras, finish mapping, stream usage, the tool guard, the SSE split boundary, and scheme-aware url parsing. The fixtures are trimmed from
  the live Anthropic answer.
- **`pricing.cyr` and `metadata.cyr` are now included by the test binary instead of mirrored**, closing the
  roadmap's mirror-drift problem for them. 2.5.0's `_req_reasoning_effort` mirror is gone too.
- **Break-tested.** Removing the effort clamp fails three assertions, and removing the Opus 4.8 price row fails
  two.

### Gates

fmt, lint, vet and deny are clean, symbol coverage is 43% (294 of 677 functions), the security scan and all four
fuzz targets pass, and the benchmark gate ran (25 benchmarks recorded). The suite was also run live against the Anthropic API from a scratch config: the catalog,
the refresh, health with a deliberately bad key, the thinking shapes on Haiku 4.5, Sonnet 4.6 and Opus 4.8, the
usage chunk, and `length`.

## [2.7.1] — 2026-09-25

**The last rust-old parity item, and three fixes found while filing the sandhi issue.** 2.7.0 listed
per-provider TLS pinning and mTLS as a non-port blocked on sandhi. That came from a stale roadmap entry,
not from sandhi. **959 assertions** (was 943).

### Added — per-provider TLS pinning and mTLS

rust-old's `tls_pinned_certs`, `client_cert` and `client_key` provider keys are ported. sandhi 1.4.6
threads a TLS policy through `sandhi_http_*` and `sandhi_http_stream`, 1.6.0 enforces pinning, trust
stores and mTLS natively, and hoosh's bundled sandhi 1.9.17 has all of it.

- **The keys.** `tls_pinned_certs` is the PEM bundle trusted as the provider's only roots, as rust-old's
  `tls_certs_only` did; in sandhi it is a trust store. `client_cert` and `client_key` present a client
  identity. The policy lives on the route (`ROUTE_TLS_POLICY`) and goes to every remote call and stream.
- **One bundle.** sandhi keeps a single trust bundle, so several paths refuse the config rather than
  silently dropping all but one.
- **Fail closed.** A missing file, or half an mTLS pair, also refuses the config. rust-old logged the
  problem and carried on with the system roots.
- **Scope.** On a plain-http route the keys are ignored, with a warning.

### Fixed — remote streams had no timeouts

The remote streaming path called `sandhi_http_stream` with no options, so it had no connect deadline and
no read timeout. A provider that accepted the connection and then went silent held a worker forever, and
seven such streams hung the gateway. 2.6.6 fixed this for `sandhi_http_post` and missed the stream.
Streams now get a 10 s connect deadline and a 300 s idle bound. They get no total deadline, so a long
generation is not cut off.

### Fixed — the semantic cache never matched through Ollama

The semantic cache embeds each query. For an Ollama route it posted the OpenAI-shaped body to the legacy
`/api/embeddings`, which reads `prompt`, not `input`. The answer held no vector, so an Ollama embedding
model never produced a semantic hit. It now uses `/api/embed` and the `/v1/embeddings` normalizer from
2.7.0.

### Filed upstream — szál tools

szál's Cyrius port (2.1.2) has its 54 MCP tools, but hoosh cannot vendor them yet. There is no dist
bundle (szál's M5 item). szál also avoids collisions by renaming its *vendored* majra and bote-core,
which leaves its own `STEP_COMPLETED` / `STEP_FAILED` / `STEP_SKIPPED` (different values from
majra's), `step_result_new` and `cache_new` (different arities), and `uuid_generate`,
`compiled_compile` and `TRIGGER_*` colliding with the upstream copies hoosh ships. Filed as
`szal/docs/development/issues/2026-09-25-hoosh-consumer-bundle.md`, with the measured collision table
and a bundle and registration shape hoosh can use.

### Filed upstream — remote SSE keep-alive

Remote streams still cannot send keep-alives: `sandhi_http_stream` gives the caller no turn while the
upstream is silent. Filed as `sandhi/docs/development/issues/2026-09-25-http-stream-no-idle-hook.md`.
The issue checks sandhi's full stream surface first, proposes `sandhi_http_options_idle_ms` / `_idle_cb`,
flags partial TLS records across a timed-out read, and specifies a gate.

### Tests

Two groups, 16 assertions:

- `rustold_provider_tls`: the TLS config decisions and the remote stream options.
- `rustold_semantic_ollama`: the semantic parser reading a normalized Ollama answer, and the empty legacy
  answer it used to get.

### Docs

- The roadmap's sandhi items are corrected. Pinning and mTLS are done. Connection pooling is no longer
  upstream-gated, since sandhi has `sandhi_http_options_pool`. Remote keep-alive now points at the new
  issue.
- ADR 004 and doc-health note the TLS implementation, and `hoosh.cyml` documents the three keys.
- `rust-old-retirement.md` moves TLS pinning to "ported". Deletion now follows 2.7.1, the last tag that
  contains `rust-old/`.

### Gates

fmt, lint, vet and deny are clean, and symbol coverage is 42%. Every CI step, the benchmark gate
included, was green when replayed against a clean export of the tracked tree. No benchmarked path
changed, so `bench-history.csv` was not re-recorded. Binary: 2,862,408 → 2,862,824 bytes.

## [2.7.0] — 2026-09-25

**Closes rust-old parity so the Rust tree can be deleted.** A second pass over `rust-old/` checked the
route table, every config key, the CLI, request validation, the response shapes rust-old's OpenAI
conformance tests assert, the model catalog entry by entry, metric names, embeddings, tools and the
hardware endpoints. The July review's items were already fixed. What remained is below. The deliberate
non-ports and the deletion checklist are in
[docs/development/rust-old-retirement.md](docs/development/rust-old-retirement.md), so nothing needs the
Rust tree after it goes. **943 assertions** (was 877).

### Fixed — `/v1/embeddings`

- **Routed by model.** Every request went to the first local route, whatever model it named. It now
  takes the route that matches the model, as chat does, and a model no route serves gets rust-old's 404
  `No provider configured for model '<model>'`.
- **Ollama.** hoosh posted the OpenAI body to Ollama's legacy `/api/embeddings`, which reads `prompt`,
  not `input`, and answers `{"embedding":[…]}`. It now calls `/api/embed` with `{model, input}`. That
  endpoint takes a string or an array, and its answer is normalized to OpenAI's
  `{"object":"list","data":[{"object":"embedding","index":i,"embedding":[…]}],"usage":…}`. rust-old
  joined an array into one string; hoosh returns one embedding per item.
- **Remote providers.** OpenAI-compatible remote routes (OpenAI, Mistral, Groq and others) now serve
  embeddings with their auth over TLS. Anthropic and Google, which have no `/v1/embeddings`, answer
  400. The DLP policy matches chat: RESTRICTED is refused, and CONFIDENTIAL goes to a local route or is
  refused.

### Fixed — `/v1/models` listed provider names as model ids

The endpoint returned one entry per route, with ids like `"ollama"`. An OpenAI client's model picker
offered those, and none of them could be routed. rust-old asked each provider for its models. hoosh
now asks local backends live (Ollama's `/api/tags`, everyone else's `/v1/models`). Remote routes, and
a local one that does not answer, list the catalog models and literal patterns they match, so the
endpoint makes no remote call. Only ids a route matches are listed, each once, with `owned_by` set to
the provider. Anyone who used this endpoint to enumerate providers can read the distinct `owned_by`
values.

### Fixed — every completion had the same id and no `created`

Every chat completion was `"id":"chatcmpl-hoosh"`, and none carried `created`. rust-old sent
`chatcmpl-<uuid>` and a unix timestamp, and its conformance tests required both. Each completion now
gets `chatcmpl-` plus 96 random bits, and a `created` time. Every chunk of a stream, including the
upstream-error frame, carries the same pair. A cache hit replays its stored envelope with a fresh head,
so it no longer repeats an earlier response's id. The id buffer is allocated once per thread and
reused, so a request allocates nothing for it.

### Fixed — smaller rust-old gaps

- **`GET /v1/health`** adds `version` and `providers_configured`. The 503 when degraded stays.
- **Model catalog**: 34 → 67 entries, and all 65 of rust-old's ids resolve. New entries include
  gpt-4.1, o3, o4-mini, gemini-2.5, claude-3-5-sonnet, llama-4, grok-2, phi, command-r and the
  embedding models. Exact rows where a family prefix gave the wrong answer: `mistral-nemo` has 128k
  context, not `mistral`'s 32k, and `o1-mini` has 128k, not `o1`'s 200k. Before this, those models
  took the unknown-model path, which skips context compaction and tier reasoning.
- **`rate_limit_rpm`**, rust-old's per-provider key and the one ADR 004 names, was silently ignored,
  so a carried-over config ran with no limit. It is now read; `rate_limit` wins if both are set.
- **`hoosh infer -m <model> <prompt>`**, rust-old's form, printed usage: `-m` consumed the model as its
  value. `-m` and `--model` are now accepted. The positional form is unchanged.
- **`/metrics`** declared `hoosh_tokens_total` but emitted its samples as `hoosh_tokens_prompt_total`
  and `hoosh_tokens_completion_total`, which had no `TYPE` line of their own. It now emits rust-old's
  `hoosh_tokens_total{type="prompt"|"completion"}`, and keeps the two 2.x names, now properly declared.
- **`POST /v1/hardware/models {model?, quantization?}`** is rust-old's form. With `model`, it reports
  whether that catalogue model runs here (`can_run`, memory, headroom), or 404. Without it, it lists
  every compatible model. `GET` keeps the fixed size table.
- **`POST /v1/hardware/placement`** computed a sharding plan and discarded it. It now returns it as
  `sharding`. When the model does not fit, it adds up to five `cloud_alternatives` in rust-old's field
  names.
- **SSE keep-alive**: local streams send `: keep-alive` after every 15 s of silence, as rust-old did.
  Without it, a proxy's 60 s idle timeout could cut a stream while a model loaded or thought before its
  first token. The backend read now times out every 15 s, and the silences are summed, so a stream
  with nothing from the backend still ends after 300 s. Remote streams are driven by sandhi, which has
  no idle hook yet; that is recorded as a known non-port.

Verified live against mock Ollama and llama.cpp backends:

- Health, `/v1/models`, distinct ids on a cache hit, and Ollama embeddings over `/api/embed` with an
  array input all behaved as described.
- The 404 for an unknown model, the 429 on the third call under `rate_limit_rpm = 2`, the metrics
  block, the hardware model lookup and placement all behaved as described.
- A stream whose backend stayed silent for 35 s received keep-alives at 15 s and 31 s, then its tokens,
  all under one id.

### Tests

Eight groups, 66 assertions. The completion-id tests run against `provider_err.cyr` itself, and the
keep-alive frame is tested through a real pipe. The rest mirror the embeddings normalization, the model
listing, the catalog lookups, the config and CLI parsing, the health body, the metrics block (with a
check that every sample's metric is declared) and the keep-alive timing.

### Gates

fmt, lint, vet and deny are clean, and symbol coverage is 42% (was 39%). Every CI step, the
benchmark gate included, was green when replayed step for step against a clean export of the tracked
tree. No benchmarked path changed, since `tests/hoosh.bcyr` covers none of these handlers, so
`bench-history.csv` was not re-recorded. Binary: 2,844,664 → 2,862,408 bytes (+17,744).

### Docs

- New: `docs/development/rust-old-retirement.md`.
- The roadmap's Embeddings item is closed and replaced by "Retire `rust-old/`".
- State, README, overview and index are refreshed.
- `hoosh.cyml` notes the `rate_limit_rpm` alias.

## [2.6.13] — 2026-09-25

**Bounds the hardware planners' numeric inputs, and closes a race that could kill a worker in
requirement-match.** Both were found while working on the simulator for 2.6.12. **877 assertions**
(was 834), fmt/lint/vet/deny clean, and every CI step green when replayed against a clean export of
the tracked tree. No benchmarked path changed, so benchmarks were not re-run.

### Fixed — `POST /v1/hardware/requirement-match` could kill its worker during a hardware refresh

The handler read `_hw_registry` twice: once for the profile count and once per profile. Since 2.5.7,
`[hardware] refresh_interval_secs` lets `hw_refresh` swap the registry at any time. If a refresh
found fewer devices between the two reads (a transient nvidia-smi failure, a driver reset), `vec_get`
indexed past the end and called `_vec_die()`. That exits only the calling worker, so the client hung
and `sock_close` and `wq_done` never ran. The handler now reads the registry once and scans that
snapshot, as `hw_summary_json` already did. Its comment still called the registry immutable, which
stopped being true in 2.5.7; it now says why no lock is needed. Only gateways with refresh enabled
were exposed. The default, refresh off, never swaps the registry.

### Fixed — the hardware planners wrapped oversized numbers into wrong answers

Reproduced on the 2.6.12 binary:

- `POST /v1/hardware/simulate` with `"memory_bytes": 18446744073709551615` added a card of −1 byte.
  The digit loop had no bound, so 2^64−1 wrapped to −1 and passed the `v == 0` check. Two cards of
  9e18 bytes each parsed fine but summed to a negative accelerator total.
- A `model_params` of 3e18 wrapped ai-hwaccel's memory estimate (parameters × bits ÷ 8). Simulate and
  `POST /v1/hardware/placement` both reported `memory_required_bytes` −1101034833169298227 with
  `fits: true`; placement also said `fits_single_device` and `fits_available`.
- `POST /v1/hardware/training-estimate` with `model_params_m` 9e15 reported negative GiB figures.
- `str_to_int` wraps and skips non-digits. `18446744073709551617` became a 1-parameter model, `7e9`
  became 79, and `7000000000.0`, which is what Python's `json.dumps` writes for `7e9`, became 7e10.

All four inputs now go through one parser, `_hw_whole_bounded`. It checks the cap before each
multiply, so it cannot wrap, and the handler answers 400 when a value is out of range or is not a
whole number:

- `memory_bytes` is capped at `SIM_MAX_DEVICE_BYTES`, 2^56 bytes (64 PiB). 64 cards at the cap total
  2^62, which leaves 2^62 for the detected devices, so the simulator's totals cannot wrap either.
- `model_params` (simulate, placement) is capped at `HW_MAX_MODEL_PARAMS`, 10^14, about 50× the
  largest disclosed models. `model_params_m` (training-estimate) is capped at 10^8. The training
  estimate is the first to overflow, near 8.4 × 10^14 parameters, so the cap keeps 8× of headroom.
- The error names the cap, as in `model_params out of range (max 100000000000000)`. A sign, an
  exponent, a real fraction or a stray byte gets `… must be a positive whole number`. 0 keeps its
  existing message, `… must be greater than 0`.
- A zero fraction such as `7000000000.0` is the same whole number, so Python clients that send floats
  get the right value. Before, `memory_bytes` read it correctly by accident and `model_params` read
  it 10× too large.
- Blanks around a value are trimmed, because the flat JSON parser leaves a trailing CR or tab in a
  bare value. An absent, empty or `null` value still means "not given", so placement still falls
  back to the model name and then to 7B.
- The `memory_bytes` scan reads the whole number token, so `1.5e9` is refused rather than read as a
  1-byte card.

Verified live on the dev host against the rebuilt binary. Each input above now gets the 400
described. 64 cards at the cap simulate without wrapping, for a total of 2^62 plus the host's 8 GiB.
Training at the cap returns 1024650000 GiB×1000. A null or absent `model_params` in placement still
falls back.

### Tests

`tests/hoosh.tcyr` gains three groups with 43 assertions:

- `hardware_whole_number_inputs` mirrors the parser, including the old loop's wraparound.
- `hardware_planner_caps` checks the arithmetic headroom at both caps, and the wraps at the values
  that used to get through.
- `hardware_requirement_snapshot` mirrors the scan with a registry swap between the count and the
  loop.

Breaking the mirrors fails them: removing the pre-multiply check fails 5 assertions, dropping the
zero-fraction rule fails 3, dropping the trimming fails 3, and reading the global once per profile
fails 2. Symbol coverage rises from 37% to 39%.

### Benchmarks

Not re-run, because no benchmarked path changed: `tests/hoosh.bcyr` covers none of these handlers.

Binary: 2,840,368 → 2,844,664 bytes (+4,296).

### Docs

The README and `state.md` stats are refreshed: 877 assertions, 156 groups, ~12,700 source lines and
39% symbol coverage. The roadmap's "Current" line moves to v2.6.13.

## [2.6.12] — 2026-09-25

**Finishes the ai-hwaccel 2.4.0 adoption.** 2.6.11 bumped `ai-hwaccel` 2.3.22 → 2.4.0 and left two
follow-ups in `src/lib/hardware.cyr` on purpose. The what-if simulator still summed per-device memory
by hand, and the serial-detector workaround from 2.5.9 had not been re-evaluated. This release closes
both. **834 assertions** (was 805), fmt/lint/vet/deny clean, benchmarks within noise of 2.6.11, and
every CI step green when replayed step for step against a clean export of the tracked tree.

### Fixed — `POST /v1/hardware/simulate` counted shared system RAM once per device

`_sim_snapshot` summed `profile_memory_bytes` over every accelerator. Since ai-hwaccel 2.3.28, a
profile's `shared_mem` marks the part of its memory that is system RAM. That is all of it for Apple's
GPU and Neural Engine and for the client NPUs, a Vulkan iGPU's heap, and GH200's 480 GiB of Grace
memory. `reg_total_accel_memory` counts that RAM once: each device adds its memory minus its shared
part, and the largest shared part is added at the end. The simulator counted it once per device. On a
48 GiB Apple M5 Pro (GPU 48 GiB + Neural Engine 4 GiB) it reported 52 GiB of accelerator memory and
`fits: true` for a 50 GiB model, and its `original` figure disagreed with `GET /v1/hardware`'s
`accel_memory_bytes`.

- Both totals now use ai-hwaccel's accounting. `accelerator_memory_bytes` is what
  `reg_total_accel_memory` reports for the same profiles, and `system_memory_bytes` counts the CPU's
  RAM once the same way. Hosts with only discrete cards are unaffected: nothing is shared, so the
  total is the plain sum it was.
- Unavailable profiles are skipped by the totals, `accelerator_count` and packing, as ai-hwaccel's
  totals skip them. rust-old's snapshot likewise counted `available_profiles()`. No 2.4.0 detector
  marks a profile unavailable, so this changes nothing on a detected registry today.
- `shards_required` double-counted the same way. The largest-first packing took each device's whole
  `memory_bytes`, so after the GPU it counted the Neural Engine's 4 GiB, RAM the GPU already covers.
  Now a device holds its own memory plus what is left of the shared pool, up to the part it can
  address, so the pool is spent once. On the Apple host with two hypothetical 2 GiB cards, a 52 GiB
  model spans 3 devices, not 2. Totals and packing now use the same accounting, so a model that
  `fits` always packs.
- Hypothetical `add_devices` cards are discrete. `_sim_profiles` sets their `shared_mem` to 0 instead
  of relying on `profile_cuda`'s default: a card marked shared would fold into the pool and add
  nothing.
- `hw_simulate_json` reads the registry once for both snapshots. It used to read `_hw_registry`
  twice, so a concurrent `hw_refresh` could put `original` and `simulated` on different device
  lists.

Verified against the real module. A scratch harness compiled `src/lib/hardware.cyr` against
ai-hwaccel 2.4.0 and matched `reg_total_accel_memory` on 3,000 random mixed device sets and on these
hosts: Apple, an Intel NPU + NVIDIA laptop, GH200, discrete-only, an unavailable device, and the
clamp edge cases. In every random set, a model that fits packed and one byte over the total did not.
On the dev host the live `original.accelerator_memory_bytes` equals `GET /v1/hardware`'s
`accel_memory_bytes`. `tests/hoosh.tcyr` gains a `hardware_simulate_shared_memory` group of 29
assertions. It mirrors the accounting and checks it against a literal transcription of
`reg_total_accel_memory`. Reverting the mirror's totals to the old sum fails 10 of them, and
reverting its packing fails 3.

### Evaluated — the threaded detector is fixed upstream, and hoosh still does not use it

2.5.9 moved to the serial detector because `registry_detect_threaded`'s post-passes were passed the
registry where a `system_io` was expected. ai-hwaccel 2.3.25 fixed that. Every entry point now shares
`registry_post_passes(r, allow_exec)`, and `registry_detect_threaded_with(builder_mask)` would keep
`[hardware] disabled_backends` working. Checked on ai-hwaccel 2.4.0 on the dev host (AMD Renoir APU):
the serial and threaded registries hold the same 2 profiles and the same 8 warnings. Both
`reg_system_io()` results are real `system_io` structs with the same storage devices, and the totals
and best device match. The symptom 2.5.9 measured, 10 warnings instead of 8, is gone.

hoosh stays on the serial detector for two reasons:

- **The threaded path orders profiles differently, and hoosh reads that order.** Serial appends
  backends in a fixed order: CPU, CUDA, ROCm, Apple, Vulkan, Intel NPU, AMD XDNA, and so on.
  Threaded appends the sysfs backends first, then the CLI backends as it joins their threads.
  `POST /v1/hardware/requirement-match` reports the first matching profile, and
  `/v1/hardware/simulate`'s `remove_count` drops the first N accelerators. On an Intel NPU + NVIDIA
  laptop, `any-accelerator` would answer `Intel NPU` instead of `CUDA GPU`, and removing one device
  would pull the NPU. `GET /v1/hardware` and `/v1/hardware/telemetry` would also list devices in a
  different order. Best-device selection goes by rank. The only tie the reorder flips is Metal against
  an unknown Neuron chip (both rank 40), which would need a host that has both.
- **The speedup is gone.** Medians of 15 alternated rounds: serial 22.5 ms, threaded 21.2 ms. 2.5.9
  measured 34 vs 20 ms. A per-stage breakdown shows why: `vulkaninfo --summary` takes 18–23 ms, every
  other probe takes under 1 ms, and threads cannot split one probe.

The `_hw_detect` comment now records this and what switching would take. `hw_init`'s comment still
said hoosh ran the threaded detector; it is corrected. Its `THREAD_STACK_SIZE` ≥ 1 MiB bump is gone.
The bump existed for the threaded detector's probe threads, the serial detector spawns none, and it
never fired anyway: cyrius's default stack has been 2 MiB since 6.3.13.

### Benchmarks

The gate ran and recorded all 25 benchmarks. No benchmarked path changed, because the suite does not
cover `src/lib/hardware.cyr`. Every row is within noise of the 2.6.11 run (`192be68`). The largest
moves are `cache_insert` 108 → 98 ns and `route_matches_model` 37 → 34 ns, both faster and both in
code this release does not touch. The Baseline column in `benchmarks.md` (`303ca06`) predates cyrius
6.6.5's harness rewrite (see 2.6.11), which accounts for its larger deltas.

Binary: 2,840,368 bytes, the same size as 2.6.11.

### Docs

The roadmap's "Upstream-gated (ai-hwaccel)" item is now a low-priority open item. It records why the
threaded detector stays off and what would justify revisiting it. The roadmap's "Current" line moves
from v2.5.11 to v2.6.12. `state.md` and the README stats are refreshed: 834 assertions, 153 groups,
~12,600 source lines. The README's testing block still said 427 assertions and 16 benchmarks, and is
corrected. The `hardware.cyr` header names the ai-hwaccel 2.4.0 bundle instead of 2.3.7; the
simulator needs 2.3.28 or later for `profile_shared_mem`.

## [2.6.11] — 2026-09-25

**Toolchain and dependency refresh.** cyrius 6.6.2 → 6.6.6, `ai-hwaccel` 2.3.22 → 2.4.0, vendored
`bote-core` 3.1.4 → 3.3.13 and `majra` 2.5.3 → 2.9.1. No gateway logic changed. **805 assertions**,
fmt/lint/vet/deny clean, and every CI step green when run step for step against a clean `git archive`
export. The build gains one benign warning, the `uname_release` duplicate explained below. 6.6.6
also reports sigil's per-bank static crypto buffer (`lib/sigil.cyr:25118`) as a located warning;
6.6.2 printed the same condition as an unlocated note.

### Changed — cyrius pin 6.6.2 → **6.6.6**

The snapshot drops nothing hoosh uses; its one new file, `alloc_cx.cyr`, is the cx-target allocator.
Bundled modules in hoosh's compile set moved: sandhi 1.9.16 → 1.9.17, sigil 3.12.16 → 3.12.18, patra
1.14.1 → 1.14.3, sakshi 2.5.1 → 2.5.2, bayan 1.5.5 → 1.5.6.

**`sys` added to `[deps].stdlib`, ahead of `sigil`.** sigil 3.12.18 (cyrius 6.6.4) changed
`agnosys_uname` from a raw `syscall(63)` (which is `read(2)` on aarch64) to `lib/sys.cyr`'s
`sys_uname`. A hand-written stdlib list does not pick up that dependency, so the 6.6.6 build warned
`undefined function 'sys_uname'` and the call compiled to a trap. The only caller is sigil's
kernel-module sign-file fallback, which hoosh never reaches (DCE NOPs it), so this was a latent trap,
not a live crash. The fix follows the upstream guidance in cyrius `docs/ecosystem.md`. sigil still
carries its own `uname_release` and `UTS_*` offsets, identical in body and value on every target, so
the build now prints one benign `duplicate fn 'uname_release'` warning from `lib/sigil.cyr`.

**`lib/` re-synced clean.** The working `lib/` held 105 files. Seven of them (ganita, niyama, yukti,
vani, mabda, sankoch, yantra) were leftovers from older toolchains that the declared-subset
`cyrius lib sync` never refreshes; hoosh includes none of them, and the build warned that `./lib/`
shadowed the pinned snapshot. `cyrius.lock` had locked those leftovers too, 111 entries in all. It now
records exactly what a fresh checkout vendors: 81 entries, written in path order, which is most of the
lock diff. The regenerated lock is byte-identical to one produced from a clean `git archive` export,
so CI's `cyrius deps --verify` sees the same set.

### Changed — `ai-hwaccel` 2.3.22 → **2.4.0**

Every function, enum constant and `AIHW_BACKEND_COUNT` that hoosh references is unchanged in arity
and value, and no new name collides with hoosh. The changes (2.3.25–2.4.0) are in detection and
accounting:

- A GPU reported by both Vulkan and CUDA/ROCm is now listed once, matched by PCI vendor:device, and
  Vulkan views of Apple Metal GPUs are dropped. Device counts and totals go down on such hosts.
- `reg_total_memory` and `reg_total_accel_memory` count shared system RAM once instead of once per
  device (unified memory, client NPUs, GH200). Vulkan iGPUs are sized from their real heap, and
  software Vulkan devices (lavapipe, SwiftShader) are no longer reported as GPUs.
- macOS reads real RAM from `hw.memsize` and detects Apple Silicon through sysctl.
- The threaded detector's post-pass bug that 2.5.9 worked around (the registry was passed where a
  `system_io` was expected) is fixed upstream at 2.3.25. hoosh still runs the serial detector:
  switching back is a separate change, and the threaded path orders profiles differently.
- ⚠ `POST /v1/hardware/simulate` still sums per-device memory itself (`_sim_snapshot`). On a
  unified-memory host its accelerator total now differs from the `reg_total_accel_memory` figure that
  `GET /v1/hardware` and placement use. The simulator's own number is unchanged by this upgrade: it
  counted shared RAM once per device before, as ai-hwaccel did. What changed is that the two no longer
  agree.

`data/cloud_pricing.json` and `data/models.json` are byte-identical between the two tags, so there
was nothing to re-sync.

### Changed — vendored `bote-core` 3.1.4 → **3.3.13**

All ten bote-core functions hoosh calls keep their signatures and bodies. `_json_emit_escaped` is
byte-identical and not `private`. The rest of the upgrade is additive. Behavior visible through
`POST /v1/tools/call`, which hands the body to bote's codec:

- `ping` now answers `{"result":{}}` instead of `-32601`.
- Protocol version `2025-06-18` is accepted.
- `initialize` reports `serverInfo` version `3.3.13`. The name is still `bote`; the new
  `dispatcher_set_server_info` (3.3.0) could make it `hoosh`.

The core profile is now 12 modules (`content.cyr` joined at 3.3.6), and `scripts/sync-bote.sh`
defaults to 3.3.13. The rationale for vendoring still holds: bote 3.3.13 still declares `[deps.libro]`
and `[deps.majra]`.

### Changed — vendored `majra` 2.5.3 → **2.9.1**

`pubsub_new`, `pubsub_publish` and `pubsub_total_published` keep their signatures and contract, so
`src/lib/events.cyr` needs no change. majra no longer declares top-level `SYS_CLOCK_GETTIME` /
`SYS_GETRANDOM` vars holding x86_64 numbers, which shared names with the stdlib's per-architecture
syscall enum. The one remaining shared name is the enum member `ERR_NONE`, which is 0 on both sides.
majra 2.8.0 renamed its codes `MAJRA_ERR_*` and keeps `ERR_*` as aliases until 3.0.0. The bundle grows
from 3,284 to 5,984 lines, and majra ships no pubsub-only profile.

### Fixed — every published event leaked 152 bytes (via majra 2.9.1)

majra 2.5.3's `pubsub_publish` called `map_keys(patterns)` on every publish. That allocates a vec (a
24-byte header plus a 16-slot buffer), and hoosh's bump allocator never frees. `event_emit` publishes
on every completed and failed inference, every rate-limit rejection and every health transition. So a
long-running gateway grew by at least 152 bytes per completed request and got nothing for it: hoosh
has no pattern subscribers. 2.9.1 snapshots the pattern slot array under the lock instead, allocates
nothing per publish, and skips the pattern walk when there are no pattern subscribers.

### Benchmarks — ⚠ the instrument changed

**cyrius 6.6.5 rewrote `lib/bench.cyr`, so `bench-history.csv` has a measurement break at this
release.** Per-op figures used to be truncated to whole ns and are now rounded half-up. Clock-read
cost is now netted at read time, and the calibration floor is re-checked. The harness's own note:
*"a row can move by up to 1 ns against 6.6.4"*. On rows of a few ns that is +5–15% with no code
change. Compiled against the **old 6.6.2 harness**, the 2.6.11 code measures exactly what 2.6.10 did
on every sub-40 ns row: `pool_available`, `work_queue_push_pop`, `rate_limit_check`,
`event_publish_ring`, `estimate_tokens_per_provider` and `route_matches_model` all read 4 / 8 / 32
ns, same as before. So the jumps `benchmarks.md` shows on those rows are the instrument, not hoosh.

Method: back to back on one machine against a pristine 6.6.2 export of HEAD, five alternating rounds
per variant, comparing medians. Three variants separated the causes: 6.6.6 alone (old bote-core, no
`sys`), the release built against the old harness, and the release as shipped. **No shift is
attributable to the dependency bumps.** Two shifts persist under the old harness and come from the
toolchain itself, since both reproduce with the old bote-core and without `sys`:

- `route_select_20_providers`: +3–9% across runs, about +25–75 ns on ~790 ns.
- The constant-time auth pair is layout-sensitive. `auth_verify_wrong_late` read 90–94 ns on 6.6.2
  and 100–105 ns on 6.6.6, while `auth_verify_wrong_early` swung between 91 and 104 ns across builds
  that differ only in code the bench never runs. Under the shipped harness all three auth rows
  (`ct_eq_bytes_lens`) sit at 101–107 ns wherever the mismatch falls. `lib/ct.cyr` and
  `src/lib/auth.cyr` are unchanged.

Every other row is within ±4% of 6.6.2 under the old harness, and within ±6% as shipped. The first
single run showed `cache_insert` at +96%; alternated runs put it at 92–96 ns against 94–97 ns, so
that was noise.

Binary: 2,765,064 → 2,840,368 bytes (+75,304, +2.7%). The toolchain adds 12,904 bytes, `sys` 4,192,
and the three dependency bumps 58,208.

### Docs

README stats (pin, source size, tests, binary) and the `docs/development/state.md` table are
refreshed; both were several releases stale. The README also said the Cyrius binary was ~15 MB and
that `CYRIUS_DCE=1` shrinks it. It is ~2.8 MB either way: DCE NOPs unreachable functions in place (~1.1
MB of them here) rather than removing them. `CLAUDE.md`'s toolchain line said 6.5.35. Stale comments
in `cyrius.cyml`, `src/main.cyr`, `src/lib/mcp.cyr` and `scripts/sync-bote.sh` are corrected: they
cited bote 2.7.6 and majra 2.4.7, a 9-module include-free core bundle, a `registry_new` that
ai-hwaccel renamed to `hw_registry_new` long ago, and a `ratelimit_*` collision that hoosh's
`hoosh_ratelimit_*` naming already avoids.

## [2.6.10] — 2026-09-11

**Migrated to the cyrius 6.6.x value form.** All 17 CI steps green.

### Changed — cyrius pin 6.5.36 → **6.6.2**

cyrius 6.6.0 flipped `Result` / `Option` / `Either` declared `: stack` to a value form — a payload
variant returns a `(tag, payload)` register pair and allocates nothing; `payload()` is gone.
6.6.2 is the repair release.

16 declarations split across `src/`, `tests/` and `fuzz/`. No propagation traps, no
reassignment sites, no `callptr` Results — the migration was mechanical throughout.

### Changed — `ai-hwaccel` 2.3.20 → **2.3.22**

### Fixed — `pricing_lookup` was defined three times, two of them colliding

`src/lib/pricing.cyr:75` defines `pricing_lookup(model, prov, out)`. `tests/hoosh.tcyr` defined it
**twice more**: an arity-3 copy at :448, and an arity-2 variant at :3024 that returns the in/out
prices packed into one i64 instead of filling an out-param — a genuinely different function
wearing the same name.

All three were in scope together. Before 6.6.2 that was a silent "last definition wins", so calls
to whichever lost mis-bound their arguments; 6.6.2 makes a same-name different-arity duplicate a
hard error. The arity-2 variant is renamed `pricing_lookup_packed`.

⚠ Unlike the ecosystem's other instances of this class — `health_check_new` (agnostik ↔ argonaut)
and `audit_entry_new` (kavach ↔ agnostik) — this one is **intra-repo**: a test file shadowing its
own source. Pre-existing, and verified so against HEAD before any migration edits.

### Fixed — two fuzz harnesses called `sys_exit` without including `lib/syscalls.cyr`

`fuzz/inference_request.fcyr` and `fuzz/message_content.fcyr` include `string`/`alloc`/`vec`/
`str`/`fmt` only. CI builds harnesses with `--strict --no-deps`, where a **reachable** undefined
symbol is an error rather than a warning, so both failed to build. Pre-existing.

## [2.6.9] — 2026-08-30

**Nothing is deferred.** This closes the last four audit findings — the DLP
escape bypass, shadowed request parameters, the unpartitioned cache, and the CLI
stream reader — so the audit backlog is empty. **805 assertions** (was 775),
`src/` lint-clean, fmt/vet/deny clean, no build warnings, no benchmark
regressions.

It also removes 26 more stale test mirrors: the DLP suite had been asserting
against a **copy** of the scanner, one that predated the 2.5.10 custom-pattern
support and defaulted to enabled where the shipped module defaults to disabled.
Those assertions now run against `src/lib/dlp.cyr` itself.

### Fixed — JSON escapes defeated every DLP pattern

`dlp_scan_level` was handed the **undecoded** request body, while
`_extract_messages` forwards that same undecoded span to the provider, which
then decodes it. hoosh scanned one string and shipped a different one, so any
pattern could be hidden by writing it with JSON escapes:

- `"ssn 123\u002d45\u002d6789"` classified as PUBLIC and was forwarded.
- `"\u0073\u006b\u002d<key>"` hid an API key from the `api_key` matcher.

This was not only an evasion path. Python's `json.dumps` defaults to
`ensure_ascii=True`, so a **default Python client** escapes every non-ASCII byte
— which meant a non-ASCII `[[dlp_pattern]]` literal never matched real traffic at
all. rust-old scanned `msg.content.text()`, i.e. the decoded value; the port lost
that.

`dlp_scan_level_json` now scans the raw bytes and an escape-decoded copy and
takes the higher classification. Decoding handles the six short escapes and
`\u` (including surrogate halves, which become U+FFFD rather than raw bytes),
and is skipped entirely when nothing was escaped. Both the chat and embeddings
paths use it.

### Fixed — a nested schema key shadowed the real request parameter

`temperature`, `top_p`, `max_tokens`, `stream`, `reasoning_effort`, `messages`
and `tools` were all located by scanning the raw body for the key at **any**
offset. Inside a tools JSON-Schema those are ordinary property names, so a
nested one won whenever it appeared first — and a tools array precedes the
trailing parameters in every OpenAI client that sends one. A client asking for
`"max_tokens": 64` could be silently capped by a schema's `max_tokens` property.

The `messages` case was worse: the validator saw a nested `"messages"`, so the
request passed validation, and `_extract_messages` then returned the **nested**
span — forwarding a fragment of the tool schema as the conversation.

All seven now resolve through one depth- and string-aware lookup
(`_top_key_value`, in `jsonlite.cyr` so it is tested against the real
implementation), which only matches a key at the top level of the request object
and skips string contents, so a key-shaped substring inside a message can never
match either.

### Fixed — the response cache was shared across tenants

Neither the exact cache key nor the semantic index carried any caller identity,
so two consumers of the same gateway sending the same prompt shared a cached
response — one produced under a different tenant's credentials and provider
account. `auth_check` now records **which** configured token authenticated the
request (thread-local slot 4, following trace/request-buffer/provider_err), and
that identity is part of both the exact key and the semantic index scope. The
semantic index needed it independently: it matches on vector similarity, so the
exact key's identity would not have protected it — a near-identical prompt from
another tenant still scored above the threshold. A rejected request clears the
slot so the next request on that worker cannot inherit it.

Note the cache is cold once after upgrade, since every key now includes the
identity prefix.

### Fixed — the CLI dropped streamed output

`hoosh infer --stream` began each scan at the **pre-read** fill level, so a
`data: ` marker delivered by an earlier read was never re-tested, and the
`acc > 60000` reset discarded a partial frame outright. This is not a rare
split: `http_sse_event` issues three `sys_write`s per event and nothing sets
`TCP_NODELAY`, so a read returning exactly the six bytes `"data: "` is the
expected case — and the loop condition `i + 6 < acc` would not even execute
against it. Against a fast local gateway the CLI could print almost nothing. It
now always scans from 0 and slides the unconsumed tail down, so a frame spanning
any number of reads is emitted exactly once.

### Performance

Interleaved A/B against 2.6.8 (`fafe31e`), 3 rounds each, same machine and
session: **no regressions and no wins** across all 25 benchmarks. The added DLP
decode pass is skipped whenever the body contains no escapes, so
`dlp_scan_clean_prompt` is unchanged.

### Audit status

The 10-dimension security audit that opened in 2.6.6 is **closed**: 42 distinct
confirmed findings, all fixed across 2.6.6 – 2.6.9. Test count over that arc:
714 → 805.

Two standing items remain, neither an audit finding:

- 103 `line exceeds 120 characters` lint warnings in `tests/hoosh.tcyr` /
  `tests/hoosh.bcyr` — long JSON fixtures, where mechanical rewrapping would
  alter the test data. Needs a project decision on the limit for fixture lines.
- `src/vendor/bote-core.cyr` fails `cyrius fmt --check` — vendored upstream code,
  deliberately not reformatted.

## [2.6.8] — 2026-08-30

**The audit closes.** This lands the remaining findings — the confidential-data
leak, the unmetered streaming path, the errors laundered into successes, and the
last of the concurrency holes — and clears the transitive-dependency collision
upstream. **775 assertions** (was 753), `src/` lint-clean, fmt/vet/deny clean,
and the build is now warning-free for the first time in this arc. Benchmarks show
**no regressions**.

### Fixed — the /v1/embeddings controls that 2.6.6 exposed

Fixing the `/v1/embeddings` SIGSEGV in 2.6.6 had a consequence the audit called
out explicitly: two findings had been refuted *only* because the handler crashed
before it ever forwarded. Removing the crash activated both. The endpoint had **no
DLP scan, no rate limiting, and no metrics**, and it forwarded through
`http_post_local`, which speaks plaintext and emits no `Authorization` header — so
a remote route would have received the body in the clear. It is now DLP-scanned
(RESTRICTED refused), restricted to local providers, rate limited, and counted.

### Fixed — confidential data left the box

`_chat_prep` enforces "CONFIDENTIAL is re-routed to a local provider" and logs
that it did so — then handed the **same raw body** to `_embed_query_body`, which
picked a route with a plain `router_select` (no locality constraint), JSON-escaped
the entire body into `"input"`, and POSTed it to a **remote** embedding provider
with the operator's key attached. The shipped `hoosh.cyml` suggests
`text-embedding-3-small`, i.e. the remote route, so the leaking configuration was
the default one, while README advertises "Confidential forced local". Confidential
requests now skip semantic embedding entirely; the exact-key cache still applies.

### Fixed — money and metering

- **`"stream": true` was never billed.** The streaming branch committed 0 to the
  budget pool and never reached `cost_record` — `/v1/tokens/pools` reported
  `used: 0` forever. Streaming is the default in every real client, so in practice
  most traffic was unmetered and no budget could be exhausted by it. The
  reservation the request already made is now converted into usage, and cost and
  metrics are recorded. (Decoding the provider's own counts — OpenAI
  `stream_options.include_usage`, Anthropic `message_delta.usage` — is the
  follow-up; charging the reservation is strictly better than charging nothing and
  never exceeds what admission approved.)
- **The budget reserved the output estimate but charged prompt + completion.**
  `est_tokens` is the client's own `max_tokens`, so `{"max_tokens": 1}` was
  admitted whenever a single token was free and could still be charged a
  quarter-million prompt tokens — one request could exceed the entire configured
  cap. The reservation now includes an estimate of the prompt, using the same
  estimator compaction uses, so it is an upper bound on the charge.
- **Provider token counts are clamped.** All four extractors packed
  `prompt * 2^32 + completion` from untrusted provider JSON, and
  `_json_extract_int` honours a leading `-`, so `completion_tokens: -5` unpacked
  to 4294967291 and flowed into the budget, the cost record and the usage block
  returned to the client.
- **The rate limiter granted 0 rpm at small limits.** `add = rpm * elapsed_ms *
  1000 / 60000` truncates to 0 for a short interval, and `last_refill` was stamped
  to `now` regardless, discarding it. Because refill runs on **every** check
  including rejected ones, a client polling faster than one token's worth of time
  never accumulated anything: measured against the real 5–6 ms tick,
  `rate_limit = 5` granted **0 rpm** — total starvation — and 10 granted about 3.
  Free tiers of 5–10 RPM are common, and one tenant polling a shared route drove
  it to zero for every other tenant. The clock now advances only by the time
  actually converted into tokens.

### Fixed — errors laundered into successes

- **The local streaming path never read the provider's HTTP status.**
  `stream_skip_headers` pulls the whole response head into a buffer and the status
  line was simply discarded, so a 404 "model not found" went into the NDJSON loop,
  extracted nothing, and finished as a clean `200 OK` with
  `finish_reason:"stop"` and no content — logged nowhere, with
  `provider_last_error_status()` still 0 and `audit_record` filing it as a
  **success**. That is the exact failure 2.6.5 fixed for the remote path. The
  status is now parsed, a bounded prefix of the error body is captured, and the
  caller takes its existing error-chunk path. The parser moved to
  `provider_err.cyr` so it is tested against the real implementation.
- **Remote provider calls had no timeouts at all.** `sandhi_http_post` was called
  with `opts = 0`, so every option getter returned 0: no connect deadline, no read
  timeout, no total deadline — the outbound twin of the accepted-socket timeouts
  added in 2.6.6. A provider that accepted and went silent parked a worker
  forever. The library's 256 KiB default response cap also made any longer
  completion fail as a protocol error, which the retry layer treats as retryable:
  four identical re-POSTs, each generating and billing a full completion upstream.
  Both call sites now pass explicit connect/read/write/total bounds and a response
  cap sized for real completions.

### Fixed — concurrency

- **`provider_err` state was four process globals** written from inside
  `provider_forward`, which every caller invokes with `_chat_lock` *released*,
  while readers take `_chat_lock` — a lock protecting nothing against writers
  holding nothing. Another worker's `reset` could wipe a captured error before its
  owner read it, silently regressing every diagnostic the module exists to provide
  back to "provider backend unreachable", and one client could be answered with
  another client's provider error text. Now per-thread.
- **The sync-batch path could deadlock the pool.** `wq_push` spins forever on a
  full ring, and the batch loop pushes up to `BATCH_MAX_ITEMS` **from a pool
  worker** before reaching its barrier; once every worker is in that spin nothing
  calls `wq_pop` and the state is terminal — restart required, `/v1/health`
  included. A sync batch is now refused with 503 when the queue lacks room, which
  makes the spin unreachable.

### Fixed — untrusted input

- **Client `traceparent` was unvalidated.** Only `1 <= n <= 200` was checked, but
  the OTLP span builder slices it at **fixed offsets**, so `traceparent: x`
  produced a 2-byte allocation from which ~50 bytes were read and emitted
  unescaped into the span batch; an embedded NUL truncated the fragment and
  corrupted the whole batch. The same buffer is forwarded verbatim into every
  outbound request header. It is now validated against the W3C shape
  (`00-<32 hex>-<16 hex>-<2 hex>`, exactly 55 bytes) and a malformed value is
  replaced by a freshly generated one. The existing tests **blessed**
  `00-abc-def-01` as valid; they were wrong and are corrected.
- **`Transfer-Encoding` was ignored entirely.** With no `Content-Length` the drain
  never ran and the handler received chunk-size lines, CRLFs and the terminator as
  the body: short bodies "worked" by accident because the JSON parser skips to the
  first `{`, while the framing bytes polluted the cache key, and any body spanning
  more than one segment was **silently truncated** — so the prompt forwarded and
  billed was a cut-off prefix. Reachable via `curl -T`, undici streams, or an
  HTTP/2-terminating proxy. Now refused with 501 rather than mis-framed.

### Fixed — semantic cache

- **A hit ignored the requested model.** The exact key hashes model + body, but
  the index stored only `(key, vec, n)` and matched on dimension and cosine score
  alone. Two bodies differing only in `"model"` are near-identical strings that
  score far above the shipped 0.85 threshold, so a `claude-opus-4-1` request was
  served the `gpt-4o-mini` answer, labelled `X-Hoosh-Cache: SEMANTIC`. Entries now
  carry the model and a hit requires an exact match.
- **Index pruning was dead on two paths.** `semantic_remove` compared keys by raw
  **pointer** identity while the caller passes a freshly allocated `sha256_hex`
  cstr, so the expired-on-read path never matched; and LRU eviction called
  `map_delete` with no `semantic_remove` at all, so under capacity pressure every
  eviction orphaned an index entry permanently. Because entries are appended, the
  orphans collect at the head — exactly where a configured `max_search` window
  looks — so hit rate decayed toward zero.

### Fixed — silent configuration failures

`_config_expand_env` returned 0 for an unset variable and the provider loop
dropped it with no `else` and no log, so the route kept no API key,
`_provider_headers` skipped the auth block, and requests went upstream with **no
credential** — surfacing as a provider 401 the operator naturally blamed on the
provider. The shipped `hoosh.cyml` has three remote providers each using
`$..._API_KEY`. `${NAME}` failed the same way, since the `$` test matched the
braced form too. Unset variables are now named in a warning and `${NAME}` is
supported.

### Changed

- **`[deps.ai-hwaccel]` 2.3.19 → 2.3.20**, which makes its `bayan` dependency
  optional and feature-gated. This removes the transitive `dist/bayan-json.cyr`
  that shadowed the stdlib bayan: **131 duplicate function definitions → 0**.
- **`lib/sankoch.cyr` refreshed** from 2.7.8 to the pinned 2.7.10, clearing the
  last `./lib/ shadows version-pinned lib/` warning. The build now emits no
  warnings at all.

### Performance

Interleaved A/B against 2.6.7 (`8279314`), 3 rounds each, same machine and
session: **no regressions and no wins** across all 25 benchmarks — every delta
within its own run-to-run spread.

### Known issues

Four audit items remain, all deferred deliberately with reasons recorded in the
repair plan:

- **Depth-aware top-level parameter scanning** — a key nested inside a tools
  schema can shadow the real `temperature` / `max_tokens` / `stream`. Correctness
  only; no security boundary is crossed.
- **DLP scans raw JSON bytes**, so a `\uXXXX` escape defeats the built-in
  patterns. Note Python's `json.dumps` defaults to `ensure_ascii=True`, so this
  also means a non-ASCII custom pattern never matches a default Python client.
- **Per-consumer cache partitioning** — the cache key carries no caller identity.
  2.6.8's model match removes the most likely cross-serve.
- **The CLI `--stream` reader drops SSE frames that straddle a read boundary.**
  Client-side only; the gateway's own streaming is unaffected.

Two pre-existing gate items also stand: 103 `line exceeds 120 characters` lint
warnings in the test files (long JSON fixtures, where mechanical rewrapping would
alter test data), and `src/vendor/bote-core.cyr` failing `cyrius fmt --check`
(vendored upstream code, deliberately not reformatted).

## [2.6.7] — 2026-08-30

**Second tranche of the security audit**, plus a test-integrity fix that turned
out to matter more than any single bug. **753 assertions** (was 722),
fmt/vet/deny clean, `src/` lint-clean, 25 benchmarks recorded with no regression
beyond 1 ns of clamp cost. Every fix below has a regression test that was
verified to FAIL against the previous code.

### Fixed — the test suite was checking stale copies of shipped code

`tests/hoosh.tcyr` includes only a handful of `src/` files and re-implements the
rest as mirrors. Five `jsonlite.cyr` functions were **both** included **and**
redefined in the test file, and since last definition wins, every assertion that
touched them exercised a stale copy rather than the shipped parser — the mirrors
had drifted and still carried the trailing-backslash bug below. Two `compact.cyr`
functions were shadowed the same way. All seven duplicates are removed, the
suite still passes, and `jsonlite`/`compact`/`auth` assertions now run against
real `src/`. The `req_content_length` tests were pointed at the real function
instead of a `_t_`-prefixed copy, and the `budget`/`dlp` mirrors were re-synced.

### Fixed — untrusted input

- **Six JSON string scanners walked past their closing quote.** Each decided
  "is this quote escaped?" from the single preceding byte, which is wrong when
  that byte is itself an escaped backslash. A value ending in `\` — a Windows
  path, LaTeX, a shell continuation, a regex, all ordinary model output that
  every provider serialises correctly — swallowed its own terminator and ran on
  into the surrounding structure: `_json_extract_str` returned a span containing
  real JSON syntax, spliced **raw** into the client's SSE frame and buffered
  body, so the client's decoder threw and the whole completion was lost. On the
  request side, `_json_obj_end` bounds each message object, so a *client*
  message ending in `\` swallowed the rest of the array into the body hoosh
  POSTs upstream. All six now use the forward-scan idiom already used correctly
  in `_extract_openai_tool_calls`.
- **`req_content_length` matched at any byte offset.** The name was tested at
  every position with no line-start requirement, so the request URI or an
  earlier header *value* carrying the literal won on first-match — a `Referer`
  was enough to make a small valid request answer 413. The digit loop was also
  uncapped and wrapped i64, so an absurd run could land on a small or negative
  value that passed the `MAX_REQUEST_BODY` check. Now anchored to a line start,
  capped at 19 digits, and a malformed value returns 400 rather than being
  treated as absent.
- **The 128 KiB provider response buffer could be written one byte past.** The
  read loop filled it exactly, and `jsonlite`'s extractors then wrote a NUL
  terminator at `body + len` — one past the allocation, into the shared bump
  arena at the offset the next `alloc` on any of 7 workers was about to receive.
  The buffer now carries a byte of headroom, and a response that reaches the cap
  is refused rather than parsed from a body cut mid-JSON.
- **`semantic_parse_embedding` could hang a worker forever.** `_sem_parse_num`
  consumes nothing for a non-numeric token, returning its start index unchanged,
  and the caller spun on that byte allocating 16 bytes per iteration against a
  never-freeing arena until `alloc` returned 0 and the store went to address 0.
  Python's `json.dumps` emits bare `NaN`/`Infinity` by default, so no malice is
  required. The exponent loop was separately uncapped — `1e9223372036854775` was
  effectively permanent. No progress now abandons the parse; the exponent is
  clamped.

### Fixed — security

- **`Authorization` matching rejected valid tokens from real clients.** The
  compare was byte-exact, so a client sending the header lowercase — which
  hyper/reqwest, undici/fetch and every HTTP/2-terminating proxy do, since
  HTTP/2 mandates lowercase field names — got 401 with a correct token. The
  match was also unanchored, so it fired inside `Proxy-Authorization:` and then
  abandoned the scan without reaching the real header, and it scanned the whole
  request including the body. Now case-insensitive per RFC 9110, anchored to a
  line start, bounded to the header block, and a non-Bearer line no longer stops
  the scan.
- **`api_key` DLP detection missed every current key format.** The body required
  20+ *consecutive* alphanumerics after the prefix, but every modern key carries
  internal separators, so the run stopped almost immediately: `sk-proj-…` scored
  a run of 4, `sk-ant-api03-…` a run of 3, and both classified as PUBLIC. The
  matcher now consumes `-`/`_` as part of the token while counting only
  alphanumerics against the threshold, and the `api`/`key` prefix words are
  case-folded.
- **Negative token counts corrupted the budget permanently.** `/v1/tokens/report`
  passed the client's `actual` straight to `pool_commit` and `str_to_int` honours
  a leading `-`, so `{"actual": -1000000000}` erased recorded usage, pushed
  `pool_available` above capacity, and persisted that through
  `storage_budget_set`. `pool_reserve` had the mirror hole: a negative `tokens`
  passed the availability test trivially and then *decreased* the outstanding
  reservation. Both are clamped at the primitive.
- **Unauthenticated `GET /v1/health` did a blocking, untimed TCP connect** (and
  possible DNS resolve) per request, on a route deliberately exempt from the
  auth gate and never rate limited — so an unauthenticated caller could park a
  worker per request. It now serves from the background prober's state, as
  `handle_health_providers` has since 2.5.5.

### Fixed — concurrency and correctness

- **`/v1/costs/reset` appended to the audit chain outside `_chat_lock`** — the
  unlock was one block too early. `audit_record` is a multi-step read-modify-write
  over the chain header that writes through to a single-threaded store, so racing
  a worker's append produced duplicate ids, a shared `prev_hash` and a relinked
  tail, and `audit_verify` then reported `"valid":false` — indistinguishable from
  tampering.
- **Batch ids were generated outside the registry lock.** The "accept-thread
  only" contract went stale with the v2.4.0 worker pool. The read-modify-write
  plus a re-read straddled an allocation, so two workers could format the *same*
  id even without a torn increment; `map_set` is last-write-wins, so one client
  polled the other's results and could cancel the other's batch.
- **Async batch coordinator threads were never joined or detached**, so their
  stacks and TLS stayed mapped and the process walked toward `vm.max_map_count`;
  past it `thread_create` returned 0, the batch stayed `BATCH_QUEUED`, and the
  client polled `"queued"` with null results forever. Now detached, the result is
  checked, and a new terminal `BATCH_FAILED` status makes such a batch both
  visible to the client and reclaimable by the evictor.
- **`hw_summary_json` re-read the `_hw_registry` global inside its loop.** The
  count came from one load and each profile from another, so a concurrent refresh
  returning fewer devices made `vec_get` call `_vec_die()` → `syscall(60,1)`,
  which exits only the *calling thread*: the worker vanished without closing the
  client socket or clearing its in-flight count, hanging that client and making
  every later shutdown drain time out. It now snapshots the registry once, as
  every sibling reader already did.
- **`STRAT_LOWEST_LATENCY` treated an unmeasured route as 0 ms.** `map_get`
  returns 0 for a missing key, indistinguishable from a real measurement, and
  every failure path returns before `router_report_latency` — so a route that
  never succeeds stayed "unmeasured" and won every selection forever. An expired
  API key became a permanent outage for that model while a healthy sibling idled.
  Unmeasured routes now get a finite optimistic value that a genuinely fast route
  can beat.

### Performance

Interleaved A/B against 2.6.6 (`222ce77`), 3 rounds each, same machine and
session. No wins, and one movement outside its spread: `pool_reserve_commit`
29 → 30 ns (+3.4 %), which is the three added comparisons that clamp negative
token counts. Everything else is flat.

### Known issues — carried forward

The audit's remaining items are not in this release. The most significant:

- The **semantic-cache embedding path ships DLP-CONFIDENTIAL request bodies to a
  remote provider**, defeating the "confidential forced local" policy enforced a
  few lines earlier, and does it while holding the global `_chat_lock`.
- **`"stream": true` is charged zero tokens and zero cost** — the streaming path
  never reaches `cost_record` and commits 0 to the budget pool.
- **The local streaming path never reads the provider's HTTP status**, so a 404
  is laundered into a clean `200 OK` with empty content and audited as a success
  — the same class of defect 2.6.5 fixed for the remote path.
- **`wq_push` blocks without draining**, so a full ring plus workers in the
  sync-batch push loop can deadlock the pool.
- **`provider_err` state is a single process-global slot** written from the
  deliberately unlocked forward, so one client can be answered with another's
  provider error.
- A **semantic cache hit ignores the requested model**, and the cache key carries
  no caller identity.
- **Client `traceparent` is unvalidated** before being sliced at fixed offsets by
  the OTLP span builder and forwarded verbatim into outbound headers.

Two pre-existing gate items also remain: 103 `line exceeds 120 characters` lint
warnings in the test files (long JSON fixtures), and the 131 duplicate-fn
warnings from ai-hwaccel's transitive bayan dep — **fixed upstream in
ai-hwaccel 2.3.20**, which makes that dep optional and feature-gated; bump
`[deps.ai-hwaccel]` to 2.3.20 once it is tagged and the warnings go away.

## [2.6.6] — 2026-08-30

**Two client-triggerable remote crashes, and credential revocation that silently
did nothing.** Toolchain pin 6.5.35 → 6.5.36. **722 assertions** (was 714),
fmt/vet/deny clean, `src/` lint-clean, 25 benchmarks recorded with no
regressions. Every defect below was reproduced before it was fixed.

This release is the **first tranche** of a 10-dimension security audit that
produced 42 distinct confirmed findings. It lands the crashers, the auth defect
and the wire-correctness bugs; the remainder are listed under *Known issues*.

### Fixed — remote crashes

- **`POST /v1/embeddings` segfaulted the whole gateway.** `src/main.cyr` called
  `http_post_local(eport, …)` while the signature is
  `http_post_local(base_url, …)` — an integer port where a URL cstr was
  expected. `http_req_local` then ran `url_host(base_url)` → `load8(port + 7)`,
  an unmapped low address. One authenticated request killed all 7 workers.
  Every other call site already passed a URL; this one was stale.
- **`compact_messages` overflowed its offset table (heap corruption → SIGSEGV).**
  The boundary scan tested every byte for `{` + `"role"` with no notion of
  nesting or string context and stored into a fixed `alloc(1024)` — 128 slots —
  with **no bound check**. `handlers.cyr` caps a request at `MAX_MESSAGES = 256`
  via `_json_array_count`, which counts only depth-1 elements, so the two
  disagreed: a **single** valid message whose `content` is a nested array of
  objects carrying a `"role"` key produced 301 boundaries from a 3.9 KB body —
  173 entries, 1,384 bytes, past the allocation. Slots beyond the end were then
  read back after `str_builder_new()` had reallocated over them. Compaction is
  on by default. The scan is now depth- and string-aware (so it agrees with
  `_json_array_count` exactly, and a `{"role"` inside a string value is no
  longer a boundary), the table is sized to `MAX_MSG_OFFSETS`, and the store is
  bounded regardless. Verified: the new regression test exits **139 (SIGSEGV)**
  against the old code and 0 against the fix.

### Fixed — security

- **Config reload appended auth tokens without clearing, so revocation and
  rotation silently did nothing.** `_auth_tokens` was `vec_new()`'d once and
  only ever `vec_push`ed, while `load_config` is a live reload entry point
  (`POST /v1/admin/reload`, SIGHUP) that reports success either way. Removing a
  leaked key from `hoosh.cyml` and reloading left it authenticating every
  endpoint — `/v1/admin/reload` included — until restart; every key ever loaded
  accumulated as permanently valid. The vec is now rebuilt whenever `[auth]` is
  present. It is deliberately **not** cleared when the section is absent:
  `auth_check` allows all on an empty vec, so clearing unconditionally would
  fail *open*. The same function already called `dlp_custom_clear()`
  unconditionally — auth was the outlier.
- **Unbounded client `job_id` was concatenated into an outbound request path**
  and `memcpy`'d into a fixed 2048-byte header buffer whose builder had no
  bounds check, against a 1 MiB body limit. `job_id` is now capped at 128 bytes
  and restricted to `[A-Za-z0-9._-]`, which closes the overflow, CRLF injection
  into the request line, and path traversal in one check. Independently, both
  header builders take a capacity argument and return −1 rather than writing
  past it, so no future non-literal path can reintroduce this.
- **Accepted client sockets had no timeouts.** Only *outbound* sockets carried
  deadlines; a client that connected and went silent parked a worker in
  `sock_recv` forever, and that runs **before** `auth_check` — seven such
  connections wedged the gateway unauthenticated. `_handle_conn` now sets
  `SO_RCVTIMEO`/`SO_SNDTIMEO` (30 s), which also covers an SSE client that
  stops reading.
- **A client-sized growth allocation on the pre-auth path was unchecked.**
  `alloc(want + 1)` with `want` client-chosen up to ~1 MiB, then `memcpy` into
  it; `alloc` returns 0 on exhaustion without aborting. It now returns 503.

### Fixed — correctness

- **`Content-Type: application/jso` on every plaintext provider request.** The
  literal is 30 bytes; both header builders copied 29 and then wrote CR over the
  trailing `n`. Ollama tolerates it, but vLLM / llama-cpp-python / LocalAI /
  LM Studio return 422. The test mirror carried the identical bug and was fixed
  in the same edit.
- **Sync batches queued pointers into the thread-local request buffer.** The
  work-stealing barrier re-enters `_handle_conn` on the same thread, whose
  `sock_recv` overwrites that buffer with the next client's request — so a
  queued item could be processed against another caller's bytes. Item bodies are
  now deep-copied before `wq_push`, exactly as the async path already did.
- **`storage_restore_audit` null-dereferenced the chain.** It guarded only
  `_storage == 0`, but `audit_chain_from_config()` returns 0 when `[audit]` is
  absent (the default), so a `[[storage]]`-only config wrote to absolute address
  56 and killed the process before the listener bound.

### Changed

- **Toolchain pin 6.5.35 → 6.5.36**, matching the installed compiler; `lib/`
  resynced (67 declared modules). Clears the toolchain-drift build warning.

### Known issues — carried into the next cycle

The audit's remaining 33 findings are not in this release. The most significant:

- The **semantic-cache embedding path ships DLP-CONFIDENTIAL request bodies to a
  remote provider**, defeating the "confidential forced local" policy enforced a
  few lines earlier, and does it while holding the global `_chat_lock`.
- **Streaming requests are charged zero tokens and zero cost** — `"stream": true`
  bypasses token accounting and budgets entirely.
- **Six JSON string scanners decide "is this quote escaped?" from a single-byte
  lookback**, so a value ending in a backslash (a Windows path, LaTeX, a regex —
  ordinary model output) walks past the closing quote and corrupts the spliced
  response.
- `req_content_length` matches `content-length:` at **any** byte offset, so the
  URI or an earlier header value wins.
- A **semantic cache hit ignores the requested model**, and the cache key carries
  no caller identity.
- `semantic_parse_embedding` can **loop forever** allocating on a non-numeric
  array element (Python's `json.dumps` emits bare `NaN`/`Infinity` by default).

Two pre-existing gate items also remain: 103 `line exceeds 120 characters` lint
warnings in `tests/hoosh.tcyr` / `tests/hoosh.bcyr` (long JSON fixtures, where
mechanical rewrapping would alter test data), and 131 duplicate-fn warnings from
`ai-hwaccel 2.3.19` pulling its own `[deps.bayan] 1.5.2` (`dist/bayan-json.cyr`)
transitively, which shadows the stdlib bayan. All 131 bodies are currently
identical, so it is a supply-chain hazard rather than a live bug, and it cannot
be fixed from hoosh's side — 2.3.19 is ai-hwaccel's latest published tag.

## [2.6.5] — 2026-08-28

**A provider error is no longer laundered into a successful, empty completion.** No pin change (cyrius
6.5.35). **714 assertions** (was 689), lint/vet/deny clean, 25 benchmarks recorded. Three related fixes,
one small refactor; no behaviour change on any path where the provider succeeds.

### Fixed — an in-stream provider `error` event was silently dropped

A provider that has already sent `200 OK` cannot change its status either, so it reports a mid-stream
failure as an **event**: Anthropic sends `event: error` with
`{"type":"error","error":{"type":…,"message":…}}`, and OpenAI-compatible backends put a top-level
`"error"` object in a data frame.

`_remote_stream_cb` looked only for text deltas, thinking deltas and tool deltas. An error frame
extracted to empty in all three, so it was skipped, the stream ended normally, and the caller received a
well-formed, successful, **empty** completion. Nothing was written to hoosh's log either — the failure
left no trace anywhere.

Measured against a live stack: a request the **non-streaming** path answered fine came back from the
streaming path as `finish_reason:"stop"` with no content. Downstream, thoth reported *"response had
neither tool calls nor content"* and its operator spent a long session believing the model had gone
quiet.

Such a frame is now recognised and forwarded to the client as an `error` chunk, then the stream stops.
Recognition is **structural, never a substring search** — a model's own prose routinely contains the word
"error", and a false positive would abort a good stream mid-answer. Two precise tests: the SSE event is
*named* `error` (Anthropic), or the data frame carries `"type":"error"` (OpenAI-compatible).

### Fixed — the provider's error body was thrown away

On a non-2xx, `provider_forward` returned `FWD_PERMANENT` (or 0) and logged one fixed sentence —
`provider: permanent error, not retrying`. The response body, where every provider puts the reason, was
dropped. The single most useful fact in the whole failure reached neither the log nor the caller.

The status and the provider's own `message` are now captured (bounded to 512 bytes, control bytes
flattened to spaces so one error stays one log line) and surfaced in both directions:

```
2.6.4  log:    provider: permanent error, not retrying
       client: {"error":{"message":"provider backend unreachable","type":"server_error"}}

2.6.5  log:    provider: HTTP 404, not retrying - model: claude-does-not-exist-9
       client: {"error":{"message":"upstream provider returned HTTP 404: model: claude-does-not-exist-9",
                         "type":"upstream_error"}}
```

### Fixed — "provider backend unreachable" was the wrong words for the common case

A 400 or a 404 means the provider **was** reached and refused. Telling an operator their backend is down
when it answered instantly sends them to the wrong place. The non-streaming 502 now names the status and
the reason when one was captured, and keeps the old wording only when the backend genuinely could not be
reached.

### Changed — `src/lib/jsonlite.cyr` (pure lift, no behaviour change)

`_json_value_pos`, `_json_extract_str`, `_json_extract_int`, `_json_obj_str_field` and `_json_obj_end`
moved out of `provider.cyr` into their own module. They were never provider logic — `handlers.cyr` reads
with them too — and the move has a concrete payoff: **`provider.cyr` reaches `_router`, a global defined
in `main.cyr`, so it cannot be included by the test binary.** Anything living there is untestable, and
the new upstream-error surface is exactly the kind of untrusted-byte parsing that must be tested against
the real implementation rather than a mirror.

`src/lib/provider_err.cyr` is new for the same reason: it holds the capture, the bounding and sanitising,
the log/client renderings, the SSE error-frame builder and the two error recognisers — all reachable from
`tests/hoosh.tcyr`.

### Tests

25 new assertions, covering: capture of status + message; the pre-capture fallback wording; bounding an
over-long provider body; flattening newlines and tabs to spaces; `reset` making a stale error
unattributable to a later request; the SSE error frame's shape (`error` object, upstream status,
`finish_reason:"error"`, model named); JSON-escaping of a provider message containing quotes and
backslashes; and error **recognition** — including the two negative cases that matter, a content delta
whose text says "error" and a keep-alive ping.

### Note for consumers

This release makes hoosh **report** upstream failures instead of hiding them; it does not make them stop
happening. A client that previously saw an empty completion will now see an `error` chunk in the stream
(or a 502 naming the status), which is the outcome it needed in order to react.

⚠ Separately, and **not** fixed here: the vendored `lib/sandhi.cyr` in this toolchain snapshot silently
drops whole SSE events at read boundaries, which is what caused hoosh to forward a tool call whose
`id`/`name` frame never arrived. Fixed upstream in **sandhi 1.9.15**; it reaches hoosh when a cyrius
release re-vendors the bundle. Filed as
`cyrius/docs/development/issues/2026-08-27-revendor-sandhi-1.9.15-sse-event-loss.md`.

## [2.6.4] — 2026-08-25

**Toolchain and CI maintenance.** No behaviour change to the gateway; every gate now actually gates.
Suite **689/689**, unchanged.

### Fixed

- **The CI format gate, which could never pass.** It diffed `cyrius fmt`'s stdout against each file, but
  `cyrius fmt <file>` rewrites the file in place and prints nothing — so the diff compared an empty
  stream against every file and reported drift unconditionally, including for correctly-formatted ones.
  A gate that always fails carries the same information as one that never runs. Now uses `--check`, the
  flag built for it (exit 1 on real drift, names the file, no stdout by design), matching the shape
  daimon and kavach settled on.

  With a gate that can answer the question, the real drift was **10 of 34 files** — `src/main.cyr`,
  `src/lib/{audit,batch,compression,handlers,health,mcp,provider,retry}.cyr` and `tests/hoosh.tcyr` —
  all continuation-line indentation, reformatted here. The gate is green on its own terms now.

- **`BACKEND_COUNT` no longer exists in ai-hwaccel.** The 2.3.19 bump below namespaced the library's
  public constants (`BACKEND_COUNT` -> `AIHW_BACKEND_COUNT`), and `src/lib/hardware.cyr`'s
  `_hw_backend_from_name` still referenced the bare name, so the build failed outright. Renamed at the
  single use site.

### Changed

- **Toolchain pin `6.5.27` -> `6.5.35`**, matching daimon, bote and thoth; hoosh was two pins behind.
  `cyrius lib sync --full` re-synced the 108-file stdlib snapshot.
- **`[deps.ai-hwaccel]` `2.3.17` -> `2.3.19`.** Verified the bump took by reading the vendored
  `lib/ai-hwaccel.cyr` header rather than trusting the manifest. See the `BACKEND_COUNT` fix above — this
  bump is what surfaced it.

### Notes

- Every CI gate re-run locally and green: include-graph audit (34 deps, 0 untrusted, 0 missing), format,
  lint (src scope), build, ELF, **689/689 tests**, dependency policy (0 violations), symbol coverage
  (163/481 = 33%, floor 30%), security scan, all four fuzz targets, and the mandatory benchmark gate
  (25 benchmarks, 1091 CSV rows).
- ⚠ `tests/hoosh.tcyr` and `tests/hoosh.bcyr` carry **98 line-length warnings** between them. These are
  pre-existing and unchanged by this cut, and they do **not** gate — the lint step scopes to
  `src/main.cyr` and `src/lib/*.cyr`. Recorded so the count is not mistaken for new drift later.

## [2.6.3] — 2026-08-18

### Changed

- **`[deps.ai-hwaccel]` `2.3.16` -> `2.3.17`.** Held back from the 2.6.2 cut because ai-hwaccel
  2.3.17 was not published yet; picked up now. Verified the bump took — vendored
  `lib/ai-hwaccel.cyr` moved to `2.3.17`. Suite **689/689**, unchanged.

## [2.6.2] — 2026-08-17

### Changed

- **Cyrius pin `6.5.2` -> `6.5.27`** (2026-08-17, ecosystem-wide ML/AI-arc realign ahead of
  the arc reopening). `cyrius lib sync --full` re-vendored the whole version-matched stdlib
  snapshot, clearing the toolchain-drift and `./lib/ shadows version-pinned` warnings.
  Suite green at the new pin, identical to the pre-bump baseline.

## [2.6.1] — 2026-08-13

**Locally-served models can use tools.** They never could: on the streaming path hoosh dropped the
caller's `tools` array before it reached the backend, so a local model was asked with no tools and
answered — correctly — that it had none. Reported by thoth, whose users read it as "only a few models
can actually use tools, the rest just write code about them". Remote providers were never affected,
which is exactly why it looked like a model-quality problem rather than a gateway one.

**Patch, not minor:** no API is added or changed. Two internal signatures grow a parameter, and one
provider's request and response shapes are now translated the way the other three already were;
every externally visible contract is the one that was already documented.

### Fixed

- **The local streaming path forwards `tools`.** `handle_chat_stream`'s remote branch has always
  passed `tools_raw/tools_len` to `_remote_stream_body`; the local branch called
  `retry_forward_stream(cfg, route, model, msg_raw, msg_len, gen)` — a signature with **no tools
  parameter** — and `provider_forward_stream` hardcoded `0, 0` into `_build_chat_body_raw_stream`.
  Both signatures now carry `tools_raw/tools_len` and the handler passes them. Since clients stream
  by default (thoth's `[hoosh].stream` defaults on), this was the **default** local experience.
  Proven by holding one request constant and flipping only `stream`: blocking returned a clean
  `list_dir` tool call, streaming returned a ```` ```json ```` code block describing one.

- **Ollama's tool-call shape is converted, not forwarded raw** (`_ollama_tool_calls`). Ollama nests
  `{"function":{"index":0,"name":…,"arguments":{…}}}`, where OpenAI specifies
  `{"id":…,"type":"function","function":{"name":…,"arguments":"<JSON string>"}}`. Three differences:
  `arguments` is a JSON **object** rather than a JSON-encoded string, `index` is not an OpenAI field,
  and older builds omit `id`. Anthropic and Google were already converted here (`_anthropic_tool_calls`,
  `_gemini_tool_calls`); Ollama was routed through `_extract_openai_tool_calls`, which is a **raw span
  copy** — so it was the one provider whose non-conforming shape reached clients untouched. A client
  reading `arguments` as the spec's string got nothing and silently dropped every argument: tools whose
  parameters are optional appeared to work, while required-argument tools failed on every round. The
  converter is tolerant on input — an already-encoded string is emitted verbatim rather than escaped a
  second time, and a missing `id` is synthesised as `call_<n>`, mirroring the Gemini converter.

- **The local streaming reader emits tool-call deltas.** It only ever extracted `content`
  (`ollama_extract_text`), so a locally streamed tool call was parsed off the wire and discarded —
  fixing the two items above without this one would have forwarded the tools and then swallowed the
  answer. Ollama chunks convert through `_ollama_tool_calls`; a local OpenAI-compatible backend's
  deltas go through `_extract_openai_tool_calls`, the same decode the remote branch already does.

- **Tool-continuation requests are converted back to Ollama's shape** (`_ollama_messages_convert`,
  applied by `_sb_add_messages` on both the blocking and streaming builders). This is the outbound
  mirror of the item above, and without it the first three fixes still leave the agentic loop
  **unable to close**: the client echoes back the assistant turn hoosh handed it, whose `arguments`
  is now a JSON string, and Ollama's `/api/chat` rejects that outright —
  `{"error":"Value looks like object, but can't find closing '}' symbol"}` — so round two returns
  nothing and the caller reports an empty completion. Only assistant turns carrying `tool_calls` are
  rewritten, and only their `arguments` (string → object); `role:"tool"` results already match
  Ollama's shape, and every other message is copied span-verbatim. When no message carries
  `tool_calls` the converter returns 0 and the raw span is forwarded untouched, so ordinary
  conversations are not re-serialised. `_build_anthropic_body_x` has always done this translation
  for Anthropic; Ollama was the provider it was never done for.

### Added

- **`_emit_tool_call_encoded`** — the `_emit_tool_call` envelope for an argument span that is already
  an encoded JSON string. Running those bytes through `_json_esc_span` would escape the escapes and
  double-encode the arguments.
- **10 assertions** (`tests/hoosh.tcyr`, 679 → 689). Inbound: object arguments → encoded string with
  `type:"function"` added and `index` dropped; empty `{}` arguments preserved rather than dropping the
  call; missing `id` synthesised; already-encoded string arguments passed through undamaged; two calls
  in one round converted in order; a content-only chunk yielding 0 so the streaming reader emits
  nothing. Outbound: `arguments` string unescaped to an object with user and `role:"tool"` turns left
  verbatim; no `tool_calls` → 0; an empty argument string → `{}`; an argument that is already an object
  passed through. The object payload is captured verbatim from `qwen3.5:9b` through hoosh.

### Verified rather than assumed

- **The agentic loop closes on a local model, end to end.** `thoth 0.38.5 → hoosh 2.6.1 →
  qwen3.5:9b` on a Jetson Orin, streaming (the default), listing a real directory: round one calls
  `list_dir {"path":"."}` and gets 241 bytes back, round two answers with the actual contents. Before
  this release the same run reported `no completion returned`. `llama3.1:8b` likewise answers from a
  real listing. `nemotron-3-nano:4b` also completes both rounds, though it passes `{"path":"/"}` —
  an absolute path the caller's jail refuses — and then reports that refusal accurately; the
  transport is doing its job there and the argument choice is the model's.
- **No benchmark regression.** The two rows that cross the +20% watch threshold against the 2.6.0
  baseline are sub-100ns timers whose **absolute** deltas are 3ns and 11ns
  (`estimate_tokens_per_provider` 8 → 11, `latency_bucket_find` 51 → 62), and same-build variance was
  measured **larger than the threshold itself**: two consecutive runs of one unchanged binary moved
  `route_select_20_providers` 1108 → 906 ns (−18.2%) and `estimate_tokens_per_provider` 10 → 8 ns
  (−20.0%). Nothing this release touches is on a benchmarked path — the conversions run only on the
  Ollama request/response boundary. 25 benchmarks recorded.
- **The suite's mirrored copies match their sources byte-for-byte.** `tests/hoosh.tcyr` cannot include
  `provider.cyr`, so it mirrors the pure functions; `_ollama_tool_calls` and `_emit_tool_call_encoded`
  were diffed against the real definitions rather than eyeballed.
- Clean at the gates: `cyrius fmt --check`, `cyrius lint` (0 warn lines), `cyrius vet` (34 deps, 0
  untrusted, 0 missing), `cyrius deny` (0 violations). 685 + 1 tests passing.

### Known issues

- **`_extract_tools` matches the first `"tools":` byte-run in the request body**, including one that
  appears inside a message string (JSON-escaped `\"tools\":` still contains the needle at the quote).
  A conversation that quotes that exact text ahead of the real key yields a bogus span. Pre-existing,
  not introduced here, and unrelated to the tool-forwarding fix — a caller would have to quote
  `"tools":` verbatim in a message to trigger it.

## [2.6.0] — 2026-07-30

**A response now says what happened to it.** Until this release a client could not tell a cache hit
from a fresh inference — the two were byte-identical, `"id"` included — and the cost hoosh had
already computed never left the process. Reported by agnosai while porting its orchestrator to
Cyrius: it had to re-implement hoosh's pricing table just to attribute per-request spend, and it
re-billed every server-side cache hit because nothing on the wire said otherwise. Both facts were
sitting one function call away from the response builder. Plus the toolchain and dependency refresh.

**Minor, not patch:** the fix is additive public API — a new response header and two new `usage`
fields — so SemVer puts it at 2.6.0 even though it resolves a defect report.

### Added
- **`X-Hoosh-Cache` response header on `/v1/chat/completions`** — `HIT` (exact-key), `SEMANTIC`
  (similarity), or `MISS` (a real forward). A header rather than a body field so the payload stays
  OpenAI-compatible. `http_respond_h` carries one optional extra header; the ten existing
  `http_respond` call sites are untouched and byte-identical, which a test pins.
- **`usage.cost_micro_usd` and `usage.provider`** on chat completions. Integer micro-USD, the same
  unit `estimate_cost_micro` returns, so a consumer never converts and never rounds. `provider` is
  the **real serving route** — which matters because a model-name guess is wrong whenever DLP
  re-routes a confidential request to a local provider, and hoosh bills $0 for that while the
  consumer prices it as remote. Both fields are additive inside the existing `usage` object; an
  OpenAI-shaped client ignores unknown fields.
- Cache-warming entries are priced the same way, so a hit on a warmed entry is indistinguishable
  from a hit on a normally-served one.

### Changed
- **`cost_record` returns the micro-USD it accumulated** instead of 0. The response body reports
  that same value rather than a second call to `estimate_cost_micro` that merely ought to agree —
  so a client summing per-response costs reconciles with `/v1/costs` **by construction**, and cannot
  drift if the pricing table changes between the two calls.
- **Toolchain: cyrius `6.4.62` → `6.5.2`.** No source changes were needed. The jump removes
  `bayan_json_v_parse_str`/`json_v_parse_str` (renamed `_parse_buf`, because `X_str` is a reserved
  overload slot) and the whole `regex_*` surface; hoosh uses none of them, verified by diffing every
  public symbol across the two lib trees before compiling.
- **Dependencies:** ai-hwaccel `2.3.14` → `2.3.16`, vendored bote-core `2.7.7` → `3.1.4`, vendored
  majra `2.5.0` → `2.5.3`. ai-hwaccel 2.3.16 fixes a `json_v_parse_str` call that bayan 1.3.0
  removed — found while doing this update, filed upstream, and fixed there. It was harmless in this
  binary (the function has no callers here, so the symbol was dead), but it meant the build carried
  a permanent `warning: undefined function`. **The build is now warning-free.** bote 3.x's breaking change is in the `bote-streamable` binary's session
  lifecycle; the `[lib.core]` bundle hoosh consumes grew additively on the 2.0 handler ABI.
- `data/models.json` re-synced from ai-hwaccel 2.3.15 (26 models). Note it ships as a
  `{"models":[…]}` wrapper upstream and must be **unwrapped to a top-level array** here — the
  `hardware_data_files` test caught the wrapped form immediately.

### Tests
- 663 → **679 assertions**. The `X-Hoosh-Cache` path is tested against the **real**
  `http_respond_h` over a pipe rather than a mirror, including that the no-extra-header form differs
  from the labelled one by exactly that header line plus its CRLF and nothing else.

### Performance
- No regressions across the toolchain jump — 25 benchmarks compared before and after, all within
  noise, with the cache path slightly improved (`cache_get_miss` −20%, `cache_insert` −14%).
- The per-response cost costs nothing: `cost_record` already computed it, and now returns it instead
  of recomputing.

## [2.5.11] — 2026-07-23

**Security & hardening sweep — closes the rust-old parity arc.** A P(-1) pass over the whole gateway now that
the parity bands have settled. Two of these are the kind of defect that only shows up in production: a memory
leak proportional to requests served, and a syscall that dominated every hot path.

### Fixed
- **RSS grew 64 KiB for every request served.** `_handle_conn` called `alloc(64 KiB)` per connection against an
  arena that **never frees**, so a gateway's memory grew with total requests handled — ~128 MB per 2000
  requests, and an eventual OOM on any long-running deployment regardless of load shape. Nothing was wrong with
  the code except that it treated a bump allocator like `malloc`. Now one buffer per thread, allocated on that
  thread's first request and reused. Measured over 2000 requests: **2.0 MB of growth, down from ~128 MB (62×)**.
  Reuse is safe because the buffer's contents do not outlive the call — the sync-batch path blocks in its
  work-stealing barrier inside `_handle_conn`, and the async path deep-copies every item in `batch_submit`
  precisely so its slices own their bytes.
- **`clock_now_ms()` is a 1.351 µs syscall and it dominated the hot paths.** Cyrius has no vDSO route, so it
  bottoms out at raw `syscall(228)`. It was ~98% of a cache hit and ~96% of a rate-limit check, and hoosh calls
  it on every cache read, twice per insert, and once per rate-limit decision. A ticker thread now refreshes a
  global every 5 ms and the cache and rate limiter read that instead:

  | benchmark | before | after | |
  |---|---|---|---|
  | `cache_get_hit` | 1487 ns | **113 ns** | 13× |
  | `cache_insert` | 1447 ns | **95 ns** | 15× |
  | `rate_limit_check` | 1409 ns | **8 ns** | 176× |

  Accuracy becomes ±5 ms, which is ample for cache TTL (300 s), LRU ordering and per-minute rate limiting.
  Deliberately **not** used for audit timestamps, per-forward latency, or trace ids — those keep the real clock.
- **An abandoned SSE stream kept draining the provider.** 2.4.13 stopped a disconnected client from *killing*
  the gateway, and flagged the follow-up: with SIGPIPE ignored, writes to a dead socket merely return EPIPE and
  the stream loop carried on pulling the whole remaining response from the backend. That spends a worker and
  real provider tokens on a client that already left, and an unauthenticated caller can trigger it in a loop.
  `http_sse_event` now reports the failed write and the local streaming loop stops reading upstream.
- **`ratelimit_new`/`ratelimit_check` collided with the vendored majra distlib**, which exports the same names
  with different arities. hoosh included later so hoosh won, and every build printed "duplicate fn (last
  definition wins)". Harmless in practice — majra never calls its own — but a shadow resolved by include order
  is a trap, and the noise masked any real duplicate warning. hoosh's are now `hoosh_ratelimit_*`; the build is
  clean.
- **`make_crlf()` allocated 3 bytes on every call**, several times per response, forever. Now a
  process-lifetime constant.

- **Signal setup now precedes every `thread_create`.** `sigprocmask` masks only the *calling* thread, and threads
  inherit the mask as it stands when they are created — so any thread spawned before `signals_init()` does not
  block SIGINT/SIGTERM, the kernel may deliver the signal there, and it takes the default action: the process
  dies with no drain and no shutdown logs. 2.5.8 got this right for the worker pool and documented why; this
  release then inserted the coarse-clock ticker *above* that call and broke graceful shutdown, which the
  end-to-end regression check caught before release. The OTLP exporter thread had the same exposure whenever
  telemetry was enabled — a latent bug since 2.5.8. `signals_init()` is now the first thing in the init
  sequence, with a comment stating that nothing creating a thread may move above it.

### Added
- **A startup security warning when the gateway is reachable off-box with no auth.** Auth is fail-open by
  design when no tokens are configured — correct for a loopback sidecar, and what rust-old did — but it stops
  being correct the moment the listener is not on loopback. Binding a non-loopback address with an empty
  `[[auth]]` token set now warns on both the log and stderr rather than leaving it to be discovered.

### Known, not fixed
- **~1 KiB per request still accumulates.** After the 64 KiB fix, growth is ~1 KiB/request — response string
  building (`str_builder`, `to_cstr`, path parsing) against the same never-freeing arena. At 100 req/s that is
  still ~360 MB/hour. Point fixes will not close it; it needs a per-request arena with mark/release, and
  `alloc_reset` cannot simply be called per request because the response cache, audit chain, cost records,
  routes and health records all live in the same arena and must survive. Filed as the next piece of work rather
  than patched over. The measurement above (2.0 MB / 2000 requests) is the honest current number.

Gates: 663 tests, 25 benchmarks, 4 fuzz targets; fmt/lint/vet/deny clean and now genuinely warning-free.

## [2.5.10] — 2026-07-23

**Scaffolding parity — the tenth and final feature band of the rust-old parity closeout arc.** The gates,
fuzzers and benchmarks the Rust project had and the port did not. Only the 2.5.11 hardening sweep remains.
See [the parity review](docs/development/rust-old-parity-review.md).

### Added
- **`cyrius deny` in CI.** CLAUDE.md has listed it in the cleanliness check since the port, but the workflow
  never ran it — so a dependency violating project policy could land with a green build.
- **A coverage gate** (`scripts/coverage.sh`, CI floor 30%, currently 32%). This is **symbol** coverage, not
  line coverage, and the distinction is load-bearing: hoosh's tests are a self-contained mirror that
  re-implements the logic under test rather than linking `src/` (`src/main.cyr` is a program, not a library —
  including it drags in the gateway's globals and threads). No test executes a line of `src/`, so a line number
  would be a flat 0%. The script instead asserts every function defined in `src/` is at least *referenced* by
  the suites — a floor against adding code nothing watches. It states its own limitation plainly: it cannot
  catch **mirror drift**, where `src/` and its mirror diverge while both stay internally consistent, which has
  bitten twice (2.5.6 pricing, 2.5.7 audit linkage — in both the mirror was right and `src/` was wrong).
- **Two fuzz targets**, porting rust-old's. `fuzz/inference_request.fcyr` hammers the raw byte scanners that
  read `temperature`/`top_p`/`max_tokens`/`reasoning_effort`/`stream` and the message count — the ones that
  exist *because* `json_parse` cannot reach a key after the nested `messages` array, so they walk
  attacker-controlled bytes with no parser backstop. `fuzz/message_content.fcyr` covers untagged
  string-or-parts `content`, the whitespace-tolerant role matcher and the boundary scan. Both run every
  truncation of a hand-built corpus (unterminated strings, lone trailing escapes, unbalanced brackets,
  overflowing digit runs) plus random and structure-biased bytes.
- **8 benchmark cases** for the work every request pays for, which previously had no numbers at all:
  `auth_verify_token` / `_wrong_late` / `_wrong_early`, `rate_limit_check`, `cost_record_known_model` /
  `_local_free`, `audit_record_sign`, `event_publish_ring`. Suite is now 25 benchmarks.
- **`scripts/bench-live.sh`** — the opt-in port of `benches/e2e.rs` + `benches/live_providers.rs`. Deliberately
  outside CI, as in rust-old: it needs a live gateway and backend, so it measures a machine and a model rather
  than the code. Its header states that each iteration forks a `curl` and the ~5 ms floor is the harness, not
  the gateway — use it for relative comparisons only.
- **DLP `custom_patterns`** via `[[dlp_pattern]]` sections (`name`, `pattern`, `level`). Case-insensitive
  **literal** matching, not regex — hoosh has no regex engine (rust-old's `dlp` feature pulled in the `regex`
  crate), and compiling operator-supplied patterns would put a backtracking matcher fed by config on the
  request path, i.e. a ReDoS surface. A literal match is the capability that ports safely, and is what covers
  the actual use case: an internal codename or customer-id prefix that only this deployment knows is
  sensitive. Custom patterns can raise the classification to `restricted`, same as a built-in.

### Measured
- **The auth compare is constant-time in the way that matters.** A same-length token differing in the **first**
  byte and one differing in the **last** both cost 100 ns, against 101 ns for a match — no prefix oracle, so an
  attacker cannot recover a token byte by byte. (An earlier draft benched a 25-byte token against a 24-byte
  one and read 101 vs 52 ns; that is the length short-circuit, which is unavoidable and not secret. The
  benchmark now uses equal-length inputs so it measures the property it claims to.)
- **`rate_limit_check` costs 1.4 µs**, essentially all of it the `clock_now_ms` syscall — more evidence for the
  2.5.11 item on replacing it with a vDSO/coarse clock.

### Upstream
- **Filed: `cyrius coverage` reports on the vendored stdlib, not the local repo.** Run from a project root it
  enumerates `lib/` — hoosh's 465 functions never appear and the summary reads "Functions: 0/1097", which is
  both wrong for the project and permanently near zero for any project. Unusable as a gate, and misleading if
  trusted. Filed at `cyrius/docs/development/issues/2026-07-23-hoosh-coverage-reports-stdlib-not-local-repo.md`
  proposing a local-repo default, `--full` for today's behaviour, and `--min <pct>` for CI.

Gates: 663 tests (was 652), 25 benchmarks (was 17), 4 fuzz targets (was 2); fmt/lint/vet/deny clean.
(`pool_available` and `route_round_robin_10` show single high readings in the CSV run; both measured stable at
their previous values across three consecutive repeats afterwards — transient, and neither was touched.)

## [2.5.9] — 2026-07-23

**Hardware planning closeout — the ninth band of the rust-old parity closeout arc.**
See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **ai-hwaccel's threaded detector corrupts the registry — hoosh now uses the serial one.**
  `registry_detect_threaded` calls its post-passes with the wrong argument: `detect_interconnects(r, …)`,
  `detect_storage(r)` and `detect_environment(r)` pass the **registry** where a `system_io` is expected. The
  two layouts overlap but differ — `reg {profiles@0, warnings@8, system_io@16, schema@24}` vs
  `sio {interconnects@0, storage@8, environment@16}` — so storage devices get pushed into `reg.warnings`,
  `reg.system_io` is overwritten with a `runtime_env`, and on a machine with NVLink or InfiniBand the detected
  interconnects are pushed into **`reg.profiles`, the device list** — corrupting `hw_device_count()` and
  `reg_total_accel_memory()`, which reads `profile_memory_bytes()` off whatever is in that vec. That is
  precisely the multi-GPU case where placement decisions matter. Measured here: threaded reports 10 "warnings"
  where serial reports 8 (the extra two are storage structs). Serial costs 34 ms vs 20 ms — not a trade worth
  making — and is the only path whose `reg_system_io()` the new topology accessors can read.
  **This is an upstream ai-hwaccel bug; revert to the threaded detector once it is fixed.** 2.4.2 adopted the
  threaded detector as "verified byte-identical to serial", but that check compared profile counts on a machine
  with no interconnects and could not see it.
- **`-1` ("driver did not report") was being treated as a value.** `profile_mem_used` returns `-1` for unknown;
  subtracting it from a VRAM total *added* a byte, and emitting it as JSON would make a consumer graph a spike
  at minus one. Unknown now contributes 0 to arithmetic and serializes as `null`.

### Added
- **`POST /v1/hardware/simulate`** — what-if capacity planning: add or remove devices and see how placement
  changes, without touching the real registry. Returns `{original, simulated}` with accelerator memory, system
  memory, required memory, fit, and the device span. Validated (`model_params > 0`, ≤ 64 added devices,
  non-zero `memory_bytes`), matching rust-old (`handlers.rs:1099-1183`). Never ported and never deferred.
- **`GET /v1/hardware/telemetry`** — per-device utilization, memory used/free and temperature, plus
  interconnect topology (`has_fast_interconnect`, max bandwidth, full `system_io`) and available VRAM. With
  `[hardware] refresh_interval_secs` set, these track the machine over time instead of being frozen at startup.
  Exposes rust-old's `system_io()`, `has_fast_interconnect()`, `gpu_telemetry()` and `runtime_environment()`.
- **Hardware-aware routing** (`router_select_hw`, rust-old `Router::select_with_hardware`). When a model's size
  is readable from its name and it does not fit the VRAM actually free, local routes are deprioritized in
  favour of a remote provider that can serve it — instead of routing to a backend that will OOM. Unknown-size
  models and undetected hardware fall through to plain selection; if nothing remote can serve the model the
  local route is still tried, since a real attempt beats a guess.

### Changed
- **`/v1/hardware/simulate`'s `fits` is an accelerator question.** System RAM is reported separately as
  `system_memory_bytes` rather than folded into the total — summing both made a 70B model "fit" a box with
  7.5 GB of VRAM and 64 GB of RAM, which is not what a GPU-placement simulator is being asked.
- **Shard span is counted largest-device-first.** Counting in registry order overstated it: a 70B model on a
  3 GiB + 4 GiB + 24 GiB + 24 GiB set reported 4 devices when the two 24 GiB cards alone suffice. Now 2.
- **Breaking URL change, documented late**: rust-old's `POST /v1/hardware/format` became
  `POST /v1/hardware/model-format` in **2.4.1**, and the payload changed from a file path to raw model bytes.
  The redesign is the safer one (no server-side path traversal) and stands, but it was never announced —
  recording it here for anyone porting from the Rust-era API.

Gates: 652 tests (was 644), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.8] — 2026-07-23

**CLI & process lifecycle — the eighth band of the rust-old parity closeout arc.** hoosh can now be stopped
cleanly, reloaded in place, pointed at a running gateway, and configured from the command line.
See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **`sys_exit` is thread exit, not process exit — a clean shutdown left hoosh running.** The stdlib's
  `sys_exit` issues `SYS_EXIT` (60), which retires only the *calling thread* despite its "terminate process"
  comment. With 7 pool workers plus the health/OTLP/signal threads alive, returning from `main` left a
  process that had logged "shutdown: complete" and then sat there. It was invisible before 2.5.8 because the
  accept loop never returned. hoosh now calls `exit_group` via a local `hoosh_exit_process` — the single place
  a bare syscall appears, pending an upstream `sys_exit_group` wrapper.
- **`crypto_tls_main_init()` was inside `cmd_serve`, so every other command ran with no thread-local block.**
  Anything reaching `trace_current_or_new()` — which every outgoing request header builder does — dereferenced
  an uninstalled TLS block and **segfaulted**. Latent until this release gave the CLI a reason to make HTTP
  requests; `models --server` crashed with SIGSEGV on the first successful connect. Hoisted into `main` (it is
  CAS-guarded upstream, so `cmd_serve` calling it again is a no-op).
- **An unknown token pool silently skipped budgeting.** A request naming a pool that does not exist got a null
  pool and fell straight through, so `"pool": "typo"` was a way to bypass the token budget entirely, and a
  renamed pool stopped being enforced instead of failing loudly. Now `400 Token pool 'X' does not exist`.
- **A config that exists but fails to parse is now fatal.** Falling back to defaults meant a typo'd config
  started a gateway with no providers, no auth and no budgets — looking healthy while serving nothing the
  operator asked for. A *missing* config is still fine.

### Added
- **Graceful shutdown on SIGINT/SIGTERM.** The listener closes, then in-flight work is allowed to finish
  (bounded by a 5 s drain) before the process exits. Implemented with **signalfd** rather than a signal
  handler: a handler would need an SA_RESTORER trampoline on Linux x86-64 and could only safely set a flag,
  whereas config reload must take `_chat_lock`, which is not async-signal-safe. Blocking the signals
  process-wide and reading them from a dedicated thread turns delivery into an ordinary blocking read, so the
  work runs in normal thread context. If `signalfd` is unavailable the mask is **unblocked** again — leaving
  signals blocked with no reader would make hoosh unkillable by anything but SIGKILL.
  Verified: a request mid-flight against a deliberately slow backend completes during the drain.
- **SIGHUP reloads the config** in place (rust-old `server/mod.rs:390-401`). A failed reload keeps the running
  config and logs a warning rather than leaving the gateway half-configured.
- **CLI flags** — `serve -p/--port <n>`, `--bind <addr>`, `-c/--config <path>`. Supports `--name value`,
  `--name=value` and short forms; the pre-2.5.8 bare positional port still works.
- **`--server <url>` on `models` / `health` / `infer`** — query a running gateway over HTTP instead of
  re-reading local config, which is what rust-old's CLI did. `health --server` exits non-zero if any provider
  is unhealthy. **`infer --stream`** prints SSE deltas as they arrive.
- **`HOOSH_LOG`** sets the log level (`fatal|error|warn|info|debug|trace`) — the `RUST_LOG` equivalent. Read
  before config load so config problems are logged at the requested level.

### Changed
- The work queue tracks **in-flight** jobs, not just queued ones. The shutdown drain waits on queued +
  in-flight; queue depth alone is zero the moment a worker picks a request up, so draining on depth cut live
  requests off mid-answer. The counter is incremented inside `wq_pop` under the lock it already holds, so the
  hot path pays no extra locking; the batch work-stealing barrier calls `wq_done()` too, or the count would
  leak and every later drain would hit its timeout.

Gates: 644 tests (was 631), 17 benchmarks; fmt/lint/vet/deny clean. (`estimate_tokens_per_provider` oscillates
7↔8 ns run to run — 1 ns of timer quantization on an 8 ns benchmark, not a regression.)

## [2.5.7] — 2026-07-23

**Config-reader closeout — the seventh band of the rust-old parity closeout arc.** Config keys that existed
with no reader now have one, and two security/correctness bugs surfaced while verifying them.
See [the parity review](docs/development/rust-old-parity-review.md).

### Security
- **The audit signing key was compiled into the binary.** The HMAC chain was created unconditionally with the
  literal key `"hoosh-audit-key"`, so anyone holding a hoosh binary could forge entries or re-sign a doctored
  chain — the signature proved nothing about provenance. The key now comes from `[audit] signing_key`, which is
  `$NAME`-expandable so the secret lives in the environment; when unset, a random 32-byte per-process key is
  generated and a startup warning is logged. Auditing is now **off by default**, matching rust-old.
- **`audit_verify` could not detect a deleted record.** It recomputed each entry's own hash and HMAC — both of
  which an excised entry leaves undisturbed, because the surviving entries are byte-identical. Removing a
  record from the middle produced a chain that verified clean, which defeats the purpose of chaining. It now
  walks the linkage: each entry's `previous_hash` must equal the prior entry's hash, and the first must be
  GENESIS (rust-old `audit.rs:186-195`).

### Fixed
- **Compaction dropped the system prompt for ordinary clients.** The system-message detector matched the
  compact literal `"role":"system"`, but virtually every JSON encoder emits `"role": "system"` **with a space**
  — Python's `json.dumps` does by default. So for realistically-encoded requests no system message was detected
  and compaction discarded it like any other message. The matcher is now whitespace-tolerant, as are the
  message-boundary scans. Same class of bug the response extractors fixed in 2.2.0; it survived here because
  nothing exercised compaction with realistic input. Verified: a 13-message request that came back with 6
  messages (system missing) now returns 7.
- **Compaction could strip a conversation to nothing.** If a single recent message exceeded the remaining
  budget, the retention loop kept zero messages and forwarded the system prompt alone — or an empty array.
  The retained tail is now floored at `keep_last_messages`, and unconditionally at one non-system message.

### Added
- **`[context]`** — `enabled`, `compaction_threshold` (default 0.8), `keep_last_messages` (default 10).
- **`[audit]`** — `enabled` (default false), `signing_key`, `max_entries`.
- **`[retry] jitter_factor`** (default 0.5) — the retry jitter spread was hardcoded at 25%.
- **`inference.error` audit entry** on the provider-failure path. Recording only successes made the chain a
  record of what worked, which is the less useful half when reconstructing an incident.

### Changed
- **The compaction budget now derives from the model's context window × `compaction_threshold`.** It was
  `est_tokens * 2` — a heuristic off the requested *output* size, which is unrelated to how much input the
  model accepts, so a 200k-context model was compacted as aggressively as an 8k one. Models absent from the
  catalog keep the old heuristic rather than an invented limit.

### Added — `[hardware]`
All four keys, each with the code that acts on it:
- **`disabled_backends`** — a list of ai-hwaccel detection backends to skip (`"cuda"`, `"rocm"`, `"vulkan"`, …),
  applied as a builder mask. An unrecognized name logs a warning rather than silently disabling nothing.
- **`vram_reserve_bytes`** — VRAM held back from placement. New `hw_available_vram()` = total accelerator
  memory − what devices report in use − the reserve. **Placement previously compared against TOTAL accelerator
  memory**, which assumes an idle, exclusively-owned card; on a GPU already running a display server or another
  model that overcommits, and the load fails at the backend instead of being planned around here.
  `/v1/hardware/placement` now reports `available_vram_bytes`, `vram_reserve_bytes` and `fits_available`
  alongside the existing `fits_single_device`. Verified: a 6 GB reserve drops available VRAM by exactly 6 GB
  and flips `fits_available` to false while `fits_single_device` stays true.
- **`refresh_interval_secs`** — background re-detection thread; `0` (default) keeps the previous
  detect-once-at-startup behavior. Verified: 4 refreshes in a 10 s window at a 2 s interval.
- **`cache_ttl_secs`** — minimum snapshot age before a wake actually re-probes. Detection shells out to
  `nvidia-smi`/`vulkaninfo`, so this is what stops a short refresh interval from spawning probes constantly.
  Verified: interval 1 s with TTL 3600 s wakes repeatedly and re-probes zero times.

Config load also moved **before** `hw_init()` — `disabled_backends` has to be known before detection runs, or
the backends it names get probed once anyway.

Gates: 631 tests (was 600), 17 benchmarks; fmt/lint/vet/deny clean. (`mcp_tools_list` and
`estimate_tokens_per_provider` show >10% swings in the CSV in both directions across consecutive runs — both
are untouched by this release and sit within run-to-run variance; `mcp_tools_list` measured 3.92–4.73 µs across
repeats.)

## [2.5.6] — 2026-07-23

**Cost accounting — the sixth band of the rust-old parity closeout arc.** hoosh could price a request but never
remembered one, so `/v1/costs` could not answer the question the endpoint exists for.
See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **Local providers were billed for hosted-model names.** `pricing_lookup` checked `provider_is_local` only in
  its *fallback*, which is reached only for models absent from the price table. The table is keyed by model
  NAME, and hosted-model names are routinely served locally — so `llama-3.3-70b` on Ollama matched the
  Groq-hosted row and accrued 590/790 per million tokens, real money for tokens generated on your own GPU.
  Exactly the realistic cases were mispriced. The local check now runs first, before the table.
  Found while verifying the new accumulation (a local-only test config reported $0.0225 of spend). The test
  suite's pricing mirror had checked local-first all along; production had diverged from it.
- **Nothing accumulated cost.** `/v1/costs` reported global token counters plus a bare listing of configured
  routes with no cost attached — it could say how many tokens hoosh had processed in total, but not what any
  one provider had cost.

### Added
- **A cost tracker** (`pricing.cyr`), accumulating per `(provider, base_url)` on the successful-inference path.
  Port of rust-old's `CostTracker` (`cost/mod.rs:116-127`). Records are matched by a linear scan over a record
  vec rather than a keyed map: building a `provider:base_url` key string per request would allocate on the hot
  path against an arena that never frees, and the scan is over the route count. `base_url` is compared by
  content, so totals survive an `/v1/admin/reload`.
- **`/v1/costs` now returns `records`** — each with `provider`, `base_url`, `is_local`, `total_input_tokens`,
  `total_output_tokens`, `request_count`, `total_cost_micro_usd` and `total_cost_usd` — plus
  `total_cost_micro_usd`/`total_cost_usd` grand totals. The existing `total_requests` /
  `total_prompt_tokens` / `total_completion_tokens` aggregates are unchanged.
- **`admin.costs_reset` audit entry at `warn`** on `POST /v1/costs/reset`, plus a warn log. Clearing spend
  history destroys billing evidence, so it belongs in the chain. `POST /v1/costs/reset` now clears the cost
  records as well as the metrics counters.

### Changed
- **`/v1/costs` replaced its `providers` array with `records`.** The old array listed configured routes
  (`provider`, `base_url`, `is_local`) with no cost data; `records` carries the same identity fields plus the
  accumulated totals, so it is a superset for any consumer that was reading it.

Gates: 600 tests (was 585), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.5] — 2026-07-22

**Health & failover — the fifth and largest band of the rust-old parity closeout arc.** A dead provider used to
keep taking traffic indefinitely; hoosh now probes its backends and routes around the ones that are down.
See [the parity review](docs/development/rust-old-parity-review.md).

### Added
- **`src/lib/health.cyr` — background provider health probing.** One bankless thread probes every enabled route
  on `[server] health_check_interval_secs` (default 30; `0` disables) and maintains a per-route health record.
  It probes once inline at startup so the earliest requests route on measured state rather than an assumption.
  Port of rust-old's `health.rs`.
- **A circuit breaker per route.** `UNHEALTHY_THRESHOLD = 3` consecutive failures withdraws a route from
  selection; a single success restores it and resets the counter. Failures must be *consecutive* — an
  intermittent backend is not depooled by cumulative noise. A `health_changed` event fires on each transition
  and never on a repeat, so the bus is not spammed while a backend stays down.
- **`[[providers]] enabled = false`** — takes a route out of selection without deleting it. `route_enabled` was
  already honored by the router; nothing ever set it from config (rust-old `config.rs:272`).
- **`/v1/health/providers` gained `status` (`healthy`/`unhealthy`/`disabled`), `enabled`,
  `consecutive_failures`, `last_check_ms`, and `check_interval_secs`.** Disabled routes are now listed rather
  than omitted.

### Fixed
- **The router never consulted health, so a dead backend kept receiving requests.** The health map was written
  by the `/v1/health/providers` handler and read by *nobody* — purely decorative telemetry. `router_select` now
  filters unhealthy routes out of the candidate set (rust-old `router.rs:76-96`).
  Verified end to end with two backends: killing the priority-1 backend moved traffic to priority-2 after three
  failed probes and emitted one `health_changed`; restarting it moved traffic back.
- **`/v1/health/providers` ran N blocking TCP connects inline on a pool worker for every poll**, so a monitoring
  system scraping it tied up one of the 7 workers for as long as the slowest backend took to answer — and then
  discarded the result. It now reports the state the background prober maintains, making the endpoint a handful
  of memory reads.

### Changed
- **Probe depth is asymmetric by design.** Local (`http://`) routes get a real HTTP GET — Ollama's `/api/tags`,
  everyone else's `/v1/models` — which confirms the backend is actually serving rather than merely that a port
  is open. Remote (`https://`) routes get a TCP-connect probe instead. rust-old issued the authenticated GET
  remotely too, but that bills an API call per provider per interval and adds rate-limit pressure for a
  liveness signal a connect already provides; it would also require a sigil crypto bank, which the bankless
  prober thread does not own (banks 0..7 = main + the 7 pool workers).
- **When every candidate is unhealthy, selection falls back to the unfiltered set** rather than returning 404.
  A backend can recover between probes, so refusing traffic outright turns a provider blip into a guaranteed
  outage — attempting and failing (502) is strictly more useful. Logged when it happens. Verified: with both
  backends dead the request returns 502, not 404.
- `route` grew a `ROUTE_HEALTH` field (size 72 → 80). The health record lives on the route so `router_select`
  reads it with one load and it survives `/v1/admin/reload` the same way the route does. The `healthy` flag is
  read **unlocked** on the hot path — a single aligned i64 written by the one prober thread, so a reader sees
  the old or new value and nothing between; the worst case is one request routed on a reading milliseconds
  stale. Same reasoning as the lock-free `_router` swap in [ADR 011](docs/decisions/011-multithreaded-accept-loop.md).

Gates: 585 tests (was 562), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.4] — 2026-07-22

**Cache expiry — the fourth band of the rust-old parity closeout arc.** `ttl_secs` has shipped in `hoosh.cyml`
since the Cyrius port with nothing reading it; cached responses never expired.
See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **`[cache] ttl_secs` was inert — stale responses were served forever.** The config key shipped in `hoosh.cyml`
  since the port, but nothing parsed it and the cache had no notion of age: the entry carried only an LRU
  recency stamp, rewritten on every read, so there was nothing to measure a lifetime against. Eviction was
  capacity-only, meaning a cached answer could be returned indefinitely. Entries now carry a `created_ms` stamp
  and are checked on read. rust-old had `CacheEntry.ttl` + `is_expired` (`cache/mod.rs:40-44`).
  Verified live: with `ttl_secs = 3`, a repeated query hits the cache, then re-fetches after the deadline —
  the backend saw exactly 2 calls across 4 requests.
- **An expired entry left its embedding behind.** `semantic_remove(key)` is new; the response cache calls it on
  expiry and on sweep. Without it the semantic index outlives the response it points at, so `semantic_find`
  keeps returning a key that resolves to nothing — a wasted lookup per query, and the dead embedding still
  competes for `max_search`.

### Added
- **`cache_evict_expired`** — a full sweep, run on capacity pressure (reclaiming dead entries *before* evicting
  a live one) and before `/v1/cache/stats` reports, so `entries` counts what is actually servable. rust-old's
  `evict_expired` (`cache/mod.rs:207-214`). hoosh has no background timer, so these are the two natural points.
- **`/v1/cache/stats` gained `max_entries`, `hit_rate`, `ttl_secs`, `enabled`** — previously only `entries`,
  `hits`, `misses`, `evictions`.
- Expiry-on-read counts **both** an eviction and a miss, matching rust-old's accounting: the two answer
  different questions, and `hit_rate` is only honest if the miss is recorded.

### Changed
- `cache_new` takes a third argument, `ttl_secs`; `0` (or negative) disables expiry explicitly — the pre-2.5.4
  behavior, now something you opt into rather than get by accident.

### Benchmarks — a step change that is NOT a regression
`cache_get_hit` 108 ns → 1487 ns and `cache_insert` 71 ns → 1447 ns in `bench-history.csv`. The code did not get
14× slower; the **benchmark got honest**. The bench mirror was a bare map lookup — it now models the real entry
struct and the `clock_now_ms()` call that `cache_get` has *always* made for its LRU touch. Measured in
isolation: `clock_now_ms()` alone is **1.351 µs** (a real syscall, not vDSO), the entry alloc is 19 ns. So the
clock is essentially the entire number, and it was being paid before this release too — the old bench simply
did not measure it. Filed as a real optimization target under 2.5.11. (While hoisting, `cache_get` also stopped
taking a *second* clock reading on the hit path.)

Gates: 562 tests (was 535), 17 benchmarks; fmt/lint/vet/deny clean.

## [2.5.3] — 2026-07-22

**Provider correctness — the third band of the rust-old parity closeout arc.** Four defects that made hoosh
misbehave against real backends: a default route that 404s, retries that never gave up, sockets with no
deadlines, and sampling params that never reached Ollama.
See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **The default Groq route 404s.** `provider_default_url` returned `https://api.groq.com`, and since the URL is
  built by plain concatenation every request went to `https://api.groq.com/v1/chat/completions`. Groq's
  OpenAI-compatible surface lives under `/openai`. rust-old had this right (`provider/groq.rs:21`); the port
  dropped the prefix. Now `https://api.groq.com/openai`.
- **Retries burned the full backoff schedule on permanent errors.** `provider_forward` returned a bare 0 for
  every failure, so `retry_forward` could not tell "the backend is down" from "your request is malformed" and
  retried a 400 four times over ~30 s. It now classifies the response and returns `FWD_PERMANENT` for a 4xx that
  is not 408/429, which ends the loop immediately. rust-old gated this on `HooshError::is_retryable`
  (`provider/retry.rs:64`). Measured against a mock backend: a 400 now costs **1 attempt / ~7 ms** (was 4
  attempts / ~1.6 s at the test config); 500, 429 and 503 still get all four.
- **The local provider path had no socket timeouts at all** — neither connect nor read/write. A backend that
  accepted a connection and then went silent pinned a worker thread **forever**; with a 7-worker pool, seven
  such calls hang the gateway with no recovery short of a restart. Now `connect_timeout` 10 s (via
  `net_connect_nb`) plus 300 s `SO_RCVTIMEO`/`SO_SNDTIMEO`, matching rust-old's `client.rs:127` and
  `provider/mod.rs:26`. Measured: a request to a blackholed IP now fails at **~10 s** instead of the ~130 s
  kernel default. The streaming path gets the same deadlines — the read timeout is a whole-stream idle bound,
  so a normally-paused SSE stream is unaffected.
- **Ollama ignored every sampling parameter.** Ollama's native `/api/chat` reads
  `options.{temperature,top_p,num_predict}` and discards the OpenAI-style top-level fields — so 2.5.2's
  forwarding work was inert for the flagship local provider. The body builders now emit the native shape for
  ollama routes and the flat shape for everyone else.

### Added
- **`ANTHROPIC_API_VERSION` env override** — the `anthropic-version` header was hardcoded to `2023-06-01`, so
  adopting a new API version required a rebuild. rust-old read the same variable.
- **18 more model catalog entries** (16 → 34): the Mistral family (`mistral-large`, `mistral-small`,
  `codestral`, `pixtral-large`, `open-mistral-nemo`) and the local Ollama/llama.cpp defaults (`llama3`,
  `llama3.1`, `llama3.2`, `llama3.3`, `codellama`, `mistral`, `mixtral`, `qwen2.5`, `qwen3`, `gemma2`,
  `gemma3`, `phi3`, `deepseek-r1`). Lookup is exact-match then longest-prefix, so the general and specific keys
  coexist. Local models previously fell through the unknown-model path, which skips context compaction and
  cost/tier reasoning entirely.

### Known gap
- Ollama's **embeddings** response is still passed through rather than normalized to the OpenAI
  `{object:"list", data:[…]}` envelope. `/v1/embeddings` needs its own pass (the handler also passes a port
  where a base-url cstr is expected) — tracked in the roadmap under 2.5.7.

Gates: 535 tests (was 515), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.2] — 2026-07-22

**Request-path fidelity — the second band of the rust-old parity closeout arc.** Sampling parameters now reach
the provider, and malformed requests are rejected with a useful message instead of being forwarded or silently
altered. See [the parity review](docs/development/rust-old-parity-review.md).

### Fixed
- **`temperature` and `top_p` were silently dropped.** Neither string existed anywhere in the gateway — a client
  asking for `temperature: 0` got nondeterministic output with no error, and no way to tell. Both are now parsed,
  range-validated, and forwarded on the blocking *and* streaming paths, for OpenAI-compatible, Anthropic, and
  local providers.
- **`max_tokens` never reached the provider.** It was read once, only to size the token-budget reservation.
  Anthropic bodies hardcoded 4096 (or 16384 with extended thinking); OpenAI-compatible bodies omitted it
  entirely, so a client's output cap was ignored. It is now forwarded; the Anthropic defaults apply only when
  the request omits the field.
- **`max_tokens` was invisible when it followed `messages`.** The old `json_get` reads a flat parse that cannot
  reach a top-level key placed after the nested `messages` array — which is exactly where OpenAI clients put it.
  The budget reservation therefore fell back to its 2048 default on real client traffic. All three sampling
  params now use a raw byte scan (`_req_num_milli`), the same technique 2.5.0 needed for `reasoning_effort`.
- **A `GET` to a body-taking route answered a confusing `400 empty request body`.** Those routes now answer
  `405 Method Not Allowed`. Enforcement is deliberately one-sided — the GET routes stay verb-agnostic, because
  gating them would turn a `POST` that works today into a 405 with no correctness win.

### Added
- **Request validation** matching rust-old's `validate_chat_request`: empty `messages` → 400; more than 256
  messages → 400 carrying the count; `temperature` outside [0.0, 2.0] and `top_p` outside [0.0, 1.0] → 400
  echoing the offending value; `max_tokens` ≤ 0 → 400.
- **`[[providers]] max_tokens_limit`** — a per-provider output clamp. A request asking for more is trimmed to the
  limit with a warn log rather than refused.

### Changed
- **Per-request generation options are now one record.** `genopts` (`types.cyr`) carries the thinking effort plus
  the three sampling params and is threaded through `retry_forward` → `provider_forward` → the body builders,
  replacing the bare effort i64 that 2.5.0 introduced. Accessors tolerate a null pointer, so paths with no
  options (batch lifecycle, CLI one-shot) keep their previous behavior. Fractional values are carried as
  x1000-scaled ints — this codebase has no floats — and rendered back to wire form by `_sb_add_milli`.
- **Anthropic: sampling is forwarded only with extended thinking off.** Anthropic pins temperature to 1 under
  thinking and rejects any other value, so sending a client's `temperature` alongside `reasoning_effort` would
  turn a working request into a 400. When both are present, effort wins and sampling is dropped for that request.
- **`route` grew a field** (`ROUTE_MAX_TOKENS`, size 64 → 72) for the per-provider clamp.

Verified against a mock backend: `temperature`/`top_p`/`max_tokens` arrive on both the blocking and SSE paths;
`temperature: 0` survives as a value rather than being read as unset; `max_tokens` placed after `messages` is
found; the clamp trims 5000 → 100 and leaves 50 untouched.

Gates: 515 tests (was 482), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.1] — 2026-07-22

**Security & hard limits — the first band of the rust-old parity closeout arc.** A full behavioral diff of the
archived Rust tree against the port (1,007 behaviors catalogued, see
[the parity review](docs/development/rust-old-parity-review.md)) found the surface area is a superset but the
request path is not at parity. This cut closes the three highest-severity gaps.

### Security
- **The gateway ignored `[server] bind` and listened on `0.0.0.0`.** `hoosh.cyml` has shipped
  `bind = "127.0.0.1"` since the port, but `config.cyr` never parsed the key and `cmd_serve` bound
  `INADDR_ANY()` unconditionally — so hoosh listened on **every interface** while its own config claimed
  loopback. Combined with auth being optional (`auth.cyr`: no tokens configured ⇒ allow all), a stock
  deployment was an unauthenticated LLM gateway reachable from the whole network. `bind` is now parsed
  (via `host_addr`, so a dotted quad, a hostname, or `0.0.0.0` all work) and **defaults to loopback**,
  matching rust-old's `ServerConfig.bind` default. The startup banner prints the address actually bound.
- **`[auth] tokens` accepted only one token.** The parser pushed the raw TOML scalar as a single token, so
  `tokens = ["a", "b"]` matched nothing and multi-key deployments (per-consumer keys, key rotation) were
  inexpressible — rust-old took a list (`config.rs:166`). Now parsed as an array; `auth_check` already
  looped the vec correctly, so this was purely parse-side.

### Added
- **A 1 MiB request body cap, answering `413 Payload Too Large`** — matching rust-old's axum
  `DefaultBodyLimit`. `Content-Length` is checked *before* any large allocation, so an oversized request
  costs one response rather than a megabyte of the never-freeing arena.

### Fixed
- **Requests larger than 64 KiB were silently truncated.** `_handle_conn` did a single `sock_recv` of
  65535 bytes and never drained the rest, so any body bigger than that — or merely split across TCP
  segments — was cut short and surfaced as a malformed-JSON 400. A legitimate 100 KiB chat request could
  not be served at all. The read path now honors `Content-Length` and reads the full body into a
  right-sized buffer (bounded by the new cap). Verified: 200 KB and exactly-1 MiB bodies parse and route;
  1 MiB + 1 is refused with 413.
- **Latent heap overflow in the `models` pattern parser.** `[[providers]] models = [...]` allocated 8
  pointer slots and never bounds-checked the parse loop, so a 9th model pattern wrote past the
  allocation. Both `models` and the new `tokens` array now share a bounded `_config_str_array` helper,
  capped at 32 entries each.

### Changed
- **BREAKING — default listen address is now `127.0.0.1`, was `0.0.0.0`.** A config that omits `bind`
  (or no config at all) now serves loopback only. Set `[server] bind = "0.0.0.0"` to restore off-host
  access. Deployments using the shipped `hoosh.cyml` are unaffected in effect — it already said
  `127.0.0.1`; only the actual behavior changes to match it.
- **`BACKLOG.md` removed**; its contents are now a [Backlog section](docs/development/roadmap.md) in the
  roadmap. Two lists had drifted — it still carried P4 (multi-threaded accept loop) as open six weeks
  after 2.4.0 shipped it.

### Docs
- **New: `docs/development/rust-old-parity-review.md`** — the parity diff's evidence record, splitting
  hand-verified findings from agent-reported leads.
- **Roadmap: the v2.5.x parity-closeout arc** (10 bands), plus the missing 2.4.13 / 2.5.0 shipped rows.
  The *Shipped* preamble's "reached parity" claim is qualified — surface area matched, request path did not.

Gates: 482 tests (was 464), 17 benchmarks, no regressions; fmt/lint/vet/deny clean.

## [2.5.0] — 2026-07-14

**Extended thinking / reasoning: `reasoning_effort` control + a `reasoning_content` stream. Toolchain + deps refresh.**

### Added
- **`reasoning_effort` (`low`/`medium`/`high`) now controls Anthropic extended thinking.** hoosh maps it to the
  model's **adaptive** thinking API — `"thinking":{"type":"adaptive"},"output_config":{"effort":"..."}` — and raises
  `max_tokens` to leave room for a reasoning-influenced answer. This is the format current models (Opus 4.8, Sonnet
  4.x) require; the legacy `"thinking":{"type":"enabled"}`+`budget_tokens` is **rejected** by them ("not supported
  for this model"). The effort is read with a **raw byte scan** (`_req_reasoning_effort`, like `_req_is_stream`) —
  NOT the flat `json_parse`, which can't reach a top-level key that follows the nested `messages` array (where
  OpenAI clients put it). Threaded per-request on the prep struct (hoosh is multi-threaded — a global would race).
  Without `reasoning_effort` the request is byte-identical to before.
- **A `reasoning_content` streaming delta** — hoosh now translates a provider's reasoning stream into the de-facto
  OpenAI-compat `delta.reasoning_content` field, kept SEPARATE from `delta.content` so a client can fold the
  reasoning apart from the answer. For Anthropic, an SSE `thinking_delta` → `reasoning_content` (new
  `anthropic_extract_thinking` + `_sse_reasoning_chunk`). Inert on a normal turn, and — usefully — inert on **Opus
  4.8**, which keeps its reasoning INTERNAL (it streams only `text_delta`, never `thinking_delta`, even at high
  effort); the translation is there for models that DO expose thinking. Verified live: `reasoning_effort` requests
  complete (were rejected/empty before this cut); unit-tested (`_req_reasoning_effort` incl. the after-`messages`
  case).

### Changed
- **Toolchain + dependency refresh**: Cyrius pin **6.3.15 → 6.4.62** (`cyrius.cyml`, vendored stdlib re-synced),
  and the `ai-hwaccel` dep **2.3.12 → 2.3.14**. Build + the 459-test suite green on the new toolchain (464 with the
  new `reasoning_effort` tests).

## [2.4.13] — 2026-07-13

**Fix: a client disconnecting mid-stream (SIGPIPE) crashed the gateway.**

### Fixed
- **The gateway process died when an SSE-streaming client disconnected mid-response** — e.g. thoth aborting a
  streaming turn with its Esc-interrupt. `cmd_serve` runs hoosh's own accept loop and writes SSE chunks with bare
  `sys_write` (no `MSG_NOSIGNAL`), so the *next* chunk write after the client went away raised **SIGPIPE**, whose
  default disposition **terminates the process**. hoosh crashed a moment after any interrupted stream, so every
  following request "failed to connect." `cmd_serve` now installs the SIGPIPE→`SIG_IGN` guard at startup (sandhi's
  proven `rt_sigaction` helper — the same one `sandhi_server_run` installs, which our hand-rolled loop bypassed), so
  a write to a dead peer yields an `EPIPE` error instead of killing the gateway. Verified: repeated mid-stream client
  aborts leave hoosh alive and serving subsequent requests.
- **Follow-up (not in this patch):** the SSE writers don't yet check the `sys_write` return, so after a client
  disconnects hoosh keeps pulling the rest of the response from the provider and writing (ignored) `EPIPE`s until the
  upstream stream ends — harmless now (no crash) but wasteful. Detecting the write error to abort the upstream
  early is a follow-up.

## [2.4.12] — 2026-07-03

**Tool-continuation fix — agentic loops now complete on Anthropic backends.**
The Anthropic request-builder copied OpenAI messages verbatim, so a follow-up
turn carrying an assistant `tool_calls` message plus a `role:"tool"` result was
sent to Anthropic unchanged — shapes the Messages API rejects — and the failure
surfaced (misclassified) as `502 provider backend unreachable`. This broke every
multi-step agentic tool loop: the model could call a tool once, but feeding the
result back to continue failed. Single-turn and repeated-single-turn calls were
unaffected, which is why the break was specific to the tool-continuation shape.

### Fixed
- **`_build_anthropic_body_x` (`src/lib/provider.cyr`) now translates OpenAI tool
  messages into Anthropic content blocks** as it re-emits the `messages` array:
  - an assistant message with `tool_calls` → `{"role":"assistant","content":[…]}`
    with a `text` block (only when `content` is a non-empty string; `content:null`
    is dropped) followed by one `tool_use` block per call — the call's `arguments`
    **string** is unescaped into the `input` **object** (empty/absent → `{}`), and
    `tool_use.id` carries the OpenAI `id` so it matches the tool result;
  - a `role:"tool"` message → `{"role":"user","content":[{"type":"tool_result",
    "tool_use_id":<tool_call_id>,"content":<result string>}]}`;
  - every other message (plain `{role, content-string}`) still passes through
    verbatim — already Anthropic-shaped.
  New helper `_json_unesc_span` reverses JSON string-escaping so an `arguments`
  string becomes a literal JSON object in `input`.

### Verified
- The exact filed repro (assistant `tool_calls` + `role:"tool"` continuation) now
  returns **200** with a correct completion; streaming and a two-`tool_call` /
  two-result continuation both return 200 (streaming also restores the closing
  summary that previously came back empty).
- Full vertical re-proven: thoth → hoosh (`claude-opus-4-8`) → tool call →
  t-ron allow → daimon → bote 3.0.0 `fs_write` (file written) → **continuation
  turn returns the final summary**; the one-shot agentic loop exits 0.
- Suite green (457 assertions).

## [2.4.11] — 2026-07-01

**AGNOS cross-build readiness.** hoosh now compiles cleanly under
`cyrius build --agnos`. Host build byte-identical; 457 assertions still pass.

### Changed
- **Networking ported to the portable `net.cyr` socket API.** The six
  outbound-TCP sites (`main.cyr`, `lib/http_client.cyr` ×2, `lib/otlp.cyr`,
  `lib/handlers.cyr` ×2) used raw Linux BSD sockets (`sys_socket` / `sys_connect`)
  — undefined on agnos's frozen syscall surface. They now use `tcp_socket()` +
  `sock_connect(fd, addr, port)`, which dispatch per target (Linux
  `socket`/`connect`; agnos `sock_connect`#47 yielding a tagged fd). The existing
  `sys_write` / `sys_read` / `sys_close` on the fd are unchanged — the agnos
  syscall peer routes tagged fds to `sock_send`/`recv`/`close` (#48/#49/#50), so
  hoosh's gateway networking **runs** on agnos, not merely compiles.

### Fixed
- **`--agnos` build**: guarded the Linux-only `THREAD_STACK_SIZE` knob in
  `lib/hardware.cyr` (HW-probe threads) and `lib/pool.cyr` (worker pool) behind
  `#ifndef CYRIUS_TARGET_AGNOS` — the agnos threading lib has no such global, so
  detection and the pool fall back to their existing paths.

## [2.4.10] — 2026-06-30

Tier-4 (consumer) step of the coordinated base-security-stack migration
to cyrius **6.3.15**. Toolchain pin + re-sync of the vendored bundles to
the migrated versions, plus the stdlib boundary reconciliation the 6.3.x
line requires. All 457 assertions pass on the new stack.

### Changed

- **Cyrius toolchain pin: 6.2.39 → 6.3.15.**
- **Vendored bundles re-synced**: `src/vendor/majra.cyr` 2.4.7 → **2.5.0**
  and `src/vendor/bote-core.cyr` 2.7.6 → **2.7.7** (the migrated tiers).
- **`[deps] stdlib`**: added `atomic` + `sync` (patra's transitive
  `lib/sync.cyr` requirement on 6.3.x), `random` (sigil's `random_bytes`
  pull-through, before `sigil`), and `async` (tls/sandhi inter-dep on
  6.3.x, before `tls`).

### Fixed

- **Vendored majra var-bomb** (`src/vendor/majra.cyr`). The stale 2.4.7
  bundle carried the undersized `var ts[2]` / `var buf[2]` / `var hdr[1]`
  locals that cyrius 6.3.13's move of function-local `var X[N]` arrays to
  the guard-paged thread stack turns from a benign adjacent-scribble into
  a hard SIGSEGV. The 2.5.0 re-sync brings the correctly-sized buffers.

## [2.4.9] — 2026-06-24

**Concrete per-provider model catalog.** `/v1/models` lists provider *names* (one
per enabled route); it does not enumerate the specific models a client can switch
to. New endpoint surfaces them.

### Added
- **`GET /v1/models/catalog`** — concrete, switchable model ids from the pricing
  catalog (metadata.cyr), each tagged with the provider that would route it as
  `owned_by`. Only models an **enabled route actually matches** are listed, so the
  catalog reflects what this gateway can serve (e.g. with no DeepSeek/Groq route
  configured, those models are omitted; `o1` is omitted when only `o1-*` is
  configured). Shape mirrors `/v1/models` so the same client parser works:
  `{"object":"list","data":[{"id","object":"model","owned_by"}]}`. Clients drill
  down per provider by filtering on `owned_by`. (`handle_models_catalog`,
  `_catalog_provider_for`; new `metadata_count` / `metadata_key` accessors.)
  `/v1/models` (provider list) is unchanged.

## [2.4.8] — 2026-06-24

**Local providers reachable off-localhost + toolchain refresh.** Fixes a bug
that made any local (plaintext `http://`) provider — Ollama, LlamaCpp —
unreachable unless it ran on the *same host* as hoosh, and bumps Cyrius to
**6.2.39**.

### Fixed
- **Local providers on a non-localhost host returned `502 provider backend
  unreachable`.** The raw-socket forwarder for `http://` backends
  (`http_client.cyr`) hardcoded the connect target to `127.0.0.1` and parsed
  only the **port** from `base_url` — the host was discarded. So an Ollama box
  at e.g. `http://192.168.1.186:11434` was contacted at `127.0.0.1:11434`,
  where nothing listened. Added `url_host` (parse the host from the URL) and
  `host_addr` (resolve it: `localhost` → loopback, dotted-quad literal parsed
  directly, otherwise sandhi DNS; loopback fallback on miss), and routed every
  local-path connect through them: blocking + streaming chat forward
  (`provider.cyr`), the HTTP `/health` + provider-status probes and the CLI
  `health` command, the Ollama/Synapse lifecycle admin (pull/delete/training/
  catalog), embeddings, and OTLP export. The `Host:` header now carries the
  real host. Remote `https://` routing (via sandhi) was always host-correct and
  is unchanged.

### Changed
- **Toolchain: Cyrius 6.2.39** (pin, was 6.2.37). Clean `lib/` re-sync; no
  stdlib module migration. All 457 tests pass.

## [2.4.7] — 2026-06-23

**Toolchain refresh.** Bumps Cyrius to **6.2.37** with the one single-pass
include-order fix the new stdlib requires.

### Changed
- **Toolchain: Cyrius 6.2.37** (pin, was 6.2.11). Clean `lib/` re-sync; no
  stdlib module migration.

### Fixed
- **Test/bench harness `undefined variable 'sys_getrandom'` under 6.2.37.**
  Since stdlib 6.2.28, `tls_native_conn` takes `&sys_getrandom`. `cyrius build`
  prepends the auto-injected `[deps]` stdlib set (incl. `tls`) ahead of a file's
  explicit includes, in array order — so an explicit `include "lib/syscalls.cyr"`
  in `tests/hoosh.tcyr` / `tests/hoosh.bcyr` landed *after* the prepended `tls`,
  leaving `sys_getrandom` undefined at that point. Dropped the explicit syscalls
  include from both harnesses (it auto-injects at its array position, before
  `tls`); `sys_*` wrappers stay in scope. All 457 tests pass, benchmarks intact.

## [2.4.6] — 2026-06-15

**Toolchain + dependency refresh.** Bumps Cyrius to **6.2.11** and all
dependencies to their latest tags, with the one breaking-API fix the bote
update requires.

### Changed
- **Toolchain: Cyrius 6.2.11** (pin, was 6.1.31). Clean `lib/` re-sync; no
  stdlib module migration (the 6.2.x snapshot only *adds* modules —
  `tls_native_*` split, `*_agnos` variants — and removes none).
- **ai-hwaccel 2.3.9 → 2.3.12** (`cyrius.cyml` tag). `dist/ai-hwaccel.cyr`
  re-vendored; the vendored `data/cloud_pricing.json` + `data/models.json` are
  byte-/content-identical at the new tag, so they are unchanged.
- **bote 2.7.3 → 2.7.6** (`src/vendor/bote-core.cyr`). Code is identical apart
  from the rename below and whitespace.
- **majra 2.4.5 → 2.4.7** (`src/vendor/majra.cyr`). Version header only — the
  bundle body is unchanged.

### Fixed
- **bote renamed its registry constructor `registry_new` → `tool_registry_new`.**
  This matters because ai-hwaccel *also* defines `registry_new`, so leaving the
  call unchanged would have silently bound hoosh's MCP registry to the hardware
  registry ("last definition wins"). Updated the call sites in `src/lib/mcp.cyr`
  and the test/bench harnesses to call `tool_registry_new` explicitly. The other
  `registry_*` ops keep their names. All 457 tests pass, MCP benchmarks intact.
- **Cyrius 6.2.11 now hard-errors on duplicate same-scope variables.** The test
  harness redeclared three top-level `var`s in `main()` (`tp`, `ll1`, `ll2`);
  renamed the second occurrences (`tval`, `llv1`, `llv2`). Block-scoped
  shadowing (e.g. `rec`) is still allowed and was left as-is.

## [2.4.5] — 2026-06-10

**Hardening review** — closes the v2.4.x arc. A concurrency/correctness/security
audit of the 2.4.x code with concrete fixes, plus Cyrius **6.1.31**.

### Fixed (concurrency — v2.4.0 sync-pass misses)
- **`GET /v1/cache/stats`** read `map_count` over the cache entries map without a
  lock — racing the chat path's `cache_insert`/evict (`map_set`/`map_delete` →
  rehash) on the pool workers, a latent crash. Now snapshots the four values
  under `_chat_lock`.
- **`GET /v1/tokens/pools`** iterated `map_keys(_budget)` + read pool fields
  unlocked — racing reload's `budget_add_pool` and reserve/commit writes. Now
  under `_chat_lock`. Both verified with a 240-way concurrent stress (chat +
  stats + pools), no crash.

### Fixed (correctness)
- **Routing strategy is now configurable** — `[server] strategy =
  "priority"|"round-robin"|"lowest-latency"|"direct"`. It was hardcoded to
  `priority`, so three implemented strategies were unreachable.
- **Lowest-latency routing now works** — its EMA writer (`router_report_latency`)
  was never called (dead), and `router_select` treated untried backends as
  *max* latency, so they were never explored. Now the forward feeds the EMA
  (thread-safe via a dedicated `_lat_lock`, gated on the active strategy), and
  untried backends are explored before exploiting the fastest. Live-verified
  24:1 toward the fast backend across two mocks.

### Changed
- **Toolchain: Cyrius 6.1.31** (pin). Clean `lib/` re-sync; no stdlib migration.
- Removed dead `_cost_map` (declared + `map_new`'d, never used).

## [2.4.4] — 2026-06-10

**New backends** — vLLM, TensorRT-LLM, and ONNX Runtime, completing the v2.4.x
arc's new-backend item. **17 providers** now (was 14).

### Added
- **vLLM** (`type = "vLLM"`), **TensorRT-LLM** (`"TensorRT-LLM"` / `"TensorRT"` /
  `"trtllm"`), **ONNX Runtime** (`"ONNX"` / `"ONNXRuntime"`) provider types. All
  three serve an OpenAI-compatible API, so they route through the existing local
  OpenAI-compatible forward (`/v1/chat/completions`) — `types.cyr` gains the enum
  entries + name/is-local/parse/default-url; no forward-path change. Default base
  URLs `http://localhost:8000`; marked **local** (free in the cost optimizer,
  permitted for Confidential-classified traffic under DLP). Live-verified: all
  three parse + register + route + forward to a mock (`POST /v1/chat/completions`
  confirmed); vLLM end-to-end chat round-trip.

## [2.4.3] — 2026-06-10

**OTLP remote/https export + scaffolding** — an observability follow-up plus
sibling-repo scaffolding conventions.

### Added
- **OTLP remote / `https://` collector** — the exporter now posts to remote
  collectors over TLS via sandhi (DNS + TLS), not just a localhost `http://`
  sidecar. Because TLS needs a sigil crypto bank and the exporter thread is
  bankless (banks 0..7 = main + 7 workers), an `https` POST is enqueued as a
  `JOB_OTLP_EXPORT` so a **banked pool worker** does the TLS; `http` stays direct
  on the exporter. Live-verified: http path receipt-confirmed at a local
  collector; https path does real TLS POSTs to a public endpoint on a worker with
  the server stable under concurrent load.
- **`docs/development/state.md`** — volatile-state snapshot (version, sizes,
  test/bench counts), refreshed per release (patra/cyrius pattern); linked from
  the docs index.
- **Fuzz harnesses** (`fuzz/*.fcyr`) — adversarial-input harnesses for the
  hand-rolled parsers (`_batch_split` array splitter, `trace_extract` header
  scanner): unbalanced brackets, trailing escapes, unterminated strings,
  truncation, generated byte stress. No crash across ~8k inputs.
- **CI: security-pattern scan + fuzz step** — `scripts/security-scan.sh` (flags
  subprocess exec, hardcoded `/etc` paths, hardcoded provider keys in `src/`) and
  a CI step building + running every `fuzz/*.fcyr` under a timeout.

### Notes
- OTLP/protobuf stays upstream-gated (cyrius protobuf lib); nested spans deferred.

## [2.4.2] — 2026-06-10

**Threaded hardware detection** — the last hardware item of the v2.4.x arc, now
unblocked by the v2.4.0 thread-safe foundation.

### Changed
- **`hw_init` uses ai-hwaccel's threaded detector** (`registry_detect_threaded`)
  — the CLI-tool probes (nvidia-smi, vulkaninfo, hl-smi, neuron-ls, …) run in
  parallel threads while sysfs backends run inline; per-thread result vecs are
  merged after join (race-free). Previously serial: the threaded detector
  segfaulted under the pre-thread-safe allocator. It's safe since the v6.0.64 CAS
  spinlock — the only shared mutation is `alloc` (atomic), and the fork+exec child
  does only raw syscalls (no child-side alloc → fork-safe from a thread). Probe
  threads get 1 MB stacks (set before the first `thread_create`); internal serial
  fallback if `thread_create` fails. **Verified**: detection output is byte-identical
  to serial (3 profiles, ROCm GPU), no segfault across repeated runs, server
  startup clean.

## [2.4.1] — 2026-06-10

**Hardware planning endpoints** — the remaining ai-hwaccel surface from the
v2.4.x arc. Also bumps the toolchain to Cyrius **6.1.30**.

### Added
- **`POST /v1/hardware/model-format`** — detect a model's container format from
  the **raw model bytes** in the request body (send the file or its header; no
  filesystem path, so no arbitrary-read surface). Returns ai-hwaccel's
  `ModelMetadata` JSON (`format` + `param_count`/`dtype`/`tensor_count`/
  `format_version` when present). Detects SafeTensors / GGUF / ONNX / PyTorch;
  unrecognized → 400. Live-verified against crafted GGUF + SafeTensors headers.
- **`POST /v1/hardware/requirement-match`** — does the detected hardware satisfy a
  scheduler requirement? Body `{requirement: "gpu"|"tpu"|"gaudi"|"aws-neuron"|
  "gpu-or-tpu"|"any-accelerator"|"none", min_chips?: N}` → `{requirement,
  min_chips, satisfied, device?}`. Reads the immutable hw registry (detected at
  startup) — lock-free under the v2.4.0 worker pool. Live-verified (ROCm GPU
  detected on the dev machine; TPU/min_chips → false; unknown req → 400).

### Changed
- **Toolchain: Cyrius 6.1.30** (pin). Clean `lib/` re-sync; no stdlib migration;
  442 tests green.

## [2.4.0] — 2026-06-10

**Multi-threaded accept loop** — interactive traffic now runs concurrently, not
just batch. A unified pool of 7 banked worker threads serves all requests; the
accept loop only accepts + enqueues. See [ADR-011](docs/decisions/011-multithreaded-accept-loop.md).

### Added
- **Unified worker pool** (`src/lib/pool.cyr`) — `WORKER_COUNT = 7` threads, each
  permanently owning a sigil crypto bank (1..7), draining a bounded work-queue
  ring. The accept loop hands each connection to the pool as a job. Bounded by the
  8-crypto-bank limit (bank 0 = main); no thread-per-connection, no DoS surface.
- **Batch unified onto the pool** — `/v1/batch` items are now queue jobs run by
  the same workers (the separate crypto-lane pool is gone). Sync batch
  **work-steals** the queue while waiting (can't stall the pool); async batch uses
  a bankless coordinator thread. The full 7-bank budget is shared dynamically
  between interactive + batch, with no static split.

### Changed
- **Synchronization pass** — shared state guarded for concurrent handlers: a
  `_batch_reg_lock` for the batch registry (map/id-list), `_chat_lock` extended to
  the token handlers (budget) and the health map. Config hot-reload rebuilds
  `_router` as an atomic pointer swap (lock-free readers, safe via the never-frees
  allocator) under `_chat_lock`. Worker stacks bumped to 1 MB.

### Performance
- Live-verified: 14 concurrent 50 ms chats in **115 ms** (≈6× faster than the
  ~700 ms serialized path), 50 in 417 ms; 200 mixed requests + reload-under-load
  survived; sync/async/concurrent batches all correct. `work_queue_push_pop`
  dispatch overhead **8 ns**.

## [2.3.5] — 2026-06-10

**OpenTelemetry OTLP span export** — the deferred half of the 2.3.4 observability
work. One span per inference request is exported to a collector over OTLP/HTTP
with **JSON** encoding (no protobuf — a Cyrius protobuf lib is proposed upstream
for a future OTLP/protobuf path). Also bumps the toolchain to Cyrius **6.1.29**.

### Added
- **`src/lib/otlp.cyr`** — opt-in OTLP exporter, off unless `[[telemetry]]
  otlp_endpoint` is set. Each inference enqueues an OTLP span (traceId/spanId
  from the 2.3.4 traceparent → joins the caller's trace; epoch-ns start/end;
  attributes provider/model/latency_ms/tokens; status OK/ERROR) into a bounded
  ring; a **background thread** batches the ring into an OTLP `resourceSpans`
  document and POSTs it to the collector every second (non-blocking). Localhost
  http collector (the common sidecar); remote/https is a follow-up.
  **Live-verified** against a mock collector: success (status 1) + failure
  (status 2) spans, correct trace-id correlation, epoch timestamps, batching.
- **`[[telemetry]]` config** — `otlp_endpoint` (e.g. `http://localhost:4318/v1/traces`)
  + `service_name`. Absent → no exporter, no overhead.
- **CLOCK_REALTIME epoch-ns clock** for OTLP timestamps (`clock_now_ns` is
  monotonic).

### Changed
- **Toolchain: Cyrius 6.1.29** (pin). Clean `lib/` re-sync; no stdlib migration;
  427 tests green.

## [2.3.4] — 2026-06-10

**Observability** — per-provider latency histograms, a provider event bus
(majra pub/sub) with a recent-events endpoint, and W3C `traceparent` propagation.
Full OpenTelemetry OTLP export is deferred to 2.3.5. See ADR-010.

### Added
- **Per-provider latency histograms** (`/metrics`) — `hoosh_provider_latency_ms`
  Prometheus histogram (11 le-buckets + sum + count, per provider), recorded
  around each provider forward (atomic — batch workers record concurrently).
- **Provider event bus** (`src/lib/events.cyr`, **majra 2.4.5** pubsub) — the
  four `ProviderEvent` kinds from `events.rs` are published as JSON: HealthChanged
  (on a health-poll flip), InferenceCompleted (provider/model/latency/tokens),
  InferenceFailed (provider/model/error), RateLimited (provider). majra is the
  pub/sub substrate (`hoosh_events_published_total` in `/metrics`); since hoosh's
  loop is synchronous, observability is a bounded **recent-events ring** at
  **`GET /v1/events/recent`** (a never-drained subscriber channel would fill and
  block, so events append to the ring as they publish). Live-verified all four.
- **W3C `traceparent` propagation** (`src/lib/trace.cyr`) — the incoming
  `traceparent` is forwarded verbatim to backend requests (local + remote), or a
  fresh one is generated (`clock_now_ns` + atomic counter) when absent, so the
  gateway→backend hop joins the trace. Carried through the deep, threaded forward
  path via a **thread-local** slot (slot 1; slot 0 is sigil's crypto bank) rather
  than threading the header through every signature. Live-verified (incoming
  forwarded; absent → generated).

### Notes
- **majra is vendored** at `src/vendor/majra.cyr` (committed, not `[deps]` — same
  rationale as bote-core). Its `ratelimit_new`/`ratelimit_check` collide by name
  with hoosh's (different signatures); harmless — they're dead code in majra's
  core pubsub, so hoosh's win and work. One benign `last-definition-wins` build
  note, like the existing bayan ones.
- **Gotcha fixed during bring-up:** `thread_local_init()` is NOT idempotent — it
  installs a fresh zeroed TLS block. Calling it per-access wiped the slot (and
  would clobber sigil's crypto bank). It is now called exactly once per thread
  (main via `crypto_tls_main_init` at startup; workers via CLONE_SETTLS).

## [2.3.3] — 2026-06-10

**Concurrent async batches + registry eviction** — the deferred follow-ups to
2.3.2. Multiple async batches now execute *concurrently* (sharing a global
crypto-lane pool), and the batch registry is bounded. Also bumps the toolchain to
Cyrius **6.1.28**.

### Added
- **Global crypto-lane pool** — replaces the one-batch-at-a-time `_batch_exec_lock`.
  sigil's 7 worker crypto lanes (banks 1..7; bank 0 = accept thread) are handed
  out across **all** batches: a worker acquires a free lane → uses it as its
  crypto bank → releases it. Total live workers stay ≤ 7 globally regardless of
  how many batches run, with no cross-batch lane collision. Both sync and async
  batches draw from the pool. Live-verified: 4 async batches of 12 items
  progressed simultaneously (e.g. 8/2/1/3 → 11/9/6/9), all completed 12/12, with
  GET polls returning live progress (HTTP 200) and 15/15 concurrent chats served.
- **Registry eviction** — `BATCH_MAX_TRACKED` (64); on submit over the cap, the
  oldest *terminal* (completed/cancelled) batch is dropped from the registry +
  id list (evicted ids 404). Bounds the tracking map (not heap — hoosh's bump
  allocator never frees, same as every allocation). Live-verified (after 70
  submits, the first id → 404).

### Changed
- **Runners/barriers sleep instead of busy-spin** — `_batch_lane_acquire` and
  the completion barriers now `sleep_ms(1)` when waiting, so concurrent runners
  don't burn cores or starve the accept loop serving `GET /v1/batch/{id}`.
- **Toolchain: Cyrius 6.1.28** (pin). Clean `lib/` re-sync; no stdlib migration
  this bump; 414 tests green.

### Notes
- Concurrency is bounded by the 7-lane pool; submitting many batches at once is
  fine (they share lanes fairly), but per-batch throughput drops as lanes are
  split across active batches. See ADR-009.

## [2.3.2] — 2026-06-10

**Async batch inference** — `POST /v1/batch` with `{"async":true}` returns a
batch id immediately and runs the batch on a background thread; clients poll
`GET /v1/batch/{id}` for progress and `POST /v1/batch/{id}/cancel` to cancel.
Completes the `inference/batch.rs` port (submit → progress → cancel → registry).
The synchronous `POST /v1/batch` (2.3.1) is unchanged and remains the default.

### Added
- **`POST /v1/batch` async mode** (`{"async":true}`) → `{"id","status","total"}`.
  A background runner thread executes the items (waves of 7 crypto-bank workers,
  same engine as the sync path), updating a `BatchState`.
- **`GET /v1/batch/{id}`** → progress snapshot: `{id,status,total,completed,
  failed,results[]}`. `status` ∈ queued/running/completed/cancelled; a still-
  pending item shows `status:null,body:null`. Reads are lock-free (atomics +
  write-once result slots).
- **`POST /v1/batch/{id}/cancel`** → sets the cancel flag; in-flight items finish
  but no further waves launch (the runner checks the flag before each wave).
- **Batch registry** (`src/lib/batch.cyr`): `id → BatchState` map + atomic id
  counter, touched only by the single-threaded accept loop (no lock needed).
  Live-verified end-to-end: submit→poll→complete (6/6), cancel mid-flight (28-item
  batch → cancelled at 14/28), and **20/20 concurrent `/v1/chat/completions`
  served while a 28-item batch ran in the background** (the sync-pass guarantee).

### Changed
- **Sync pass: the accept-thread chat path now holds `_chat_lock`** around its
  shared-state phases (prep, assemble/commit; streaming audit appends too). An
  async batch runs workers concurrently with the accept loop, so a
  `/v1/chat/completions` served mid-batch would otherwise race batch workers on
  `_cache`/`_budget`/`_audit`. Uncontended (cheap) when no batch is in flight.
- **One batch executes at a time** via `_batch_exec_lock` (sync and async share
  it), so crypto-bank workers (lanes 1..7) never overlap across batches;
  additional submitted batches queue (`status:"queued"`).

### Notes
- Async batches run sequentially (one executes at a time); concurrent execution
  of multiple async batches, and completed-batch eviction from the registry, are
  future work. See ADR-009.

## [2.3.1] — 2026-06-10

**Concurrent batch inference** — `POST /v1/batch` runs an array of chat requests
on worker threads and returns all results in one response. Unblocked by
re-verifying (not assuming) that the cyrius 6.1.27 allocator is thread-safe: a
4-thread × 200k-alloc stress showed zero corruption (the v6.0.64 CAS spinlock
fixed the race the older roadmap audit flagged as a hard blocker). Async
job-id/progress-poll/cancel and a fully multi-threaded accept loop are deferred
(see roadmap); this ships the synchronous concurrent executor.

### Added
- **`POST /v1/batch`** — body `{"requests":[ <chat req>, … ]}`; runs the items
  concurrently (bounded to `BATCH_MAX_PARALLEL`=7 in flight, processed in waves;
  `BATCH_MAX_ITEMS`=64 cap) and returns `{"results":[{"index","status","body"},
  …]}`. Items are treated as non-streaming. Ports the concurrency core of
  `rust-old/src/inference/batch.rs`. Live-verified against a mock backend: 8
  forwards per wave overlap (confirmed by per-connection arrival timestamps),
  waves serialize, results are correct, and metrics stay exact under load.
- **`src/lib/batch.cyr`** — executor + string/nesting-aware request-array
  splitter (`_batch_split`, `_batch_extract_requests`). Worker completion is
  signalled through an **atomic counter barrier** rather than `thread_join`
  timing (empirically `thread_join` could return before a worker stored its
  result, which would let the next wave overlap and risk reading an unfilled
  slot). Unit-tested (split: nested arrays + bracket-bearing strings) + benched
  (`batch_split_4`).

### Changed
- **`handle_chat` refactored into a returns-body core** — `_chat_prep`
  (validation/routing/DLP/cache/rate/budget/messages → terminal descriptor or
  ready state), `_chat_forward` (the unlocked network call), and `_chat_assemble`
  (response build + cache/budget/audit writes). `handle_chat` is now a thin
  writer over these; batch workers call `_chat_produce` (prep + forward +
  assemble) and collect the body instead of writing a socket. Behavior-preserving
  (live-diffed byte-for-byte against the prior binary for stream + non-stream).
- **Coarse-lock for concurrent workers** — `_chat_produce` holds a global
  `_chat_lock` mutex around the shared-state phases (prep, assemble) and releases
  it for the network forward, so forwards overlap while cache/budget/audit
  bookkeeping stays serialized. `metrics_record` is now **atomic** (CAS adds) —
  it runs inside the unlocked forward. The single-threaded accept path is
  uncontended (the batch handler joins all workers before the accept loop
  resumes, so workers are the only concurrent actors).

### Fixed (pre-release, during 2.3.1 bring-up)
- **Worker-thread crypto crash** — the chat pipeline's sha256/HMAC/TLS (sigil)
  read a **thread-local crypto scratch bank** that is uninitialized on worker
  threads; the first threaded build corrupted the heap (batch returned fine, the
  *next* request segfaulted). Fixed per sigil's intended API: the handler calls
  `crypto_tls_main_init()` before fan-out and each worker calls
  `crypto_bank_set(1..7)` at entry (bank 0 = main). Caps `BATCH_MAX_PARALLEL` at
  7 (sigil's 8 banks − the main lane). See ADR-009 §5.

### Notes
- See **ADR-009** for the concurrency model, the allocator + sigil thread-safety
  findings, and why the barrier (not `thread_join`) is the correctness primitive.

## [2.3.0] — 2026-06-10

**MCP tool-server endpoints** — `GET /v1/tools/list` + `POST /v1/tools/call`,
backed by **bote 2.7.3**'s JSON-RPC 2.0 registry/dispatcher/codec. Closes the
last open item of the v2.2.x parity arc (was tracked as v2.2.5); the parity arc
is complete, so this ships as **2.3.0**. Also bumps the toolchain to the latest
Cyrius (**6.1.27**). Connection pooling and OpenTelemetry/event-bus observability
move to v2.3.1 and v2.3.2 respectively.

### Added
- **`GET /v1/tools/list`** — lists registered MCP tools in JSON-RPC form
  (`{"jsonrpc":"2.0","id":1,"result":{"tools":[…]}}`). hoosh synthesizes the
  `tools/list` request and returns bote's `ToolRegistry` listing verbatim.
- **`POST /v1/tools/call`** — invokes a tool by name. The request body is an MCP
  JSON-RPC request (`method:"tools/call"`, `params:{name,arguments}`); it is run
  through bote's codec + `Dispatcher`, so `initialize` and `tools/list` are also
  accepted here for full MCP-client compatibility. Unknown tools return a
  JSON-RPC error; an empty body returns 400.
- **`src/lib/mcp.cyr`** — the wiring module. `mcp_init` (called from
  `cmd_serve`) builds the registry + dispatcher and registers a built-in
  `bote_echo` smoke tool so both endpoints are live-verifiable end-to-end.
  szál's 58 tool implementations plug in here once they ship as a Cyrius
  distlib — register them alongside `bote_echo` in `mcp_init` and they appear in
  `tools/list` and dispatch through `tools/call` with no transport changes (the
  registry currently holds echo only).
  **Live-verified**: `tools/list` lists `bote_echo`; `tools/call` round-trips
  arguments as MCP text content; `initialize` negotiates the protocol version;
  unknown tool → JSON-RPC error.
- **Tests + benches** — `mcp_tools` test group (registry/dispatch/codec via the
  real bote-core: list/call/unknown-tool) and `mcp_tools_list` / `mcp_tools_call`
  benches (full JSON-RPC parse → dispatch → serialize: ~4 µs / ~9 µs).

### Changed
- **Toolchain: Cyrius 6.1.21 → 6.1.27** (pin in `cyrius.cyml`). 384 → 392 tests
  green.
- **Stdlib migration: `bigint` + `toml` + `json` → `bayan`.** 6.1.27 consolidated
  the standalone `bigint`/`toml`/`json` snapshot modules into a single `bayan`
  module (it provides `json_parse`/`json_get`, `toml_parse`, and the bigint/u256
  surface). `[deps].stdlib` now lists `bayan` (placed before `sigil`, which
  references its u256 surface) in place of the three; `tests/hoosh.tcyr`'s
  explicit `lib/toml.cyr` + `lib/json.cyr` includes collapse to `lib/bayan.cyr`.
  Without this, a clean checkout's `cyrius deps` fails — `cannot read
  ./lib/{bigint,toml,json}.cyr` — since those files no longer ship.

### Notes
- **bote is vendored, not a `[deps.bote]` block.** Unlike ai-hwaccel (no git
  sub-deps), bote's manifest declares `[deps.libro]` + `[deps.majra]`; `cyrius
  deps` resolves those transitively, pulling libro/majra → bayan/ganita/agnosys
  into the compile set, where the agnos superset collides with bote-core's
  `registry_new` and trips an agnosys slice-include error. bote's `[lib.core]`
  bundle (`dist/bote-core.cyr`) is fully self-contained (9 transport-free
  modules, no includes), so it is committed at `src/vendor/bote-core.cyr` and
  included directly. Living under `src/` keeps `cyrius vet` trust intact and the
  generated file out of the fmt/lint globs. Re-sync with
  `./scripts/sync-bote.sh <tag>`.

## [2.2.4] — 2026-06-10

**Tool calling** across all three remote families (OpenAI, Anthropic, Gemini) —
forward `tools`, surface unified OpenAI `tool_calls`, both non-streaming and
streaming — plus the stale tool-pair prune that completes the compression port.
(The MCP server endpoints `/v1/tools/list` + `/v1/tools/call` are deferred to
v2.2.5, pending szál as a Cyrius distlib.)

### Added
- **Streaming tool-call assembly — OpenAI, Anthropic, Gemini** — `stream:true`
  requests forward `tools` on all three families and convert each provider's
  streaming tool deltas into OpenAI `chat.completion.chunk` `tool_calls` deltas
  (`_sse_tool_chunk` in `_remote_stream_cb`):
  - **OpenAI-compat**: deltas pass straight through (client assembles id/name/
    arguments fragments).
  - **Anthropic**: `content_block_start` (tool_use) → id+name delta;
    `input_json_delta.partial_json` → incremental `arguments` deltas
    (`_emit_anthropic_tool_delta`).
  - **Gemini**: complete `functionCall` parts re-emitted as one tool_call delta
    (reusing `_gemini_tool_calls`).

  The buffered early-error fallback also emits any tool calls.
  **Live-verified** against all three (streamed `get_weather` produced the call
  id+name then incremental arguments for OpenAI/Anthropic, one complete delta for
  Gemini); plain streaming unchanged.
- **Compression: stale tool-pair pruning** (`compression.cyr` `prune_tool_pairs`)
  — completes the `context/compression.rs` port (the half deferred from 2.2.3,
  now that tool-call message structure exists). In a long agentic conversation,
  assistant messages carrying `tool_calls` (and their matching `role:"tool"`
  results, paired by id) that fall before the last 3 tool turns are dropped;
  ordinary turns are untouched. Applied after whitespace collapse when
  `[compression]` is enabled. Self-contained byte-level helpers (`_cmp_*`).
  Unit-tested (5-turn conversation → first 2 call+result pairs pruned, recent 3
  and user turns kept; ≤3 turns unchanged).
- **Tool calling — OpenAI, Anthropic, and Gemini** — the gateway forwards a
  request's `tools` to the provider (converting to each native format) and
  surfaces the model's tool calls back as OpenAI `tool_calls` +
  `finish_reason:"tool_calls"`. Ports `tools/convert.rs`.
  - **Request**: `_extract_tools` lifts the `tools` array (balanced-bracket scan);
    threaded through `retry_forward`/`provider_forward` into the body. OpenAI-compat
    passes them verbatim (`_build_chat_body_raw`); `_tools_convert` maps them to
    Anthropic `input_schema` and Gemini `functionDeclarations`.
  - **Response**: `_extract_openai_tool_calls` (OpenAI), `_anthropic_tool_calls`
    (`content[].tool_use` → `tool_calls`, `input` stringified), and
    `_gemini_tool_calls` (`functionCall` parts → `tool_calls`, synthesised ids).
    Tool-call argument objects are JSON-string-escaped (incl. control chars, so
    Gemini's pretty-printed `args` stay valid).
  - **Live-verified** against all three: `get_weather` called for Paris
    (Anthropic), Tokyo (Gemini), London (OpenAI). Plain (no-tools) requests
    unchanged. Unit-tested: `_extract_tools`, OpenAI→Anthropic/Gemini conversion
    (incl. the `"function"`-as-value-vs-key case), Anthropic `tool_use` parsing.

## [2.2.3] — 2026-06-10

The **Cost & cache intelligence** parity arc — semantic cache, cost optimizer,
prompt compression, cache warming, and the now-functional response cache — plus
the native-TLS-by-default toolchain flip (cyrius 6.1.21) and the `hoosh.toml` →
`hoosh.cyml` config rename.

### Added
- **Semantic cache** (`src/lib/semantic.cyr`, `[semantic_cache]` config) — ports
  `cache/semantic.rs`. On an exact-cache miss, the query is embedded via the
  configured `embedding_model` provider and compared (cosine similarity) to
  stored query embeddings; a match above `threshold` reuses that entry's cached
  response — so semantically-similar (differently-worded) requests hit.
  - **Core**: fixed-point cosine (`cosine_x1000` over ×10000 integer vectors,
    magnitudes via integer sqrt to stay within i64), embedding store
    (`semantic_insert`/`semantic_find`, nearest neighbour above threshold with a
    `max_search` cap), and an embeddings-response float-vector parser
    (`semantic_parse_embedding`, sign/decimal/exponent).
  - **Chat-path wiring** (`handle_chat`): on miss, `_embed_query_body` POSTs the
    (JSON-escaped) request to the embedding provider's `/v1/embeddings`
    (`/api/embeddings` for Ollama), parses the vector, `semantic_find`s a hit, or
    stores the embedding under the exact key after forwarding. Any embedding
    failure degrades silently to a normal forward. Streaming requests are not
    semantically cached.
  - Config: `enabled`, `threshold` (0–1 → ×1000), `embedding_model`, `max_search`.
  - **Unit-tested** (cosine, integer sqrt, store, float parsing incl. scientific
    notation) **and live-verified**: two paraphrased "capital of France" queries
    — the second (different exact-key) hit the semantic cache and returned the
    first's response without re-forwarding.
- **Cost optimizer — cheapest *capable* model recommendation** (`pricing.cyr` +
  `metadata.cyr`, ports `cost/{mod,optimizer}.rs` + the needed slice of
  `provider/metadata.rs`).
  - **Pricing** (`pricing.cyr`): 16-model per-token table (input/output $/M,
    carried ×1000 since no floats), exact → longest-prefix → provider-fallback
    lookup; cost in micro-USD. `POST /v1/cost/estimate` `{model,input_tokens,
    output_tokens}` → estimated cost.
  - **Capability filter** (`metadata.cyr`): per-model tier/context-window/
    vision/tools/system metadata + `classify_complexity` (request → min tier) +
    `meets_requirements`. `POST /v1/cost/recommend` `{input_tokens,
    max_output_tokens,uses_tools,has_vision,has_system_prompt}` classifies the
    request, keeps only configured exact-model routes whose metadata satisfies
    tier/modality/tool/system/context, and returns the **cheapest capable** one
    (with its `tier` + `required_tier`). Wildcard-only/unknown-metadata routes
    are skipped.
  - **Live-verified**: pricing (exact/prefix/fallback/local-free/cheapest-wins);
    capability (plain text → cheapest economy; `has_vision`/large-input → Standard
    tier picks `gpt-4o` over economy models; `uses_tools` excludes a no-tools
    model; over-context → 404). Note: per-token pricing is hardcoded, **not**
    `data/cloud_pricing.json` (which is cloud-GPU $/hour for hardware planning).
- **Prompt compression** (`src/lib/compression.cyr`, `[compression]` config) —
  opt-in whitespace collapse over JSON `content` *values*: runs of whitespace
  (incl. `\n`/`\t`/`\r` escapes) collapse to a single space with leading/trailing
  trim; keys, structure, and other escapes (`\"`, `\\`, `\uXXXX`) are preserved.
  Applied in `handle_chat` before compaction when enabled. Ports the
  whitespace-collapse half of `context/compression.rs`; the stale tool-pair
  prune is deferred to v2.2.4 (needs tool-call message structure). Distinct from
  compaction (which drops whole messages). Unit-tested (6 cases) + live-verified.
- **Cache warming** (`handlers.cyr` `warming_run`/`warming_add`/`_warming_body`,
  `[[warming]]` config) — pre-populates the response cache at startup with
  operator-configured `(model, prompt)` prompts so common requests are instant
  before traffic arrives. Synchronous (the single-threaded runtime has no
  background task); fires one inference per prompt, skips already-cached keys,
  logs `cache warming: N/M entries cached`. Each warmed entry is stored under the
  exact key a client hits by POSTing the canonical body
  `{"model","messages":[{"role":"user","content"}]}` — unit-tested that the
  warmed key equals the client-request key. **Live-verified**: startup warmed
  1/1, client request → cache hit returning the warmed response without
  forwarding.

### Fixed
- **Build output always lands in `build/`** — the `build/` directory is now
  tracked via `build/.gitkeep` so it always exists; the compiler no longer falls
  back to dropping the binary at the repo root when `build/` is absent.
- **Response cache was inert — now wired into `/v1/chat/completions`.** The
  exact-key LRU cache (`cache.cyr`) was configured and exposed via
  `/v1/cache/stats`, but `handle_chat` never read or wrote it, so every request
  hit the provider. Non-streaming requests now compute an exact key
  (`_cache_key` = sha256-hex over model + raw body), short-circuit on a hit
  before rate-limit/budget/forward, and store the response body on a miss.
  **Live-verified**: identical request twice → 1 miss + 1 hit (`/v1/cache/stats`
  `hits:1`). Foundation for cache warming (v2.2.3). Streaming responses are not
  cached.

### Changed
- **Runtime config renamed `hoosh.toml` → `hoosh.cyml`** — matches the project's
  `cyrius.cyml` manifest convention (TOML syntax in a `.cyml` file; still parsed
  by `toml_parse_file`). `load_config` + `/v1/admin/reload` + all docs/comments
  updated. Verified: server loads providers + serves inference from `hoosh.cyml`.
- **Cyrius pin 6.1.20 → 6.1.21**; stdlib re-synced, `cyrius.lock` refreshed. This
  ships sandhi's **native-TLS-by-default** flip.
- **Native TLS is now the default — the gateway builds with no TLS flag**
  (`cyrius build src/main.cyr build/hoosh`). The old opt-in `-D CYRIUS_TLS_NATIVE`
  is gone from CI/release/CLAUDE.md (kept as a deprecated, harmless no-op
  upstream). The crash-prone libssl fdlopen bridge is now the explicit **opt-out**
  via `-D CYRIUS_TLS_LIBSSL`, which hoosh never passes. `main()` still asserts
  native via `sandhi_tls_use_native()` and warns only if a libssl-only build
  disabled it. **Verified**: default build → native active (5 sequential remote
  HTTPS, clean exit, no warning); `-D CYRIUS_TLS_LIBSSL` build → startup warning
  as expected.

## [2.2.2] — 2026-06-10

Data Loss Prevention — the **v2.2.2 parity item**. Ports `dlp/scanner.rs`: the
Cyrius gateway previously had only `ERR_DLP_BLOCKED` + a test stub. Now requests
are scanned for PII/secrets, classified, and routed by privacy policy.

### Added
- **DLP scanner** (`src/lib/dlp.cyr`) — eight built-in PII/secret matchers,
  hand-rolled as byte-level scanners (the Cyrius port carries no regex engine,
  and adding one would be an unnecessary dependency). Patterns and levels mirror
  `BuiltinPatterns::all`: `email`/`ipv4` → Internal, `phone_us` → Confidential,
  `ssn`/`credit_card`/`api_key`/`aws_key`/`github_token` → Restricted. Each
  honours `\b` word boundaries; `dlp_scan_level` returns the highest level found
  and short-circuits on the first Restricted match.
- **Classification levels** (`DlpClass`: Public/Internal/Confidential/Restricted,
  ordered) + `dlp_class_name`/`dlp_class_from_str`.
- **Privacy-aware routing** (`handle_chat`) — when DLP is enabled: Restricted
  content is **blocked** (`403`); Confidential content is forced to a **local
  provider** via the new `router_select_local` (blocked `403` if no local route
  serves the model); Internal and Public pass through. **Live-verified
  end-to-end**: SSN → blocked, US phone on a remote-only model → local-required
  block, clean prompt → normal inference.
- **`[dlp]` config section** (`src/lib/config.cyr`) — `enabled` (default false)
  and `default_level`; documented (disabled) in `hoosh.toml`.

### Tests & benchmarks
- `+19` assertions (**317 pass**): every pattern at its level, highest-level-wins,
  clean/empty → default, disabled → default, and two false-positive guards
  (20-digit run is not a card; leading-dot domain is not an email).
- New bench `dlp_scan_clean_prompt` (~4µs for a typical prompt) — 12 benches total.

## [2.2.1] — 2026-06-09

Provider correctness & completeness — the **v2.2.1 parity items**. Restores the
per-provider token estimation lost in the port, and adds the provider lifecycle
and native-Ollama surface the Rust gateway exposed.

### Added
- **Provider lifecycle endpoints** (`src/lib/handlers.cyr`) — restore the Rust-era
  `ollama.pull_model`/`delete_model` and `synapse.training_status`/`sync_catalog`:
  - `POST /v1/models/pull` `{"model":"..."}` → forwards `POST /api/pull`
    `{"name":"..."}` to the configured Ollama backend.
  - `POST /v1/models/delete` `{"model":"..."}` → forwards `DELETE /api/delete`
    `{"name":"..."}` to Ollama.
  - `POST /v1/training/status` `{"job_id":"..."}` → forwards `GET
    /v1/training/<job>` to the Synapse backend.
  - `POST /v1/catalog/sync` → forwards `POST /v1/catalog/sync` to Synapse.

  Each resolves the target via `_router_find_provider` (first enabled route of
  the provider type), returns `404` when that provider is not configured, and
  `502` when the backend is unreachable or errors. **Live-verified** against a
  mock backend: pull/delete forward the correct method, path, and body.
- **Native Ollama `/api/tags` inbound route** (`handle_ollama_tags`) — lists each
  enabled route's model patterns in Ollama's tags shape
  (`{"models":[{"name","model","modified_at","size","digest","details"}]}`), so
  Ollama clients pointed at the gateway can enumerate available models.
- **Generic local HTTP client** (`http_req_local`, `_build_req_header` in
  `src/lib/http_client.cyr`) — arbitrary method (GET/POST/DELETE) over the
  loopback fast path, with bodyless requests supported. `http_post_local` is now
  a thin `POST` wrapper (no behaviour change for existing callers).

### Changed
- **Per-provider token estimation** (`src/lib/compact.cyr`) — context compaction
  no longer hardcodes 4 chars/token for every provider. Restores
  `ProviderTokenCounter::for_provider` from the Rust `context/tokens.rs`: ratios
  are carried scaled by 10 (no floats) — OpenAI-family 3.8, Anthropic 3.5 (denser
  tokenizer), local LLaMA-family 3.7, others a conservative 4.0. `compact_messages`
  takes the target provider's ratio (via `chars_per_token_x10(route_provider)`),
  so the budget math matches how the destination actually tokenizes. Unknown
  providers keep the prior 4.0 default.

### Tests
- `+20` assertions (**298 pass**): per-provider ratio table + `estimate_tokens`
  (incl. denser-provider-estimates-higher and zero-ratio fallback), lifecycle
  route lookup (found/missing/disabled), and generic GET/DELETE request headers.

## [2.2.0] — 2026-06-09

Remote provider transport — the **v2.2.0 criticals**. Cloud providers were enum +
default-URL entries the loopback-only client could never reach; they now forward
over TLS. All families work end-to-end: OpenAI-compatible (OpenAI, DeepSeek,
Mistral, Groq, Grok, OpenRouter), Anthropic, and Google/Gemini. The sandhi P1
that blocked production is fixed upstream; streaming is now incremental.

### Added
- **Remote forward path** (`src/lib/provider.cyr`) — `https://` routes forward
  through **sandhi**'s high-level client (`sandhi_http_post`: URL parse + DNS +
  TLS + connect), scheme-dispatched by `route_is_remote`. Local `http://`
  backends keep the raw-socket fast path unchanged.
- **Provider auth headers** (`_provider_headers`) — `Authorization: Bearer` for
  OpenAI-compat; `x-api-key` + `anthropic-version` for Anthropic — built from the
  route's `api_key` (loaded from `[providers].api_key` in hoosh.toml).
- **Anthropic `/v1/messages` shaping** (`_build_anthropic_body`,
  `anthropic_extract_text`, `extract_anthropic_tokens`) — `max_tokens` body,
  `content[].text` + `usage.{input,output}_tokens` extraction. **Live-verified
  against the real Anthropic API** (single request returns correctly).
- **Token extraction** for OpenAI `usage` (`extract_openai_tokens`) and Anthropic
  usage; Ollama's `*_eval_count` fields fall back.
- **`$ENV` key expansion** in config (`_config_expand_env`, `src/lib/config.cyr`)
  — `api_key = "$ANTHROPIC_API_KEY"` resolves from the environment, so the secret
  never lives in hoosh.toml. The Anthropic provider block in `hoosh.toml` is
  enabled (key from env).
- **Anthropic system-message hoist** (`_build_anthropic_body`, with byte-level
  JSON helpers `_json_obj_str_field` / `_obj_is_system` / `_json_obj_end`) —
  `role:"system"` turns lift to the top-level `system` field (multiple joined
  with `\n`); remaining `{role,content}` turns pass through unchanged. Anthropic
  rejects a system role inside `messages`, so this is required for system prompts.
- **Google/Gemini shaping** (`_build_gemini_body`, `_gemini_url`,
  `gemini_extract_text`, `extract_gemini_tokens`) — maps OpenAI-style messages to
  Gemini `contents` (assistant→`model`, system→`systemInstruction`), builds the
  model-scoped `:generateContent?key=` URL (key as a **query param**, no auth
  header — `_provider_headers` skips `Authorization` for Google), and extracts
  `candidates[].content.parts[].text` + `usageMetadata.{prompt,candidates}TokenCount`.
- **Incremental remote streaming** — `handle_chat_stream`'s remote branch drives
  `sandhi_http_stream` with a per-event SSE callback (`_remote_stream_cb`),
  decoding each provider's delta (OpenAI `choices[].delta.content`, Anthropic
  `content_block_delta.delta.text`, Gemini `:streamGenerateContent?alt=sse`) and
  re-emitting them as OpenAI SSE chunks. Replaces the buffered one-shot fallback,
  which is retained only for the error-before-first-byte case (so a stream that
  fails early degrades to buffered without duplicating output).
- `tls`, `sandhi`, `mmap`, `dynlib`, `fdlopen` added to `cyrius.cyml` `[deps]`
  (in include order). No `main()` init needed — sandhi/tls self-initialize.
- Tests: `remote_transport` group (scheme dispatch, path/URL/bearer building,
  OpenAI + Anthropic + Gemini token/text extraction, system-message hoist, stream
  request shaping, SSE delta extraction) — **269 tests pass**.

### Changed
- **Cyrius pin 6.1.18 → 6.1.20** (`cyrius.cyml`); stdlib re-synced, `cyrius.lock`
  refreshed (sandhi 1.4.4 → **1.4.5**). This ships the fix for the P1 below.
- **Build the gateway with `-D CYRIUS_TLS_NATIVE`** (CI + release workflows;
  flag precedes the source). Compiles in sandhi's native TLS stack so the binary
  never fdlopen-loads libssl/libcrypto/glibc. `main()` now calls
  `sandhi_tls_use_native()` at startup and prints a loud stderr WARNING if the
  native backend is not active (so a dropped flag can't silently regress to the
  crash-prone libssl path). See CLAUDE.md Key Principles + sandhi architecture/004.

### Fixed
- ✅ **Whitespace-tolerant response extraction — OpenAI & Gemini non-streaming
  responses were silently dropped.** Live verification revealed the text/token
  extractors (`ollama_extract_text`, `anthropic_extract_text`, and the
  `extract_*_tokens` scanners) matched compact-only needles (`"content":"`,
  `"prompt_tokens":`). OpenAI and Gemini **pretty-print** their REST responses
  (`"content": "ok"`, `"prompt_tokens": 14`), and `atoi` doesn't skip leading
  spaces — so both **text and token extraction returned empty/0** for those
  providers (Anthropic returns compact JSON, which is why it passed earlier).
  Replaced all six extractors with three shared, whitespace-tolerant,
  quote-anchored scanners (`_json_value_pos` / `_json_extract_str` /
  `_json_extract_int`) that skip whitespace around the `:` and require a colon to
  confirm a key — the latter also cleanly skips a string *value* equal to a key
  name (Anthropic's `"type":"text"` vs the real `"text":` field) and subsumes the
  hand-rolled `prompt_eval_count`/`eval_count` disambiguation. Added a
  pretty-printed-JSON regression group (OpenAI/Anthropic/Gemini text + tokens).
  **Live-verified**: `gpt-4o-mini`, `claude-haiku-4-5`, and `gemini-2.5-flash`
  all return through the gateway.
- ✅ **Remote-transport repeated-request SIGSEGV — fixed by switching hoosh to the
  native TLS backend.** Live smoke testing revealed the gateway crashed (SIGSEGV)
  on the *2nd–4th* remote request (stream *and* non-stream, intermittent). Root
  cause: hoosh was building **without** `-D CYRIUS_TLS_NATIVE`, so it ran on the
  deprecated libssl fdlopen bridge; the fault was inside the loaded libssl/glibc
  TLS layer (`cmp …,%fs:…` — the brk-malloc/TLS-arena family of the upstream P1).
  Building native (flag + `sandhi_tls_use_native()`) means **no libssl is ever
  loaded** (verified: 0 libssl maps), and the crash is gone — 10/10 non-stream and
  8/8 streaming requests to Anthropic succeed with the server staying up.
- ✅ **sandhi P1 repeated-HTTPS SIGSEGV resolved upstream** (cyrius 6.1.20 /
  sandhi 1.4.5). cyrius `alloc.cyr`'s `brk` bump heap collided with glibc malloc's
  `brk` arena (pulled in by `fdlopen` loading libssl); fixed by moving the alloc
  heap onto an anonymous-`mmap` chunk-bump allocator and default-switching sandhi
  to the native TLS backend. (hoosh additionally had to *opt into* native — see
  above — to stop loading libssl at all.)

### Verified (live)
- **All three cloud families live end-to-end through the gateway** (`hoosh infer`,
  native TLS backend): **OpenAI** (`gpt-4o-mini`), **Anthropic**
  (`claude-haiku-4-5`), and **Google/Gemini** (`gemini-2.5-flash`,
  `gemini-flash-latest`) each return correct text. OpenAI and Gemini surfaced the
  pretty-printed-JSON extraction bug above; verified fixed.
- **Anthropic** also verified for the system-message hoist (a 3-word system
  instruction is obeyed, which only works if the system turn is hoisted out of
  `messages`), incremental streaming, and repeated requests — no crash.
- **Config**: `$ENV` key expansion verified live — provider blocks resolve
  `api_key = "$GEMINI_KEY"` / `"$ANTHROPIC_AGNOS_KEY"` / `"$OPENAI_KEY"` from the
  environment; secrets stay out of `hoosh.toml`.
- Gemini auth confirmed as the `?key=` query param (Bearer returns 401); the
  gateway degrades gracefully on a `404`/`429` upstream (unknown model / quota)
  without crashing.

### Notes
- **Deferred — blocked on a sandhi P1:** certificate pinning + optional mTLS for
  local providers. sandhi already has live pinning/mTLS
  (`sandhi_tls_policy_new_pinned` → `sandhi_conn_open_with_policy`), but the
  high-level `sandhi_http_post`/`_stream` client never threads a policy
  (`sandhi_http_options` has no policy field), so it's unreachable without
  hand-rolling HTTP+chunked+SSE over `sandhi_conn`. Filed P1 on sandhi to thread a
  policy through `sandhi_http_options`
  (`sandhi/docs/issues/2026-06-09-https-client-tls-policy-threading.md`).
- Binary grows (~sandhi/tls/libssl); local-only deployments are unaffected at
  runtime — the TLS path is reached only for `https://` routes.

## [2.1.4] — 2026-06-09

Toolchain and dependency refresh. No API or behavior changes.

### Changed
- **Cyrius pin 6.0.57 → 6.1.18** (`cyrius.cyml`). Stdlib re-synced (`cyrius lib
  sync`); `cyrius.lock` refreshed. `cyrius fmt`/`lint`/`vet`/`deny` clean, 242
  tests pass, benchmark suite green under 6.1.18.
- **ai-hwaccel pin 2.3.7 → 2.3.9** — `dist/ai-hwaccel.cyr` bundle re-vendored
  (`cyrius deps`). The vendored `data/cloud_pricing.json` + `data/models.json`
  were re-checked against 2.3.9 and are content-unchanged (`models.json` stays a
  top-level array per the `hardware_data_files` guard).

## [2.1.3] — 2026-06-04

Optional durable persistence via the `patra` embedded SQL DB (stdlib). Opt-in and
fully backward compatible — without `[[storage]]`, hoosh runs in-memory exactly as
before.

### Added
- **`src/lib/storage.cyr`** (new) — patra-backed persistence for the HMAC audit
  chain and token-budget usage. Enabled by `[[storage]] path = "..."` in
  hoosh.toml; tables `audit` + `budgets` created on open.
- **Audit chain durability** — `audit_record` writes each entry through to disk
  (typed `patra_insert_row`, so messages with quotes/commas can't break or
  inject SQL); on startup the chain is rebuilt in id-order with `last_hash` +
  `next_id` restored so new entries continue the existing chain.
- **Token-budget durability** — `pool_commit` persists each pool's `used`;
  restored on startup. Verified end-to-end (`/v1/tokens/report` → restart →
  `used` restored).
- ADR [008-persistence-via-patra](docs/decisions/008-persistence-via-patra.md).
- `*.patra` added to `.gitignore`; commented `[[storage]]` example in hoosh.toml.

### Notes
- patra requires `fl_init()` + `patra_init()` before use — called in `main()`
  before opening storage.
- patra is single-threaded; storage access will need serialization when the
  threaded accept loop lands (next milestone).

## [2.1.2] — 2026-06-04

Structured operational logging via the `sakshi` stdlib module. Internal — no API
or response changes; the CLI surface is untouched.

### Added
- **Structured logging** (`src/lib/logging.cyr`, new) — leveled operational logs
  to **stderr** with timestamps, via sakshi. `hlog_info/warn/error/debug` cstr
  wrappers + `hlog_request(method, path)`. Log points: server startup, per
  request (`http_route`), auth rejections, config reload, chat "no provider"
  (warn) and "backend unreachable" (error), embeddings backend failure.
- **`[[logging]] level = ...`** in hoosh.toml (fatal/error/warn/info/debug/trace;
  default info) → `sakshi_set_level`. Parsed in `config.cyr`.
- Test group `logging_levels` (level-string mapping + set/get round-trip).

### Notes
- The CLI banner / `info` / `help` / `version` output stays on **stdout** as
  plain presentation; operational logs go to **stderr**, so piping stdout stays
  clean.
- `[[logging]]` uses the double-bracket table form because the TOML parser only
  honors `[[table]]` sections today (single-bracket support is a queued
  improvement) — consistent with `[[budgets]]`/`[[providers]]`.

## [2.1.1] — 2026-06-04

Surfaces ai-hwaccel 2.3.7 planning capabilities that the 2.1.0 dep upgrade pulled
in but didn't yet expose. Additive — existing endpoints unchanged.

### Added
- **`POST /v1/hardware/cost`** — cloud instance $/inference recommendations for a
  model size + quantization (ai-hwaccel `cost.cyr`; AWS/GCP/Azure).
- **`POST /v1/hardware/training-estimate`** — training-memory breakdown
  (model/optimizer/activation/total) for a model size + method
  (full/lora/qlora/dpo/…) + target (gpu/tpu/gaudi) (ai-hwaccel `training.cyr`).
- **`GET /v1/hardware/compatible-models`** — catalogue models that fit the
  detected accelerator memory at int8, with headroom % (ai-hwaccel `model.cyr`).
- **`data/cloud_pricing.json` + `data/models.json`** vendored from ai-hwaccel
  (read cwd-relative at runtime; cost/compatible-models degrade to empty if
  absent). `models.json` ships as a **top-level JSON array** — `load_models`
  scans for bare `{…}` objects, so the `{"models":[…]}` wrapper would yield only
  the first model. A test (`hardware_data_files`) guards this shape.

### Changed
- `src/lib/hardware.cyr` header refreshed for the 2.3.7 module set.

### Notes
- ai-hwaccel's threaded detector (`registry_detect_threaded`) was evaluated for
  faster startup but segfaults under hoosh's single-threaded runtime — deferred
  to the concurrency milestone. Startup still uses serial `registry_detect`.
- Still TODO on the 2.1.x line: `/v1/hardware/model-format` and
  `/v1/hardware/requirement-match` (ai-hwaccel `model_format.cyr` /
  `requirement.cyr`).

## [2.1.0] — 2026-06-04

Toolchain & scaffolding modernization to current Cyrius (6.0.x) conventions. No
gateway behavior changes; the binary builds, tests (231/231), and benchmarks
clean under the new pin. Two latent correctness fixes shipped along the way
(audit HMAC + config parsing — see Fixed).

### Changed
- **Cyrius toolchain pin 4.5.0 → 6.0.57.**
- **ai-hwaccel dependency 2.0.0 → 2.3.7**, now consumed as the single-file
  distlib bundle (`[deps.ai-hwaccel] modules = ["dist/ai-hwaccel.cyr"]`,
  vendored to `lib/ai-hwaccel.cyr` and `include`d from `src/main.cyr`) instead
  of the old per-source-module list.
- **Manifest `cyrius.toml` → `cyrius.cyml`** with `version = "${file:VERSION}"`
  interpolation (VERSION is the single source of truth) and a `repository` field.
- **Retired `.cyrius-toolchain`** — the pin now lives only in `cyrius.cyml`.
- **Syscalls go through the `sys_*` stdlib wrappers** (`sys_write`, `sys_read`,
  `sys_close`, `sys_socket`, `sys_connect`, `sys_exit`) instead of raw
  `syscall(N, …)` / bare `SYS_*` enum members (no longer global in 6.x).
- **stdlib deps** now list `ct`, `keccak`, `thread`, `thread_local` explicitly
  (split out of `sigil` / required by the ai-hwaccel bundle; Cyrius does not
  resolve transitive deps).
- **CI/release workflows modernized** — canonical installer reading the pin from
  `cyrius.cyml`, `cyrius lib sync` + `cyrius deps`, and hard `fmt`/`lint`/`vet`
  gates; release verifies tag == VERSION == `${file:VERSION}` and that the
  version is in this changelog.
- **Scripts de-Rusted** — `bench-history.sh` parses `cyrius bench` output (was
  `cargo bench`/criterion); `version-bump.sh` drives VERSION + CLAUDE.md +
  CHANGELOG (was `Cargo.toml`/`cargo generate-lockfile`).
- Whole tree formatted with `cyrius fmt`.

### Added
- **Benchmarks are now a hard, CI-enforced release gate** — CI runs
  `./scripts/bench-history.sh` and fails if the suite does not run or records no
  data (maintainer waiver via `CYRIUS_SKIP_BENCH=1`). Documented in CLAUDE.md.
- ADR [007-cyrius-6-modernization](docs/decisions/007-cyrius-6-modernization.md).

### Fixed
- **Audit chain HMAC** — replaced the removed `hmac_sign` with
  `hmac_sha256(...)` + `hex_encode` (new `_hmac_hex` helper in `audit.cyr`).
- **Config parsing under 6.x** — `toml_get_sections`/`toml_get` now take a
  **cstr** name; `config.cyr` was wrapping every lookup in `str_from(...)`,
  which silently parsed no sections. Stripped the wrappers (21 sites). Matching
  test drift (`vec_new(8)` arity, `ct_eq` → `ct_eq_bytes_lens`) fixed too.
- Stale hardcoded `"version":"2.0.0"` in the `/` response now tracks
  `HOOSH_VERSION`.

### Removed
- Rust-era cruft: `cyrius.toml`, `.cyrius-toolchain`, `tarpaulin-report.json`,
  `tarpaulin.toml`, and Rust/criterion entries in `.gitignore`.

## [2.0.0] — 2026-04-13

Complete rewrite from Rust to Cyrius. Binary drops from multi-MB to 636KB. All core gateway functionality preserved and ported.

### Added — Core Gateway
- **18 Cyrius modules** — types, ratelimit, route, router, budget, cache, metrics, auth, http_server, http_client, provider, compact, audit, retry, hardware, handlers, config, main
- **13 provider backends** — Ollama (native `/api/chat`), LlamaCPP, Synapse, LMStudio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter — all via OpenAI-compatible forwarding (Ollama uses native API)
- **SSE streaming** — `stream:true` in `/v1/chat/completions` proxies NDJSON (Ollama) or SSE (OpenAI-compat) from backend to client as OpenAI-format `chat.completion.chunk` events
- **Provider routing** — Priority, RoundRobin, LowestLatency strategies; model pattern matching with glob (`llama*`, `gpt-*`)
- **Token budget system** — named pools with capacity, reserve/commit lifecycle; `/v1/tokens/check`, `/v1/tokens/reserve`, `/v1/tokens/report`, `/v1/tokens/pools`
- **HMAC-SHA256 audit chain** — cryptographically linked log entries with tamper detection and verification; `/v1/audit` endpoint with chain validation
- **Retry with exponential backoff** — jittered delays (nanosecond clock bits for jitter), configurable max_retries/base_delay_ms/max_delay_ms via `[[retry]]` config section
- **Per-provider rate limiting** — RPM token bucket with continuous refill; `rate_limit` field in `[[providers]]` config
- **Response cache with LRU eviction** — timestamp-based access tracking, evict-oldest-on-full; hit/miss/eviction counters at `/v1/cache/stats`
- **Context compaction** — preserves system message, keeps recent N messages within token budget; runs before inference to prevent oversized requests
- **Bearer token auth** — constant-time comparison via sigil; skips `/v1/health` and `/metrics`
- **CORS** — full preflight handling on all endpoints

### Added — Hardware
- **ai-hwaccel 2.0.0 integration** — git tag dep (kybernet-style), 27 modules for hardware detection across 18 accelerator types (CUDA, ROCm, Metal, Vulkan, TPU, Gaudi, Neuron, Intel NPU, AMD XDNA, etc.)
- **`/v1/hardware`** — device summary JSON (count, memory, best device, all profiles)
- **`/v1/hardware/placement`** — model placement recommendation given model_params and quantization
- **`/v1/hardware/models`** — compatibility matrix for common model sizes (1B–405B) against detected hardware
- **Hardware on startup** — device count and best device shown in server banner and `hoosh info`

### Added — API Endpoints
- `POST /v1/chat/completions` — streaming + non-streaming inference
- `GET /v1/models` — list configured providers
- `GET /v1/health` — first provider connectivity check
- `GET /v1/health/providers` — per-provider health with TCP probe
- `GET /v1/health/heartbeat` — node status
- `POST /v1/embeddings` — routed through provider system (not hardcoded)
- `GET /v1/costs` — request/token counters per provider
- `POST /v1/costs/reset` — reset counters
- `GET /v1/cache/stats` — hit/miss/eviction stats
- `GET /v1/tokens/pools` — pool capacity/usage
- `GET /v1/queue/status` — queue depth
- `GET /v1/audit` — audit chain with verification
- `POST /v1/admin/reload` — hot-reload config
- `GET /v1/hardware`, `POST /v1/hardware/placement`, `GET /v1/hardware/models`
- `GET /metrics` — Prometheus format
- `GET /` — server info

### Added — CLI
- `hoosh serve [port]` — start gateway (default: 8088)
- `hoosh models` — list configured providers with URLs
- `hoosh health` — check provider connectivity
- `hoosh infer <model> <prompt>` — one-shot inference from CLI
- `hoosh info` — system info with hardware summary
- `hoosh help` / `hoosh version`

### Added — Configuration
- `hoosh.toml` with sections: `[[server]]`, `[[providers]]` (type, base_url, priority, models, api_key, rate_limit), `[[budgets]]`, `[[auth]]`, `[[retry]]`, `[[cache]]`
- `cyrius.toml` with `[package]`, `[build]`, `[deps]` (stdlib + ai-hwaccel git tag dep)

### Changed
- **Language**: Rust → Cyrius (cyrius 3.10.0)
- **Binary size**: multi-MB → 636KB
- **Dependencies**: 200+ crates → 29 Cyrius deps (stdlib + ai-hwaccel)
- **HTTP server**: axum/tokio → raw TCP sockets with syscalls
- **Build system**: cargo → `cyrius build`
- **Dep management**: Cargo.toml → cyrius.toml with git tag deps (kybernet-style)

### Removed
- Rust codebase (preserved in `rust-old/` for reference)
- axum, tokio, reqwest, serde, and all Rust dependencies
- Feature flags (all features compiled in)
- OpenTelemetry integration (deferred to v2.1)
- DLP content filtering (deferred to v2.1)
- TLS/mTLS support (blocked on Cyrius TLS lib)
- Audio endpoints (deferred to svara migration)
- Tool calling / MCP bridge (deferred to v2.1)
- Multi-threaded concurrency (single-threaded accept loop)

---

## Rust-era releases (pre-Cyrius port)

See `rust-old/` for source. These versions used Rust + axum + tokio.

- **1.2.0** (2026-04-03) — License change to GPL-3.0, binary size optimization, TLS provider decoupling
- **1.1.0** (2026-03-29) — GPU telemetry heartbeats, heartbeat eviction, majra ConcurrentPriorityQueue
- **1.0.0** (2026-03-27) — Context management, model metadata (63 models), semantic cache, retry manager, batch inference, cost optimizer, DLP scanner, multi-modal support, ai-hwaccel 1.0.0, 613 tests
- **0.23.4** (2026-03-23) — Tool use & MCP via bote/szál, model metadata registry, hot_path benchmarks
- **0.23.3** (2026-03-23) — Sentiment analysis via bhava
- **0.21.5** (2026-03-21) — Auth, rate limiting, TLS pinning, Prometheus, OpenTelemetry, audit chain, health checks, heartbeat, event bus, queue
- **0.21.3** (2026-03-21) — E2E benchmarks, connection tuning, HTTP/2, documentation
- **0.20.4** (2026-03-21) — Benchmark suite, CI/CD pipelines, version management
- **0.20.3** (2026-03-20) — Initial release: 14 backends, routing, caching, budgets, streaming, hardware placement, CLI, 185 tests
