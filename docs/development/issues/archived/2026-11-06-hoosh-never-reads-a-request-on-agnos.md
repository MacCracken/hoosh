# hoosh never reads a request on AGNOS

**Status:** ✅ RESOLVED in hoosh 2.10.1. The fix is the patch below, keyed on `THREADS_CONCURRENT` throughout, with `otlp_init` guarded and a startup warning, as `wq_serial_turn` and `pool_serial_idle`. See ADR 011's addendum. *(Was: 🔴 OPEN.)*
**Filed:** 2026-11-06, by thoth, against hoosh 2.10.0 on agnos 1.57.17 and cyrius 6.7.18.
**Affects:** every consumer of hoosh on AGNOS. It holds thoth's v1.0 gate 1, rung 2: a real turn on AGNOS passes every gate with the patch below, and fails without it.
**Severity:** hoosh's to rank. On AGNOS the binary starts, says it is serving, and serves nothing.

## What happens

On AGNOS, `hoosh serve` prints `hoosh 2.10.0 serving on :8088` and never reads a request. A client connects, sends, and waits until its own deadline. Nothing reaches the provider.

## Why

`thread_create` has no second thread on AGNOS. `lib/thread_agnos.cyr` runs the body **inline** and returns when it ends (`THREADS_CONCURRENT = 0`). Several hoosh threads are `while (1 == 1)`, so the first one started never returns to `main`:

| Body | Started by | When |
|---|---|---|
| `_coarse_clock_thread` | `coarse_clock_start` (`src/lib/logging.cyr:40`) | always, and it is first |
| `_pool_worker` × 7 | `pool_start` (`src/lib/pool.cyr:221`) | always |
| `_health_thread` | `src/lib/health.cyr:151` | `health_check_interval_secs` > 0 |
| `_cat_timer_thread` | `catalog_start` (`src/lib/catalog_fetch.cyr:254`) | a refresh interval is set |
| `_hw_refresh_thread` | `hw_refresh_start` (`src/lib/hardware.cyr:199`) | a refresh interval is set |
| `_otlp_exporter_thread` | `otlp_init` (`src/lib/otlp_export.cyr:84`) | `otlp_endpoint` is set |

So `main` sits in the coarse clock's `sleep_ms` loop and the accept loop never runs. `sleep_ms`#41 also holds the CPU on AGNOS, which `daimon` learned and replaced with `pause`#14 (its ADR-007). The serving log line comes before all of this, so nothing looks wrong.

## The proof

`agnos/scripts/smoke/spine-turn-smoke.sh`, on a plain kernel, stages thoth, hoosh and daimon on one image with a stub provider on the host:

| | stock 2.10.0 | with the patch |
|---|---|---|
| A request reaches the provider | no | yes |
| `thoth -p hello` | exit 1, `gateway unreachable` | `hello from stub`, exit 0 |
| A tool round through t-ron and daimon | fails | `{"ok":true}` reaches the model |
| `thoth --json` | fails | prints its object |

thoth runs it with `./scripts/agnos-run.sh --turn`. `HOOSH_BIN` names a binary to test. hoosh's own suite stays at 1,727 of 1,727 on Linux with the patch.

## A patch that passes

Tested as written, on a copy of 2.10.0. It keeps one thread on a serial backend: the accept loop handles each job itself.

```diff
--- a/src/lib/logging.cyr   coarse_clock_start: hoosh_now_ms already falls back to the real clock
+    if (THREADS_CONCURRENT == 0) { return 0; }
--- a/src/lib/pool.cyr      pool_start, after wq_init()
+    if (THREADS_CONCURRENT == 0) { return 0; }
--- a/src/lib/health.cyr
-    thread_create(&_health_thread, 0);
+    if (THREADS_CONCURRENT == 1) { thread_create(&_health_thread, 0); }
--- a/src/lib/catalog_fetch.cyr
-        if (catalog_refresh_secs() > 0) {
+        if ((catalog_refresh_secs() > 0) && (THREADS_CONCURRENT == 1)) {
--- a/src/lib/hardware.cyr  hw_refresh_start
+    if (THREADS_CONCURRENT == 0) { return 0; }
--- a/src/main.cyr          the accept loop
-        if (is_err_result(client_t) == 1) { continue; }
+        if (is_err_result(client_t) == 1) {
+            #ifdef CYRIUS_TARGET_AGNOS
+            var sj = wq_pop();
+            if (sj != 0) { job_process(sj); wq_done(); }
+            sys_pause();
+            #endif
+            continue;
+        }
         ...
         wq_push(_job_conn(client_v));
+        #ifdef CYRIUS_TARGET_AGNOS
+        var cj = wq_pop();
+        if (cj != 0) { job_process(cj); wq_done(); }
+        #endif
```

## What this leaves for hoosh to decide

- **`otlp_init` is not gated above.** The smoke does not set an endpoint. It needs the same guard.
- **One request at a time.** A long stream holds the accept loop, so a second client waits. That is what a serial backend can do, and it suits one consumer on one machine, but a batch job runs on the accept loop too.
- **Nothing runs on a timer.** Health probes, catalog refresh and hardware re-detection stay at their startup values. An idle turn in the accept loop could run them.
- **`THREADS_CONCURRENT` reads better than `CYRIUS_TARGET_AGNOS`.** The accept loop uses the target only because `sys_pause` is AGNOS's wait. `lib/thread.cyr`'s header asks for the capability and not the name.
- **A CI lane.** Nothing in hoosh's CI or scripts builds `--agnos`, which is how this went unseen. `roadmap.md` already names it.
