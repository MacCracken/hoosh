# Hoosh — Current State

> Refreshed every release. CLAUDE.md is preferences/process (durable); this file
> is **state** (volatile). Per-release detail is canonical in
> [CHANGELOG.md](../../CHANGELOG.md); the forward plan is in [roadmap.md](roadmap.md);
> doc currency is tracked in [doc-health.md](../doc-health.md).

## Current state

| | |
|---|---|
| **Version** | **2.4.5** (v2.4.x — Concurrency & completeness arc, **complete**; see [roadmap.md](roadmap.md)) |
| **Toolchain** | Cyrius pin **6.1.31** (`cyrius.cyml`) |
| **Binary** (x86_64 static ELF) | ~2.04 MB (`CYRIUS_DCE=1` build) |
| **Source** | ~7,900 lines / 31 files (`src/main.cyr` + 30 `src/lib/*.cyr`) + 2 vendored distlib bundles (~5,150 lines) |
| **Tests** | 442 assertions · 102 groups (`tests/hoosh.tcyr`) |
| **Benchmarks** | 17 (`tests/hoosh.bcyr`); CSV history + `benchmarks.md` (release gate) |
| **Fuzz** | `fuzz/*.fcyr` (parser harnesses) |
| **Providers** | 17 (9 local incl. vLLM/TensorRT-LLM/ONNX + Whisper-STT→svara, 8 remote) |
| **ADRs** | 11 (`docs/decisions/`) |
| **Concurrency** | unified 7-worker pool (banks 1..7); accept loop enqueues — [ADR 011](../decisions/011-multithreaded-accept-loop.md) |

## Active cycle — v2.4.x arc

Shipped: 2.4.0 (multi-threaded accept loop), 2.4.1 (hardware planning endpoints),
2.4.2 (threaded hw detection), 2.4.3 (OTLP remote/https + scaffolding),
2.4.4 (new backends — vLLM / TensorRT-LLM / ONNX), **2.4.5 (hardening review)** —
**arc complete**.

Open (post-arc): OTLP nested spans; **upstream-gated** — OTLP/protobuf (cyrius
protobuf lib), cert pinning + connection pooling (sandhi TLS-policy threading),
szál MCP tools; tests/bench split (only if the suite grows).

> **Handoff (2026-06-10):** 2.4.5 is a hardening review — fixed two unlocked
> shared-map iterations from the v2.4.0 sync pass (`cache_stats`/`tokens_pools`,
> crash-risk), made the routing strategy configurable + un-deaded a working
> lowest-latency (explore/exploit, `_lat_lock`), removed dead `_cost_map`, pin →
> 6.1.31. Crypto-bank ceiling (8: main + 7 workers) remains the concurrency cap.
