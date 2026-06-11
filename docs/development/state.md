# Hoosh — Current State

> Refreshed every release. CLAUDE.md is preferences/process (durable); this file
> is **state** (volatile). Per-release detail is canonical in
> [CHANGELOG.md](../../CHANGELOG.md); the forward plan is in [roadmap.md](roadmap.md);
> doc currency is tracked in [doc-health.md](../doc-health.md).

## Current state

| | |
|---|---|
| **Version** | **2.4.3** (v2.4.x — Concurrency & completeness arc; see [roadmap.md](roadmap.md)) |
| **Toolchain** | Cyrius pin **6.1.30** (`cyrius.cyml`) |
| **Binary** (x86_64 static ELF) | ~2.04 MB (`CYRIUS_DCE=1` build) |
| **Source** | ~7,900 lines / 31 files (`src/main.cyr` + 30 `src/lib/*.cyr`) + 2 vendored distlib bundles (~5,150 lines) |
| **Tests** | 442 assertions · 102 groups (`tests/hoosh.tcyr`) |
| **Benchmarks** | 17 (`tests/hoosh.bcyr`); CSV history + `benchmarks.md` (release gate) |
| **Fuzz** | `fuzz/*.fcyr` (parser harnesses) |
| **Providers** | 14 (6 local incl. Whisper-STT→svara, 8 remote) |
| **ADRs** | 11 (`docs/decisions/`) |
| **Concurrency** | unified 7-worker pool (banks 1..7); accept loop enqueues — [ADR 011](../decisions/011-multithreaded-accept-loop.md) |

## Active cycle — v2.4.x arc

Shipped: 2.4.0 (multi-threaded accept loop), 2.4.1 (hardware planning endpoints),
2.4.2 (threaded hw detection), **2.4.3 (OTLP remote/https + scaffolding)**.

Next in arc: **2.4.4** new backends (vLLM / TensorRT-LLM / ONNX); **2.4.5**
hardening / refactor / security / optimization review. Upstream-gated:
OTLP/protobuf (cyrius protobuf lib), cert pinning + connection pooling (sandhi
TLS-policy threading).

> **Handoff (2026-06-10):** 2.4.3 adds OTLP remote/`https://` export (worker-routed
> POST so a banked worker does the TLS — the bankless exporter thread can't) plus
> scaffolding: this `state.md`, fuzz harnesses, and a CI security-pattern scan +
> fuzz step. Crypto-bank ceiling (8: main + 7 workers) remains the concurrency cap.
