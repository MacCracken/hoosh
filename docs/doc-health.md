---
name: Hoosh Documentation Health
description: Living state of doc currency in the hoosh repo — fresh / point-in-time / open, refreshed as docs are touched
type: state
---

# Documentation Health — hoosh

> **Last refresh**: 2026-07-22 (**v2.5.4** — cache expiry, the fourth closeout
> band). Doc touch: CHANGELOG [2.5.4], roadmap (2.5.4 shipped + a bench
> step-change note so the CSV jump is not misread as a regression; **+a 2.5.11
> item**: `clock_now_ms()` measured at 1.351 µs, a syscall), hoosh.cyml
> (`ttl_secs` comment), CLAUDE.md pin. Gates: 562 tests, 17 benches, clean.
> **Prior**: 2026-07-22 (**v2.5.3** — provider correctness, the third
> closeout band). Doc touch: CHANGELOG [2.5.3], roadmap (2.5.3 marked shipped +
> an embeddings-normalization gap deferred to 2.5.7), CLAUDE.md pin. Gates: 535
> tests, 17 benches, clean.
> **Prior**: 2026-07-22 (**v2.5.2** — request-path fidelity, the second
> closeout band). Doc touch: CHANGELOG [2.5.2], roadmap (2.5.2 marked shipped +
> an Anthropic thinking-vs-sampling note), hoosh.cyml (per-provider knob comment
> incl. `max_tokens_limit`), CLAUDE.md pin. Gates: 515 tests, 17 benches, clean.
> **Prior**: 2026-07-22 (**v2.5.1** — rust-old parity review + the first
> closeout band). Doc touch: **+`development/rust-old-parity-review.md`** (new — the
> parity evidence record, 1,007 behaviors catalogued, 12 findings hand-verified),
> roadmap (**+v2.5.x parity-closeout arc**, 11 bands incl. the 2.5.11 P(-1)
> security/hardening sweep that closes the arc, 2.5.1 marked shipped; parity
> claim in *Shipped* qualified — surface area matched, request path not; missing
> 2.4.13 + 2.5.0 rows added; **+Backlog section** absorbing the deleted root
> `BACKLOG.md`, whose P4 had sat "open" six weeks after 2.4.0 shipped it),
> CHANGELOG [2.5.1], hoosh.cyml (`bind` + a commented `[[auth]] tokens` example),
> CLAUDE.md pin (version-bump). Gates: 482 tests, 17 benches, clean.
> **Prior**: 2026-06-15 (**v2.4.6** — toolchain + dependency refresh;
> Cyrius 6.2.11, ai-hwaccel 2.3.12, bote 2.7.6, majra 2.4.7). Doc touch:
> CHANGELOG [2.4.6], roadmap (shipped row + bump note), state.md (version /
> toolchain / handoff), **ADR-005** (bote 2.7.6 `tool_registry_new` rename),
> CLAUDE.md pin. Gates: 457 tests, 17 benches, clean.
> **Prior**: 2026-06-10 (**v2.4.5** — hardening review; v2.4.x arc complete,
> Cyrius 6.1.31). Doc touch: CHANGELOG [2.4.5], **ADR-011 §2.4.5** (sync-pass audit
> follow-ups), roadmap (shipped row), state.md (arc complete + handoff),
> README/hoosh.cyml (`strategy` option). Gates: 457 tests, 17 benches, clean.
> **Prior**: 2026-06-10 (**v2.4.4** — new backends vLLM/TensorRT-LLM/ONNX).
> Doc touch: CHANGELOG [2.4.3], roadmap (OTLP/scaffolding checked + shipped row),
> overview/hoosh.cyml (otlp https), **+state.md** (volatile-state doc — closes the
> open follow-up), index (+state.md link). New: `fuzz/*.fcyr`,
> `scripts/security-scan.sh`. Gates: 444 tests, 17 benches, clean. **Prior**:
> 2026-06-10 (**v2.4.2** — threaded hardware detection). **Prior**: 2026-06-10 (**v2.4.1** — hardware planning endpoints + Cyrius
> 6.1.30). Doc touch: CHANGELOG [2.4.1], roadmap (hw bullets checked + shipped
> row), overview.md (`hardware.cyr` desc), CLAUDE.md pin. **Prior**: 2026-06-10 (**v2.4.0** — multi-threaded
> accept loop). Doc
> touch: +**ADR-011** (unified worker pool), CHANGELOG [2.4.0], roadmap (2.4.0 →
> shipped row + arc anchor marked done), overview.md (+`pool.cyr`, accept-loop
> line), index.md (+ADR-011). Gates: 436 tests, 17 benches, clean. **Prior**:
> 2026-06-10 (**v2.3.5**, Cyrius 6.1.29). **Full doc-staleness
> sweep** — the user-facing + reference docs had drifted to the Rust era and to
> pre-2.3.x counts. Touched: **README** (stats → ~7,750 L / 30 files / ~2.0 MB /
> 427 tests / 16 benches / pin 6.1.29; +batch/MCP/observability/OTLP; config
> example → real `hoosh.cyml`; test table → prose), **overview.md** (de-Rusted —
> Cyrius module map + AGNOS-distlib deps + current endpoints, replacing the `.rs`
> tree / Rust-crate table / trait types), **roadmap.md** (416 → ~120 L: shipped
> collapsed to a CHANGELOG-pointer table, v2.4.0 mis-ordering + a duplicate line
> fixed, open work reframed as the **v2.4.x concurrency & completeness arc**), **CONTRIBUTING.md** (de-Rusted — cyrius
> fmt/lint/vet/deny + dev loop, replacing cargo/clippy/`Cargo.toml`/MSRV),
> **BACKLOG.md** (P1–P9 reconciled: 7 cleared with release pointers, P4/P5 → roadmap,
> stale "no DNS" D2 retired), **performance.md** (de-Rusted — points at the
> `hoosh.bcyr` + `bench-history.sh` + `benchmarks.md` system, no fabricated
> numbers), **index.md** (ADRs 007–010 + doc-health added; `benchmarks.md` link
> case fixed), and **this file** (new). CHANGELOG + benchmarks.md were already
> current. Gates after sweep: 427 tests, fmt/lint/vet/deny clean, 16 benches.

---

## Currency summary

| Tier | Count | Notes |
|------|-------|-------|
| 🟢 **Fresh** | 11 | Swept this pass or auto-maintained — accurate as of v2.3.5 |
| 🔵 **Point-in-time** | 11 | ADRs — decision records, frozen by design; re-read, not rewritten |
| 🟢 **Policy / legal** | 3 | No version drift (SECURITY, CODE_OF_CONDUCT, LICENSE) |
| ❓ **Open** | 1 | `docs/development/state.md` not yet created (roadmap scaffolding item) |

---

## Fresh (user-facing + reference)

| Doc | Status | Last touched |
|-----|--------|--------------|
| `README.md` | 🟢 Fresh — stats/capabilities/config/tests refreshed | 2026-06-10 sweep |
| `CHANGELOG.md` | 🟢 Canonical — current through 2.3.5 | per release |
| `CLAUDE.md` | 🟢 Fresh — pin 6.1.29 / v2.3.5 | per release (version-bump) |
| `CONTRIBUTING.md` | 🟢 Fresh — de-Rusted to the Cyrius workflow | 2026-06-10 sweep |
| `benchmarks.md` | 🟢 Auto-generated — 17 benches, latest 2026-07-23Z | `bench-history.sh` |
| `docs/index.md` | 🟢 Fresh — all 11 ADRs + doc-health linked | 2026-06-10 sweep |
| `docs/architecture/overview.md` | 🟢 Fresh — Cyrius module map / deps / endpoints | 2026-06-10 sweep |
| `docs/development/roadmap.md` | 🟢 Fresh — v2.5.x arc + absorbed Backlog section | 2026-07-22 (v2.5.1) |
| `docs/development/rust-old-parity-review.md` | 🟢 Fresh — port parity evidence record | 2026-07-22 (new) |
| `docs/development/performance.md` | 🟢 Fresh — bench-system guide → benchmarks.md | 2026-06-10 sweep |
| `docs/doc-health.md` | 🟢 This file | 2026-06-10 (new) |

## Point-in-time (ADRs — frozen by design)

Decision records: accurate as of their date, not rewritten on drift. Re-read on a
slow cadence; supersede with a new ADR or an `## Update` section rather than edit.

| ADR | Status | Note |
|-----|--------|------|
| 001 HTTP Gateway | Accepted | Evergreen rationale |
| 002 Audit Chain | Accepted | Evergreen |
| 003 Majra Messaging | Accepted | Predates the 2.3.4 event-bus use of majra — that landing is in ADR-010; reconcile if 003 is ever revised |
| 004 Auth & Security | Accepted | Mentions TLS pinning — **deferred** (sandhi policy-threading gap; see roadmap) |
| 005 MCP via Bote + Szál | Accepted | szál (58 tools) still pending; `bote_echo` smoke tool live |
| 006 Kavach Tool Sandbox | **Proposed** | Not implemented |
| 007 Cyrius 6 Modernization | Accepted | Historical (toolchain has since advanced to 6.1.29) |
| 008 Persistence via Patra | Accepted | Current |
| 009 Concurrent Batch Inference | Accepted | Current (2.3.1–2.3.3, with Update sections) |
| 010 Observability | Accepted | Current (2.3.4–2.3.5, with Update section) |
| 011 Multi-threaded accept loop | Accepted | Current (2.4.0) |

## Policy / legal

| Doc | Status |
|-----|--------|
| `SECURITY.md` | 🟢 No rust-era drift detected (not deeply reviewed this sweep) |
| `CODE_OF_CONDUCT.md` | 🟢 Policy — evergreen |
| `LICENSE` | 🟢 GPL-3.0-only |

---

## Open follow-ups

- ADR-003 ↔ ADR-010 reconciliation (majra is now also the event-bus substrate) —
  cosmetic; do it only if ADR-003 is otherwise revised.

## Convention

Refresh this file whenever docs are touched: bump the **Last refresh** line (date
+ version + one-line summary of what moved), and adjust any row whose status
changed. Point-in-time docs (ADRs) don't count as "stale" when reality moves past
them — that's what `## Update` sections and successor ADRs are for.
