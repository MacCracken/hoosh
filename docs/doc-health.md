---
name: Hoosh Documentation Health
description: Living state of doc currency in the hoosh repo — fresh / point-in-time / open, refreshed as docs are touched
type: state
---

# Documentation Health — hoosh

> **Last refresh**: 2026-11-05 (**v2.9.22**: CHANGELOG, README stats, state, the roadmap's mirror-drift item marked
> done, `architecture/overview.md`'s new `models.cyr` and `jsonlite.cyr` rows, `jsonlite.cyr`'s stale header). Also
> 2026-11-05: **v2.9.21** (the list of what the test file copied, the six moved pieces' rows), **v2.9.20** (the `wq.cyr` / `pool.cyr` rows, the roadmap's bench-file
> item, the CI coverage step's comment), **v2.9.19** (the `cli.cyr` row). On
> 2026-11-04: **v2.9.18** (the `config.cyr` / `cfgparse.cyr` rows),
> **v2.9.17** (bote 3.8.9 in state), **v2.9.16**, **v2.9.15**, **v2.9.14** (`architecture/overview.md`'s module map
> for the `otlp.cyr` / `otlp_export.cyr` split), **v2.9.13** (`scripts/shutdown-smoke.sh` and its CI step),
> **v2.9.12** (the audit-restore item), **v2.9.11** (the roadmap's layout item), **v2.9.10**, **v2.9.9** (the
> `HW_MAX_MODEL_PARAMS` comment in `hardware.cyr`), **v2.9.8**, **v2.9.7**, **v2.9.6** (the dependency scan),
> **v2.9.5**, **v2.9.4**, **v2.9.3** (the pooling measurement, the sandhi TLS-allocator item), **v2.9.2** (README
> *Hardware data files*; the roadmap re-banded; the rust-old retirement carried out), **v2.9.1** (cyrius 6.7.18; ADR
> 011's crypto-lane update), **v2.9.0** (cyrius 6.7.9; the zugot-recipe step dropped) and **v2.8.2** (cyrius
> 6.6.20); before that 2026-10-09 (**v2.8.1**).
> The last FULL sweep was 2026-07-23 (**v2.5.11**, closing the rust-old parity arc), when every doc below was
> re-read against the source.
>
> **What moved**: `roadmap.md` rewritten as forward-only (557 → 153 lines; ten
> shipped `v2.5.x` band records and the absorbed Backlog tables removed — that
> history is the CHANGELOG's job). `state.md` refreshed to v2.5.11 with the arc
> retrospective. `README.md` stats corrected (pin 6.1.29 → 6.4.62, ~7,750 → ~11,070
> lines, 427 → 663 tests, 16 → 25 benches, +fuzz count), capability table updated
> for sampling controls / health failover / hardware planning / lifecycle, endpoint
> table gained the hardware + audit rows, and the port-comparison binary claim was
> corrected (the Cyrius binary is **larger**, not smaller — the old table said
> ~2.0 MB against a real 15 MB). `overview.md` diagram gained the health/hardware
> routing filters, background threads and cost accumulation; module and endpoint
> tables refreshed. `CLAUDE.md` pin corrected (6.2.37 → 6.4.62). `hoosh.cyml`
> gained `[[dlp_pattern]]`. **This file** was rewritten — its tables claimed
> "accurate as of v2.3.5", pin 6.1.29, and "state.md not yet created" (it exists).
>
> **Prior refreshes** are in the CHANGELOG per release; this file no longer keeps
> a running log of them — that duplication is what made it drift.

---

## Currency summary

| Tier | Count | Notes |
|------|-------|-------|
| 🟢 **Fresh** | 12 | Re-read against source in the 2026-07-23 sweep |
| 🔵 **Point-in-time** | 11 | ADRs — decision records, frozen by design; re-read, not rewritten |
| 🟢 **Policy / legal** | 3 | No version drift (SECURITY, CODE_OF_CONDUCT, LICENSE) |
| ❗ **Open** | 2 | See *Open follow-ups* |

---

## Fresh (user-facing + reference)

| Doc | Status | Last touched |
|-----|--------|--------------|
| `README.md` | 🟢 Fresh — stats, capabilities, endpoints, CLI all verified against source | 2026-07-23 sweep |
| `CHANGELOG.md` | 🟢 Canonical — current through 2.5.11 | per release |
| `CLAUDE.md` | 🟢 Fresh — pin 6.4.62 / v2.5.11 | per release (version-bump) |
| `CONTRIBUTING.md` | 🟢 Fresh — Cyrius workflow (fmt/lint/vet/deny) | 2026-07-23 re-read |
| `hoosh.cyml` | 🟢 Fresh — every config section hoosh reads has a documented example | 2026-07-23 sweep |
| `benchmarks.md` | 🟢 Auto-generated — 25 benches | `bench-history.sh` |
| `docs/index.md` | 🟢 Fresh — all 11 ADRs + doc-health linked | 2026-07-23 re-read |
| `docs/architecture/overview.md` | 🟢 Fresh — module map, data flow, endpoints | 2026-07-23 sweep |
| `docs/development/roadmap.md` | 🟢 Fresh — **forward-only**, banded by release (2.9.x, 2.10.x–2.12.x, unscheduled, upstream-gated); every item re-checked against code and upstream; 2.9.2's quick wins shipped and left it | 2026-11-04 re-band |
| `docs/development/state.md` | 🟢 Fresh — v2.5.11 + arc retrospective | 2026-07-23 sweep |
| `docs/development/rust-old-parity-review.md` | 🟢 Point-in-time — the 2026-07-22 parity diff that drove the arc; not rewritten as items shipped | 2026-07-22 |
| `docs/development/rust-old-retirement.md` | 🟢 Fresh — the 2.7.0 parity check, known non-ports, deletion checklist (carried out in 2.9.2) | 2026-11-04 |
| `docs/development/performance.md` | 🟢 Fresh — bench-system guide → benchmarks.md | 2026-07-23 re-read |
| `docs/doc-health.md` | 🟢 This file | 2026-07-23 rewrite |

## Point-in-time (ADRs — frozen by design)

Decision records: accurate as of their date, not rewritten on drift. Re-read on a
slow cadence; supersede with a new ADR or an `## Update` section rather than edit.

| ADR | Status | Note |
|-----|--------|------|
| 001 HTTP Gateway | Accepted | Evergreen rationale |
| 002 Audit Chain | Accepted | **Re-read needed** — 2.5.7 changed the threat model materially: the signing key moved out of the binary into `[audit] signing_key`, auditing became opt-in, and `audit_verify` gained chain-link verification. ADR-002 predates all three |
| 003 Majra Messaging | Accepted | Predates the 2.3.4 event-bus use of majra — that landing is in ADR-010; reconcile if 003 is ever revised |
| 004 Auth & Security | Accepted | TLS pinning + mTLS implemented in 2.7.0 on sandhi TLS policies (note in the ADR) |
| 005 MCP via Bote + Szál | Accepted | szál (58 tools) still pending; `bote_echo` smoke tool live |
| 006 Kavach Tool Sandbox | **Proposed** | Not implemented |
| 007 Cyrius 6 Modernization | Accepted | Historical (toolchain has since advanced to 6.4.62) |
| 008 Persistence via Patra | Accepted | Current |
| 009 Concurrent Batch Inference | Accepted | Current (2.3.1–2.3.3, with Update sections). Note: `batch_submit`'s deep copy is now load-bearing for the 2.5.11 buffer reuse |
| 010 Observability | Accepted | Current (2.3.4–2.3.5, with Update section) |
| 011 Multi-threaded accept loop | Accepted | Current (2.4.0, §2.4.5). The 7-worker/crypto-bank ceiling still stands |
| 012 Live model catalog | Accepted | New (2.8.0). The catalog asked of each route's provider; capability-shaped requests |

## Policy / legal

| Doc | Status |
|-----|--------|
| `SECURITY.md` | 🟢 Policy — no rust-era drift |
| `CODE_OF_CONDUCT.md` | 🟢 Policy — evergreen |
| `LICENSE` | 🟢 GPL-3.0-only |

---

## Open follow-ups

- **ADR-002 (audit chain) needs a re-read or an `## Update`.** v2.5.7 moved the
  signing key out of the binary, made auditing opt-in, and added chain-link
  verification — the ADR describes the pre-2.5.7 design.
- **ADR-003 ↔ ADR-010 reconciliation** (majra is now also the event-bus
  substrate) — cosmetic; do it only if ADR-003 is otherwise revised.

## Convention

Refresh this file whenever docs are touched: bump the **Last refresh** line (date
+ version + what moved) and adjust any row whose status changed. Do **not** keep a
running per-release log here — that is the CHANGELOG's job, and duplicating it is
what let this file drift. Point-in-time docs (ADRs) don't count as "stale" when
reality moves past them; that's what `## Update` sections and successor ADRs are
for — but a *materially* superseded ADR belongs in *Open follow-ups*.
