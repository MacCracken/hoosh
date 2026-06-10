# ADR-005: MCP Integration via Bote + Szál

**Status:** Accepted (protocol layer shipped in 2.3.0; szál + kavach pending)
**Date:** 2026-03-21 (proposed) · 2026-06-10 (protocol layer accepted/shipped)

> **Update 2026-06-10 (2.3.0):** the bote protocol layer shipped. See
> [Implementation (2.3.0)](#implementation-230) below. The Rust-era framing in
> the original proposal (crates.io links, `v0.23.4`) is superseded by the Cyrius
> port — bote and szál are Cyrius distlibs, not crates.

## Context

The Model Context Protocol (MCP) is becoming a standard for LLM tool use (adopted by Claude, Cursor, and other AI applications). Hoosh needs to support tool use / function calling across providers, and MCP compatibility would make hoosh a natural integration point for MCP-aware clients.

The AGNOS ecosystem has three relevant crates:
- **[bote](https://crates.io/crates/bote)** — MCP core protocol: JSON-RPC 2.0 handling, `ToolDef`/`ToolRegistry`/`Dispatcher`
- **[szál](https://crates.io/crates/szal)** — Workflow engine + 58 built-in MCP tools (file, process, git, network, hash, template, math, system), implements bote's `Tool` trait
- **[kavach](https://crates.io/crates/kavach)** — Sandbox execution with 8 isolation backends, secret scanning externalization gate

## Decision

Integrate all three as the MCP tool execution stack:

```
Client → hoosh (inference + tool routing)
  → LLM returns tool_call
  → bote (MCP dispatch via JSON-RPC 2.0)
  → szál (tool implementation: 58+ tools, workflow engine)
  → kavach (sandboxed execution, secret scanning)
  → hoosh (tool result back to LLM context)
```

### Ownership boundaries

| Concern | Owner |
|---------|-------|
| JSON-RPC 2.0 protocol, `ToolDef`, `ToolRegistry`, `Dispatcher` | bote |
| `Tool` trait, 58 built-in tools, workflow engine (DAG, retry, rollback) | szál |
| Sandbox isolation (8 backends), secret scanning (17 patterns) | kavach |
| Provider-specific tool format mapping (Anthropic/OpenAI/Gemini/Ollama) | hoosh |
| HTTP routing of tool calls, streaming tool assembly | hoosh |
| Tool result forwarding to LLM context | hoosh |

### Integration layers (incremental)

1. **v0.23.4**: Map provider tool formats to bote's `ToolCall`. Register szál tools with bote's registry. Hoosh exposes MCP-compatible `tools/list` and `tools/call` endpoints.

2. **Post-v1**: Add kavach sandboxing. Tool execution runs inside kavach with externalization gate scanning outputs before they re-enter the inference context.

3. **Post-v1**: Expose szál's workflow engine as a hoosh API — multi-step agentic tool chains through the gateway, with DAG execution, step retry/rollback, and state machine tracking.

## Rationale

- **No duplication**: bote owns protocol, szál owns tools, kavach owns isolation
- **58 tools out of the box**: file ops, git, network, process execution, crypto, templates — ready for LLM tool use without writing custom handlers
- **Workflow orchestration**: szál's DAG engine handles multi-step tool chains (retry, rollback, parallel execution) that would be complex to build in hoosh
- **Security boundary**: kavach's externalization gate prevents secret leakage (API keys, tokens, PII) through tool outputs entering the LLM context

## Consequences

- Adds bote + szál as dependencies (szál is larger: 58 tools, ~6k LOC)
- kavach remains optional (sandboxing only needed for untrusted tool execution)
- szál's roadmap (v0.23) plans to port hoosh's LLM routing as szál tools — bidirectional integration
- Tool count scales: szál targets ~200 tools by v0.23, ~490 by v0.25

## Alternatives Considered

- **Reimplement tools in hoosh**: Duplicates szál's 58 tools and workflow engine
- **No MCP support**: Limits tool use to OpenAI-compatible function calling only
- **Only bote, no szál**: Requires writing all tool implementations from scratch

## Implementation (2.3.0)

The **protocol layer** (bote) shipped in 2.3.0 — the first concrete step of
"Integration layer 1" above. szál (tool implementations) and kavach (sandbox)
remain pending.

**Endpoints** (`src/lib/mcp.cyr`, wired in `src/main.cyr`):

- `GET /v1/tools/list` — synthesizes a JSON-RPC `tools/list` request and returns
  bote's `ToolRegistry` listing verbatim.
- `POST /v1/tools/call` — the request body is an MCP JSON-RPC request, run through
  bote's codec + `Dispatcher`; `initialize` and `tools/list` are also accepted.

`mcp_init` (called from `cmd_serve`) builds the registry + dispatcher and
registers a single built-in `bote_echo` smoke tool so the endpoints are
live-verifiable end-to-end. **szál plugs in here**: when its 58 tools ship as a
Cyrius distlib, register them alongside `bote_echo` in `mcp_init` and they appear
in `tools/list` and dispatch through `tools/call` with no transport changes.

### bote is vendored, not a `[deps.bote]` block

bote is consumed as a single committed file, `src/vendor/bote-core.cyr`
(bote 2.7.3's `[lib.core]` profile — 9 transport-free modules: error, protocol,
jsonx, registry, events, audit, dispatch, codec, schema), **not** via a
`[deps.bote]` entry in `cyrius.cyml`.

**Why.** Unlike ai-hwaccel (whose manifest declares no git sub-deps), bote's
manifest declares `[deps.libro]` + `[deps.majra]` for its *full* bundle's
`events_majra` / `audit_libro` modules. `cyrius deps` resolves those
transitively — vendoring libro/majra → bayan/ganita/agnosys into the compile set,
where the agnos superset collides with bote-core's `registry_new`
("last-definition-wins", a correctness hazard) and trips an `agnosys.cyr`
slice-include compile error. The `bote-core` bundle is fully self-contained (no
includes, no libro/majra symbols), so vendoring just that file sidesteps the
transitive tree entirely.

**Why under `src/`.** `cyrius vet` (cyaudit) trusts the authored `src/` tree and
hash-locked `cyrius.lock` deps; a top-level `vendor/` file reads as *untrusted*
and fails the vet gate. Placing the bundle at `src/vendor/bote-core.cyr` keeps
vet green, and because the fmt/lint CI globs are `src/main.cyr` + `src/lib/*.cyr`,
the generated bundle is excluded from those gates. Re-sync with
`./scripts/sync-bote.sh <tag>`.

### Tested

`mcp_tools` unit group (registry/dispatch/codec via the real bote-core:
list / call / unknown-tool) and `mcp_tools_list` / `mcp_tools_call` benches
(full JSON-RPC parse → dispatch → serialize, ~4 µs / ~9 µs). Live-verified
end-to-end against the running gateway.
