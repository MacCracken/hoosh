# ADR-005: MCP Integration via Bote + Szál

**Status:** Proposed
**Date:** 2026-03-21

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
