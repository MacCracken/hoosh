# ADR-006: Kavach for Sandboxed Tool Execution

**Status:** Proposed
**Date:** 2026-03-21

## Context

Hoosh's v0.22.3 milestone adds tool use / function calling. When LLMs invoke tools, those tools may:
1. Execute arbitrary code (bash, Python scripts)
2. Access files, network, or credentials
3. Return output that contains secrets (API keys, tokens, PII)

The AGNOS ecosystem has [kavach](https://crates.io/crates/kavach) — a sandbox execution framework with 8 isolation backends (Process, OCI, WASM, gVisor, SGX, SEV, SyAgnos, Firecracker) and quantitative security scoring (0-100).

## Decision

Integrate kavach as an optional dependency for sandboxed tool execution, combined with bote for MCP protocol handling.

### Execution flow

```
Client → hoosh (inference) → LLM returns tool_call
  → bote (MCP dispatch) → kavach (sandboxed execution)
  → kavach externalization gate (secret scanning)
  → hoosh (tool result back to LLM context)
```

### Three integration layers (incremental)

1. **Externalization gate only** (post-v1): Apply kavach's secret scanner to tool outputs flowing through hoosh, even when tools are executed externally. No sandbox dependency — just the scanning patterns.

2. **Sandbox metadata** (post-v1): When tool results arrive with kavach metadata (backend, strength score), hoosh passes them through in the API response so agents can make trust decisions.

3. **Direct tool execution** (post-v1): Hoosh optionally executes tools in kavach sandboxes, using bote for MCP dispatch. This eliminates the need for a separate tool execution service.

## Rationale

- **Defense in depth**: LLMs can be prompt-injected into calling tools that leak secrets. The externalization gate catches this at the gateway level.
- **Backend flexibility**: kavach's 8 backends let operators choose isolation strength vs performance tradeoff.
- **Ecosystem reuse**: kavach is already production-proven in SecureYeoman (279 MCP tools sandboxed).

## Consequences

- kavach is an optional dependency — hoosh works without it for inference-only deployments
- The externalization gate (17 secret patterns) adds scanning overhead to tool result processing
- Direct tool execution makes hoosh a more capable agent runtime, but increases its responsibility surface
