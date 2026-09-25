# Retiring `rust-old/`

**Date**: 2026-09-25 · **Release**: 2.7.0 · **Compared**: `rust-old/` (Rust v1.3.0, 22,956 lines) against
`src/` (Cyrius, ~13,200 lines)

`rust-old/` is the archived Rust implementation hoosh 2.x was ported from. This document records the
last parity check before it is deleted, the items 2.7.0 ported to close it, and every behavior that was
deliberately not carried over, so nobody needs the Rust tree to know what it did.

After the deletion, the tree stays in git history: `git show 2.7.0:rust-old/src/server/handlers.rs` and
so on. The `rust-old <file>:<line>` citations in `src/` comments and in the CHANGELOG refer to that
tree as of tag **2.7.0**.

## Method

The [2026-07-22 parity review](rust-old-parity-review.md) found the gaps that the 2.5.1–2.5.11 arc
closed. This second pass went back to the Rust tree and checked, against `src/`:

- the route table and methods;
- every config key in `config.rs`;
- the CLI commands and flags in `main.rs`;
- chat request validation;
- the response shapes that rust-old's OpenAI conformance tests assert;
- the model catalog, entry by entry;
- Prometheus metric names;
- embeddings, tools, and the hardware endpoints;
- each item of the July review.

## Ported in 2.7.0

| Gap | rust-old | Now |
|---|---|---|
| `/v1/embeddings` routing and shape | routed by model, local or remote; Ollama `/api/embed` normalized to OpenAI | same; array `input` supported |
| `/v1/models` | real model ids from each provider | real ids: local backends live, remote from the catalog |
| chat `id` / `created` | `chatcmpl-<uuid>`, unix `created` | `chatcmpl-<96 random bits>`, `created`; one per stream |
| `/v1/health` | `version`, `providers_configured` | same |
| model catalog | 65 entries | all 65 ids resolve (67 entries); see below |
| `rate_limit_rpm` | the provider key | accepted alongside `rate_limit` |
| `hoosh infer -m <model> <prompt>` | the syntax | accepted |
| `hoosh_tokens_total{type=…}` | labelled series | same; the 2.x names stay |
| `POST /v1/hardware/models {model?, quantization?}` | per-model lookup | same; GET keeps the size table |
| placement `recommendation` / `cloud_alternatives` | sharding plan + cloud options | `sharding` + `cloud_alternatives` |
| SSE keep-alive | comment every 15 s | local streams; remote see below |

The July review's items were already fixed in 2.5.x and 2.6.x; the CHANGELOG entries name each one.

The catalog resolves every rust-old model id by exact match or longest prefix. Ten entries that 2.5.1
added (codestral, llama3, llama3.1–3.3, mistral, mistral-small, mixtral, pixtral-large, qwen2.5) carry
tier, tool or context values that differ from rust-old's. They are kept: for example, Codestral's 256k
context and tool support on llama3.1+ and qwen2.5 are current where rust-old's figures were not.

## Known non-ports

Each of these is a decision, not an omission.

| rust-old | Why it is not in hoosh 2.x |
|---|---|
| `/v1/audio/transcriptions`, `/v1/audio/speech`, `whisper` / `tts` config, `transcribe` / `speak` CLI | Audio moved to **svara**; hoosh keeps the provider interface only |
| `POST /v1/hardware/format` (a file path) | Became `POST /v1/hardware/model-format` taking raw bytes in 2.4.1: no server-side path access |
| szál's 58 MCP tools, tool discovery and announce, `hoosh_workflow_step_*` metrics | Waiting on a szál Cyrius distlib; `/v1/tools/*` runs on bote with a smoke tool until then |
| Per-provider `tls_pinned_certs`, `client_cert`, `client_key` | sandhi's high-level HTTP client does not thread a TLS policy yet (roadmap, upstream-gated) |
| SSE keep-alive on **remote** streams | sandhi drives the remote stream loop and has no idle hook; local streams have it |
| Live model listing from **remote** providers in `/v1/models` | Deliberate: no outbound call per `/v1/models`; remote routes list the catalog models they match |
| `hoosh_request_duration_seconds{provider,model}`, `hoosh_requests_total{provider,model,status}` | Replaced by the per-provider `hoosh_provider_latency_ms` histogram ([ADR 010](../decisions/010-observability.md)); `hoosh_requests_total` is unlabelled |
| OpenTelemetry via OTLP gRPC (`telemetry.rs`) | OTLP/HTTP+JSON export instead ([ADR 010](../decisions/010-observability.md)) |
| The Rust library crate (`HooshClient`, `lib.rs`) | hoosh 2.x is a binary; consumers use the HTTP API |
| `hoosh.toml` | The config file is `hoosh.cyml` |
| `/v1/hardware` carrying available VRAM, interconnect and environment | Split out to `GET /v1/hardware/telemetry` (2.5.9) |
| `/v1/health/providers` `last_error`; `/v1/audit` `total` / `chain_valid`; `/v1/queue/status` `queued` | Same information under hoosh's names (`consecutive_failures`, `count` / `valid`, `pending` / `processing`); rust-old's queue was never fed, so its `queued` was always 0 |
| Ollama embeddings joining array inputs into one string | Not reproduced: an array `input` returns one embedding per item |

## Deleting `rust-old/`

Nothing in the build, CI, scripts, tests, fuzz targets or coverage reads `rust-old/`; every reference is
a comment or a document. After 2.7.0 is tagged:

1. `git rm -r rust-old/`.
2. Keep the `rust-old <file>:<line>` provenance comments in `src/`, `tests/` and the CHANGELOG. They
   resolve against tag 2.7.0, as stated above.
3. README "Port comparison": keep the Rust figures (22,956 lines / 58 files, ~5.1 MB) as historical
   numbers, and say the Rust tree was removed after 2.7.0.
4. `docs/development/state.md`, `docs/development/roadmap.md`, `docs/index.md`, `docs/doc-health.md`:
   change present-tense mentions of `rust-old/` as a live reference to past tense, and point to this
   document.
5. Comments in `scripts/coverage.sh`, `scripts/bench-live.sh` and `fuzz/*.fcyr` that say "port of rust-old
   …" stay true as provenance and need no change.
6. `rust-old/bench-history.csv` holds one line, and `LINES_OF_RUST.txt` holds the 22,956 the README
   already cites. Nothing else in the tree is data to keep.
