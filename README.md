# hoosh

**AI inference gateway — written in Cyrius.**

Multi-provider LLM routing, token budgets, caching, and cost tracking. OpenAI-compatible HTTP API with zero external dependencies.

> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI.
> Part of the [AGNOS](https://github.com/MacCracken/agnosticos) ecosystem. Ported from Rust.

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)

---

## Stats

| Metric | Value |
|--------|-------|
| **Language** | Cyrius (pin 6.1.29) |
| **Source** | ~7,750 lines / 30 files (+ 2 vendored distlib bundles) |
| **Binary** | ~2.0 MB (static ELF, x86_64) |
| **Dependencies** | 0 third-party — AGNOS distlibs (ai-hwaccel, bote, majra) + cyrius stdlib |
| **Tests** | 427 assertions, 100 groups, 0 failures |
| **Benchmarks** | 16 operations |
| **Providers** | 17 (9 local, 8 remote) |
| **API** | OpenAI-compatible `/v1/chat/completions` |

### Port comparison (Rust v1.3.0 → Cyrius)

| | Rust | Cyrius | Ratio |
|---|------|--------|-------|
| Source | 22,956 lines / 58 files | ~7,750 lines / 30 files | **~3x fewer** |
| Binary | ~5.1 MB | ~2.0 MB | **~2.5x smaller** |
| Dependencies | 40+ crates | 0 third-party | **Zero third-party** |

---

## What it does

hoosh is the **inference backend** — it routes, caches, rate-limits, and budget-tracks LLM requests across providers. Applications build their AI features on top of hoosh.

| Capability | Details |
|------------|---------|
| **17 LLM providers** | Ollama, llama.cpp, Synapse, LM Studio, LocalAI, vLLM, TensorRT-LLM, ONNX, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter, Whisper |
| **OpenAI-compatible API** | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` |
| **Batch inference** | `POST /v1/batch` — concurrent (in-process worker pool) sync, or `"async":true` → job id with `GET /v1/batch/{id}` progress + cancel |
| **MCP tool server** | `/v1/tools/list` + `/v1/tools/call` (JSON-RPC 2.0 via bote) |
| **Provider routing** | Priority, round-robin, lowest-latency (EMA), direct — with model pattern matching |
| **Tool calling** | Forwards `tools` (OpenAI/Anthropic/Gemini native formats), surfaces unified OpenAI `tool_calls` |
| **Authentication** | Bearer token auth with constant-time comparison (HMAC-SHA256 via sigil) |
| **Rate limiting** | Per-provider RPM limits |
| **Token budgets** | Per-agent named pools with reserve/commit lifecycle |
| **Data Loss Prevention** | PII/secret scanning (8 built-ins) with privacy-aware routing — Restricted blocked, Confidential forced local (opt-in `[dlp]`) |
| **Cost tracking** | Per-provider cost accumulation, reset endpoint |
| **Observability** | Prometheus `/metrics` (incl. per-provider latency histograms), majra event bus + `GET /v1/events/recent`, W3C `traceparent` propagation, opt-in OpenTelemetry OTLP/JSON span export, HMAC-SHA256 audit chain |
| **Health checks** | Per-provider TCP probe via `/v1/health/providers` |
| **Response caching** | Exact-key LRU cache (eviction stats) + opt-in semantic cache (embedding cosine similarity) + startup warming |
| **Hot-reload** | `POST /v1/admin/reload` — re-reads `hoosh.cyml` |
| **Local-first** | Prefers on-device inference; remote APIs as fallback |

---

## Architecture

```
Clients (curl, tarang, daimon, consumer apps)
    |
    v
Auth --> Rate Limiter --> Router (priority, round-robin, lowest-latency)
                              |
    +-------------------------+
    |                         |
    v                         v
Local backends            Remote APIs
(Ollama, llama.cpp, ...)  (OpenAI, Anthropic, DeepSeek, ...)
    |                         |
    +------------+------------+
                 v
    Cache <-- Budget <-- Cost Tracker
                 |
    Metrics <-- Audit Log
```

---

## Quick start

```bash
# Build
cyrius build src/main.cyr build/hoosh

# Start the gateway (default port 8088, bound to 127.0.0.1)
./build/hoosh serve

# Custom port / bind address / config file
./build/hoosh serve --port 9000
./build/hoosh serve --bind 0.0.0.0 --config /etc/hoosh/prod.cyml

# Talk to a RUNNING gateway instead of reading the local config
./build/hoosh models --server http://gateway:8088
./build/hoosh health --server http://gateway:8088    # non-zero exit if degraded
./build/hoosh infer  --server http://gateway:8088 --stream llama3 "hello"

# System info / version
./build/hoosh info
./build/hoosh version
```

`hoosh serve` binds **loopback only** unless you set `[server] bind` (or `--bind`).
It shuts down gracefully on `SIGINT`/`SIGTERM` — the listener closes and in-flight
requests are allowed to finish — and reloads its config on `SIGHUP`. Set
`HOOSH_LOG=debug` for verbose logging.

### OpenAI-compatible API

```bash
curl http://localhost:8088/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:1b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Response:
```json
{
  "id": "chatcmpl-hoosh",
  "object": "chat.completion",
  "model": "llama3.2:1b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello! How can I help?"},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 26,
    "completion_tokens": 8,
    "total_tokens": 34
  }
}
```

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Gateway info |
| GET | `/v1/health` | Health check (probes first provider) |
| GET | `/v1/health/providers` | Per-provider health status |
| GET | `/v1/models` | List configured providers |
| GET | `/api/tags` | List models, native Ollama-compatible shape |
| POST | `/v1/chat/completions` | Inference (OpenAI-compatible) |
| POST | `/v1/batch` | Concurrent batch inference — `{"requests":[…]}` → `{"results":[…]}`; add `"async":true` for a job id |
| GET | `/v1/batch/{id}` | Async batch progress (status, completed/failed, results) |
| POST | `/v1/batch/{id}/cancel` | Cancel an async batch (in-flight items finish) |
| POST | `/v1/embeddings` | Embeddings (forwarded to Ollama) |
| POST | `/v1/models/pull` | Pull a model (forwarded to Ollama `/api/pull`) |
| POST | `/v1/models/delete` | Delete a model (forwarded to Ollama `/api/delete`) |
| POST | `/v1/training/status` | Synapse training job status |
| POST | `/v1/catalog/sync` | Synapse model-catalog sync |
| GET | `/v1/tokens/pools` | Token budget pools |
| GET | `/v1/costs` | Cost tracking summary |
| POST | `/v1/costs/reset` | Reset cost counters |
| POST | `/v1/cost/estimate` | Estimate per-token cost for a model + token profile |
| POST | `/v1/cost/recommend` | Cheapest *capable* model for a request profile (tier/vision/tools/context-aware) |
| GET | `/v1/cache/stats` | Cache hit/miss/eviction stats |
| GET | `/v1/events/recent` | Recent provider events (health/inference/errors/rate-limit) |
| GET | `/v1/tools/list` | List registered MCP tools (JSON-RPC 2.0, via bote) |
| POST | `/v1/tools/call` | Invoke an MCP tool by name (MCP JSON-RPC body, via bote) |
| GET | `/v1/queue/status` | Request queue status |
| GET | `/metrics` | Prometheus metrics (text format) |
| POST | `/v1/admin/reload` | Hot-reload config from `hoosh.cyml` |
| OPTIONS | `*` | CORS preflight |

---

## Configuration

`hoosh.cyml`:

```toml
[server]
port = 8088
# strategy = "priority"   # priority | round-robin | lowest-latency | direct

[cache]
max_entries = 10000
enabled = true

[[providers]]
type = "Ollama"
# base_url = "http://localhost:11434"   # default
priority = 1
models = ["llama*", "mistral*", "qwen*"]

[[providers]]
type = "OpenAi"
api_key = "$OPENAI_API_KEY"             # $ENV expansion — keep secrets out of the file
priority = 10
models = ["gpt-*", "o1-*"]

[[budgets]]
name = "default"
capacity = 100000

[[auth]]
tokens = "your-secret-token"

# Opt-in: OpenTelemetry OTLP/JSON span export to a collector
# [[telemetry]]
# otlp_endpoint = "http://localhost:4318/v1/traces"
```

---

## Testing

```bash
# Run tests (427 assertions across 100 groups)
cyrius test tests/hoosh.tcyr

# Run benchmarks (16 operations)
cyrius bench tests/hoosh.bcyr
```

`tests/hoosh.tcyr` is self-contained (stdlib + the vendored bote-core, mirroring
src logic) and covers: error/HTTP mapping, routing (4 strategies), token budgets,
cache + cost, queue priority, health FSM, rate limiting, metrics, audit chain
(HMAC-SHA256), config (TOML), JSON, token counting, DLP, cosine similarity, auth,
tool-call conversion + pruning, MCP dispatch, batch parsing + lane pool, and
observability (latency buckets, traceparent extraction, OTLP path/attribute JSON).

---

## Who uses this

| Project | Usage |
|---------|-------|
| **[AGNOS](https://github.com/MacCracken/agnosticos)** | System-wide inference gateway |
| **[tarang](https://github.com/MacCracken/tarang)** | Transcription, AI media analysis |
| **[daimon](https://github.com/MacCracken/daimon)** | Inference routing daemon |
| All AGNOS consumer apps | Via daimon or direct HTTP |

---

## Rust reference

The original Rust implementation (v1.3.0, 22,956 lines) is preserved in `rust-old/` for reference during the port. It includes the full test suite (605 tests), Criterion benchmarks, and all 15 provider implementations.

---

## License

GPL-3.0-only. See [LICENSE](LICENSE) for details.
