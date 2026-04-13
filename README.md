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
| **Language** | Cyrius (cc3 3.6.7+) |
| **Source** | 1,515 lines / 1 file |
| **Binary** | 479 KB (static ELF, x86_64) |
| **Compile time** | 225ms |
| **Dependencies** | 0 external (stdlib only) |
| **Tests** | 231 assertions, 81 groups, 0 failures |
| **Benchmarks** | 10 operations |
| **Providers** | 14 (6 local, 8 remote) |
| **API** | OpenAI-compatible `/v1/chat/completions` |

### Port comparison (Rust v1.3.0 → Cyrius v2.0.0)

| | Rust | Cyrius | Ratio |
|---|------|--------|-------|
| Source | 22,956 lines / 58 files | 1,515 lines / 1 file | **15x fewer** |
| Binary | ~5.1 MB | 479 KB | **10.6x smaller** |
| Compile | ~15s | 225ms | **67x faster** |
| Dependencies | 40+ crates | 0 | **Zero deps** |

---

## What it does

hoosh is the **inference backend** — it routes, caches, rate-limits, and budget-tracks LLM requests across providers. Applications build their AI features on top of hoosh.

| Capability | Details |
|------------|---------|
| **14 LLM providers** | Ollama, llama.cpp, Synapse, LM Studio, LocalAI, OpenAI, Anthropic, DeepSeek, Mistral, Google, Groq, Grok, OpenRouter, Whisper |
| **OpenAI-compatible API** | `/v1/chat/completions`, `/v1/models`, `/v1/embeddings` |
| **Provider routing** | Priority, round-robin, lowest-latency (EMA), direct — with model pattern matching |
| **Authentication** | Bearer token auth with constant-time comparison (HMAC-SHA256 via sigil) |
| **Rate limiting** | Per-provider RPM limits |
| **Token budgets** | Per-agent named pools with reserve/commit lifecycle |
| **Cost tracking** | Per-provider cost accumulation, reset endpoint |
| **Observability** | Prometheus `/metrics`, HMAC-SHA256 audit chain |
| **Health checks** | Per-provider TCP probe via `/v1/health/providers` |
| **Response caching** | Hashmap cache with eviction stats |
| **Hot-reload** | `POST /v1/admin/reload` — re-reads `hoosh.toml` |
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

# Start the gateway (default port 8088)
./build/hoosh serve

# Custom port
./build/hoosh serve 9000

# System info
./build/hoosh info

# Version
./build/hoosh version
```

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
| POST | `/v1/chat/completions` | Inference (OpenAI-compatible) |
| POST | `/v1/embeddings` | Embeddings (forwarded to Ollama) |
| GET | `/v1/tokens/pools` | Token budget pools |
| GET | `/v1/costs` | Cost tracking summary |
| POST | `/v1/costs/reset` | Reset cost counters |
| GET | `/v1/cache/stats` | Cache hit/miss/eviction stats |
| GET | `/v1/queue/status` | Request queue status |
| GET | `/metrics` | Prometheus metrics (text format) |
| POST | `/v1/admin/reload` | Hot-reload config from `hoosh.toml` |
| OPTIONS | `*` | CORS preflight |

---

## Configuration

`hoosh.toml`:

```toml
[[server]]
port = 8088

[[cache]]
max_entries = 10000
enabled = true

[[providers]]
type = "Ollama"
base_url = "http://localhost:11434"
priority = 1
models = ["llama*", "mistral*", "qwen*"]

[[providers]]
type = "OpenAi"
base_url = "https://api.openai.com"
priority = 10
models = ["gpt-*", "o1-*"]

[[budgets]]
name = "default"
capacity = 100000

[[budgets]]
name = "agent-1"
capacity = 50000

[[auth]]
tokens = "your-secret-token"
```

---

## Testing

```bash
# Run tests (231 assertions)
cyrius test tests/hoosh.tcyr

# Run benchmarks (10 operations)
cyrius bench tests/hoosh.bcyr
```

### Test coverage

| Module | Assertions |
|--------|-----------|
| Error types | 22 |
| Provider types | 13 |
| Router (4 strategies) | 32 |
| Token budget | 19 |
| Cache + stats | 14 |
| Cost tracker | 15 |
| Queue (5-tier priority) | 15 |
| Health FSM | 30 |
| Rate limiter | 5 |
| Metrics | 8 |
| Provider registry | 8 |
| Audit chain (HMAC-SHA256) | 12 |
| Config (TOML) | 8 |
| JSON parse/build | 5 |
| Inference types | 13 |
| Token counting | 8 |
| DLP scanning | 6 |
| Cosine similarity | 3 |
| Auth (constant-time) | 3 |

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
