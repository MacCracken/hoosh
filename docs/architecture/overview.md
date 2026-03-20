# Hoosh Architecture

> AI inference gateway — multi-provider LLM routing, local model serving,
> speech-to-text, and token budget management.
>
> **Name**: Hoosh (Persian: هوش) — intelligence, the word for AI.
> Extracted from the [AGNOS](https://github.com/MacCracken/agnosticos) LLM gateway as a standalone, reusable crate.

---

## Design Principles

1. **Provider-agnostic** — uniform API across 14+ backends (local and remote)
2. **Local-first** — prefer on-device inference when hardware is available; remote APIs as fallback
3. **Model-agnostic inference** — the `LlmProvider` trait abstracts over all backends
4. **Token-aware** — every request tracked against per-agent budgets
5. **OpenAI-compatible** — `/v1/chat/completions` API for drop-in replacement

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│  Clients                                              │
│  (tarang, daimon, agnoshi, tazama, consumer apps)     │
│                                                       │
│  HooshClient ──HTTP──▶ hoosh server (:8088)           │
└──────────────────────────┬───────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────┐
│  Router                                               │
│  ┌──────────────┐  ┌──────────────┐                  │
│  │   Priority   │  │  Round-Robin │  Strategy         │
│  │   Routing    │  │  Routing     │  selection        │
│  └──────┬───────┘  └──────┬───────┘                  │
│         └────────┬─────────┘                          │
│                  ▼                                    │
│  ┌───────────────────────────────────────────┐       │
│  │  Provider Registry                         │       │
│  │                                            │       │
│  │  Local:                                    │       │
│  │    Ollama │ llama.cpp │ Synapse │ LM Studio│       │
│  │    LocalAI │ Whisper (STT)                 │       │
│  │                                            │       │
│  │  Remote:                                   │       │
│  │    OpenAI │ Anthropic │ DeepSeek │ Mistral │       │
│  │    Google │ Groq │ Grok │ OpenRouter       │       │
│  └───────────────────────────────────────────┘       │
│                  │                                    │
│  ┌───────────────▼───────────────────────────┐       │
│  │  Middleware                                 │       │
│  │    Cache → Rate Limiter → Token Budget     │       │
│  └────────────────────────────────────────────┘       │
│                                                       │
│  ai-hwaccel: hardware detection for model placement   │
└───────────────────────────────────────────────────────┘
```

---

## Module Structure

```
src/
├── lib.rs              Public API re-exports
├── main.rs             CLI binary (serve, models, infer, health, info)
├── error.rs            HooshError enum
├── inference/          Core types
│   └── mod.rs          InferenceRequest, InferenceResponse, ModelInfo,
│                       TranscriptionRequest/Response, Message, Role
├── provider/           Backend implementations
│   ├── mod.rs          LlmProvider trait, ProviderType enum
│   ├── ollama.rs       Ollama REST API
│   ├── llamacpp.rs     llama.cpp server API
│   ├── openai.rs       OpenAI API (also: DeepSeek, Groq, OpenRouter, LM Studio)
│   ├── anthropic.rs    Anthropic Messages API
│   ├── mistral.rs      Mistral API
│   ├── google.rs       Google Gemini API
│   ├── synapse.rs      Synapse LLM manager
│   └── whisper.rs      whisper.cpp bindings (feature-gated)
├── client.rs           HooshClient — HTTP client for consumers
├── router.rs           Provider selection, load balancing, fallback
├── cache/
│   └── mod.rs          ResponseCache with TTL eviction
├── budget/
│   └── mod.rs          TokenPool, TokenBudget (per-agent accounting)
└── tests/
    └── mod.rs          Integration tests
```

---

## Key Types

### `LlmProvider` (trait)
Every backend implements this. Abstracts inference, streaming, model listing, and health checks.

### `Router`
Selects which provider handles a request based on model name patterns, priority, and routing strategy (Priority, RoundRobin, LowestLatency, Direct).

### `HooshClient`
HTTP client for downstream consumers. Speaks the OpenAI-compatible API. This is what tarang, daimon, and consumer apps import.

### `TokenBudget`
Named token pools with capacity limits. Supports reserve → commit/release lifecycle for accurate accounting even with streaming.

### `ResponseCache`
Thread-safe DashMap cache with TTL. Keyed on prompt hash. Skips caching for streaming requests.

---

## API Endpoints (server mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Inference (streaming + non-streaming) |
| `/v1/models` | GET | List available models |
| `/v1/health` | GET | Gateway health |
| `/v1/health/providers` | GET | Per-provider health |
| `/v1/tokens/check` | POST | Check token budget |
| `/v1/tokens/reserve` | POST | Reserve tokens |
| `/v1/tokens/report` | POST | Report usage |
| `/v1/tokens/pools` | GET | List token pools |
| `/v1/transcribe` | POST | Speech-to-text (whisper) |

---

## Dependencies

| Crate | Role |
|-------|------|
| [ai-hwaccel](https://crates.io/crates/ai-hwaccel) | Hardware detection for model placement decisions |
| [reqwest](https://crates.io/crates/reqwest) | HTTP client for remote providers |
| [axum](https://crates.io/crates/axum) | HTTP server framework |
| [dashmap](https://crates.io/crates/dashmap) | Thread-safe response cache |
| [whisper-rs](https://crates.io/crates/whisper-rs) | whisper.cpp Rust bindings (optional) |

---

## Consumers

| Project | Usage |
|---------|-------|
| **[AGNOS](https://github.com/MacCracken/agnosticos)** (llm-gateway) | Thin wrapper — routes all LLM traffic through hoosh |
| **[tarang](https://crates.io/crates/tarang)** | Transcription, content description, AI media analysis |
| **[aethersafta](https://github.com/MacCracken/aethersafta)** | Real-time transcription/captioning for streams |
| **[AgnosAI](https://github.com/MacCracken/agnosai)** | Agent crew LLM routing (agnosai-llm crate) |
| **[Synapse](https://github.com/MacCracken/synapse)** | Model management + inference backend |
| **All AGNOS consumer apps** | Via daimon agent runtime or direct HTTP |
