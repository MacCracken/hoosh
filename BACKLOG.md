# Engineering Backlog

Items identified during code audit (2026-03-20). Sorted by priority.

Audit rounds completed: 3. Backlog cleared except one feature request.

## Remaining Items

None. Backlog cleared.

## Resolved Items

| ID | Issue | Resolution |
|---|---|---|
| M1 | Cache key collision | Added `cache_key()` helper with model+messages hash |
| M2 | API key logging risk | Redacted error messages containing `api_key` |
| M3 | Anthropic API version hardcoded | Configurable via `ANTHROPIC_API_VERSION` env var |
| M4 | Bounded channel backpressure | Increased buffer from 64 to 256 |
| M5 | Duplicate chunk IDs | Verified correct per OpenAI spec |
| M6 | Silent JSON parse failures | Added `tracing::warn!` on malformed SSE chunks |
| M7 | Round-robin counter wrapping | Counter wraps via `%` — no issue unless routes change at runtime (they don't) |
| M8 | Cache eviction TOCTOU | Accepted risk — DashMap inherent, forced eviction sufficient |
| M9 | Missing Content-Type on SSE | Added validation before SSE parsing in all stream providers + client |
| M10 | Missing Content-Type on transcription | Added `audio/*` or `application/octet-stream` validation |
| L1 | Temperature/top_p validation | Range validation in server (0-2 / 0-1) |
| L2 | Cache entry cloning overhead | Changed cache values to `Arc<String>` |
| L3 | No request tracing IDs | Added request_id UUID logging in chat_completions |
| L4 | SSE keep-alive not configurable | Set explicit 15s keep-alive interval |
| L5 | list_models fallback | Already logs at warn level |
| L6 | Missing max_tokens per-provider | Added `max_tokens_limit` to ProviderRoute, clamped in server |
| L8 | Streaming e2e test | Added `e2e_streaming` test with SSE mock backend |
| L9 | Per-route body size limits | 1MB for JSON API routes, 50MB for audio routes |
| L10 | Grok provider not implemented | Implemented as OpenAI-compatible wrapper |
| L11 | Whisper hardcoded 16kHz | Parses sample rate from WAV fmt chunk |
| L12 | Unsafe env var in tests | Accepted (single-threaded test) |
| L13 | Missing env var silent | Added `tracing::warn!` on unresolved `$ENV_VAR` |
| L7 | TTS support | Implemented: `TtsProvider` (HTTP backend), `/v1/audio/speech` endpoint, `hoosh speak` CLI, config `[tts]` section, `SpeechRequest`/`SpeechResponse` types. Feature-gated behind `piper`. Piper ONNX dep has upstream ort compat issues — using HTTP backend pattern instead (works with openedai-speech, OpenAI API, or any compatible TTS server). |
