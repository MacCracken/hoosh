# Engineering Backlog

Items identified during code audit (2026-03-20). Sorted by priority.

Audit rounds completed: 3. All CRITICAL, HIGH, and most MEDIUM/LOW items fixed.

## Remaining Items

### M4: Bounded channel backpressure
- **Location**: `src/client.rs`, `src/server.rs`, all provider stream impls
- **Issue**: `mpsc::channel(64)` — slow consumers block producers. May cause upstream timeouts.
- **Fix**: Consider larger buffer, backpressure metrics, or timeout on send.

### M7: Round-robin counter wrapping with route changes
- **Location**: `src/router.rs`
- **Issue**: Atomic counter grows unbounded. If routes are enabled/disabled at runtime, distribution becomes uneven.
- **Fix**: Reset counter when route configuration changes, or use modular arithmetic directly.

### M8: Cache eviction TOCTOU race (DashMap)
- **Location**: `src/cache/mod.rs`
- **Issue**: Between `entries.len()` check and removal, concurrent inserts can cause cache to briefly exceed max_entries.
- **Status**: Accepted risk. DashMap doesn't support atomic check-and-remove. Forced eviction is sufficient.

### M9: Missing Content-Type validation on SSE responses
- **Location**: `src/provider/openai_compat.rs`, `src/client.rs`
- **Issue**: Streaming parsers don't validate `Content-Type: text/event-stream`. HTML error pages would be silently parsed as SSE.
- **Fix**: Check `Content-Type` header before starting SSE parsing, return explicit error.

### M10: Missing Content-Type validation on transcription endpoint
- **Location**: `src/server.rs` transcribe handler
- **Issue**: Accepts any body without checking Content-Type. Should require `audio/wav` or `multipart/form-data` for OpenAI compatibility.
- **Fix**: Add Content-Type validation, support multipart form upload matching OpenAI's API.

### L2: Cache entry cloning overhead
- **Location**: `src/cache/mod.rs`
- **Issue**: Cache returns cloned `String` values. Large responses expensive to clone.
- **Fix**: Use `Arc<String>` for cache values.

### L3: No request-level tracing/correlation IDs
- **Location**: `src/server.rs`
- **Issue**: No request IDs in logs, making concurrent request debugging difficult.
- **Fix**: Add `x-request-id` header generation, propagate through tracing spans.

### L4: SSE keep-alive not configurable
- **Location**: `src/server.rs`
- **Issue**: `KeepAlive::default()` used without configuration.
- **Fix**: Add `sse_keepalive_secs` to server config.

### L6: Missing max_tokens per-provider limits
- **Location**: `src/inference/mod.rs`
- **Issue**: No per-provider max_tokens enforcement. Requests may be silently rejected by backends.
- **Fix**: Add provider-specific limits to ProviderRoute config, validate before dispatch.

### L7: TTS (text-to-speech) support
- **Issue**: Whisper handles STT but no TTS counterpart exists.
- **Fix**: Add TTS provider trait and endpoint (`/v1/audio/speech`) matching OpenAI's API.

### L8: Streaming end-to-end test
- **Issue**: E2e tests cover non-streaming only. No test for SSE streaming through the full server stack.
- **Fix**: Add mock that returns SSE data, test client `infer_stream()` through hoosh server.

### L9: Per-route body size limits
- **Location**: `src/server.rs`
- **Issue**: 50MB `DefaultBodyLimit` applies to all routes. Chat completions should be much smaller (~1MB).
- **Fix**: Apply route-specific limits using axum's `RequestBodyLimit` layer.

## Resolved Items

| ID | Issue | Resolution |
|---|---|---|
| M1 | Cache key collision | Added `cache_key()` helper with model+messages hash |
| M2 | API key logging risk | Redacted error messages containing `api_key` |
| M3 | Anthropic API version hardcoded | Made configurable via `ANTHROPIC_API_VERSION` env var |
| M5 | Duplicate chunk IDs | Verified correct per OpenAI spec |
| M6 | Silent JSON parse failures | Added `tracing::warn!` on malformed SSE chunks |
| L1 | Temperature/top_p validation | Added range validation in server (0-2 / 0-1) |
| L5 | list_models fallback | Already logs at warn level |
| L10 | Grok provider not implemented | Implemented as OpenAI-compatible wrapper |
| L11 | Whisper hardcoded 16kHz | Parses sample rate from WAV fmt chunk |
| L12 | Unsafe env var in tests | Accepted (single-threaded test) |
| L13 | Missing env var silent | Added `tracing::warn!` on unresolved `$ENV_VAR` |
