# Engineering Backlog

Items identified during code audit (2026-03-20). Sorted by priority.

Audit rounds completed: 3 (2026-03-20). All CRITICAL and HIGH items fixed.

## Medium Priority

### M1: Cache key collision risk
- **Location**: `src/cache/mod.rs`
- **Issue**: Cache uses simple `String` keys without namespacing. Different requests with identical prompts but different models/contexts could share cached responses.
- **Fix**: Use composite key like `format!("{model}:{prompt_hash}")` or hash the full request.

### M2: API key logging risk on config parse failure
- **Location**: `src/config.rs:167-169`
- **Issue**: If `hoosh.toml` parsing fails, the error message may include inline API keys in the TOML content.
- **Fix**: Sanitize error messages before logging, or redact values matching `api_key` patterns.

### M3: Anthropic API version hardcoded
- **Location**: `src/provider/anthropic.rs:11`
- **Issue**: `ANTHROPIC_VERSION = "2023-06-01"` is hardcoded and will break when deprecated.
- **Fix**: Make configurable via config file or environment variable.

### M4: Bounded channel backpressure
- **Location**: `src/client.rs:181`, `src/server.rs`, all provider stream impls
- **Issue**: `mpsc::channel(64)` — slow consumers block producers. May cause upstream timeouts.
- **Fix**: Consider larger buffer, backpressure metrics, or timeout on send.

### M5: Duplicate streaming chunk IDs (OpenAI compat)
- **Location**: `src/server.rs` streaming handler
- **Issue**: All SSE chunks in a stream share the same `chatcmpl-{uuid}` ID. OpenAI sends the same ID per stream, so this is actually correct per spec.
- **Status**: Verified correct — OpenAI uses same ID across chunks within a single stream.

### M6: Silent JSON parse failures in streaming
- **Location**: `src/provider/openai_compat.rs`, `src/provider/anthropic.rs`
- **Issue**: Malformed JSON chunks from providers are silently dropped without logging.
- **Fix**: Add `tracing::warn!` on parse failure so corrupt responses are visible.

### M7: Round-robin counter wrapping with route changes
- **Location**: `src/router.rs`
- **Issue**: Atomic counter grows unbounded. If routes are enabled/disabled at runtime, distribution becomes uneven.
- **Fix**: Reset counter when route configuration changes, or use modular arithmetic directly.

### M8: Cache eviction TOCTOU race (DashMap)
- **Location**: `src/cache/mod.rs`
- **Issue**: Between `entries.len()` check and removal, concurrent inserts can cause cache to briefly exceed max_entries.
- **Impact**: Minor — cache size may temporarily exceed limit by a few entries under high concurrency.
- **Status**: Accepted risk. DashMap doesn't support atomic check-and-remove. The current forced eviction is sufficient for a gateway cache.

### M9: Missing Content-Type validation on SSE responses
- **Location**: `src/provider/openai_compat.rs`, `src/client.rs`
- **Issue**: Streaming parsers don't validate that the response has `Content-Type: text/event-stream`. HTML error pages would be silently parsed as SSE.
- **Fix**: Check `Content-Type` header before starting SSE parsing, return explicit error.

### M10: Missing Content-Type validation on transcription endpoint
- **Location**: `src/server.rs` transcribe handler
- **Issue**: Accepts any body without checking Content-Type header. Should require `audio/wav` or `multipart/form-data` for OpenAI compatibility.
- **Fix**: Add Content-Type validation, support multipart form upload matching OpenAI's API.

## Low Priority

### L1: No temperature/top_p range validation
- **Location**: All providers
- **Issue**: Invalid ranges (e.g. temperature=99.0) are passed through to backends.
- **Fix**: Validate at gateway level: temperature in [0.0, 2.0], top_p in (0.0, 1.0].

### L2: Cache entry cloning overhead
- **Location**: `src/cache/mod.rs`
- **Issue**: Cache returns cloned `String` values. Large responses are expensive to clone.
- **Fix**: Use `Arc<String>` for cache values.

### L3: No request-level tracing/correlation IDs
- **Location**: `src/server.rs`
- **Issue**: No request IDs in logs, making concurrent request debugging difficult.
- **Fix**: Add `x-request-id` header generation, propagate through tracing spans.

### L4: SSE keep-alive not configurable
- **Location**: `src/server.rs`
- **Issue**: `KeepAlive::default()` used without configuration.
- **Fix**: Add `sse_keepalive_secs` to server config.

### L5: list_models fallback hides errors
- **Location**: `src/server.rs` list_models handler
- **Issue**: When a provider's `list_models()` fails, falls back silently to route patterns.
- **Fix**: Already logs at warn level. Consider adding a `warnings` field to the models response.

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

### L10: Grok provider listed but not implemented
- **Location**: `src/provider/mod.rs`
- **Issue**: `ProviderType::Grok` exists in enum but has no provider implementation. Configuring it silently registers nothing.
- **Fix**: Either implement (it's OpenAI-compatible at `api.x.ai`) or remove from enum.

### L11: Whisper hardcoded 16kHz sample rate
- **Location**: `src/provider/whisper.rs`
- **Issue**: Duration calculation assumes 16kHz (`samples.len() / 16000.0`), but WAV may have different sample rate.
- **Fix**: Parse the `fmt ` chunk sample rate and use it for duration calculation.

### L10: Unsafe env var access in tests
- **Location**: `src/config.rs:335-346`
- **Issue**: Uses `unsafe { std::env::set_var }` which can race with concurrent tests.
- **Fix**: Use dependency injection for env var resolution or `#[serial_test]`.
