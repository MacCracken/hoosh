# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v0.20.3 — Core Gateway + Provider Backends ✅

All items complete. See CHANGELOG.md for details.

---

## v0.21.5 — Server Hardening, Observability & Messaging ✅

All items complete. See CHANGELOG.md for details.

---

## v0.22.3 — Advanced Inference

### Tool use & function calling
- [ ] Unified `ToolCall` abstraction mapping Anthropic/OpenAI/Gemini/Ollama tool formats — extract from secureyeoman's provider implementations
- [ ] Streaming tool call assembly (incremental delta → complete ToolCall)
- [ ] Tool result message type for multi-turn tool use
- [ ] MCP integration via `bote` + `szál` — bote for protocol/dispatch, szál for 58+ built-in MCP tools (file, process, git, network, hash, template, math, system), enabling Claude/Cursor-style tool use through hoosh
- [ ] Workflow-as-tool — invoke szál workflows (DAG steps with retry/rollback) as single MCP tool calls through hoosh
- [ ] Shared tool schema types — depend on bote for `ToolDef`/`ToolRegistry`/`Dispatcher`, szál for `Tool` trait and tool implementations

### Error handling improvements
- [ ] Protocol-level error code mapping — `HooshError::http_status_code()` method for consistent OpenAI-compatible error responses (inspired by bote's `rpc_code()` pattern)
- [ ] Separate metadata registry from routing dispatch — decouple model/provider metadata from routing logic (following bote's registry/dispatch separation)

### Context management
- [ ] Model registry with detailed metadata — context windows (60+ models), capability flags (chat/vision/reasoning/tool_use/code/streaming), performance tiers, cost tiers, extended thinking support — port from secureyeoman's `model-registry.ts`
- [ ] Context compactor — proactive 80% threshold check + conversation summarization before API call, preserving recent turns — port from secureyeoman's `context-compactor.ts`
- [ ] Token counting per provider (tiktoken for OpenAI, Anthropic tokenizer, etc.)

### Caching improvements
- [ ] Semantic cache (embedding similarity via cosine distance, configurable threshold) — port pattern from secureyeoman's `semantic-cache.ts`
- [ ] Cache warming for common prompts
- [ ] Cache statistics API

### Provider acceleration
- [ ] Batch inference manager — concurrent batching with progress tracking, per-prompt timeout, cancellation support — port from secureyeoman's `batch-inference-manager.ts`
- [ ] Retry manager — jittered exponential backoff, retryable vs permanent error classification, Retry-After header support — port from secureyeoman's `retry-manager.ts`
- [ ] Cost optimizer — dynamic model selection based on workload complexity (simple→Haiku, complex→Opus), 30-day forecasting — port from secureyeoman's `cost-optimizer.ts`
- [ ] Prompt compression for long conversations

### Privacy & data classification
- [ ] DLP classification engine — PII regex scanning (email, SSN, credit card), keyword scanning, custom patterns, 4-level classification (Public/Internal/Confidential/Restricted) — port from secureyeoman's `sy-privacy` crate
- [ ] Privacy-aware routing — route sensitive content to local models only, GPU-aware model selection, configurable policy (auto/local-preferred/local-only/cloud-only) — port from secureyeoman's `privacy-router.ts`

### Multi-modal
- [ ] Vision models — image input support (GPT-4V, Claude, Gemini)
- [ ] Multi-modal routing — route text/image/audio to appropriate backends

---

## v0.23.3 — Speech & Audio

### Whisper improvements
- [ ] Model download and management (tiny, base, small, medium, large)
- [ ] Audio format support beyond WAV: MP3, OGG, FLAC
- [ ] Streaming transcription (chunked audio input)
- [ ] Speaker diarization — who-spoke-when

### TTS improvements
- [ ] Native piper-rs integration (blocked on upstream ort compatibility)
- [ ] Multiple voice support with voice selection API
- [ ] Real-time streaming TTS
- [ ] Bark / XTTS integration for voice cloning

---

## v1.0.0 Criteria

All of the following must be true before cutting 1.0:

- [ ] Public API reviewed and marked stable
- [ ] `LlmProvider` trait finalized
- [ ] Core types (`InferenceRequest`, `InferenceResponse`, `ModelInfo`) frozen
- [ ] 90%+ line coverage
- [x] At least 5 provider backends fully implemented and tested (12 done)
- [ ] OpenAI-compatible API passing conformance tests
- [ ] Token budget accounting verified against real provider billing
- [ ] docs.rs documentation complete with examples
- [ ] No `unsafe` blocks without `// SAFETY:` comments
- [ ] `cargo-semver-checks` in CI
- [ ] Whisper transcription tested against reference audio corpus

---

## Post-v1

### New backends
- [ ] **ONNX Runtime** — local ONNX model inference
- [ ] **vLLM** — high-throughput serving with PagedAttention
- [ ] **TensorRT-LLM** — NVIDIA-optimised inference
- [ ] **Candle** — pure-Rust inference (Hugging Face)
- [ ] **Custom model serving** — user-trained models via hoosh plugin API
- [ ] **TPU** — Google TPU inference via libtpu
- [ ] **Gaudi** — Intel Gaudi accelerator backend
- [ ] **Inferentia** — AWS Inferentia/Trainium via neuron SDK
- [ ] **OneAPI** — Intel Arc/Data Center GPU Max via oneAPI
- [ ] **Qualcomm AI 100** — Qualcomm Cloud AI 100 backend
- [ ] **AMD XDNA** — AMD Ryzen AI NPU backend
- [ ] **Metal** — Apple Metal Performance Shaders backend
- [ ] **Vulkan** — Vulkan compute backend
- [ ] **GGUF** — direct GGUF model loading (currently via llama.cpp)

### Ifran integration (library-mode API)
- [ ] **Embeddings trait** — expose embeddings generation through `LlmProvider` so Ifran can drop its per-backend embedding wrappers
- [ ] **Privacy-aware routing** — route-level privacy classification (local-only vs remote-allowed) so Ifran can enforce data residency without its own `BackendRouter`
- [ ] **Backend capability reporting** — expose accelerator type, context length, streaming support, vision support per provider so Ifran can query capabilities without `ai-hwaccel` directly
- [ ] **Programmatic server builder** — `HooshServer::builder()` API for embedding the gateway inside another Axum app (Ifran's API server) without spawning a separate process

### Kavach + Szál integration (sandboxed tool execution)
- [ ] **Externalization gate** — apply kavach's secret scanner (17 patterns: AWS keys, GitHub tokens, JWTs, PII) to tool outputs before returning them through hoosh's inference context, preventing secret leakage through LLMs
- [ ] **Sandbox strength metadata** — tool execution results carry isolation strength scores (Process: 50, gVisor: 70, Firecracker: 90) in API response metadata, enabling agents to make trust decisions
- [ ] **Direct tool execution** — hoosh invokes szál tools in kavach sandboxes: bote for MCP dispatch → szál for tool logic → kavach for isolation → externalization gate for output scanning
- [ ] **Workflow orchestration** — expose szál's workflow engine (DAG execution, step retry/rollback, state machine) as a hoosh API, enabling multi-step agentic tool chains through the gateway

### Advanced features
- [ ] **Embeddings server** — local embedding generation for RAG
- [ ] **Agent memory** — conversation history with vector retrieval

### Platform
- [ ] **WASM target** — browser-based inference client
- [ ] **C FFI bindings** — `hoosh.h` for C/C++ consumers
- [ ] **Python bindings** — PyO3 package

---

## Non-goals

- **Model training** — hoosh is for inference. Training is Ifran's domain.
- **Model storage** — hoosh doesn't manage model files. Ifran, Ollama, or the filesystem own that.
- **Direct GPU compute** — hoosh delegates to backends that own the GPU. It uses ai-hwaccel for detection, not for running kernels.
- **Web UI** — hoosh is an API gateway. Dashboard/UI is a separate application.
