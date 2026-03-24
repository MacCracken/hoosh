# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

---

## v0.24.0 — Context, Caching & Privacy

### Context management
- [ ] Model registry with detailed metadata — context windows (60+ models), capability flags, performance/cost tiers
- [ ] Context compactor — proactive 80% threshold check + conversation summarization
- [ ] Token counting per provider (tiktoken for OpenAI, Anthropic tokenizer, etc.)

### Caching improvements
- [ ] Semantic cache (embedding similarity via cosine distance, configurable threshold)
- [ ] Cache warming for common prompts
- [ ] Cache statistics API

### Provider acceleration
- [ ] Batch inference manager — concurrent batching with progress tracking, cancellation
- [ ] Retry manager — jittered exponential backoff, retryable vs permanent error classification
- [ ] Cost optimizer — dynamic model selection based on workload complexity
- [ ] Prompt compression for long conversations

### Privacy & data classification
- [ ] DLP classification engine — PII regex scanning, keyword scanning, custom patterns, 4-level classification
- [ ] Privacy-aware routing — route sensitive content to local models only

### Multi-modal
- [ ] Vision models — image input support (GPT-4V, Claude, Gemini)
- [ ] Multi-modal routing — route text/image/audio to appropriate backends

---

## v0.25.0 — Speech & Audio

### Cross-platform audio preprocessing

- [ ] **macOS: CoreAudio for STT audio preprocessing** — replace
  PipeWire-only audio capture with CoreAudio (`AudioQueue` /
  `AVAudioEngine`) for microphone input on macOS. Required for Whisper
  transcription on non-Linux hosts.
- [ ] **Windows: WASAPI for audio preprocessing** — WASAPI loopback and
  microphone capture via `windows-rs` for STT input on Windows.
- [ ] **Cross-platform: abstract audio capture behind platform trait** —
  `AudioCapture` trait with `open_device()`, `read_frames()`,
  `list_devices()`. Feature-gated backends: `pipewire` (Linux, default),
  `coreaudio` (macOS), `wasapi` (Windows). Consumers call the same API
  regardless of platform.

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
