# Hoosh Roadmap

> **Principle**: Local inference first, remote APIs as fallback. Model-agnostic API — backends are swappable without consumer changes.

Completed items are in [CHANGELOG.md](../../CHANGELOG.md).

**Note**: Speech-to-text and text-to-speech capabilities are migrating to **svara** as a dedicated speech/audio project. Hoosh retains the provider interface but svara owns the audio pipeline.

---

## v1.1.0 — New Backends & Integrations

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

### Kavach + Szál integration (sandboxed tool execution)
- [ ] **Externalization gate** — apply kavach's secret scanner (17 patterns: AWS keys, GitHub tokens, JWTs, PII) to tool outputs before returning them through hoosh's inference context, preventing secret leakage through LLMs
- [ ] **Sandbox strength metadata** — tool execution results carry isolation strength scores (Process: 50, gVisor: 70, Firecracker: 90) in API response metadata, enabling agents to make trust decisions
- [ ] **Direct tool execution** — hoosh invokes szál tools in kavach sandboxes: bote for MCP dispatch → szál for tool logic → kavach for isolation → externalization gate for output scanning
- [ ] **Workflow orchestration** — expose szál's workflow engine (DAG execution, step retry/rollback, state machine) as a hoosh API, enabling multi-step agentic tool chains through the gateway

---

## v1.2.0 — Ifran Integration (library-mode API)

- [ ] **Embeddings trait** — expose embeddings generation through `LlmProvider` so Ifran can drop its per-backend embedding wrappers
- [ ] **Backend capability reporting** — expose accelerator type, context length, streaming support, vision support per provider so Ifran can query capabilities without `ai-hwaccel` directly
- [ ] **Programmatic server builder** — `HooshServer::builder()` API for embedding the gateway inside another Axum app (Ifran's API server) without spawning a separate process

---

## v2.0.0 — Advanced Features

- [ ] **Embeddings server** — local embedding generation for RAG
- [ ] **Agent memory** — conversation history with vector retrieval
- [ ] **WASM target** — browser-based inference client
- [ ] **C FFI bindings** — `hoosh.h` for C/C++ consumers
- [ ] **Python bindings** — PyO3 package

---

## Non-goals

- **Model training** — hoosh is for inference. Training is Ifran's domain.
- **Model storage** — hoosh doesn't manage model files. Ifran, Ollama, or the filesystem own that.
- **Direct GPU compute** — hoosh delegates to backends that own the GPU. It uses ai-hwaccel for detection, not for running kernels.
- **Web UI** — hoosh is an API gateway. Dashboard/UI is a separate application.
- **Audio pipeline** — speech-to-text and text-to-speech processing is svara's domain. Hoosh provides the provider interface; svara owns capture, decoding, and synthesis.
