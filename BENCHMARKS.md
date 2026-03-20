# hoosh Performance Matrix

Benchmark results from `cargo bench`. Live benchmarks require running provider backends.

## Test Environment

| | |
|---|---|
| **CPU** | AMD Ryzen 7 5800H (8C/16T) |
| **RAM** | 60 GB DDR4 |
| **GPU** | AMD Radeon Vega (Cezanne iGPU, 3 GB VRAM, shared memory via Vulkan) |
| **OS** | Linux 6.12.71-1-lts x86_64 |
| **Rust** | 1.89 (edition 2024) |
| **Model** | llama3.2:1b (Q8_0, ~1.3 GB) |

**GPU notes:** The Cezanne APU (gfx90c) does not support ROCm properly — `libggml-hip.so` crashes during rocBLAS init. Vulkan backend works via `OLLAMA_VULKAN=true` and reports ~33 GB addressable memory (shared system RAM). ROCm requires a discrete AMD GPU (RDNA2+/CDNA).

## Synthetic Benchmarks (`cargo bench`)

No backends required. Measures gateway overhead.

### Routing (`benches/routing.rs`)

| Benchmark | Time |
|---|---|
| route_select (20 providers, priority) | 199 ns |
| route_round_robin (10 providers) | 104 ns |

### Provider Infrastructure (`benches/providers.rs`)

| Benchmark | Time |
|---|---|
| registry_register (20 routes) | 66 µs |
| registry_lookup (hit) | 36 ns |
| registry_lookup (miss) | 36 ns |
| OpenAiCompatibleProvider::new | 2.3 µs |
| OpenAiCompatibleProvider::new (with API key) | 2.3 µs |
| InferenceRequest construction (simple) | 19 ns |
| InferenceRequest construction (4 messages) | 47 ns |
| Serialize simple request (JSON) | 162 ns |
| Serialize 10-message request (JSON) | 758 ns |

### Hardware Detection (`benches/live_providers.rs -- hwaccel`)

Results vary significantly based on which ai-hwaccel backends are compiled in.

| Benchmark | No backends | With vulkan+rocm+cuda |
|---|---|---|
| HardwareManager::detect | 15-24 µs | 31 ms |
| HardwareManager::summary | 94-200 ns | 316 ns |
| recommend_placement (1B params) | 44-89 ns | 47 ns |
| recommend_placement (7B params) | 44-90 ns | 47 ns |
| recommend_placement (70B params) | 44-85 ns | 47 ns |

**Note:** With backends enabled, `detect()` takes ~31 ms because it actually probes GPU hardware via ROCm/Vulkan drivers. Without backends it's a no-op returning CPU info only. This cost is paid once at startup. The placement functions are pure CPU math — the small differences are cache effects.

## Live Benchmarks (`cargo bench --bench live_providers`)

Requires `ollama serve` with a pulled model. Measures end-to-end latency including network, tokenization, and inference.

### Ollama — llama3.2:1b (CPU-only)

| Benchmark | Time | Notes |
|---|---|---|
| health_check | 230 µs | GET /api/tags |
| list_models | 228 µs | GET /api/tags + parse |
| infer (short, 5 tokens) | 432 ms | "Say hi." → ~5 completion tokens |
| infer (medium, 50 tokens) | 2.96 s | "Explain Rust ownership" → ~50 tokens |
| infer (multiturn, 4 msgs, 30 tokens) | 1.65 s | System + 3 turns → ~30 tokens |
| stream (50 tokens) | 1.61 s | NDJSON streaming, ~50 tokens |

### Derived Metrics (CPU-only, llama3.2:1b)

| Metric | Value |
|---|---|
| Tokens/sec (short prompt) | ~12 tok/s |
| Tokens/sec (medium prompt) | ~17 tok/s |
| Time-to-first-token (estimated) | ~400 ms |
| Gateway overhead per request | < 1 ms |

### Ollama — llama3.2:1b (Vulkan iGPU — AMD Radeon Vega)

100% GPU-offloaded (4.3 GB VRAM via Vulkan shared memory, 32K context).

| Benchmark | Time | vs CPU |
|---|---|---|
| health_check | 223 µs | ~same |
| list_models | 227 µs | ~same |
| infer (short, 5 tokens) | 413 ms | -4.4% |
| infer (medium, 50 tokens) | 2.47 s | -16.5% |
| infer (multiturn, 4 msgs, 30 tokens) | 1.39 s | -15.6% |
| stream (50 tokens) | 1.39 s | -13.4% |

### Derived Metrics (Vulkan iGPU, llama3.2:1b)

| Metric | Value | vs CPU |
|---|---|---|
| Tokens/sec (short prompt) | ~12 tok/s | ~same |
| Tokens/sec (medium prompt) | ~20 tok/s | +18% |
| Time-to-first-token (estimated) | ~390 ms | -2% |
| Gateway overhead per request | < 1 ms | ~same |

### CPU vs Vulkan iGPU Summary

The Vulkan iGPU provides a modest 13-17% speedup on longer generations. Short prompts see minimal benefit since prompt evaluation dominates. A discrete GPU (RDNA2+, CUDA) would show much larger gains — expect 5-20x improvement on a dedicated card.

## Running Benchmarks

```bash
# All synthetic benchmarks (no backends needed)
cargo bench --bench routing --bench providers

# Hardware detection only
cargo bench --bench live_providers -- hwaccel

# Live Ollama benchmarks (requires ollama serve)
cargo bench --bench live_providers

# Everything
cargo bench
```

## Adding Results

When benchmarking on new hardware, run `cargo bench` and append a section below with your hardware specs. Key variables that affect results:

- **GPU vs CPU**: GPU inference is 10-100x faster depending on model size
- **Model size**: 1B vs 7B vs 70B dramatically changes latency
- **Quantization**: Q4_K_M vs FP16 trades quality for speed/memory
- **Batch size**: Concurrent requests benefit from GPU batching
