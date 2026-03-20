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
| **ai-hwaccel** | 0.20.3 (vulkan + rocm + cuda backends) |
| **Model** | llama3.2:1b (Q8_0, ~1.3 GB) |

**GPU notes:** The Cezanne APU (gfx90c) does not support ROCm properly — `libggml-hip.so` crashes during rocBLAS init. Vulkan backend works via `OLLAMA_VULKAN=true` and reports ~33 GB addressable memory (shared system RAM). ROCm requires a discrete AMD GPU (RDNA2+/CDNA).

## Synthetic Benchmarks (`cargo bench --bench routing --bench providers`)

No backends required. Measures gateway overhead.

### Routing (`benches/routing.rs`)

| Benchmark | Time |
|---|---|
| route_select (20 providers, priority) | 220 ns |
| route_round_robin (10 providers) | 112 ns |

### Provider Infrastructure (`benches/providers.rs`)

| Benchmark | Time |
|---|---|
| registry_register (20 routes) | 71 µs |
| registry_lookup (hit) | 36 ns |
| registry_lookup (miss) | 31 ns |
| OpenAiCompatibleProvider::new | 2.2 µs |
| OpenAiCompatibleProvider::new (with API key) | 2.3 µs |
| InferenceRequest construction (simple) | 21 ns |
| InferenceRequest construction (4 messages) | 48 ns |
| Serialize simple request (JSON) | 156 ns |
| Serialize 10-message request (JSON) | 820 ns |

### Cache (`benches/providers.rs`)

| Benchmark | Time |
|---|---|
| cache_get (hit, 1K entries) | 62 ns |
| cache_get (miss) | 25 ns |
| cache_insert (below capacity) | 189 µs |
| cache_insert (at capacity, triggers eviction) | 4.0 µs |
| cache_key (5 messages) | 131 ns |

### Token Budget (`benches/providers.rs`)

| Benchmark | Time |
|---|---|
| budget reserve+report cycle | 32 ns |
| budget check | 16 ns |
| pool available() | 0.24 ns |

### Config Parsing (`benches/providers.rs`)

| Benchmark | Time |
|---|---|
| parse minimal (empty TOML) | 175 ns |
| parse full (3 providers, 2 pools) | 18 µs |
| parse + convert to ServerConfig | 17 µs |

### Hardware Detection (`benches/live_providers.rs`)

| Benchmark | Time | Notes |
|---|---|---|
| HardwareManager::detect | 5.0 s | ai-hwaccel 0.20.3 with full GPU probing (vulkan+rocm+cuda) |
| HardwareManager::summary | 320 ns | |
| recommend_placement (1B params) | 47 ns | |
| recommend_placement (7B params) | 46 ns | |
| recommend_placement (70B params) | 46 ns | |

**Note:** `detect()` takes ~5s because ai-hwaccel 0.20.3 probes VRAM bandwidth, PCIe links, power/thermal, and runs `vulkaninfo` + `nvidia-smi`. This is a one-time startup cost. Prior version (0.19.3 with no backends) took 15µs.

## Live Benchmarks (`cargo bench --bench live_providers`)

Requires `ollama serve` with a pulled model. Measures end-to-end latency including network, tokenization, and inference.

### Ollama — llama3.2:1b (Vulkan iGPU)

100% GPU-offloaded (4.3 GB VRAM via Vulkan shared memory, 32K context).

| Benchmark | Time |
|---|---|
| health_check | 231 µs |
| list_models | 235 µs |
| infer (short, 5 tokens) | 257 ms |
| infer (medium, 50 tokens) | 1.69 s |
| infer (multiturn, 4 msgs, 30 tokens) | 1.05 s |
| stream (50 tokens) | 1.39 s |

### Derived Metrics (Vulkan iGPU, llama3.2:1b)

| Metric | Value |
|---|---|
| Tokens/sec (short prompt) | ~19 tok/s |
| Tokens/sec (medium prompt) | ~30 tok/s |
| Time-to-first-token (estimated) | ~240 ms |
| Gateway overhead per request | < 1 ms |

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

## Benchmark Coverage

| Area | Benchmarks | Suite |
|---|---|---|
| Routing | select, round-robin | `routing.rs` |
| Provider registry | register, lookup hit/miss | `providers.rs` |
| Provider construction | new, new with key | `providers.rs` |
| Request building | simple, multi-message, serialization | `providers.rs` |
| Cache | get hit/miss, insert, eviction, key generation | `providers.rs` |
| Token budget | reserve+report, check, available | `providers.rs` |
| Config parsing | minimal, full, conversion | `providers.rs` |
| Hardware detection | detect, summary, placement (1B/7B/70B) | `live_providers.rs` |
| Ollama inference | health, models, short/medium/multiturn, stream | `live_providers.rs` |

## Adding Results

When benchmarking on new hardware, run `cargo bench` and append a section with your hardware specs. Key variables:

- **GPU vs CPU**: GPU inference is 10-100x faster depending on model size
- **Model size**: 1B vs 7B vs 70B dramatically changes latency
- **Quantization**: Q4_K_M vs FP16 trades quality for speed/memory
- **ai-hwaccel version**: 0.20.3 has significantly more thorough detection than 0.19.x
