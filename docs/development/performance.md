# Performance Testing & Benchmarks

> Benchmark results, testing matrix, and performance targets for hoosh.

All benchmarks use [Criterion.rs](https://bheisler.github.io/criterion.rs/book/) with statistical
analysis. Results are from the development machine and are relative, not absolute.

Last updated: 2026-03-21

---

## Running Benchmarks

```bash
# All synthetic benchmarks (no backends needed)
cargo bench --bench routing --bench providers

# Live Ollama benchmarks (requires ollama serve)
cargo bench --bench live_providers

# End-to-end through hoosh server (requires ollama serve)
cargo bench --bench e2e

# Hardware detection only
cargo bench --bench live_providers -- hwaccel

# Everything
cargo bench
```

---

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

---

## Benchmark Matrix

### Routing (`benches/routing.rs`)

| Benchmark | Scale | Median |
|---|---|---|
| route_select (priority) | 20 providers | 220 ns |
| route_round_robin | 10 providers | 112 ns |

### Provider Infrastructure (`benches/providers.rs`)

| Benchmark | Scale | Median |
|---|---|---|
| registry_register | 20 routes | 71 us |
| registry_lookup (hit) | 20 routes | 36 ns |
| registry_lookup (miss) | 20 routes | 31 ns |
| OpenAiCompatibleProvider::new | -- | 2.2 us |
| OpenAiCompatibleProvider::new (with key) | -- | 2.3 us |
| InferenceRequest construction | simple | 21 ns |
| InferenceRequest construction | 4 messages | 48 ns |
| Serialize simple request | JSON | 156 ns |
| Serialize 10-message request | JSON | 820 ns |

### Cache (`benches/providers.rs`)

| Benchmark | Scale | Median |
|---|---|---|
| cache_get (hit) | 1K entries | 62 ns |
| cache_get (miss) | 1K entries | 25 ns |
| cache_insert | below capacity | 189 us |
| cache_insert (at capacity, eviction) | 100 entries | 4.0 us |
| cache_key generation | 5 messages | 131 ns |

### Token Budget (`benches/providers.rs`)

| Benchmark | Median |
|---|---|
| budget reserve+report cycle | 32 ns |
| budget check | 16 ns |
| pool available() | 0.24 ns |

### Config Parsing (`benches/providers.rs`)

| Benchmark | Payload | Median |
|---|---|---|
| parse minimal | empty TOML | 175 ns |
| parse full | 3 providers, 2 pools | 18 us |
| parse + convert to ServerConfig | 3 providers, 2 pools | 17 us |

### Hardware Detection (`benches/live_providers.rs`)

| Benchmark | Median | Notes |
|---|---|---|
| HardwareManager::detect | 5.0 s | One-time startup cost (probes VRAM, PCIe, vulkaninfo) |
| HardwareManager::summary | 860 ns | |
| recommend_placement (1B) | 46 ns | |
| recommend_placement (7B) | 47 ns | |
| recommend_placement (70B) | 55 ns | |

### Ollama Direct (`benches/live_providers.rs`)

| Benchmark | Median |
|---|---|
| health_check | 626 us |
| list_models | 649 us |
| infer (short, 5 tokens) | 570 ms |
| infer (medium, 50 tokens) | 2.74 s |
| infer (multiturn, 4 msgs, 30 tokens) | 1.76 s |
| stream (50 tokens) | 1.83 s |

### End-to-End Through Hoosh Server (`benches/e2e.rs`)

| Benchmark | Median | Notes |
|---|---|---|
| e2e_health_check | 51 us | Full round-trip through hoosh |
| e2e_list_models | 324 us | |
| e2e_infer (short, 5 tokens) | 284 ms | |
| e2e_infer (medium, 50 tokens) | 1.85 s | |
| e2e_stream (50 tokens) | 1.46 s | |
| e2e_sequential (3 requests) | 1.32 s | |

### Connection Reuse (`benches/e2e.rs`)

| Benchmark | Median | Notes |
|---|---|---|
| cold (new client per request) | 151 us | TCP connect + request |
| warm (reused connection) | 52 us | **2.9x faster** — pooled connection |

### Concurrency Scaling (`benches/e2e.rs`)

| Concurrent requests | Total | Per-request |
|---|---|---|
| 1 | 56 us | 56 us |
| 4 | 99 us | 25 us |
| 8 | 212 us | 26 us |
| 16 | 306 us | 19 us |

### Gateway Overhead (`benches/e2e.rs`)

| Path | Median (5 tokens) |
|---|---|
| Direct to Ollama | 284 ms |
| Through hoosh server | 264 ms |
| **Overhead** | **~0 ms** (within noise) |

---

## Connection Tuning

All HTTP clients are tuned for low-latency local communication:

| Setting | Value | Why |
|---|---|---|
| TCP_NODELAY | true | Disables Nagle's algorithm — avoids 40ms batching delay |
| tcp_keepalive | 60s | OS-level keepalive probes prevent connection drops |
| pool_idle_timeout | 600s | Keep pooled connections alive 10 min (default 90s) |
| pool_max_idle_per_host | 32 | Allow more concurrent pooled connections |
| HTTP/2 adaptive window | true | Multiplexed requests with adaptive flow control |

---

## Performance Summary by Hot Path

| Hot Path | Operation | Per-call | Budget | Status |
|---|---|---|---|---|
| **Per-request** | Route selection | 220 ns | <1 ms | OK |
| **Per-request** | Registry lookup | 36 ns | <1 ms | OK |
| **Per-request** | JSON serialize (10 msgs) | 820 ns | <1 ms | OK |
| **Per-request** | Budget reserve+report | 32 ns | <1 ms | OK |
| **Per-request** | Cache lookup (hit) | 62 ns | <1 ms | OK |
| **Per-request** | Gateway overhead (e2e) | ~0 ms | <5 ms | OK |
| **Connection** | Cold TCP connect | 151 us | <1 ms | OK |
| **Connection** | Warm pooled reuse | 52 us | <1 ms | OK |
| **Startup** | Hardware detection | 5.0 s | <10 s | OK |
| **Startup** | Config parse | 18 us | <100 ms | OK |

---

## Benchmark Suite

4 benchmark files:

| File | Benchmarks | Backend required |
|---|---|---|
| `benches/routing.rs` | 2 | No |
| `benches/providers.rs` | 15 | No |
| `benches/live_providers.rs` | 10 | Ollama + hwaccel |
| `benches/e2e.rs` | 8 | Ollama |

---

## How to Update This Document

After running benchmarks, update the median values:

```bash
cargo bench 2>&1 | grep -B1 'time:' | grep -v '^--$' | paste - -
```

Copy the median values (middle number in the `[low median high]` range) into
the corresponding table cells. Update the "Last updated" date at the top.
