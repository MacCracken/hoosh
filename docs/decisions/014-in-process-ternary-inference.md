# ADR-014: In-process ternary inference through tentib

**Status:** Accepted (2.11.0)
**Date:** 2026-11-09

## Context

hoosh is local-first, and every local provider it has is a server on the same host: Ollama, llama.cpp, vLLM and
the rest are reached over a socket (`provider.cyr`'s raw-socket path). murti, AGNOS's own local inference engine,
is meant to sit behind hoosh as another of those, but it is still a Rust tree and has no Cyrius port to depend on.

tentib, AGNOS's ternary (1.58-bit) inference reference, has been a library since 1.3.0. It ships a distlib bundle
(`dist/tentib.cyr`) and, since 1.2.0, a packed-model file format: `tx_pack_serialize` writes a trained model,
`tx_pack_deserialize` loads it under a caller-set memory cap and returns a status code for any malformed byte, and
`tx_fwd_packed` serves it with a multiply-free integer kernel. There is no tentib server to forward to. The choice
was between building one and serving tentib in hoosh's own process.

## Decision

**A provider type `tentib` that serves a packed ternary model in hoosh's process** (`src/lib/tentib_local.cyr`),
from the file a `[[providers]]` block names:

```toml
[[providers]]
type = "tentib"
model_file = "/var/lib/hoosh/model.tpk"
models = ["tentib-hello"]
```

1. **The dependency.** `[deps.tentib]` with `modules = ["dist/tentib.cyr"]`, which brings rosnet and tyche with it
   (akshara is tentib's test-scope dependency and does not come). None of the three shares a function or global name
   with hoosh, its vendored bundles or its stdlib set; a build would warn of a duplicate.
2. **One model per process.** tentib keeps its model in module globals, so one tentib route serves: the first whose
   model loads, and any later tentib route is disabled with the reason logged. Every call into tentib holds
   `_tnt_lock`, **for one decode step**, not across a reply, so streams interleave and no lock is held across a
   socket write.
3. **Reloads.** The file is read at startup and on every reload (`config.cyr`). When it holds the bytes of the model
   serving (the same SHA-256), nothing is loaded and replies in progress go on: an operator reloads to change a key or
   a route far more often than to change the model. A load of another model bumps `_tnt_epoch` under the lock, and a
   reply that sees the epoch change stops: a buffered reply fails and is retried on the new model, and a stream ends
   with an error frame. Neither mixes two models.
4. **Loading treats the file as untrusted.** The read is capped (256 MiB), and the load's memory is capped (4 GiB,
   passed to `tx_pack_deserialize`, which computes the need and refuses before allocating, because a failed
   allocation would end the process). tentib validates every field. A refused load leaves the current model
   serving, and the route whose file was refused is disabled, with tentib's reason in words in the log. Every load
   reads into one buffer (`file_read_whole_into`), grown only for a file larger than any before, so a refused or
   unchanged reload keeps nothing.
5. **Bytes are tokens.** tentib's format v1 carries no vocabulary, so a served model uses the byte-identity one:
   V = 256, token id = byte value. A model with another V is refused at load, before tentib replaces the current
   model (the V header word is read first).
6. **tentib's `guard()` exits the process** on a precondition it refuses. Every value hoosh passes is in range by
   construction: tokens are bytes and V is 256, the window never exceeds the model's T, and nothing runs before a
   load succeeds.
7. **Decoding.** The prompt is every message's content, JSON-decoded, joined by `"\n"`. Decoding is greedy over a
   sliding window of the model's last T positions, to `max_tokens` (128 when unset, never more than 4096) or a 0
   byte. Output bytes become valid UTF-8, each invalid or unfinished sequence turning into U+FFFD, and are
   JSON-escaped. A stream sends each character once it is complete.
8. **What the route is to the rest of hoosh.** Local (DLP-confidential content may use it), healthy exactly when
   its model is loaded, with a static catalog (its patterns are its list), no embeddings (`/v1/embeddings` refuses it
   by name), no tools, and cost 0.

## Consequences

- **Positive.** No second server and no socket: a reply never leaves the process. The served logits are tentib's
  own, bit for bit (the same `tx_fwd_packed` tentib's suite gates). A bad model file cannot take the gateway down;
  it disables one route.
- **Negative.** One tentib model per process, and the worker pool shares one model lock. Each new token costs a full
  forward over the window (tentib has no incremental decode yet): about 395 µs for an 8-token reply on the bench's
  T = 25, C = 16 model. **A load of another model keeps the one it replaces in memory for the life of the process**:
  tentib allocates from the allocator that never frees, so each model loaded costs `tx_model_alloc_bytes(V, T, C, F)`
  for good (802,376 bytes at T = 30, C = 16; the load's log line states it). That figure includes tentib's training
  buffers, not just what serving reads. Greedy only: temperature and top_p are ignored. Today's tentib models are
  toy-scale.
- **Neutral.** The byte-vocabulary rule lasts until tentib's format carries a vocabulary. The memory a changed model
  keeps lasts until tentib has freeable model handles (its 2.0.0 backlog); its 1.4.x serving-only allocation shrinks it.

## Alternatives considered

- **A tentib HTTP server, reached like Ollama.** That needs a second binary, a protocol and its own failure modes,
  for a model hoosh can run itself. Rejected while tentib has no server.
- **Wait for murti.** murti is the intended general local engine, but it has no Cyrius port. When it has one,
  whether ternary models move behind it is a new decision. This ADR does not block it.
- **One model per route.** That would need tentib's state per instance, a breaking tentib change (its 2.0.0 backlog
  item, model handles).
