# ADR-016: In-process speech through shabda and svara

**Status:** Accepted (2.12.2)
**Date:** 2026-11-11

## Context

OpenAI's `POST /v1/audio/speech` turns text into audio. AGNOS's speech stack is ported to Cyrius on hoosh's pin:

- **shabda 3.1.1** is grapheme-to-phoneme: text through normalizing, its dictionary and rules, syllables and prosody
  to svara phoneme events. It brings shabdakosh 3.1.1 (the dictionaries) and varna 2.5.2 (the phoneme inventories).
- **svara 3.6.3** is formant synthesis: phoneme events and a voice become f64 samples. It brings hisab 3.4.1, naad
  2.3.3 and goonj 2.3.1.

None of them is a server. As with tentib (ADR 014), the choice was between building one and rendering in hoosh's
own process.

Three facts shaped the design:
- **svara's sequence render keeps everything it makes.** `svara_sequence_render` builds a glottal source, a tract and
  a sample vector for each phoneme and concatenates them into another. All of it comes from the allocator that never
  frees: about 88 bytes per output sample, 4.5 MB for two seconds of speech.
- **svara keeps scratch in module globals between the calls of one render.** `svara_tract_set_formants_from_target`
  writes a target's formants into a module vector and then reads them back; `svara_glottal_apply_rd` does the same
  with a triple. Two workers rendering at once would read each other's formants.
- **shabda's conversion allocates on the default heap** and has no allocator-taking forms.

## Decision

**A provider type `svara` that renders `POST /v1/audio/speech` in hoosh's process** (`src/lib/svara_local.cyr`;
the handler is `handle_audio_speech` in `handlers.cyr`):

```toml
[[providers]]
type = "svara"
models = ["tts-1", "svara"]
```

1. **The dependencies.** Seven `[deps.*]` blocks in shabda's dependency order: hisab, goonj, naad, svara, varna,
   shabdakosh, shabda. Their sidecars name the stdlib leaves they need (ganita, result, simd, slice), which `cyrius
   deps` vendors. hisab's git sakshi 2.5.10 is the stdlib snapshot's file, byte for byte, so `cyrius deps` keeps the
   stdlib's and warns; the warning is benign. One name is shared with hoosh's compile set: goonj's `var ERR_NONE = 0`,
   the same 0 as hoosh's enum member.
2. **The request** is OpenAI's shape:
   - `model` routes it, among svara routes only;
   - `input` is a string of 1 to 4096 bytes, decoded, with no NUL;
   - `voice` is one of svara's profiles: `male` (the default), `female` or `child`. OpenAI's voice names are refused
     with the list, rather than mapped to a profile that is not that voice;
   - `response_format` is `wav` (the default) or `pcm`. OpenAI's default, `mp3`, and `opus`, `aac` and `flac` need an
     encoder AGNOS does not have, and are refused with a 400;
   - `speed` is 0.25 to 4.0;
   - `stream_format` may be `audio`. OpenAI's `sse` is refused: the audio streams as it renders anyway;
   - `instructions` is ignored, since svara takes no style prompt.
3. **The reply** is 16-bit mono PCM at 24 kHz, the rate OpenAI's `pcm` has, in a WAV container or raw. Its length is
   known before the first sample, so it is sent with a `Content-Length`. svara's level is kept: its voices peak near
   0.25 of full scale (about −12 dBFS). A sample is clamped to [−1, 1], scaled by 32767 and rounded half up.
4. **The render is `svara_sequence_render`'s, as a stream.** The plan uses svara's stress pitch (×1.10 primary, ×1.05
   secondary), its durations (stress and consonant-cluster scaling), its anticipatory nasalization, its crossfade
   lengths and its sigmoid crossfade, in its order. Each phoneme is rendered by svara's reusable synthesis context
   (`svara_synthctx_synthesize`, the integration guide's real-time path) into a buffer the worker thread keeps. It is
   crossfaded in the thread's pending buffer, and written to the client 8192 samples at a time once no later fade-in
   can reach it.
   - **One context per request** (1.9 KB), over the thread's buffer, so the same input renders the same bytes,
     whatever the thread rendered before.
   - **`speed` divides the durations and svara's transition window** (the crossfade's floor), so a reply's length
     scales by 1 / speed. A fixed 50 ms floor overlapped whole phonemes at speed 4 and rendered a 0.7 s phrase in
     22 ms. At speed 1 both are svara's, bit for bit.
5. **One lock, never across a write.** Every call into shabda or svara holds `_spk_lock`: one conversion (a request's
   input), or one phoneme's render. Replies interleave phoneme by phoneme, and a client that reads slowly holds no one
   else up. The engine (6.1 MB) and the voices are built once per process, by the first enabled svara route, at
   startup or on the reload that adds one.
6. **What the route is to the rest of hoosh.** Local, healthy exactly when speech is loaded, with a static catalog.
   It serves no text: `router_new` keeps a second array of the routes that do (`route_serves_text`), and every text
   selection walks that one. So a svara route with no `models`, which matches every model, never takes a chat
   request, and the router's hot loops are the ones they were. `GET /v1/models` lists a svara route's literal model
   names (as OpenAI lists `tts-1`), the chat catalog and Ollama's tags list none, and a speech request considers only
   svara routes (`router_select_speech`). DLP applies as to chat; the route is local, so confidential input may use
   it. A request is rate-limited by its route, counted in the metrics with its latency under `svara`, and audited as
   `speech.request`.

## Consequences

- **Positive.**
  - Speech with no second server and no socket: the input never leaves the process.
  - Samples never reach the heap. A reply of any length costs the same per-thread buffers: the synthesis buffer and
    the pending buffer, each grown to the longest phoneme the thread has rendered, and 16 KB of PCM.
  - The output is checked against svara's own code. The suite runs the streamed crossfade against
    `svara_seq_crossfade` bit for bit (including fade-ins longer than their segments), a plan's durations and
    nasalizations against `svara_seq_durations` and `svara_detect_nasalization`, its length against
    `svara_sequence_render`'s, and a whole render against `svara_seq_crossfade` over fresh-context renders.
    `scripts/wire-smoke.py` checks the wire: a WAV a strict reader opens, `pcm` the same samples, four requests at
    once answering the bytes one did.
  - It renders about 48× real time on one core: 50.6 ms for a 2.4 s sentence (`speech_render_sentence`); the plan,
    mostly shabda's conversion, takes 102 µs (`speech_plan_sentence`).
- **Negative.**
  - **A request still keeps what shabda and svara allocate inside**, about 1 KB per input byte
    (`scripts/memory-growth.py --speech`: 9.8 KB a request at 6 bytes of input, 62.6 KB at 60, 408 KB at 400). For a
    378-byte input, shabda's conversion keeps 251 KB, and svara's renders 134 KB: a context and a formant filter for
    each consonant. hoosh's own part, the plan, keeps nothing. Both are filed upstream; until they ship, a gateway
    serving speech should rate-limit its svara routes.
  - The release binary grows by 899 KB (2.97 to 3.87 MB), whether or not a svara route is configured: DCE pads what
    it drops. Less than the doubling that would have called for a build without speech.
  - Speech is serialized at phoneme granularity: renders on seven workers share one lock, about 0.15 ms a phoneme.
  - English only: shabda's other languages have empty dictionaries, and OpenAI's request names no language.
  - hoosh carries a copy of svara's sequence arithmetic (the crossfade and the plan), checked against svara by the
    suite, until svara offers a streaming render.
- **Neutral.** The per-phoneme synthesis is svara's context path, not its free path: svara documents the differences
  (plosive and affricate timing; glottal and noise state carried from phoneme to phoneme).

## Alternatives considered

- **`svara_sequence_render` as it is.** Simplest, and it keeps 88 bytes per sample for good: a long reply is tens of
  megabytes. Rejected.
- **A context per worker thread, kept across requests.** That saves 1.9 KB a request, but the output would depend on
  what the thread rendered before. Rejected for a context per request.
- **A lock per reply.** Simpler, but a client that reads slowly would hold every other speech request.
- **Map OpenAI's voices to svara's profiles.** `alloy` is not svara's male voice; an explicit refusal is clearer.
- **A speech server reached over HTTP.** A second binary and protocol for what hoosh can do itself, as in ADR 014.

## References

- svara `docs/development/integration-guide.md` (*Real-time streaming*), `docs/adr/0006-scope-boundaries.md`;
  `svara_sequence_render`, `svara_seq_crossfade`, `svara_synthctx_*` in `dist/svara.cyr` 3.6.3.
- shabda `README.md` (consuming the distlib, the dependency order), `docs/adr/0001-shabdakosh-declared-optional.md`.
- OpenAI, *Create speech* (`POST /v1/audio/speech`): the request fields and the `pcm` format (24 kHz, 16-bit,
  little-endian).
- Microsoft/IBM, *Multimedia Programming Interface and Data Specifications 1.0* (1991): the RIFF WAVE `fmt ` and
  `data` chunks.
- Issues filed 2026-11-11: shabda `docs/development/issues/2026-11-11-conversion-allocates-on-the-default-heap.md`;
  svara `docs/development/issues/2026-11-11-a-server-render-keeps-what-each-phoneme-allocates.md` and
  `docs/development/issues/2026-11-11-module-scratch-makes-concurrent-renders-unsafe.md`.
- CHANGELOG 2.12.2.
