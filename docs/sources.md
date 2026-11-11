# Sources

Citations for the specifications and algorithms hoosh implements, by the module that implements them.

## Trace context — `src/lib/trace.cyr`, `src/lib/otlp.cyr`

- **W3C Trace Context** (Level 1), W3C Recommendation. <https://www.w3.org/TR/trace-context/>
  - §3.2, the `traceparent` header: `version-traceid-parentid-flags`, 55 characters for version `00`, its hex
    lowercase (the ABNF's `HEXDIGLC`). hoosh reads a value leniently and writes it in lowercase.
  - The field definitions under §3.2: version `ff` is invalid, and so are an all-zero `trace-id` and an all-zero
    `parent-id`, which is the id of the caller's span. hoosh refuses those values (2.10.3), and forwards its own span
    id as `parent-id`, in the caller's trace and with its flags.
- **OpenTelemetry Protocol (OTLP)**, `opentelemetry/proto/trace/v1/trace.proto`: `Span.trace_id`, `span_id` and
  `parent_span_id`, empty for a root. The JSON mapping writes them as hex strings and 64-bit integers as decimal
  strings. <https://opentelemetry.io/docs/specs/otlp/>

## Ids — `src/lib/trace.cyr` (`_trace_mix`, `_trace_id64`)

- **SplitMix64's finalizer**: Guy L. Steele Jr., Doug Lea and Christine H. Flood, "Fast Splittable Pseudorandom Number
  Generators", OOPSLA 2014, doi:10.1145/2660193.2660195. The constants (`0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`,
  shifts 30, 27, 31) are David Stafford's "Mix13", as Sebastiano Vigna's reference <https://prng.di.unimi.it/splitmix64.c>
  uses them. It is a bijection of 64-bit values, so distinct counter values give distinct ids.
- hoosh applies it twice with a secret key xored in before each, the shape of a two-round key-alternating cipher
  (Even and Mansour, "A Construction of a Cipher from a Single Pseudorandom Permutation", J. Cryptology 10(3), 1997).
  The claim is only that an id neither repeats nor reads as the counter or the clock, not cryptographic strength.
- The keys come from `getrandom(2)` with `GRND_INSECURE`, as `lib/hashmap.cyr` draws its seed, and are not that seed.

## Audit chain — `src/lib/audit.cyr`

- **HMAC**: RFC 2104, with SHA-256 (FIPS 180-4), from sigil. [ADR 002](decisions/002-cryptographic-audit-chain.md).

## In-process ternary inference — `src/lib/tentib_local.cyr`

- **BitNet b1.58**: Shuming Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits", 2024,
  arXiv:2402.17764. The model shape tentib implements: ternary {−1, 0, +1} weights with a per-layer scale, int8
  activations. hoosh serves it; tentib trains and packs it. [ADR 014](decisions/014-in-process-ternary-inference.md).
- **tentib's packed-model format, version 1**: tentib's `docs/architecture/001-packed-model-format.md` (the bytes, the
  decoder's checks and its status codes, and since tentib 1.4.0 flags bit 0, the byte vocabulary). hoosh reads the header
  with `tx_pack_peek` before tentib loads anything (V must be 256), and passes the rest to `tx_pack_deserialize_serving`
  (tentib 1.5.0, its ADR 0004: the same checks and status codes, and only what serving reads, since 2.11.2).
- **Incremental decode**: tentib's `docs/architecture/002-incremental-decode.md`, why a cached step is the same bits as
  the full forward's last row and why the cache stops at T. hoosh shares the one cache between interleaved replies
  ([ADR 014](decisions/014-in-process-ternary-inference.md)'s 2.11.1 update).
- **UTF-8**: The Unicode Standard, Version 15.0, §3.9, Table 3-7 ("Well-Formed UTF-8 Byte Sequences"), for the byte
  ranges a character's second byte may take (E0 A0–BF, ED 80–9F, F0 90–BF, F4 80–8F, else 80–BF); and §3.9's "U+FFFD
  Substitution of Maximal Subparts", the practice hoosh follows: each maximal subpart of an ill-formed sequence becomes
  one U+FFFD. <https://www.unicode.org/versions/Unicode15.0.0/>
- **JSON strings**: RFC 8259 §7, for the escapes a reply's text is written with and a message's content is decoded from.
- **SHA-256** (FIPS 180-4), from sigil: whether a reload's file holds the model already serving.
