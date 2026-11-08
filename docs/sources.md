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
