# An error with a quote in it is not JSON — RESOLVED

**Status:** ✅ **RESOLVED** in hoosh 2.10.2 ([CHANGELOG](../../../../CHANGELOG.md), *Fixed*). Every row below parses, row 4 decodes to the provider's sentence, and the spoof writes nothing; `scripts/wire-smoke.py` runs them against a real gateway on every CI run, and 12 of its error checks fail on 2.10.1. The seen-in-passing 404 on a local stream is fixed too, and a permanent refusal now ends a stream's retries, as it ends a buffered chat's. Three things went differently from what is filed here: `http_error` could call the escaper as the tree stood, as a function resolves wherever it is defined, so nothing moved and `catalog_add_esc` folded into `sb_add_json_escaped`; the decode is the stdlib's `str_json_unescape_cstr`, as `_json_unesc_span` leaves `\u003c` escaped, which is how Go writes `<`; and two messages did reach the writers holding a request's raw JSON slice (`Token pool '…'` and `No provider configured for model '…'`), so both decode first. The clip also falls on a whole UTF-8 character now. **agnosai can drop `_agnosai_hoosh_detail_a`'s scan.**
**Placement:** 2.10.2.
**Discovered:** 2026-11-06 during agnosai 2.3.1 (its roadmap B33, ADR 030), when a model an Ollama-shaped provider
no longer had came back to agnosai as a bare `hoosh returned HTTP 502`.
**Severity:** Medium — agnosai has a workaround. A `/v1/batch` client has none, as one failed item spoils the whole
answer.
**Affects:** hoosh 2.6.5 through 2.10.1.

## Summary

hoosh builds its error envelope by pasting text between quotes. `_resp_err` (`src/lib/handlers.cyr:2170`), `http_error`
(`src/lib/http_server.cyr:171`) and `http_bad_request` (`src/lib/http_server.cyr:147`) all add `msg` with
`str_builder_add_cstr`, so a `"` in it ends the JSON string early. For an upstream failure `msg` holds a provider's
bytes, and when those have no `message` key it holds the provider's raw body.

| What happened | Endpoint | hoosh answers | Parses |
|---|---|---|---|
| 1. Provider 404, `{"error": "model 'llama3.2:1b' not found"}` (Ollama's shape) | chat | 502 | no |
| 2. Provider 404, `<html lang="en">…` | chat | 502 | no |
| 3. Provider 400, a `message` of 510 bytes and then `\"tail\"` | chat | 502 | no |
| 4. Provider 400, `"message":"the \"quoted\" parameter is invalid"` | chat | 502 | yes |
| 5. Ollama's 404 | `POST /v1/embeddings` | 502 | no |
| 6. Two items, one hits Ollama's 404 | `POST /v1/batch` | **200** | no |
| 7. The same with `"async": true` | `GET /v1/batch/{id}` | **200** | no |
| 8. A request with no `model`: hoosh's own text, no provider involved | `POST /v1/hardware/cost`, `/training-estimate` | 400 | no |

Row 1 reads:

```json
{"error":{"message":"upstream provider returned HTTP 404: {"error": "model 'llama3.2:1b' not found"}","type":"upstream_error"}}
```

So a provider that has no `message` key to say a model is gone is the one whose answer no client can read. Rows 6 and 7
are the worst of it: a batch answer is a 200 that embeds each item's body as it stands, so one failed item takes the
items that succeeded down with it.

⚠ **The same gap lets a provider write fields.** A body of `nope","type":"spoofed","injected":"yes` parses, and
`"injected":"yes"` arrives at the client as a key of hoosh's `error` object. The comment on `provider_err_set` calls
these bytes untrusted, and only control bytes are cleaned.

## Reproduction

```python
# stub.py: a provider that answers every chat with Ollama's 404.
import http.server
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        b = b'{"error": "model \'llama3.2:1b\' not found"}'
        self.send_response(404)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
http.server.HTTPServer(("127.0.0.1", 11500), H).serve_forever()
```

```toml
# hoosh.cyml
[server]
bind = "127.0.0.1"
port = 8088
health_check_interval_secs = 0

[[providers]]
type = "ollama"
base_url = "http://127.0.0.1:11500"
priority = 1
models = ["*"]
```

```sh
python3 stub.py &
build/hoosh serve -p 8088 -c hoosh.cyml &
curl -s localhost:8088/v1/chat/completions \
  -d '{"model":"llama3.2:1b","messages":[{"role":"user","content":"hi"}]}' | python3 -m json.tool
# Expecting ',' delimiter: line 1 column 61 (char 60)
```

## Root cause

1. The three writers paste. `upstream_error_sse_body` (`src/lib/provider_err.cyr:356`) is the one that escapes.
2. `_provider_err_capture` (`src/lib/provider.cyr:1226`) falls back to the raw body when there is no `message` key.
3. `_json_extract_str` (`src/lib/jsonlite.cyr:68`) keeps a value's escapes, which is why row 4 survives. That is luck:
   `provider_err_set` (`src/lib/provider_err.cyr:254`) clips to 511 bytes without regard for them, so a clip can leave a
   `\` that swallows the closing quote, which is row 3. A provider with a well-formed `message` can still break the body.
4. `src/lib/batch.cyr:253` and `:373` add each item's body raw, which is correct only if every body is JSON.
5. Row 8 is `src/lib/handlers.cyr:1325` and `:1355`, whose messages hold a literal `"70b"` and `"7b"`.

## Proposed fix

1. **Escape in the writers.** `_resp_err` can call `sb_add_json_escaped` (`src/lib/provider_err.cyr:331`). ⚠ `http_error`
   and `http_bad_request` cannot as the tree stands: `http_server.cyr` is included first (`src/main.cyr:53` and `:60`), and
   resolution is single-pass. `catalog_add_esc` (`src/lib/catalog.cyr:1015`, included at `src/main.cyr:37`) does the same
   job with the same signature, or the escaper can move earlier. This alone makes every row parse.
2. **Decode at capture.** ⚠ Escaping alone double-escapes row 4: the text is still escaped when it is stored, so a client
   would read `the \"quoted\" parameter`, backslashes included. The streaming path does this today: `provider_last_error_msg()`
   goes into `sb_add_json_escaped` still escaped. That is from reading the code and was not run, since only an `https`
   route reaches it and `tests/hoosh.tcyr:6397` hands the function plain text. `_json_unesc_span`
   (`src/lib/provider.cyr:464`) is above `_provider_err_capture` in the same file and undoes `\"` and `\\`. With it every
   writer gets plain text, and the clip falls on plain text, so it cannot cut an escape.

No `msg` reaches these writers already escaped, so nothing else changes on the wire. **Done means** each row above
parses, and row 4 decodes to the provider's own sentence.

`sb_add_json_escaped`, `catalog_add_esc` and `_json_escape_into` now do nearly one job. The first two differ only on byte
127. Whether that is worth folding is hoosh's call.

## Consumer-side workaround

agnosai's `_agnosai_hoosh_detail_a` (its `src/llm/hoosh.cyr`) scans a body that did not parse for the words
`upstream provider returned HTTP 404` and keeps those words only, so a retired model can still be selected again.
Everything else the provider said is lost. **With this fixed agnosai drops the scan** and takes the detail from the
parsed body alone.

agnosai calls neither `/v1/batch` nor `/v1/embeddings`. Those rows are for whoever does.

**Seen in passing, not this issue:** a local streamed chat reports a provider's 404 as `provider backend unreachable`
(`src/lib/handlers.cyr:1938`, a fixed string). That body is valid JSON with the wrong reason, the mistake
`provider_err_client_msg`'s comment explains.

Filed rather than patched: hoosh is a separate project.
