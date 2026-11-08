#!/usr/bin/env python3
"""wire-smoke.py — what a client reads off a real gateway, against stub providers (2.10.2).

The unit tests cannot reach handlers.cyr, which reads main.cyr's globals, so nothing there can say what a whole request
puts on the wire. This runs a real gateway at local stubs and checks the answers as a strict client would: `json.loads`
on the raw bytes, which also refuses a body that is not UTF-8.

  errors    every shape of provider refusal, through chat, a stream, /v1/embeddings and /v1/batch, parses, and says
            what the provider said. docs/development/issues/archived/2026-11-06-agnosai-error-bodies-with-a-quote-are-not-json.md
  retry     a permanent refusal is asked once, on a stream too, and a 5xx is still retried.
  catalog   GET /v1/models/catalog says tier, tools and kind, and omits what nobody said.
            docs/development/issues/archived/2026-11-06-agnosai-catalog-cannot-say-tier-tools-or-chat.md
  held      the catalog answers from a held body, and a change to the router, the snapshot or [[models]] replaces it.
            docs/development/issues/archived/2026-11-06-agnosai-every-catalog-get-keeps-what-it-built.md
  race      reads across all the listings while reloads and refreshes replace what they read.
  stream    (2.10.3) a many-token reply, with quotes, backslashes and non-ASCII text, through an Ollama and an
            OpenAI-compatible route: every chunk parses, all carry one id, and the deltas join into the reply. The
            chunks are written from one reused buffer since 2.10.3.
  events    (2.10.3) GET /v1/events/recent read while chats fill the ring from several threads: every read parses,
            holds at most 128 events of the shapes hoosh writes, and the last read the newest. Events are copied into
            recycled slots since 2.10.3.
  trace     (2.10.3) hoosh's span is the caller's child: the provider is sent hoosh's span id, and the exported span
            names the caller's as its parentSpanId. A batch item is a span of its own under the request's caller.
            docs/development/issues/archived/2026-11-07-agnosai-inference-span-takes-the-callers-span-id.md

Exits non-zero on the first run with a failed check. No network beyond 127.0.0.1.

Usage: scripts/wire-smoke.py [build/hoosh]
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

LONG_TAIL = "x" * 510 + '"tail"'
FAILS = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what)
    if not ok:
        FAILS.append(what)


# What the error stub answers, by the prompt or embedding input it was sent: (status, body bytes).
REFUSALS = {
    "ollama404": (404, b'{"error": "model \'llama3.2:1b\' not found"}'),
    "html404": (404, b'<html lang="en"><body>no such thing</body></html>'),
    "clip": (400, json.dumps({"error": {"message": LONG_TAIL}}).encode()),
    "quoted": (400, b'{"error":{"message":"the \\"quoted\\" parameter is invalid"}}'),
    "spoof": (400, b'nope","type":"spoofed","injected":"yes'),
    "go": (400, b'{"error":{"message":"a \\u003cmodel\\u003e \\u0026 caf\\u00e9"}}'),
    "latin1": (400, b"<p>caf\xe9 \"closed\"</p>"),
    "boom500": (500, b'{"error":{"message":"boom"}}'),
    "cjkcut": (400, json.dumps({"error": {"message": "中" * 400}}, ensure_ascii=False).encode()),
}


class ErrorStub(http.server.BaseHTTPRequestHandler):
    """An Ollama-shaped provider that refuses each request the way its text asks, and answers 'ok' otherwise."""
    protocol_version = "HTTP/1.0"
    hits = {}

    def log_message(self, *args):
        pass

    def _send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, json.dumps({"models": [{"name": "stub:latest"}]}).encode())

    def do_POST(self):
        text = self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8", "replace")
        for key, (code, body) in REFUSALS.items():
            if key in text:
                ErrorStub.hits[key] = ErrorStub.hits.get(key, 0) + 1
                self._send(code, body)
                return
        if self.path.startswith("/api/show"):
            self._send(404, b"{}")
            return
        self._send(200, json.dumps({"model": "stub", "message": {"role": "assistant", "content": "ok"}, "done": True,
                                    "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}).encode())


# The catalog stubs. What Ollama's own docs and source say: details.family in the tags, `capabilities` in /api/show.
TAGS = [
    {"name": "llama3.2:8b", "details": {"family": "llama"}},
    {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
    {"name": "bge-m3:latest", "details": {"family": "bert"}},
    {"name": "mystery:1b", "details": {"family": "gemma3"}},
    {"name": "oldshow:1b", "details": {"family": "gemma3"}},
]
SHOW = {
    "llama3.2:8b": ["completion", "tools"],
    "nomic-embed-text:latest": ["embedding"],
    "bge-m3:latest": ["embedding"],
    "mystery:1b": ["completion", "vision"],
}


class OllamaStub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    shows = 0

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": TAGS} if self.path.startswith("/api/tags") else {}, 200)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
        OllamaStub.shows += 1
        name = req.get("model")
        if name == "oldshow:1b":          # an Ollama from before `capabilities`
            self._send({"details": {"family": "gemma3"}, "template": "x"})
        elif name in SHOW:
            self._send({"details": {}, "license": "x" * 20000, "capabilities": SHOW[name]})
        else:
            self._send({"error": "no such model"}, 404)


class OpenAIStub(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    ids = ["gpt-6-astra", "gpt-6-luna", "gpt-4o", "text-embedding-3-small", "whisper-1", "tts-1", "gpt-99-new"]

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": i, "object": "model"} for i in OpenAIStub.ids]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def serve(handler):
    s = Server(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


class Gateway:
    def __init__(self, binary, cfg_body):
        probe = Server(("127.0.0.1", 0), ErrorStub)
        self.port = probe.server_address[1]
        probe.server_close()
        self.url = f"http://127.0.0.1:{self.port}"
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = os.path.join(self.tmp.name, "hoosh.cyml")
        self.write_config(cfg_body)
        self.log = open(os.path.join(self.tmp.name, "gateway.log"), "w")
        self.proc = subprocess.Popen([binary, "serve", "-p", str(self.port), "-c", self.cfg],
                                     stdout=self.log, stderr=self.log)
        for _ in range(100):
            try:
                urllib.request.urlopen(self.url + "/v1/health", timeout=1).read()
                return
            except OSError:
                time.sleep(0.05)
        raise SystemExit("gateway did not start")

    def write_config(self, cfg_body):
        with open(self.cfg, "w") as f:
            f.write(f'[server]\nbind = "127.0.0.1"\nport = {self.port}\nhealth_check_interval_secs = 0\n\n' + cfg_body)

    def req(self, path, body=None):
        """(status, raw bytes)."""
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.url + path, data=data, method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(r, timeout=15) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def stop(self):
        self.proc.terminate()
        self.proc.wait(timeout=10)
        self.tmp.cleanup()


def strict(raw):
    """json.loads of the raw bytes, as a strict client reads them: refuses bad JSON and bad UTF-8. None on refusal."""
    try:
        return json.loads(raw)
    except ValueError:
        return None


def chat(prompt, stream=False):
    b = {"model": "llama3.2:1b", "messages": [{"role": "user", "content": prompt}]}
    if stream:
        b["stream"] = True
    return b


def run_errors(binary):
    print("errors")
    stub = serve(ErrorStub)
    gw = Gateway(binary, f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{stub.server_address[1]}"\n'
                         'priority = 1\nmodels = ["*"]\n\n[[retry]]\nmax_retries = 0\n')
    try:
        for key, want in (("ollama404", "model 'llama3.2:1b' not found"),
                          ("html404", '<html lang="en">'),
                          ("clip", "xxxx"),
                          ("quoted", 'the "quoted" parameter is invalid'),
                          ("go", "a <model> & café"),
                          ("latin1", 'caf  "closed"'),
                          ("cjkcut", "中中")):
            st, raw = gw.req("/v1/chat/completions", chat(key))
            doc = strict(raw)
            msg = doc["error"]["message"] if doc else ""
            check(st == 502 and doc is not None and want in msg and "\\" not in msg, f"chat {key}: parses, says {want!r}")

        st, raw = gw.req("/v1/chat/completions", chat("spoof"))
        doc = strict(raw)
        check(doc is not None and "injected" not in doc["error"] and doc["error"]["type"] == "upstream_error",
              "chat spoof: a provider writes no keys into hoosh's error object")

        odd = chat("fine")
        odd["pool"] = 'no"such\\pool'
        st, raw = gw.req("/v1/chat/completions", odd)
        doc = strict(raw)
        check(st == 400 and doc is not None and doc["error"]["message"] == "Token pool 'no\"such\\pool' does not exist",
              "a client's own text in an error reads as the client wrote it, escaped once")

        st, raw = gw.req("/v1/chat/completions", chat("ollama404", stream=True))
        doc = strict(raw)
        check(st == 502 and doc is not None and "HTTP 404" in doc["error"]["message"]
              and "not found" in doc["error"]["message"] and "unreachable" not in doc["error"]["message"],
              "local stream: a refusal is reported as the refusal, not as an unreachable backend")

        st, raw = gw.req("/v1/embeddings", {"model": "m", "input": "ollama404"})
        check(strict(raw) is not None and st == 502, "embeddings: parses")

        st, raw = gw.req("/v1/hardware/cost", {"quantization": "int8"})
        doc = strict(raw)
        check(st == 400 and doc is not None and '"70b"' in doc["error"]["message"], "hardware cost: hoosh's own quote")
        st, raw = gw.req("/v1/hardware/training-estimate", {"method": "lora"})
        doc = strict(raw)
        check(st == 400 and doc is not None and '"7b"' in doc["error"]["message"], "training estimate: hoosh's own quote")

        batch = {"requests": [chat("fine"), chat("ollama404")]}
        st, raw = gw.req("/v1/batch", batch)
        doc = strict(raw)
        check(st == 200 and doc is not None and len(doc["results"]) == 2
              and doc["results"][0]["body"]["choices"][0]["message"]["content"] == "ok"
              and "not found" in doc["results"][1]["body"]["error"]["message"],
              "batch: a failed item leaves the good one readable")

        st, raw = gw.req("/v1/batch", dict(batch, **{"async": True}))
        bid = strict(raw)["id"]
        doc = None
        for _ in range(100):
            st, raw = gw.req("/v1/batch/" + bid)
            doc = strict(raw)
            if doc is None or doc.get("status") not in ("queued", "running", "pending", "in_progress"):
                break
            time.sleep(0.05)
        check(doc is not None and len(doc["results"]) == 2, "async batch: parses")
    finally:
        gw.stop()
        stub.shutdown()


def run_retry(binary):
    print("retry")
    stub = serve(ErrorStub)
    gw = Gateway(binary, f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{stub.server_address[1]}"\n'
                         'priority = 1\nmodels = ["*"]\n\n'
                         '[[retry]]\nmax_retries = 2\nbase_delay_ms = 100\nmax_delay_ms = 100\n')
    try:
        for stream, name in ((False, "buffered"), (True, "stream")):
            ErrorStub.hits.clear()
            t0 = time.time()
            st, raw = gw.req("/v1/chat/completions", chat("ollama404", stream=stream))
            took = time.time() - t0
            doc = strict(raw)
            check(st == 502 and doc is not None and ErrorStub.hits.get("ollama404") == 1 and took < 0.15,
                  f"{name}: a permanent refusal is asked once and answered at once ({ErrorStub.hits}, {took:.2f}s)")
        ErrorStub.hits.clear()
        gw.req("/v1/chat/completions", chat("boom500", stream=True))
        check(ErrorStub.hits.get("boom500") == 3, f"stream: a 500 is still retried ({ErrorStub.hits})")
    finally:
        gw.stop()
        stub.shutdown()


def catalog(gw):
    st, raw = gw.req("/v1/models/catalog")
    doc = strict(raw)
    return {} if doc is None else {e["id"]: e for e in doc["data"]}


def caps(e):
    return e.get("capabilities", {})


def run_catalog(binary):
    print("catalog")
    oa = serve(OpenAIStub)
    ol = serve(OllamaStub)
    closed = Server(("127.0.0.1", 0), ErrorStub)
    cport = closed.server_address[1]
    closed.server_close()
    gw = Gateway(binary, f'''[[providers]]
type = "openai"
base_url = "http://127.0.0.1:{oa.server_address[1]}"
priority = 1
models = ["gpt-*", "text-embedding-*", "whisper-*", "tts-*"]

[[providers]]
type = "ollama"
base_url = "http://127.0.0.1:{ol.server_address[1]}"
priority = 2
models = ["*"]

[[models]]
id = "gpt-6-luna"
tools = false
tier = "premium"

[[models]]
id = "mystery:1b"
kind = "other"
''')
    try:
        c = {}
        for _ in range(100):
            c = catalog(gw)
            if "bge-m3:latest" in c and c["bge-m3:latest"].get("source") == "live":
                break
            time.sleep(0.05)

        e = c.get("gpt-6-astra", {})
        check(e.get("tier") == "premium" and e.get("kind") == "chat" and caps(e).get("tools") is False
              and caps(e).get("vision") is True and e.get("context_window") == 1050000 and e.get("source") == "live",
              "a live entry gets the table's tier, kind, tools, vision and context window")
        check(all(i not in c for i in ("text-embedding-3-small", "whisper-1", "tts-1")),
              "the OpenAI-compatible list stays filtered")
        e = c.get("gpt-6-luna", {})
        check(caps(e).get("tools") is False and e.get("tier") == "premium", "[[models]] wins over the table")
        e = c.get("gpt-99-new", {})
        check(e and all(k not in e for k in ("tier", "kind", "capabilities", "context_window")),
              "a model nobody knows says nothing rather than a default")
        e = c.get("llama3.2:8b", {})
        check(e.get("kind") == "chat" and caps(e).get("tools") is True and e.get("tier") == "economy",
              "Ollama: a chat model, tools from /api/show and tier from the table")
        for name in ("nomic-embed-text:latest", "bge-m3:latest"):
            check(c.get(name, {}).get("kind") == "embedding", f"Ollama: {name} is an embedding model")
        check(caps(c.get("bge-m3:latest", {})).get("tools") is False, "Ollama: /api/show's list is an answer on tools")
        e = c.get("mystery:1b", {})
        check(e.get("kind") == "other" and caps(e).get("vision") is True, "[[models]] kind wins over /api/show")
        e = c.get("oldshow:1b", {})
        check(e and "kind" not in e and "tools" not in caps(e), "an Ollama with no `capabilities` leaves them unknown")
    finally:
        gw.stop()

    # The table path: a route that has not answered, with literal patterns.
    gw = Gateway(binary, f'''[[providers]]
type = "openai"
base_url = "http://127.0.0.1:{cport}"
priority = 1
models = ["gpt-6-astra", "nomic-embed-text", "whisper-1", "llama3", "text-embedding-ada-002", "gpt-4o-transcribe"]
''')
    try:
        time.sleep(0.3)
        c = catalog(gw)
        e = c.get("gpt-6-astra", {})
        check(e.get("source") == "builtin" and e.get("kind") == "chat" and caps(e).get("tools") is False,
              "table path: gpt-6-astra")
        check(c.get("nomic-embed-text", {}).get("kind") == "embedding", "table path: the embedder's row says so")
        check(c.get("text-embedding-ada-002", {}).get("kind") == "embedding", "table path: the words with no row")
        check(c.get("whisper-1", {}).get("kind") == "other", "table path: a speech model is neither")
        check(c.get("llama3", {}).get("kind") == "chat", "table path: a local tag")
        e = c.get("gpt-4o-transcribe", {})
        check(e.get("kind") == "other" and "tier" not in e and "capabilities" not in e,
              "table path: a prefix row lends nothing to a model of another kind")
    finally:
        gw.stop()
        oa.shutdown()
        ol.shutdown()


def run_held(binary):
    print("held")
    a = serve(OllamaStub)
    b = serve(OpenAIStub)

    def cfg(models, extra=""):
        return f'''[[providers]]
type = "ollama"
base_url = "http://127.0.0.1:{a.server_address[1]}"
priority = 1
models = {models}
''' + extra
    gw = Gateway(binary, cfg('["llama*", "nomic*"]'))
    try:
        time.sleep(0.4)
        first = gw.req("/v1/models/catalog")[1]
        check(first == gw.req("/v1/models/catalog")[1] and b"llama3.2:8b" in first, "the same body twice")

        # /v1/admin/reload replaces the router at once; the old snapshot still serves until the refresh publishes.
        st, raw = gw.req("/v1/embeddings", {"model": 'gpt"x\\y', "input": "hi"})
        doc = strict(raw)
        check(st == 404 and doc is not None
              and doc["error"]["message"] == "No provider configured for model 'gpt\"x\\y'",
              "embeddings: a model name that routes nowhere reads as the client wrote it, escaped once")

        gw.write_config(cfg('["llama*"]'))
        gw.req("/v1/admin/reload", {})
        check(b"nomic-embed-text" not in gw.req("/v1/models/catalog")[1], "a reload that drops a pattern shows at once")

        gw.write_config(cfg('["llama*"]', '\n[[models]]\nid = "llama3.2:8b"\ntier = "reasoning"\n'))
        gw.req("/v1/admin/reload", {})
        check(b'"tier":"reasoning"' in gw.req("/v1/models/catalog")[1], "a reload that changes [[models]] shows")

        TAGS.append({"name": "llama-new:1b", "details": {"family": "llama"}})
        try:
            st = 202
            for _ in range(100):             # 202: the refresh the reload kicked off is still running
                st, raw = gw.req("/v1/models/refresh", {})
                if st != 202:
                    break
                time.sleep(0.05)
            check(st == 200 and b"llama-new:1b" in gw.req("/v1/models/catalog")[1], "a refresh shows")
        finally:
            TAGS.pop()
    finally:
        gw.stop()
        a.shutdown()
        b.shutdown()


def run_race(binary):
    """Workers read the held body, lock-free, while reloads and refreshes replace everything under it."""
    print("race")
    a = serve(OllamaStub)
    cfg = (f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{a.server_address[1]}"\n'
           'priority = 1\nmodels = ["*"]\n\n[[models]]\nid = "llama3.2:8b"\ntier = "reasoning"\n')
    gw = Gateway(binary, cfg)
    stop = threading.Event()
    bad = []
    reads = [0]

    def reader():
        while not stop.is_set():
            try:
                for path in ("/v1/models/catalog", "/v1/models", "/api/tags"):
                    st, raw = gw.req(path)
                    doc = strict(raw)
                    if st != 200 or doc is None:
                        bad.append((path, st, raw[:120]))
                    reads[0] += 1
            except OSError as e:
                bad.append(("os", str(e)))

    threads = [threading.Thread(target=reader) for _ in range(6)]
    try:
        for t in threads:
            t.start()
        t0 = time.time()
        rounds = 0
        while time.time() - t0 < 3:
            gw.req("/v1/admin/reload", {})
            gw.req("/v1/models/refresh", {})
            rounds += 1
        stop.set()
        for t in threads:
            t.join()
        print(f"       {reads[0]} reads across {rounds} reload and refresh rounds")
        check(not bad and gw.proc.poll() is None, f"every read parsed and the gateway stayed up {bad[:2]}")
    finally:
        stop.set()
        gw.stop()
        a.shutdown()


# The reply StreamStub streams, one token per line or event: escapes and characters a chunk builder could mangle.
STREAM_TOKENS = ["Hello", ", ", "\"quoted\"", " back\\slash", " café", " 中文", " tab\there", " " + "z" * 9000, "."]


class StreamStub(http.server.BaseHTTPRequestHandler):
    """A local backend that streams STREAM_TOKENS: Ollama NDJSON on /api/chat, OpenAI SSE on /v1/chat/completions."""
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "stub:latest"}], "data": [{"id": "stub"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.startswith("/api/chat"):
            lines = [json.dumps({"message": {"role": "assistant", "content": t}, "done": False}, ensure_ascii=False)
                     for t in STREAM_TOKENS]
            lines.append(json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                                     "done_reason": "stop", "prompt_eval_count": 3, "eval_count": len(STREAM_TOKENS)}))
            body = ("\n".join(lines) + "\n").encode()
            ctype = "application/x-ndjson"
        else:
            ev = [json.dumps({"choices": [{"index": 0, "delta": {"content": t}, "finish_reason": None}]},
                             ensure_ascii=False) for t in STREAM_TOKENS]
            ev.append(json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}))
            body = ("".join(f"data: {e}\n\n" for e in ev) + "data: [DONE]\n\n").encode()
            ctype = "text/event-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def sse_events(raw):
    """The data payloads of an SSE body, in order."""
    out = []
    for block in raw.split(b"\n\n"):
        for line in block.split(b"\n"):
            if line.startswith(b"data: "):
                out.append(line[6:])
    return out


def run_stream(binary):
    print("stream")
    stub = serve(StreamStub)
    port = stub.server_address[1]
    for kind in ("ollama", "openai"):
        gw = Gateway(binary, f'[[providers]]\ntype = "{kind}"\nbase_url = "http://127.0.0.1:{port}"\n'
                             'priority = 1\nmodels = ["*"]\n')
        try:
            for rnd in range(3):
                st, raw = gw.req("/v1/chat/completions", chat(f"stream {kind} {rnd}", stream=True))
                evs = sse_events(raw)
                docs = [strict(e) for e in evs if e != b"[DONE]"]
                deltas = [d["choices"][0]["delta"].get("content", "") for d in docs
                          if d and d.get("choices") and d["choices"][0].get("delta")]
                ids = {d.get("id") for d in docs if d}
                check(st == 200 and evs and evs[-1] == b"[DONE]" and all(d is not None for d in docs),
                      f"{kind} stream {rnd}: every chunk parses as a strict client reads it")
                check("".join(deltas) == "".join(STREAM_TOKENS),
                      f"{kind} stream {rnd}: the deltas join into the reply, escapes and all")
                check(len(ids) == 1 and docs[0].get("object") == "chat.completion.chunk",
                      f"{kind} stream {rnd}: one id for every chunk")
                check(docs and docs[-1]["choices"][0].get("finish_reason") == "stop",
                      f"{kind} stream {rnd}: it ends with the provider's finish reason")
        finally:
            gw.stop()
    stub.shutdown()


def run_events(binary):
    print("events")
    stub = serve(ErrorStub)
    gw = Gateway(binary, f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{stub.server_address[1]}"\n'
                         'priority = 1\nmodels = ["*"]\n\n[cache]\nenabled = false\n')
    stop = threading.Event()
    bad = []
    reads = [0]
    sent = [0]

    def chatter(t):
        i = 0
        while not stop.is_set() and i < 60:
            body = chat(f"event {t} {i}")
            body["model"] = f'm"{t}\\{i}'          # a model the event must escape
            gw.req("/v1/chat/completions", body)
            sent[0] += 1
            i += 1

    def reader():
        while not stop.is_set():
            st, raw = gw.req("/v1/events/recent")
            doc = strict(raw)
            reads[0] += 1
            if st != 200 or doc is None or len(doc.get("events", [])) > 128:
                bad.append((st, raw[:120]))
                continue
            for e in doc["events"]:
                if e.get("event") not in ("inference_completed", "inference_failed", "rate_limited", "health_changed"):
                    bad.append(("shape", e))
                    break

    try:
        writers = [threading.Thread(target=chatter, args=(t,)) for t in range(6)]
        readers = [threading.Thread(target=reader) for _ in range(2)]
        for t in writers + readers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()
        print(f"       {reads[0]} reads while {sent[0]} chats filled the ring")
        check(not bad, f"every read parsed and held only hoosh's event shapes {bad[:2]}")
        st, raw = gw.req("/v1/events/recent")
        evs = (strict(raw) or {}).get("events", [])
        models = [e.get("model", "") for e in evs]
        exact = re.compile(r'm"[0-5]\\[0-9]+')
        check(len(evs) == 128 and all(exact.fullmatch(m) for m in models),
              f"the ring holds 128, each model read back exactly as sent, escaped once ({len(evs)}, {models[-1:]})")
        check(gw.proc.poll() is None, "the gateway stayed up")
    finally:
        stop.set()
        gw.stop()
        stub.shutdown()


CALLER = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class TraceStub(http.server.BaseHTTPRequestHandler):
    """An Ollama-shaped provider that records each request's traceparent, and an OTLP collector that keeps each span."""
    protocol_version = "HTTP/1.0"
    sent = []      # the traceparent of each chat request hoosh made, by prompt: (prompt, header or None)
    spans = []     # every span the collector was sent

    def log_message(self, *args):
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "stub:latest"}]})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.path.startswith("/v1/traces"):
            for rs in json.loads(raw)["resourceSpans"]:
                for ss in rs["scopeSpans"]:
                    TraceStub.spans.extend(ss["spans"])
            self._send({})
            return
        if self.path.startswith("/api/show"):
            self._send({})
            return
        try:
            prompt = json.loads(raw)["messages"][-1]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            prompt = None
        TraceStub.sent.append((prompt, self.headers.get("traceparent")))
        if self.path.startswith("/api/chat") and b'"stream":true' in raw.replace(b" ", b""):
            body = (json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": False}) + "\n"
                    + json.dumps({"message": {"role": "assistant", "content": ""}, "done": True,
                                  "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1}) + "\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send({"model": "stub", "message": {"role": "assistant", "content": "ok"}, "done": True,
                    "done_reason": "stop", "prompt_eval_count": 1, "eval_count": 1})


def tp_parts(tp):
    """(trace id, span id, flags) of a traceparent, or None."""
    if not tp or len(tp) != 55:
        return None
    return tp[3:35], tp[36:52], tp[53:55]


def sent_for(prompt):
    return [tp for p, tp in TraceStub.sent if p == prompt]


def spans_for(trace_id, n, wait=4.0):
    """The spans of a trace, once the exporter has sent n of them (it batches every second)."""
    t0 = time.time()
    while time.time() - t0 < wait:
        got = [sp for sp in TraceStub.spans if sp.get("traceId") == trace_id]
        if len(got) >= n:
            return got
        time.sleep(0.1)
    return [sp for sp in TraceStub.spans if sp.get("traceId") == trace_id]


def traced(gw, path, body, tp):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if tp:
        headers["traceparent"] = tp
    r = urllib.request.Request(gw.url + path, data=data, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def run_trace(binary):
    print("trace")
    stub = serve(TraceStub)
    port = stub.server_address[1]
    gw = Gateway(binary, f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{port}"\n'
                         'priority = 1\nmodels = ["*"]\n\n'
                         f'[telemetry]\notlp_endpoint = "http://127.0.0.1:{port}/v1/traces"\n')
    ctrace, cspan, _ = tp_parts(CALLER)
    try:
        # The issue's repro: a request that names the caller's span.
        traced(gw, "/v1/chat/completions", chat("child-1"), CALLER)
        up = tp_parts((sent_for("child-1") or [None])[0])
        check(up is not None and up[0] == ctrace and up[1] != cspan and up[2] == "01",
              f"the provider is sent the caller's trace under hoosh's own span id ({up})")
        sp = spans_for(ctrace, 1)
        check(len(sp) == 1 and up is not None and sp[0]["spanId"] == up[1] and sp[0].get("parentSpanId") == cspan,
              f"the span is hoosh's, and its parent is the caller's ({sp[:1]})")

        # No header: a root, and the provider's parent is still hoosh's span.
        traced(gw, "/v1/chat/completions", chat("root-1"), None)
        up = tp_parts((sent_for("root-1") or [None])[0])
        sp = spans_for(up[0], 1) if up else []
        check(up is not None and up[0] != ctrace and len(sp) == 1 and sp[0]["spanId"] == up[1]
              and "parentSpanId" not in sp[0], f"without a header the span is a root ({sp[:1]})")

        # A stream forwards hoosh's own too.
        traced(gw, "/v1/chat/completions", chat("stream-1", stream=True), CALLER)
        up = tp_parts((sent_for("stream-1") or [None])[0])
        check(up is not None and up[0] == ctrace and up[1] != cspan, f"a stream is sent hoosh's own span id ({up})")

        # A sync batch: each item a span of its own under the caller's.
        btrace = "0af7651916cd43dd8448eb211c80319c"
        bcaller = f"00-{btrace}-b7ad6b7169203331-01"
        traced(gw, "/v1/batch", {"requests": [chat("item-a"), chat("item-b"), chat("item-c")]}, bcaller)
        ups = [tp_parts(t) for t in sent_for("item-a") + sent_for("item-b") + sent_for("item-c")]
        sp = spans_for(btrace, 3)
        check(len(ups) == 3 and all(u and u[0] == btrace and u[1] != "b7ad6b7169203331" for u in ups)
              and len({u[1] for u in ups}) == 3, f"batch: each item is sent its own span id in the caller's trace")
        check(len(sp) == 3 and all(s.get("parentSpanId") == "b7ad6b7169203331" for s in sp)
              and {s["spanId"] for s in sp} == {u[1] for u in ups if u},
              f"batch: three spans, each the caller's child ({len(sp)})")

        # An async batch runs after its request has ended, on any worker.
        atrace = "11111111111111111111111111111111"
        acaller = f"00-{atrace}-2222222222222222-01"
        st, raw = traced(gw, "/v1/batch", {"requests": [chat("async-a"), chat("async-b")], "async": True}, acaller)
        bid = (strict(raw) or {}).get("id")
        for _ in range(100):
            if len(sent_for("async-a") + sent_for("async-b")) == 2:
                break
            time.sleep(0.05)
        ups = [tp_parts(t) for t in sent_for("async-a") + sent_for("async-b")]
        sp = spans_for(atrace, 2)
        check(bid is not None and len(ups) == 2 and all(u and u[0] == atrace for u in ups)
              and len(sp) == 2 and all(s.get("parentSpanId") == "2222222222222222" for s in sp),
              f"async batch: the items continue the request that submitted them ({len(sp)})")

        # A worker keeps its last connection's slots: fill every worker's with the caller's, then send items with none.
        for i in range(14):
            traced(gw, "/v1/chat/completions", chat(f"fill-{i}"), CALLER)
        traced(gw, "/v1/batch", {"requests": [chat(f"bare-{i}") for i in range(7)]}, None)
        ups = [tp_parts(t) for i in range(7) for t in sent_for(f"bare-{i}")]
        check(len(ups) == 7 and all(u and u[0] != ctrace for u in ups),
              "batch: an item of a request without a header is not sent under another request's trace")

        # Values the spec calls invalid: a root, with no zeros for a parent.
        ztrace = "33333333333333333333333333333333"
        traced(gw, "/v1/chat/completions", chat("zero-1"), f"00-{ztrace}-0000000000000000-01")
        up = tp_parts((sent_for("zero-1") or [None])[0])
        check(up is not None and up[0] != ztrace, "an all-zero parent id is not continued")
        check(gw.proc.poll() is None, "the gateway stayed up")
    finally:
        gw.stop()
        stub.shutdown()


if __name__ == "__main__":
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "build/hoosh"
    run_errors(bin_path)
    run_retry(bin_path)
    run_catalog(bin_path)
    run_held(bin_path)
    run_race(bin_path)
    run_stream(bin_path)
    run_events(bin_path)
    run_trace(bin_path)
    if FAILS:
        print(f"\n{len(FAILS)} failed:")
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    print("\nall ok")
