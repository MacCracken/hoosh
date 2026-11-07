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

Exits non-zero on the first run with a failed check. No network beyond 127.0.0.1.

Usage: scripts/wire-smoke.py [build/hoosh]
"""
import http.server
import json
import os
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


if __name__ == "__main__":
    bin_path = sys.argv[1] if len(sys.argv) > 1 else "build/hoosh"
    run_errors(bin_path)
    run_retry(bin_path)
    run_catalog(bin_path)
    run_held(bin_path)
    run_race(bin_path)
    if FAILS:
        print(f"\n{len(FAILS)} failed:")
        for f in FAILS:
            print("  " + f)
        sys.exit(1)
    print("\nall ok")
