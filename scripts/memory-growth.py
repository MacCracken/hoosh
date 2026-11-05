#!/usr/bin/env python3
"""memory-growth.py — how much a running gateway keeps per request (2.10.0, ADR 013).

hoosh's allocator never frees, so whatever a request allocates and does not hand to a reused buffer stays for the
life of the process. This measures it the way ADR 013 did: a real gateway with one Ollama route at a fake local
backend (started here, in a thread), sent chat requests with UNIQUE prompts so the response cache never answers
(a repeated prompt never exercises the request path), and the gateway's VmRSS read from /proc after a warm-up.

RSS grows in whole chunks (the allocator maps 8 MB at a time), so use a few thousand requests or more. Each run
uses two sockets per request; a long run right after another can run out of ephemeral ports (TIME_WAIT).

Usage: scripts/memory-growth.py [build/hoosh] [requests]   (defaults: build/hoosh, 3000)
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
import urllib.request


class FakeOllama(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _send(self, obj, code=200, newline=False):
        body = json.dumps(obj).encode() + (b"\n" if newline else b"")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            self._send({"models": [{"name": "llama3:latest", "model": "llama3:latest"}]})
        else:
            self._send({}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            req = json.loads(self.rfile.read(length))
        except ValueError:
            req = {}
        if not self.path.startswith("/api/chat"):
            self._send({}, 404)
            return
        # One NDJSON line for a stream (with its newline, as Ollama sends it), one object otherwise.
        self._send({"model": req.get("model", "llama3"), "created_at": "2026-11-05T00:00:00Z",
                    "message": {"role": "assistant", "content": "A reply of moderate length to exercise the path."},
                    "done": True, "done_reason": "stop", "prompt_eval_count": 42, "eval_count": 17},
                   newline=bool(req.get("stream")))


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


def rss_kib(pid):
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def run(binary, n, stream):
    backend = Server(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    probe = Server(("127.0.0.1", 0), FakeOllama)
    port = probe.server_address[1]
    probe.server_close()
    cfg = (f'[server]\nbind = "127.0.0.1"\nport = {port}\nhealth_check_interval_secs = 0\n\n'
           f'[[providers]]\ntype = "ollama"\nbase_url = "http://127.0.0.1:{backend.server_address[1]}"\n'
           f'priority = 1\nmodels = ["llama*"]\n')
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "hoosh.cyml")
        with open(path, "w") as f:
            f.write(cfg)
        log = open(os.path.join(tmp, "gateway.log"), "w")
        gw = subprocess.Popen([binary, "serve", "-p", str(port), "-c", path], stdout=log, stderr=log)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(url + "/v1/health", timeout=1).read()
                    break
                except OSError:
                    time.sleep(0.05)

            def chat(i):
                body = {"model": "llama3",
                        "messages": [{"role": "user", "content": f"unique prompt {i} on topic {i * 7919 % 10007}"}]}
                if stream:
                    body["stream"] = True
                req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
                return urllib.request.urlopen(req, timeout=10).read()

            for i in range(200):
                chat(i)
            time.sleep(0.5)
            before = rss_kib(gw.pid)
            ok = 0
            t0 = time.time()
            for i in range(200, 200 + n):
                try:
                    chat(i)
                    ok += 1
                except OSError:
                    pass
            time.sleep(0.5)
            after = rss_kib(gw.pid)
            kind = "streamed" if stream else "buffered"
            print(f"{kind}: {ok}/{n} ok in {time.time() - t0:.1f}s; VmRSS {before} -> {after} KiB, "
                  f"{(after - before) * 1024 / max(ok, 1):.0f} bytes per request")
        finally:
            gw.terminate()
            gw.wait(timeout=10)
            backend.shutdown()


if __name__ == "__main__":
    binary = sys.argv[1] if len(sys.argv) > 1 else "build/hoosh"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    run(binary, n, False)
    time.sleep(2)
    run(binary, n, True)
