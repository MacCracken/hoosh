#!/usr/bin/env python3
"""memory-growth.py — how much a running gateway keeps per request (2.10.0, ADR 013).

hoosh's allocator never frees, so whatever a request allocates and does not hand to a reused buffer stays for the
life of the process. This measures it the way ADR 013 did: a real gateway with one Ollama route at a fake local
backend (started here, in a thread), sent chat requests with UNIQUE prompts so the response cache never answers
(a repeated prompt never exercises the request path), and the gateway's VmRSS read from /proc after a warm-up.

RSS grows in whole chunks (the allocator maps 8 MB at a time), so use a few thousand requests or more. Each run
uses two sockets per request; a long run right after another can run out of ephemeral ports (TIME_WAIT).

2.10.2: --get PATH sends that GET instead of chats, for the model listings, with the fake's list at --tags N models.
--shape claude lists them from a `type = "anthropic"` route, whose entries carry the capability tree; the default is
bare Ollama tags. In this mode the output also says what a GET kept if the gateway serves GET /v1/debug/alloc-used,
which only a scratch build does (see ADR 013).

Usage: scripts/memory-growth.py [build/hoosh] [requests]   (defaults: build/hoosh, 3000)
       scripts/memory-growth.py build/hoosh 3000 --get /v1/models/catalog --tags 200 --shape claude
"""
import argparse
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


def claude_model(i):
    """One entry of Anthropic's GET /v1/models, with the capability tree."""
    return {"id": f"claude-opus-5-5-2026{i:04d}", "type": "model", "display_name": f"Claude Opus 5.5 ({i})",
            "created_at": "2026-10-01T00:00:00Z", "max_input_tokens": 1000000, "max_tokens": 128000,
            "capabilities": {"image_input": {"supported": True},
                             "thinking": {"types": {"adaptive": {"supported": True}, "enabled": {"supported": False},
                                                    "disabled": {"supported": True}}},
                             "effort": {"supported": True, "low": {"supported": True}, "medium": {"supported": True},
                                        "high": {"supported": True}, "xhigh": {"supported": True},
                                        "max": {"supported": True}}}}


class FakeOllama(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"
    tags = 1

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
            names = ["llama3:latest"] + [f"llama3-pad{i}:latest" for i in range(1, self.tags)]
            self._send({"models": [{"name": n, "model": n} for n in names]})
        elif self.path.startswith("/v1/models"):
            self._send({"data": [claude_model(i) for i in range(self.tags)], "has_more": False})
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


def run(binary, n, stream, get=None, shape="ollama"):
    backend = Server(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=backend.serve_forever, daemon=True).start()
    probe = Server(("127.0.0.1", 0), FakeOllama)
    port = probe.server_address[1]
    probe.server_close()
    kind, pattern = ("anthropic", "claude-*") if shape == "claude" else ("ollama", "llama*")
    cfg = (f'[server]\nbind = "127.0.0.1"\nport = {port}\nhealth_check_interval_secs = 0\n\n'
           f'[[providers]]\ntype = "{kind}"\nbase_url = "http://127.0.0.1:{backend.server_address[1]}"\n'
           f'priority = 1\nmodels = ["{pattern}"]\n')
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

            def fetch(i):
                return urllib.request.urlopen(url + get, timeout=10).read()

            def alloc_used():
                try:
                    return int(urllib.request.urlopen(url + "/v1/debug/alloc-used", timeout=2).read())
                except (OSError, ValueError):
                    return None

            once = fetch if get else chat
            if get:
                # The first refresh runs on a worker; wait until the catalog has what the fake lists.
                want = {"ollama": b"llama3-pad", "claude": b"claude-opus-5-5-2026"}[shape]
                for _ in range(200):
                    if FakeOllama.tags == 1 or want in fetch(0):
                        break
                    time.sleep(0.05)
            for i in range(200):
                once(i)
            time.sleep(0.5)
            before = rss_kib(gw.pid)
            used0 = alloc_used() if get else None
            ok = 0
            t0 = time.time()
            for i in range(200, 200 + n):
                try:
                    once(i)
                    ok += 1
                except OSError:
                    pass
            used1 = alloc_used() if get else None
            time.sleep(0.5)
            after = rss_kib(gw.pid)
            what = get or ("streamed" if stream else "buffered")
            line = (f"{what}: {ok}/{n} ok in {time.time() - t0:.1f}s; VmRSS {before} -> {after} KiB, "
                    f"{(after - before) * 1024 / max(ok, 1):.0f} bytes per request")
            if used0 is not None and used1 is not None:
                line += f"; alloc_used +{(used1 - used0) / max(ok, 1):.0f} bytes per request"
            print(line)
        finally:
            gw.terminate()
            gw.wait(timeout=10)
            backend.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="how much a running gateway keeps per request")
    ap.add_argument("binary", nargs="?", default="build/hoosh")
    ap.add_argument("requests", nargs="?", type=int, default=3000)
    ap.add_argument("--get", help="send this GET instead of chats")
    ap.add_argument("--tags", type=int, default=1, help="models the fake backend lists")
    ap.add_argument("--shape", choices=("ollama", "claude"), default="ollama")
    args = ap.parse_args()
    FakeOllama.tags = args.tags
    if args.get:
        run(args.binary, args.requests, False, args.get, args.shape)
    else:
        run(args.binary, args.requests, False)
        time.sleep(2)
        run(args.binary, args.requests, True)
