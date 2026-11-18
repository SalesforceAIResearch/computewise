# usage: python3 -I isolation/tests/gate_pin_known_answer.py <forwarder.py>
# Known answer for the gate's pin_backend_by_model, on a scratch gate and a stub upstream on free loopback ports (never a live
# gate): a pinned model reaches the upstream with the configured backend header, other models without it; a client's copy of
# that header is dropped; suffixed ids are pinned; an edge, a bad map, a non-dict or a pin with no backend header is refused at load.
import http.client, http.server, json, os, socket, subprocess, sys, tempfile, threading, time
FWD = os.path.abspath(sys.argv[1])
seen = []

class Stub(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        seen.append({k.lower(): v for k, v in self.headers.items()})
        blob = b'{"type":"message","content":[]}'
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob))); self.end_headers(); self.wfile.write(blob)
    def log_message(self, *a):
        pass

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p

stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
threading.Thread(target=stub.serve_forever, daemon=True).start()
d = tempfile.mkdtemp(); port = free_port()
HDR = {"cost_tag": "X-Cost-Tag", "session": "X-Session-Id", "backend": "X-Upstream-Backend"}
cfg = {"mode": "gate", "listen_port": port, "upstream": f"http://127.0.0.1:{stub.server_port}", "cost_tag": "T-pin",
       "log_path": os.path.join(d, "gate.jsonl"), "pin_backend_by_model": {"model-a": "backend-a"}, "upstream_headers": HDR}
json.dump(cfg, open(os.path.join(d, "gate.json"), "w"))
gate = subprocess.Popen([sys.executable, "-I", FWD, os.path.join(d, "gate.json")], stderr=subprocess.PIPE)
for _ in range(50):
    try:
        socket.create_connection(("127.0.0.1", port), 0.2).close(); break
    except OSError:
        time.sleep(0.1)
ok = True
def want(label, cond):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {label}")
def post(model, extra=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request("POST", "/v1/responses", json.dumps({"model": model, "input": "x"}), {"Content-Type": "application/json", **(extra or {})})
    r = c.getresponse(); r.read(); c.close(); return r.status
try:
    for model, extra, expect in (("model-a", None, "backend-a"), ("model-a:high", None, "backend-a"),
                                 ("model-a[1m]", None, "backend-a"), ("model-b:medium", None, None),
                                 ("model-c", {"X-Upstream-Backend": "backend-c"}, None),
                                 ("model-a", {"X-Upstream-Backend": "backend-b"}, "backend-a")):
        n = len(seen); status = post(model, extra)
        got = seen[n].get("x-upstream-backend") if len(seen) > n else "no request"
        note = f" (client sent {extra['X-Upstream-Backend']})" if extra else ""
        want(f"{model}{note}: status {status}, X-Upstream-Backend {got!r}", status == 200 and got == expect)
    want("cost tag still injected", all(h.get("x-cost-tag") == "T-pin" for h in seen))
    logs = [json.loads(l) for l in open(cfg["log_path"]) if '"path"' in l]
    want("gate log records the pin", [r.get("pinned") for r in logs] == ["backend-a"] * 3 + [None, None, "backend-a"])
finally:
    gate.terminate(); gate.wait(5)
UP = "http://127.0.0.1:1"
for label, bad in (("an edge with pins", {"mode": "edge", "upstream": UP, "cost_tag": "T-pin", "upstream_headers": HDR,
                                          "pin_backend_by_model": {"model-a": "backend-a"}}),
                   ("a non-string backend", {"mode": "gate", "upstream": UP, "upstream_headers": HDR, "pin_backend_by_model": {"model-a": 1}}),
                   ("a header-unsafe backend", {"mode": "gate", "upstream": UP, "upstream_headers": HDR,
                                                "pin_backend_by_model": {"model-a": "backend a\r\nX: y"}}),
                   ("a list instead of a map", {"mode": "gate", "upstream": UP, "upstream_headers": HDR, "pin_backend_by_model": ["model-a"]}),
                   ("a pin with no backend header", {"mode": "gate", "upstream": UP, "pin_backend_by_model": {"model-a": "backend-a"}})):
    p = os.path.join(d, "bad.json"); json.dump(bad, open(p, "w"))
    r = subprocess.run([sys.executable, "-I", FWD, p], capture_output=True, text=True, timeout=10)
    want(f"refused at load: {label}", r.returncode != 0 and "pin_backend_by_model" in r.stderr)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
