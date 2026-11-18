# usage: python3 -I isolation/tests/gate_upstream_key_known_answer.py <forwarder.py>
# Known answer for the gate's upstream_key_env, on scratch gates and a stub upstream on free loopback ports (never a live
# gate): the gate sends the key from the named variable as x-api-key (or as "Authorization: Bearer"), never the client's
# own credentials; without upstream_key_env no key is sent; an unset variable, an edge with the setting, a bad header name
# or a non-http(s) upstream is refused at load.
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
d = tempfile.mkdtemp()
UP = f"http://127.0.0.1:{stub.server_port}"
ok = True
def want(label, cond):
    global ok
    ok &= bool(cond); print(f"{'PASS' if cond else 'FAIL'} {label}")

def with_gate(extra_cfg, env_extra):
    port = free_port()
    cfg = {"mode": "gate", "listen_port": port, "upstream": UP, "cost_tag": "T-key",
           "log_path": os.path.join(d, f"gate-{port}.jsonl"), **extra_cfg}
    p = os.path.join(d, f"gate-{port}.json"); json.dump(cfg, open(p, "w"))
    env = {k: v for k, v in os.environ.items() if k != "CW_TEST_KEY"}; env.update(env_extra)
    gate = subprocess.Popen([sys.executable, "-I", FWD, p], stderr=subprocess.PIPE, env=env)
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), 0.2).close(); break
        except OSError:
            time.sleep(0.1)
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        c.request("POST", "/v1/messages", json.dumps({"model": "m", "max_tokens": 1, "messages": []}),
                  {"Content-Type": "application/json", "x-api-key": "client-key", "Authorization": "Bearer client-token"})
        r = c.getresponse(); r.read(); c.close()
        return r.status, (seen[-1] if seen else {})
    finally:
        gate.terminate(); gate.wait(5)

status, h = with_gate({"upstream_key_env": "CW_TEST_KEY"}, {"CW_TEST_KEY": "sk-test-123"})
want(f"x-api-key from the variable: status {status}, x-api-key {h.get('x-api-key')!r}, authorization {h.get('authorization')!r}",
     status == 200 and h.get("x-api-key") == "sk-test-123" and "authorization" not in h)
status, h = with_gate({"upstream_key_env": "CW_TEST_KEY", "upstream_key_header": "authorization"}, {"CW_TEST_KEY": "sk-test-456"})
want(f"Bearer from the variable: status {status}, authorization {h.get('authorization')!r}, x-api-key {h.get('x-api-key')!r}",
     status == 200 and h.get("authorization") == "Bearer sk-test-456" and "x-api-key" not in h)
status, h = with_gate({}, {})
want(f"no upstream_key_env: no credentials sent (x-api-key {h.get('x-api-key')!r}, authorization {h.get('authorization')!r})",
     status == 200 and "x-api-key" not in h and "authorization" not in h)
want("cost tag still injected", all(x.get("x-cost-tag") == "T-key" for x in seen))

for label, bad, env_extra, needle in (
        ("an unset key variable", {"mode": "gate", "upstream": UP, "upstream_key_env": "CW_TEST_KEY"}, {}, "not set"),
        ("an edge with a key", {"mode": "edge", "upstream": UP, "cost_tag": "T-key", "upstream_key_env": "CW_TEST_KEY"},
         {"CW_TEST_KEY": "k"}, "gate setting"),
        ("a bad key header", {"mode": "gate", "upstream": UP, "upstream_key_env": "CW_TEST_KEY", "upstream_key_header": "x-key"},
         {"CW_TEST_KEY": "k"}, "upstream_key_header"),
        ("an ftp upstream", {"mode": "gate", "upstream": "ftp://127.0.0.1:1"}, {}, "http:// or https://")):
    p = os.path.join(d, "bad.json"); json.dump(bad, open(p, "w"))
    env = {k: v for k, v in os.environ.items() if k != "CW_TEST_KEY"}; env.update(env_extra)
    r = subprocess.run([sys.executable, "-I", FWD, p], capture_output=True, text=True, timeout=10, env=env)
    want(f"refused at load: {label}", r.returncode != 0 and needle in r.stderr)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
