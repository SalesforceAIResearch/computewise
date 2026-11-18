#!/usr/bin/env python3
"""job_bye is written however a connection ends (isolation's idle_after_harness
rule, 2026-10-22): explicit bye, abrupt close without bye (a crash or SIGKILL),
and an error inside the handler such as the reply write failing.
    python3 -I tests/test_job_end.py
"""
import json, os, socket, sys, tempfile, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src" / "daemon")); sys.path.insert(0, str(HERE.parent / "src" / "torch_patch"))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402
import gpu_daemon  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

def wait_byes(occ, k, t=5.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if sum(1 for e in read_occ(occ) if e["kind"] == "job_bye") >= k:
            return True
        time.sleep(0.05)
    return False

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    d, sock, occ = start_daemon(tmp / "d", speed=1000.0)
    # 1. explicit bye
    c = DaemonClient(str(sock)); j1 = c.hello(argv=["a.py"]); c.enqueue(gpu=0, op="x", modelled_s=0.001, mem_delta=1 << 30, step=1, unit="step"); c.bye(); c.close()
    chk("bye: job_bye written", wait_byes(occ, 1))
    # 2. abrupt close, no bye (what a crash or SIGKILL does to the socket)
    c = DaemonClient(str(sock)); j2 = c.hello(argv=["b.py"]); c.enqueue(gpu=1, op="x", modelled_s=0.001, mem_delta=1 << 30, step=1, unit="step"); c.close()
    chk("abrupt close: job_bye written", wait_byes(occ, 2))
    # 3. an error inside the handler: a raw socket sends a request and closes before reading the reply,
    #    and a bad command makes the dispatch raise as well
    orig = d._dispatch
    def boom(job, req):
        if req.get("cmd") == "boom":
            raise RuntimeError("dispatch failure")
        return orig(job, req)
    d._dispatch = boom
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(str(sock))
    s.sendall((json.dumps({"cmd": "hello", "pid": os.getpid(), "argv": ["c.py"], "cwd": "/"}) + "\n").encode())
    s.recv(65536)
    s.sendall(b'{"cmd":"boom"}\n'); s.close()
    chk("handler error: job_bye written", wait_byes(occ, 3))
    time.sleep(0.2)
    byes = [e for e in read_occ(occ) if e["kind"] == "job_bye"]
    chk("reasons: bye, eof, error:RuntimeError", [e.get("reason") for e in byes] == ["bye", "eof", "error:RuntimeError"], str([e.get("reason") for e in byes]))
    chk("no job left registered, no memory held", not d.jobs and all(not g.mem for g in d.gpus), f"jobs {list(d.jobs)} mem {[dict(g.mem) for g in d.gpus]}")
    chk("each job ended once", len(byes) == 3 and len({e['job'] for e in byes}) == 3)
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
