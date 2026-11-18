#!/usr/bin/env python3
"""A per_example unit sent with host_gap_s is back-dated by that gap (clamped to
the GPU's cursor), so host work between two examples does not add to the
paced time: cursor_B − cursor_A is one slot, not slot + gap."""
import json, os, sys, tempfile, time
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(__file__))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0; n = 0


def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


with tempfile.TemporaryDirectory() as td:
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1.0)
    c = DaemonClient(str(sock)); c.hello(argv=["x.py"])
    a = c.enqueue(gpu=0, op="replay", modelled_s=0.2, step=1, unit="example", host_gap_s=0.0)
    ca = a["cursor"]
    time.sleep(0.5)                      # host work: 0.5 s, longer than the 0.2 s slot
    b = c.enqueue(gpu=0, op="replay", modelled_s=0.2, step=2, unit="example", host_gap_s=0.5)
    cb = b["cursor"]
    chk("back-dated: the second unit starts at the first's cursor, so cursor_B − cursor_A is one slot (0.2 s), not 0.5 + 0.2", 0.17 < cb - ca < 0.25, f"{cb - ca:.3f}")
    time.sleep(0.5)
    c2 = c.enqueue(gpu=0, op="replay", modelled_s=0.2, step=3, unit="example", host_gap_s=0.1)
    cc = c2["cursor"]
    # B's cursor is t0 + 0.4 (back-dated); C is sent at t0 + 1.0 with a 0.1 s gap, so it starts at t0 + 0.9 and ends at t0 + 1.1
    chk("a gap smaller than the time elapsed back-dates by the gap only (cursor_C − cursor_B ≈ 0.7)", 0.64 < cc - cb < 0.78, f"{cc - cb:.3f}")
    d0 = c.enqueue(gpu=0, op="replay", modelled_s=0.2, step=4, unit="example")
    chk("no host_gap_s: the old behaviour (starts at max(cursor, now))", d0["cursor"] >= cc, f"{d0['cursor'] - cc:.3f}")
    c.bye(); c.close()
    ev = read_occ(occ)
    enq = [e for e in ev if e["kind"] == "enqueue"]
    # since next/ r2b the daemon clamps a claimed gap to the real time since the
    # job's previous example unit ended: B claims 0.5 s, but A's unit ended 0.2 s
    # after it was sent, so B's real gap is about 0.3 s (the patch measures from its
    # drain, which is that end, so it never claims more)
    g = [e.get("host_gap_s") for e in enq]
    chk("host_gap_s logged on the enqueue events (rounded, clamped to the real gap since the previous unit ended), absent where not sent", g[0] == 0.0 and 0.25 < g[1] < 0.35 and g[2] == 0.1 and g[3] is None, str(g))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
