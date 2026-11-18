#!/usr/bin/env python3
"""A paced wait keeps the host thread busy on a core, as a CUDA sync does, not
asleep (the host-CPU tell, docs/DETECTION-SCOPE-2026-10-26.md, tell 1).

DaemonClient.sync waits for the GPU cursor by spinning with sched_yield. Checked:
the wait lasts as long as before (the cursor's), the process burns about one
core of CPU while it waits, a second thread of the job keeps running at its full
rate during the wait, and NVRT_WAIT=sleep (the control) restores the old
sleeping wait at about 0 CPU. No torch needed:
    python3 -I tests/test_host_cpu_wait_daemon.py
"""
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_daemon_e2e import start_daemon  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0
n = 0


def chk(name, cond, detail=""):
    global fails, n
    n += 1
    fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def paced_wait(c, seconds, worker_counts=None):
    """Enqueue `seconds` of GPU work and sync on it; returns (wall, process CPU)."""
    c.enqueue(gpu=0, op="kernel", modelled_s=seconds)
    w0, c0 = time.monotonic(), time.process_time()
    c.sync(gpu=0)
    return time.monotonic() - w0, time.process_time() - c0


def worker_rate(c, seconds):
    """Progress of a counting thread during a paced wait, against the same thread
    while the main thread sleeps for the same time (no wait in the job)."""
    stop = [False]
    cnt = [0]

    def run():
        while not stop[0]:
            cnt[0] += 1
    t = threading.Thread(target=run)
    t.start()
    time.sleep(0.2)
    b = cnt[0]
    paced_wait(c, seconds)
    during = cnt[0] - b
    b = cnt[0]
    time.sleep(seconds)
    idle = cnt[0] - b
    stop[0] = True
    t.join()
    return during / max(1, idle)


with tempfile.TemporaryDirectory() as td:
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1.0)
    c = DaemonClient(str(sock))
    c.hello(argv=["job.py"])
    c.lookup(fingerprint="none")

    os.environ.pop("NVRT_WAIT", None)
    wall, cpu = paced_wait(c, 1.5)
    chk("the wait lasts the paced 1.5 s (the cursor's time is unchanged)", 1.4 < wall < 1.7, f"{wall:.3f} s")
    # a yielding spin hands its slice to any other runnable process, so on a
    # loaded host it shows less than a full core (0.6–0.9 on the controller at load
    # 28 of 16 cores); on an idle box it is about 1.0. The line it must clear is
    # far from the sleeping wait's 0.0
    chk("the process burns CPU while it waits (≥ half a core; the sleeping wait shows 0)", cpu >= 0.5 * wall, f"cpu {cpu:.3f} s over {wall:.3f} s ({cpu / wall:.2f} of a core)")
    r = worker_rate(c, 1.0)
    chk("another thread of the job keeps its full rate during the wait (≥ 0.8 of its rate when nothing waits)", r >= 0.8, f"{r:.2f}")

    os.environ["NVRT_WAIT"] = "sleep"
    wall_s, cpu_s = paced_wait(c, 1.5)
    chk("control, NVRT_WAIT=sleep: the same 1.5 s wait", 1.4 < wall_s < 1.7, f"{wall_s:.3f} s")
    chk("control, NVRT_WAIT=sleep: asleep, about 0 CPU (under 10% of the wait)", cpu_s < 0.1 * wall_s, f"cpu {cpu_s:.3f} s")
    os.environ.pop("NVRT_WAIT", None)

    wall0, cpu0 = paced_wait(c, 0.0)
    chk("nothing queued: sync returns at once", wall0 < 0.05, f"{wall0:.3f} s")
    c.bye()
    c.close()
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
