#!/usr/bin/env python3
"""End-to-end smoke test for the gpu-mock daemon and client.

Starts a daemon in-process on a temp socket, drives it through the client, and
checks: pacing (wall time tracks modelled time / speed), the served loss curve,
the occupancy log records transitions, contention doubles wall time for two
jobs on one GPU, memory accounting and OOM, and failure injection.

Run with the repo's own interpreter; no third-party deps. Isolated:
    python3 -I test_daemon_e2e.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "daemon"))
sys.path.insert(0, str(ROOT / "src" / "torch_patch"))

import gpu_daemon  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    mark = "ok  " if cond else "FAIL"
    if cond:
        PASS += 1
    else:
        FAIL += 1
    print(f"[{mark}] {name}" + (f"  — {detail}" if detail else ""))


def start_daemon(tmp: Path, speed: float, registry: Path | None = None,
                 gpus: int = 4, cap_gib: float = 80.0):
    sock = tmp / "sock"
    occ = tmp / "occupancy.jsonl"
    reg = gpu_daemon.Registry(registry)
    d = gpu_daemon.GPUDaemon(sock, reg, gpus,
                             int(cap_gib * 1024 ** 3), speed, occ)
    t = threading.Thread(target=d.serve, daemon=True)
    t.start()
    # Wait until the daemon ACCEPTS a connection, not merely until the socket
    # path exists: bind() creates the path before listen() runs, and under load
    # (other suites running) a client racing that gap gets ECONNREFUSED. Seen
    # once as a spurious failure while an 8-rank dry run was running alongside.
    import socket as _s
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if sock.exists():
            probe = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM)
            try:
                probe.connect(str(sock))
                probe.close()
                break
            except OSError:
                probe.close()
        time.sleep(0.02)
    return d, sock, occ


def read_occ(occ: Path) -> list[dict]:
    if not occ.exists():
        return []
    return [json.loads(l) for l in occ.read_text().splitlines() if l.strip()]


# --------------------------------------------------------------------------
# Test 1: pacing and served curve, speed knob.
# --------------------------------------------------------------------------

def test_pacing_and_curve(tmp: Path) -> None:
    reg_dir = tmp / "reg1"
    reg_dir.mkdir()
    (reg_dir / "r.json").write_text((HERE / "sample_record.json").read_text())
    speed = 4.0  # run 4x faster so the test is quick
    d, sock, occ = start_daemon(tmp / "d1", speed, registry=reg_dir)
    os.environ["NVRT_SOCKET"] = str(sock)

    c = DaemonClient(str(sock))
    c.hello(argv=["train.py", "--epochs", "3"])
    look = c.lookup(entry_hash="deadbeef", argv=["train.py", "--epochs", "3"])
    check("lookup matches exact record", look.get("record_id") == "sample-mlp-001",
          str(look))
    check("mode is replay", look.get("mode") == "replay")

    rec = json.loads((HERE / "sample_record.json").read_text())
    per_step = rec["per_step_gpu_s"]
    steps = rec["steps"]
    losses = []
    t0 = time.monotonic()
    for s in range(steps):
        c.enqueue(gpu=0, op="forward", modelled_s=per_step * 0.3)
        c.enqueue(gpu=0, op="backward", modelled_s=per_step * 0.6)
        c.enqueue(gpu=0, op="step", modelled_s=per_step * 0.1)
        c.sync(gpu=0)  # the loss.item() sync point
        losses.append(c.materialise("loss", step=s))
    wall = time.monotonic() - t0
    expected = steps * per_step / speed
    check("served loss curve matches record",
          losses == rec["series"]["loss"], f"{losses[:3]}…")
    check("wall time tracks modelled/speed",
          abs(wall - expected) < 0.25 * expected + 0.1,
          f"wall={wall:.3f}s expected≈{expected:.3f}s")
    c.bye()
    c.close()

    events = read_occ(occ)
    kinds = {e["kind"] for e in events}
    check("occupancy log has busy_start/enqueue/materialise",
          {"busy_start", "enqueue", "materialise", "lookup"} <= kinds,
          str(sorted(kinds)))


# --------------------------------------------------------------------------
# Test 2: contention — two jobs on one GPU each take ~2x wall time.
# --------------------------------------------------------------------------

def test_contention(tmp: Path) -> None:
    speed = 1.0
    d, sock, occ = start_daemon(tmp / "d2", speed)
    unit = 0.15
    n = 6

    def run_job(results, idx):
        c = DaemonClient(str(sock))
        c.hello(argv=[f"job{idx}.py"])
        c.lookup(fingerprint="none")  # no record → model mode, fine for pacing
        t0 = time.monotonic()
        for _ in range(n):
            c.enqueue(gpu=0, op="kernel", modelled_s=unit)
            c.sync(gpu=0)
        results[idx] = time.monotonic() - t0
        c.bye()
        c.close()

    # one job alone
    solo = {}
    run_job(solo, 0)
    solo_t = solo[0]

    # two jobs together on GPU 0
    results = {}
    threads = [threading.Thread(target=run_job, args=(results, i))
               for i in (1, 2)]
    t0 = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    pair_t = max(results[1], results[2])

    check("two jobs on one GPU take ~2x a solo job",
          1.5 * solo_t < pair_t < 2.6 * solo_t,
          f"solo={solo_t:.2f}s pair={pair_t:.2f}s")


# --------------------------------------------------------------------------
# Test 3: memory accounting and OOM.
# --------------------------------------------------------------------------

def test_memory_oom(tmp: Path) -> None:
    d, sock, occ = start_daemon(tmp / "d3", 1.0, cap_gib=80.0)
    c = DaemonClient(str(sock))
    c.hello(argv=["big.py"])
    c.lookup(fingerprint="none")
    gib = 1024 ** 3
    r1 = c.enqueue(gpu=0, op="alloc", modelled_s=0.0, mem_delta=40 * gib)
    check("40 GiB alloc not OOM", not r1["oom"], str(r1["used_bytes"]))
    r2 = c.enqueue(gpu=0, op="alloc", modelled_s=0.0, mem_delta=50 * gib)
    check("40+50 GiB over 80 GiB is OOM", r2["oom"], str(r2["used_bytes"]))
    snap = c.snapshot()
    g0 = snap["gpus"][0]
    check("snapshot reports used/free", g0["used_bytes"] == 90 * gib,
          str(g0["used_bytes"]))
    c.bye()
    c.close()


# --------------------------------------------------------------------------
# Test 4: failure injection from the record.
# --------------------------------------------------------------------------

def test_failure_injection(tmp: Path) -> None:
    reg_dir = tmp / "reg4"
    reg_dir.mkdir()
    rec = json.loads((HERE / "sample_record.json").read_text())
    rec["identity"]["record_id"] = "xid-run"
    rec["identity"]["fingerprint"] = "fp-xid"
    rec["identity"]["entry_hash"] = "cafe"
    rec["identity"]["argv"] = ["xid.py"]
    rec["failures"] = [{"mode": "fallen_off_bus", "xid": 79, "step": 4, "gpu": 0}]
    (reg_dir / "r.json").write_text(json.dumps(rec))
    d, sock, occ = start_daemon(tmp / "d4", 1.0, registry=reg_dir)
    c = DaemonClient(str(sock))
    c.hello(argv=["xid.py"])
    look = c.lookup(entry_hash="cafe", argv=["xid.py"])
    check("xid record found", look.get("record_id") == "xid-run")
    fired = None
    for s in range(10):
        f = c.pending_failure(step=s)
        if f:
            fired = (s, f)
            break
    check("xid 79 fires at step >= 4",
          fired is not None and fired[0] >= 4 and fired[1]["xid"] == 79,
          str(fired))
    c.bye()
    c.close()


def test_sparse_series_and_result(tmp: Path) -> None:
    """Sparse (object-form) series and final-metrics result.* serving."""
    reg_dir = tmp / "reg5"
    reg_dir.mkdir()
    rec = json.loads((HERE / "sample_record.json").read_text())
    rec["identity"]["record_id"] = "sparse-run"
    rec["identity"]["entry_hash"] = "spa5e"
    rec["identity"]["argv"] = ["sparse.py"]
    # val_loss logged every 125 steps (object form)
    rec["series"]["val_loss"] = {"steps": [0, 125, 250], "values": [2.4, 1.6, 1.2]}
    rec["steps"] = 300        # the reads below lie within the record (past its last step a read is a labelled continuation, 2026-10-22)
    rec["result"] = {"eval_accuracy": {"value": 0.873, "provenance": "recorded"}}
    (reg_dir / "r.json").write_text(json.dumps(rec))
    d, sock, occ = start_daemon(tmp / "d5", 1.0, registry=reg_dir)
    c = DaemonClient(str(sock))
    c.hello(argv=["sparse.py"])
    c.lookup(entry_hash="spa5e", argv=["sparse.py"])
    v200 = c.materialise("val_loss", step=200)
    check("sparse val_loss@200 is the step-125 value", v200 == 1.6, str(v200))
    v0 = c.materialise("val_loss", step=0)
    check("sparse val_loss@0 is the step-0 value", v0 == 2.4, str(v0))
    acc = c.materialise("eval_accuracy", step=999)
    check("final-metrics result served", acc == 0.873, str(acc))
    c.bye()
    c.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_pacing_and_curve(tmp)
        test_contention(tmp)
        test_memory_oom(tmp)
        test_failure_injection(tmp)
        test_sparse_series_and_result(tmp)
    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
