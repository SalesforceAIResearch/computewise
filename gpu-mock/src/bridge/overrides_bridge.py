#!/usr/bin/env python3
"""Bridge: gpu-mock daemon state → Mokka overrides.yaml.

Mokka's mock libnvidia-ml re-reads overrides.yaml on a 1 s TTL and deep-merges it
over the pristine profile, so nvidia-smi and NVML report whatever we write here.
Left alone, Mokka moves utilization on elapsed time — a busy-looking number on an
idle node, which is a tell. This bridge overwrites it with the real busy fraction
the daemon measures, plus memory and the running-process list, so nvidia-smi and
torch.cuda tell the same story.

It samples the daemon's snapshot every 100 ms, reports utilization as the busy
fraction over the trailing 1 s window (so a data-loader-starved job shows the GPU
waiting), and writes memory as the used/free triple Mokka wants set together.

Schema (Mokka's config_override_merge.go, docs/configuration.md):
    { "all": {...}, "devices": {"0": {...}, ...} }
with per-device keys utilization{gpu,memory}, memory{used_bytes,free_bytes},
processes[], failure{mode,xid}. We write only devices[], one per GPU.

overrides.yaml is a strict YAML subset; we emit JSON, which every YAML parser
reads, so there is no yaml dependency. The write is atomic (temp + rename) so the
mock never reads a half-written file.

Run:  python3 overrides_bridge.py --socket <daemon.sock> \\
                                  --overrides /var/lib/nvml-mock/.../overrides.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Optional

SAMPLE_HZ = 10          # 100 ms
WINDOW_S = 1.0          # utilisation reported over the trailing 1 s
MIB = 1024 * 1024


class DaemonConn:
    def __init__(self, socket_path: str):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._f = self._sock.makefile("rwb", buffering=0)
        # The bridge is an observer, not a job: it says hello so the daemon can
        # log it as observer_hello (never job_hello, which the runner's scorer
        # takes a run's makespan from), and it enqueues no work.
        self._rpc({"cmd": "hello", "pid": os.getpid(), "observer": True,
                   "argv": ["gpu-mock-bridge"], "cwd": os.getcwd()})

    def _rpc(self, obj: dict) -> dict:
        self._f.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        line = self._f.readline()
        if not line:
            raise ConnectionError("daemon closed")
        return json.loads(line.decode())

    def snapshot(self) -> dict:
        return self._rpc({"cmd": "snapshot"}).get("snapshot", {})


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".overrides-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_overrides(snapshot: dict,
                    busy_windows: dict[int, deque]) -> dict:
    """Turn one snapshot + the busy-fraction history into an overrides doc."""
    devices: dict[str, dict] = {}
    for g in snapshot.get("gpus", []):
        idx = g["index"]
        win = busy_windows[idx]
        util_gpu = round(100 * (sum(win) / len(win))) if win else 0
        used = int(g["used_bytes"])
        free = int(g["free_bytes"])
        procs = []
        for p in g.get("processes", []):
            procs.append({
                "pid": int(p["pid"]),
                "type": "C",
                "name": "python",
                "used_memory_mib": int(p["used_bytes"] // MIB),
                "sm_util": util_gpu,
                "mem_util": util_gpu,
            })
        dev = {
            "utilization": {"gpu": util_gpu, "memory": util_gpu},
            "memory": {"used_bytes": used, "free_bytes": free},
        }
        if procs:
            dev["processes"] = procs
        devices[str(idx)] = dev
    return {"devices": devices}


def run(socket_path: str, overrides_path: Path) -> None:
    # Connect inside the loop, so the bridge may start before the daemon's
    # socket exists (it died with FileNotFoundError when started alongside the
    # daemon, 2026-10-20) and reconnects after the daemon restarts; the old loop
    # kept a dead connection forever.
    conn = None
    window_len = int(WINDOW_S * SAMPLE_HZ)
    busy_windows: dict[int, deque] = {}
    period = 1.0 / SAMPLE_HZ
    last_write = 0.0
    while True:
        t = time.monotonic()
        try:
            if conn is None:
                conn = DaemonConn(socket_path)
            snap = conn.snapshot()
        except (ConnectionError, OSError, ValueError):
            if conn is not None:
                try:
                    conn._sock.close()
                except OSError:
                    pass
            conn = None
            time.sleep(period)
            continue
        for g in snap.get("gpus", []):
            idx = g["index"]
            busy_windows.setdefault(idx, deque(maxlen=window_len))
            busy_windows[idx].append(1 if g.get("util_gpu", 0) > 0 else 0)
        # Write the merged overrides once per ~100 ms; the mock reads on its own
        # 1 s TTL, so writing faster than that only sharpens the window edge.
        if t - last_write >= period:
            doc = build_overrides(snap, busy_windows)
            atomic_write(overrides_path,
                         json.dumps(doc, separators=(",", ":"), indent=0))
            last_write = t
        sleep = period - (time.monotonic() - t)
        if sleep > 0:
            time.sleep(sleep)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/var/run/nvidia-persistenced/socket")
    ap.add_argument("--overrides", required=True)
    args = ap.parse_args()
    run(args.socket, Path(args.overrides))


if __name__ == "__main__":
    main()
