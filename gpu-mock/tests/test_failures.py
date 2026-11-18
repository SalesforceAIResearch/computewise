#!/usr/bin/env python3
"""Recorded failures: which fire, and when (RECORD-FORMAT `failures`). Uses the
daemon's own _pending_failure on records from gpu-tasks' registry (read-only)
and on synthetic ones. No socket, no torch.
    python3 -I tests/test_failures.py
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "daemon"))
import gpu_daemon  # noqa: E402
REG = ROOT.parent / "gpu-tasks" / "registry" / "records"
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else "")); PASS += bool(cond); FAIL += (not cond)


class _Occ:
    def event(self, *a, **k):
        pass


d = gpu_daemon.GPUDaemon.__new__(gpu_daemon.GPUDaemon)
d.occ = _Occ()


def fires(rec, step, phase="training", job=None):
    job = job or gpu_daemon.Job(job_id=1, pid=1, argv=[], cwd="", record=rec)
    return d._pending_failure(job, {"step": step, "phase": phase}).get("failure")


sft = json.load(open(REG / "trl-sft-h100-sxmx1-0022af66-s1-afosdmfe.json"))
check("the SFT record's Hub-push crash is written as a suppressed value with provenance",
      sft["failures"][0]["replay"].get("value") == "suppress")
check("it does not fire at its step", fires(copy.deepcopy(sft), 30) is None)
check("it does not fire after training", fires(copy.deepcopy(sft), 30, "after_training") is None)
serving = gpu_daemon._serving_info(sft)
check("serving info carries replay as a plain string", serving["failures"][0]["replay"] == "suppress")

def rec_with(**f):
    return {"failures": [dict({"mode": "oom", "step": 3, "at": "during_training"}, **f)]}

check("a fire-by-default failure fires at its step", (fires(rec_with(), 3) or {}).get("mode") == "oom")
check("not before its step", fires(rec_with(), 2) is None)
check("a fire value with provenance fires", fires(rec_with(replay={"value": "fire"}), 3) is not None)
r = rec_with(); j1 = gpu_daemon.Job(job_id=1, pid=1, argv=[], cwd="", record=r); fires(r, 3, job=j1)
check("fires once per job", fires(r, 4, job=j1) is None)
j2 = gpu_daemon.Job(job_id=2, pid=2, argv=[], cwd="", record=r)   # another job, the same record dict
check("fires again for another job drawing the same record", (fires(r, 3, job=j2) or {}).get("mode") == "oom")
check("the record itself carries no fired flag", not any("_fired" in f for f in r["failures"]))
ra = rec_with(mode="crash", at="after_training", replay="fire")
check("an after-training failure never fires at a step", fires(ra, 99) is None)
check("it fires when training has ended", (fires(ra, 99, "after_training") or {}).get("mode") == "crash")
check("a during-training failure does not fire after training", fires(rec_with(), 3, "after_training") is None)
print(f"\n{PASS} passed, {FAIL} failed"); sys.exit(1 if FAIL else 0)
