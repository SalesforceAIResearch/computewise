#!/usr/bin/env python3
"""next/ r7.1: the lookup reply carries no result. A client that reaches the
daemon socket and sends `lookup` for a record learns how to pace the job and
which seams to install, never what the job scores: no measured per-example
scores, no final score, no outcome, no failure text. Results reach a job only
through `materialise` (gated on its paced unit) and `fail?` (at its step).

Checked on the real registry records (gpu-tasks/registry/records), every one:
  1  no reply anywhere carries a result-bearing key (searched recursively by name);
  2  a KV record's reply carries the per_example fields the patch reads
     (generate_seam, score_seam, n_examples_full, score_kind) and nothing else of
     the block: no per_example_scores, decode, results_file, final_print;
  3  a failure entry carries mode, where, replay, at (and armed when the lookup
     arms it) and no cause, exception, stderr, trigger, threshold;
  4  through a live daemon: a KV lookup's reply has no measured score, and the
     same record still serves its measured score at position 0 to a job that
     paced example 0.
On r7's daemon checks 1-4 fail (r6 put per_example_scores in the reply); on r7.1 they pass.

    python3 -I tests/test_lookup_no_results_daemon.py [<daemon tree src dir>]
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE.parent / "src"
# the registry: REG in the environment (the E2 keeps it under /opt/computewise-deploy, apart from the gate tree),
# else the repository layout beside gpu-mock
REG = Path(os.environ["REG"]) if os.environ.get("REG") else HERE.parent.parent.parent / "gpu-tasks" / "registry" / "records"
sys.path.insert(0, str(SRC / "torch_patch"))
from daemon_client import DaemonClient  # noqa: E402

spec = importlib.util.spec_from_file_location("gd", SRC / "daemon" / "gpu_daemon.py")
gd = importlib.util.module_from_spec(spec)
sys.modules["gd"] = gd
spec.loader.exec_module(gd)

n = fails = 0


def chk(name, cond, detail=""):
    global n, fails
    n += 1
    if not cond:
        fails += 1
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


# keys whose value is (or can carry) what a job scores or how it failed
RESULT_KEYS = {"per_example_scores", "scores", "final_score", "outcome", "result", "results_file",
               "final_print", "cause", "exception", "stderr", "decode", "measured", "example_scores"}


def result_keys(obj, path=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in RESULT_KEYS:
                out.append(f"{path}.{k}")
            out += result_keys(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += result_keys(v, f"{path}[{i}]")
    return out


recs = []
if not REG.is_dir() or not any(REG.glob("mls-llm-kv-*.json")):
    print(f"[FAIL] the registry {REG} holds no KV records: nothing to check (set REG)")
    sys.exit(1)
for p in sorted(REG.glob("*.json")):
    try:
        r = json.load(open(p))
    except Exception:
        continue
    if isinstance(r, dict) and "identity" in r:
        recs.append(r)
leaks = {}
for r in recs:
    found = result_keys(gd._serving_info(r))
    if found:
        leaks[r["identity"]["record_id"]] = found
ex = next(iter(leaks.items()), (None, []))
chk(f"1: no reply over {len(recs)} registry records carries a result-bearing key", not leaks,
    f"{len(leaks)} records leak, e.g. {ex[0]}: {ex[1][:4]}")

kv = [r for r in recs if gd._per_example(r)]
need = {"generate_seam", "score_seam", "n_examples_full", "score_kind"}
bad = [r["identity"]["record_id"] for r in kv
       if not need <= set((gd._serving_info(r).get("per_example") or {}))
       or set(gd._serving_info(r).get("per_example") or {}) & {"per_example_scores", "decode", "results_file", "final_print"}]
chk(f"2: each of {len(kv)} KV replies carries the seam and count fields and none of the block's results",
    kv and not bad, f"{len(bad)} bad, e.g. {bad[:2]}")

fl = [f for r in recs for f in (gd._serving_info(r).get("failures") or [])]
badf = [sorted(f) for f in fl if set(f) & {"cause", "exception", "stderr", "trigger", "threshold", "step", "time_s"}]
chk(f"3: each of {len(fl)} failure entries carries only mode/where/replay/at", fl and not badf,
    f"{len(badf)} carry more, e.g. {badf[:1]}")

# 4: through a live daemon
# a record whose first measured score is nonzero, so a served 0.0 (the score
# seam's fallback) cannot pass for the measured value
kvrec = next((r for r in kv if gd._measured_scores(r) and gd._measured_scores(r)["scores"][0] > 0), None)
if kvrec is None:
    chk("4: a KV record with measured scores to test against", False, "none in the registry")
else:
    tmp = Path(tempfile.mkdtemp())
    (tmp / "reg").mkdir()
    (tmp / "reg" / "kv.json").write_text(json.dumps(kvrec))
    sock = tmp / "d.sock"
    d = subprocess.Popen([sys.executable, "-I", str(SRC / "daemon" / "gpu_daemon.py"), "--socket", str(sock),
                          "--registry", str(tmp / "reg"), "--gpus", "4", "--speed", "1000",
                          "--occupancy", str(tmp / "occ.jsonl")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            if sock.exists():
                break
            time.sleep(0.05)
        ident = kvrec["identity"]
        c = DaemonClient(str(sock))
        c.hello(argv=list(ident["argv"]))
        rep = c.lookup_v2(job_key="probe", rank=0, world_size=1, entry_hash=ident["entry_hash"], argv=list(ident["argv"]),
                          env={}, subset_env={}, script_path=None, fingerprint=None, fingerprint_inputs=None, stage="exact")
        chk("4a: a live KV lookup is a replay and its reply carries no result-bearing key",
            rep.get("mode") == "replay" and not result_keys(rep), str(result_keys(rep)[:4]))
        meas = gd._measured_scores(kvrec)
        wall = float(gd._val((kvrec.get("timing") or {}).get("wall_total_s")))
        nf = int(gd._val(gd._per_example(kvrec)["n_examples_full"]))
        c.enqueue(gpu=0, op="replay", modelled_s=wall / nf, step=1, unit="example")
        c.sync(gpu=0)
        v = c.materialise("example_score", 0)
        chk("4b: the same record still serves its measured (nonzero) score at position 0 after example 0 is paced",
            v == meas["scores"][0] and v > 0, f"{ident['record_id']}: served {v}, measured {meas['scores'][0]}")
        c.bye(); c.close()
    finally:
        d.terminate()
        d.wait(timeout=10)

print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
