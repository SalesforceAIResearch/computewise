#!/usr/bin/env python3
"""A launch that ends before the short-run floor (SHORT_RUN_STEPS, capped at a
quarter of the record) gives its record back:
an agent's smoke test must not cost its real run the record. The next launch
of the configuration draws the same record as a first draw; a run past the
floor keeps its draw; a seeded smoke test returns the seed's record."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(__file__))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402
import gpu_daemon as gd  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

H = "a" * 64
def rec(rid, seed, steps=1000):
    r = {"identity": {"record_id": rid, "entry_hash": H, "argv": ["train_gpt.py"], "match_policy": "exact_only", "replicate_group": "g"},
         "mode": "replay", "steps": steps, "timing": {"pacing": "per_step", "per_step_gpu_s": {"value": 0.01, "provenance": "recorded"}, "steady_step_s": 0.01},
         "launch": {"world_size": 1}, "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
         "series": {"train_loss": {"steps": list(range(1, steps + 1)), "values": [1.0] * steps}}, "readiness": {"replay": True}, "failures": []}
    if seed is not None:
        r["identity"]["seed"] = {"value": seed, "provenance": "recorded"}
    return r
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "s7.json").write_text(json.dumps(rec("run-s7", 7)))
    (R / "s8.json").write_text(json.dumps(rec("run-s8", 8)))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    jobs = 0
    def job(argv, steps_run):
        global jobs; jobs += 1
        c = DaemonClient(str(sock)); c.hello(argv=argv)
        r = c.lookup_v2(job_key=f"J{jobs}", rank=0, world_size=1, entry_hash=H, argv=argv, env={}, subset_env={}, argv_effective=None,
                        script_path="/w/train_gpt.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        for st in range(1, steps_run + 1):
            c.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=(1 << 30) if st == 1 else 0, step=st, unit="step")
        c.bye(); c.close(); return r
    chk("floor: 50 for a 1000-step record, a quarter for a short one", gd._short_run_floor(rec("x", None)) == 50 and gd._short_run_floor(rec("x", None, steps=120)) == 30 and gd._short_run_floor(None) == 50, str((gd._short_run_floor(rec("x", None)), gd._short_run_floor(rec("x", None, steps=120)))))
    r1 = job(["train_gpt.py", "--seed", "7", "--stop-at", "4"], 4)
    chk("A: the seeded smoke test (--seed 7 --stop-at 4) is the first 4 steps of the seed's record: match exact_prefix, draw exact_seed", r1.get("record_id") == "run-s7" and r1.get("draw") == "exact_seed" and r1.get("match") == "exact_prefix", str({k: r1.get(k) for k in ("record_id", "draw", "match")}))
    r2 = job(["train_gpt.py", "--seed", "7"], 60)
    chk("B: the real run of the same seed gets the SAME record as a first draw, not recycled", r2.get("record_id") == "run-s7" and r2.get("draw") == "exact_seed", str({k: r2.get(k) for k in ("record_id", "draw")}))
    r3 = job(["train_gpt.py", "--seed", "7"], 60)
    chk("C: the 60-step run of a 1000-step record was INCOMPLETE (returned), so another seed-7 launch gets its record again as a first draw", r3.get("record_id") == "run-s7" and r3.get("draw") == "exact_seed", str({k: r3.get(k) for k in ("record_id", "draw")}))
    r4 = job(["train_gpt.py", "--stop-at", "3"], 3)
    chk("D: an unseeded smoke test (--stop-at 3) takes the first unused record (run-s7, returned by C) through exact_prefix and returns it", r4.get("record_id") == "run-s7" and r4.get("draw") == "without_replacement" and r4.get("match") == "exact_prefix", str({k: r4.get(k) for k in ("record_id", "draw", "match")}))
    r5 = job(["train_gpt.py"], 200)
    chk("E: the next unseeded launch gets run-s7 again as a first draw (D returned it); run-s8 stays unused", r5.get("record_id") == "run-s7" and r5.get("draw") == "without_replacement", str({k: r5.get(k) for k in ("record_id", "draw")}))
    # I. an incomplete run past the floor (the agent cancelled at step 384 of 1000) returns its record too,
    #    so a resubmit of the same seed shows the same curve; a COMPLETED run keeps its draw
    (R / "s9.json").write_text(json.dumps(rec("run-s9", 9)))
    try:
        d.registry.__init__(R)
    except Exception:
        pass
    r9a = job(["train_gpt.py", "--seed", "9"], 384)
    r9b = job(["train_gpt.py", "--seed", "9"], 1000)
    r9c = job(["train_gpt.py", "--seed", "9"], 10)
    chk("I: seed 9 cancelled at 384/1000 → the resubmit of seed 9 gets its own record again as a first draw", r9a.get("draw") == "exact_seed" and r9b.get("record_id") == "run-s9" and r9b.get("draw") == "exact_seed", str([(r.get("record_id"), r.get("draw")) for r in (r9a, r9b)]))
    chk("I: after the COMPLETED run (1000/1000) the next seed-9 launch is a reuse, not a fresh draw", r9c.get("draw") not in ("exact_seed", "without_replacement", "seed_substituted"), str(r9c.get("draw")))
    r6 = job(["train_gpt.py", "--stop-at", "5000"], 1)
    chk("G: a --stop-at beyond the record's length is not a prefix: miss", r6.get("match") == "none", str(r6.get("match")))
    ev = read_occ(occ)
    lk = [e for e in ev if e["kind"] == "lookup"]
    chk("H: the prefix lookups carry stop_at on the event", [e.get("stop_at") for e in lk if e.get("match") == "exact_prefix"] == [4, 3], str([e.get("stop_at") for e in lk]))
    byes = [e for e in ev if e["kind"] == "job_bye"]
    chk("F: job_bye labels the smoke tests draw_returned_short_run (with the floor), the cancelled 384-step run draw_returned_incomplete, the 60/200/1000-step runs nothing (60 and 200 < 1000 are incomplete too: returned)",
        [(e.get("last_step"), e.get("draw_returned_short_run"), e.get("short_run_floor"), e.get("draw_returned_incomplete")) for e in byes] ==
        [(4, True, 50, None), (60, None, None, True), (60, None, None, True), (3, True, 50, None), (200, None, None, True), (384, None, None, True), (1000, None, None, None), (10, None, None, None), (1, None, None, None)],
        str([(e.get("last_step"), e.get("draw_returned_short_run"), e.get("short_run_floor"), e.get("draw_returned_incomplete")) for e in byes]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
