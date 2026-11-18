#!/usr/bin/env python3
"""Prefix upgrade: a launch that draws a
prefix / prepared record but whose stop does not fit it — no `--stop-at`, or
one past the prefix's steps — is served the record's full-schedule parent,
with the drawn (listed) record kept as the job's identity; a smoke test whose
stop fits keeps the prefix; a parent that is missing or not replay-ready
leaves the prefix; a completed full run keeps its draw, an incomplete one
returns it (through the drawn record)."""
import json, os, sys, tempfile
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

HP, HQ, HX = "a" * 64, "b" * 64, "e" * 64
def parent(rid, h, steps, ready=True):
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": ["train_gpt_simple_muonh.py"], "match_policy": "exact_only", "replicate_group": rid},
            "mode": "replay", "steps": steps, "timing": {"pacing": "per_step", "per_step_gpu_s": {"value": 0.01, "provenance": "recorded"}, "steady_step_s": 0.01},
            "launch": {"world_size": 8}, "memory": {"peak_allocated_bytes": 8 << 30, "peak": 8 << 30},
            "series": {"val_loss": {"steps": list(range(125, steps + 1, 125)), "values": [3.0] * (steps // 125)}},
            "readiness": {"replay": ready, "missing": ([] if ready else ["source log"])}, "failures": []}
def prefix(rid, par, steps, bp):
    r = parent(rid, HP, steps)
    r["identity"].update({"bound_path": bp, "match_policy": "path_bound", "replicate_group": bp.strip("/"), "process_model": "single",
                          "derived_from": {"kind": "prepared", "record": par, "steps": [1, steps], "stop_after_step": steps}})
    return r
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "P.json").write_text(json.dumps(parent("muonh-full-run1", HP, 3250)))
    (R / "X.json").write_text(json.dumps(prefix("muonh-run1-stop1000-h1prep", "muonh-full-run1", 1000, "setups/muonh-tuned/")))
    (R / "Q.json").write_text(json.dumps(parent("ema-full-seed8", HQ, 2900, ready=False)))
    (R / "Y.json").write_text(json.dumps(prefix("ema-seed8-stop1000-h1prep", "ema-full-seed8", 1000, "setups/ema-nesterov-aurora/")))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    jobs = 0
    def job(argv, path, steps_run=None):
        global jobs; jobs += 1
        c = DaemonClient(str(sock)); c.hello(argv=argv, cwd="/home/researcher/project")
        r = c.lookup_v2(job_key=f"J{jobs}", rank=0, world_size=1, entry_hash=HX, argv=argv, env={}, subset_env={}, argv_effective=None,
                        script_path="/home/researcher/project/" + path, fingerprint=None, fingerprint_inputs=None, stage="exact")
        if steps_run:
            for st in range(1, steps_run + 1):
                c.enqueue(gpu=0, op="replay", modelled_s=0.0001, mem_delta=(1 << 30) if st == 1 else 0, step=st, unit="step")
            c.bye(); c.close(); return r, None
        return r, c
    r, c = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py")
    chk("1: no --stop-at under setups/muonh-tuned/: the draw is the listed prefix, the SERVED record is its 3250-step parent", r.get("match") == "path" and r.get("record_id") == "muonh-run1-stop1000-h1prep" and (r.get("serving") or {}).get("steps") == 3250, str({k: r.get(k) for k in ("match", "record_id", "draw")}) + " serving.steps=" + str((r.get("serving") or {}).get("steps")))
    c.bye(); c.close()
    r, c = job(["train_gpt.py", "--stop-at", "125"], "setups/muonh-tuned/train_gpt.py")
    chk("2: --stop-at 125 fits the prefix: the prefix itself is served (1000 steps); a fresh draw, since launch 1 ended at step 0 and returned its record", (r.get("serving") or {}).get("steps") == 1000 and r.get("record_id") == "muonh-run1-stop1000-h1prep", "serving.steps=" + str((r.get("serving") or {}).get("steps")) + " draw=" + str(r.get("draw")))
    c.bye(); c.close()
    r, c = job(["train_gpt.py", "--stop-at", "2000"], "setups/muonh-tuned/train_gpt.py")
    chk("3: --stop-at 2000 is past the prefix: the parent is served", (r.get("serving") or {}).get("steps") == 3250, "serving.steps=" + str((r.get("serving") or {}).get("steps")))
    c.bye(); c.close()
    r, c = job(["train_gpt.py", "--seed", "99"], "setups/ema-nesterov-aurora/train_gpt.py")
    chk("4: a parent that is not replay-ready leaves the prefix in place (1000 steps)", (r.get("serving") or {}).get("steps") == 1000 and r.get("record_id") == "ema-seed8-stop1000-h1prep", "serving.steps=" + str((r.get("serving") or {}).get("steps")))
    c.bye(); c.close()
    r, _ = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py", steps_run=3250)
    chk("5: a full run to 3250 on the served parent keeps its draw (completed)", (r.get("serving") or {}).get("steps") == 3250 and r.get("draw") == "without_replacement", str(r.get("draw")))
    r, c = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py")
    chk("6: the next launch of the setup is a recycle (the unit was completed), still served the parent", r.get("draw") == "recycled" and (r.get("serving") or {}).get("steps") == 3250, str(r.get("draw")))
    c.bye(); c.close()
    # the W0-h1-speed1 defect: a second prefix record in the pool; a seedless
    # launch upgraded to the parent whose script stops at the prefix's own 1000 is COMPLETE and keeps its draw
    (R / "P2.json").write_text(json.dumps(parent("muonh-full-run2", "f" * 64, 3250)))
    (R / "X2.json").write_text(json.dumps(prefix("muonh-run2-stop1000-h1prep", "muonh-full-run2", 1000, "setups/muonh-tuned/")))
    (R / "P3.json").write_text(json.dumps(parent("muonh-full-run3", "d" * 64, 3250)))
    (R / "X3.json").write_text(json.dumps(prefix("muonh-run3-stop1000-h1prep", "muonh-full-run3", 1000, "setups/muonh-tuned/")))
    try:
        d.registry.__init__(R)       # one re-read, before these launches: it also resets the used sets
    except Exception:
        pass
    r1, _ = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py", steps_run=1000)
    r2, _ = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py", steps_run=1000)
    chk("9: two seedless launches, each stopping at the prefix's own 1000 while served the 3250 parent, draw DIFFERENT records (no return at the drawn stop)", r1.get("record_id") != r2.get("record_id") and {r1.get("record_id"), r2.get("record_id")} == {"muonh-run1-stop1000-h1prep", "muonh-run2-stop1000-h1prep"} and r1.get("draw") == "without_replacement" and r2.get("draw") == "without_replacement", str([(r1.get("record_id"), r1.get("draw")), (r2.get("record_id"), r2.get("draw"))]))
    # the third pair is still unused, so the kill case is a first draw (a recycled draw is never returned, by design)
    r3, _ = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py", steps_run=1500)
    r4, c = job(["train_gpt.py"], "setups/muonh-tuned/train_gpt.py")
    chk("10: a fresh launch killed at 1500, between the prefix's stop and the parent's end, returns its record; the next launch gets that record again as a first draw", r3.get("record_id") == "muonh-run3-stop1000-h1prep" and r3.get("draw") == "without_replacement" and r4.get("record_id") == r3.get("record_id") and r4.get("draw") == "without_replacement", str([(r3.get("record_id"), r3.get("draw")), (r4.get("record_id"), r4.get("draw"))]))
    c.bye(); c.close()
    ev2 = read_occ(occ)
    byes2 = [e for e in ev2 if e["kind"] == "job_bye" and e.get("last_step") in (1000, 1500)]
    chk("11: job_bye: the two 1000-step runs show no draw_returned_incomplete; the 1500-step one does", [(e.get("last_step"), e.get("draw_returned_incomplete")) for e in byes2] == [(1000, None), (1000, None), (1500, True)], str([(e.get("last_step"), e.get("draw_returned_incomplete")) for e in byes2]))
    ev = read_occ(occ)
    lk = [e for e in ev if e["kind"] == "lookup"]
    chk("7: lookup events carry served_from / served_steps / prefix_steps on the upgraded ones only", [(e.get("served_from"), e.get("served_steps"), e.get("prefix_steps")) for e in lk][:6] == [("muonh-full-run1", 3250, 1000), (None, None, None), ("muonh-full-run1", 3250, 1000), (None, None, None), ("muonh-full-run1", 3250, 1000), ("muonh-full-run1", 3250, 1000)], str([(e.get("served_from"), e.get("served_steps"), e.get("prefix_steps")) for e in lk]))
    byes = [e for e in ev if e["kind"] == "job_bye"]
    chk("8: job_bye keeps the drawn record id and reports last_step 3250 for the full run", any(e.get("last_step") == 3250 for e in byes), str([(e.get("last_step"), e.get("draw_returned_incomplete")) for e in byes]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
