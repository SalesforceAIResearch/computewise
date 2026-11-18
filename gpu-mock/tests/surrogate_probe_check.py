#!/usr/bin/env python3
"""The checks of tests/test_surrogate_in_image.sh, run inside the box after
the three launches: occupancy log, lr_stream.jsonl and the three run outputs.
    python surrogate_probe_check.py <out dir> <exit A> <exit B> <exit C> <run dir>
"""
import json, math, re, sys
from pathlib import Path

out = Path(sys.argv[1]); ra, rb, rc = (int(x) for x in sys.argv[2:5]); run = Path(sys.argv[5])
ev = [json.loads(l) for l in (out / "occupancy.jsonl").read_text().splitlines() if l.strip()]
fails = 0; n = 0


def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def parse(path):
    text = (run / path).read_text()
    tr = [float(m.group(1)) for m in re.finditer(r"train_loss (\S+)", text)]
    va = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"val step (\d+) val_loss (\S+)", text)}
    return tr, va


ta, va = parse("a.out"); tb, vb = parse("b.out"); tc, vc = parse("c.out")
jobs = sorted({e["job"] for e in ev if e["kind"] == "job_bye"})
chk("three jobs seen (job_bye each)", len(jobs) == 3, str(jobs))
if len(jobs) < 3:
    print(f"{n - fails} passed, {fails} failed"); sys.exit(1)
A, B, C = jobs
look = [e for e in ev if e["kind"] == "lookup"]
la = [e for e in look if e["job"] == A]
chk("run A exits 0, 30 steps, validations at 10/20/30", ra == 0 and len(ta) == 30 and sorted(va) == [10, 20, 30], f"exit {ra} steps {len(ta)} val {sorted(va)}")
chk("A: pending at the first CUDA touch, then surrogate at the first step", [e.get("match") for e in la][:2] == ["surrogate_pending", "surrogate"], str([e.get("match") for e in la]))
chk("A: draw synthesised, record id names the family", la and all(e.get("draw") == "synthesised" and str(e.get("record_id", "")).startswith("surrogate:toy-gpt:v1:") for e in la), str([e.get("record_id") for e in la]))
fin = next((e for e in la if e.get("match") == "surrogate"), {})
sur = fin.get("surrogate") or {}
chk("A: the surrogate block carries the argv knobs, steps and ordinal 0", sur.get("config", {}).get("lr") == 0.02 and sur.get("steps") == 30 and sur.get("launch_ordinal") == 0 and sur.get("pending") is False, json.dumps(sur)[:200])
mats = [e for e in ev if e["kind"] == "materialise" and e["job"] == A]
chk("A: every served value is tier surrogate (at least 33 reads)", len(mats) >= 33 and all(e.get("tier") == "surrogate" for e in mats), f"{len(mats)} reads, tiers {sorted({str(e.get('tier')) for e in mats})}")
chk("A: printed losses are the family curve, not the CPU arithmetic (first train loss above 2)", bool(ta) and ta[0] > 2.0 and all(math.isfinite(v) for v in ta), str(ta[:3]))
chk("A: train loss falls, val loss falls", bool(ta) and ta[-1] < ta[0] and va.get(30, 9) < va.get(10, 0), f"{ta[:1]}->{ta[-1:]}, val {va}")
chk("A: train loss sits at or below the val loss (gap 0.03, noise 0.03)", all(ta[s - 1] < va[s] + 0.12 for s in va if s - 1 < len(ta)), str([(round(ta[s - 1], 3), round(va[s], 3)) for s in sorted(va) if s - 1 < len(ta)]))
lb = [e for e in look if e["job"] == B and e.get("match") == "surrogate"]
chk("B: a rerun of the same configuration is launch ordinal 1 with another seed", bool(lb) and lb[0]["surrogate"]["launch_ordinal"] == 1 and lb[0]["surrogate"]["seed"] != sur.get("seed") and lb[0]["surrogate"]["config_hash"] == sur.get("config_hash"), str(lb[0].get("surrogate", {}).get("launch_ordinal") if lb else None))
chk("B: same curve, different realisation", rb == 0 and all(abs(va[s] - vb.get(s, 99)) < 0.08 for s in va) and any(va[s] != vb.get(s) for s in va), str([(round(va[s], 4), round(vb.get(s, 0), 4)) for s in sorted(va)]))
ctext = (run / "c.out").read_text()
chk("C: lr 0.2 diverges after 5 steps, the script's own guard exits 3", rc == 3 and "nan at step 5" in ctext, f"exit {rc}: {ctext.strip().splitlines()[-1] if ctext.strip() else ''}")
fl = [e for e in ev if e["kind"] == "failure"]
chk("C: one failure event, mode surrogate_divergence, at step 5", len(fl) == 1 and fl[0]["job"] == C and fl[0].get("mode") == "surrogate_divergence" and fl[0].get("step") == 5, str(fl))
byes = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
chk("job_bye: surrogate true, lr_steps 30, last_step 30 for A; last_step 5 for C", byes[A].get("surrogate") is True and byes[A].get("lr_steps") == 30 and byes[A].get("last_step") == 30 and byes[C].get("last_step") == 5, str({k: (byes[k].get("surrogate"), byes[k].get("lr_steps"), byes[k].get("last_step")) for k in byes}))
enq = [e for e in ev if e["kind"] == "enqueue" and e["job"] == A and e.get("unit") == "step" and e.get("lr") is not None]
# lr_at(21) equals the hold value (0.02·10/10), so the first change of the cooldown is step 22
chk("A: the LR stream logs on change only: warm-up 1..3 and cooldown 22..30 (12 events, two groups)", [e["step"] for e in enq] == [1, 2, 3] + list(range(22, 31)) and all(len(e["lr"]) == 2 for e in enq), str([e["step"] for e in enq]))
sp = out / "lr_stream.jsonl"
stream = [json.loads(l) for l in sp.read_text().splitlines()] if sp.exists() else []
sa = next((s for s in stream if s["job"] == A), None)
chk("lr_stream.jsonl: A has 30 steps of two groups, step 1 at lr/3 and step 30 at lr/10", sa is not None and sa["steps"] == 30 and sa["groups"] == 2 and abs(sa["lr"][0][1][0] - 0.02 / 3) < 1e-6 and abs(sa["lr"][29][1][0] - 0.002) < 1e-6, str((sa["lr"][:2] + sa["lr"][-1:]) if sa else stream))
chk("no record-tier read anywhere (nothing in the registry)", not [e for e in ev if e["kind"] == "materialise" and e.get("tier") != "surrogate"])
chk("no exception events (the NaN exit is the script's own SystemExit)", not [e for e in ev if e["kind"] == "exception"], str([e.get("etype") for e in ev if e["kind"] == "exception"]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
