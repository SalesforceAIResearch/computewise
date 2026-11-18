#!/usr/bin/env python3
"""Checks for tests/test_per_example_in_image.sh: five launches of the stand-in."""
import json, math, re, sys
from pathlib import Path
out = Path(sys.argv[1]); ra, rb, rc, rd, re_ = (int(x) for x in sys.argv[2:7]); work = Path(sys.argv[7])
ev = [json.loads(l) for l in (out / "occupancy.jsonl").read_text().splitlines() if l.strip()]
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
def metrics(f):
    m = re.search(r"TEST_METRICS: final_score=(\S+) mean_retained_fraction=(\S+) runtime_seconds=(\S+)", (out / f).read_text())
    return tuple(float(x) for x in m.groups()) if m else None
A, B, C, Dm, E = metrics("a.out"), metrics("b.out"), metrics("c.out"), metrics("d.out"), metrics("e.out")
chk("all five launches exit 0 and print TEST_METRICS", (ra, rb, rc, rd, re_) == (0, 0, 0, 0, 0) and None not in (A, B, C, Dm, E), f"exits {(ra, rb, rc, rd, re_)} {A} {B} {C} {Dm} {E}")
if None in (A, B, C, Dm, E):
    print(f"{n - fails} passed, {fails} failed"); sys.exit(1)
chk("full_attention full: final_score is the recorded 32.5 through the script's own mean (13 of 40)", abs(A[0] - 32.5) < 1e-6, str(A))
chk("full_attention: mean_retained_fraction 1.0 (disable_compression)", abs(A[1] - 1.0) < 1e-9, str(A))
lengths = json.loads((out / "lengths-streamingllm-40.json").read_text())
expect_rf = sum(int(L * 0.8) / L for L in lengths) / len(lengths)
chk("streamingllm: mean_retained_fraction is the script's own function of the real truncated prompt lengths", abs(B[1] - expect_rf) < 1e-6 and 0.7 < B[1] < 0.8, f"{B[1]:.6f} expected {expect_rf:.6f} from lengths {lengths[:5]}…")
chk("streamingllm: final_score is its own record's 5.0", abs(B[0] - 5.0) < 1e-6, str(B))
chk("full runs: runtime_seconds is the recorded wall (20 s) within 15 % (load real, examples paced)", 17.0 < A[2] < 23.0 and 17.0 < B[2] < 23.0, f"{A[2]:.2f} {B[2]:.2f}")
mats = [e for e in ev if e["kind"] == "materialise" and e.get("role") == "example_score"]
byes = [e for e in ev if e["kind"] == "job_bye"]
jobs = [e["job"] for e in byes]
vecA = [e.get("value") for e in mats if e["job"] == jobs[0]]
vecC = [e.get("value") for e in mats if e["job"] == jobs[2]]
chk("subset k=10: the first 10 entries of the same vector; final_score their mean", vecC == vecA[:10] and abs(C[0] - sum(vecA[:10]) / 10) < 1e-6, f"{C[0]} vs {sum(vecA[:10]) / 10 if vecA else None}")
chk("subset k=10: runtime about load + 10/40 of the paced examples (under 9 s)", 2.0 < C[2] < 9.0, f"{C[2]:.2f}")
look = [e for e in ev if e["kind"] == "lookup"]
chk("lookups: subset on the k=10 launches only (flag, then env); the fourth full_attention launch is a deterministic_rerun of the same record", [e.get("subset") for e in look] == [None, None, {"k": 10, "n_full": 40}, None, {"k": 10, "n_full": 40}] and look[3].get("draw") == "deterministic_rerun" and look[3].get("record_id") == look[0].get("record_id"), str([(e.get("record_id"), e.get("draw"), e.get("subset")) for e in look]))
chk("env knob only (SELECTION_KV_MAX_EXAMPLES=10, no flag): the same subset as the flag, identical printed numbers", E[0] == C[0] and E[1] == C[1] and [e.get("value") for e in mats if e["job"] == jobs[4]] == vecC, f"{E} vs {C}")
chk("the rerun prints the identical numbers", Dm[0] == A[0] and Dm[1] == A[1], f"{Dm} vs {A}")
chk("every served score is tier result; the subset's carry provenance synthesised_subset and subset_model", all(e.get("tier") == "result" for e in mats) and all(e.get("provenance") == "synthesised_subset" and e.get("subset_model") == {"kind": "bernoulli"} for e in mats if e["job"] == jobs[2]) and all(e.get("provenance") == "result" for e in mats if e["job"] == jobs[0]))
su = sorted({e.get("startup_s") for e in mats if e["job"] == jobs[0]}); sb = [e.get("startup_s") for e in byes]
chk("startup_s: one start-up per launch, on every served score and on job_bye (seconds, under 60)", len(su) == 1 and isinstance(su[0], float) and 0.0 <= su[0] < 60.0 and sb[0] == su[0] and all(isinstance(x, float) for x in sb), f"scores {su} byes {sb}")
chk("startup_real_s beside it (real seconds, at least the box-clock figure minus 1 s)", all(isinstance(e.get("startup_real_s"), float) and e["startup_real_s"] >= e["startup_s"] - 1.0 for e in mats if e["job"] == jobs[0]) and all(isinstance(e.get("startup_real_s"), float) for e in byes))
chk("job_bye: served result on all, subset on both k=10 launches", all(e.get("served") == "result" for e in byes) and [e.get("subset") for e in byes] == [None, None, {"k": 10, "n_full": 40}, None, {"k": 10, "n_full": 40}], str([e.get("subset") for e in byes]))
enq = [e for e in ev if e["kind"] == "enqueue" and e.get("unit") == "example" and e["job"] == jobs[0]]
chk("40 example units paced for the full launch, each (20 − load)/40 s", len(enq) == 40 and all(0.3 < float(e.get("modelled_s") or 0) <= 0.5 for e in enq), f"{len(enq)} units, first {enq[0].get('modelled_s') if enq else None}")
mj = json.loads((out / "artifacts" / "gsm8k_metrics.json").read_text())
chk("the metrics file holds the printed numbers of the last launch (the env-knob subset)", abs(mj["final_score"] - E[0]) < 1e-9 and abs(mj["mean_retained_fraction"] - E[1]) < 1e-9)
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
