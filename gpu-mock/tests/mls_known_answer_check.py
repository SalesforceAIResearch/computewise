#!/usr/bin/env python3
"""Checks for tests/mls_known_answer_in_image.sh."""
import json, os, math, re, sys
from pathlib import Path
out, work, policy, workload, k, rc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
text = (out / "run.out").read_text() + (out / "run.err").read_text()
m = re.search(r"TEST_METRICS: (.*)", text)
kv = dict(p.split("=", 1) for p in m.group(1).split() if "=" in p) if m else {}
chk("the script exits 0 and prints TEST_METRICS", rc == 0 and m is not None and "final_score" in kv, f"exit {rc}; {'no TEST_METRICS line' if m is None else m.group(0)}")
if m is None or "final_score" not in kv:
    print(f"{n - fails} passed, {fails} failed"); sys.exit(1)
quant = "kv_compression_ratio" in kv       # the quant family prints effective_kv_bits and kv_compression_ratio instead of mean_retained_fraction
fs = float(kv["final_score"]); rt = float(kv.get("runtime_seconds", "nan")); rf = float(kv.get("mean_retained_fraction", kv.get("kv_compression_ratio", "nan")))
ev = [json.loads(l) for l in (out / "occupancy.jsonl").read_text().splitlines() if l.strip()]
look = [e for e in ev if e["kind"] == "lookup"]
chk("one lookup, " + ("exact_effective (the evaluator launched directly, defaults filled in)" if os.environ.get("DIRECT") == "1" else "exact match") + ", replay, the record for this policy and workload", len(look) == 1 and look[0].get("match") == ("exact_effective" if os.environ.get("DIRECT") == "1" else "exact") and look[0].get("mode") == "replay" and policy.replace("_", "-") in str(look[0].get("record_id")) and workload.replace("_", "-") in str(look[0].get("record_id")), str([(e.get("match"), e.get("mode"), e.get("record_id"), e.get("subset")) for e in look]))
rec = json.load(open(next(Path("/tmp/reg").glob(f"*{policy.replace('_', '-')}*{workload.replace('_', '-')}*.json"))))
final = rec["result"]["final_score"]["value"]; wall = rec["timing"]["wall_total_s"]["value"]
ser = rec["materialisation"]["serving"]; n_full = ser["n_examples_full"]["value"]
mats = [e for e in ev if e["kind"] == "materialise" and e.get("role") == "example_score"]
vals = [e.get("value") for e in mats]
if k == 0 or k >= n_full:
    chk("full launch: final_score is the recorded value through the script's own mean", abs(fs - final) < 1e-5, f"{fs} vs {final}")
    chk("full launch: n_full scores served", len(vals) == n_full, str(len(vals)))
    chk("full launch: runtime_seconds is the recorded wall within 5 % (the script's own clock, under the virtual clock)", abs(rt - wall) / wall < 0.05, f"{rt:.1f} vs {wall}")
    chk("no subset on the lookup", look[0].get("subset") is None)
else:
    chk(f"subset k={k}: final_score is the mean of the first {k} served scores", len(vals) == k and abs(fs - sum(vals) / k) < 1e-5, f"{fs} vs {sum(vals)/k if vals else None} ({len(vals)} served)")
    chk("subset on the lookup and provenance synthesised_subset on every score", look[0].get("subset") == {"k": k, "n_full": n_full} and all(e.get("provenance") == "synthesised_subset" for e in mats), str(look[0].get("subset")))
    load_guess = 6.0
    exp = load_guess + k * (wall - load_guess) / n_full
    chk("subset: runtime_seconds about load + k/n of the paced examples", 0.3 * exp < rt < 3.0 * exp + 30, f"{rt:.1f} s vs about {exp:.1f}")
def _find(o, key):
    if isinstance(o, dict):
        for kk, v in o.items():
            if kk == key:
                return v.get("value") if isinstance(v, dict) and "value" in v else v
            r_ = _find(v, key)
            if r_ is not None:
                return r_
    if isinstance(o, list):
        for v in o:
            r_ = _find(v, key)
            if r_ is not None:
                return r_
    return None
rec_rf = _find(rec, "kv_compression_ratio" if quant else "mean_retained_fraction")
if quant:
    chk("quant: kv_compression_ratio is the script's own CPU computation, equal to the recorded ratio", isinstance(rec_rf, (int, float)) and abs(rf - float(rec_rf)) < 1e-5, f"{rf} vs recorded {rec_rf}")
elif policy == "full_attention":
    chk("mean_retained_fraction: 1.0 for full_attention", abs(rf - 1.0) < 1e-9, str(rf))
elif (k == 0 or k >= n_full) and isinstance(rec_rf, (int, float)):
    chk("full launch: mean_retained_fraction is the recorded figure, computed from the real prompt lengths", abs(rf - float(rec_rf)) < 5e-4, f"{rf} vs recorded {rec_rf}")
else:
    chk("mean_retained_fraction in (0, 1) for a compressing policy", 0.0 < rf < 1.0, str(rf))
chk("every served score is tier result", mats and all(e.get("tier") == "result" for e in mats))
byes = [e for e in ev if e["kind"] == "job_bye"]
chk("job_bye: served result, reason recorded", len(byes) == 1 and byes[0].get("served") == "result", str([(e.get("served"), e.get("reason"), e.get("subset")) for e in byes]))
mf = work / "output" / f"{workload}_metrics.json"
mj = json.loads(mf.read_text()) if mf.exists() else {}
chk("the metrics file holds the printed numbers (the line prints 6 decimals)", mf.exists() and abs(mj.get("final_score", -1) - fs) < 1e-6 and abs(mj.get("kv_compression_ratio" if quant else "mean_retained_fraction", -1) - rf) < 1e-6, str(mj))
tf = work / "output" / f"{workload}_trace.json"
tj = json.loads(tf.read_text()) if tf.exists() else {}
chk("the trace file holds aggregates only (no per-example text)", tf.exists() and all(not isinstance(v, (list, dict)) for v in tj.values()), str(list(tj.keys())))
su = sorted({e.get("startup_s") for e in mats}) if mats else []
bye_su = [e.get("startup_s") for e in byes]
ex_enq = [e for e in ev if e["kind"] == "enqueue" and e.get("unit") == "example"]
pace = float(ex_enq[0].get("modelled_s")) if ex_enq else None
wall_total = float((rec.get("timing") or {}).get("wall_total_s", {}).get("value") or (rec.get("result") or {}).get("runtime_seconds", {}).get("value") or 0.0) if rec else 0.0
chk("startup_s: the launch's real start-up on every served score and on job_bye, and wall_total − n_full × pace recovers it", len(su) == 1 and isinstance(su[0], float) and bye_su == [su[0]] and pace is not None and n_full and abs((wall_total - n_full * pace) - su[0]) < 0.05, f"scores {su} bye {bye_su} wall {wall_total} n {n_full} pace {pace}")
# the pacing identity itself: runtime = startup_s + k × pace (a subset) and = wall_total (a full launch)
if su and pace is not None:
    k_eff = k if (k and k < n_full) else n_full
    ident = su[0] + k_eff * pace
    chk("the identity runtime_seconds = startup_s + k × pace holds within 0.2 s (a full launch: the recorded wall)", abs(rt - ident) < 0.2 and (k_eff < n_full or abs(rt - wall_total) < 0.2), f"runtime {rt:.3f} identity {ident:.3f} wall {wall_total}")
    sr = sorted({e.get("startup_real_s") for e in mats})
    chk("startup_real_s beside startup_s on every score and on job_bye", len(sr) == 1 and isinstance(sr[0], float) and [e.get("startup_real_s") for e in byes] == sr, f"{sr}")
err_text = (out / "run.err").read_text()
warn = [l for l in err_text.splitlines() if "per_example:" in l and ("failed" in l or "not computed" in l)]
chk("no per_example seam warnings in run.err (a seam fallback is a defect, not a result)", not warn, (warn[0][:160] + f" ... {len(warn)} lines") if warn else "")
if tf.exists() and "method" in tj:
    chk("the trace's method is the policy's own plan, not the seam's fallback", tj.get("method") not in (None, "", "selection_policy"), str(tj.get("method")))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
