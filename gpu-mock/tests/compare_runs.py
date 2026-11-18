#!/usr/bin/env python3
"""Compare two run_one.py run directories of the SAME plan: the known-answer
check for an image change. Reports, per job log:
served validation values (must be identical), the script's printed train_time
per validation (expected to differ only by what the change moved), and the
wall clock per phase from the two occupancy logs: process start to first
enqueue, each paced unit bucketed by its recorded length, host gaps beyond
pacing, and last cursor to exit.

    python3 -I tests/compare_runs.py <old run dir> <new run dir> [expected train_time shift s]

Nothing here asserts; it prints both sides and the deltas, and a final line
with the comparison's verdict on the two things the study named: same
served values, and wall clock moved only where the change says it did.
"""
from __future__ import annotations
import glob, json, os, re, sys
from collections import defaultdict

VAL = re.compile(r"^step:(\d+)/\d+ val_loss:([0-9.]+) train_time:([0-9.]+)s", re.M)


def vals(run):
    out = {}
    for p in sorted(glob.glob(os.path.join(run, "agent-logs", "*.log"))):
        out[os.path.basename(p)[:-4]] = [(int(a), float(b), float(c)) for a, b, c in VAL.findall(open(p, errors="replace").read())]
    return out


def phases(run):
    ev = [json.loads(l) for l in open(os.path.join(run, "occupancy.jsonl")) if l.strip()]
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    enq = defaultdict(list); cursor = defaultdict(float); last = {}
    for e in ev:
        if e["kind"] != "enqueue":
            continue
        start = max(cursor[e["gpu"]], e["t_mono"])
        cursor[e["gpu"]] = start + float(e.get("wall_cost") or 0.0)
        enq[e["job"]].append(e); last[e["job"]] = cursor[e["gpu"]] + (e["t_wall"] - e["t_mono"])
    out = {}
    for j, es in enq.items():
        h = hello[j]; t0 = h.get("t_start") or h["t_wall"]
        buckets = defaultdict(lambda: [0, 0.0, 0.0])      # rounded recorded length -> count, recorded s, wall s
        for e in es:
            b = buckets[round(float(e.get("modelled_s") or 0.0), 3)]
            b[0] += 1; b[1] += float(e.get("modelled_s") or 0.0); b[2] += float(e.get("wall_cost") or 0.0)
        out[j] = {
            "argv": " ".join(str(a) for a in (h.get("argv") or []))[-80:],
            "start_to_first_enqueue": es[0]["t_wall"] - t0,
            "first_enqueue_recorded_s": float(es[0].get("modelled_s") or 0.0),
            "paced_wall": sum(float(e.get("wall_cost") or 0.0) for e in es),
            "recorded_total": sum(float(e.get("modelled_s") or 0.0) for e in es),
            "gaps_beyond_paced": sum(max(0.0, b["t_mono"] - a["t_mono"] - float(a.get("wall_cost") or 0.0)) for a, b in zip(es, es[1:])),
            "last_cursor_to_bye": (bye[j]["t_wall"] - last[j]) if j in bye else float("nan"),
            "enqueues": len(es),
            "buckets": dict(sorted(buckets.items(), key=lambda kv: -kv[0])),
        }
    return out


def main(old, new, shift=None):
    verdict = []
    for run in (old, new):
        s = json.load(open(os.path.join(run, "status.json"))); r = json.load(open(os.path.join(run, "run.json")))
        img = r.get("image") or {}; sc = r.get("sidecar") or {}
        print(f"{run}: status {s.get('status')} exit {s.get('harness_exit_code')} | box {img.get('name')} {str(img.get('id'))[:19]} | sidecar {sc.get('image')} {str(sc.get('id'))[:19]} src {str(sc.get('src_sha256'))[:12]}")
    vo, vn = vals(old), vals(new)
    print("\n=== served values and printed train_time, per job log")
    for name in sorted(set(vo) | set(vn)):
        a, b = vo.get(name, []), vn.get(name, [])
        same_steps = [x[0] for x in a] == [x[0] for x in b]
        same_vals = same_steps and all(abs(x[1] - y[1]) < 1e-9 for x, y in zip(a, b))
        print(f"--- {name}: old {len(a)} validations, new {len(b)}; same steps: {same_steps}; same val_loss: {same_vals}")
        verdict.append(("same served values: " + name, same_vals))
        if same_steps and a:
            d = [y[2] - x[2] for x, y in zip(a, b)]
            print(f"    train_time new-old: step {a[0][0]}: {d[0]:+.3f} s; steps>0: min {min(d[1:]):+.3f} max {max(d[1:]):+.3f} s"
                  + (f"; expected {shift:+.3f} on every validation after step 0" if shift is not None else ""))
            if shift is not None:
                ok = abs(d[0]) < 1e-3 and all(abs(x - shift) < 1e-3 for x in d[1:])
                verdict.append(("train_time shifted by exactly the startup after step 0: " + name, ok))
            print(f"    final: old {a[-1][2]:.3f} s, new {b[-1][2]:.3f} s")
    po, pn = phases(old), phases(new)
    print("\n=== wall clock per phase, per job (old -> new)")
    for (jo, o), (jn, n) in zip(sorted(po.items()), sorted(pn.items())):
        print(f"--- old job {jo} / new job {jn}: …{o['argv']}")
        for k in ("start_to_first_enqueue", "first_enqueue_recorded_s", "paced_wall", "recorded_total", "gaps_beyond_paced", "last_cursor_to_bye", "enqueues"):
            print(f"    {k:>24}: {o[k]:>10.3f} -> {n[k]:>10.3f}   ({n[k] - o[k]:+.3f})")
        print("    paced units by recorded length (s): count, recorded s, wall s")
        for key in sorted(set(o["buckets"]) | set(n["buckets"]), reverse=True):
            a = o["buckets"].get(key, [0, 0.0, 0.0]); b = n["buckets"].get(key, [0, 0.0, 0.0])
            print(f"      {key:>10.3f}: {a[0]:>5} {a[1]:>10.3f} {a[2]:>8.3f}  ->  {b[0]:>5} {b[1]:>10.3f} {b[2]:>8.3f}")
        verdict.append((f"recorded total unchanged (job {jo}->{jn})", abs(o["recorded_total"] - n["recorded_total"]) < 1e-3))
        verdict.append((f"paced wall unchanged within 1% (job {jo}->{jn})", abs(o["paced_wall"] - n["paced_wall"]) <= 0.01 * max(o["paced_wall"], 1e-9)))
    print("\n=== verdict")
    for k, ok in verdict:
        print(f"  [{'ok' if ok else 'DIFF'}] {k}")
    print(f"COMPARE {'PASS' if all(ok for _, ok in verdict) else 'FAIL'}: {sum(ok for _, ok in verdict)}/{len(verdict)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
