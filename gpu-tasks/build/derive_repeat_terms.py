#!/usr/bin/env python3
# gpu-tasks, 2026-10-20: per-launch start-up terms across several scripted runs of one task, each launch labelled by the host
# memory the guest could not account for during its start-up (MemTotal - MemFree - Buffers - Cached - AnonPages - Slab -
# KernelStack - PageTables - Percpu - VmallocUsed, from resources.jsonl host_mem; the E2 memory balloon's signature), in two
# views: all launches, and launches whose start-up saw at most 10 GiB unaccounted (the runner's split). Runs whose
# samples lack MemFree leave their launches unlabelled: counted in "all", not in "quiet". Reuses build/derive_host_terms.py.
# A record of run-to-run spread; it does not set the deadline floor (build/gen_tasks.py reads survey/host-terms/<task_id>.json).
# Gaps from runs with overlapping jobs include the next job's interleaved units, not host time: take gaps from a serial run.
# usage: python3 -I build/derive_repeat_terms.py <gpu-tasks root> <lower bound s> <run dir>...   (on the ComputeWise VM)
import json, os, statistics, subprocess, sys
G, LB, runs = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
KEYS = ("Buffers_kb", "Cached_kb", "AnonPages_kb", "Slab_kb", "KernelStack_kb", "PageTables_kb", "Percpu_kb", "VmallocUsed_kb")
def unacc(hm):
    if "MemFree_kb" not in hm or "MemTotal_kb" not in hm:
        return None                      # older runs' samples lack MemFree: the launch stays unlabelled
    return (hm["MemTotal_kb"] - hm["MemFree_kb"] - sum(hm.get(k, 0) for k in KEYS)) / 1048576.0
launches = []
for R in runs:
    st = json.load(open(os.path.join(R, "status.json")))
    out = subprocess.run([sys.executable, "-I", os.path.join(G, "build", "derive_host_terms.py"), R], capture_output=True, text=True)
    if out.returncode:
        raise SystemExit(f"{R}: {out.stderr.strip()[-300:]}")
    d = json.loads(out.stdout)
    samples = []
    for l in open(os.path.join(R, "resources.jsonl")):
        r = json.loads(l)
        if r.get("kind") == "sample" and isinstance(r.get("host_mem"), dict):
            u = unacc(r["host_mem"])
            if u is not None:
                samples.append((r["t_wall"] - st["agent_start_wall"], u))
    ev = [json.loads(l) for l in open(os.path.join(R, "occupancy.jsonl")) if l.strip()]
    bye = max(e["t_wall"] for e in ev if e["kind"] == "job_bye") - st["agent_start_wall"]
    base = None
    p = os.path.join(R, "mem_baseline_pre.json")
    if os.path.exists(p):
        base = json.load(open(p)).get("unaccounted_gib", {}).get("median")
    for k, pj in enumerate(d["per_job"]):
        t0 = pj["launch_s"] + pj["hello_s"]
        t1 = t0 + pj["pre_s"]
        win = [u for t, u in samples if t0 <= t <= t1]
        launches.append({"run": os.path.basename(R)[-16:], "job": pj["job_id"], "pos": k + 1, "pre_s": pj["pre_s"], "hello_s": pj["hello_s"],
                         "post_s": pj["post_s"], "gap_s": pj["gap_s"], "gap_after_step": pj["gap_after_step"],
                         "max_unacc_gib": max(win) if win else None, "n_samples": len(win), "run_end_s": bye, "pre_baseline_gib": base})
print(f"{'run':17s} {'pos':>3s} {'job':14s} {'start-up s':>10s} {'max unaccounted GiB':>20s} {'samples':>7s} {'run end s':>9s} {'baseline GiB':>12s}")
for x in launches:
    print(f"{x['run']:17s} {x['pos']:3d} {x['job']:14s} {x['pre_s']:10.3f} {x['max_unacc_gib'] if x['max_unacc_gib'] is None else round(x['max_unacc_gib'], 1)!s:>20s} "
          f"{x['n_samples']:7d} {x['run_end_s']:9.1f} {x['pre_baseline_gib']!s:>12s}")
def view(xs, name):
    first = [x["pre_s"] for x in xs if x["pos"] == 1]
    rest = [x["pre_s"] for x in xs if x["pos"] > 1]
    gaps = [x["gap_s"] for x in xs if x["pos"] == 1 or True]
    print(f"== {name}: {len(xs)} launches ({len(first)} first, {len(rest)} later)")
    for lab, v in (("pre_s_first", first), ("pre_s_rest", rest), ("pre_s (all)", first + rest), ("hello_s", [x["hello_s"] for x in xs]),
                   ("post_s", [x["post_s"] for x in xs]), ("gap_s", [x["gap_s"] for x in xs])):
        if v:
            print(f"   {lab:12s} median {statistics.median(v):8.3f} | mean {statistics.fmean(v):8.3f} | min {min(v):8.3f} | max {max(v):8.3f}"
                  + (f" | sd {statistics.stdev(v):6.2f}" if len(v) > 1 else ""))
    return {"pre_s_first": statistics.median(first) if first else None, "pre_s_rest": statistics.median(rest) if rest else None,
            "hello_s": statistics.median(x["hello_s"] for x in xs), "post_s": statistics.median(x["post_s"] for x in xs),
            "gap": statistics.median(x["gap_s"] for x in xs), "n": len(xs)}
unk = [x for x in launches if x["max_unacc_gib"] is None]
if unk:
    print("launches with no MemFree sample in their start-up (unlabelled; in 'all', not in 'quiet'):", [(x["run"][-7:], x["job"]) for x in unk])
allv = view(launches, "all launches")
quiet = view([x for x in launches if x["max_unacc_gib"] is not None and x["max_unacc_gib"] <= 10.0], "launches whose start-up saw at most 10 GiB unaccounted")
hit = [x for x in launches if x["max_unacc_gib"] is not None and x["max_unacc_gib"] > 10.0]
print(f"== hit launches: {len(hit)}: " + ", ".join(f"{x['run'][-7:]} job{x['pos']} {x['pre_s']:.1f} s at {x['max_unacc_gib']:.1f} GiB" for x in hit))
ends = [x["run_end_s"] for x in launches if x["pos"] == 1]
print(f"== run makespans from agent start: {[round(e, 1) for e in ends]}; implied F = makespan - lower bound {LB}: {[round(e - LB, 1) for e in ends]}")
print("JSON " + json.dumps({"all": allv, "quiet": quiet}))
