"""The design table of an MLS-KV package: list schedules of its reference plans on the box's one-GPU lanes (gpu-tasks).

usage: python3 -I build/kv_design_table.py <registry> <task_id> <output json>

Written on 2026-10-22 for mls-kv-deploy (survey/mls-kv-deploy-expected-design-2026-10-22.json, accepted by
the analysis) and kept as a script on 2026-10-23 for mls-kv-report, so the two tables come from one rule.
Policies: serial; waves of four in list order; fixed per-GPU queues; greedy in list order; and the floor, critical-path list
scheduling. Order-dependent ones are given in the methodical order and over 100 random orders. Paced durations, no host terms.
"""
import json
import os
import random
import statistics
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
REG, TASK_ID, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
recs = C.load_records(os.path.join(REG, "records"))
task = json.load(open(os.path.join(REG, "tasks", TASK_ID + ".json")))
G = task["box"]["gpus"]
by = {j["job_id"]: j for j in task["jobs"]}
def plan_jobs(entries):
    """An `after` names "<job_id> screen": the job's --max-examples screen where the plan has one, else its full evaluation
    (mls-kv-gate's release waits on full ones). `after_all`, where present, lists every evaluation the release waits on; the
    job's `after` is then the set, and it is released when the last of them ends."""
    ids = {}
    for e in entries:
        ids.setdefault(e["job_id"], e["job_id"] + (":screen" if e.get("max_examples") else ""))
    def dep(x):
        return ids.get(x.split(" ")[0], x.split(" ")[0])
    out = []
    for e in entries:
        r = recs[by[e["job_id"]]["records"][0]]
        deps = [dep(x) for x in (e.get("after_all") or ([e["after"]] if e.get("after") else []))]
        jid = e["job_id"] + (":screen" if e.get("max_examples") else "")
        if jid in {j["id"] for j in out}:
            jid += ":full"
        out.append({"id": jid, "after": deps or None, "d": C.subset_duration(r, e.get("max_examples"), 1)})
    nxt = {}
    for j in out:
        for a in j["after"] or []:
            nxt[a] = max(nxt.get(a, 0.0), j["d"])
    for j in out:
        j["path"] = j["d"] + nxt.get(j["id"], 0.0)          # the job plus the longest job that waits on it: its remaining path
    return [j for j in out if not j["after"]] + [j for j in out if j["after"]]
def list_schedule(jobs, policy):
    """Non-preemptive list schedules on G one-GPU lanes; a confirm never starts before its own screen ends.
    serial: one lane, list order. waves: groups of G in list order, each after the last group and its jobs' screens.
    queues: lane i % G in list order. greedy: each freed lane takes the first released job in list order.
    floor: each freed lane takes the released job with the longest remaining path (the job plus any confirm waiting on it),
    critical-path list scheduling; it reaches the lower bound here."""
    iv = [[] for _ in range(G)]; end = {}
    rel = lambda j: max(end[a] for a in j["after"]) if j["after"] else 0.0
    done = lambda j, t: all(a in end and end[a] <= t for a in (j["after"] or []))
    if policy == "serial":
        t = 0.0
        for j in jobs:
            s = max(t, rel(j)); iv[0].append((s, s + j["d"])); t = end[j["id"]] = s + j["d"]
    elif policy == "waves":
        # groups of G in list order, each launched together and waited on with `wait`; a job whose dependency is in the
        # group being formed closes the group, as an agent that has to read a result before launching would
        t, i = 0.0, 0
        while i < len(jobs):
            w = []
            while i < len(jobs) and len(w) < G and all(a in end for a in (jobs[i]["after"] or [])):
                w.append(jobs[i]); i += 1
            s = max([t] + [rel(j) for j in w])
            for g, j in enumerate(w): iv[g].append((s, s + j["d"])); end[j["id"]] = s + j["d"]
            t = s + max(j["d"] for j in w)
    elif policy == "queues":
        lane = [0.0] * G
        for i, j in enumerate(jobs):
            g = i % G; s = max(lane[g], rel(j)); iv[g].append((s, s + j["d"])); lane[g] = end[j["id"]] = s + j["d"]
    else:
        free = [0.0] * G; pending = list(jobs)
        while pending:
            g = min(range(G), key=lambda k: free[k]); t = free[g]
            ready = [j for j in pending if done(j, t)]
            if not ready:
                # the earliest time a pending job's last dependency ends, among dependencies already scheduled; with none
                # scheduled yet, this lane waits for the next lane to free
                cand = [max(end[a] for a in j["after"]) for j in pending
                        if j["after"] and all(a in end for a in j["after"]) and max(end[a] for a in j["after"]) > t]
                free[g] = min(cand) if cand else min(f for k, f in enumerate(free) if k != g and f > t) if any(
                    f > t for k, f in enumerate(free) if k != g) else t + 1.0
                continue
            j = max(ready, key=lambda x: x["path"]) if policy == "floor" else ready[0]
            iv[g].append((t, t + j["d"])); free[g] = end[j["id"]] = t + j["d"]; pending.remove(j)
    mk = max(b for lane in iv for _, b in lane)
    busy = sum(b - a for lane in iv for a, b in lane)
    return mk, 1 - busy / (G * mk)
rng = random.Random(20261022)
pct = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p / 100 * len(xs)))]
out = {"schema": "p006-expected-design/0.1", "task_id": task["task_id"], "box_gpus": G, "plans": {},
       "lower_bound_h": round(task["stats"]["lower_bound_makespan_s"] / 3600, 2),
       "how": ("the package's design claim (the analysis, 2026-10-22): non-preemptive list schedules of the "
               "built task's two reference plans on 4 one-GPU lanes, at paced durations (common.subset_duration: a "
               "--max-examples k screen pays the model load, 6.0 s, once plus k of n examples). A confirm never starts before its "
               "own screen ends, and every order lists the screens before the confirms, since an agent queues a confirm only once "
               "its screen is read (mls-kv-report has no confirms: all 40 in full). The floor is critical-path list scheduling (the released job with the longest remaining path "
               "first) and reaches the task's lower bound. No host terms: the scripted measuring runs (W0-mlskv-check10, "
               "W0-mlskv-measure) produce registry/expected/<task>.<machine type>.host.json with them, which replaces this for scoring. "
               "Order-dependent policies: the methodical order (task, method, workload) and the median [p10, p90] over 100 random "
               "orders (seed 20261022; within each phase). Packing loss is GPU-idle share minus the floor's own")}
rp = task["reference_plan"]
plans = ([("clairvoyant", rp["clairvoyant"]["instances"]), ("adaptive", rp["instances"])] if "clairvoyant" in rp
         else [("all_full", rp["instances"])])
for name, entries in plans:
    jobs = plan_jobs(entries)
    fl, fl_idle = list_schedule(jobs, "floor")
    row = {"instances": len(jobs), "gpu_hours": round(sum(j["d"] for j in jobs) / 3600, 2), "floor_h": round(fl / 3600, 2),
           "floor_gpu_idle": round(fl_idle, 3), "policies": {}}
    first = [j for j in jobs if not j["after"]]; then = [j for j in jobs if j["after"]]
    # a random order keeps every job after the jobs it waits on: shuffle within the released set, and the dependants after
    for pol in ("serial", "waves", "queues", "greedy", "floor"):
        mk, idle = list_schedule(jobs, pol)
        r = {"methodical": {"makespan_h": round(mk / 3600, 2), "x_floor": round(mk / fl, 2), "gpu_idle": round(idle, 3),
                            "packing_loss": round(idle - fl_idle, 3)}}
        if pol in ("waves", "queues", "greedy"):
            def order():
                # waves and queues read `end` for a dependency in list order, so a dependant must come after everything it
                # waits on: shuffle the released jobs, then place each dependant after its last dependency
                o = rng.sample(first, len(first))
                for d_ in rng.sample(then, len(then)):
                    o.append(d_)
                pos = {j["id"]: i for i, j in enumerate(o)}
                return sorted(o, key=lambda j: max([pos[j["id"]]] + [pos[a] + 0.5 for a in (j["after"] or []) if a in pos]))
            runs = [list_schedule(order(), pol) for _ in range(100)]
            xs = [m / fl for m, _ in runs]; ids = [i for _, i in runs]
            r["random_orders"] = {"x_floor_median": round(statistics.median(xs), 2), "x_floor_p10": round(pct(xs, 10), 2),
                                  "x_floor_p90": round(pct(xs, 90), 2), "gpu_idle_median": round(statistics.median(ids), 3),
                                  "gpu_idle_p10": round(pct(ids, 10), 3), "gpu_idle_p90": round(pct(ids, 90), 3)}
        row["policies"][pol] = r
    out["plans"][name] = row
path = OUT
C.write_json(path, out)
print("task lower bound", out["lower_bound_h"], "h")
for name, row in out["plans"].items():
    print(f"\n{name}: {row['instances']} launches, {row['gpu_hours']} GPU-h, floor {row['floor_h']} h, floor idle {row['floor_gpu_idle']}")
    for pol, r in row["policies"].items():
        m = r["methodical"]; ro = r.get("random_orders")
        print(f"  {pol:7s} {m['makespan_h']:6.2f} h {m['x_floor']:4.2f}x  idle {m['gpu_idle']:.3f} loss {m['packing_loss']:+.3f}"
              + (f"  | random {ro['x_floor_median']}x [{ro['x_floor_p10']}, {ro['x_floor_p90']}] idle {ro['gpu_idle_median']} [{ro['gpu_idle_p10']}, {ro['gpu_idle_p90']}]" if ro else ""))
print(path)
