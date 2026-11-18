"""C1's tolerance from repeated packed_overlap runs of one task on one machine type.

usage: python3 -I c1_tolerance.py <host_terms.json> <expected.host.json> <run_dir> [<run_dir> ...] [--json]

Per run, from its own occupancy log: each task job's start-up (launch to its slowest rank's first paced unit, the
pre_s convention) against the host terms' prediction, cold (the first launch of its script, pre_s_first) or warm
(pre_s_rest), plus hello_s; the window and the whole-box idle share against the expected file's packed_overlap; and
the most memory owned by nothing during the run (the balloon check, MemTotal minus every meminfo category).

The tolerance, as stated before the repeats (C1 note, 2026-10-20): per launch, 0.25 s of jitter plus the largest
start-up deviation seen, cold and warm separately; on the window, the sum over the plan's launches. Because a
tolerance taken from the runs trivially covers those same runs, the pass count is also given leave-one-out: the
tolerance from the other runs, applied to each run in turn.
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "score"))
import idle, score_run

def unaccounted_max(d):
    best = None
    for l in open(os.path.join(d, "resources.jsonl")):
        try:
            r = json.loads(l)
        except json.JSONDecodeError:
            continue
        h = r.get("host_mem") or {}
        if "MemFree_kb" not in h:
            continue
        k = lambda f: h.get(f + "_kb", 0)
        u = (k("MemTotal") - sum(k(f) for f in ("MemFree", "Buffers", "Cached", "AnonPages", "Slab", "KernelStack",
                                                 "PageTables", "Percpu", "VmallocUsed"))) / 2 ** 20
        best = u if best is None else max(best, u)
    return best

def per_run(d, terms):
    ev = score_run.load_events(os.path.join(d, "occupancy.jsonl"))
    task = json.load(open(os.path.join(d, "task.json")))
    st = json.load(open(os.path.join(d, "status.json")))
    a0 = st["agent_start_wall"]
    s0 = next(e for e in ev if e["kind"] == "daemon_start"); off = s0["t_wall"] - s0["t_mono"]
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    first, cur = {}, {}
    for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
        s = max(cur.get(g, t), t); cur[g] = s + c
        first.setdefault(e["job"], s)
    groups = {}
    for dj, h in hello.items():
        groups.setdefault(idle.task_job_of(h, lk.get(dj), task["jobs"]), []).append(dj)
    t = terms["dpo.py"] if "dpo.py" in terms else next(iter(terms.values()))
    launches = []
    for k, (tj, djs) in enumerate(sorted(groups.items(), key=lambda kv: min(idle.job_start(hello[d]) for d in kv[1]))):
        start = min(idle.job_start(hello[d]) for d in djs)
        pre = max(first[d] for d in djs) - start
        cold = k == 0
        want = t["pre_s_first"] if cold else t["pre_s_rest"]
        launches.append({"task_job": tj, "cold": cold, "pre_s": pre, "predicted_s": want, "deviation_s": pre - want,
                         "process_start_s": start - a0})
    r = idle.analyse(d)
    return {"run": os.path.basename(os.path.abspath(d)), "window_s": r["window"]["wall_s"],
            "whole_box_share": r["whole_box_idle_share_of_wall"], "launches": launches,
            "unaccounted_max_gib": unaccounted_max(d),
            "run_one": (json.load(open(os.path.join(d, "run.json"))).get("code") or {}).get("run_one_sha256", "")[:16]}

def tolerance(runs):
    cold = [l["deviation_s"] for r in runs for l in r["launches"] if l["cold"]]
    warm = [l["deviation_s"] for r in runs for l in r["launches"] if not l["cold"]]
    tc = 0.25 + max(abs(x) for x in cold) if cold else None
    tw = 0.25 + max(abs(x) for x in warm) if warm else None
    return tc, tw

def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    terms = json.load(open(args[0]))["terms"] if "terms" in json.load(open(args[0])) else json.load(open(args[0]))
    exp = json.load(open(args[1]))["policies"]["packed_overlap"]["idle"]
    runs = [per_run(d, terms) for d in args[2:]]
    tc, tw = tolerance(runs)
    n_launch = len(runs[0]["launches"]) if runs else 0
    win_tol = tc + tw * (n_launch - 1) if tc is not None and tw is not None else None
    ew, es = exp["window"]["wall_s"], exp["whole_box_idle_share_of_wall"]
    def passes(r, tc_, tw_):
        wt = tc_ + tw_ * (len(r["launches"]) - 1)
        return abs(r["window_s"] - ew) <= wt and abs(r["whole_box_share"] - es) <= wt / ew
    loo = []
    for i, r in enumerate(runs):
        others = runs[:i] + runs[i + 1:]
        a, b = tolerance(others)
        loo.append(passes(r, a, b))
    res = {"expected": {"window_s": ew, "whole_box_share": es}, "per_launch_tolerance_s": {"cold": tc, "warm": tw},
           "window_tolerance_s": win_tol, "runs": runs, "pass_all_from_all": sum(passes(r, tc, tw) for r in runs),
           "pass_leave_one_out": sum(loo), "n": len(runs)}
    if "--json" in sys.argv:
        print(json.dumps(res, indent=1)); return
    print(f"expected window {ew:.1f} s, whole-box idle {es:.4f}")
    for r in runs:
        devs = " ".join(f"{'C' if l['cold'] else 'W'}{l['deviation_s']:+.1f}" for l in r["launches"])
        print(f"  {r['run'][-26:]:26s} window {r['window_s']:8.1f} ({r['window_s'] - ew:+6.1f})  whole {r['whole_box_share']:.4f}  "
              f"start-up dev {devs}  unaccounted max {r['unaccounted_max_gib']:.1f} GiB  run_one {r['run_one']}")
    print(f"per-launch tolerance: cold {tc:.2f} s, warm {tw:.2f} s; window {win_tol:.2f} s over {n_launch} launches")
    print(f"pass: {res['pass_all_from_all']} of {len(runs)} against the tolerance from all runs (trivially); "
          f"{res['pass_leave_one_out']} of {len(runs)} leave-one-out")

if __name__ == "__main__":
    main()
