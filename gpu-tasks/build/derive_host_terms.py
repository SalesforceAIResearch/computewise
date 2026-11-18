#!/usr/bin/env python3
"""Host terms for build/expected_scores.py, derived from a finished scripted run directory.

usage: python3 -I build/derive_host_terms.py <run dir> --machine-type <gcp type> [--measured-on YYYY-MM-DD] [--source TEXT]

Prints the registry's host-terms file for the run's task (survey/host-terms/<task_id>.<machine type>.json): the per-script terms, the run's
speed, caps and image from its run.json, the run directory, and the per-job figures the terms come from. Written independently
of the runner's scripted/host_terms.py, so that the two derivations can be compared.

Per task job. Its ranks are the gpu-mock daemon jobs whose job_hello argv names the job's output directory (the job id itself,
or a name ending in "-<job id>"; the longest matching id wins). A launch line with no output directory (the modded-nanogpt
package tasks: each run writes logs/<uuid>.txt, so the task's shared_state.isolation is null) is matched instead by its
command: a rank whose argv (its sys.argv, which torchrun passes with the script path as launched) equals the job's launch line
from the script on, and the tool_use whose command contains that launch line. The script path is kept whole, so two groups
that share a script name in different directories (contra-soft-muon/ and soda-fade/ both hold contra-muon-to-soft-muon.py)
stay apart. Two jobs with one launch line would be ambiguous, and the script refuses them.
  launch   the scripted agent's tool_use whose command names that output directory or launch line, from the transcript (ms resolution)
  hello    launch to the earliest rank's process start (job_hello t_start)
  pre      the earliest rank's process start to the slowest rank's first paced unit. A multi-rank job's GPU work cannot run
           ahead of its last rank: the earlier ranks wait for it at their first collective
  gap      the job's GPU span beyond its paced time: the last rank's last unit end, minus the slowest rank's first unit start,
           minus the paced time (the longest rank's sum of unit wall costs). It is placed after the step of the largest single
           gap on any rank that begins once the slowest rank is ready (in the DPO smoke task, rank 0's checkpoint save after
           step 500)
  post     the last rank's last unit end to the last rank's job_bye
  per_event  (modded-nanogpt jobs, whose records carry a shard profile; since 2026-10-20) the same in-run GPU-idle time split by
           cause, so it transfers between runs of different lengths (a prefix and a full run of one script). The slowest rank's
           units are classified against the drawn record (the lookup event's record_id) as step, validation or one-off by their
           modelled seconds, and each gap between two units takes one class by common.gap_classes, the rule the simulator also
           uses: once (beside a one-off unit), val_after, val_before, shard (before the step at which the loader reads its next
           shard) or step. Per job, each class's mean gap per occurrence; per script, the median over its jobs.
Per script: hello_s, pre_s and post_s are medians over its jobs; pre_s_first is the median pre of its cold jobs (launched
before any job of that script had reached its first paced unit, the rule the simulators apply) and pre_s_rest the median over
its warm jobs, present when the script ran at least twice; gaps is [[step, median gap]]. The cold/warm split follows the per-script rule agreed with the runner on 2026-10-20.
All values are wall seconds. They equal recorded seconds only at speed 1, which build/gen_tasks.py requires of a run that sets
an environment floor or a horizon. Standard library only; run it where the run directory is (the ComputeWise VM).

per_example tasks (MLS-Bench KV, whose jobs carry n_examples_full; since 2026-10-23). A launch is `bash
methods/<method>/scripts/<workload>.sh` with OUTPUT_DIR in its environment, so neither tie above applies: a daemon job is tied
to its task job by its lookup's record_id (the job whose records list it), and to its launch by order among that job's
tool_uses that name its script. Per script: hello_s, pre_s (process start to the first example's GPU start), post_s (the last
example's end to job_bye), with pre_s_first / pre_s_rest by the same cold/warm rule; per script and workload,
example_gaps_by_workload, the host time before each example after the first, by index, median over the launches that reached
it (the same prompt's tokenising in every launch), and example_s_by_workload, its mean. example_gap_mode says how the box
combined that host time with the paced slot, which build/expected_scores.py needs to replay it: "overlap" when the enqueues
carry host_gap_s (gpu-mock 73dc on, 2026-10-23: each example is back-dated by its gap, so it takes max(slot, gap)), read from
that field; else "additive" (d1da and earlier: slot + gap), the gap being the enqueue time less the previous example's end.
"""
import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OD = re.compile(r"--output_dir[= ](\S+)")


def job_of(name, ids):
    base = os.path.basename(name.rstrip("/"))
    m = [j for j in ids if base == j or base.endswith("-" + j)]
    return max(m, key=len) if m else None


def output_dir(argv):
    for i, a in enumerate(argv or []):
        if a.startswith("--output_dir="):
            return a.split("=", 1)[1]
        if a == "--output_dir" and i + 1 < len(argv):
            return argv[i + 1]
    return None


def script_argv(tokens):
    """The argv from the script on: the first token ending in .py, as a normalised relative path, and everything after it. A
    launch line (torchrun ... dir/x.py --seed 2) and a rank's job_hello argv (dir/x.py --seed 2, or ./dir/x.py) give the same key."""
    toks = [str(t) for t in tokens]
    for i, t in enumerate(toks):
        if t.endswith(".py"):
            return tuple([os.path.normpath(t)] + toks[i + 1:])
    return tuple(toks)


def contains(tokens, key):
    """Whether the script argv `key` appears in a shell command's tokens, the script path normalised as in script_argv."""
    n = len(key)
    for i, t in enumerate(tokens):
        if t.endswith(".py") and os.path.normpath(t) == key[0] and tuple([key[0]] + tokens[i + 1:i + n]) == tuple(key):
            return True
    return False


def per_event(rec, u, ms, ready, speed):
    """In-run GPU-idle seconds per gap class (common.gap_classes) from one rank's units `u` ([(start, end)]) and their modelled
    seconds `ms`, for a record with a shard profile: each class's mean gap per occurrence, in wall seconds (host time does not
    scale with the speed knob). None when the
    units cannot be classified against the record."""
    import collections
    import common as C
    shards = set(C.shard_load_steps(rec))
    if not shards:
        return None
    vl = (rec.get("series") or {}).get("val_loss") or {}
    n_val = len(vl.get("steps") or [])
    vals = collections.Counter(round(x, 9) for x in ms)
    step_v = vals.most_common(1)[0][0]
    val_v = next((v for v, c in vals.most_common() if v != step_v and c == n_val), None)
    if val_v is None or vals[step_v] < rec["steps"] - 1:
        return None
    kinds = C.unit_kinds(ms, [round(x, 9) == val_v for x in ms])
    cls = C.gap_classes(kinds, shards)
    tot, cnt = collections.Counter(), collections.Counter()
    for i in range(1, len(u)):
        if u[i - 1][1] < ready - 1e-9:
            continue
        cnt[cls[i]] += 1
        tot[cls[i]] += max(0.0, u[i][0] - u[i - 1][1])
    out = {f"{c}_s": (tot[c] / cnt[c] if cnt[c] else None) for c in C.GAP_CLASSES}
    out["counts"] = {c: cnt[c] for c in C.GAP_CLASSES}
    return out


def per_example_terms(ev, task, off, status):
    """Host terms of a per_example run (see the module docstring): (terms per script, per-launch rows)."""
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup" and e.get("record_id")}
    job_of_rec = {rid: j["job_id"] for j in task["jobs"] for rid in (j.get("records") or [])}
    script_of = {j["job_id"]: os.path.dirname(j["script_path"]) + "/scripts/" + j["launch"].split("/")[-1] for j in task["jobs"]}
    cursor, units, has_gap = {}, collections.defaultdict(list), False
    for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
        hg = e.get("host_gap_s")
        has_gap |= hg is not None
        s = max(cursor.get(g, t), t - max(0.0, float(hg or 0.0)))     # where the daemon put it (gpu_daemon.py enqueue)
        cursor[g] = s + c
        units[e["job"]].append((s, s + c, t, hg))
    tools = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(status["_path"])), "transcript", "session", "*", "*.jsonl"))):
        for l in open(f):
            r = json.loads(l) if l.strip() else {}
            if r.get("type") != "assistant":
                continue
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    cmd = (b.get("input") or {}).get("command", "") or ""
                    for jid, sc in script_of.items():
                        if sc in cmd:
                            tools[jid].append(ts(r["timestamp"]))
    by_job = collections.defaultdict(list)
    for d, h in hello.items():
        jid = job_of_rec.get((lookup.get(d) or {}).get("record_id"))
        if jid and units.get(d) and d in bye:
            by_job[jid].append(d)
    rows = []
    for jid, ds in by_job.items():
        ds.sort(key=lambda d: hello[d]["t_start"])
        ls = sorted(tools.get(jid) or [])
        if len(ls) != len(ds):
            raise SystemExit(f"{jid}: {len(ds)} launches in the occupancy log, {len(ls)} tool_uses naming {script_of[jid]}")
        for d, L in zip(ds, ls):
            u, argv = units[d], hello[d].get("argv") or []
            wl = argv[argv.index("--workload") + 1].replace("_", "-") if "--workload" in argv else None
            gaps = ([max(0.0, float(x[3] or 0.0)) for x in u[1:]] if has_gap
                    else [max(0.0, u[i][2] - u[i - 1][1]) for i in range(1, len(u))])
            rows.append({"job_id": jid, "script": os.path.basename(str(argv[0])), "workload": wl, "examples": len(u),
                         "launch_s": L - status["agent_start_wall"], "hello_s": hello[d]["t_start"] - L,
                         "pre_s": u[0][0] - hello[d]["t_start"], "post_s": bye[d]["t_wall"] - u[-1][1],
                         "host_gap_sum_s": sum(gaps), "_gaps": gaps})
    terms = {}
    for sc in sorted({r["script"] for r in rows}):
        ps = [r for r in rows if r["script"] == sc]
        med = lambda k: statistics.median(r[k] for r in ps)
        t = {"hello_s": round(med("hello_s"), 3), "pre_s": round(med("pre_s"), 3), "post_s": round(med("post_s"), 3), "gaps": [],
             "example_gap_mode": "overlap" if has_gap else "additive"}
        first_ready = min(r["launch_s"] + r["hello_s"] + r["pre_s"] for r in ps)
        cold = [r for r in ps if r["launch_s"] < first_ready - 1e-6]
        warm = [r for r in ps if r["launch_s"] >= first_ready - 1e-6]
        if len(ps) >= 2:
            t["pre_s_first"] = round(statistics.median(r["pre_s"] for r in cold), 3)
            if warm:
                t["pre_s_rest"] = round(statistics.median(r["pre_s"] for r in warm), 3)
            t["cold_jobs"], t["warm_jobs"] = len(cold), len(warm)
        vec, mean = {}, {}
        for wl in sorted({r["workload"] for r in ps}):
            gs = [r["_gaps"] for r in ps if r["workload"] == wl]
            m = max(len(g) for g in gs)
            v = [round(statistics.median(g[i] for g in gs if len(g) > i), 4) for i in range(m)]
            vec[wl], mean[wl] = v, round(sum(v) / len(v), 4) if v else 0.0
        t["example_gaps_by_workload"], t["example_s_by_workload"] = vec, mean
        terms[sc] = t
    for r in rows:
        r.pop("_gaps")
    return terms, rows


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--machine-type",
                    help="the GCP machine type the run was on (e.g. c3-highmem-44). Default: run.json host.machine_type, which "
                         "run_one.py records since 2026-10-20 16:14 UTC. Host terms are per machine type, caps and image, and the "
                         "file is written as survey/host-terms/<task_id>.<machine type>.json")
    ap.add_argument("--registry", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"),
                    help="the registry the run drew from, for the records named in its lookup events (default: this tree's)")
    ap.add_argument("--measured-on")
    ap.add_argument("--source")
    a = ap.parse_args()
    D = a.run_dir
    run = json.load(open(os.path.join(D, "run.json")))
    status = json.load(open(os.path.join(D, "status.json")))
    task = json.load(open(os.path.join(D, "task.json")))
    # A scripted measure runs each decision slot as its recorded winner's launch (the runner's plan.py), and a slot's
    # own launch is a template, so tie its ranks by the winner's launch.
    for j in task["jobs"]:
        w = j.get("recorded_winner") or {}
        if j.get("bind") and w.get("launch"):
            j["launch"], j["records"] = w["launch"], w.get("records") or j.get("records")
    if status.get("status") != "completed":
        raise SystemExit(f"{D}: status {status.get('status')!r}, not completed")
    ids = [j["job_id"] for j in task["jobs"]]
    ev = [json.loads(l) for l in open(os.path.join(D, "occupancy.jsonl")) if l.strip()]
    s0 = next(e for e in ev if e["kind"] == "daemon_start")
    speed = float(run.get("inputs", {}).get("speed") or s0.get("speed") or 1.0)
    off = s0["t_wall"] - s0["t_mono"]
    per_example = any(j.get("n_examples_full") for j in task["jobs"])
    if per_example:
        terms, per_job = per_example_terms(ev, task, off, dict(status, _path=os.path.join(D, "status.json")))
    else:
        hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
        bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
        lookup = {e["job"]: e for e in ev if e["kind"] == "lookup" and e.get("record_id")}
        cursor, units, modelled = {}, collections.defaultdict(list), collections.defaultdict(list)
        for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
            g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
            s = max(cursor.get(g, t), t)
            cursor[g] = s + c
            units[e["job"]].append((s, s + c))
            modelled[e["job"]].append(float(e.get("modelled_s", 0.0)))
        tr = sorted(glob.glob(os.path.join(D, "transcript", "session", "*", "*.jsonl")))
        recs = [json.loads(l) for f in tr for l in open(f) if l.strip()]
        by_cmd = {}                                   # script argv (past the launcher) -> job ids, for jobs with no output dir
        for j in task["jobs"]:
            if OD.search(j.get("launch") or ""):
                continue
            key = script_argv(j["launch"].split())
            by_cmd.setdefault(key, []).append(j["job_id"])
        # Jobs that share a launch line (an unseeded script run k times: H1's muonh-tuned, G4n's newton-muon) are told apart by
        # order: the k-th launch of the line and the k-th process start of its argv belong to the k-th such job. Terms are per
        # script, so which of the identical jobs a rank is credited to does not change them. Only single-rank jobs: with several
        # ranks a launch's ranks could not be grouped by argv alone
        for key, js in by_cmd.items():
            if len(js) > 1 and sum(1 for h in hello.values() if script_argv(h.get("argv") or []) == key) != len(js):
                raise SystemExit(f"{', '.join(js)} share a launch line and do not have one rank each, so their ranks cannot be "
                                 "told apart")
        launch = {}
        for r in recs:
            if r.get("type") != "assistant":
                continue
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    cmd = (b.get("input") or {}).get("command", "") or ""
                    for m in OD.finditer(cmd):
                        j = job_of(m.group(1), ids)
                        if j:
                            launch.setdefault(j, ts(r["timestamp"]))
                    for key, js in by_cmd.items():
                        if contains(cmd.split(), list(key)):
                            j = next((x for x in js if x not in launch), None)
                            if j:
                                launch[j] = ts(r["timestamp"])
        ranks = collections.defaultdict(list)
        nth = collections.Counter()
        for d, h in sorted(hello.items(), key=lambda kv: kv[1].get("t_start") or 0.0):
            od = output_dir(h.get("argv"))
            if od:
                j = job_of(od, ids)
            else:
                key = script_argv(h.get("argv") or [])
                js = sorted(by_cmd.get(key) or [], key=lambda x: launch.get(x, float("inf")))
                j = js[nth[key]] if nth[key] < len(js) else None
                nth[key] += 1
            if j:
                ranks[j].append(d)
        per_job = []
        for j in sorted(launch, key=launch.get):
            rk = ranks.get(j) or []
            if not rk or any(d not in bye or not units.get(d) for d in rk):
                raise SystemExit(f"{j}: a rank has no paced units or no job_bye")
            L = launch[j]
            start = min(hello[d]["t_start"] for d in rk)
            ready = max(units[d][0][0] for d in rk)
            end = max(units[d][-1][1] for d in rk)
            paced = max(sum(e - s for s, e in units[d]) for d in rk)
            big = None
            for d in rk:
                u = units[d]
                for k in range(1, len(u)):
                    g = u[k][0] - u[k - 1][1]
                    if u[k - 1][1] >= ready - 1e-9 and g > 1e-6 and (big is None or g > big[1]):
                        big = (k, g)
            argv = hello[rk[0]].get("argv") or []
            script = os.path.normpath(str(next((x for x in argv if str(x).endswith(".py")), argv[0] if argv else "?")))
            row = {"job_id": j, "script": script, "ranks": len(rk), "launch_s": round(L - status["agent_start_wall"], 3),
                   "hello_s": start - L, "pre_s": ready - start, "gap_s": max(0.0, end - ready - paced),
                   "gap_after_step": big[0] if big else None, "post_s": max(bye[d]["t_wall"] for d in rk) - end,
                   "paced_wall_s": paced}
            slow = max(rk, key=lambda d: units[d][-1][1])
            rid = (lookup.get(slow) or {}).get("record_id")
            if rid:
                rec = json.load(open(os.path.join(a.registry, "records", rid + ".json")))
                pe = per_event(rec, units[slow], modelled[slow], ready, speed)
                if pe:
                    row["record_id"], row["per_event"] = rid, pe
            per_job.append(row)
        # Terms are keyed by script basename, as build/expected_scores.py and the runner's planner look them up, unless two different
        # scripts of the task share one (contra-soft-muon/ and soda-fade/ both hold contra-muon-to-soft-muon.py): those are keyed by
        # their path in the task tree, which expected_scores.py tries first.
        paths = collections.defaultdict(set)
        for p in per_job:
            paths[os.path.basename(p["script"])].add(p["script"])
        for p in per_job:
            if len(paths[os.path.basename(p["script"])]) == 1:
                p["script"] = os.path.basename(p["script"])
        terms = {}
        for sc in sorted({p["script"] for p in per_job}):
            ps = [p for p in per_job if p["script"] == sc]
            med = lambda k, xs=ps: statistics.median(p[k] for p in xs)
            steps = [p["gap_after_step"] for p in ps if p["gap_after_step"] is not None]
            t = {"hello_s": round(med("hello_s"), 3), "pre_s": round(med("pre_s"), 3), "post_s": round(med("post_s"), 3),
                 "gaps": [[int(statistics.median_low(steps)), round(med("gap_s"), 3)]] if steps and med("gap_s") > 0 else []}
            if len(ps) >= 2:
                # Cold or warm by the rule both simulators apply (expected_scores.Sim.startup, the runner's plan.py and expect.py):
                # a job is cold when no job of its script had reached its first paced unit by its launch. Several jobs of one script
                # launched together are all cold, as packed launches them; until 2026-10-20 only the earliest-launched counted.
                first_ready = min(p["launch_s"] + p["hello_s"] + p["pre_s"] for p in ps)
                cold = [p for p in ps if p["launch_s"] < first_ready - 1e-6]
                warm = [p for p in ps if p["launch_s"] >= first_ready - 1e-6]
                t["pre_s_first"] = round(statistics.median(p["pre_s"] for p in cold), 3)
                if warm:
                    t["pre_s_rest"] = round(statistics.median(p["pre_s"] for p in warm), 3)
                t["cold_jobs"], t["warm_jobs"] = len(cold), len(warm)
            pes = [p["per_event"] for p in ps if p.get("per_event")]
            if pes:
                t["per_event"] = {f"{c}_s": round(statistics.median(x[f"{c}_s"] for x in pes if x[f"{c}_s"] is not None), 4)
                                  if any(x[f"{c}_s"] is not None for x in pes) else None
                                  for c in ("once", "val_after", "val_before", "shard", "step")}
                t["per_event"]["jobs"] = len(pes)
            terms[sc] = t
    im = run.get("image") or {}
    mf = im.get("manifest") or {}
    host = run.get("host") or {}
    machine_type = a.machine_type or host.get("machine_type")
    if not machine_type:
        raise SystemExit(f"{D}: run.json records no host.machine_type; pass --machine-type")
    out = {"schema": "p006-host-terms/0.2", "task_id": task["task_id"], "machine_type": machine_type,
           **{k: host[k] for k in ("cpu_platform", "zone", "nproc") if k in host}, "terms": terms, "speed": speed,
           "caps": run.get("caps"), "image": {"name": im.get("name"), "id": im.get("id"),
           "sidecar_image_id": mf.get("sidecar_image_id"), "torch_patch_sha256": mf.get("torch_patch_sha256"),
           "gpu_mock_bundle_manifest_sha256": mf.get("gpu_mock_bundle_manifest_sha256")},
           "run_dir": os.path.abspath(D), "run_task_sha256": run.get("task_sha256"),
           "measured_on": a.measured_on or status.get("start_utc", "")[:10],
           "source": a.source or (f"scripted {run.get('scripted', {}).get('policy')} run {run.get('cost_tag')}, speed {speed:g}, caps "
                                  f"cpus {(run.get('caps') or {}).get('cpus')} cpuset {(run.get('caps') or {}).get('cpuset')} memory "
                                  f"{(run.get('caps') or {}).get('memory')}; status {status.get('status')}, {status.get('teardown_reason')}"),
           "derived_by": "build/derive_host_terms.py (gpu-tasks)",
           "per_job": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in p.items()} for p in per_job]}
    json.dump(out, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
