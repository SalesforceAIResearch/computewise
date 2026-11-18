"""The box scheduler's event log, read for the scorer (the scheduler's spec, experiments/scheduler/AGENTS.md
"The event log"; runner, 2026-10-21).

usage: python3 -I sched_events.py <run_dir> [--json]

The log is one JSON object per transition, with `t` (epoch seconds) and `ev`: submit, pending_reason, eligible, start
(gpus, pid), end (state, exit_code), and the controller's own events. Found at <run>/scheduler.jsonl (S rungs: isolation
copies it there), else the workspace's logs/sched-events.jsonl (the in-box probe), else any sched-events.jsonl under the
workspace. None when there is none.

Per scheduler job: submit, eligible, start and end times from the agent's start, its state and exit code, the GPUs it was
given, its time pending by reason, and the GPU ranks (gpu-mock's daemon jobs) it ran. A rank is tied to a scheduler job
through the box's process records (resources.jsonl): the job's batch step is `slurm_script` under the spool's
job<NNNNN>/, and a rank whose argv ends a descendant's command line, said hello while the job ran, is that job's. Without
process records, a rank ties by time and GPU: the job that was running on the rank's GPUs when it said hello.
Checked for each tied rank: its GPUs (the daemon's enqueue GPUs) are among the job's start.gpus, which is
CUDA_VISIBLE_DEVICES honoured; a rank on another GPU is gpu_mask_mismatch, and score_all sends the run to review.

A job asking for no GPU (submit.gpus_req 0: data prep, eval, a barrier) is a job (gpu_job false), with its pending and
run times, kept out of the latency, the scheduler idle and the GPU-job counts. A controller_error (the scheduler's own fault) is counted, with the first
message; score_all sends the run to review on any. A rank
alive more than 5 s after its job's end ran outside the job (the step's leftovers are killed when its script ends):
ranks_outlived_job. A GPU job pending on Resources while enough GPUs were free for longer than GRACE_S (5 s: the controller's
1-s tick plus a pass) waited on CPUs or memory, not on the scheduler: resources_wait_with_gpus_free, cause cpu/mem, its
GPU-seconds out of the scheduler idle, and a start right after it out of the latency figures. A shorter free stretch is
the scheduler's own latency.

The scheduler's own latency (the spec's rule): for a GPU job's start that followed a wait on Resources, the gap from the
latest end at or before it (the event that freed the GPUs) to the start. Also the queue wait (eligible to start) per job, and
the GPU-seconds a GPU sat unallocated while an eligible job that fit was pending: the scheduler's share of idle, which
should be about its latency times the jobs it delayed.
"""
import argparse, glob, json, os, statistics, sys
from datetime import datetime

GRACE_S = 5.0      # longer than the controller's 1-s tick plus a pass: a free GPU unfilled past this is not its latency

def _ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def find_log(run_dir):
    for p in (os.path.join(run_dir, "scheduler.jsonl"),
              os.path.join(run_dir, "workspace", "home", "project", "logs", "sched-events.jsonl")):
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(run_dir, "workspace", "**", "sched-events.jsonl"), recursive=True))
    return hits[0] if hits else None

def read_jsonl(p):
    out = []
    for line in open(p, errors="replace"):
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out

def proc_tree(run_dir):
    """Host processes from resources.jsonl: pid -> {ppid, cmdline, first}. Empty when the run has no process records."""
    p = os.path.join(run_dir, "resources.jsonl")
    procs = {}
    for l in (read_jsonl(p) if os.path.exists(p) else []):
        if l.get("kind") != "procs":
            continue
        try:
            t = _ts(l["t_utc"])
        except (KeyError, ValueError):
            continue
        for q in (l.get("procs") if l.get("full") else l.get("started")) or []:
            if isinstance(q, dict) and q.get("pid") is not None and int(q["pid"]) not in procs:
                procs[int(q["pid"])] = {"ppid": q.get("ppid"), "cmdline": q.get("cmdline") or "", "first": t}
            # a process first sampled between fork and exec has an empty command line; keep its first non-empty one
            # (G-k-haiku-S1-r4, 2026-11-06: job 459769's slurm_script read '' at 15:27:17Z and its script 22 s later, so
            # its normuonh rank was tied by command line to job 459768 on another GPU and read as a gpu_mask_mismatch)
            elif isinstance(q, dict) and q.get("pid") is not None and not procs[int(q["pid"])]["cmdline"] and q.get("cmdline"):
                procs[int(q["pid"])]["cmdline"] = q["cmdline"]
    return procs

def descendants(procs, root):
    kids = {}
    for pid, q in procs.items():
        kids.setdefault(q["ppid"], []).append(pid)
    out, todo = [], [root]
    while todo:
        x = todo.pop()
        for k in kids.get(x, []):
            out.append(k); todo.append(k)
    return out

def analyse(run_dir):
    path = find_log(run_dir)
    if not path:
        return None
    ev = sorted(read_jsonl(path), key=lambda e: e.get("t") or 0)
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    a0 = st.get("agent_start_wall") or (ev[0]["t"] if ev else 0)
    rel = lambda t: None if t is None else t - a0
    ctl = next((e for e in ev if e.get("ev") == "controller_start"), {})
    # the controller's own faults (today: a signal to a job that failed): any one sends the run to review (fake-slurm,
    # 2026-10-21), since a schedule the scheduler got wrong is not the agent's
    cerr = [e for e in ev if e.get("ev") == "controller_error"]
    jobs = {}
    for e in ev:
        k = e.get("ev")
        if e.get("job") is None or k == "controller_start":
            continue
        j = jobs.setdefault(e["job"], {"job": e["job"], "display": e.get("display"), "name": e.get("name"), "pending": [],
                                       "submit": None, "eligible": None, "start": None, "end": None, "gpus": None})
        if k == "submit":
            j.update(submit=e["t"], gpus_req=e.get("gpus_req"), dependency=e.get("dependency"), kind=e.get("kind"),
                     cpus=e.get("cpus"), mem_mb=e.get("mem_mb"))
        elif k == "pending_reason":
            j["pending"].append((e["t"], e.get("reason")))
        elif k == "eligible":
            j["eligible"] = e["t"]
        elif k == "start":
            j.update(start=e["t"], gpus=e.get("gpus") or [], pid=e.get("pid"), launch_error=e.get("launch_error"))
        elif k == "end":
            j.update(end=e["t"], state=e.get("state"), exit_code=e.get("exit_code"), end_reason=e.get("reason"),
                     gpus=j.get("gpus") if j.get("gpus") is not None else e.get("gpus"))
    last = max((e.get("t") or 0 for e in ev), default=a0)
    # the daemon's ranks: hello (argv, start), their GPUs from the enqueues
    occ = os.path.join(run_dir, "occupancy.jsonl")
    hello, rgpus, rbye, rkey, rcost = {}, {}, {}, {}, {}
    for e in (read_jsonl(occ) if os.path.exists(occ) else []):
        if e.get("kind") == "job_hello":
            hello[e["job"]] = e
        elif e.get("kind") == "lookup":
            rkey.setdefault(e["job"], e.get("job_key"))
        elif e.get("kind") == "enqueue":
            rgpus.setdefault(e["job"], set()).add(e["gpu"])
            rcost[e["job"]] = rcost.get(e["job"], 0.0) + float(e.get("wall_cost") or 0.0)
        elif e.get("kind") == "job_bye":
            rbye[e["job"]] = e["t_wall"]
    for j in jobs.values():
        # a job that asks for no GPU (data prep, eval, a barrier) is a job, kept out of the GPU sums
        j["gpu_job"] = int(j.get("gpus_req") or 0) > 0
    procs = proc_tree(run_dir)
    step_of = {}                      # scheduler job -> its batch step's host pid
    for pid, q in procs.items():
        for jid in jobs:
            # a job's batch step is `slurm_script` under its spool dir, or fake-slurm's `slurmstepd <spool>/jobN/step.json` when
            # the process record caught the step before it exec'd the script (G-mlskv-haiku-S1, 2026-10-24 03:47Z: job 863425's
            # step was recorded only as slurmstepd, so its rank was tied by command line to an earlier job running the same
            # script on another GPU, and read as a mask mismatch)
            if f"/job{int(jid):05d}/" in q["cmdline"] and ("slurm_script" in q["cmdline"] or
                                                            ("slurmstepd" in q["cmdline"] and "step.json" in q["cmdline"])):
                step_of.setdefault(jid, pid)
    cmds_of = {jid: [procs[d]["cmdline"] for d in descendants(procs, pid)] for jid, pid in step_of.items()}
    tied, how = {}, {}
    for r, h in hello.items():
        t0 = h.get("t_start") or h.get("t_wall")
        argv = " ".join(str(x) for x in h.get("argv") or [])
        live = [j for j in jobs.values() if j["start"] is not None and j["start"] - 2 <= t0 <= (j["end"] or last) + 2]
        by_tree = [j for j in live if argv and any(c.endswith(argv) for c in cmds_of.get(j["job"], []))]
        broke = False
        if len(by_tree) > 1 and rgpus.get(r):
            # several live steps run a byte-identical command line (fake-slurm, 2026-10-23 00:16Z: three MLS-KV selection
            # methods run custom_selection_eval.py with the same flags, only the directory differing, and gpu-mock's argv[0]
            # is the bare script name): the tree alone cannot tell them apart, so the step whose GPUs hold the rank's is
            # preferred before the latest start. A tie-break among tree matches only, so it cannot create a mismatch
            on_gpu = [j for j in by_tree if set(rgpus[r]) <= set(j["gpus"] or [])]
            if on_gpu:
                by_tree, broke = on_gpu, True
        if by_tree:
            j = max(by_tree, key=lambda j: j["start"])
            how[r] = "process tree, GPU tie-break" if broke else "process tree"
        else:
            by_gpu = [j for j in live if rgpus.get(r) and set(rgpus[r]) <= set(j["gpus"] or [])]
            if not by_gpu:
                continue
            j = max(by_gpu, key=lambda j: j["start"]); how[r] = "time and GPU"
        tied[r] = j["job"]
    pts_all = sorted({t for x in jobs.values() for t in (x["eligible"], x["start"], x["end"]) if t is not None})
    out_jobs, mism, lat, outlived, cpu_wait = [], [], [], [], []
    ends = sorted((j["end"], j["job"]) for j in jobs.values() if j["end"] is not None and j.get("gpus"))
    gpus_n = int(ctl.get("gpus") or 0)
    def gpus_used(m):
        return sum(len(x["gpus"] or []) for x in jobs.values() if x["start"] is not None and x["start"] <= m < (x["end"] or last))
    for j in sorted(jobs.values(), key=lambda j: (j["submit"] or 0, str(j["job"]))):
        ranks = sorted(r for r, x in tied.items() if x == j["job"])
        off = [r for r in ranks if not set(rgpus.get(r) or []) <= set(j["gpus"] or [])]
        for r in off:
            mism.append({"sched_job": j["display"], "rank": r, "rank_gpus": sorted(rgpus.get(r) or []), "start_gpus": j["gpus"]})
        # the step's leftovers are killed when its script ends, so a rank alive well past its job's end ran outside it
        # (nohup with a new session, say)
        for r in ranks:
            if j["end"] is not None and rbye.get(r, last) > j["end"] + 5:
                outlived.append({"sched_job": j["display"], "rank": r, "past_end_s": rbye.get(r, last) - j["end"]})
        pend = {}
        reasons = j["pending"] + [(j["start"] if j["start"] is not None else (j["end"] or last), None)]
        for (t, why), (t1, _) in zip(reasons, reasons[1:]):
            if why:
                pend[why] = pend.get(why, 0.0) + max(0.0, t1 - t)
        lat_s = None
        if j["gpu_job"] and j["start"] is not None and any(w == "Resources" for _, w in j["pending"]):
            freed = [t for t, x in ends if t <= j["start"] + 1e-6 and x != j["job"]]
            if freed:
                lat_s = j["start"] - max(freed)
                lat.append({"sched_job": j["display"], "latency_s": lat_s})
        # a GPU job pending on Resources while enough GPUs were free waited on CPUs or memory (a 0-GPU job holding them,
        # say), not on the scheduler: its seconds so labelled
        # a free stretch up to GRACE_S is the scheduler's own pass (its latency); a longer one is cpu/mem
        cw, cw_before_start = 0.0, False
        if j["gpu_job"]:
            for (t, why), (t1, _) in zip(reasons, reasons[1:]):
                if why != "Resources":
                    continue
                cuts = sorted({t, t1, *[x for x in pts_all if t < x < t1]})
                runs, cur_ = [], None
                for a_, b_ in zip(cuts, cuts[1:]):
                    if gpus_n - gpus_used((a_ + b_) / 2) >= int(j["gpus_req"] or 0):
                        cur_ = [cur_[0], b_] if cur_ and abs(cur_[1] - a_) < 1e-9 else [a_, b_]
                        if not runs or runs[-1] is not cur_:
                            runs.append(cur_)
                    else:
                        cur_ = None
                for a_, b_ in runs:
                    if b_ - a_ > GRACE_S:
                        cw += b_ - a_
                        if j["start"] is not None and abs(b_ - j["start"]) < 1e-6:
                            cw_before_start = True
            if cw:
                cpu_wait.append({"sched_job": j["display"], "waited_s": cw, "cause": "cpu/mem"})
        if cw_before_start and lat and lat[-1]["sched_job"] == j["display"]:
            lat[-1]["cause"] = "cpu/mem"           # not the scheduler's latency: kept out of its median and max
            lat_s = None
        # runs per job: the task-job launches the job ran, a launch
        # being its ranks grouped by gpu-mock's job key (one key per launch; a multi-GPU launch has a rank per GPU). More
        # than one launch in sequence (one starting after another ended) makes the job a lane: the fixed per-GPU list
        # rebuilt inside the scheduler, which cannot repack it
        launches_ = {}
        for r in ranks:
            k = rkey.get(r) or f"rank-{r}"
            h = hello.get(r) or {}
            a_, b_ = h.get("t_start") or h.get("t_wall"), rbye.get(r)
            x = launches_.setdefault(k, {"start": a_, "end": b_, "gpu_s": 0.0})
            x["start"] = min(v for v in (x["start"], a_) if v is not None) if (x["start"] or a_) else None
            x["end"] = max(v for v in (x["end"], b_) if v is not None) if (x["end"] or b_) else None
            x["gpu_s"] += rcost.get(r, 0.0)
        seq = sorted((x for x in launches_.values() if x["start"] is not None), key=lambda x: x["start"])
        in_seq = sum(1 for p_, q_ in zip(seq, seq[1:]) if p_["end"] is not None and q_["start"] >= p_["end"] - 1.0)
        runs_per_job = {"runs": len(launches_), "in_sequence": in_seq, "lane": len(launches_) > 1 and in_seq > 0,
                        "task_gpu_s": sum(x["gpu_s"] for x in launches_.values())}
        out_jobs.append({"display": j["display"], "name": j["name"], "gpu_job": j["gpu_job"], "runs_per_job": runs_per_job, "kind": j.get("kind"), "dependency": j.get("dependency"),
                         "gpus_req": j.get("gpus_req"), "gpus": j["gpus"],
                         "submit_s": rel(j["submit"]), "eligible_s": rel(j["eligible"]), "start_s": rel(j["start"]),
                         "end_s": rel(j["end"]), "state": j.get("state"), "exit_code": j.get("exit_code"),
                         "end_reason": j.get("end_reason"), "launch_error": j.get("launch_error"),
                         "pending_s_by_reason": pend,
                         "queue_wait_s": (j["start"] - j["eligible"]) if j["start"] is not None and j["eligible"] is not None else None,
                         "scheduler_latency_s": lat_s, "resources_wait_with_gpus_free_s": cw, "ranks": ranks, "gpu_mask_mismatch": off})
    # GPU-seconds unallocated while an eligible job that fit was pending (the scheduler's own idle). "Fit" is the scheduler's
    # own test (fake-slurm's ctld.py: gpus <= free, cpus <= free_c, mem <= free_m) where the log carries the node's and the
    # jobs' cpus and mem_mb: a job whose GPUs fit but whose CPUs or memory do not (G-h4-haiku-S1, fake-slurm 2026-10-26
    # 16:12Z: 21 one-GPU trainings at -c 8 behind 42-CPU preps) loses those GPU-seconds to cpu/mem, whatever its reason
    # string. A log without them (an older controller) keeps the GPU-only test, and the Resources detector above takes its
    # cpu/mem waits out after
    pts = pts_all
    cpus_n, mem_n = ctl.get("cpus"), ctl.get("mem_mb")
    known = lambda j: cpus_n is not None and mem_n is not None and j.get("cpus") is not None and j.get("mem_mb") is not None
    sched_idle, cpu_mem_fit_s = 0.0, 0.0
    for a, b in zip(pts, pts[1:]):
        m = (a + b) / 2
        run_ = [j for j in jobs.values() if j["start"] is not None and j["start"] <= m < (j["end"] or last)]
        used = sum(len(j["gpus"] or []) for j in run_)
        free_c = (cpus_n - sum(int(j.get("cpus") or 0) for j in run_)) if cpus_n is not None else None
        free_m = (mem_n - sum(int(j.get("mem_mb") or 0) for j in run_)) if mem_n is not None else None
        cur = lambda j: next((w for t, w in reversed(j["pending"]) if t <= m), None)
        want = sorted((j for j in jobs.values()
                       if j["gpu_job"] and j["eligible"] is not None and j["eligible"] <= m and (j["start"] is None or m < j["start"])
                       and (j["end"] is None or m < j["end"]) and cur(j) not in ("JobHeldUser", "JobArrayTaskLimit")),
                      key=lambda j: (int(j.get("gpus_req") or 0), j["submit"] or 0))
        free, fit, blocked, fc, fm = max(0, gpus_n - used), 0, 0, 0, 0
        for j in want:
            w = int(j.get("gpus_req") or 0)
            if not 0 < w <= free - fit - blocked:
                continue
            if known(j) and (int(j["cpus"]) > free_c - fc or int(j["mem_mb"]) > free_m - fm):
                blocked += w                     # its GPUs are free; its CPUs or memory are not
                continue
            fit += w
            if known(j):
                fc += int(j["cpus"]); fm += int(j["mem_mb"])
        sched_idle += fit * (b - a)
        cpu_mem_fit_s += blocked * (b - a)
    # a GPU job's wait on CPUs or memory with its GPUs free is not the scheduler's idle: taken out, and reported apart. The
    # Resources detector's seconds count only for a job the fit test above could not judge (no cpus/mem in the log), so
    # nothing is taken out twice
    by_disp = {v["display"]: v for v in jobs.values()}
    cpu_mem_detector_s = sum(x["waited_s"] * int(by_disp[x["sched_job"]].get("gpus_req") or 0)
                             for x in cpu_wait if not known(by_disp[x["sched_job"]]))
    cpu_mem_gpu_s = cpu_mem_fit_s + cpu_mem_detector_s
    sched_idle = max(0.0, sched_idle - cpu_mem_detector_s)
    lats = [x["latency_s"] for x in lat if x.get("cause") != "cpu/mem"]
    untied = sorted(r for r in hello if r not in tied)
    return {"log": os.path.relpath(path, run_dir), "events": len(ev), "controller": {k: ctl.get(k) for k in ("node", "gpus", "cpus", "mem_mb", "version")},
            "jobs": out_jobs, "ranks_tied": len(tied), "ranks_untied": untied,
            "tied_by": {k: sum(1 for v in how.values() if v == k) for k in ("process tree", "process tree, GPU tie-break", "time and GPU")},
            "controller_errors": len(cerr), "controller_error_first": (str(cerr[0].get("error") or cerr[0])[:300] if cerr else None),
            "gpu_jobs": sum(1 for j in out_jobs if j["gpu_job"]), "non_gpu_jobs": sum(1 for j in out_jobs if not j["gpu_job"]),
            "gpu_mask_mismatch": mism, "ranks_outlived_job": outlived, "resources_wait_with_gpus_free": cpu_wait,
            "scheduler_latency": lat,
            "scheduler_latency_median_s": statistics.median(lats) if lats else None,
            "scheduler_latency_max_s": max(lats) if lats else None,
            "scheduler_idle_gpu_s": sched_idle, "cpu_mem_wait_gpu_s": cpu_mem_gpu_s,
            # lanes: GPU jobs that ran more than one task-job launch in sequence, and the task GPU-seconds inside them
            # (the enqueued wall cost of their ranks)
            "lane_jobs": sum(1 for j in out_jobs if j["gpu_job"] and j["runs_per_job"]["lane"]),
            "lane_task_gpu_s": sum(j["runs_per_job"]["task_gpu_s"] for j in out_jobs if j["gpu_job"] and j["runs_per_job"]["lane"]),
            "task_gpu_s_in_jobs": sum(j["runs_per_job"]["task_gpu_s"] for j in out_jobs if j["gpu_job"]),
            "states": {s: sum(1 for j in out_jobs if j["state"] == s) for s in sorted({j["state"] for j in out_jobs if j["state"]})},
            "how": "scheduler jobs from the event log; ranks tied through the box's process tree (slurm_script under "
                   "job<NNNNN>/ and its descendants' command lines), else by time and GPU; latency is end-that-freed to start "
                   "for jobs that waited on Resources; scheduler idle is unallocated GPUs while an eligible job that fit pended"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = analyse(a.run_dir)
    if r is None:
        sys.exit("no scheduler event log in the run directory")
    if a.json:
        print(json.dumps(r, indent=1)); return
    print(f"scheduler log   {r['log']}: {r['events']} events, {len(r['jobs'])} jobs ({r['gpu_jobs']} GPU, {r['non_gpu_jobs']} without) "
          f"{r['states']}; ranks tied {r['ranks_tied']} ({r['tied_by']}), untied {len(r['ranks_untied'])}; GPU mask mismatches "
          f"{len(r['gpu_mask_mismatch'])}; ranks outliving their job {len(r['ranks_outlived_job'])}; GPU jobs waiting on "
          f"cpu/mem with GPUs free {len(r['resources_wait_with_gpus_free'])} ({r['cpu_mem_wait_gpu_s']:.0f} GPU-s)")
    if r["scheduler_latency"]:
        print(f"sched latency   median {r['scheduler_latency_median_s']:.2f} s, max {r['scheduler_latency_max_s']:.2f} s over "
              f"{len(r['scheduler_latency'])} start(s) after a wait on Resources; scheduler idle {r['scheduler_idle_gpu_s']:.1f} GPU-s")
    for j in r["jobs"]:
        f = lambda x: "-" if x is None else f"{x:.1f}"
        print(f"  {str(j['display']):>8} {str(j['name'])[:24]:24s} submit {f(j['submit_s'])} start {f(j['start_s'])} end {f(j['end_s'])} "
              f"{j['state'] or '':10s} gpus {j['gpus']} ranks {j['ranks']} pending {({k: round(v, 1) for k, v in j['pending_s_by_reason'].items()})}")

if __name__ == "__main__":
    main()
