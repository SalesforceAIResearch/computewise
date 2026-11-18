"""Score one ComputeWise run directory: GPU utilisation, makespan, and the gap to the task's lower bound.

usage: python3 -I score_run.py <run_dir> [--task <task.json>] [--occupancy <file>] [--json]

Reads gpu-mock's occupancy log (one JSON event per line, written by
experiments/gpu-mock/src/daemon/gpu_daemon.py:OccupancyLog) and the task JSON from
experiments/gpu-tasks/registry/tasks/. Prints the three smoke-run numbers and, per job,
whether it replayed a record or ran modelled.

Makespan is reported from the first job launch and, when status.json carries
agent_start_wall (the agent's start on the daemon's wall clock), also from the agent's
start, so time spent before the first launch counts against the run.

Busy time is rebuilt from `enqueue` events, not from `busy_start`/`busy_end`: the daemon
logs `busy_start` on every idle-to-busy transition but `busy_end` only when a job ends
(`_end_job` -> `_recompute_idle`), so idle gaps between steps leave no event. Each GPU is
replayed the way the daemon's cursor works: a unit enqueued at t with wall cost c occupies
[max(cursor, t), max(cursor, t) + c). Busy GPU-seconds are the union of those intervals.
All times are wall seconds on the run's clock, so they include the speed knob; the
recorded-time equivalents multiply by `speed`, which every event carries.
"""
import argparse, json, os, sys

TRIAL_JOB_BASE = 1_000_000     # a later trial's rank id: job + TRIAL_JOB_BASE * (trial - 1), clear of the daemon's ids

def split_trials(ev):
    """gpu-mock's per-trial records (pair b23, 2026-10-22): a script that loops trials in one process is served one record
    per trial. The first trial's events carry no `trial`; from the second on, every lookup, enqueue and materialise
    carries `trial: n`, steps restart at 1, and the one job_bye says `trials` and the last trial's `last_step`. Each later
    trial becomes a rank of its own (job + TRIAL_JOB_BASE * (n - 1)), with a job_hello at its lookup (the process's argv,
    so it ties to the same task job), its events, and a job_bye where the next trial's lookup begins (the last trial's
    at the process's bye). Every scorer that keys per rank then reads each trial as one launch of its task job, judged
    against its own record. Runs with no `trial` field are returned unchanged."""
    if not any("trial" in e for e in ev):
        return ev
    hello = {e["job"]: e for e in ev if e.get("kind") == "job_hello"}
    bye = {e["job"]: e for e in ev if e.get("kind") == "job_bye"}
    starts = {}                                   # (job, trial) -> its lookup event
    for e in ev:
        if e.get("kind") == "lookup":
            starts.setdefault((e.get("job"), e.get("trial") or 1), e)
    trials = {}
    for (j, t) in starts:
        trials.setdefault(j, []).append(t)
    out = []
    for e in ev:
        t = e.get("trial") or 1
        j = e.get("job")
        if j is not None and t > 1:
            e = dict(e, job=j + TRIAL_JOB_BASE * (t - 1), trial_of=j)
        if e.get("kind") == "job_bye" and j in trials and len(trials[j]) > 1:
            continue                                # re-emitted below, one per trial
        out.append(e)
    for j, ts in trials.items():
        if len(ts) < 2:
            continue
        ts = sorted(ts)
        for i, t in enumerate(ts):
            nj = j + TRIAL_JOB_BASE * (t - 1)
            lk = starts[(j, t)]
            if t > 1:
                h = hello.get(j) or {}
                out.append(dict(h, job=nj, trial=t, trial_of=j, t_wall=lk["t_wall"], t_mono=lk["t_mono"], t_start=lk["t_wall"]))
            nxt = starts.get((j, ts[i + 1])) if i + 1 < len(ts) else None
            b0 = bye.get(j)
            if nxt is not None:
                out.append({"kind": "job_bye", "job": nj, "gpu": -1, "speed": lk.get("speed", 1.0), "trial": t, "trial_of": j,
                            "t_wall": nxt["t_wall"], "t_mono": nxt["t_mono"], "reason": "trial_end"})
            elif b0 is not None:
                out.append(dict(b0, job=nj, trial=t, trial_of=j))
    out.sort(key=lambda e: e.get("t_mono", 0))
    return out

def load_events(path):
    ev = []
    with open(path) as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path}:{n} is not JSON; skipped", file=sys.stderr)
    return split_trials(whole_job_lookups(ev))

def whole_job_lookups(ev):
    """gpu-mock's whole_job serving (r8.1 and later, the TTC tabular packages): a launch served one whole result is matched by
    an early `whole_job_probe` at sitecustomize and logs no `lookup` (gpu-mock's r891 evidence note, 2026-11-03). Every scorer
    keys a rank's record on its lookup, so W0-tsct-check10 read 54 of 54 exact-served jobs as "no lookup" and unverified. A
    probe for a job with no lookup of its own becomes that job's lookup (mode replay, its record and match, the probe's
    job_key), placed where the probe is; a job that also has a lookup keeps its own. The event keeps whole_job: true."""
    seen = {e.get("job") for e in ev if e.get("kind") == "lookup"}
    if not any(e.get("kind") == "whole_job_probe" and e.get("job") not in seen for e in ev):
        return ev
    out = []
    for e in ev:
        out.append(e)
        if e.get("kind") == "whole_job_probe" and e.get("job") not in seen:
            seen.add(e.get("job"))
            out.append(dict(e, kind="lookup", mode="replay", whole_job=True, from_event="whole_job_probe"))
    return out

def busy_intervals(events):
    """Per GPU, the merged busy intervals implied by the enqueue stream."""
    cursor, ivs = {}, {}
    for e in sorted((e for e in events if e.get("kind") == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"], float(e.get("wall_cost", 0.0))
        start = max(cursor.get(g, t), t)
        end = start + c
        cursor[g] = end
        lst = ivs.setdefault(g, [])
        if lst and start <= lst[-1][1]:
            lst[-1][1] = max(lst[-1][1], end)
        else:
            lst.append([start, end])
    return ivs

def score(events, task, gpus=None, agent_start_wall=None):
    if not events:
        raise SystemExit("no occupancy events")
    start_ev = next((e for e in events if e["kind"] == "daemon_start"), None)
    gpus = gpus or (start_ev or {}).get("gpus") or (task or {}).get("box", {}).get("gpus")
    speed = (start_ev or events[0]).get("speed", 1.0)
    hello = {e["job"]: e for e in events if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in events if e["kind"] == "job_bye"}
    lookup = {e["job"]: e for e in events if e["kind"] == "lookup"}
    ivs = busy_intervals(events)
    if not hello:
        # No job was launched: a real outcome (an agent that stops before launching anything, C4), not a scorer
        # failure. There is no makespan or utilisation over a launch span; idle.py scores the window as idle.
        out = {"gpus": gpus, "speed": speed, "launched_jobs": 0, "makespan_wall_s": None, "makespan_recorded_s": None,
               "busy_gpu_wall_s": 0.0, "utilisation": None, "jobs": [], "unfinished_jobs": [],
               "note": "no job reached the daemon (no job_hello events): nothing was launched"}
        if task:
            st_ = task.get("stats") or {}
            out.update({"task_id": task.get("task_id"), "task_class": task.get("task_class"),
                        "lower_bound_s": st_.get("lower_bound_makespan_s"), "serial_s": st_.get("serial_makespan_s")})
        return out
    # a job starts when its process starts (t_start, wall clock), not at its first CUDA use, which
    # is when gpu-mock now logs job_hello (2026-10-20); convert t_start to the daemon's mono clock
    def start_mono(e):
        t = e.get("t_start")
        return e["t_mono"] - (e["t_wall"] - float(t)) if isinstance(t, (int, float)) else e["t_mono"]
    t0 = min(start_mono(e) for e in hello.values())
    t_last = max([e["t_mono"] for e in bye.values()] + [iv[1] for l in ivs.values() for iv in l])
    makespan = t_last - t0
    # The agent's start, if the run directory gives it on the daemon's wall clock
    # (status.json agent_start_wall). t_mono and t_wall advance together within one daemon,
    # so the offset of any event converts between them.
    ref = next(iter(hello.values()))
    agent_start_mono = (agent_start_wall - ref["t_wall"] + ref["t_mono"]) if agent_start_wall else None
    busy = sum(min(b, t_last) - max(a, t0) for l in ivs.values() for a, b in l if b > t0)
    util = busy / (gpus * makespan) if makespan > 0 and gpus else float("nan")
    out = {
        "gpus": gpus, "speed": speed,
        "makespan_wall_s": makespan, "makespan_recorded_s": makespan * speed,
        "busy_gpu_wall_s": busy, "utilisation": util,
        "jobs": [],
        "unfinished_jobs": sorted(set(hello) - set(bye)),
        "makespan_from": "first job's process start (job_hello t_start, else job_hello)",
    }
    if agent_start_mono is not None:
        span_a = t_last - agent_start_mono
        out.update({"makespan_from_agent_start_wall_s": span_a,
                    "makespan_from_agent_start_recorded_s": span_a * speed,
                    "before_first_launch_wall_s": t0 - agent_start_mono,
                    "utilisation_from_agent_start": busy / (gpus * span_a) if span_a > 0 else float("nan")})
    for j in sorted(hello):
        lk = lookup.get(j, {})
        out["jobs"].append({
            "job": j, "argv0": (hello[j].get("argv") or ["?"])[0],
            "mode": lk.get("mode", "no lookup"), "record_id": lk.get("record_id"),
            "match": lk.get("match"),
            "wall_s": (bye[j]["t_mono"] - start_mono(hello[j])) if j in bye else None,
        })
    if task:
        lb = task.get("stats", {}).get("lower_bound_makespan_s")
        serial = task.get("stats", {}).get("serial_makespan_s")
        out.update({"task_id": task.get("task_id"), "task_class": task.get("task_class"),
                    "lower_bound_s": lb, "serial_s": serial})
        if lb:
            m = out["makespan_recorded_s"]
            out["gap_to_lower_bound_s"] = m - lb
            out["makespan_over_lower_bound"] = m / lb
            if "makespan_from_agent_start_recorded_s" in out:
                ma = out["makespan_from_agent_start_recorded_s"]
                out["gap_to_lower_bound_from_agent_start_s"] = ma - lb
                out["makespan_from_agent_start_over_lower_bound"] = ma / lb
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--task", help="task JSON; default <run_dir>/task.json")
    ap.add_argument("--occupancy", help="occupancy log; default <run_dir>/occupancy.jsonl")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--live", action="store_true", help="score a run that has not finished")
    a = ap.parse_args()
    occ = a.occupancy or os.path.join(a.run_dir, "occupancy.jsonl")
    tpath = a.task or os.path.join(a.run_dir, "task.json")
    task = json.load(open(tpath)) if os.path.exists(tpath) else None
    spath = os.path.join(a.run_dir, "status.json")
    st = json.load(open(spath)) if os.path.exists(spath) else {}
    if not (st.get("end_utc") or st.get("status") in ("completed", "killed-on-budget", "killed-on-time", "stopped", "crashed", "harness-error")) and not a.live:
        sys.exit(f"{a.run_dir}: status.json has no end time and no final status, so the run has not finished; pass --live to score it as it stands")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import idle
    events = load_events(occ)
    try:
        speed = idle.run_speed(a.run_dir, events)
    except idle.SpeedUnknown as e:
        sys.exit(str(e))
    r = score(events, task, agent_start_wall=st.get("agent_start_wall"))
    r["status"] = st.get("status")
    r["smoke_test"] = speed != 1.0
    if a.json:
        print(json.dumps(r, indent=1)); return
    if r["smoke_test"]:
        print(idle.SMOKE_NOTE.format(speed=r["speed"]))
    print(f"task            {r.get('task_id', '(no task.json)')}  class {r.get('task_class')}  status {r.get('status') or '(no status.json)'}")
    print(f"GPUs            {r['gpus']}   speed knob {r['speed']}x")
    if r.get("launched_jobs") == 0:
        print(f"launched        nothing: {r['note']}")
        return
    print(f"utilisation     {r['utilisation']:.3f}   (busy {r['busy_gpu_wall_s']:.1f} GPU-s over {r['gpus']} x {r['makespan_wall_s']:.1f} s)")
    print(f"makespan        {r['makespan_recorded_s']:.1f} s recorded-time  ({r['makespan_wall_s']:.1f} s wall)")
    if r.get("lower_bound_s"):
        print(f"lower bound     {r['lower_bound_s']:.1f} s   gap {r['gap_to_lower_bound_s']:+.1f} s   ratio {r['makespan_over_lower_bound']:.2f}x")
    if "makespan_from_agent_start_recorded_s" in r:
        print(f"from agent start {r['makespan_from_agent_start_recorded_s']:.1f} s recorded-time, of which {r['before_first_launch_wall_s'] * r['speed']:.1f} s before the first launch;"
              f" utilisation {r['utilisation_from_agent_start']:.3f}" + (f"; gap {r['gap_to_lower_bound_from_agent_start_s']:+.1f} s, ratio {r['makespan_from_agent_start_over_lower_bound']:.2f}x" if r.get('lower_bound_s') else ""))
    else:
        print("from agent start (no agent_start_wall in status.json; makespan is from the first job launch)")
    for j in r["jobs"]:
        w = f"{j['wall_s']:.1f} s wall" if j["wall_s"] is not None else "UNFINISHED"
        print(f"  job {j['job']:>3}  {j['mode']:<9} {j['match'] or '':<6} {j['record_id'] or '-'}  {w}")
    if r["unfinished_jobs"]:
        print(f"warning: jobs without job_bye: {r['unfinished_jobs']}")

if __name__ == "__main__":
    main()
