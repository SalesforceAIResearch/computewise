"""What an S-rung run's agent did with the scheduler, beyond what the scheduler itself did: submit order against job
length, and on staged tasks the decision latency and what the agent waited with.

For each run directory (a collected run: scheduler.jsonl, occupancy.jsonl, task.json, transcript/):
  - jobs: each sbatch job tied to the task job(s) its ranks replayed (gpu-mock's job_hello argv against the task's
    launch lines; a lane job runs several), with its recorded length (the replay's paced wall time, from the run's own
    occupancy: job_hello to job_bye of the rank);
  - submit order: for each stage (jobs released together), the rank of each job's length in the order submitted, and
    whether the longest went first;
  - decision latency (staged tasks): for each later-stage submit, submit time minus the last end among the jobs its
    stage depends on (the task's `after`), and how the agent learned of that end: a background `sbatch --wait`/srun
    task (Claude Code) or a wait primitive, a `--dependency` (no wait needed), or a poll (sleep/squeue/sacct loop).

usage: python3 -I s_rung_readout.py <run dir>... [--json out.json]
"""
import argparse, json, os, re, shlex, sys


def load_jsonl(p):
    out = []
    try:
        for l in open(p):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def norm_launch(cmd):
    """A launch line's identity: the script path and the args, order-insensitive, with torchrun's own flags dropped."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    toks = [t for t in toks if not (t in ("torchrun", "python", "python3", "-u") or t.startswith("--standalone")
                                    or t.startswith("--nproc_per_node") or t.startswith("--nproc-per-node"))]
    script = next((t for t in toks if t.endswith(".py")), None)
    rest = toks[toks.index(script) + 1:] if script else toks
    pairs, i = [], 0
    while i < len(rest):
        a = rest[i]
        if a.startswith("--") and "=" in a:
            k, v = a.split("=", 1); pairs.append((k, v)); i += 1
        elif a.startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
            pairs.append((a, rest[i + 1])); i += 2
        else:
            pairs.append((a, None)); i += 1
    return (os.path.normpath(script) if script else None, tuple(sorted(pairs)))


def task_jobs(task):
    """job_id -> {key, after, stage}; stage 0 = no deps, else 1 + max stage of deps. Decision launches have no fixed
    line in the task, so their keys come from every candidate's confirm_launch."""
    jobs = {j["job_id"]: dict(after=list(j.get("after") or []), keys=set()) for j in task["jobs"]}
    for j in task["jobs"]:
        if j.get("launch") and "<" not in j["launch"]:
            jobs[j["job_id"]]["keys"].add(norm_launch(j["launch"]))
    for d in task.get("decisions") or []:
        for jid in d.get("launches") or []:
            for c in d.get("candidates") or []:
                for k in ("confirm_launch", "launch"):
                    if c.get(k):
                        jobs.setdefault(jid, dict(after=list(d.get("after") or []), keys=set()))["keys"].add(norm_launch(c[k]))
    stage = {}
    def st(j):
        if j not in stage:
            stage[j] = 0 if not jobs[j]["after"] else 1 + max(st(a) for a in jobs[j]["after"] if a in jobs)
        return stage[j]
    for j in jobs:
        st(j)
        jobs[j]["stage"] = stage[j]
    return jobs


def tool_calls(run):
    """(t_rel, kind, command, background) for every shell call the agent made, from Claude Code's stream or Codex's
    rollout; t_rel in seconds from the first record."""
    calls = []
    s = os.path.join(run, "transcript", "stream.jsonl")
    rec = load_jsonl(s)
    if rec and rec[0].get("type") in ("system", "assistant", "user"):
        # Claude Code stream: no timestamps per line; order only
        for i, r in enumerate(rec):
            if r.get("type") != "assistant":
                continue
            for c in (r.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_use" and c.get("name") == "Bash":
                    inp = c.get("input") or {}
                    calls.append((i, "bash", inp.get("command", ""), bool(inp.get("run_in_background"))))
                elif c.get("type") == "tool_use" and c.get("name") in ("Task", "Agent"):
                    calls.append((i, "agent", json.dumps(c.get("input"))[:400], True))
        return calls, "claude-code"
    import glob
    roll = sorted(glob.glob(os.path.join(run, "transcript", "codex-home", "sessions", "**", "*.jsonl"), recursive=True),
                  key=os.path.getsize, reverse=True)
    for i, r in enumerate(load_jsonl(roll[0]) if roll else []):
        p = r.get("payload") or {}
        if p.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
            a = str(p.get("arguments") or p.get("input") or json.dumps(p.get("action")))
            calls.append((i, "codex", a, False))
    return calls, "codex"


def classify_wait(calls):
    """How the agent waited for jobs to end, over the whole run: background sbatch --wait/srun tasks, --dependency,
    polls (a loop or repeated sleep + squeue/sacct), fixed sleeps."""
    n = dict(bg_wait=0, dependency=0, poll_loop=0, sleep_check=0, fixed_sleep=0)
    for _, kind, cmd, bg in calls:
        if "sbatch" in cmd and "--dependency" in cmd or re.search(r"\s-d\s+after", cmd):
            n["dependency"] += 1
        if bg and (("sbatch" in cmd and ("--wait" in cmd or re.search(r"\s-W\b", cmd))) or cmd.lstrip().startswith("srun")):
            n["bg_wait"] += 1
        if re.search(r"\b(while|until)\b", cmd) and re.search(r"squeue|sacct|scontrol", cmd):
            n["poll_loop"] += 1
        if kind == "codex" and '"duration_ms"' in cmd:
            n["sleep_check"] += 1
        for m in re.finditer(r"\bsleep\s+(\d+(?:\.\d+)?)", cmd):
            if float(m.group(1)) >= 5 and not re.search(r"\b(while|until)\b", cmd):
                n["fixed_sleep"] += 1
    return n


def readout(run):
    task = json.load(open(os.path.join(run, "task.json")))
    tj = task_jobs(task)
    ev = load_jsonl(os.path.join(run, "scheduler.jsonl"))
    occ = load_jsonl(os.path.join(run, "occupancy.jsonl"))
    if not ev:
        return dict(run=os.path.basename(run.rstrip("/")), error="no scheduler.jsonl")
    t0 = ev[0]["t"]
    sub = {e["job"]: e for e in ev if e["ev"] == "submit"}
    start = {e["job"]: e for e in ev if e["ev"] == "start"}
    end = {e["job"]: e for e in ev if e["ev"] == "end"}
    # ranks: gpu-mock jobs with their argv and wall span; tie each to the sbatch job running when it said hello on a
    # GPU that job held
    hello, bye, gpu_of = {}, {}, {}
    for r in occ:
        if r["kind"] == "job_hello":
            hello[r["job"]] = r
        elif r["kind"] == "job_bye":
            bye[r["job"]] = r
        elif r.get("job") in hello and r.get("gpu", -1) >= 0 and r["job"] not in gpu_of:
            gpu_of[r["job"]] = r["gpu"]
    ranks = []
    claimed = {}                    # launch key -> decision job it was given (two picks share every candidate's key)
    for k, h in sorted(hello.items(), key=lambda kv: kv[1]["t_wall"]):
        key = norm_launch(" ".join(h.get("argv") or []))
        opts = [j for j, v in tj.items() if key in v["keys"]]
        if not opts:
            continue
        if key in claimed:          # a resubmit of the same launch is the same task job
            tid = claimed[key]
        else:
            used = set(claimed.values())
            tid = next((j for j in opts if j not in used), opts[0])
            claimed[key] = tid
        span = (bye[k]["t_wall"] - h["t_wall"]) if k in bye else None
        sj = None
        for jid, s in start.items():
            e = end.get(jid)
            if s["t"] - 1 <= h["t_wall"] <= (e["t"] if e else 1e18) + 1 and (gpu_of.get(k) in (s.get("gpus") or [])
                                                                           or gpu_of.get(k) is None):
                if sj is None or s["t"] > start[sj]["t"]:
                    sj = jid
        ranks.append(dict(task_job=tid, stage=tj[tid]["stage"], hello=h["t_wall"] - t0, len_s=span, sbatch=sj))
    # one row per task job that ran to the end (the last rank of it, so a cancelled-and-resubmitted job counts once)
    by_tj = {}
    for r in sorted(ranks, key=lambda x: x["hello"]):
        by_tj[r["task_job"]] = r
    rows = list(by_tj.values())
    out = dict(run=os.path.basename(run.rstrip("/")), task=task["task_id"], stages={})
    for stg in sorted({r["stage"] for r in rows}):
        rs = [r for r in rows if r["stage"] == stg and r["sbatch"] in sub]
        rs.sort(key=lambda r: (sub[r["sbatch"]]["t"], r["sbatch"]))
        lens = [r["len_s"] or 0 for r in rs]
        order = sorted(range(len(lens)), key=lambda i: -lens[i])
        rank_of = {i: order.index(i) + 1 for i in range(len(lens))}
        lane = len({r["sbatch"] for r in rs}) < len(rs)
        info = dict(jobs=[dict(task_job=r["task_job"], submit_s=round(sub[r["sbatch"]]["t"] - t0, 1),
                               len_s=round(r["len_s"] or 0, 1), len_rank=rank_of[i], sbatch=r["sbatch"])
                          for i, r in enumerate(rs)],
                    longest_first=bool(rs) and rank_of[0] == 1,
                    len_rank_in_submit_order=[rank_of[i] for i in range(len(rs))], lanes=lane)
        if stg > 0:
            lat = []
            for r in rs:
                deps = tj[r["task_job"]]["after"]
                dep_ends = [end[by_tj[d]["sbatch"]]["t"] for d in deps if d in by_tj and by_tj[d]["sbatch"] in end]
                if dep_ends:
                    dep_sub = sub[r["sbatch"]]
                    lat.append(dict(task_job=r["task_job"], latency_s=round(dep_sub["t"] - max(dep_ends), 1),
                                    by_dependency=bool(dep_sub.get("dependency"))))
            info["decision_latency"] = lat
        out["stages"][stg] = info
    calls, harness = tool_calls(run)
    out["harness"] = harness
    out["waits"] = classify_wait(calls)
    out["cancelled"] = sum(1 for e in end.values() if e.get("state") == "CANCELLED")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    res = [readout(r) for r in a.runs]
    for r in res:
        if r.get("error"):
            print(f"{r['run']}: {r['error']}")
            continue
        print(f"{r['run']}  ({r['task']}, {r['harness']})  waits {r['waits']}  cancelled {r['cancelled']}")
        for stg, s in r["stages"].items():
            lat = s.get("decision_latency")
            print(f"  stage {stg}: {len(s['jobs'])} jobs, longest first: {s['longest_first']}, length rank in submit order "
                  f"{s['len_rank_in_submit_order']}{', LANES' if s['lanes'] else ''}"
                  + (f"; decision latency {[(x['latency_s'], 'dep' if x['by_dependency'] else 'reactive') for x in lat]}"
                     if lat else ""))
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
