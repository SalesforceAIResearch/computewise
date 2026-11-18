"""The hindsight bound: how fast the jobs the agent actually ran could have finished, given when each became
possible.

usage: python3 -I hindsight.py <run_dir> [--json]

Jobs: every launch that completed (one gpu-mock job key; every rank ended cleanly, as idle.py's work_remaining rules
it). For a task with a job list, a task job's completing launch; for an open-ended package, each completed launch. Each
keeps the GPUs it used (width) and its own run time (first rank's process start to its last rank's bye).

Release, the earliest legitimate start of job i:
  inputs ready   the end of the last tool call before i's launch that wrote a file i's command names (its script and
                 any config or data path in its arguments: a Write, Edit or apply_patch of it, or a shell command naming
                 it with a redirect or heredoc); the agent's start when nothing wrote them (files given with the task)
  dependency     job j's completion, when the agent read j's results (a tool call naming j's log or output, whose
                 result came after j completed) before it first named i's script in its own output (tool input, text
                 or thinking). A job the agent named before that read was planned independently of it: reading results
                 while waiting does not make the plan depend on them
  release        max(inputs ready, every dependency's completion)

The bound: the shortest of several non-preemptive list schedules of those jobs on the box's GPUs (by release, longest
first, widest first, the agent's own launch order), each job at its release or later. Beside it a floor, max(the latest
release + run time, the earliest release + total GPU-seconds / GPUs), which no schedule beats. Times are recorded
seconds from the agent's start.

Per job: release, launch, how late (launch - max(release, the moment enough GPUs were free in the actual run)), and what
the agent was doing then (the call it was in). This generalises prep.py's prep-aware bound: on a package whose jobs are
all named at the start, the release is the prep end, and the two agree up to how prep is sequenced.
"""
import argparse, json, os, re, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idle, prep, score_run

WRITE_RE = re.compile(r">>?|<<|\btee\b|\bcp\b|\bmv\b|\bsed\s+-i|apply_patch|cat\s*>")

def writes_path(name, text, p):
    """Does this call write file p? A file tool on it, or a shell command whose redirect, tee, cp/mv/install
    destination, sed -i or apply_patch target is it. Running a script is not writing it: a launch line that names the
    script and redirects its log elsewhere writes only the log (Q-p6par-haiku-L0: the relaunch commands were read as new
    inputs, and the bound followed the agent's own launches)."""
    if not p:
        return False
    if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
        return p in text
    if name != "Bash":
        return False
    tgt = []
    tgt += re.findall(r"(?:>>?|\btee(?:\s+-a)?)\s*([^\s;&|)<>]+)", text)
    tgt += re.findall(r"apply_patch\s+(\S+)", text)
    tgt += re.findall(r"\bsed\s+-i\S*\s+(?:'[^']*'|\"[^\"]*\"|\S+)\s+(\S+)", text)
    tgt += [m[-1] for m in re.findall(r"\b(?:cp|mv|install)\s+(?:-\S+\s+)*(\S+)\s+(\S+)", text)]
    return any(t and (t.rstrip("/").endswith(p) or p.endswith(t.strip("./"))) and not t.startswith(("&", "/dev/")) for t in tgt)

GEN_RE = re.compile(r"\b(?:python3?|bash|sh|node)\s+(?:-\S+\s+)*\S+\.(?:py|sh|js)\b")   # runs a script
READ_RE = re.compile(r"\b(?:grep|egrep|rg|tail|head|cat|less|wc|diff|find|ls)\b")       # only reads
ACT_WAIT = re.compile(r"(^|[;&|\s(])(sleep\s+\d|wait\b|watch\b|tail\s+-[fF])")
OUT_OPT = re.compile(r"^--(?:output[_-]?dir|out(?:put)?|save[_-]?dir|log[_-]?dir|run[_-]?dir)$")
REDIRECT_RE = re.compile(r">>?\s*(\S+)")

def _argv_paths(argv):
    """The files a launch names: its script and any argument that looks like a path."""
    return [t for t in argv if ("/" in t or re.search(r"\.(py|sh|ya?ml|json|toml|cfg|txt|pt|bin|jsonl)$", t))
            and not t.startswith("-")]

def _outputs(argv, launch_cmd):
    outs = []
    for k, t in enumerate(argv):
        if OUT_OPT.match(t) and k + 1 < len(argv):
            outs.append(argv[k + 1])
        elif "=" in t and OUT_OPT.match(t.split("=", 1)[0]):
            outs.append(t.split("=", 1)[1])
    if launch_cmd:
        outs += [m.group(1) for m in REDIRECT_RE.finditer(launch_cmd) if not m.group(1).startswith(("&", "/dev/"))]
    return [o for o in outs if o]

def _tail(p):
    return p.strip("./") if p else p

def schedule(jobs, gpus):
    """Non-preemptive list schedules; returns (makespan, order name, starts).

    A job with `rel_deps` (its dependencies, each with `lag`) is released in the schedule at max(its own release,
    each dependency's end IN THIS SCHEDULE + that lag), not at an absolute time: when the agent wrote a job's inputs
    after its dependencies had ended, the bound keeps the agent's own latency (the write minus the dependencies' actual
    end) but lets an earlier stall upstream move with the bound (the analysis, 2026-10-22 20:50Z: Q-halving-
    sonnet-L0's round 3 was written the moment round 2 ended, 3,000 s late because of a self-matching wait, and the bound
    kept that absolute time, reading 1.00x with 66.8% of the box idle). A job is placed only once its dependencies are."""
    best = None
    rules = {"release": lambda j: (j["release"], -j["run"]), "longest": lambda j: (-j["run"], j["release"]),
             "widest": lambda j: (-j["width"], -j["run"], j["release"]), "agent_order": lambda j: j["launch"]}
    for name, key in rules.items():
        free = [0.0] * gpus                    # when each GPU frees
        pending = sorted(jobs, key=key)
        starts, t = {}, 0.0
        ends = {}
        def ready(j):
            return all(d in ends for d, _ in j.get("rel_deps") or [])
        def release_of(j):
            return max([j["release"]] + [ends[d] + lag for d, lag in j.get("rel_deps") or []])
        while pending:
            placed = False
            for j in pending:
                # the earliest time j can start: its release and enough GPUs free
                avail = sorted(free)
                if j["width"] > gpus or not ready(j):
                    continue
                s = max(release_of(j), avail[j["width"] - 1])
                # list scheduling: take the first pending job (in priority order) that can start soonest at or
                # before any higher-priority job could
                starts[j["id"]] = s
                ends[j["id"]] = s + j["run"]
                used = sorted(range(gpus), key=lambda g: free[g])[:j["width"]]
                for g in used:
                    free[g] = s + j["run"]
                pending.remove(j); placed = True
                break
            if not placed:
                break
        if pending:
            continue                           # a dependency cycle or an unplaceable job: this order gives no schedule
        mk = max((starts[j["id"]] + j["run"] for j in jobs), default=0.0)
        if best is None or mk < best[0]:
            best = (mk, name, dict(starts))
    return best

def analyse(run_dir):
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    t0 = st.get("agent_start_wall")
    if t0 is None:
        return {"bound_s": None, "why": "no agent start"}
    speed = float(idle.run_inputs(run_dir).get("speed") or 1.0)
    gpus = int((task.get("box") or {}).get("gpus") or 4)
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    idle.attach_decisions(task, [e for e in ev if e["kind"] == "lookup"])
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    bye = {e["job"]: e["t_wall"] for e in ev if e["kind"] == "job_bye"}
    bad = {e["job"] for e in ev if e["kind"] in ("failure", "exception")}
    gpus_of = {}
    for e in ev:
        if e["kind"] == "enqueue":
            gpus_of.setdefault(e["job"], set()).add(e["gpu"])
    rem = idle.work_remaining(ev, task) if task.get("jobs") else idle.work_remaining(ev, {"jobs": []})
    done_launch = rem.get("done_launch_wall") or {}
    share = rem.get("share_delay_by_rank_s") or {}
    # the jobs a launch ties to (idle.tie_jobs: G4n's unneeded_jobs too, so the rejected setup's full runs are counted as the
    # work they were, not dropped as untied)
    jobs_l = idle.tie_jobs(task)
    # a task whose jobs carry required_width ties its launches by width, as completion does (idle.work_remaining's width_ties):
    # the unseeded rows of one script share every launch option, so task_job_of alone ties none of them (pkg-astra-widths'
    # ten unseeded muonh-tuned rows; W0-awid-measure counted 49 of 49 complete but its bound held 39; Astra independent and the
    # the study, 2026-11-02 20:12 MST)
    width_tie = {}
    if any(j.get("required_width") for j in jobs_l or []):
        _reached = {}
        for e in ev:
            if e["kind"] == "enqueue" and e.get("unit") in ("step", "whole_job") and isinstance(e.get("step"), (int, float)):
                _reached[e["job"]] = max(_reached.get(e["job"], -1), e["step"])
        width_tie = idle.width_ties(hello, lookup, jobs_l, bye, bad, _reached)
    # and interchangeable rows (same launch, same record pool) for ranks task_job_of leaves untied, as completion does (idle.pool_ties)
    _reached_p = {}
    for e in ev:
        if e["kind"] == "enqueue" and e.get("unit") in ("step", "whole_job") and isinstance(e.get("step"), (int, float)):
            _reached_p[e["job"]] = max(_reached_p.get(e["job"], -1), e["step"])
    pool_tie = idle.pool_ties(hello, lookup, jobs_l, bye, bad, _reached_p, skip=width_tie) if jobs_l else {}
    tie_of = lambda d: (width_tie.get(d) or (idle.task_job_of(hello[d], lookup.get(d), jobs_l) if jobs_l else None)
                        or pool_tie.get(d))
    # launches: ranks grouped by job key
    launches = {}
    for d, h in hello.items():
        k = (lookup.get(d) or {}).get("job_key") or f"rank-{d}"
        launches.setdefault(k, []).append(d)
    mtimes = prep._manifest_mtimes(run_dir)
    recs = idle.transcript_records(run_dir)
    calls = prep.calls_with_inputs(recs, open_until=st.get("teardown_wall"))
    main_calls = [c[:2] + (c[2], prep.call_text(c[2], c[3])) for c in calls]   # what the main agent itself was in
    # subagents write and launch too (Q-prepP6-haiku-L0-v2: six subagents wrote the six scripts): their calls count
    import glob as _glob
    for f in sorted(_glob.glob(os.path.join(run_dir, "transcript", "session", "**", "subagents", "*.jsonl"), recursive=True)):
        sub = [r for r in idle.read_jsonl(f) if r.get("timestamp")]
        sub.sort(key=lambda r: idle.ts(r["timestamp"]))
        calls += prep.calls_with_inputs(sub, open_until=st.get("teardown_wall"))
        recs = recs + sub
    calls.sort(key=lambda c: (c[0], c[1]))
    texts = [(c[0], c[1], c[2], prep.call_text(c[2], c[3]), c[7] or "") for c in calls]
    said = [(idle.ts(r["timestamp"]), (b.get("text") or b.get("thinking") or ""))
            for r in recs if r.get("type") == "assistant" for b in ((r.get("message") or {}).get("content") or [])
            if isinstance(b, dict) and b.get("type") in ("text", "thinking")]
    jobs = []
    for k, ds in launches.items():
        if not all(d in bye and d not in bad for d in ds):
            continue                           # only launches that completed
        start = min(idle.job_start(hello[d]) for d in ds)
        tj = tie_of(ds[0])
        if jobs_l:
            is_open = next((j.get("draw") == "open" for j in jobs_l if j["job_id"] == tj), False)
            if tj is None or (not is_open and (tj not in done_launch or abs(done_launch[tj] - start) > 1.0)):
                continue                       # a task job counts once, at its completing launch; an open pool's
                                               # every completed instance counts
        argv = list(hello[ds[0]].get("argv") or [])
        # the shell command that launched it: the latest tool call before the launch naming its script
        script = next((a for a in argv if a.endswith((".py", ".sh"))), argv[0] if argv else "")
        lc = next((t for (a, b, n, t, r) in sorted(texts, key=lambda x: -x[0]) if a <= start + 2 and _tail(script) and _tail(script) in t), "")
        jobs.append({"id": (f"{tj}#{sum(1 for x in jobs if x['task_job'] == tj) + 1}" if tj and next((j.get("draw") == "open" for j in jobs_l if j["job_id"] == tj), False)
                            else tj or k), "task_job": tj, "argv": argv, "script": script, "launch_cmd": lc,
                     "launch": start, "end": max(bye[d] for d in ds), "width": len(set().union(*[gpus_of.get(d, set()) for d in ds])) or len(ds),
                     "share_delay": max((share.get(d, 0.0) for d in ds), default=0.0)})
    if not jobs:
        return {"bound_s": None, "why": "no completed launch", "jobs": []}
    def inputs_ready(paths, before):
        """The last write of a file the launch names, before `before`: (time, how) or (agent start, given)."""
        writes = [b for (a, b, n, t, r) in texts if b <= before + 1 and any(writes_path(n, t, p) for p in paths)]
        # a generator that never names the files it writes (`python3 build.py`): a writing call whose own output names
        # one, or the workspace manifest's write time of one (prep.py's two rules)
        # (a generator runs a script; a grep or tail whose output prints the path, a traceback most often, read it)
        writes += [b for (a, b, n, t, r) in texts if b <= before + 1 and n == "Bash" and r and any(p and p in r for p in paths)
                   and GEN_RE.search(t) and not READ_RE.search(t)
                   and not (prep.BLOCKING_WAIT_RE.search(t) or prep.GPU_LAUNCH_RE.search(t) or prep.LIST_RE.search(t))]
        mt = [m for p in paths for k, m in mtimes.items() if p and (k == p or k.endswith("/" + p) or k.endswith(p)) and m <= before + 1]
        if not (writes or mt):
            return t0, "files given with the task"
        w = max(writes + mt)
        if w < t0:
            # a file staged before the agent started (the task tree's cache copy keeps its mtime): given with the task
            # (Q-halving-haiku-L0 read a release of -2,534 s and a floor of -1,874 s from it)
            return t0, "files given with the task (written before the agent started)"
        return w, "last write of a file it names" + (" (manifest mtime)" if mt and max(mt) >= max(writes, default=0) else "")
    # every attempt of each task job (completed or not), and every failed one, for the regenerated-job rule
    attempts, failed_ends = {}, []
    # for the two T-score fields below: every ended attempt's interval, and each failed
    # one's cause, the first failure or exception its ranks logged (a served OOM logs its failure, then the excepthook's exception)
    tried, charged, cause_of, fail_t = {}, {}, {}, {}
    agent_d = {u["job"] for u in rem.get("agent_caused_crashes") or []}
    for e in ev:
        if e["kind"] in ("failure", "exception") and e.get("job") is not None:
            fail_t.setdefault(e["job"], []).append(e["t_wall"])
            cause_of.setdefault(e["job"], f"failure {e.get('label') or e.get('mode')}" if e["kind"] == "failure"
                                else f"exception {e.get('etype') or ''}".strip())
    for k, ds in launches.items():
        tj_ = tie_of(ds[0])
        st_ = min(idle.job_start(hello[d]) for d in ds)
        ok_ = all(d in bye and d not in bad for d in ds)
        attempts.setdefault(tj_ or k, []).append((st_, ok_))
        if not ok_:
            failed_ends.append(max([bye.get(d) or st_ for d in ds] + [e["t_wall"] for e in ev if e["kind"] in ("exception", "failure") and e.get("job") in ds]))
        ft = [t for d in ds for t in fail_t.get(d, [])]
        if all(d in bye for d in ds) or ft:
            en_ = max([bye[d] for d in ds if d in bye] + ft)
            tried.setdefault(tj_ or k, []).append((st_, en_))
            cause = next((cause_of[d] for d in ds if d in cause_of), None)
            # charged to the environment: a failed attempt with a failure or exception that is not the agent's. Not one the agent
            # stopped itself (a signal's KeyboardInterrupt, SystemExit, or a kill with neither), not an oversubscription OOM (the agent
            # put two jobs on one card: gpu-mock's `failure` label oversubscribed), and not one work_remaining reads as agent-caused
            # (an exception in a script the agent wrote or edited, or the data guard). A served failure (label served_failure, a
            # record's failure armed by the launch, pkg-oom-relaunch's OOM) is the environment's
            if (not ok_ and cause and cause not in ("exception KeyboardInterrupt", "exception SystemExit", "failure oversubscribed")
                    and not (set(ds) & agent_d)):
                charged.setdefault(tj_ or k, []).append({"start": st_, "end": en_, "cause": cause,
                                                         "width": len(set().union(*[gpus_of.get(d, set()) for d in ds])) or len(ds)})
    try:
        import launches as launches_mod
        for x in (launches_mod.scan(run_dir).get("launches_failed_before_gpu") or []):
            failed_ends.append(t0 + x["end_s"] / speed)
            attempts.setdefault(x["task_job"], []).append((t0 + x["start_s"] / speed, False))
            # a launch that failed before reaching a GPU (its command line, most often) holds its job's place while it runs, and is
            # the agent's, never charged to the environment
            tried.setdefault(x["task_job"], []).append((t0 + x["start_s"] / speed, t0 + x["end_s"] / speed))
    except Exception:
        pass
    call_ends = sorted(b for (a, b, n, t, r) in texts)
    def j_logs(j):
        """The logs a job's launches write: redirect targets in its launch command."""
        return [_tail(m.group(1)) for m in REDIRECT_RE.finditer(j.get("launch_cmd") or "") if not m.group(1).startswith(("&", "/dev/"))]
    for j in jobs:
        # its solo run time: a job that waited behind another on its GPU ran stretched (Q-prepP6-haiku-L2)
        j["run"] = j["end"] - j["launch"] - j["share_delay"]
        paths = [_tail(p) for p in _argv_paths(j["argv"])]
        j["inputs_ready"], j["inputs_how"] = inputs_ready(paths, j["launch"])
        # a regenerated or relaunched job (an earlier attempt failed, or completed and was superseded): its release is
        # the time a failure became observable plus the agent's own fix time (its first look after the failure to the
        # rewrite), so the delay before it looked, a blind sleep most often, stays on the agent's side of the bound
        #
        # (an open pool's instances are each their own job: an earlier instance is no earlier attempt of this one)
        earlier = [] if j["id"] != j["task_job"] and j["task_job"] and "#" in str(j["id"]) else \
            sorted(a for a, ok in attempts.get(j["id"], []) if a < j["launch"] - 1.0)
        j["regenerated"] = None
        if earlier:
            first_in, first_how = inputs_ready(paths, earlier[0])
            j["first_in"] = first_in
            fails = [f for f in failed_ends if earlier[0] - 1.0 <= f <= j["launch"]]
            # only a job whose input was rewritten after its first attempt is regenerated; a launch that failed on
            # its command line, with the files as they were, delays nothing (a failed launch is no input)
            rewritten = j["inputs_ready"] > earlier[0] + 1.0
            if fails and rewritten:
                obs = min(fails)
                # the agent's first look at it: the first call after the failure that names the job (its script, its
                # directory or its log); one that starts with a blocking sleep looks only when the sleep ends
                keys = [k_ for k_ in ([_tail(j["script"]), _tail(os.path.dirname(j["script"]))] + j_logs(j)) if k_]
                looks = []
                for (a, b, n, t, r) in texts:
                    if b <= obs or not any(k_ in t for k_ in keys):
                        continue
                    looks.append(b if re.match(r"\s*(?:cd\s+\S+\s*(?:&&|;)\s*)*sleep\s+\d", t) else max(a, obs))
                look = min(looks) if looks else obs
                fix_end = min(max(j["inputs_ready"], look), j["launch"])
                rel_ = max(first_in, obs + max(0.0, fix_end - look))
                j["regenerated"] = {"failure_observable": obs, "first_look": look, "fixed": fix_end, "release": rel_}
                j["inputs_ready"], j["inputs_how"] = rel_, "regenerated: a failure observable, plus the agent's own fix time"
            else:
                j["inputs_ready"], j["inputs_how"] = first_in, ("relaunched with its inputs unchanged: its first attempt's inputs"
                                                                if fails else "re-run without a failure: its first attempt's inputs")
        # first named: the agent's own output naming its script's directory or file
        key = _tail(os.path.dirname(j["script"])) or _tail(j["script"])
        named = [a for (a, b, n, t, r) in texts if key and key in t] + [a for a, s in said if key and key in s]
        j["first_named"] = min(named) if named else j["launch"]
        j["outputs"] = [_tail(o) for o in _outputs(j["argv"], j["launch_cmd"])]
    # dependencies: j's results read (after j ended) before i was first named; and the task's own `after` list, which
    # no schedule may break: a decision-bound confirm needs every screen's result whenever the agent wrote its script
    # (pkg-sc-q: the scripted reference read 1.28x its bound, the confirms released at their checkpoint's write, 419 s,
    # before the last screen ended at 664 s; the analysis, 2026-10-22 15:35Z)
    after = {j["job_id"]: j.get("after") or [] for j in jobs_l}
    end_of = {j["task_job"]: j["end"] for j in jobs if j["task_job"] and "#" not in str(j["id"])}
    for i in jobs:
        deps = []
        for j in jobs:
            if j is i or j["end"] >= i["launch"] or not j["outputs"]:
                continue
            reads = [b for (a, b, n, t, r) in texts if b >= j["end"] and b <= i["first_named"]
                     and any(o and (o in t or os.path.basename(o) in t) for o in j["outputs"])]
            if reads:
                deps.append((j["id"], j["end"], min(reads)))
        named = {d[0] for d in deps}
        deps += [(a, end_of[a], None) for a in after.get(i["task_job"], []) if a in end_of and a not in named]
        i["deps"] = deps
        i["release"] = max([i["inputs_ready"]] + [e for _, e, _ in deps])
        # relative to its dependencies: the agent's own latency after the
        # dependencies ended, kept; the dependencies' actual end, not. The absolute part is the inputs' write only when it
        # came before any dependency launched (the job was ready before its predecessors ran); an input written after a
        # dependency launched is that dependency's own output or the agent's work alongside it (Q-halving-sonnet-L0's
        # round 3 resumes from round 2's checkpoint, written by round 2 at 4,201 s, 2 s before its end), so it moves with
        # the dependency in the bound
        last_dep = max((e for _, e, _ in deps), default=None)
        launch_of = {j["id"]: j["launch"] for j in jobs}
        first_dep_launch = min((launch_of.get(d, e) for d, e, _ in deps), default=None)
        i["rel_deps"] = []
        if deps:
            lag = max(0.0, i["inputs_ready"] - last_dep)
            i["rel_deps"] = [(d, lag) for d, _, _ in deps]
            i["abs_release"] = i["inputs_ready"] if i["inputs_ready"] < first_dep_launch else t0
        else:
            i["abs_release"] = i["release"]
    rel = lambda x: (x - t0) * speed
    sj = [{"id": j["id"], "release": rel(j["abs_release"]), "run": j["run"] * speed, "width": j["width"], "launch": rel(j["launch"]),
           "rel_deps": [(d, lag * speed) for d, lag in j["rel_deps"]]} for j in jobs]
    mk, rule, starts = schedule(sj, gpus)
    # the interval form (the paper's figure): every idle GPU interval in the actual run, cut where the agent's activity
    # changes, with the released and unlaunched jobs that could have run there and what the agent was doing
    # busy intervals on the daemon's monotonic clock, shifted to wall time as idle.py does
    off = next((e["t_wall"] - e["t_mono"] for e in ev if e.get("t_mono") is not None and e.get("t_wall") is not None), 0.0)
    ivs = {g: [[a + off, b + off] for a, b in l] for g, l in score_run.busy_intervals(ev).items()}
    # modelled jobs busy on their GPU, as in idle.py
    idle.with_modelled_busy(run_dir, ev, ivs, gpus, st)
    t_end = max(j["end"] for j in jobs)
    a_end = st.get("harness_end_wall") or st.get("agent_end_wall") or st.get("teardown_wall") or t_end
    # cut where the agent's activity changes and where a job is released or launched, so an interval's could-run set
    # holds across it
    cuts = sorted({t0, t_end} | {x for (a, b, n, t, r) in texts for x in (a, b) if t0 < x < t_end}
                  | {x for j in jobs for x in (j["release"], j["launch"]) if t0 < x < t_end})
    def activity(m):
        if m >= a_end:
            return "ended", ""
        # the main agent's own activity: a subagent's foreground launch-and-wait is that subagent's, while the main
        # agent may be ended or idle (Q-prepP6-haiku-L2: six subagents each ran a job in the foreground)
        c = next(((n, t, b - a) for (a, b, n, t) in main_calls if a <= m < b), None)
        if c is None:
            sub = next((t for (a, b, n, t, r) in texts if a <= m < b and prep.GPU_LAUNCH_RE.search(t)), None)
            return ("subagent running a job", sub[:100]) if sub else ("model turn", "")
        n, t, dur = c
        # a long call that launches in the foreground and waits on it is the agent blocked, as in a sleep
        kind = ("sleeping" if (n == "Bash" and ACT_WAIT.search(t)) or t.startswith("wait ")
                or (prep.GPU_LAUNCH_RE.search(t) and dur > 30.0) else
                "preparing" if (n in ("Write", "Edit", "MultiEdit", "NotebookEdit") or (n == "Bash" and WRITE_RE.search(t)
                                and not prep.GPU_LAUNCH_RE.search(t))) else
                "launching" if prep.GPU_LAUNCH_RE.search(t) else "reading or running a command")
        return kind, t[:100]
    gaps, pieces = [], []
    for g in range(gpus):
        busy = sorted(ivs.get(g, []))
        free, cur = [], t0
        for a, b in busy:
            if a > cur:
                free.append((cur, min(a, t_end)))
            cur = max(cur, b)
        if cur < t_end:
            free.append((cur, t_end))
        for a, b in free:
            pts = [a] + [x for x in cuts if a < x < b] + [b]
            for x, y in zip(pts, pts[1:]):
                if y - x < 0.5:
                    continue
                pieces.append((x, y))          # the idle pieces idle_fillable_share reads, the same ones as above
                m = (x + y) / 2
                could = [j["id"] for j in jobs if j["release"] <= m < j["launch"]]
                kind, what = activity(m)
                if gaps and gaps[-1]["gpu"] == g and abs(gaps[-1]["_end"] - x) < 1e-6 and gaps[-1]["agent"] == kind and gaps[-1]["could_run"] == could:
                    gaps[-1]["_end"] = y
                else:
                    gaps.append({"gpu": g, "_start": x, "_end": y, "could_run": could, "agent": kind, "call": what})
    for x in gaps:
        x["start_s"], x["end_s"] = rel(x.pop("_start")), rel(x.pop("_end"))
        x["idle_s"] = x["end_s"] - x["start_s"]
    by_act = {}
    for x in gaps:
        if x["could_run"]:
            by_act[x["agent"]] = by_act.get(x["agent"], 0.0) + x["idle_s"]
    # the share of the box's GPU capacity lost while a runnable job waited, by agent activity: comparable across
    # packages of different length
    cap = gpus * rel(t_end)
    share = {k: v / cap for k, v in by_act.items()} if cap > 0 else {}
    # the floor, no schedule beats it: the critical path through the relative releases (each job's earliest end, its
    # dependencies' earliest ends plus the lags), and the work spread over the box from the earliest release
    by_id = {j["id"]: j for j in sj}
    ef = {}
    def earliest_end(i, seen=()):
        if i in ef:
            return ef[i]
        j = by_id[i]
        r = max([j["release"]] + [earliest_end(d, seen + (i,)) + lag for d, lag in j["rel_deps"] if d in by_id and d not in seen])
        ef[i] = r + j["run"]
        return ef[i]
    floor = max(max(earliest_end(j["id"]) for j in sj), min(j["release"] for j in sj) + sum(j["run"] * j["width"] for j in sj) / gpus)
    run_of = {j["id"]: j["run"] for j in sj}
    bound_release = {j["id"]: max([j["release"]] + [starts[d] + run_of[d] + lag for d, lag in j["rel_deps"] if d in starts])
                     for j in sj}
    actual = rel(max(j["end"] for j in jobs))
    # how late each job launched, and what the agent was in at its release
    out = []
    for j, s in zip(jobs, sj):
        r = j["release"]
        doing = next((n + ": " + t[:80] for (a, b, n, t, rr) in texts if a <= r <= b), "between calls")
        out.append({"job": j["id"], "width": j["width"], "release_s": rel(j["release"]), "inputs_ready_s": rel(j["inputs_ready"]),
                    "inputs_how": j["inputs_how"], "first_named_s": rel(j["first_named"]), "depends_on": [d[0] for d in j["deps"]],
                    # the release the bound schedule used: relative to its dependencies' ends in that schedule
                    "bound_release_s": bound_release[j["id"]], "agent_lag_after_deps_s": (j["rel_deps"][0][1] * speed) if j["rel_deps"] else None,
                    "launch_s": s["launch"], "late_s": s["launch"] - rel(j["release"]), "run_s": s["run"],
                    "bound_start_s": starts[j["id"]], "agent_at_release": doing,
                    "regenerated": None if not j["regenerated"] else {k: rel(v) for k, v in j["regenerated"].items()}})
    out.sort(key=lambda x: x["launch_s"])
    # hindsight_with_failures_s: the bound with every failed attempt of a completed job
    # as required work, charged to the environment. The attempt runs as long as it did, released when the job's first attempt
    # was; each relaunch is released at the previous attempt's end in the schedule, with no lag, so only the agent's lag after a
    # failure is waste (a fix it wrote included). Relative, as the dependencies are: an agent that launched the failing attempt late
    # keeps that lateness. The bound above treats every failed attempt as avoidable and is kept as it was
    sjf, failed_out = [], []
    for j, s in zip(jobs, sj):
        own = [] if "#" in str(j["id"]) else sorted((a for a in charged.get(j["id"], []) if a["start"] < j["launch"] - 1.0),
                                                    key=lambda a: a["start"])
        if not own:
            sjf.append(s)
            continue
        base = rel(min(j["abs_release"], j.get("first_in", j["abs_release"])))
        prev = None
        for n, a in enumerate(own, 1):
            fid = f"{j['id']}~failed{n}"
            sjf.append({"id": fid, "release": base, "run": (a["end"] - a["start"]) * speed, "width": a["width"] or j["width"],
                        "launch": rel(a["start"]), "rel_deps": list(s["rel_deps"]) if prev is None else [(prev, 0.0)]})
            failed_out.append({"job": j["id"], "attempt": fid, "start_s": rel(a["start"]), "end_s": rel(a["end"]),
                               "width": a["width"] or j["width"], "cause": a["cause"]})
            prev = fid
        sjf.append(dict(s, release=base, rel_deps=[(prev, 0.0)]))
    mkf = schedule(sjf, gpus) if failed_out else (mk, rule, starts)
    for x in failed_out:
        x["bound_start_s"] = (mkf[2].get(x["attempt"]) if mkf else None)
    # idle_fillable_share: per stretch of the actual run, the
    # free GPUs the waiting jobs could fill, the largest sum of their widths that fits in the free GPUs, so a free GPU no waiting
    # job fits counts nothing (an 8-wide job waiting on 7 free GPUs; two 3-wide jobs on 4 free fill 3); and a job is not waiting
    # while one of its own earlier attempts runs (a relaunched job's failed attempt holds its place). Same window, capacity and
    # releases and idle pieces as idle_with_released_job_share, so it never reads above it
    held = {j["id"]: ([] if "#" in str(j["id"]) else [(a, b) for a, b in tried.get(j["id"], []) if a < j["launch"] - 1.0])
            for j in jobs}
    # the free-GPU count over time from the idle pieces, by a sweep, split also where a held attempt starts or ends
    deltas = {}
    for a, b in pieces:
        deltas[a] = deltas.get(a, 0) + 1
        deltas[b] = deltas.get(b, 0) - 1
    for v in held.values():
        for a, b in v:
            for x in (a, b):
                if t0 < x < t_end:
                    deltas.setdefault(x, 0)
    pts = sorted(deltas)
    fill_s, fill_by, fill_iv, free = 0.0, {}, [], 0
    for x, y in zip(pts, pts[1:]):
        free += deltas[x]
        if y - x <= 1e-9:
            continue
        m = (x + y) / 2
        wait = [j for j in jobs if j["release"] <= m < j["launch"] and not any(a <= m < b for a, b in held[j["id"]])]
        if not free or not wait:
            continue
        reach = 1
        for j in wait:
            if j["width"] <= free:
                reach |= reach << j["width"]
        fill = (reach & ((1 << (free + 1)) - 1)).bit_length() - 1
        kind, _ = activity(m)
        fill_s += fill * (y - x) * speed
        if fill:
            fill_by[kind] = fill_by.get(kind, 0.0) + fill * (y - x) * speed
        ids = [j["id"] for j in wait]
        if fill_iv and abs(fill_iv[-1]["end_s"] - rel(x)) < 1e-6 and (fill_iv[-1]["free_gpus"], fill_iv[-1]["waiting"],
                                                                      fill_iv[-1]["fillable_gpus"], fill_iv[-1]["agent"]) == (free, ids, fill, kind):
            fill_iv[-1]["end_s"] = rel(y)
        else:
            fill_iv.append({"start_s": rel(x), "end_s": rel(y), "free_gpus": free, "waiting": ids, "fillable_gpus": fill, "agent": kind})
    t_score = {"hindsight_with_failures_s": mkf[0] if mkf else None, "hindsight_with_failures_rule": mkf[1] if mkf else None,
               "actual_over_hindsight_with_failures": (actual / mkf[0]) if mkf and mkf[0] else None,
               "failed_attempts": failed_out,
               "idle_fillable_gpu_s": fill_s, "idle_fillable_share": fill_s / cap if cap > 0 else 0.0,
               "idle_fillable_gpu_s_by_agent_activity": fill_by, "idle_fillable_intervals": fill_iv}
    return {"bound_s": mk, "bound_rule": rule, "floor_s": floor, "actual_s": actual,
            "actual_over_bound": actual / mk if mk else None, "jobs": out, "gpus": gpus,
            "idle_intervals": [x for x in gaps if x["idle_s"] >= 1.0],
            "idle_with_released_job_gpu_s_by_agent_activity": by_act,
            "idle_with_released_job_share_by_agent_activity": share,
            "idle_with_released_job_share": sum(share.values()) if share else 0.0,
            "busy_intervals_s": {g: [[rel(a), rel(b)] for a, b in ivs.get(g, [])] for g in range(gpus)},
            **t_score,
            "how": "completed launches, each released at max(its inputs' last write, the completion of every job whose "
                   "results the agent read before first naming it); the best of four list schedules on the box's GPUs"}

COLOURS = {"sleeping": "#d62728", "model turn": "#ff7f0e", "reading or running a command": "#9467bd",
           "preparing": "#2ca02c", "launching": "#17becf", "subagent running a job": "#8c564b", "ended": "#cfcfcf"}
BUSY = "#3c4650"

def svg(r, title=""):
    """One run's GPU lanes: busy grey; idle coloured by what the agent was doing, hatched where a released job waited;
    below, each job's wait from release to launch. Self-contained SVG, times in recorded s from the agent's start."""
    if not r.get("bound_s"):
        return None
    W, L, lane, top = 900, 70, 26, 34
    T = max(r["actual_s"], 1.0)
    x = lambda t: L + (W - L - 20) * max(0.0, t) / T
    g_ = r["gpus"]
    jobs = r["jobs"]
    H = top + g_ * lane + 24 + len(jobs) * 12 + 76
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica,Arial,sans-serif" font-size="11">',
         f'<rect x="0" y="0" width="{W}" height="{H}" fill="#fff"/>',
         '<defs><pattern id="h" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
         '<line x1="0" y1="0" x2="0" y2="6" stroke="#000" stroke-opacity="0.35" stroke-width="2"/></pattern></defs>',
         f'<text x="{L}" y="16" font-size="13" font-weight="bold">{title}</text>',
         f'<text x="{L}" y="28" fill="#444">actual {r["actual_s"]:.0f} s from agent start; hindsight bound {r["bound_s"]:.0f} s '
         f'({r["actual_over_bound"]:.2f}x); {100 * r.get("idle_with_released_job_share", 0):.1f}% of GPU capacity idle while a released job waited</text>']
    for g in range(g_):
        y = top + g * lane
        o.append(f'<text x="{L - 8}" y="{y + 16}" text-anchor="end">GPU {g}</text>')
        o.append(f'<rect x="{x(0)}" y="{y + 3}" width="{x(T) - x(0)}" height="{lane - 6}" fill="#f4f4f4"/>')
        merged = []
        for a, b in sorted(r["busy_intervals_s"].get(g, r["busy_intervals_s"].get(str(g), []))):
            if merged and a - merged[-1][1] < 2.0:      # a unit-to-unit gap under 2 s is drawn as busy
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        for a, b in merged:
            o.append(f'<rect x="{x(a):.1f}" y="{y + 3}" width="{max(0.5, x(b) - x(a)):.1f}" height="{lane - 6}" fill="{BUSY}"/>')
        for iv in r["idle_intervals"]:
            if iv["gpu"] != g or iv["end_s"] - iv["start_s"] < 2.0:
                continue
            c = COLOURS.get(iv["agent"], "#bbbbbb")
            o.append(f'<rect x="{x(iv["start_s"]):.1f}" y="{y + 3}" width="{max(0.5, x(iv["end_s"]) - x(iv["start_s"])):.1f}" '
                     f'height="{lane - 6}" fill="{c}" fill-opacity="{0.85 if iv["could_run"] else 0.25}"/>')
            if iv["could_run"]:
                o.append(f'<rect x="{x(iv["start_s"]):.1f}" y="{y + 3}" width="{max(0.5, x(iv["end_s"]) - x(iv["start_s"])):.1f}" '
                         f'height="{lane - 6}" fill="url(#h)"/>')
    y0 = top + g_ * lane + 14
    o.append(f'<text x="{L - 8}" y="{y0}" text-anchor="end" fill="#444">waits</text>')
    for k, j in enumerate(sorted(jobs, key=lambda j: j["release_s"])):
        y = y0 + 4 + k * 12
        o.append(f'<line x1="{x(j["release_s"]):.1f}" y1="{y + 4}" x2="{x(j["launch_s"]):.1f}" y2="{y + 4}" stroke="#d62728" stroke-width="3"/>')
        o.append(f'<circle cx="{x(j["launch_s"]):.1f}" cy="{y + 4}" r="2.5" fill="#000"/>')
        o.append(f'<text x="{x(j["launch_s"]) + 5:.1f}" y="{y + 8}" font-size="9" fill="#444">{str(j["job"])[:24]}</text>')
    yt = y0 + 8 + len(jobs) * 12
    for t in range(0, int(T) + 1, max(60, int(T / 8) // 60 * 60 or 60)):
        o.append(f'<line x1="{x(t):.1f}" y1="{top}" x2="{x(t):.1f}" y2="{yt}" stroke="#000" stroke-opacity="0.06"/>')
        o.append(f'<text x="{x(t):.1f}" y="{yt + 12}" text-anchor="middle" fill="#666">{t}</text>')
    # the legend in two rows, so it fits the width
    items = [("busy", BUSY, False)] + [(k, c, False) for k, c in COLOURS.items()] + [("a released job waiting (faint: none)", "#bbb", True)]
    lx, ly = L, yt + 22
    for name, c, hatch in items:
        w_ = 14 + int(6.6 * len(name)) + 14
        if lx + w_ > W - 10:
            lx, ly = L, ly + 16
        o.append(f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="{c}"/>' + (f'<rect x="{lx}" y="{ly}" width="10" height="10" fill="url(#h)"/>' if hatch else "")
                 + f'<text x="{lx + 14}" y="{ly + 9}">{name}</text>')
        lx += w_
    o.append("</svg>")
    return "\n".join(o)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--svg", default=None, help="write the run's GPU-lane figure here")
    a = ap.parse_args()
    r = analyse(a.run_dir)
    if a.svg and r.get("bound_s"):
        open(a.svg, "w").write(svg(r, os.path.basename(os.path.normpath(a.run_dir))))
    if a.json:
        print(json.dumps(r, indent=1)); return
    if r.get("bound_s") is None:
        print(f"hindsight       - ({r.get('why')})"); return
    print(f"hindsight       bound {r['bound_s']:.0f} s ({r['bound_rule']}), floor {r['floor_s']:.0f} s, actual {r['actual_s']:.0f} s "
          f"from agent start: {r['actual_over_bound']:.2f}x the bound")
    print(f"  {'job':28s} {'width':>5s} {'release':>8s} {'launch':>8s} {'late':>7s} {'run':>6s}  depends on / at release")
    for j in r["jobs"]:
        print(f"  {str(j['job'])[:28]:28s} {j['width']:5d} {j['release_s']:8.0f} {j['launch_s']:8.0f} {j['late_s']:7.0f} {j['run_s']:6.0f}  "
              f"{','.join(j['depends_on']) or '-'} / {j['agent_at_release'][:60]}")

if __name__ == "__main__":
    main()
