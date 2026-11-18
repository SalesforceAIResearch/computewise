"""Preparation in a run: when the agent prepared each job, how long the GPUs sat idle around it, and what makespan
its own preparation allowed (the study, 2026-10-20, "preparation packages"). Labels, never gates.

A task job that carries `prep` (gpu-tasks: kind code | data, paths, produces, cpu_step) must be prepared before it
can run; a job without `prep` is prepared from the start. Per job:

  prep calls      the agent's tool calls whose input names one of the job's prep.paths (a Bash command, or the
                  file_path of Write, Edit, MultiEdit, Read and NotebookEdit; other tools' whole input), up to the
                  job's launch. The call that launched it (a Bash command with torchrun, accelerate launch, python,
                  nohup, bash or sh that names the job's script) is not prep. A path matches with or without its
                  trailing slash ("cd runs/a && ..."), never as a prefix of a longer name.
  cpu steps       for kind data: the box's processes (resources.jsonl procs) whose cmdline holds prep.cpu_step and
                  one of the job's paths, from start to exit.
  prep start      the start of the model turn that issued the job's first prep call.
  prep end        the latest of: the mtime of each prep.produces file in the workspace manifest, when it is not after
                  the launch (an edit after launch is a fix, not prep); the end of the last prep call; the exit of
                  the last cpu step before launch. Which one set it is recorded.
  ready           from max(prep end, the end of every job in its `after`) to its launch.
  prep busy       the agent's own time on the job: each prep call plus its share of the turn that issued it, split
                  evenly when a turn issues several calls and a call names several jobs.

Over the run's window (idle.py's: agent start to the last job's end, or to the horizon):

  ready_but_idle_gpu_s            idle GPU-seconds while at least one job was ready and not launched;
                                  usable_gpu_s caps each instant at the GPUs the ready jobs would take
  waiting_instead_of_preparing    idle GPU-seconds while the agent was in a wait, sleep or poll call (idle.py's
                                  wait_or_poll) and some job needing prep was neither prepared nor launched;
                                  usable_gpu_s caps each instant at the GPUs those jobs would take
  prep_aware_bound                the makespan from agent start had the agent prepared its jobs back to back in its
                                  own order and with its own prep busy times, starting when it actually began
                                  preparing, and launched each the moment it was ready (cpu steps run beside the
                                  agent), each job taking the GPUs and the time it took in the run (launch to end),
                                  placed first come first served on free GPUs. It drops the agent's time on
                                  anything else, so it is a floor for this agent's own preparation, not a target.
  agent_busy_vs_job               median prep busy per job beside the median job length (paced GPU seconds)

All times are reported in recorded seconds (wall x the run's speed), from the agent's start.
"""
import json, os, re, statistics

LAUNCH_RE = re.compile(r"(^|[;&|\s(])(torchrun|accelerate\s+launch|python3?|nohup|bash|sh)\b")
GPU_LAUNCH_RE = re.compile(r"(^|[;&|\s(])(torchrun|accelerate\s+launch)\b|torch\.distributed\.run")
# a call that blocks on something else (a sleep, the shell's wait, watch, tail -f); not the scorer's wider WAIT_RE,
# which also takes a trailing `nvidia-smi -L` (Q-prepP10-haiku-L0's generator call ends with one)
BLOCKING_WAIT_RE = re.compile(r"(^|[;&|\s(])(sleep\s+\d|wait\b|watch\b|tail\s+-[fF])")
# a command that only looks (after any leading cd): it cannot have written a job's file
LIST_RE = re.compile(r"^\s*(?:cd\s+\S+\s*(?:&&|;)\s*)*(?:ls|cat|grep|egrep|rg|find|head|tail|wc|pgrep|ps|sed\s+-n|awk|stat|du|diff|"
                     r"tree|echo|printf|nvidia-smi|sleep|wait|test|file|md5sum|sha256sum)\b")
FILE_TOOLS = ("Write", "Edit", "MultiEdit", "Read", "NotebookEdit")
WRITE_FIELDS = ("content", "new_string", "old_string", "edits", "new_source")

def _ts(s):
    from datetime import datetime
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def calls_with_inputs(recs, open_until=None):
    """Tool calls as (start, end, name, input, turn_start, turn_id, calls_in_turn, result_text), timed as idle.activity() times
    them: a call starts at its assistant record and ends at its result; a call with no result is dropped unless
    open_until is given. turn_start is when the issuing model turn could start (the end of its input)."""
    calls, results, last_input_end, first_input, turn_calls = {}, {}, None, {}, {}
    for r in recs:
        t = _ts(r["timestamp"])
        typ = r.get("type")
        if typ == "assistant":
            msg = r.get("message") or {}
            mid = msg.get("id") or r.get("uuid")
            first_input.setdefault(mid, last_input_end if last_input_end is not None else t)
            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b["id"]] = (t, b.get("name"), b.get("input") or {}, mid)
                    turn_calls.setdefault(mid, []).append(b["id"])
        elif typ in ("user", "system", "queue-operation", "attachment"):
            msg = r.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            for b in content or [] if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    rc = b.get("content")
                    txt = rc if isinstance(rc, str) else " ".join(x.get("text", "") for x in rc or [] if isinstance(x, dict))
                    results[b.get("tool_use_id")] = (t, txt[:20000])
            last_input_end = t
    out = []
    for tid, (t0, name, inp, mid) in calls.items():
        t1, res = results.get(tid, (None, ""))
        if t1 is None and open_until is not None and open_until > t0:
            t1 = open_until
        if t1 is None or t1 < t0:
            continue
        out.append((t0, t1, name, inp, first_input[mid], mid, len(turn_calls[mid]), res))
    # by time only: calls with the same start and end (a scripted agent's, logged in one instant) must never fall
    # through to comparing their input dicts (the scheduler probe, 2026-10-21: TypeError)
    return sorted(out, key=lambda c: (c[0], c[1]))

def call_text(name, inp):
    if not isinstance(inp, dict):
        return str(inp)
    if name == "Bash":
        return inp.get("command") or ""
    if name in FILE_TOOLS:
        return str(inp.get("file_path") or inp.get("notebook_path") or "")
    return json.dumps({k: v for k, v in inp.items() if k not in WRITE_FIELDS})

def path_re(p):
    """A job path matches with or without its trailing slash, never as a prefix of a longer name."""
    return re.compile(re.escape(p.rstrip("/")) + r"(?![A-Za-z0-9_.\-])")

def _manifest_mtimes(run_dir):
    p = os.path.join(run_dir, "workspace", "manifest.jsonl")
    out = {}
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("path") and isinstance(m.get("mtime"), (int, float)):
                out[m["path"]] = float(m["mtime"])
    return out

def _procs(run_dir, full=False):
    """[(start, end, cmdline)] for the box's processes, from resources.jsonl procs lines (isolation's sampler); with
    full=True, [(start, end, cmdline, pid, ppid)]."""
    p = os.path.join(run_dir, "resources.jsonl")
    if not os.path.exists(p):
        return []
    seen, gone, last, parent = {}, {}, None, {}
    for line in open(p, errors="replace"):
        try:
            l = json.loads(line)
        except ValueError:
            continue
        if l.get("kind") != "procs":
            continue
        try:
            t = _ts(l["t_utc"])
        except (KeyError, ValueError):
            continue
        last = t
        live = set()
        for q in (l.get("procs") if l.get("full") else l.get("started")) or []:
            if isinstance(q, dict) and q.get("pid") is not None:
                k = (int(q["pid"]), q.get("cmdline") or "")
                seen.setdefault(k, t)
                parent.setdefault(k, q.get("ppid"))
                live.add(k)
        if l.get("full"):
            for k in list(seen):
                if k not in live and k not in gone:
                    gone[k] = t
        for q in l.get("exited") or []:
            pid = int(q["pid"]) if isinstance(q, dict) else int(q)
            for k in seen:
                if k[0] == pid and k not in gone:
                    gone[k] = t
    if full:
        return [(t0, gone.get(k, last), k[1], k[0], parent.get(k)) for k, t0 in seen.items()]
    return [(t0, gone.get(k, last), k[1]) for k, t0 in seen.items()]

def data_step_cores(run_dir, shared_pats):
    """Whether the agent's data step used more than one core: the step's processes are the
    python processes whose command line names one of the step's paths, and all their descendants (a multiprocessing
    pool's workers, forked or spawned); the peak is the most of them alive at once, from the box's process records
    (sampled each second, so a worker living under a second can be missed). A label, never a void."""
    procs = _procs(run_dir, full=True)
    if not procs:
        return {"peak_concurrent": None, "multi_core": None, "why": "no process records"}
    roots = {pid for a, b, cmd, pid, ppid in procs if re.search(r"\bpython", cmd) and any(p.search(cmd) for p in shared_pats)}
    members, grew = set(roots), True
    while grew:
        grew = False
        for a, b, cmd, pid, ppid in procs:
            if pid not in members and ppid in members:
                members.add(pid); grew = True
    ivs = [(a, b) for a, b, cmd, pid, ppid in procs if pid in members and re.search(r"python", cmd) and b is not None]
    if not ivs:
        return {"peak_concurrent": 0, "multi_core": None, "why": "no process of the data step was recorded"}
    pts = sorted([(a, 1) for a, b in ivs] + [(b, -1) for a, b in ivs], key=lambda x: (x[0], x[1]))
    cur = peak = 0
    for _, d in pts:
        cur += d
        peak = max(peak, cur)
    return {"processes": len(ivs), "peak_concurrent": peak, "multi_core": peak > 1,
            "start": min(a for a, b in ivs), "end": max(b for a, b in ivs),
            "how": "python processes naming the data step's paths, with their descendants, from resources.jsonl procs"}

def _segments(points, w0, w1):
    pts = sorted({w0, w1, *(x for x in points if w0 < x < w1)})
    return list(zip(pts, pts[1:]))

def analyse(run_dir, task, recs, ivs, gpus, w0, w1, speed, launches, done_at, gpus_by_job, run_s_by_job,
            open_until=None, wait_re=None):
    jobs = (task or {}).get("jobs") or []
    need = {j["job_id"]: j["prep"] for j in jobs if isinstance(j.get("prep"), dict)}
    if not need:
        return None
    # The agent may hand preparation to subagents (Q-prepP6-haiku-L0: six background subagents, one per setup), whose
    # calls are in their own transcripts (session/**/subagents/*.jsonl). Every preparer's calls count as prep; each
    # call carries its preparer ("main" or the subagent's file), and the bound gives each preparer its own lane.
    calls = [c + ("main",) for c in calls_with_inputs(recs, open_until)]
    import glob as _glob
    for f in sorted(_glob.glob(os.path.join(run_dir, "transcript", "session", "**", "subagents", "*.jsonl"), recursive=True)):
        sub = []
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("timestamp"):
                sub.append(r)
        sub.sort(key=lambda r: _ts(r["timestamp"]))
        calls += [c + (os.path.basename(f)[:-6],) for c in calls_with_inputs(sub, open_until)]
    calls.sort(key=lambda c: (c[0], c[1]))
    mtimes = _manifest_mtimes(run_dir)
    procs = _procs(run_dir)
    pats = {jid: [path_re(p) for p in (pr.get("paths") or [f"runs/{jid}/"])] for jid, pr in need.items()}
    # A shared data step (task.data_prep, gpu-tasks' data-preparation packages): every job marked `shared` is prepared
    # by the one step, so a call or process naming any of its paths (the step's script, its raw input, the reference
    # copy's path, the shard directory) is prep for all of them, its time split evenly
    dp = (task or {}).get("data_prep") if isinstance((task or {}).get("data_prep"), dict) else None
    shared_pats = [path_re(x) for x in ([dp.get(k) for k in ("script", "raw", "scripted_write", "produces")] if dp else [])
                   if isinstance(x, str) and x]
    for jid, pr in need.items():
        if pr.get("shared") and shared_pats:
            pats[jid] = pats[jid] + shared_pats
    # A call that names none of a job's paths can still have written its file (a generator: Q-prepP10-haiku-L0's
    # compose_quick.py wrote all ten scripts). Such a call is prep for a job when its own output names one of the job's
    # produces files, or when it was running at the moment the workspace manifest says one of them was last written
    #.
    # Either way only a call that could have written it counts: a Bash call that is not a wait, a GPU launch or a
    # listing. With subagents preparing in parallel, other calls are always running (Q-prepP6-haiku-L0-v2: another
    # subagent's `sleep 600`, a torchrun, a Write of another file), so at a write's mtime only the latest-started such
    # call is credited, and none when a call naming the job's paths was running then (that call is the writer).
    def may_write(name, text):
        return name == "Bash" and not (BLOCKING_WAIT_RE.search(text) or GPU_LAUNCH_RE.search(text) or LIST_RE.search(text))
    texts = [call_text(c[2], c[3]) for c in calls]
    named = {jid: [any(p.search(t) for p in pats[jid]) for t in texts] for jid in need}
    gen = {}
    for jid, pr in need.items():
        prod_re = [path_re(f) for f in pr.get("produces") or []]
        got = {i for i, c in enumerate(calls) if not named[jid][i] and may_write(c[2], texts[i])
               and any(p.search(c[7] or "") for p in prod_re)}
        for m in (m for f in pr.get("produces") or [] for k, m in mtimes.items() if k == f or k.endswith("/" + f)):
            # the manifest's mtimes are whole seconds, so the write was in [m, m + 1): inside a call when m <= its end
            # and m + 1 > its start
            live = [i for i, c in enumerate(calls) if c[0] - 1.0 < m <= c[1]]
            if any(named[jid][i] for i in live):
                continue
            cand = [i for i in live if may_write(calls[i][2], texts[i])]
            if cand:
                got.add(max(cand, key=lambda i: calls[i][0]))
        gen[jid] = got
    per = {}
    for jid, pr in need.items():
        L = launches.get(jid)
        scripts = [os.path.basename(p) for p in pr.get("produces") or []]
        mine = []
        for i, c in enumerate(calls):
            t0, t1, name, inp, turn_s, mid, n_in_turn, res, who = c
            text = texts[i]
            if not (named[jid][i] or i in gen[jid]):
                continue
            if L is not None and t0 > L:
                continue
            if name == "Bash" and LAUNCH_RE.search(text) and (any(s in text for s in scripts) if scripts else ".py" in text) \
                    and (L is None or t1 >= L - 5.0):
                continue        # the call that launched the job (it ends at or just before the launch); a write it
                                # made first still sets prep end, by the produces mtime
            # its time is shared by every job it prepared
            n_jobs = sum(1 for k in need if named[k][i] or i in gen[k])
            mine.append((c, n_jobs))
        busy, by_who = 0.0, {}
        for (t0, t1, name, inp, turn_s, mid, n_in_turn, res, who), n_jobs in mine:
            # a call's time counts as prep only up to the job's launch (a call that writes the script and then launches
            # and waits on it, Q-prepP10-haiku-L0's regeneration, is not 563 s of preparation)
            end = min(t1, L) if L is not None else t1
            b = (max(0.0, end - t0) + max(0.0, t0 - turn_s) / n_in_turn) / n_jobs
            busy += b
            by_who[who] = by_who.get(who, 0.0) + b
        preparer = max(by_who, key=by_who.get) if by_who else None
        steps = []
        if pr.get("cpu_step"):
            steps = [(a, b) for a, b, cmd in procs if pr["cpu_step"] in cmd and any(p.search(cmd) for p in pats[jid])
                     and (L is None or a <= L)]
        elif pr.get("shared") and shared_pats:
            # the shared step's processes: a python (or shell) process naming one of the step's paths, short-lived
            # shells excluded by requiring a python interpreter in the command line
            steps = [(a, b) for a, b, cmd in procs if re.search(r"\bpython", cmd) and any(p.search(cmd) for p in shared_pats)
                     and (L is None or a <= L)]
        ends, how = [], []
        if mine:
            ends.append(max(min(c[1], L) if L is not None else c[1] for c, _ in mine)); how.append("last prep call")
        prod = [m for p in pr.get("produces") or [] for k, m in mtimes.items() if k == p or k.endswith("/" + p)]
        prod = [m for m in prod if L is None or m <= L + 1.0]
        if prod:
            ends.append(max(prod)); how.append("produces mtime")
        if steps:
            ends.append(max(b for a, b in steps)); how.append("cpu step exit")
        start = min([c[4] for c, _ in mine] + [a for a, b in steps], default=None)
        end = max(ends) if ends else None
        set_by = how[ends.index(end)] if ends else None
        after = next((j.get("after") or [] for j in jobs if j["job_id"] == jid), [])
        deps = [done_at.get(a) for a in after]
        deps_done = None if any(d is None for d in deps) else max(deps, default=None)
        ready = None if end is None else max([end] + ([deps_done] if deps_done is not None else []))
        if after and deps_done is None:
            ready = None
        per[jid] = {"kind": pr.get("kind"), "shared": bool(pr.get("shared")), "prep_calls": len(mine), "preparer": preparer, "prep_start": start, "prep_end": end,
                    "prep_end_set_by": set_by, "ready": ready, "launch": L, "done": done_at.get(jid),
                    "prep_busy": busy, "cpu_steps": len(steps), "cpu_step_s": sum(b - a for a, b in steps)}
    # jobs without prep are ready from the start (once their `after` jobs are done)
    for j in jobs:
        if j["job_id"] in need:
            continue
        deps = [done_at.get(a) for a in j.get("after") or []]
        per[j["job_id"]] = {"kind": None, "shared": False, "prep_calls": 0, "preparer": None, "prep_start": None, "prep_end": w0, "prep_end_set_by": "no prep",
                            "ready": None if any(d is None for d in deps) else max([w0] + deps),
                            "launch": launches.get(j["job_id"]), "done": done_at.get(j["job_id"]), "prep_busy": 0.0,
                            "cpu_steps": 0, "cpu_step_s": 0.0}

    # the sweeps: idle GPUs, the ready set, the agent's waits, the unprepared set
    def busy_at(a, b):
        m = (a + b) / 2
        return sum(1 for l in ivs.values() if any(x <= m < y for x, y in l))
    # the main agent's waits only: a subagent polling its own job while others prepare is not the agent idling
    waits = [(c[0], c[1]) for c in calls if c[8] == "main" and c[2] == "Bash" and wait_re
             and wait_re.search(call_text("Bash", c[3])) and not (isinstance(c[3], dict) and c[3].get("run_in_background"))]
    pts = [x for l in ivs.values() for a, b in l for x in (a, b)] + [x for a, b in waits for x in (a, b)]
    for v in per.values():
        pts += [x for x in (v["ready"], v["launch"], v["prep_end"]) if x is not None]
    rbi = usable = wip = wip_usable = rbi_whole = wip_whole = 0.0
    for a, b in _segments(pts, w0, w1):
        m = (a + b) / 2
        idle = gpus - busy_at(a, b)
        if idle <= 0:
            continue
        ready = [jid for jid, v in per.items() if v["ready"] is not None and v["ready"] <= m and (v["launch"] is None or m < v["launch"])]
        if ready:
            rbi += idle * (b - a)
            usable += min(idle, sum(gpus_by_job.get(jid, 1) for jid in ready)) * (b - a)
            rbi_whole += (b - a) if idle == gpus else 0.0
        # a job launched with no observed preparation (the agent wrote its script by a route the paths do not name,
        # e.g. a generator script that writes every job's file: Q-prepP10-haiku-L0) cannot be judged unprepared; only
        # a job whose prep end is known, or one never launched, counts
        unprepared = [jid for jid in need if ((per[jid]["prep_end"] is not None and per[jid]["prep_end"] > m)
                                              or (per[jid]["prep_end"] is None and per[jid]["launch"] is None))
                      and (per[jid]["launch"] is None or m < per[jid]["launch"])]
        if unprepared and any(x <= m < y for x, y in waits):
            wip += idle * (b - a)
            wip_usable += min(idle, sum(gpus_by_job.get(jid, 1) for jid in unprepared)) * (b - a)
            wip_whole += (b - a) if idle == gpus else 0.0

    bound, why = prep_aware_bound(per, need, gpus, w0, gpus_by_job, run_s_by_job, jobs)
    data_identical = data_check(run_dir, dp, launches) if dp else None
    cores = data_step_cores(run_dir, shared_pats) if dp and shared_pats else None
    if cores and cores.get("start") is not None:
        cores["start"], cores["end"] = (cores["start"] - w0) * speed, (cores["end"] - w0) * speed
    jl = [v for v in run_s_by_job.values() if v]
    pb = [per[j]["prep_busy"] for j in need if per[j]["prep_calls"]]
    rec = lambda t: None if t is None else (t - w0) * speed
    unobserved = sorted(j for j in need if per[j]["prep_end"] is None and per[j]["launch"] is not None)
    return {
        "jobs_needing_prep": len(need),
        "prep_unobserved_jobs": unobserved,
        "prep_unobserved_why": ("launched with no preparation the scorer could tie to it (no call naming its prep.paths, "
                                "no produces mtime before its launch, e.g. a generator script writing every job's file); "
                                "left out of waiting-instead-of-preparing and ready-but-idle") if unobserved else None,
        "preparers": sorted({v["preparer"] for v in per.values() if v.get("preparer")}),
        "by_job_recorded_s": {jid: {"kind": v["kind"], "preparer": v.get("preparer"), "prep_calls": v["prep_calls"], "prep_start": rec(v["prep_start"]),
                                    "prep_end": rec(v["prep_end"]), "prep_end_set_by": v["prep_end_set_by"],
                                    "ready": rec(v["ready"]), "launch": rec(v["launch"]), "done": rec(v["done"]),
                                    "ready_wait": None if v["ready"] is None or v["launch"] is None
                                    else (v["launch"] - v["ready"]) * speed,
                                    "prep_busy": v["prep_busy"] * speed, "cpu_step_s": v["cpu_step_s"] * speed}
                              for jid, v in sorted(per.items(), key=lambda kv: (kv[1]["launch"] is None, kv[1]["launch"] or 0))},
        "ready_but_idle_recorded_gpu_s": rbi * speed,
        "ready_but_idle_usable_recorded_gpu_s": usable * speed,
        "ready_but_idle_whole_box_recorded_s": rbi_whole * speed,
        "waiting_instead_of_preparing_recorded_gpu_s": wip * speed,
        "waiting_instead_of_preparing_usable_recorded_gpu_s": wip_usable * speed,
        "waiting_instead_of_preparing_whole_box_recorded_s": wip_whole * speed,
        "prep_aware_bound_recorded_s": None if bound is None else (bound - w0) * speed,
        "prep_aware_bound_why": why,
        "data_identical_by_job": data_identical,
        "data_step_cores": cores,
        "agent_busy_vs_job": {"median_prep_busy_s": (statistics.median(pb) * speed) if pb else None,
                              "median_job_s": (statistics.median(jl) * speed) if jl else None,
                              "ratio": (statistics.median(pb) / statistics.median(jl)) if pb and jl else None},
    }

def data_check(run_dir, dp, launches):
    """Per job, whether every data file it read is byte-identical to the recorded runs' shards (task.data_prep:
    needs, job -> files; shard_sha256, file name -> sha256): a label, never a void (the study, 2026-10-21; gpu-mock
    serves the recorded values whatever the bytes). Hashes come from the workspace manifest's `sha256`, taken at run
    end; a file whose mtime is no later than the job's launch held that content when the job started, and one modified
    after its launch is marked so. Without hashes in the manifest the label is None, with the reason."""
    needs = dict(dp.get("needs") or [])
    want = dict(dp.get("shard_sha256") or [])
    man = {}
    p = os.path.join(run_dir, "workspace", "manifest.jsonl")
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("path"):
                man[m["path"]] = m
    out = {}
    for jid, files in needs.items():
        L = launches.get(jid)
        diff, missing, unhashed, later = [], [], [], []
        for f in files:
            m = next((v for k, v in man.items() if k == f or k.endswith("/" + f)), None)
            if m is None:
                missing.append(f); continue
            if not m.get("sha256"):
                unhashed.append(f); continue
            if L is not None and isinstance(m.get("mtime"), (int, float)) and m["mtime"] > L + 1.0:
                later.append(f)
            if want.get(os.path.basename(f)) and m["sha256"] != want[os.path.basename(f)]:
                diff.append(f)
        out[jid] = {"data_identical": None if (unhashed or missing) and not diff else not diff,
                    "differing": diff, "missing_at_run_end": missing, "not_hashed": unhashed,
                    "modified_after_launch": later,
                    "hashed_at": "run end (mtime no later than launch, so the content at job start)" if not later
                    else "run end; some files changed after the job launched, so their content at job start is unknown"}
    return out

def prep_aware_bound(per, need, gpus, w0, gpus_by_job, run_s_by_job, jobs):
    """(bound end wall time, why): the agent prepares back to back in its own order (by prep start) from its first
    prep start, each job taking its own prep busy time, then its cpu steps beside the agent; each job runs for its
    run's launch-to-end time on its run's GPU count, first come first served, after its `after` jobs end."""
    missing = [j for j in need if per[j]["prep_calls"] == 0 and per[j]["cpu_steps"] == 0]
    if missing:
        return None, f"the agent never prepared {len(missing)} job(s): {', '.join(sorted(missing))[:200]}"
    unrun = [j for j in per if not run_s_by_job.get(j)]
    if unrun:
        return None, f"{len(unrun)} job(s) never ran to the end, so their run time is unknown: {', '.join(sorted(unrun))[:200]}"
    order = sorted(need, key=lambda j: per[j]["prep_start"] if per[j]["prep_start"] is not None else per[j]["prep_end"])
    ready = {j: w0 for j in per if j not in need}
    shared = [j for j in order if per[j]["shared"]]
    # one lane per preparer (the main agent, each subagent), back to back in its own order from its own first prep
    # start; a shared step is one block in the lane of whoever did most of it
    lanes = {}
    for j in order:
        lanes.setdefault("shared" if per[j]["shared"] else (per[j]["preparer"] or "main"), []).append(j)
    for lane, js in lanes.items():
        starts = [per[j]["prep_start"] for j in js if per[j]["prep_start"] is not None]
        t = min(starts) if starts else w0
        if lane == "shared":
            # one shared step readies all its jobs together: its whole busy time (split per job in the readout,
            # summed back here), then its processes' run beside the agent
            t += sum(per[k]["prep_busy"] for k in js)
            r = t + max(per[k]["cpu_step_s"] for k in js)
            for k in js:
                ready[k] = r
            continue
        for j in js:
            t += per[j]["prep_busy"]
            ready[j] = t + per[j]["cpu_step_s"]
    after = {j["job_id"]: list(j.get("after") or []) for j in [*jobs]}
    def place(key):
        free = [w0] * int(gpus)
        end = {}
        pending = sorted(per, key=key)
        while pending:
            progressed = False
            for j in list(pending):
                if any(a not in end for a in after.get(j, [])):
                    continue
                w = max(1, int(gpus_by_job.get(j, 1)))
                free.sort()
                start = max(ready[j], free[w - 1], max((end[a] for a in after.get(j, [])), default=w0))
                for k in range(w):
                    free[k] = start + run_s_by_job[j]
                end[j] = start + run_s_by_job[j]
                pending.remove(j)
                progressed = True
                break
            if not progressed:
                return None
        return max(end.values())
    # the best of several placements, not first come first served alone: an agent that oversubscribed its GPUs and
    # gave the longest job a GPU early (Q-prepP6-haiku-L2) can beat FCFS by ready time, and a bound it beats is no bound
    launch = {j: (per[j].get("launch") if per[j].get("launch") is not None else ready[j]) for j in per}
    tries = {"ready time": lambda j: (ready[j], j), "longest first": lambda j: (-run_s_by_job[j], ready[j], j),
             "the agent's launch order": lambda j: (launch[j], j)}
    got = {k: place(f) for k, f in tries.items()}
    if any(v is None for v in got.values()):
        return None, "the jobs' `after` order has a cycle or names a missing job"
    best = min(got, key=got.get)
    n_lanes = len([k for k in lanes if k != "shared"]) + (1 if "shared" in lanes else 0)
    return got[best], ("own prep order and busy times, back to back from the first prep start, one lane per "
                       f"preparer ({n_lanes}); each job's solo run time (its run less any wait behind another job on "
                       f"its GPU) and GPU count from the run; the best of three placements (by {best})")
