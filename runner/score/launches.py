"""Launches that died before reaching a GPU (the study, 2026-10-21: mechanism evidence, "blind sleeps hide broken
launches"). A label, never a score: such an attempt has no rank in the occupancy log, so the idle time around it is
already the agent's, and this only says why.

usage: python3 -I launches.py <run_dir> [--json]

An attempt is a launch of a task job's script in the box's process records (resources.jsonl procs): a torchrun or
accelerate launcher, or a python process running the script directly whose parent is no such launcher. It is tied to
a task job by the scorer's own rules (idle.task_job_of: output dir, then the options that tell the jobs apart). It
failed before the GPU when no rank of that task job said hello to the daemon (job_hello's process start) while the
attempt was alive. A `--help` or `-h` call is no attempt. Each failure records the job, its start and end (recorded
seconds from the agent's start), the log the shell sent it to and that log's tail if the run kept the file (a later
attempt writing to the same log overwrites it, which is then said), and the time until the agent next launched the
same job. The process records are sampled each second, so an attempt that dies inside a second can be missed, and a
launch that fails before any python starts (a shell syntax error, a missing file) never appears in them.

Those are read from the agent's own calls instead (the analysis 2026-11-01 15:48 MST, G-rnd8-haiku-L0's five and
G-oom-opus-L0-r2's chained rerun: no daemon hello, so failed_attempts [] and this field 0): the shell's own error line,
"<shell or script>: line N: <path>: No such file or directory" (or Permission denied, Is a directory, command not found), in
a Bash call's result or in the output file of a background call (workspace/tmp/claude-*/.../tasks/<id>.output). The failed
command line is the one that names the path: the call's own command (which must name it: a later `cat` of a background
call's output shows an error it did not run), or the script the error names (as the run kept it), at line N, else the one
line that names the path, else the one that redirects into it (the harness wraps a call, so N is offset). Its task job, by
the first rule that applies: the segment's own script and options (idle.task_job_of, or the job launching that .sh for the
KV packages), and a script no owed job runs ties to nothing; else the one owed job the error's path names by setup, seed
and resume step (a wrapper's log built from its arguments: G-rnd8-haiku-L0's triage/logs/radial-damp-seed8-
tocheckpoints/seed8-step875.pt.log); else the jobs the same wrapper ran the next time the agent ran it (G-oom-opus-L0-r2's
./run_mb64.sh, from the process records). A redirect that failed opens no file, so the command never ran: such a job failed
before reaching a GPU and is listed with source "shell_error" and its tie, at the call's start, or for a background call
its output file's last write (an upper bound: the error is written as the command fails). An error that ties to no owed job
(an ambiguous job, an unneeded one, a smoke line whose offset is unclear) is counted in shell_errors_untied, never guessed;
a job a daemon rank started within that second is not listed (it ran).
"""
import argparse, json, os, re, shlex, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idle, prep, score_run

CMDLINE_CAP = 380      # a process record's command line at or beyond this length may have been cut short

def _norm(toks):
    return [os.path.normpath(t) if t.endswith(".py") else t for t in toks]

LAUNCHER_RE = re.compile(r"(^|/)(torchrun|accelerate)(\s|$)|torch\.distributed\.run")
REDIRECT_RE = re.compile(r">>?\s*(\S+\.(?:log|txt|out))")

def _argv(cmd):
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    i = next((k for k, t in enumerate(toks) if t.endswith(".py") and not t.endswith(("torchrun", "launch.py"))), None)
    return None if i is None else toks[i:]

def scan(run_dir):
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    jobs = task.get("jobs") or []
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    t0 = st.get("agent_start_wall")
    procs = prep._procs(run_dir, full=True)
    if not jobs or not procs or t0 is None:
        return {"launches_failed_before_gpu": None, "why": "no task jobs, no process records or no agent start"}
    speed = float(idle.run_inputs(run_dir).get("speed") or 1.0)
    by_pid = {pid: (a, b, cmd, ppid) for a, b, cmd, pid, ppid in procs}
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    idle.attach_decisions(task, [e for e in ev if e["kind"] == "lookup"])
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hellos = [(idle.task_job_of(e, lk.get(e["job"]), jobs), idle.job_start(e), list(e.get("argv") or []))
              for e in ev if e["kind"] == "job_hello"]
    attempts = []
    for a, b, cmd, pid, ppid in procs:
        argv = _argv(cmd)
        if not argv or re.search(r"(^|\s)(--help|-h)(\s|$)", cmd):
            continue
        launcher = bool(LAUNCHER_RE.search(cmd))
        first = os.path.basename((cmd.split() or [""])[0])
        if not launcher and not first.startswith("python"):
            continue                      # a shell wrapping the command (sh -c ...): its python child is the attempt
        parent = by_pid.get(ppid)
        if parent and (LAUNCHER_RE.search(parent[2]) or (_argv(parent[2]) and os.path.basename((parent[2].split() or [""])[0]).startswith("python"))):
            continue                      # a rank under a launcher, or a worker forked by a running script
        # The box's process records cut long command lines short (an accelerate launch lost its --loss_type), so a
        # tie from the attempt's own argv can be wrong. A rank whose full argv starts with the attempt's argv (its last
        # token dropped, it may be cut mid-word) and that started while the attempt was alive is that attempt's rank:
        # the attempt reached the GPU, and it is that rank's task job
        head = argv[:-1] if len(cmd) >= CMDLINE_CAP else argv
        rank = next(((tj, t) for tj, t, hv in hellos if a - 2 <= t <= (b if b is not None else a) + 2
                     and _norm(hv[:len(head)]) == _norm(head)), None)
        tj = rank[0] if rank else idle.task_job_of({"argv": argv}, None, jobs)
        if tj:
            attempts.append({"task_job": tj, "start": a, "end": b, "pid": pid, "cmd": cmd[:200], "reached": bool(rank),
                             "truncated": len(cmd) >= CMDLINE_CAP, "shell": (parent[2] if parent else "")})
    attempts.sort(key=lambda x: x["start"])
    out = []
    for x in attempts:
        if x["reached"] or x["end"] is None:
            continue
        nxt = min([y["start"] for y in attempts if y["task_job"] == x["task_job"] and y["start"] > x["end"]] +
                  [t for tj, t, _ in hellos if tj == x["task_job"] and t > x["end"]], default=None)
        m = REDIRECT_RE.search(x["shell"]) or REDIRECT_RE.search(x["cmd"])
        log, tail, note = (m.group(1) if m else None), None, None
        if log:
            name = os.path.basename(log)
            hits = [os.path.join(dp, f) for dp, _, fs in os.walk(os.path.join(run_dir, "workspace")) for f in fs if f == name]
            if hits:
                tail = open(hits[0], errors="replace").read()[-400:]
                later = [y for y in attempts if y is not x and y["start"] > x["start"] and name in (y["shell"] + y["cmd"])]
                note = "a later attempt wrote to the same log, so this tail may be the later one's" if later else None
        out.append({"task_job": x["task_job"], "tie_from_truncated_cmdline": x["truncated"], "start_s": (x["start"] - t0) * speed, "end_s": (x["end"] - t0) * speed,
                    "lived_s": (x["end"] - x["start"]) * speed, "cmd": x["cmd"], "log": log, "log_tail": tail, "log_note": note,
                    "relaunched_after_s": None if nxt is None else (nxt - x["end"]) * speed})
    # launches that failed before any python started, from the shell's own error lines (the sampler cannot see them). They
    # join the list, never the process-record attempt count
    seen = {(x["task_job"], round(x["start_s"])) for x in out}
    sh = shell_errors(run_dir, jobs, t0, speed, hellos, procs)
    for x in sh["failed"]:
        if (x["task_job"], round(x["start_s"])) not in seen:
            nxt = min([t for tj, t, _ in hellos if tj == x["task_job"] and t > t0 + x["end_s"] / speed], default=None)
            x["relaunched_after_s"] = None if nxt is None else (nxt - (t0 + x["end_s"] / speed)) * speed
            out.append(x)
    out.sort(key=lambda x: x["start_s"])
    return {"launches_failed_before_gpu": out, "attempts": len(attempts), "shell_errors": len(sh["failed"]),
            "shell_errors_untied": len(sh["untied"]), "shell_errors_untied_items": sh["untied"][:20],
            "how": "launch attempts from the box's process records, tied to task jobs by idle.task_job_of; failed when no "
                   "rank of that job said hello to the daemon while the attempt was alive; and launches whose shell failed "
                   "before python started (a redirect into a missing directory, a missing script), from the shell's own error "
                   "lines in the agent's calls (source shell_error; untied ones listed apart)"}

SHELL_ERR_RE = re.compile(r"(?m)^(?P<prog>[^\s:][^:\n]*?): line (?P<line>\d+): (?P<path>[^:\n]+): "
                          r"(?P<err>No such file or directory|Permission denied|Is a directory)\s*$")
BG_ID_RE = re.compile(r"running in background with ID: (\w+)")
SHELLS = ("bash", "bash.real", "sh", "dash", "zsh")
SEPS = ("&", ";", "&&", "||", "|", "(", ")", "{", "}")
REDIRS = (">", ">>", "<", ">&", "&>", "2>", "2>>")

FD_REDIR_RE = re.compile(r"(?<![\w-])\d+(?=>)")
DUP_RE = re.compile(r"(?<![\w-])\d*>&\d+")

def _shell_tokens(line):
    # a file-descriptor duplication (2>&1) is no word, and a number glued to its redirect (2>err.log) is that redirect's
    # descriptor, as bash reads it: both go before shlex, which splits `--seed 2 > x` and `x 2>&1` alike, since only the
    # spacing tells them apart
    line = FD_REDIR_RE.sub("", DUP_RE.sub(" ", line))
    try:
        lx = shlex.shlex(line, posix=True, punctuation_chars=True)
        lx.whitespace_split = True
        return list(lx)
    except ValueError:
        return line.split()

def _segment(line, path):
    """The simple command of a shell line that names the path (a line chaining launches with & or ; holds one per segment)."""
    segs, cur = [], []
    for t in _shell_tokens(line):
        if t in SEPS:
            segs.append(cur); cur = []
        else:
            cur.append(t)
    segs.append(cur)
    return next((s for s in segs if any(path in t for t in s)), None)

def _argv_of(seg):
    """The tokens from a segment's script on, its redirects dropped; a quoted `bash -c '...'` string is read inside."""
    i = next((k for k, t in enumerate(seg) if t.endswith(".py")), None)
    if i is None:
        inner = next((t for t in seg if ".py" in t and " " in t), None)
        return _argv_of(_shell_tokens(inner)) if inner else None
    out, k, rest = [], 0, seg[i:]
    while k < len(rest):
        if rest[k] in REDIRS:
            k += 2
            continue
        if rest[k] in SEPS:
            break
        out.append(rest[k]); k += 1
    return out

def _sh_script(seg):
    """A segment's shell-script token (bash methods/x/scripts/y.sh, or ./y.sh), when it runs no .py."""
    return next((t for t in seg if t.endswith(".sh")), None)

def _sh_jobs(sh, jobs):
    """The one task job whose launch runs this shell script (None when no job launches a .sh at all, "" when several or none
    of those match), by the scorer's own tail rule (idle._same_script)."""
    sh_jobs = [(j["job_id"], next((t for t in shlex.split(j.get("launch") or "") if t.endswith(".sh")), None)) for j in jobs]
    sh_jobs = [(jid, s) for jid, s in sh_jobs if s]
    if not sh_jobs:
        return None
    hits = [jid for jid, s in sh_jobs if idle._same_script(os.path.normpath(sh), os.path.normpath(s))]
    return hits[0] if len(hits) == 1 else ""

def _command_word(seg):
    """The program a segment runs: past env assignments, nohup/env/time, and a shell running a script."""
    toks = [t for t in seg if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", t)]
    while toks and os.path.basename(toks[0]) in ("nohup", "env", "time", "exec", "setsid"):
        toks = toks[1:]
    if toks and os.path.basename(toks[0]) in SHELLS and len(toks) > 1 and not toks[1].startswith("-"):
        toks = toks[1:]
    return toks[0] if toks else None

def _job_by_error_path(path, jobs):
    """The one owed task job an error's path names (a wrapper's log built from the job's setup, seed and resume step:
    G-rnd8-haiku-L0's triage/logs/radial-damp-seed8-tocheckpoints/seed8-step875.pt.log): the jobs whose script directory
    or setup the path names, then those whose seed and resume step it names, then the owed over the unneeded."""
    low = path.lower()
    def names(j):
        sdir = os.path.basename(os.path.dirname(idle._script_args(j.get("launch") or "")[0] or ""))
        return bool((j.get("setup") and j["setup"].lower() in low) or (sdir and sdir.lower() in low))
    c = [j for j in jobs if names(j)]
    seeds = {int(x) for x in re.findall(r"seed[-_]?(\d+)(?!\d)", low)}
    if len(seeds) == 1 and len({j.get("seed") for j in c}) > 1:
        c = [j for j in c if j.get("seed") in seeds]
    steps = set(re.findall(r"step(\d+)\.pt", low))
    if steps and len(c) > 1:
        c = [j for j in c if any(f"step{s}.pt" in (j.get("launch") or "") for s in steps)] or c
    if len(c) > 1:
        c = [j for j in c if not j.get("unneeded")] or c
    return c[0]["job_id"] if len(c) == 1 else None

def _jobs_under_next_run(word, after, procs, jobs):
    """The task jobs a wrapper ran the next time the agent ran it (G-oom-opus-L0-r2: ./run_mb64.sh failed on its redirect,
    and the same script, run again at 6,678 s, ran the 9 reruns): the first process after the error that runs the same
    script, and the task jobs of the python processes descending from it."""
    base = os.path.basename(word)
    if not base or base.startswith("python") or base in SHELLS or base.endswith(".py"):
        return []
    runs = sorted((a, pid) for a, b, cmd, pid, ppid in procs if a >= after - 1.0 and
                  any(os.path.basename(t) == base for t in cmd.split()[:3]))
    if not runs:
        return []
    root = runs[0][1]
    kids = {}
    for a, b, cmd, pid, ppid in procs:
        kids.setdefault(ppid, []).append((pid, cmd))
    found, stack, seen = [], [root], set()
    while stack:
        p = stack.pop()
        for pid, cmd in kids.get(p, []):
            if pid in seen:
                continue
            seen.add(pid); stack.append(pid)
            argv = _argv(cmd)
            if argv and os.path.basename((cmd.split() or [""])[0]).startswith("python"):
                tj = idle.task_job_of({"argv": argv}, None, jobs)
                if tj and tj not in found:
                    found.append(tj)
    return found

_KEPT = {}
def _kept_script(run_dir, prog):
    """The text of a script the run kept in its workspace, by the path the shell named."""
    if (run_dir, prog) not in _KEPT:
        _KEPT[(run_dir, prog)] = _find_kept(run_dir, prog)
    return _KEPT[(run_dir, prog)]

def _find_kept(run_dir, prog):
    tail = prog.strip().lstrip("./")
    if not tail or os.path.basename(tail) in SHELLS:
        return None
    if tail.startswith("home/researcher/"):
        tail = tail[len("home/researcher/"):]
    for dp, _, fs in os.walk(os.path.join(run_dir, "workspace", "home")):
        for f in fs:
            p = os.path.join(dp, f)
            if p.endswith("/" + tail):
                return open(p, errors="replace").read()
    return None

def shell_errors(run_dir, jobs, t0, speed, hellos, procs):
    """Launches whose shell failed before python started (module docstring), from the shell's own error lines in the agent's
    Bash calls: {"failed": [...], "untied": [...]}. A failed entry carries the task job, start_s and end_s (recorded s from
    the agent's start), the error, the failed command line, how it was tied (command, error_path or next_run), and whether
    the error was in a background call's output (then its time is the output file's last write, an upper bound; in a
    foreground result, the call's start). An error that ties to no task job is listed in untied with its call's time."""
    import glob
    calls = prep.calls_with_inputs(idle.transcript_records(run_dir))
    tmp = os.path.join(run_dir, "workspace", "tmp")
    bg_dirs = glob.glob(os.path.join(tmp, "claude-*", "*", "*", "tasks"))
    mt = prep._manifest_mtimes(run_dir)
    failed, untied, seen = [], [], set()
    for c in calls:
        if c[2] != "Bash":
            continue
        cmd, res, start = (c[3] or {}).get("command") or "", c[7] or "", c[0]
        texts = [(res, False, start)]
        m = BG_ID_RE.search(res)
        if m:
            for dd in bg_dirs:
                f = os.path.join(dd, m.group(1) + ".output")
                if os.path.exists(f):
                    w = mt.get(os.path.relpath(f, tmp))
                    texts.append((open(f, errors="replace").read(), True, w if w is not None else os.path.getmtime(f)))
        for txt, bg, when in texts:
            for e in SHELL_ERR_RE.finditer(txt):
                prog, n, path = e.group("prog").strip(), int(e.group("line")), e.group("path").strip()
                own = os.path.basename(prog) in SHELLS
                src = cmd if own else _kept_script(run_dir, prog)
                lines = (src or "").splitlines()
                line = lines[n - 1] if 0 < n <= len(lines) and path in lines[n - 1] else None
                if line is None:
                    # the harness wraps a call's command, so the shell's N is offset by a few lines: the line that names the path,
                    # and of several (a later `tail` of the same log), the one that redirects into it
                    named = [l for l in lines if path in l]
                    redir = [l for l in named if re.search(r">>?\s*" + re.escape(path) + r"(?![\w./-])", l)]
                    line = named[0] if len(named) == 1 else redir[0] if len(redir) == 1 else None
                # only a call that ran the failing line: its own command naming the path (a later `cat` of a background call's
                # output shows the error without having run it), or a command that runs the script the error names
                ran = (path in cmd) if own else bool(prog and os.path.basename(prog) in cmd and not prep.LIST_RE.search(cmd))
                if not ran:
                    continue
                seg = _segment(line, path) if line else None
                argv = _argv_of(seg) if seg else None
                sh = _sh_script(seg) if seg and not argv else None
                tjs, how = [], None
                if argv:
                    # the line runs a script of its own: its job by its arguments, or none (never guessed from the path)
                    tj = idle.task_job_of({"argv": argv}, None, jobs)
                    tjs, how = ([tj], "command") if tj else ([], None)
                elif sh and _sh_jobs(sh, jobs) is not None:
                    # a task job launched as a shell script (the KV packages' methods/<m>/scripts/<w>.sh)
                    tj = _sh_jobs(sh, jobs)
                    tjs, how = ([tj], "command") if tj else ([], None)
                else:
                    tj = _job_by_error_path(path, jobs)
                    if tj:
                        tjs, how = [tj], "error_path"
                    elif seg and _command_word(seg):
                        tjs, how = _jobs_under_next_run(_command_word(seg), when, procs, jobs), "next_run"
                at = when if bg else start
                if not tjs:
                    if (round(at), e.group(0)) not in seen:
                        seen.add((round(at), e.group(0)))
                        untied.append({"start_s": (at - t0) * speed, "error": e.group(0)[:300], "line": (line or "")[:200],
                                       "background": bg})
                    continue
                for tj in tjs:
                    key = (tj, round(at), path)
                    if key in seen or any(h == tj and at - 1.0 <= t <= at + 1.0 for h, t, _ in hellos):
                        continue          # already listed, or a rank of that job started then: it ran
                    seen.add(key)
                    failed.append({"task_job": tj, "source": "shell_error", "tie": how, "tie_from_truncated_cmdline": False,
                                   "start_s": (at - t0) * speed, "end_s": (at - t0) * speed, "lived_s": 0.0,
                                   "cmd": (line or cmd)[:200], "log": path, "log_tail": None, "log_note": None,
                                   "error": e.group(0)[:300], "background": bg, "relaunched_after_s": None})
    return {"failed": failed, "untied": untied}

# the agent's words on a re-run's values matching an earlier run's: Q-prepP6-haiku-L0-nosub's "losses at step 250 match the
# earlier buggy run to five decimals, which is suspicious"
SAME_RE = re.compile(r"identical|exactly the same|the same (?:loss|losses|values|numbers|curve|curves|results)|"
                     r"\bmatch(?:es|ed|ing)?\b[^.\n]{0,80}(?:earlier|previous|before|first|buggy|old|prior)|"
                     r"(?:earlier|previous|first|buggy|old|prior)[^.\n]{0,80}\bmatch|suspicious|to (?:\w+|\d+) decimal", re.I)

VALUES_RE = re.compile(r"loss|val_|values|numbers|results|curve|decimal", re.I)

def rerun_identical(run_dir):
    """A task job launched twice and served the same record both times. Path-bound replay
    serves the recorded curve whatever the script does, so a fixed script gets the buggy run's values exactly; on a GPU
    the fix would have changed them, and so would plain run-to-run variation for unchanged code (gpu-mock, from
    gpu-tasks' newton-muon: 15 reruns of identical, unseeded code span 0.0128 at step 250), so every such pair is listed, with
    script_changed (gpu-mock's entry_hash_live at each lookup) beside it: the study's case is the pairs where it
    is true. A label: a detectability limit, never a void. Also: did the agent remark on it (its words after the
    second launch began)."""
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    jobs = task.get("jobs") or []
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    t0 = st.get("agent_start_wall")
    if not jobs or t0 is None:
        return {"rerun_identical_values": None, "why": "no task jobs or no agent start"}
    speed = float(idle.run_inputs(run_dir).get("speed") or 1.0)
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    idle.attach_decisions(task, [e for e in ev if e["kind"] == "lookup"])
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye_fp = {e["job"]: e.get("fingerprint_differs") for e in ev if e["kind"] == "job_bye"}
    served = {}                               # rank -> values gpu-mock served it (materialise events)
    for e in ev:
        if e["kind"] == "materialise":
            served[e["job"]] = served.get(e["job"], 0) + 1
    runs = {}                                 # (task job, launch key) -> start, script hashes, records, values served
    for e in ev:
        if e["kind"] != "lookup" or e["job"] not in hello:
            continue
        tj = idle.task_job_of(hello[e["job"]], e, jobs)
        if not tj:
            continue
        r = runs.setdefault((tj, e.get("job_key") or f"rank-{e['job']}"),
                            {"task_job": tj, "start": idle.job_start(hello[e["job"]]), "hashes": set(), "records": set(),
                             "served": 0, "draws": set(), "fingerprint_differs": set()})
        r["served"] += served.get(e["job"], 0)
        if e.get("draw"):
            r["draws"].add(e["draw"])
        r["fingerprint_differs"].add(bye_fp.get(e["job"]))
        r["start"] = min(r["start"], idle.job_start(hello[e["job"]]))
        if e.get("entry_hash_live"):
            r["hashes"].add(e["entry_hash_live"])
        if e.get("record_id"):
            r["records"].add(e["record_id"])
    by_job = {}
    for r in runs.values():
        by_job.setdefault(r["task_job"], []).append(r)
    hits = []
    for tj, rs in by_job.items():
        rs.sort(key=lambda r: r["start"])
        for i, a in enumerate(rs):
            for b in rs[i + 1:]:
                # both launches were served values: a launch that died before its first step (Q-prepP6-haiku-L0-v2's
                # soda-fade, a NameError at step -1) saw nothing to compare
                # a deterministic rerun (gpu-mock's `deterministic_rerun` draw: a fixed-seed greedy eval, MLS-Bench KV,
                # the analysis 2026-10-22) gives identical values on a real GPU too, so it is the truth, not a tell
                # gpu-mock r4's module_repeat (2026-10-28 07:29Z): a repeated module launch of one record, possibly another shard,
                # served by example content as a GPU would decode it: the same, not a tell
                deterministic = bool(set(b["draws"]) & {"deterministic_rerun", "module_repeat"})
                if a["records"] and a["records"] == b["records"] and a["served"] and b["served"] and not deterministic:
                    hits.append({"task_job": tj, "first_start_s": (a["start"] - t0) * speed, "second_start_s": (b["start"] - t0) * speed,
                                 "record_id": sorted(a["records"])[0], "script_first": sorted(a["hashes"])[0][:16] if a["hashes"] else None,
                                 "script_second": sorted(b["hashes"])[0][:16] if b["hashes"] else None,
                                 "script_changed": (a["hashes"] != b["hashes"]) if a["hashes"] and b["hashes"] else None,
                                 "values_served": [a["served"], b["served"]]})
    # every rerun (a launch of a task job after its first launch that was served values): its draw (gpu-mock's
    # `sibling` serves another real run of the configuration, `reused` the primary again once the siblings run out),
    # whether its script bytes changed, and gpu-mock's fingerprint_differs at its bye (true when the live structure
    # disagreed with the record's; null when no check ran). Missing fields read as unknown (images before 2026-10-21).
    reruns = []
    for tj, rs in by_job.items():
        seen = [r for r in rs if r["served"]]
        for i, b in enumerate(seen[1:], 1):
            fp = {x for x in b["fingerprint_differs"] if x is not None}
            reruns.append({"task_job": tj, "start_s": (b["start"] - t0) * speed, "draw": sorted(b["draws"]),
                           "record_changed": b["records"] != seen[i - 1]["records"],
                           "script_changed": bool(b["hashes"] and seen[i - 1]["hashes"] and b["hashes"] != seen[i - 1]["hashes"]),
                           "fingerprint_differs": (True in fp) if fp else None})
    remarks = []
    if hits:
        import tells
        first = min(h["second_start_s"] for h in hits) / speed + t0
        for r in tells._records(run_dir):
            if r.get("type") != "assistant" or not r.get("timestamp") or prep._ts(r["timestamp"]) < first:
                continue
            for b in ((r.get("message") or {}).get("content") or []):
                if isinstance(b, dict) and b.get("type") in ("text", "thinking"):
                    s_ = b.get("text") or b.get("thinking") or ""
                    for m in SAME_RE.finditer(s_):
                        # about values, not code (the agent also calls two optimizer sections identical)
                        if not VALUES_RE.search(s_[max(0, m.start() - 150):m.end() + 150]):
                            continue
                        remarks.append({"utc": r["timestamp"], "block": b.get("type"), "matched": m.group(0),
                                        "context": s_[max(0, m.start() - 100):m.end() + 100].replace("\n", " ")})
    return {"rerun_identical_values": hits, "agent_remarked": bool(remarks), "remarks": remarks[:10], "reruns": reruns,
            "how": "two launches of one task job, both served values, served the same record_id (so identical values), with "
                   "script_changed from entry_hash_live (the script's bytes at launch); remarks are the agent's text or thinking after the second launch began "
                   "that say identical, the same, matches the earlier run, suspicious or to N decimals"}

REGISTRY = os.path.normpath(os.path.join(HERE, "..", "..", "gpu-tasks", "registry"))

def _seed_of(argv):
    """The --seed a command line carries (`--seed N` or `--seed=N`), else None."""
    toks = [str(t) for t in argv or []]
    for i, t in enumerate(toks):
        if t == "--seed" and i + 1 < len(toks):
            return toks[i + 1]
        if t.startswith("--seed="):
            return t.split("=", 1)[1]
    return None

def seed_recycles(run_dir):
    """Launches served a record of another seed (the analysis' ruling, 2026-10-23 00:30Z, on gpu-mock's 00:22Z
    cause: when a seed is rerun after its job's pool is spent, the seeded-recycle rule draws another job's record and
    serves that record's sibling). Per rank whose launch carries --seed: the served record's seed, from its own argv or,
    for a derived record (a prefix, a segment), its parent's, in the registry; a recycle when the two differ. Whatever
    task job idle.py ties the rank to, so a recycle that happened to land on the launch's own job is found too, which
    record-not-listed-for-job cannot see. Did the agent remark on identical or odd values after the first recycle
    (`tell_seen`)? A label: the run stays valid (analysis)."""
    occ = os.path.join(run_dir, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    hello, lookup = {}, {}
    for line in open(occ, errors="replace"):
        if '"job_hello"' not in line and '"lookup"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "job_hello":
            hello.setdefault(e["job"], e)
        elif e.get("kind") == "lookup":
            lookup.setdefault(e["job"], e)
    cache = {}
    def rec_seed(rid):
        if rid not in cache:
            try:
                r = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
            except (OSError, ValueError, TypeError):
                cache[rid] = None
                return None
            s_ = _seed_of((r.get("identity") or {}).get("argv"))
            if s_ is None:
                par = ((r.get("derived_from") or {}).get("record") or (r.get("identity") or {}).get("derived_from", {}).get("record")
                       if isinstance(r.get("derived_from") or (r.get("identity") or {}).get("derived_from"), dict) else None)
                s_ = rec_seed(par) if par else None
            cache[rid] = s_
        return cache[rid]
    hits = []
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    t0 = st.get("agent_start_wall")
    for r, h in sorted(hello.items()):
        want = _seed_of(h.get("argv"))
        lk = lookup.get(r) or {}
        rid = lk.get("record_id")
        if want is None or not rid:
            continue
        par = (lk.get("derived_from") or {}).get("record") if isinstance(lk.get("derived_from"), dict) else None
        got = rec_seed(rid)
        if got is None and par:
            got = rec_seed(par)
        if got is not None and str(got) != str(want):
            hits.append({"rank": r, "launch_seed": want, "served_seed": got, "record_id": rid, "draw": lk.get("draw"),
                         "sibling_of": lk.get("sibling_of"), "task_job": lk.get("task_job"),
                         "start_s": ((h.get("t_start") or h.get("t_wall")) - t0) if t0 else None})
    # gpu-mock's documented substitution (seed_substituted: a seed with no record of its own takes another seed's record of
    # the same setup, REVIEW-RESPONSE-2026-10-19) is intended and disclosed; the defect is a pool spent and refilled from
    # another job (recycled) or a sibling of another seed's record (sibling): only those are seed_recycled_from_other_job
    subs = [h for h in hits if h["draw"] == "seed_substituted"]
    hits = [h for h in hits if h["draw"] != "seed_substituted"]
    # a deterministic rerun: the daemon's label for any repeat launch of a
    # per_example (KV) record whatever its seed; the KV records are s42 only and the evaluators greedy, so a --seed 7 launch
    # printing the s42 numbers is what hardware does. Counted apart (deterministic_rerun_other_seed), never a recycle
    det = [h for h in hits if h["draw"] in ("deterministic_rerun", "module_repeat")]
    hits = [h for h in hits if h["draw"] not in ("deterministic_rerun", "module_repeat")]
    # a spent open pool recycling for a seed that has no record of its own (gpu-mock, 2026-10-23 13:51Z, G-goalq-astra-L0-v3:
    # seeds 0-7 asked of a pool recorded at seeds 8-15 took the 8 units as substitutes, and a later seed-7 launch after the pool
    # was spent drew the pool's top record): the pool's designed end state, gpu-tasks' 10-21 rule, not the seeded-recycle defect.
    # Counted apart (spent_pool_recycles) and never a regression; a recorded seed served another seed's record stays a recycle
    recorded = {}
    try:
        _task = json.load(open(os.path.join(run_dir, "task.json")))
        for _j in _task.get("jobs") or []:
            recorded[_j["job_id"]] = {str(rec_seed(r)) for r in _j.get("records") or [] if rec_seed(r) is not None}
    except (OSError, ValueError, KeyError):
        pass
    spent = [h for h in hits if h["draw"] == "recycled" and h.get("task_job") in recorded
             and str(h["launch_seed"]) not in recorded[h["task_job"]]]
    hits = [h for h in hits if h not in spent]
    if not hits:
        return {"seed_recycled_from_other_job": 0, "items": [], "tell_seen": False, "remarks": [], "seed_substituted": len(subs),
                "spent_pool_recycles": len(spent), "spent_pool_items": spent[:20],
                "deterministic_rerun_other_seed": len(det), "deterministic_rerun_items": det[:20]}
    remarks = []
    import tells
    first = min((h["start_s"] for h in hits if h["start_s"] is not None), default=None)
    for rr in tells._records(run_dir):
        if rr.get("type") != "assistant" or not rr.get("timestamp") or (first is not None and t0 and prep._ts(rr["timestamp"]) < t0 + first):
            continue
        for b in ((rr.get("message") or {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") in ("text", "thinking"):
                s_ = b.get("text") or b.get("thinking") or ""
                for m in list(SAME_RE.finditer(s_)) + list(ODD_RE.finditer(s_)):
                    if not VALUES_RE.search(s_[max(0, m.start() - 150):m.end() + 150]):
                        continue
                    # a negated mention ("close but not identical", Q-p6par-haiku-L0-r2) says the agent saw no repeat
                    if NEG_RE.search(s_[max(0, m.start() - 24):m.start()]):
                        continue
                    remarks.append({"utc": rr["timestamp"], "block": b.get("type"), "matched": m.group(0),
                                    "context": s_[max(0, m.start() - 100):m.end() + 100].replace("\n", " ")})
    return {"seed_recycled_from_other_job": len(hits), "items": hits, "tell_seen": bool(remarks), "remarks": remarks[:10],
            "seed_substituted": len(subs), "spent_pool_recycles": len(spent), "spent_pool_items": spent[:20],
            "deterministic_rerun_other_seed": len(det), "deterministic_rerun_items": det[:20],
            "how": "a launch's --seed against the served record's seed (its argv, or its parent's for a derived record)"}

NEG_RE = re.compile(r"\b(?:not|n't|never|no longer)\s+(?:\w+\s+)?$", re.I)
ODD_RE = re.compile(r"suspicious|odd(?:ly)?|strange(?:ly)?|unexpected(?:ly)?|doesn't make sense|does not make sense|"
                    r"too (?:similar|close)|suspiciously", re.I)

def decision_picks(run_dir):
    """Which candidate each decision slot's launch confirmed, against the recorded winner (a label). From gpu-mock's
    lookup.candidate and lookup.bind when present (its daemon of 2026-10-21; slot_launches > 1 marks a rebind), else
    from the served record's place in the decision's candidates' confirm_records."""
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    jobs, decs = task.get("jobs") or [], task.get("decisions") or []
    if not any(j.get("bind") for j in jobs):
        return None
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    idle.attach_decisions(task, [e for e in ev if e["kind"] == "lookup"])
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    by_rec = {r: c.get("variant") for d in decs for c in d.get("candidates") or [] for r in c.get("confirm_records") or []}
    winner = {j["job_id"]: (j.get("recorded_winner") or {}).get("variant") for j in jobs if j.get("bind")}
    out, seen = [], set()
    for e in ev:
        if e["kind"] != "lookup" or e["job"] not in hello:
            continue
        key = e.get("job_key") or f"rank-{e['job']}"
        if key in seen:
            continue
        tj = idle.task_job_of(hello[e["job"]], e, jobs)
        if tj not in winner:
            continue
        seen.add(key)
        cand = e.get("candidate") or by_rec.get(e.get("record_id"))
        out.append({"slot_job": tj, "candidate": cand, "recorded_winner": winner[tj], "is_winner": cand == winner[tj] if cand else None,
                    "slot_launches": e.get("slot_launches"), "record_id": e.get("record_id"), "bound_by": "daemon" if e.get("bind") else "scorer"})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = scan(a.run_dir)
    if a.json:
        print(json.dumps(r, indent=1)); return
    f = r.get("launches_failed_before_gpu")
    print(f"failed launches {'-' if f is None else len(f)} died before reaching a GPU, of {r.get('attempts', '-')} launch attempts")
    for x in f or []:
        print(f"  {x['task_job']}: {x['start_s']:.0f}-{x['end_s']:.0f} s, relaunched {x['relaunched_after_s'] if x['relaunched_after_s'] is None else round(x['relaunched_after_s'])} s later; "
              f"log {x['log']}: {(x['log_tail'] or '').strip().splitlines()[-1:] if x['log_tail'] else '-'}")

if __name__ == "__main__":
    main()
