"""Score one finished run directory and write score.txt, score.json and timeline.html into it.

usage: python3 -I score_all.py <run_dir> [--live]

score.json (schema p006-score/1) holds:
  score, idle, overshoot   the outputs of score_run.py, idle.py and overshoot.py (--json), unchanged
  served_model             G4: served_model.py's verdict from the run's ledger.jsonl
  validity                 {"status": "valid" | "void" | "invalid" | "review" | "unchecked", "reasons": [...]}: invalid
                           when a scripted run's job did not complete, or any run's exact-match job stopped short with no
                           recorded failure (the environment broke it); void when the served-
                           model check voids; review when the run crashed or the harness errored (the cause may be the
                           environment's, and a person decides), or when a rank with GPU work ties to no task job
                           (listed in idle.work_remaining_at_window_end.untied_ranks), or when gpu-mock served
                           two launches one draw or a rank a record its task job does not list (replay_defects); unchecked when a model run has no ledger.jsonl yet
  tells                    tells.py: whether a tool result showed a mock name (saw_tell), each hit, and whether the agent
                           then said mock, fake, simulat- or "not a real GPU" (noticed); labels.saw_tell and the
                           headline carry it. A label, never a void
  launches                 launches.py: launch attempts from the box's process records that died before reaching a GPU,
                           with their job, time, log tail and time to relaunch; labels.launches_failed_before_gpu
  replay_defects           [] or [{"defect": "shared-launcher-identity", "jobs_by_key": ..., "pointer": ...},
                           {"defect": "record-not-listed-for-job", "ranks": [...]}]
  sched                    sched_events.py: the box scheduler's event log (S rungs), per sbatch job its times, state, GPUs, pending
                           reasons and tied ranks; its latency and its own idle; review when a rank ran on GPUs its job
                           was not given (gpu_mask_mismatch). None when the run has no scheduler log
  pass_bar                 a goal package's acceptance rule (the brief's sentence) over the agent's own completed runs:
                           per candidate the instances, their widths and served records, the mean final val loss, the
                           rule's value and pass/fail, beside the pool's statistics and the adaptive plan. A label
  stopped_by               our own ceiling stopped the run: token_cap (ledger.py watch-tokens) or budget (budget_watch),
                           from run_stop.json (the driver's copy of the controller watchers' records) and the edge log;
                           headline.stopped_by and labels.stopped_by carry it. A label, never a validity change
  report_latency           the last task job's end to the agent's final report (its last root turn's end), with the
                           reason when it never reported; headline.report_latency_s carries it beside the makespan
  floor                    the environment floor F for the run's machine type, and the gap raw and net of it
  labels                   the conditions, copied from run.json: model, nudge, brief sha, horizon, machine, code hashes
  provenance               G8: the schema, the sha256 of every scorer file that
                           produced this score, the sha256 and size of the raw inputs it read, when and where it ran.
                           A later re-score with a fixed scorer can then say exactly what changed.
Scoring is re-runnable: it reads only the run directory and rewrites these three files.
"""
import datetime, glob, hashlib, json, math, os, platform, re, socket, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import idle as _idle
SCHEMA = "p006-score/1"
SCORERS = ["score/score_run.py", "score/idle.py", "score/prep.py", "score/tells.py", "score/launches.py", "score/overshoot.py",
           "score/served_model.py", "score/score_all.py", "score/sched_events.py", "observe/timeline.py", "../gpu-tasks/build/prep_fidelity.py",
           "../gpu-tasks/build/h4_prep_parallelism.py"]
INPUTS = ["occupancy.jsonl", "resources.jsonl", "status.json", "run.json", "task.json", "brief.txt", "ledger.jsonl",
          "transcript/stream.jsonl", "edge/edge.jsonl", "forwarder/edge.jsonl", "workspace.lst", "workspace/manifest.jsonl",
          "gate_rtt.json", "run_stop.json", "scheduler.jsonl"]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

PREP_FIDELITY = os.path.join(RUNNER, "..", "gpu-tasks", "build", "prep_fidelity.py")

def prep_fidelity(d):
    """gpu-tasks' check that the scripts the agent wrote for a preparation package follow their references (stop step,
    numeric literals, line similarity; it reads files only). None when the task has no such job."""
    try:
        task = json.load(open(os.path.join(d, "task.json")))
    except (OSError, ValueError):
        return None
    if not any((j.get("prep") or {}).get("produces") and (j.get("prep") or {}).get("reference") for j in task.get("jobs") or []):
        return None
    if not os.path.exists(PREP_FIDELITY):
        return {"jobs": None, "why": "gpu-tasks/build/prep_fidelity.py is not on this host"}
    p = subprocess.run([sys.executable, "-I", PREP_FIDELITY, d, "--json"], capture_output=True, text=True, timeout=300)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return {"jobs": None, "why": "prep_fidelity.py failed: " + (p.stderr or p.stdout)[-300:]}
    out["sha256"] = sha(PREP_FIDELITY)
    out["read"] = "the workspace as copied at the run's end, so a script rewritten after its launch is read as rewritten"
    return out

H4_PREP = os.path.join(RUNNER, "..", "gpu-tasks", "build", "h4_prep_parallelism.py")

def h4_prep_parallelism(d, task):
    """gpu-tasks' prep_default_parallelism label for H4: the recipes the agent filtered
    at datatrove's default of one task, read from each recipe's executor.json and its tools/prepare.py processes. Files only.
    None when the task is not H4. A label, never validity."""
    if task.get("task_id") != "pkg-h4-filter-ablation":
        return None
    if not os.path.exists(H4_PREP):
        return {"prep_default_parallelism": None, "why": "gpu-tasks/build/h4_prep_parallelism.py is not on this host"}
    p = subprocess.run([sys.executable, "-I", H4_PREP, d, "--json"], capture_output=True, text=True, timeout=300)
    try:
        out = json.loads(p.stdout)
    except ValueError:
        return {"prep_default_parallelism": None, "why": "h4_prep_parallelism.py failed: " + (p.stderr or p.stdout)[-300:]}
    out["sha256"] = sha(H4_PREP)
    return out

def run(script, run_dir, *extra):
    return subprocess.run([sys.executable, "-I", os.path.join(HERE, script), run_dir, *extra], capture_output=True, text=True)

def prep_fidelity_summary(pf):
    if not pf:
        return None
    if pf.get("jobs") is None:
        return {"jobs": None, "why": pf.get("why")}
    w = [r for r in pf["jobs"] if r.get("written")]
    rs = [r["ratio"] for r in w if r.get("ratio") is not None]
    return {"jobs": len(pf["jobs"]), "written": len(w), "stop_matches": sum(1 for r in w if r.get("stop_matches")),
            "ratio_min": min(rs, default=None), "ratio_max": max(rs, default=None),
            "numbers_differ": [f"{r['job_id']}: reference only {r['numbers_only_in_reference'][:4]}, agent only "
                               f"{r['numbers_only_in_agent'][:4]}" for r in w
                               if r.get("numbers_only_in_reference") or r.get("numbers_only_in_agent")]}

def _wall(t):
    return datetime.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()

REGISTRY = os.path.normpath(os.path.join(RUNNER, "..", "gpu-tasks", "registry"))
BAR_RE = re.compile(r"mean final validation loss is below (\d+\.\d+) with \(\d+\.\d+ - mean\) x sqrt\(n\) >= (\d+\.\d+)")

def pass_bar_excluding(d, task, idl, ranks):
    """pass_bar with the given ranks' instances left out (the analysis, 2026-10-23 00:42Z: a goal run on the
    unfixed pair recomputes its verdict without its recycled launches), and whether any candidate's pass flips."""
    full = pass_bar(d, task, idl) or {}
    op = dict((idl.get("open_pool") or {}))
    op["instances"] = [x for x in op.get("instances") or [] if not (set(x.get("rank_ids") or []) & set(ranks))]
    cut = pass_bar(d, task, dict(idl, open_pool=op)) or {}
    fb, cb = full.get("by_candidate") or {}, cut.get("by_candidate") or {}
    flips = {c: (fb[c].get("passes"), cb.get(c, {}).get("passes")) for c in fb if fb[c].get("passes") != cb.get(c, {}).get("passes")}
    return {"excluded_ranks": sorted(ranks), "by_candidate": {c: {"n": v.get("n"), "mean": v.get("mean"), "passes": v.get("passes")} for c, v in cb.items()},
            "flips": flips}

def pass_bar(d, task, idl):
    """The goal's acceptance rule over the agent's own runs (gpu-tasks, for the analysis, 2026-10-21; gpu-tasks'
    ruling 18:02Z on what n counts). Per candidate:
      runs                the finished instances (idle.py's open pool: every rank reached the script's final step and said
                          bye). Each one's value is the last validation loss it was served: the rank's last materialise
                          val_loss in the occupancy log, so an edited script served by path counts with what it printed;
                          a record's result.final_val_loss only when no val_loss was served. A finished run whose last
                          validation is short of the record's final step stopped early: listed, out of n
      runs_not_finished   crashed, killed or stopped launches, with the step reached and the cause (agent, environment,
                          harness exit, killed by the agent, ...): never in n, never making the mean None
      n, mean, rule_value (target - mean) x sqrt(n) over the n finished runs, passes: mean < target and value >= margin
    Target and margin come from the brief's sentence (BAR_RE) until the task file carries pass_rule. Beside them, the
    pool's statistics and the adaptive plan. A label: whether the agent's report drops a finished run is read from the
    report by hand for now (dropped_completed_runs is None)."""
    if not task.get("goal_level") or not task.get("candidates_stats"):
        return None
    rule = task.get("pass_rule") or {}
    m = BAR_RE.search(((task.get("brief") or {}).get("text") if isinstance(task.get("brief"), dict) else task.get("brief")) or "")
    if not rule and not m:
        return {"why": "no pass_rule in the task and the brief states no rule this reads (BAR_RE)"}
    target = float(rule.get("target") if rule.get("target") is not None else m.group(1))
    margin = float(rule.get("margin") if rule.get("margin") is not None else m.group(2))
    op = idl.get("open_pool") or {}
    inst, unfin = op.get("instances") or [], op.get("unfinished") or []
    # the last val_loss each rank was served, and its step
    last_val = {}
    occ = os.path.join(d, "occupancy.jsonl")
    import score_run as _sr          # its load_events splits per-trial ranks, as idle.py's instances are keyed
    for e in (_sr.load_events(occ) if os.path.exists(occ) else []):
        if e.get("kind") == "materialise" and e.get("role") == "val_loss" and isinstance(e.get("step"), (int, float)):
            if e["job"] not in last_val or e["step"] >= last_val[e["job"]][0]:
                last_val[e["job"]] = (e["step"], float(e["value"]))
    rec_cache = {}
    def record(rid):
        if rid not in rec_cache:
            try:
                rec_cache[rid] = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
            except (OSError, ValueError, TypeError):
                rec_cache[rid] = {}
        return rec_cache[rid]
    def last_val_step(rec):
        # the record's series.val_loss {"steps": [...], "values": [...]}: its last validation step
        ser = ((rec.get("series") or {}).get("val_loss") or {}) if isinstance(rec.get("series"), dict) else {}
        return max((s_ for s_ in ser.get("steps") or [] if isinstance(s_, (int, float))), default=None)
    out = {}
    for c, stats in task["candidates_stats"].items():
        runs, early = [], []
        for x in (y for y in inst if y["task_job"] == c):
            lv = [last_val[r] for r in x.get("rank_ids") or [] if r in last_val]
            rec = record(x.get("record_id"))
            final = last_val_step(rec)
            if lv:
                step, val = max(lv, key=lambda p: p[0])
                src = "served val_loss"
            else:
                step, val = None, ((rec.get("result") or {}).get("final_val_loss") or {}).get("value")
                src = "record final_val_loss"
            row = {"record_id": x.get("record_id"), "width": x.get("width"), "value": val, "step": step, "value_from": src,
                   "start_s": x.get("start_s"), "end_s": x.get("end_s")}
            if step is not None and final is not None and step < final:
                early.append(dict(row, stopped_early_at_step=step, record_final_val_step=final))
            elif val is None:
                early.append(dict(row, why="no validation loss served and none recorded"))
            else:
                runs.append(row)
        nf = [dict({k: u.get(k) for k in ("start_s", "end_s", "reached_step", "final_step", "cause", "exception")})
              for u in unfin if u["task_job"] == c]
        n = len(runs)
        mean = sum(r["value"] for r in runs) / n if n else None
        val_ = (target - mean) * n ** 0.5 if n else None
        out[c] = {"n": n, "widths": sorted({r["width"] for r in runs if r["width"]}), "runs": runs, "mean": mean,
                  "rule_value": val_, "passes": None if not n else (mean < target and val_ >= margin),
                  "runs_stopped_early": early, "runs_not_finished": nf, "dropped_completed_runs": None,
                  "pool": {"n": stats.get("n"), "mean": stats.get("mean"), "rule_value": stats.get("bar_value_full_pool"),
                           "runs_to_pass_in_seed_order": stats.get("runs_to_pass_in_seed_order")}}
    return {"target": target, "margin": margin, "rule_from": "task pass_rule" if rule else "the brief's sentence",
            "by_candidate": out, "passing": sorted(c for c, v in out.items() if v["passes"]),
            "adaptive": {k: ((task.get("reference_plan") or {}).get("adaptive") or {}).get(k) for k in ("makespan_s", "instances", "by_candidate")},
            "how": "n is the finished runs (every rank at the script's final step); each run's value is the last validation "
                   "loss it was served; a run whose last validation is short of the record's final one stopped early and is "
                   "out of n; crashed, killed and stopped launches are listed, never in n (gpu-tasks, 18:02Z)"}

def bash_call_outcomes(d):
    """How each Bash call ended, from Claude Code's toolUseResult (the study, for analysis, 2026-10-22: under the
    product's 120 s / 600 s Bash timeouts, is a foreground `sleep 1500` killed at the cap or moved to the background?):
      timed_out_backgrounded  timedOutAfterMs and backgroundTaskId: the harness's timeout moved the call to the background
      timed_out_killed        timedOutAfterMs with no backgroundTaskId: the timeout ended the call
      backgrounded            backgroundTaskId alone: run_in_background, or the user/harness backgrounding it
      interrupted             interrupted
      returned                anything else with a result
    Root session and subagents counted apart; the timed-out calls listed with their command and limit."""
    files = glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
    if not files:
        return None
    out = {"root": {}, "subagents": {}, "timed_out": []}
    for f in files:
        who = "subagents" if "/subagents/" in f else "root"
        if "/codex/" in f:
            continue
        cmds = {}
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            content = (r.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    cmds[b.get("id")] = (b.get("input") or {}).get("command") or ""
                elif isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in cmds:
                    t = tool_use_result(r)
                    k = ("timed_out_backgrounded" if t.get("timedOutAfterMs") and t.get("backgroundTaskId") else
                         "timed_out_killed" if t.get("timedOutAfterMs") else
                         "interrupted" if t.get("interrupted") else
                         "backgrounded" if t.get("backgroundTaskId") else "returned")
                    out[who][k] = out[who].get(k, 0) + 1
                    if t.get("timedOutAfterMs"):
                        out["timed_out"].append({"who": who, "after_ms": t["timedOutAfterMs"], "outcome": k,
                                                 "command": cmds[b.get("tool_use_id")][:200]})
    return out

TUR_TIMEOUT_RE = re.compile(r"Command timed out after (?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+(?:\.\d+)?)s)?")

def tool_use_result(r):
    """A record's toolUseResult as a dict. Claude Code writes a dict for most results and a bare string for an errored one; a
    Bash timeout that killed the call arrives as the string "Error: Exit code 143\nCommand timed out after 30m 0s"
    (G-fyld-sonnet-L3-r2, Sol independent 2026-11-02 22:1x MST), which carries no timedOutAfterMs key. That string is read as
    {timedOutAfterMs: its duration}, so the timeout is not taken for a return."""
    t = r.get("toolUseResult")
    if isinstance(t, dict):
        return t
    if isinstance(t, str):
        m = TUR_TIMEOUT_RE.search(t)
        if m and any(m.groups()):
            h, mi, se = (float(x) if x else 0.0 for x in m.groups())
            return {"timedOutAfterMs": int(round((h * 3600 + mi * 60 + se) * 1000)), "error_text": t[:200]}
    return {}

PGREP_LOOP_RE = re.compile(r"\b(?:while|until)\s+(?:!\s*)?p(?:grep|kill)\s+((?:-\S+\s+)*)(\"[^\"]*\"|'[^']*'|\$\([^)]*\)\S*|[^\s;&|>]+)")
# a pgrep -f joined into the loop's condition by || (G-kvgate-sonnet-L1, 2026-10-29: `while [ -s jobs.txt ] || pgrep -f worker.sh
# >/dev/null; do sleep 30; done`): the condition holds while the call's shell matches, whatever the test before it reads
PGREP_OR_LOOP_RE = re.compile(r"\b(?:while)\s+[^;\n]*?\|\|\s*(?:!\s*)?p(?:grep|kill)\s+((?:-\S+\s+)*)(\"[^\"]*\"|'[^']*'|[^\s;&|>]+)")
# the counting form (the study, 2026-10-24 00:47Z, G-h3-haiku-L0: `for i in $(seq 1 170); do n=$(pgrep -fc "python runs/");
# if [ "$n" -eq 0 ]; then break; fi; sleep 60; done`): a loop whose exit is a pgrep -f count reaching 0 inside the loop body, the
# pattern on the call's own command line, so the count includes the call's shell and never reaches 0
PGREP_COUNT_LOOP_RE = re.compile(r"\b(?:for|while|until)\b[^\n]*?\$\(\s*p(?:grep)\s+((?:-\S+\s+)*)(\"[^\"]*\"|'[^']*'|[^\s;&|>)]+)\s*\)"
                                 r"[^\n]*?(?:-eq|-le|-lt\s*['\"]?1|==|=)\s*['\"]?0?['\"]?[^\n]*?\bbreak\b")
HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1.*?\n\2\s*(?:\n|$)", re.S)
# a nested shell, `bash -c "..."` / `sh -c "..."` with double quotes, run by the call itself (not inside a function the call only defines)
NESTED_SHELL_RE = re.compile(r'\b(?:bash|sh)\s+-c\s+"((?:[^"\\]|\\.)*)"')
FUNC_DEF_RE = re.compile(r'(?:\bfunction\s+)?\b([A-Za-z_]\w*)\s*\(\s*\)\s*\{')
SLEEP_RE = re.compile(r"(?:^|[;&|\s(])sleep\s+(\d+(?:\.\d+)?)")

def self_matching_wait(d, st):
    """A wait that cannot end: a Bash call whose loop condition is
    `pgrep -f <pattern>` (or pkill -f) and whose own command line contains <pattern>, so the call's shell matches itself
    and the loop runs while the call does (procps-ng 4.0.4 on the c3s, checked 2026-10-22). Per call: its start, whether
    it ran in the foreground or with run_in_background, and how it ended: returned (a result came), timed_out (Claude
    Code's Bash timeout moved it to the background or killed it), background_cap (a run_in_background wait killed at Claude Code's
    background time limit), or hung_to_teardown (no result before the run ended).
    The pattern is the matching text fnmatch-free: pgrep's regex is taken literally, which is what a shell command line
    containing it means for -f. Root and subagents alike; a label, not a verdict."""
    files = [f for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
             if "/codex/" not in f]
    a0 = st.get("agent_start_wall")
    out = []
    for f in files:
        who = "subagent" if "/subagents/" in f else "root"
        calls, ends, bg_task, bg_done, bg_capped = {}, {}, {}, {}, set()
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            content = (r.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and b.get("name") == "Bash" and r.get("timestamp"):
                    inp = b.get("input") or {}
                    cmd = inp.get("command") or ""
                    # a loop written into a file through a heredoc runs later in that script's own process, whose
                    # command line is the script's name, not the pattern (G-passbar-haiku-L0's queue.sh): only the
                    # command as the call's shell runs it counts
                    _body = HEREDOC_RE.sub("", cmd)
                    for m in list(PGREP_LOOP_RE.finditer(_body)) + list(PGREP_OR_LOOP_RE.finditer(_body)) + list(PGREP_COUNT_LOOP_RE.finditer(_body)):
                        if "f" not in "".join(o.lstrip("-") for o in m.group(1).split()):
                            continue
                        pat = m.group(2).strip("\"'")
                        # a pattern built by the shell ($(cat job.pid), $PID) is not on the command line as written
                        if pat and "$" not in pat and pat in cmd:
                            calls[b.get("id")] = (_wall(r["timestamp"]), cmd, pat, bool(inp.get("run_in_background")))
                            break
                    else:
                        # a wait inside a double-quoted `bash -c "while pgrep -f '...$s...'"`: the OUTER shell expands $s the same way in
                        # the pattern and in the inner shell's own argv, so the inner shell matches itself (G-fyld-haiku-L3-r2's four
                        # waiters, `while pgrep -f 'c4/train_gpt.py --seed $s'`; Sol independent 2026-11-02 19:4x MST). An escaped \$ is
                        # left to the inner shell and is not on its argv; a `bash -c` inside a function the call only defines never runs
                        # here (G-g4na-haiku-L1's lane(), its loops in lanes.sh)
                        for nm in NESTED_SHELL_RE.finditer(_body):
                            # inside a function body this call defines: name(){ ... } spans from its brace to the next "}" at the
                            # same depth; the wait runs only if the call later invokes that name
                            _fn = None
                            for fm in FUNC_DEF_RE.finditer(_body):
                                depth, j = 0, fm.end() - 1
                                while j < len(_body):
                                    depth += {"{": 1, "}": -1}.get(_body[j], 0)
                                    if depth == 0:
                                        break
                                    j += 1
                                if fm.end() - 1 < nm.start() < j:
                                    _fn = (fm.group(1), j)
                                    break
                            if _fn and not re.search(r"(?m)(?:^|[;&|(]\s*)" + re.escape(_fn[0]) + r"(?:\s|$)(?!\s*\(\s*\))", _body[_fn[1] + 1:]):
                                continue
                            inner = nm.group(1)
                            for m in list(PGREP_LOOP_RE.finditer(inner)) + list(PGREP_OR_LOOP_RE.finditer(inner)) + list(PGREP_COUNT_LOOP_RE.finditer(inner)):
                                if "f" not in "".join(o.lstrip("-") for o in m.group(1).split()):
                                    continue
                                pat = m.group(2).strip("\"'")
                                if pat and "\\$" not in pat and "$(" not in pat:
                                    calls[b.get("id")] = (_wall(r["timestamp"]), cmd, pat, bool(inp.get("run_in_background")), True)
                                    break
                            if b.get("id") in calls:
                                break
                elif b.get("type") == "tool_result" and r.get("timestamp"):
                    t = tool_use_result(r)
                    if t.get("backgroundTaskId") and not t.get("timedOutAfterMs"):
                        # a run_in_background call returns at once with its task id (G-g4na-sonnet-L3, 2026-10-24: open_s read
                        # 0.016 s for a wait that never ended); the wait itself ends with the task's completion notice, or never
                        bg_task[b.get("tool_use_id")] = t["backgroundTaskId"]
                        continue
                    ends[b.get("tool_use_id")] = (_wall(r["timestamp"]), "timed_out" if t.get("timedOutAfterMs") else "returned",
                                                  t.get("timedOutAfterMs"))
            if r.get("type") == "user" and r.get("timestamp"):
                txt = json.dumps((r.get("message") or {}).get("content") or "")
                for tid in re.findall(r"<task-id>([\w-]+)</task-id>", txt):
                    if "<status>" in txt:
                        bg_done.setdefault(tid, _wall(r["timestamp"]))
                # a background wait Claude Code killed at its background time limit (BASH_MAX_TIMEOUT_MS, 10 h on the boxes) also ends
                # with a <status> notice, so it read "returned" (G-k2p-sonnet-L0-v2-r3's two `while pgrep -f ...worker.sh` waits, each
                # open 36,000 s; the analysis 06:19 MST 11-07): the notice says killed and names the limit, so it is the cap
                for blk in re.findall(r"<task-notification>(.*?)</task-notification>", txt, re.S):
                    m = re.search(r"<task-id>([\w-]+)</task-id>", blk)
                    if m and re.search(r"<status>\s*killed\s*</status>", blk) and "background time limit" in blk:
                        bg_capped.add(m.group(1))
        for k, v in calls.items():
            t0, cmd, pat, bg = v[:4]
            nested = len(v) > 4
            e = ends.get(k)
            if e is None and bg and k in bg_task and bg_task[k] in bg_done:
                e = (bg_done[bg_task[k]], "background_cap" if bg_task[k] in bg_capped else "returned", None)
            how = e[1] if e else "hung_to_teardown"
            end = e[0] if e else (st.get("teardown_wall") or st.get("harness_end_wall"))
            # a nested shell is launched with nohup ... & by the call, which returns at once: the wait runs in the detached inner
            # shell, so its own end is not the call's; "detached" says so, and open_s is the call's, not the wait's
            out.append({"who": who, "call_s": (t0 - a0) if a0 else None, "background": bg, "pattern": pat[:80],
                        "command": cmd[:200], "ended": "detached" if nested else how, "timed_out_after_ms": e[2] if e else None,
                        "open_s": None if nested else ((end - t0) if end else None), "nested_shell": nested})
    out.sort(key=lambda x: x["call_s"] or 0)
    # a nested-shell wait is neither foreground nor a Bash background task: it runs detached from the call (nohup ... &), so it
    # never blocks the agent's turn; it can only hold the work queued behind it (the GPU idles until the agent notices)
    return {"calls": out, "foreground": sum(1 for x in out if not x["background"] and not x.get("nested_shell")),
            "hung_to_teardown": sum(1 for x in out if x["ended"] == "hung_to_teardown"),
            "timed_out": sum(1 for x in out if x["ended"] == "timed_out"),
            "detached": sum(1 for x in out if x.get("nested_shell")),
            # background waits (run_in_background, not a nested shell), and those of them the background time limit ended
            "background": sum(1 for x in out if x["background"] and not x.get("nested_shell")),
            "background_cap": sum(1 for x in out if x["ended"] == "background_cap")} if out else None

def sleep_lengths(d):
    """The waits the agent chose: every `sleep N`
    in a root-session Bash call, foreground or background, with the call's own `timeout` argument when it set one."""
    files = [f for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
             if "/subagents/" not in f and "/codex/" not in f]
    fg, bg, call_timeouts = [], [], []
    for f in files:
        for line in open(f, errors="replace"):
            if '"Bash"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            content = (r.get("message") or {}).get("content")
            for b in content if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    inp = b.get("input") or {}
                    s = [float(x) for x in SLEEP_RE.findall(inp.get("command") or "")]
                    (bg if inp.get("run_in_background") else fg).extend(s)
                    if inp.get("timeout") is not None:
                        call_timeouts.append(inp["timeout"])
    if not (fg or bg or call_timeouts):
        return None
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    return {"foreground_sleeps_s": fg, "background_sleeps_s": bg, "foreground_max_s": max(fg) if fg else None,
            "foreground_median_s": med(fg), "call_timeouts_ms": call_timeouts}

def replay_exhausted_in_process(d):
    """Ranks served past the end of their record (the study's scan, /tmp/p006-coord-gen/pastend.py; gpu-mock's
    definition, 2026-10-22 04:33Z): a rank whose enqueue or materialise step exceeds the `steps` of the record its last
    lookup served, with no later lookup for it. On the pairs in service that is a script looping trials in one process
    (the step counter never resets), and its reads from then on are the record's last value repeated, so the values it
    reported are not a real run's. On the per-trial pair (b23e9a84) a trial boundary makes a new lookup, so the same
    condition means one trial outlasting its record. A label; analysis decides what it voids."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    last_lookup, max_step, after, marked = {}, {}, {}, {}
    for line in open(occ, errors="replace"):
        if '"lookup"' not in line and '"enqueue"' not in line and '"materialise"' not in line and '"record_exhausted"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k, j = e.get("kind"), e.get("job")
        key = (j, e.get("trial") or 1)
        if k == "record_exhausted":
            # the revision after b23 marks the read past a record's end itself: one event per
            # (job, trial), with record_steps and step; its later reads are tier extrapolated, not the last value repeated
            marked[key] = e
            continue
        if k == "lookup":
            last_lookup[key] = e; max_step[key] = -1; after[key] = 0
        elif k in ("enqueue", "materialise") and isinstance(e.get("step"), (int, float)):
            max_step[key] = max(max_step.get(key, -1), e["step"])
    hits = [{"rank": key[0], "trial": key[1], "record_id": (last_lookup.get(key) or {}).get("record_id"),
             "record_steps": e.get("record_steps"), "max_step": max(max_step.get(key, -1), e.get("step") or -1),
             "past_end_steps": None, "marked": True, "tier": "extrapolated"} for key, e in marked.items()]
    for key, lk in last_lookup.items():
        if key in marked:
            continue
        rid = lk.get("record_id")
        try:
            n = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json"))).get("steps") if rid else None
        except (OSError, ValueError):
            n = None
        if n and max_step.get(key, -1) > int(n):
            hits.append({"rank": key[0], "trial": key[1], "record_id": rid, "record_steps": int(n), "max_step": max_step[key],
                         "past_end_steps": max_step[key] - int(n)})
    return {"ranks": len(hits), "hits": sorted(hits, key=lambda h: (h["rank"], h["trial"]))[:50],
            "how": "a rank's enqueue or materialise step above its last-served record's steps, with no later lookup"}

TRIAL_LINE_RE = re.compile(r"num_trials\s*=\s*int\(\s*sys\.argv\[-1\]\s*\)\s*if\s+len\(\s*sys\.argv\s*\)\s*>\s*1\s+else\s+1")

def trial_loop_launches(d, rx=None):
    """Launches whose script loops trials on its last argument (gpu-tasks, from the study's audit, 2026-10-22: 27
    served scripts carry `num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1`, so any trailing argument is the
    trial count, the value after --seed included). Matched by the script's own content in the workspace, that line
    unchanged (an agent that rewrote it, as Astra did to an argparse positional, is not matched). Per launch:
      trial_loop_unserved  a trailing integer >= 2 on an image before per-trial serving (no `trial` field in the run's
                           events): one real run, then the record's last values repeated; counts as one instance, not N
      trial_loop_served    the same on the per-trial pair (b23): each trial its own record, read as its own rank
      trial_loop_zero      a trailing 0: the real script also trains nothing (a 2-s exit), faithful, not a defect
    From the box's process records (resources.jsonl) and the workspace copy of each script."""
    ws = os.path.join(d, "workspace", "home", "project")
    res = os.path.join(d, "resources.jsonl")
    if not (os.path.isdir(ws) and os.path.exists(res)):
        return None
    occ = os.path.join(d, "occupancy.jsonl")
    per_trial = False
    if os.path.exists(occ):
        for line in open(occ, errors="replace"):
            if '"trial"' in line:
                per_trial = True
                break
    loops = {}
    def loops_on_last_arg(path):
        if path not in loops:
            try:
                loops[path] = bool(TRIAL_LINE_RE.search(open(path, errors="replace").read()))
            except OSError:
                loops[path] = None
        return loops[path]
    seen, out = set(), []
    for line in open(res, errors="replace"):
        if '"started"' not in line:
            continue
        try:
            l = json.loads(line)
        except ValueError:
            continue
        for q in l.get("started") or []:
            cmd = (q.get("cmdline") or "") if isinstance(q, dict) else ""
            toks = cmd.split()
            py = [i for i, t in enumerate(toks) if t.endswith(".py") and "/site-packages/" not in t and not t.endswith("torchrun")]
            if not py or "torchrun" in toks[0] or q.get("comm") in ("torchrun",):
                continue
            script, args = toks[py[0]], toks[py[0] + 1:]
            key = (q.get("pid"), cmd)
            if key in seen or not args or not re.fullmatch(r"-?\d+", args[-1]):
                continue
            seen.add(key)
            rel = script.split("/home/researcher/project/", 1)[-1]
            path = os.path.join(ws, rel) if not os.path.isabs(rel) else os.path.join(d, "workspace", rel.lstrip("/"))
            if not loops_on_last_arg(path):
                continue
            n = int(args[-1])
            kind = ("trial_loop_zero" if n == 0 else
                    "trial_loop_served" if n >= 2 and per_trial else
                    "trial_loop_unserved" if n >= 2 else None)
            if kind:
                out.append({"label": kind, "trials": n, "script": rel, "argv_tail": " ".join(args)[-80:], "t_utc": l.get("t_utc")})
    if not out:
        return {"launches": 0, "unserved": 0, "served": 0, "zero": 0, "items": []}
    return {"launches": len(out), "unserved": sum(1 for x in out if x["label"] == "trial_loop_unserved"),
            "served": sum(1 for x in out if x["label"] == "trial_loop_served"), "zero": sum(1 for x in out if x["label"] == "trial_loop_zero"),
            "items": out[:40], "how": "the trailing integer of a launch whose workspace script has the unchanged sys.argv[-1] trial line"}

def near_misses(d):
    """Launches of a script the registry knows that matched no record (gpu-mock, 2026-10-22 23:37Z: kind near_miss,
    logged just before its lookup, with the closest record and the argv tokens each side has and the other lacks).
    A near miss on a task job's script is a launch the testbed served as unknown: X-scq-astra-S1's 658 s stall on a
    `match: none` launch was the argument-order defect, not the agent's scheduling. Per near miss: the rank, the closest
    record, the differing tokens, and whether its lookup then matched none. A label, never a validity change: the
    the study decides whether a run so served is a testbed defect."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    nm, none_lookup = [], set()
    for line in open(occ, errors="replace"):
        if '"near_miss"' not in line and '"lookup"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "near_miss":
            nm.append({"rank": e.get("job"), "stage": e.get("stage"), "closest_record": e.get("record_id"),
                       "only_launch": e.get("only_launch"), "only_record": e.get("only_record"),
                       "records_of_script": e.get("records_of_script")})
        elif e.get("kind") == "lookup" and e.get("match") == "none":
            none_lookup.add(e.get("job"))
    for x in nm:
        x["lookup_matched_none"] = x["rank"] in none_lookup
    return {"near_misses": len(nm), "items": nm} if nm else None

def off_plan_launches(d, task):
    """Launches the brief's rule did not ask for (gpu-tasks-h3, 2026-10-23 17:18Z, for pkg-h3-chains; the analysis reads
    them as decision errors, not loss). gate_violation: a lookup whose record is one of a gate decision's stopped pieces
    (decisions[].stopped[].records: the runs the brief's cut says go no further, served exactly but listed by no task job).
    off_plan_whole_run: a lookup whose record is a whole run that no task job lists (the agent ran a run in one launch instead of
    the pieces). A whole run is a segment record covering its parent from its first step to its last, identity.derived_from kind
    "segment" with steps [1, parent_steps] (gpu-tasks 2026-11-01 16:13Z; the analysis 09:15 MST: pkg-rounds, -8gpu and
    pkg-chain-side as on H3'), not a name: H3''s `-h3seg0-500`/`-1000` first pieces are unlisted on rounds and are no whole run.
    Neither is work remaining; labels, not validity. None when the task has no gate decision and no such lookup."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    stopped = {}
    for dec in task.get("decisions") or []:
        for x in dec.get("stopped") or []:
            for rid in x.get("records") or []:
                stopped[rid] = {"decision": dec.get("decision_id"), "piece": x.get("job_id")}
    listed = {r for j in task.get("jobs") or [] for r in j.get("records") or []}
    gate, whole = [], []
    for line in open(occ, errors="replace"):
        if '"lookup"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        rid = e.get("record_id")
        if e.get("kind") != "lookup" or not rid:
            continue
        if rid in stopped:
            gate.append(dict(stopped[rid], rank=e.get("job"), record_id=rid, t_wall=e.get("t_wall")))
        elif rid not in listed and _is_whole_run(rid):
            whole.append({"rank": e.get("job"), "record_id": rid, "t_wall": e.get("t_wall")})
    if not stopped and not whole:
        return None
    return {"gate_violation": len(gate), "gate_items": gate[:20], "off_plan_whole_run": len(whole), "whole_run_items": whole[:20],
            "how": "lookups whose record is a gate decision's stopped piece, or a whole run no task job lists"}

_WHOLE_RUN = {}
def _is_whole_run(rid):
    """A segment record that covers its parent whole: identity.derived_from {kind: segment, steps: [1, parent_steps]}. False when the
    record is missing or is no segment. Read once per record."""
    if rid not in _WHOLE_RUN:
        try:
            df = (json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json"))).get("identity") or {}).get("derived_from") or {}
        except (OSError, ValueError):
            df = {}
        _WHOLE_RUN[rid] = (df.get("kind") == "segment" and isinstance(df.get("steps"), list)
                           and df.get("parent_steps") is not None and df["steps"] == [1, df["parent_steps"]])
    return _WHOLE_RUN[rid]

def _first_val_step(rid):
    """The first step at which a record logs validation (series.val_loss.steps[0]), or None when the record is missing or logs none."""
    if not rid:
        return None
    try:
        r = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
    except (OSError, ValueError):
        return None
    st = (((r.get("series") or {}).get("val_loss") or {}).get("steps") or [])
    return int(st[0]) if st else None

TELL_CLASS_ORDER = ("traceback_through_patch", "probed_runtime", "inspected_box_tooling", "inferred_from_served_values",
                    "saw_mock_name")

def ledger_list_usd(d):
    """Sum of the ledger's priced rows at list price (upstream_usd where the router has it, else usd), rounded to cents; None without a
    ledger or with no priced row."""
    lp = os.path.join(d, "ledger.jsonl")
    if not os.path.exists(lp):
        return None
    tot, priced = 0.0, 0
    for line in open(lp, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        v = r.get("upstream_usd") if r.get("upstream_usd") is not None else r.get("usd")
        if v is not None:
            tot += float(v); priced += 1
    return round(tot, 2) if priced else None

# analysis_labels that void a run by hand: stopped by us for a reason outside the run, never the agent's doing.
# testbed_outage: the controller's 01:44-08:15 MDT panic took the gate away, so a
# session that died on its 502 or was left with no agent is void, not the agent's failure. stopped_for_r6 (the analysis
# 2026-10-29 09:51 MDT): a run the study stopped to free its machine for gpu-mock's r6 decode, requeued.
# lifecycle_cron_unsupported: the agent ended its turn on a
# CronCreate check, and our headless lifecycle closed the session where the interactive product would have fired the cron
# disengaged_smoke_ran_real: a correct whole_job refusal ran the agent's honest
# subset smoke test for real on the CPU device, an impossible observation that led it to the box; void, in attempts.py's simulator column
# zero_mib_on_busy_gpu and killed_because_of_zero_mib: the simulator showed 0 MiB on a busy GPU
# (records with no memory figure) and the agent acted on it; void, in attempts.py's simulator column
HAND_VOID = ("stopped_to_protect_subscription", "testbed_outage", "stopped_for_r6", "lifecycle_cron_unsupported", "disengaged_smoke_ran_real",
             "unmatched_launch_ran_real", "scheduler_deadlock",
             "zero_mib_on_busy_gpu", "killed_because_of_zero_mib",
             # closed_by_work_done_grace_with_gpu_work_live: the work-done grace
             # counted from the task jobs' end while the agent's own gpu-harness runs were still training; void (testbed)
             "closed_by_work_done_grace_with_gpu_work_live",
             # closed_by_work_done_grace_before_wait_bound: the grace closed
             # the session 1,800 s after the work while the agent's bounded background wait still had about 25 min to its timeout; void (testbed)
             "closed_by_work_done_grace_before_wait_bound",
             # closed_by_work_done_grace_before_wait_returned (the analysis 2026-11-05 21:26 MST, G-g4na-astracc-L0-r2, G-g4na-fable-L0-r2,
             # G-g4nb-astracc-L0-r2): the grace closed the session while the agent's own wait would have returned before the horizon; void (testbed)
             "closed_by_work_done_grace_before_wait_returned")

def hand_voids(labels):
    """The (label, value) pairs that void a run: every HAND_VOID label present, except one written note-only. A label whose value is a dict
    with "note_only": true is a note, not a void (the analysis 15:19 MST 11-12: unmatched_launch_ran_real on nine runs whose
    readings stand, with the missed minutes and a bound; the label's other uses stay voids)."""
    out = []
    for hv in HAND_VOID:
        v = labels.get(hv)
        if v and not (isinstance(v, dict) and v.get("note_only") is True):
            out.append((hv, v))
    return out

def controller_crash(d, sc):
    """The reason a scripted controller run is invalid, or None: no job reached the daemon (score_run launched_jobs 0) and the
    controller crashed (a Traceback in agent-logs/controller.log, which plan.py's controller policy writes; its last line is
    the error). A scripted run without a controller log, an agent run, or one that launched anything reads None."""
    if (_idle_run_inputs(d).get("harness") or "scripted") != "scripted" or sc.get("launched_jobs") != 0:
        return None
    p = os.path.join(d, "agent-logs", "controller.log")
    if not os.path.exists(p):
        return None
    lines = [l.rstrip() for l in open(p, errors="replace") if l.strip()]
    if not any(l.startswith("Traceback (most recent call last)") for l in lines):
        return None
    return f"controller crashed before any launch: {lines[-1][:200]} (agent-logs/controller.log; 0 jobs reached the daemon)"

def tell_class(tl, ri, al):
    """How the mock showed itself to the agent, on every run (the analysis 2026-10-24 23:14Z: the detectability table
    needs it on every run). The classes are the channels agreed at 14:20Z plus the one analysis named at 22:49Z:
    traceback_through_patch (an error handed it a frame in the mock's files), probed_runtime (a result named the mock's modules
    in answer to a call that named none), inspected_box_tooling (the agent named the mock's path itself),
    inferred_from_served_values (the agent's own words doubt that the environment or what it served is genuine, tells.py
    DOUBT_RE: G-g4na-sonnet-L0's "looks suspiciously synthetic", "replaying old logs", "timings might be simulated rather than
    measured"; or analysis's hand label), and saw_mock_name (a mock name in a tool result through none of these). tell_class is
    the earliest by its first time (a class with no time, such as a frame found only in a workspace file, goes after the timed
    ones); analysis's hand label wins it; "none" when no channel fired. Words about the evaluator ("quantisation is simulated")
    are not doubt. artifacts_remarked lists the agent remarking on a served artifact while blaming something else (an identical
    rerun put down to seeds, 0 MiB beside full utilisation put down to a reporting quirk): not a tell class, listed so analysis
    can rule on it. A label, never validity."""
    tl, ri, al = tl or {}, ri or {}, al or {}
    first = {}
    if tl.get("traceback_through_patch"):
        first["traceback_through_patch"] = ((tl.get("patch_frames_in_tool_results") or [{}])[0] or {}).get("utc")
    if tl.get("probed_runtime"):
        first["probed_runtime"] = ((tl.get("probed_runtime_results") or [{}])[0] or {}).get("utc")
    if tl.get("inspected_box_tooling"):
        first["inspected_box_tooling"] = ((tl.get("box_tooling_calls") or [{}])[0] or {}).get("utc")
    hand = al.get("tell_class")
    if tl.get("doubted_served_values"):
        first["inferred_from_served_values"] = ((tl.get("doubted_served_values_items") or [{}])[0] or {}).get("utc")
    if tl.get("saw_tell") and not first:
        first["saw_mock_name"] = tl.get("first_tell_utc")
    if hand and hand not in first:
        first[hand] = None
    order = sorted(first, key=lambda k: (first[k] is None, first[k] or "", TELL_CLASS_ORDER.index(k) if k in TELL_CLASS_ORDER else 99))
    art = []
    if ri.get("rerun_identical_values") and ri.get("agent_remarked"):
        x = (ri.get("remarks") or [{}])[0] or {}
        art.append({"artifact": "rerun_identical_remarked", "utc": x.get("utc"), "context": (x.get("context") or "")[:200]})
    if tl.get("zero_mib_remarked"):
        x = (tl.get("zero_mib_remarks") or [{}])[0] or {}
        art.append({"artifact": "zero_mib_remarked", "utc": x.get("utc"), "context": (x.get("context") or "")[:200]})
    return {"tell_class": hand or (order[0] if order else "none"), "tell_classes": order, "first_utc": first,
            "artifacts_remarked": art or None,
            "how": "analysis's hand label (run_stop analysis_labels.tell_class)" if hand else
                   "the earliest channel by first time" if order else "no channel fired"}

def no_gpu_job_by_horizon(d, st, task):
    """Whether any task job reached a GPU before the window closed, split two ways:
    no_gpu_job_by_horizon, the agent launched no task job at all before the window closed (G-h4-haiku-S1 queued seven 42-CPU prep
    jobs ahead of its 21 GPU training jobs in one SLURM queue and reached its 57,600 s horizon with none started): agent behaviour,
    valid with no deliverable, its ratio a lower bound capped by the horizon; and modelled_path_only, task jobs were launched but none
    reached a GPU (G-h1-haiku-L0, G-h1-haiku-L3, G-mlskv-haiku-L0-r3, W0-h4-measure): a testbed defect. A task-job launch is a
    job_hello whose script is one of the task's launch scripts, or whose lookup names a task job or has a near_miss (a script with
    records); a prep step or probe that sends a hello with gpu -1 is not one. None when a task job reached a GPU (a busy_start, or a
    task-job hello on gpu >= 0), or the run has no occupancy log or the task no jobs."""
    occ = next((p for p in (os.path.join(d, "occupancy.jsonl"), os.path.join(d, "occupancy.jsonl.gz")) if os.path.exists(p)), None)
    jobs = task.get("jobs") or []
    if not occ or not jobs:
        return None
    import gzip as _gz
    scripts = {os.path.basename(t) for j in jobs for t in str(j.get("launch") or "").split() if t.endswith((".py", ".sh"))}
    end = st.get("teardown_wall") or st.get("harness_end_wall")
    hello, look, near, busy = {}, {}, set(), False
    for line in (_gz.open(occ, "rt") if occ.endswith(".gz") else open(occ, errors="replace")):
        if not any(k in line for k in ('"job_hello"', '"busy_start"', '"lookup"', '"near_miss"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if end and (e.get("t_wall") or 0) > end:
            continue
        k = e.get("kind")
        if k == "busy_start":
            busy = True
            break
        if k == "job_hello":
            hello.setdefault(e.get("job"), e)
        elif k == "lookup":
            look[e.get("job")] = e
        elif k == "near_miss":
            near.add(e.get("job"))
    if busy:
        return None
    def is_task(j, h):
        argv = [os.path.basename(str(a)) for a in (h.get("argv") or [])]
        return bool(scripts & set(argv)) or bool((look.get(j) or {}).get("task_job")) or j in near
    task_hellos = {j: h for j, h in hello.items() if is_task(j, h)}
    if any((isinstance(h.get("gpu"), int) and h["gpu"] >= 0) or h.get("gpus") for h in task_hellos.values()):
        return None
    if not task_hellos:
        return {"no_gpu_job_by_horizon": True, "modelled_path_only": False, "task_job_launches": 0, "other_hellos": len(hello)}
    return {"no_gpu_job_by_horizon": False, "modelled_path_only": True, "task_job_launches": len(task_hellos), "other_hellos": len(hello) - len(task_hellos)}

def codex_turn_failed(d, st):
    """A Codex root turn that ended failed, not completed (G-kvrep-astra-L0-r3, 2026-10-24: at 1,853 s the provider answered
    "Selected model is at capacity" (codexErrorInfo serverOverloaded, willRetry false), the thread went to systemError and the
    turn closed failed; codex_session counts any turn/completed as the answer, so run_one closed the session on it and the
    box ran the agent's queued jobs to 47,709 s with no agent). The provider ended the session, not the agent: its "report" is
    the last text before the failure. From box/codex-rpc.jsonl. None when the run has no Codex rpc log or no failed turn.
    Resumed turns (the Codex driver and the analysis, 2026-10-28 22:20Z: codex_session resumes a capacity-failed root
    turn with a "continue" turn/start on the same thread, after a 30-480 s backoff, up to 5 times): a failed turn followed by
    an outbound turn/start on its thread is a resume, listed in `resumed` with its gap, and not a failure. Only a run whose
    last root-thread turn failed with no turn/start after it has failed_turns; a run that only resumed returns
    {"failed_turns": 0, "items": [], "resumed": [...]} for the label resumed_after_capacity."""
    p = os.path.join(d, "box", "codex-rpc.jsonl")
    if not os.path.exists(p):
        return None
    a0 = st.get("agent_start_wall")
    at = lambda t: round((t or 0) - a0, 1) if a0 else None
    fails, starts = [], []   # (t, thread, item) per failed turn; (t, thread) per outbound turn/start
    for line in open(p, errors="replace"):
        if '"turn/completed"' not in line and '"turn/start"' not in line:
            continue
        try:
            x = json.loads(line)
        except ValueError:
            continue
        m = x.get("msg") or {}
        pr = m.get("params") or {}
        if x.get("dir") == "out" and m.get("method") == "turn/start":
            starts.append((x.get("t") or 0, pr.get("threadId")))
            continue
        t = pr.get("turn") or {}
        if m.get("method") == "turn/completed" and t.get("status") != "completed":
            err = t.get("error") or {}
            fails.append((x.get("t") or 0, pr.get("threadId"), {"at_s": at(x.get("t")), "status": t.get("status"),
                          "error": (err.get("message") or "")[:200], "codex_error": err.get("codexErrorInfo")}))
    out, resumed = [], []
    for tf, th, it in fails:
        nxt = next((ts for ts, sth in starts if ts >= tf and (sth == th or th is None)), None)
        if nxt is None:
            out.append(it)
        else:
            resumed.append({**it, "resumed_at_s": at(nxt), "gap_s": round(nxt - tf, 1)})
    if not out and not resumed:
        return None
    return {"failed_turns": len(out), "items": out, "resumed": resumed}

def jobs_live_at_report(d, st, harness, pick_rep=None):
    """The task's jobs still running when the agent gave its final report (the study and the analysis, 2026-10-24
    10:49Z): ranks whose hello came before the report and whose bye came after it (or never), each with its record and the
    recorded seconds it ran on after the report. The report is the root session's last end_turn (Claude Code), as
    report_latency reads it; a scripted run makes none. Whether the report called such a job useless is analysis' reading,
    by hand (kept_known_useless_job_running); this lists the candidates. None when the run has no report or no occupancy log."""
    if harness in (None, "scripted"):
        return None
    occ = os.path.join(d, "occupancy.jsonl")
    a0 = st.get("agent_start_wall")
    if not os.path.exists(occ) or a0 is None:
        return None
    ends = []
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "assistant" and r.get("timestamp") and (r.get("message") or {}).get("stop_reason") == "end_turn":
                ends.append(_wall(r["timestamp"]))
    if not ends:
        return None
    rep = pick_rep[0] if pick_rep else max(ends)
    hello, bye, rec = {}, {}, {}
    speed = 1.0
    for line in open(occ, errors="replace"):
        if not any(k in line for k in ('"job_hello"', '"job_bye"', '"lookup"', '"daemon_start"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = e.get("kind")
        if k == "job_hello":
            hello[e["job"]] = e.get("t_wall")
        elif k == "job_bye":
            bye[e["job"]] = e.get("t_wall")
        elif k == "lookup":
            rec.setdefault(e["job"], e.get("record_id"))
        elif k == "daemon_start" and e.get("speed"):
            speed = float(e["speed"])
    live = [{"rank": j, "record_id": rec.get(j), "ran_after_report_s": round(((bye.get(j) or max(list(bye.values()) + [rep])) - rep) * speed, 1)}
            for j, t in sorted(hello.items()) if t is not None and t <= rep and (bye.get(j) is None or bye[j] > rep)]
    return {"jobs_live_at_report": len(live), "items": live[:20], "report_s": round((rep - a0) * speed, 1),
            "how": "ranks whose hello came before the report (the root session's last end_turn, or for an mls-kv task the last "
                   "that states a pick, pick_report) and whose bye came after it"}

def wide_launches(d, task):
    """Launches wider than one process for a task whose scripts are single-process (gpu-tasks-h3, 2026-10-23 19:41Z: H3' task
    957de6c6's records lacked launch.process_model, so fix:gpumock-single-process could not pace a torchrun --nproc_per_node>1
    launch at one GPU; such a launch got the multi-GPU speed-up). A launch is its ranks sharing one job_key (torchrun's rendezvous);
    wide when it has more than one rank. Reported only for a task whose jobs all have gpu_options [1]. None otherwise."""
    if not task.get("jobs") or any(list(j.get("gpu_options") or [1]) != [1] for j in task["jobs"]):
        return None
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    ranks, first = {}, {}
    for line in open(occ, errors="replace"):
        if '"lookup"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "lookup":
            k = e.get("job_key") or f"rank:{e.get('job')}"
            ranks.setdefault(k, []).append(e.get("job"))
            first.setdefault(k, e)
    wide = [{"job_key": k, "ranks": v, "record_id": first[k].get("record_id"), "task_job": first[k].get("task_job")}
            for k, v in ranks.items() if len(v) > 1]
    return {"wide_launches": len(wide), "launches": len(ranks), "items": wide[:20],
            "how": "lookups grouped by job_key (one launch); more than one rank on a task whose jobs all take 1 GPU"}

def silent_failure(d, task, ev=None):
    """A silent-failure package's waste and repair (pkg-sf-zero-lr: Astra independent and the analysis, 2026-11-02 09:25 MST:
    primary evidence is the first effective-LR print, after the first optimizer step). Per logical job with silent_failure: its broken
    launches (served a broken_records curve; idle.work_remaining's broken_launches) and its first launch served a successful record.
    For each broken launch: start, end, its GPU-seconds, and the wasted GPU-seconds after its evidence, end minus (start +
    silent_failure.first_update_evidence_s / speed), with the first_eval figure beside it (evidence_s); whether it ran to the end or was
    stopped early. repair_delay_s: from the first broken launch's evidence to the fixed launch's start. how_repaired, per run:
    'pre_launch' (no job ever served a broken record), 'global' (every broken-as-briefed job's first launch already fixed after the
    first repair, i.e. fixed before it ran), 'local' (each repaired after its own broken attempt), 'mixed', or 'unrepaired' (a job never
    served a successful record). A label and fields, never validity. None for a task without silent_failure jobs."""
    jobs = [j for j in (task.get("jobs") or []) if j.get("silent_failure")]
    if not jobs:
        return None
    import idle as _idle, score_run as _sr
    occ = os.path.join(d, "occupancy.jsonl")
    if ev is None:
        if not os.path.exists(occ):
            return None
        ev = _sr.load_events(occ)
    rem = _idle.work_remaining(ev, task)
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    a0 = st.get("agent_start_wall")
    speed = float(_idle.run_inputs(d).get("speed") or 1.0)
    rel = lambda t: round((t - a0) * speed, 1) if (t is not None and a0) else None
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    tied = {}
    for dj, h in hello.items():
        tj = _idle.task_job_of(h, lk.get(dj), _idle.tie_jobs(task))
        if tj:
            tied.setdefault(tj, []).append(dj)
    by_job = {j["job_id"]: j for j in jobs}
    # the first optimizer update's LR, per rank, when the daemon made it available: its step-1 unit's enqueue (gpu-mock 12:30 MST
    # 11-02: the enqueue is the LR's availability and launch time, its pacing async; wall_cost is a duration, never an evidence time;
    # the analysis 12:22 MST: the evidence time is the first observable surface, never a modelled completion. The old
    # enqueue-plus-cost put the evidence after the agent had already read it, and repair delays read negative, W0-sfz-reference
    # -4.5 s). A launch with no step-1 unit falls back to the record's paced offset, labelled estimated
    step1 = {}
    for e in ev:
        if e["kind"] == "enqueue" and e.get("unit") == "step" and e.get("step") == 1 and e["job"] not in step1:
            step1[e["job"]] = e["t_wall"]
    broken = {}
    for b in rem.get("broken_launches") or []:
        broken.setdefault(b["task_job"], []).append(b)
    rows, waste, waste_eval, firsts_fixed = [], 0.0, 0.0, []
    for jid, j in by_job.items():
        sf = j["silent_failure"]
        good = set(sf.get("successful_records") or [])
        fixed = sorted((_idle.job_start(hello[d_]) for d_ in tied.get(jid, []) if (lk.get(d_) or {}).get("record_id") in good))
        bl = sorted(broken.get(jid, []), key=lambda b: b["start_wall"])
        items = []
        for b in bl:
            seen = [step1[r] for r in b["ranks"] if r in step1]
            ev_t = min(seen) if seen else b["start_wall"] + float(sf.get("first_update_evidence_s") or 0.0) / speed
            ev_e = b["start_wall"] + float(sf.get("evidence_s") or 0.0) / speed
            end = b["end_wall"]
            items.append({"start_s": rel(b["start_wall"]), "end_s": rel(end), "ran_to_end": b["ran_to_end"],
                          "first_update_s": rel(ev_t), "first_update_source": "daemon step-1 LR availability (its enqueue)" if seen else
                          "estimated: launch start + paced first_update_evidence_s (host set-up omitted)",
                          "first_eval_estimated_s": rel(ev_e),
                          "gpu_s": round((end - b["start_wall"]) * speed, 1) if end else None,
                          "wasted_after_first_update_s": round(max(0.0, end - ev_t) * speed, 1) if end else None,
                          "wasted_after_first_eval_s": round(max(0.0, end - ev_e) * speed, 1) if end else None})
            if end:
                waste += max(0.0, end - ev_t) * speed; waste_eval += max(0.0, end - ev_e) * speed
        first_fixed = fixed[0] if fixed else None
        _s1 = [step1[r] for r in (bl[0]["ranks"] if bl else []) if r in step1]
        _ev0 = (min(_s1) if _s1 else bl[0]["start_wall"] + float(sf.get("first_update_evidence_s") or 0.0) / speed) if bl else None
        delay = round((first_fixed - _ev0) * speed, 1) if (bl and first_fixed) else None
        # never a negative repair delay: a fix launched before the evidence time read means the
        # evidence came from a surface earlier than the one read here, so the delay is not reported, and the row says why
        delay_note = None
        if delay is not None and delay < 0:
            delay, delay_note = None, "the fixed launch started before the evidence time read; no delay reported"
        rows.append({"task_job": jid, "broken_as_briefed": bool(sf.get("broken_as_briefed")), "broken_launches": len(bl),
                     "broken": items, "fixed_start_s": rel(first_fixed), "repaired": first_fixed is not None,
                     "repair_delay_s": delay, "repair_delay_note": delay_note, "done": jid in set(rem.get("done_jobs") or [])})
        if sf.get("broken_as_briefed"):
            firsts_fixed.append((first_fixed is not None and not bl))
    bab = [r for r in rows if r["broken_as_briefed"]]
    unrep = [r["task_job"] for r in rows if not r["repaired"]]
    any_broken = any(r["broken_launches"] for r in rows)
    # how_repaired is a heuristic summary of the counts beside it, never proof of queue-wide reasoning:
    # 'global' when at most max(1, 10%) of the broken-as-briefed jobs ran broken
    if unrep:
        how = "unrepaired"
    elif not any_broken:
        how = "pre_launch"
    elif all(r["broken_launches"] for r in bab):
        how = "local"
    else:
        # some broken-as-briefed jobs were never run broken: after a repair the agent fixed the rest before launching them
        how = "global" if sum(1 for r in bab if r["broken_launches"]) <= max(1, len(bab) // 10) else "mixed"
    return {"jobs": len(rows), "broken_as_briefed": len(bab), "broken_launches": sum(r["broken_launches"] for r in rows),
            "jobs_run_broken": sum(1 for r in rows if r["broken_launches"]), "repaired": sum(1 for r in rows if r["repaired"]),
            "unrepaired": unrep, "done": sum(1 for r in rows if r["done"]),
            "wasted_gpu_s_after_first_update": round(waste, 1), "wasted_gpu_s_after_first_eval": round(waste_eval, 1),
            "repair_delay_median_s": (sorted(x for x in (r["repair_delay_s"] for r in rows) if x is not None) or [None])[
                len([x for x in (r["repair_delay_s"] for r in rows) if x is not None]) // 2] if any(r["repair_delay_s"] is not None for r in rows) else None,
            "how_repaired": how, "how_repaired_rule": "heuristic: global when at most max(1, 10%) of the broken-as-briefed jobs ran broken",
            "jobs_never_broken": sum(1 for r in bab if not r["broken_launches"]),
            "jobs_repaired_after_broken": sum(1 for r in rows if r["broken_launches"] and r["repaired"]),
            "evidence_from_daemon": sum(1 for r in rows for x in r["broken"] if x["first_update_source"].startswith("daemon")),
            "repair_delay_unreported": sum(1 for r in rows if r.get("repair_delay_note")),
            "evidence_estimated": sum(1 for r in rows for x in r["broken"] if x["first_update_source"].startswith("estimated")),
            "items": rows,
            "how": "broken launches from idle.work_remaining (served a broken_records curve); waste is GPU time after the first-update "
                   "evidence (first_update_evidence_s), first-eval beside it; a job is done only by a successful record"}

TTC_REJECT_RE = re.compile(r"\b(?:do not|don'?t|should not|shouldn'?t|not|no|reject(?:s|ed|ing)?|against)\b[^.?\n]{0,60}\badopt|"
                           r"\b(?:recommend(?:ation)?|decision|answer)\b[^.?\n]{0,40}\b(?:no|not adopt|reject|keep the (?:baseline|current))|"
                           r"\badopt(?:ion)?\s*[:\-]\s*no\b|\breject(?:ion)?\b[^.?\n]{0,20}\badopt|"
                           # 'adoption is already ruled out'
                           r"\badopt(?:ion|ing)?\b[^.?\n]{0,30}\b(?:is |are |was )?(?:already )?ruled out\b", re.I)
TTC_ACCEPT_RE = re.compile(r"\b(?:recommend(?:ation)?|decision|answer|should|we)\b[^.?\n]{0,40}\badopt(?:ing)?\b(?!ion)(?![^.?\n]{0,10}\bnot\b)|"
                           r"\badopt(?:ion)?\s*[:\-]\s*yes\b", re.I)
TTC_NUM_RE = re.compile(r"(?<![\w.])-?\d+\.\d{3,}(?!\d)")
# a sentence saying the decision does not exist yet states none, and is skipped like a question (G-tops-haiku-L0-r1b at 148.8 s, 11-06:
# "The adoption decision isn't ready yet, because no cell has finished" read as a reject through 'decision ... no'; so did
# G-tsci-opus-L0's 1,707.8-s "There's no recommendation yet, because no DiagScale results exist so far"); and one naming both outcomes
# as the open choice (G-tsci-sonnet-L0-r2 at 139.5 s: "The adopt/don't-adopt call depends on all three datasets clearing it.")
TTC_PENDING_RE = re.compile(r"\b(?:decision|recommendation|answer|call)\b[^.?\n]{0,25}\b(?:isn'?t|is not|not)\s+(?:yet\s+)?"
                            r"(?:ready|available|possible|final|made)\b|\bno (?:\w+ )?(?:decision|recommendation|answer)\b[^.?\n]{0,10}\byet\b|"
                            r"\badopt\s*/\s*(?:don'?t|do not|not|no)[- ]adopt\b", re.I)

def ttc_decision(text):
    """A tabular-adoption answer's decision from its lead, sentence by sentence: the first sentence that is not a question and states
    a reject or adopt; a question ('Should we adopt?') is never affirmative, and a negated adopt ('do not adopt') is a reject. A sentence
    saying the decision is not ready yet (TTC_PENDING_RE) is skipped."""
    for sent in re.split(r"(?<=[.!\n])\s+", (text or "").strip()[:1200]):
        if sent.rstrip().endswith("?") or TTC_PENDING_RE.search(sent):
            continue
        if TTC_REJECT_RE.search(sent):
            return "reject"
        if TTC_ACCEPT_RE.search(sent):
            return "accept"
    return None

def ttc_means_reported(text, rows, tol=5e-4):
    """Whether an answer reports the paired means a decision rests on: per dataset, a line (or sentence) naming the dataset that also
    carries both arms' completed means, each within rounding (relative tol, or the printed digits) of the records' mean. Dataset names
    are matched case-insensitively with - and _ alike. Returns {dataset: True/False/None (not complete)}."""
    out = {}
    norm = lambda s: re.sub(r"[-_\s]+", "", s.lower())
    chunks = [c for c in re.split(r"\n|(?<=[.!?])\s+", text or "") if c.strip()]
    # ... or a dataset's block: from a line naming it to the next line naming another dataset, or to the first blank line after a
    # table, so a header followed by a table of the two arms' means counts (the analysis 2026-11-05 08:31 MST:
    # G-tops-opus-L0-r2 gave '**credit_card_clients_default** (...)' then a two-row table, each mean on its own row)
    lines = (text or "").split("\n")
    def block(ds):
        out, i = [], 0
        while i < len(lines):
            if norm(ds) in norm(lines[i]):
                j, seen_table = i + 1, False
                while j < len(lines):
                    ln = lines[j]
                    if any(norm(o) in norm(ln) for o in rows if o != ds):
                        break
                    if ln.strip().startswith("|"):
                        seen_table = True
                    elif seen_table and not ln.strip():
                        break
                    j += 1
                out.append("\n".join(lines[i:j]))
                i = j
            else:
                i += 1
        return out
    for ds, r in rows.items():
        b, a = (r.get("baseline") or {}).get("mean_error"), (r.get("adapted") or {}).get("mean_error")
        if not r.get("complete") or b is None or a is None:
            out[ds] = None
            continue
        def close(x, v):
            digits = len(x.split(".")[1]) if "." in x else 0
            return abs(float(x) - v) <= max(tol * abs(v), 0.5 * 10 ** (-digits) + 1e-12)
        hit = False
        for c in [c for c in chunks if norm(ds) in norm(c)] + block(ds):
            nums = TTC_NUM_RE.findall(c)
            if any(close(x, b) for x in nums) and any(close(x, a) for x in nums):
                hit = True
                break
        out[ds] = hit
    return out

def tabular_adoption(d, task, ev=None):
    """A paired adoption package (task.tabular_adoption: Sol independent, pkg-sol-tabular-*-adoption, 2026-11-02). Each cell (dataset,
    arm, split) is a task job; a cell's result is its record's metric_error, and a cell counts only when work_remaining has it done (the
    occupancy log's completion by record, never the agent's own JSON). Per dataset: the completed baseline and adapted cells, whether
    both arms' prescribed splits are all complete, their means, and whether adapted <= baseline x (1 - minimum_relative_error_reduction).
    The key's decision is accept iff every dataset passes. Proofs: a rejection is proved by any one dataset with both arms complete
    that fails, an acceptance by every dataset complete and passing; every valid failing dataset is a proof (no reference dataset).
    proof_wall_s: when the earliest valid proof completed (its last cell's completion), whichever dataset it is. avoidable_gpu_s: all
    paced GPU work after the earliest proof, speculative tails included (Sol independent, 2026-11-02 09:52 MST); work before it that is
    not on the earliest proof's dataset is speculative_pre_proof_gpu_s, reported apart and never avoidable. over_proof_bound: time to
    the answer over the task's objective-specific bound (reference_arithmetic.rejection_proof_lower_bound_s, the minimum over every
    failing dataset of the nonpreemptive bound for its proof; for an accept key the task's own accept bound if it gives one), never a
    bound over the launched work. The answer's decision is read from the root answer text (adopt / reject), with supported = it
    matches the key and a proof completed before it. A label and fields, never validity. None for a task without tabular_adoption."""
    ta = task.get("tabular_adoption")
    if not ta:
        return None
    import idle as _idle, score_run as _sr
    occ = os.path.join(d, "occupancy.jsonl")
    if ev is None:
        if not os.path.exists(occ):
            return None
        ev = _sr.load_events(occ)
    rem = _idle.work_remaining(ev, task)
    # a cell is done when a launch of it completed: an open-draw job's completions are its open_instances (none owed, so done_jobs
    # holds them without a done_at), an ordinary job's its done_at; the earliest completion times the cell
    done_at = dict(rem.get("done_at_wall") or {})
    for inst in rem.get("open_instances") or []:
        t = inst.get("end_wall")
        if t is not None and (inst["task_job"] not in done_at or t < done_at[inst["task_job"]]):
            done_at[inst["task_job"]] = t
    done = set(done_at)
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    a0 = st.get("agent_start_wall")
    speed = float(_idle.run_inputs(d).get("speed") or 1.0)
    rel = lambda t: round((t - a0) * speed, 1) if (t is not None and a0) else None
    arms = list(ta.get("arms") or [])
    base, adapt = (arms + [None, None])[:2]
    need = float(ta.get("minimum_relative_error_reduction") or 0.0)
    by_job = {j["job_id"]: j for j in task.get("jobs") or []}
    cells = ta.get("cells") or []
    err = {}
    for c in cells:
        j = by_job.get(c["job_id"]) or {}
        rid = (j.get("records") or [None])[0]
        try:
            rec = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
            fm = (rec.get("result") or {}).get("final_metrics")
            fm = fm.get("value") if isinstance(fm, dict) and "value" in fm else fm
            err[c["job_id"]] = float(fm["metric_error"])
        except (OSError, ValueError, TypeError, AttributeError):
            err[c["job_id"]] = None
    rows = {}
    for ds in ta.get("datasets") or []:
        mine = [c for c in cells if c["dataset"] == ds]
        out = {}
        for arm in (base, adapt):
            cs = [c for c in mine if c["arm"] == arm]
            got = [c for c in cs if c["job_id"] in done and err.get(c["job_id"]) is not None]
            out[arm] = {"prescribed": len(cs), "completed": len(got),
                        "mean_error": (sum(err[c["job_id"]] for c in got) / len(got)) if got else None,
                        "complete_wall": max(done_at[c["job_id"]] for c in got) if (got and len(got) == len(cs)) else None}
        complete = all(out[a]["prescribed"] and out[a]["completed"] == out[a]["prescribed"] for a in (base, adapt))
        passes = (out[adapt]["mean_error"] <= out[base]["mean_error"] * (1 - need)) if complete else None
        rows[ds] = {"baseline": out[base], "adapted": out[adapt], "complete": complete, "passes": passes,
                    "proof_wall": max(out[base]["complete_wall"], out[adapt]["complete_wall"]) if complete else None,
                    "jobs": [c["job_id"] for c in mine]}
    # the key, from the records alone: every dataset's full pair, as the task's rule reads it
    def key_mean(ds, arm):
        v = [err[c["job_id"]] for c in cells if c["dataset"] == ds and c["arm"] == arm and err.get(c["job_id"]) is not None]
        return sum(v) / len(v) if v else None
    key_pass = {ds: (key_mean(ds, adapt) <= key_mean(ds, base) * (1 - need)) for ds in ta.get("datasets") or []
                if key_mean(ds, base) is not None and key_mean(ds, adapt) is not None}
    key_decision = "accept" if key_pass and all(key_pass.values()) else "reject"
    if key_decision == "reject":
        proofs = sorted((r["proof_wall"], ds) for ds, r in rows.items() if r["complete"] and r["passes"] is False)
    else:
        proofs = ([(max(r["proof_wall"] for r in rows.values()), "all")] if rows and all(r["complete"] and r["passes"] for r in rows.values()) else [])
    t_proof = proofs[0][0] if proofs else None
    proof_ds = {ds for _, ds in proofs}
    gpu_s = {}
    for e in ev:
        if e["kind"] == "enqueue":
            gpu_s.setdefault(e["job"], []).append((e["t_wall"], float(e.get("wall_cost") or 0.0)))
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    jobs_l = _idle.tie_jobs(task)
    ds_of = {c["job_id"]: c["dataset"] for c in cells}
    avoid = spec = 0.0
    for dj, h in hello.items():
        tj = _idle.task_job_of(h, lk.get(dj), jobs_l)
        ds = ds_of.get(tj)
        units = gpu_s.get(dj, [])
        if t_proof is not None:
            avoid += sum(c for t, c in units if t >= t_proof) * speed
        if proofs and ds != proofs[0][1]:
            spec += sum(c for t, c in units if t < t_proof) * speed
    ans = root_final_text(d) if os.path.exists(os.path.join(d, "transcript")) else None
    said = ttc_decision(ans) if ans else None
    rep = report_wall(d) if ans else None
    # the means the decision rests on (Sol independent 09:57 MST: the brief asks for them): a reject on the proof datasets reported, an
    # accept on every dataset; checked against the records' means, never the agent's own files
    reported = ttc_means_reported(ans, rows) if ans else {}
    need_ds = ([ds for _, ds in proofs] if said == "reject" else list(rows)) if proofs else []
    means_ok = bool(need_ds) and (any(reported.get(ds) for ds in need_ds) if said == "reject" else all(reported.get(ds) for ds in need_ds))
    ra = task.get("reference_arithmetic") or {}
    bound = ra.get("rejection_proof_lower_bound_s") if key_decision == "reject" else ra.get("acceptance_proof_lower_bound_s")
    end = rep if rep is not None else (max(done_at.values()) if done_at else None)
    # the deliverable rule, as over_ref: an agent's figure reads to the later of its answer and
    # the end of work still running at the answer, so known-useless cells it launched and left on the GPUs count (G-tops-opus-L0-r2
    # answered at 6,764.5 s with 3 cells running, the last to 8,165.9 s); a launch made after the answer does not exist (the session is over)
    still_running_end = None
    if rep is not None:
        _hello = {e["job"]: e["t_wall"] for e in ev if e["kind"] == "job_hello"}
        _bye = {}
        for e in ev:
            if e["kind"] == "job_bye":
                _bye[e["job"]] = e["t_wall"]
        _open = [_bye[j] for j, t0 in _hello.items() if t0 <= rep and j in _bye and _bye[j] > rep]
        if _open:
            still_running_end = max(_open)
            end = max(end, still_running_end)
    # a reading only once a valid proof completed: with none, the run proved nothing and the ratio is no reading
    over_proof = round((end - a0) * speed / float(bound), 6) if (bound and end is not None and a0 and t_proof is not None) else None
    # ... and only when the answer states a decision at or after the earliest proof: G-tsci-sonnet-L0
    # handed back at 286 s ("I don't have a recommendation yet because the experiment is still running") and its proofs completed from
    # 25,086 s, so the hand-back read as the fastest answer possible (0.039). A decision stated BEFORE any proof completed is not
    # silently dropped: it is flagged decided_before_proof for analysis to rule, and the ratio is held back until then
    # the first supported answer, kept apart from the final one (the analysis 2026-11-05 09:50 MST, on Sol independent's read of
    # G-tsci-opus-L0: at 27,326 s a status turn gave Bioresponse's paired means and 'adoption is already ruled out', and the run went on to
    # a full-table answer at 47,185 s). over_proof_bound times the final answer and the bar stays on it. first_supported_answer_s: the first
    # root end_turn at or after the earliest proof that states the key's decision and reports at least one proven dataset's paired means;
    # a hand label analysis_labels.first_supported_answer {s} wins. A field, never validity or a bar
    first_supported_s, first_how = None, None
    if ans is not None and t_proof is not None and proofs:
        proven = {ds: rows[ds] for _, ds in proofs}
        for w, txt in root_end_turns(d):
            if w < t_proof or ttc_decision(txt) != key_decision:
                continue
            # a proven dataset counts only once its own proof had completed by this turn
            done_by = {ds: r for ds, r in proven.items() if r.get("proof_wall") is not None and r["proof_wall"] <= w}
            if any(ttc_means_reported(txt, done_by).values()):
                first_supported_s, first_how = rel(w), "the first root end_turn at or after the earliest proof stating the key's decision with a proven dataset's means"
                break
    try:
        _fsa = (json.load(open(os.path.join(d, "run_stop.json"))).get("analysis_labels") or {}).get("first_supported_answer")
    except (OSError, ValueError):
        _fsa = None
    if isinstance(_fsa, dict) and _fsa.get("s") is not None:
        first_supported_s, first_how = float(_fsa["s"]), "hand label analysis_labels.first_supported_answer"
    over_proof_first = round(first_supported_s / float(bound), 6) if (first_supported_s is not None and bound) else None
    decided_before_proof = None
    # an agent run only: a scripted reference (whose root text is the controller's "done") keeps its work end, as before
    if over_proof is not None and _idle.run_inputs(d).get("harness") not in (None, "scripted"):
        if said is None:
            over_proof = None
        elif rep is not None and t_proof is not None and rep < t_proof:
            decided_before_proof = {"decision": said, "report_s": rel(rep), "earliest_proof_s": rel(t_proof)}
            over_proof = None
    return {"key_decision": key_decision, "key_passes": key_pass, "datasets": rows,
            "objective_bound_s": bound, "over_proof_bound": over_proof, "decided_before_proof": decided_before_proof,
            "proofs": [{"dataset": ds, "proof_s": rel(t)} for t, ds in proofs], "earliest_proof_s": rel(t_proof),
            "work_running_at_answer_end_s": rel(still_running_end),
            "first_supported_answer_s": first_supported_s, "over_proof_first": over_proof_first, "first_supported_how": first_how,
            "answer_decision": said, "answer_matches_key": (said == key_decision) if said else None,
            "means_reported": reported,
            "supported": bool(said == key_decision and t_proof is not None and (rep is None or rep >= t_proof) and means_ok) if said else None,
            "avoidable_gpu_s": round(avoid, 1), "speculative_pre_proof_gpu_s": round(spec, 1),
            "how": "cells done by work_remaining (occupancy completion by record); errors from each cell's record; the key from every "
                   "dataset's full pair; proofs per tabular_adoption's rule; avoidable work = paced GPU work after the earliest proof"}

SFCK_TRUSTED = os.path.join(RUNNER, "survey", "sfck-trusted-digests-2026-11-04", "trusted-digests.json")
SFCK_CORRUPT = {"truncated_or_crc"}
SFCK_UNSUPPORTED = {"legacy_format", "not_zip", "no_data_pkl", "unexpected_record", "too_large", "time_cap", "stat",
                    "unsupported_or_repacked", "pickle_parse"}

CKPT_LOAD_RE = re.compile(r"\btorch\.load\s*\(|\.load_state_dict\s*\(|\bsafetensors\b[^\n]{0,40}\bload|\bload_file\s*\(")
PY_RUN_RE = re.compile(r"(?:^|[;&|\s(`])(?:python[\d.]*|ipython|uv\s+run|torchrun)\b")
CKPT_FILE_RE = re.compile(r"\.(?:pt|pth|bin|safetensors)\b|\bckpt\b|\bcheckpoint", re.I)
CKPT_TRAINING_RE = re.compile(r"\.backward\s*\(")

def _ckpt_load(body):
    """Python code that loads a checkpoint when it runs: a torch.load / load_state_dict / safetensors load call in the parsed code
    (never in a string literal: a generator that writes a training script holds its resume code as text), outside an
    `if args.resume`-style branch (a training script's resume path), naming a checkpoint file, and no .backward( (training code).
    Code that does not parse is read as text, with the same file-name and training tests."""
    if not (CKPT_LOAD_RE.search(body) and CKPT_FILE_RE.search(body)) or CKPT_TRAINING_RE.search(body):
        return False
    import ast as _ast
    try:
        tree = _ast.parse(body)
    except (SyntaxError, ValueError, RecursionError):
        return True
    resume = lambda t: isinstance(t, _ast.If) and re.search(r"resume", _ast.unparse(t.test), re.I) is not None
    hits = []
    def walk(n, in_resume=False):
        for c in _ast.iter_child_nodes(n):
            r = in_resume or resume(c)
            if isinstance(c, _ast.Call):
                f = c.func
                nm = f.attr if isinstance(f, _ast.Attribute) else f.id if isinstance(f, _ast.Name) else ""
                full = _ast.unparse(f)
                if (nm in ("load_state_dict", "load_file") or full in ("torch.load", "safetensors.torch.load_file")
                        or full.endswith(".load") and "torch" in full) and not r:
                    hits.append(c)
            walk(c, r)
    walk(tree)
    return bool(hits)
CKPT_REMARK_RE = re.compile(r"\buntrained\b|\bnot (?:been )?trained\b|\binitial(?:i[sz]ed)? (?:weights|parameters|state)\b|"
                            r"\brandom(?:ly)?[ -]init|\bidentical (?:weights|checkpoints|tensors|models?)\b|\bsame (?:weights|tensors|parameters)\b|"
                            r"\bweights (?:did ?n[o']t|do not|never) (?:change|update|move)", re.I)

CKPT_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n(.*?)\n\s*\2\s*(?:\n|$)", re.S)
CKPT_PY_C_RE = re.compile(r"(?:python[\d.]*|ipython)\s+(?:-[A-Za-z]+\s+)*-c\s+(['\"])(.*?)(?<!\\)\1", re.S)
CKPT_PY_FILE_RE = re.compile(r"(?:^|[;&|\s(`])(?:python[\d.]*|ipython|torchrun|uv\s+run(?:\s+python[\d.]*)?)\s+(?:-[A-Za-z][^\s]*\s+)*([^\s;&|<>'\"]+\.py)\b")
CKPT_WRITE_TO_RE = re.compile(r"(?:\bcat|\btee(?:\s+-a)?)\s*>?>?\s*([^\s;&|<>'\"]+)")

def _ckpt_exec_texts(cmd):
    """(code Python runs from this command, {basename: body} of files it writes): a heredoc whose line starts a Python process is run,
    one fed to cat or tee is a file written; python -c's argument is run. Reading a file's source runs nothing."""
    run, wrote = [], {}
    for m in CKPT_HEREDOC_RE.finditer(cmd or ""):
        line = cmd[cmd.rfind("\n", 0, m.start()) + 1:m.start()]
        if PY_RUN_RE.search(line) and not re.search(r"\bcat\b|\btee\b", line):
            run.append(m.group(3))
        else:
            w = CKPT_WRITE_TO_RE.search(line)
            if w:
                wrote[os.path.basename(w.group(1))] = m.group(3)
    run += [m.group(2) for m in CKPT_PY_C_RE.finditer(cmd or "")]
    return run, wrote

def loaded_checkpoint_weights(d):
    """Whether the agent loaded a saved checkpoint's weights (the analysis 2026-11-03 20:42 MST, SF checkpoint collisions: under
    replay the saved tensors are the script's initial weights, so an agent that loads and inspects a member can meet that limit; the
    label shows whether any does). A load is Python the agent ran whose code calls torch.load, load_state_dict or a safetensors load
    and names a checkpoint file (.pt, .pth, .bin, .safetensors, ckpt, checkpoint), and that is not training code (no .backward( in it:
    a training script's --resume branch, or a generator that writes one, H1's on 2026-11-04, loads nothing to look at):
    python -c, a heredoc fed to python, or a .py file it wrote (Write, Edit, or a heredoc to cat/tee) and then ran as python <file>.
    The task's training scripts only save, so their launches never count; writing a file, reading its source (cat, grep) or
    `python -m py_compile` runs no load. remarked: the agent's own text after its first load says the weights look untrained,
    initial, random or identical. A label; None without a transcript."""
    import prep as _prep, tells as _tl
    recs = sorted((r for r in _tl._records(d) if isinstance(r, dict) and r.get("timestamp")), key=lambda r: r["timestamp"])
    if not recs:
        return None
    calls = _prep.calls_with_inputs(recs, open_until=float("inf"))
    written = {}
    loads = []
    for t0, t1, name, inp, _ts0, _mid, _n, _res in calls:
        if not isinstance(inp, dict):
            continue
        if name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
            body = " ".join(str(inp.get(k) or "") for k in ("content", "new_string", "new_source"))
            body += " ".join(str(e.get("new_string") or "") for e in inp.get("edits") or [] if isinstance(e, dict))
            fp = str(inp.get("file_path") or inp.get("notebook_path") or "")
            if fp and _ckpt_load(body):
                written[os.path.basename(fp)] = t0
            continue
        if name != "Bash":
            continue
        cmd = _prep.call_text(name, inp) or ""
        run, wrote = _ckpt_exec_texts(cmd)
        for b_, body in wrote.items():
            if _ckpt_load(body):
                written[b_] = t0
        if any(_ckpt_load(x) for x in run):
            loads.append({"utc": t0, "how": "inline", "command": cmd[:240]})
            continue
        ran = [os.path.basename(m.group(1)) for m in CKPT_PY_FILE_RE.finditer(cmd)
               if not re.search(r"\s-m\s", cmd[max(0, m.start() - 1):m.end()])]
        hit = next((b_ for b_ in ran if b_ in written and written[b_] <= t0), None)
        if hit:
            loads.append({"utc": t0, "how": f"runs {hit}, written with a load", "command": cmd[:240]})
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    a0 = st.get("agent_start_wall")
    remark = None
    if loads:
        first = loads[0]["utc"]
        for r in recs:
            if r.get("type") != "assistant" or not r.get("timestamp") or _prep._ts(r["timestamp"]) < first:
                continue
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") in ("text", "thinking"):
                    m = CKPT_REMARK_RE.search(b.get("text") or b.get("thinking") or "")
                    if m:
                        txt = b.get("text") or b.get("thinking") or ""
                        remark = {"utc": _prep._ts(r["timestamp"]), "text": txt[max(0, m.start() - 80):m.end() + 80].replace("\n", " ")}
                        break
            if remark:
                break
    rel = lambda t: round(t - a0, 1) if a0 and t is not None else None
    return {"loaded_checkpoint_weights": bool(loads), "loads": len(loads),
            "first_load_s": rel(loads[0]["utc"]) if loads else None,
            "calls": [dict(x, at_s=rel(x["utc"]), utc=None) for x in loads[:10]],
            "scripts_written_with_a_load": sorted(written), "remarked": bool(remark),
            "remark": dict(remark, at_s=rel(remark["utc"]), utc=None) if remark else None}

def sf_checkpoint_artifact(d, task, ev=None, done=None, trusted_path=None):
    """Which required ensemble members a checkpoint-collision run delivered (task.checkpoint_design: the SF lane's
    pkg-sf-checkpoint-collision; SF lane and the analysis 2026-11-03 00:16-00:17 MST). Identity is read only from the files'
    contents: isolation's weights_digest.py full_digest and model_digest of each checkpoint the run's ensemble index names
    (workspace/weights-digests.jsonl, captured at teardown), against the trusted table bootstrapped from W0-sfck-reference's 40 members
    (runner/survey/sfck-trusted-digests-2026-11-04). The index's own recipe/seed labels are never identity; they are checked beside it.
    A file identifies a member by its full_digest, or by its model_digest when no other member shares it; a model_digest shared by
    several members (9 pairs on 11-04: the same seed and architecture give the same initial weights under replay) is ambiguous between
    them. Per member: identified (and delivered when its training also completed by work_remaining), ambiguous, unresolved (not
    identified while some file could not be interpreted: an unsupported or repacked format, oversized, time-capped; or table_mismatch,
    interpretable files none of which matches any trusted member), or lost: a proven loss, no interpretable file identifies it and no
    file escaped interpretation. A corrupt file (truncated_or_crc) is a lost member, never unsupported. Labels and fields, never
    validity. None for a task without checkpoint_design."""
    cd = task.get("checkpoint_design")
    if not cd:
        return None
    try:
        tr = json.load(open(trusted_path or SFCK_TRUSTED))
    except (OSError, ValueError):
        return {"pending": True, "why": "no trusted digest table"}
    if tr.get("task_id") != task.get("task_id"):
        return {"pending": True, "why": f"the trusted table is {tr.get('task_id')}'s"}
    cap = os.path.join(d, "workspace", "weights-digests.jsonl")
    if not os.path.exists(cap):
        return {"pending": True, "why": "no weights-digests.jsonl: the capture did not run (fix:capture-weights-digest)"}
    rows = []
    for line in open(cap, errors="replace"):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    name = lambda m: f"{m['recipe']}/seed{m['seed']}"
    mem = {name(m): m for m in tr.get("members") or []}
    by_full = {m["full_digest"]: name(m) for m in mem.values()}
    by_model = {}
    for m in mem.values():
        by_model.setdefault(m["model_digest"], []).append(name(m))
    import idle as _idle
    if done is None:
        import score_run as _sr
        if ev is None:
            occ = os.path.join(d, "occupancy.jsonl")
            ev = _sr.load_events(occ) if os.path.exists(occ) else []
        done = set(_idle.work_remaining(ev, task).get("done_jobs") or [])
    # the save's wall time, as the host saw it: each member rank's last paced unit's end to its job_bye (the final torch.save plus exit).
    # The script's own save_time_s reads 0.0 under replay (time.perf_counter on the replay clock; SF lane 2026-11-03 20:43 MST), so this
    # is the measure, an upper bound on the save's I/O
    post = {}
    if ev:
        lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
        # each unit on its GPU's queue, as host_terms.py rebuilds it: it starts at the later of its enqueue and the GPU's previous unit's end
        s0 = next((e for e in ev if e["kind"] == "daemon_start"), ev[0])
        off = s0["t_wall"] - s0["t_mono"]
        last_end, cursor = {}, {}
        for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
            g, t0_, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost") or 0.0)
            st_ = max(cursor.get(g, t0_), t0_); cursor[g] = st_ + c
            last_end[e["job"]] = max(last_end.get(e["job"], 0.0), st_ + c)
        jl = _idle.tie_jobs(task)
        for e in ev:
            if e["kind"] == "job_bye" and e["job"] in last_end:
                h = next((x for x in ev if x["kind"] == "job_hello" and x["job"] == e["job"]), None)
                tj = _idle.task_job_of(h, lk.get(e["job"]), jl) if h else None
                if tj:
                    post.setdefault(tj, []).append(round(e["t_wall"] - last_end[e["job"]], 3))
    job_of = {f"{j['checkpoint_member']['recipe']}/seed{j['checkpoint_member']['seed']}": j for j in task.get("jobs") or []
              if j.get("checkpoint_member")}
    files, ident, amb, unsup, corrupt, unrec, interp = [], {}, {}, [], [], [], 0
    index_error = None
    for r in rows:
        if "i" not in r:
            index_error = r.get("error")
            continue
        err, merr = r.get("error"), r.get("model_error")
        f = {"i": r["i"], "listed": r.get("listed")}
        if err == "no_path":
            f["class"] = "empty_slot"
        elif err == "duplicate_of":
            f["class"], f["of"] = "duplicate", r.get("of")
        elif err == "not_found":
            f["class"] = "missing_file"
        elif err in ("outside_workspace", "not_a_file"):
            f["class"] = "invalid_path"
        elif err in SFCK_CORRUPT or merr in SFCK_CORRUPT:
            f["class"] = "corrupt"; corrupt.append(r["i"])
        elif err:
            f["class"], f["label"] = "unsupported", err; unsup.append(r["i"])
        elif r.get("full_digest") in by_full:
            f["class"], f["member"], f["by"] = "identified", by_full[r["full_digest"]], "full_digest"
            ident.setdefault(f["member"], []).append(r["i"]); interp += 1
        elif r.get("model_digest") in by_model and len(by_model[r["model_digest"]]) == 1:
            f["class"], f["member"], f["by"] = "identified", by_model[r["model_digest"]][0], "model_digest"
            ident.setdefault(f["member"], []).append(r["i"]); interp += 1
        elif r.get("model_digest") in by_model:
            f["class"], f["among"] = "ambiguous", sorted(by_model[r["model_digest"]])
            amb.setdefault(r["model_digest"], []).append(r["i"]); interp += 1
        elif merr:
            f["class"], f["label"] = "unsupported", merr; unsup.append(r["i"])
        elif r.get("full_digest"):
            f["class"] = "unrecognised"; unrec.append(r["i"]); interp += 1
        else:
            f["class"] = "unsupported"; f["label"] = "no digest"; unsup.append(r["i"])
        files.append(f)
    table_mismatch = bool(unrec) and not ident and not amb
    status, classes = {}, []
    for md, ms in by_model.items():
        if len(ms) > 1:
            left = [m for m in ms if m not in ident]
            fs = amb.get(md, [])
            classes.append({"members": sorted(ms), "not_identified": sorted(left), "ambiguous_files": fs,
                            "at_most_preserved": min(len(fs), len(left)), "at_least_lost": max(0, len(left) - len(fs)) if not unsup else None})
            for m in left:
                if fs:
                    status[m] = "ambiguous"
    for m in mem:
        if m in ident:
            status[m] = "identified"
        elif m not in status:
            status[m] = "unresolved" if (unsup or table_mismatch) else "lost"
    delivered = sorted(m for m, s_ in status.items() if s_ == "identified" and (job_of.get(m) or {}).get("job_id") in done)
    # the index's own rows, beside identity: the row naming an identified file should name its member and that member's loss
    labels = {"rows_checked": 0, "label_mismatch": [], "loss_mismatch": [], "loss_missing": []}
    try:
        idx = json.load(open(os.path.join(d, "workspace", "home", "project", (task.get("capture") or {}).get("weights_index", {}).get("path") or "ensemble.json")))
        irows = idx.get("members") if isinstance(idx, dict) else idx
    except (OSError, ValueError, AttributeError):
        irows = None
    for f in files:
        if f["class"] != "identified" or not isinstance(irows, list) or f["i"] >= len(irows) or not isinstance(irows[f["i"]], dict):
            continue
        row, m = irows[f["i"]], mem[f["member"]]
        labels["rows_checked"] += 1
        if (row.get("recipe"), row.get("seed")) != (m["recipe"], m["seed"]):
            labels["label_mismatch"].append({"i": f["i"], "index_says": f"{row.get('recipe')}/seed{row.get('seed')}", "file_is": f["member"]})
        want = None
        try:
            want = json.load(open(os.path.join(REGISTRY, "records", m["record"] + ".json")))["series"]["val_loss"]["values"][-1]
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass
        got = row.get("validation_loss")
        if got is None:
            labels["loss_missing"].append(f["i"])
        elif want is not None and abs(float(got) - float(want)) > 5e-5:
            labels["loss_mismatch"].append({"i": f["i"], "member": f["member"], "index_says": got, "record": want})
    count = lambda k: sum(1 for v in status.values() if v == k)
    _ps = sorted(x for v in post.values() for x in v)
    save = ({"members_timed": len(_ps), "median_s": _ps[len(_ps) // 2], "max_s": _ps[-1], "min_s": _ps[0],
             "by_job": {k: v for k, v in sorted(post.items())},
             "how": "each rank's last paced unit's end (its GPU's queue rebuilt as host_terms.py does) to its job_bye, wall seconds: the "
                    "final save plus process exit (host_terms' post_s); the scripts' own save_time_s reads 0.0 under replay (perf_counter "
                    "on the replay clock), so this observed span is the measure"}
            if _ps else None)
    return {"pending": False, "required": len(mem), "delivered": len(delivered), "identified": count("identified"),
            "save_timing": save,
            "ambiguous": count("ambiguous"), "unresolved": count("unresolved"), "lost": count("lost"),
            "identified_by_model_only": sum(1 for f in files if f.get("by") == "model_digest"),
            "files": {k: sum(1 for f in files if f["class"] == k) for k in
                      ("identified", "ambiguous", "unrecognised", "unsupported", "corrupt", "duplicate", "missing_file", "empty_slot", "invalid_path")},
            "corrupt_files": corrupt, "unsupported_files": unsup, "table_mismatch": table_mismatch, "index_error": index_error,
            "alias_classes": classes, "members": status, "delivered_members": delivered, "index_labels": labels,
            "trusted_table": os.path.relpath(trusted_path or SFCK_TRUSTED, os.path.dirname(RUNNER)),
            "trusted_table_sha256": hashlib.sha256(open(trusted_path or SFCK_TRUSTED, "rb").read()).hexdigest()[:12],
            "how": "identity from content digests only (full_digest, else a model_digest no other member shares), against W0-sfck-reference's "
                   "trusted table; ambiguous / unresolved (unsupported, repacked, table mismatch) kept apart from proven losses; delivered = "
                   "identified and its training completed (work_remaining); the index's labels and losses checked beside it"}

# adopt (the analysis 2026-11-06 03:38 MST, G-topt-sonnet-L0-r2: a table whose Recommendation column reads 'Adopt DiagScale')
TT_LOWER_RE = re.compile(r"\b(?:recommend(?:ation|ed)?|prefer(?:red)?|choose|use|pick|keep|adopt|lower(?: mean)?(?: error)?|better|wins?|best)\b", re.I)

def _tt_arm_named(sent, arms):
    """The arms a sentence names, longest first so tabpfn3_diagscale_aggregation_96 is never read as tabpfn3_aggregation_96; a bare
    'DiagScale' or 'adapted' names the second arm, 'baseline' or 'without DiagScale' the first."""
    norm = lambda x: re.sub(r"[-_\s]+", "", x.lower())
    t = norm(sent)
    hit = [a for a in sorted(arms, key=len, reverse=True) if norm(a) in t]
    if hit:
        found, rest = [], t
        for a in sorted(arms, key=len, reverse=True):
            if norm(a) in rest:
                found.append(a)
                rest = rest.replace(norm(a), " ")
        return found
    base, adapt = arms[0], arms[1]
    # a bare 'base' names the first arm too ('Keep base', G-topt-sonnet-L0-r2, the analysis 2026-11-06 03:38 MST)
    if re.search(r"without diag\s*scale|\bbase(?:line)?\b|\bno diag\s*scale\b", sent, re.I):
        return [base]
    if re.search(r"diag\s*scale|\badapted\b|\badaptation\b", sent, re.I):
        return [adapt]
    return []

def tabular_table_answer(text, datasets, arms):
    """Per dataset, the arm an answer recommends: in a sentence or line naming the dataset (case-insensitive, - and _ alike), the clause
    (split at ; , : and dashes) holding a recommendation word (recommend, prefer, use, lower, better, wins, best) that names exactly one
    arm; a line that lists both arms' means ("A mean 0.1203, B mean 0.1204; recommend A") so reads its recommendation clause. A clause
    naming both arms, or none, settles nothing and the next is tried. None for a dataset nothing settles."""
    norm = lambda x: re.sub(r"[-_\s]+", "", x.lower())
    chunks = [c for c in re.split(r"\n|(?<=[.!?])\s+", text or "") if c.strip()]
    out = {}
    for ds in datasets:
        pick = None
        for c in chunks:
            if norm(ds) not in norm(c) or not TT_LOWER_RE.search(c):
                continue
            for cl in [c] + [x for x in re.split(r"[;,:]|\s[-\u2013\u2014]\s", c) if x.strip()]:
                if not TT_LOWER_RE.search(cl):
                    continue
                named = _tt_arm_named(cl, arms)
                if len(named) == 1:
                    pick = named[0]
                    break
            if pick:
                break
        out[ds] = pick
    return out

def tabular_table(d, task, ev=None):
    """A full-table package (task.tabular_table: Sol independent's pkg-sol-tabular-{science,operations}, the matched controls of the
    adoption twins, 2026-11-02). Every prescribed cell (dataset, arm, split) is owed: there is no adoption threshold, no proof and no
    post-rejection unneeded work. A cell counts only when work_remaining has it done (occupancy completion by record, never the agent's
    own JSON); its error is its record's final_metrics.metric_error. Per dataset: each arm's completed and prescribed cells, its mean
    error over completed cells, complete when both arms' prescribed splits all are, and the lower mean error arm on the full-precision
    means (no significance or adoption threshold). The answer key is the same from every cell's record. A model run's answer (the root
    final text) is read for each dataset's arm means (ttc_means_reported, within displayed rounding) and its recommended arm
    (tabular_table_answer); supported only when every cell is done, every dataset's means are reported, and every recommendation
    matches the key. A scripted run's answer fields stay None. Labels and fields, never validity. None for a task without tabular_table."""
    tt = task.get("tabular_table")
    if not tt:
        return None
    import idle as _idle, score_run as _sr
    occ = os.path.join(d, "occupancy.jsonl")
    if ev is None:
        if not os.path.exists(occ):
            return None
        ev = _sr.load_events(occ)
    rem = _idle.work_remaining(ev, task)
    done = set(rem.get("done_jobs") or [])
    arms = list(tt.get("arms") or [])
    by_job = {j["job_id"]: j for j in task.get("jobs") or []}
    cells = tt.get("cells") or []
    err, metric = {}, {}
    for c in cells:
        rid = ((by_job.get(c["job_id"]) or {}).get("records") or [None])[0]
        try:
            rec = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
            fm = (rec.get("result") or {}).get("final_metrics")
            fm = fm.get("value") if isinstance(fm, dict) and "value" in fm else fm
            err[c["job_id"]], metric[c["job_id"]] = float(fm["metric_error"]), fm.get("metric")
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            err[c["job_id"]] = None
    mean = lambda xs: (math.fsum(xs) / len(xs)) if xs else None
    rows, key = {}, {}
    for ds in tt.get("datasets") or []:
        mine = [c for c in cells if c["dataset"] == ds]
        out, kmeans = {}, {}
        for arm in arms:
            cs = [c for c in mine if c["arm"] == arm]
            got = [c for c in cs if c["job_id"] in done and err.get(c["job_id"]) is not None]
            out[arm] = {"prescribed": len(cs), "completed": len(got), "mean_error": mean([err[c["job_id"]] for c in got])}
            kmeans[arm] = mean([err[c["job_id"]] for c in cs if err.get(c["job_id"]) is not None])
        complete = all(out[a]["prescribed"] and out[a]["completed"] == out[a]["prescribed"] for a in arms)
        lower = min(arms, key=lambda a: out[a]["mean_error"]) if complete else None
        rows[ds] = {"metric": next((metric.get(c["job_id"]) for c in mine if metric.get(c["job_id"])), None), "arms": out,
                    "complete": complete, "lower_mean_error_arm": lower,
                    # the shape ttc_means_reported reads (baseline, adapted)
                    "baseline": out[arms[0]], "adapted": out[arms[1]]}
        key[ds] = {"mean_metric_error": kmeans, "lower_mean_error_arm": (min(arms, key=lambda a: kmeans[a])
                                                                          if all(kmeans[a] is not None for a in arms) else None)}
    cells_done = sum(1 for c in cells if c["job_id"] in done)
    all_done = cells_done == len(cells) and bool(cells)
    harness = _idle.run_inputs(d).get("harness")
    ans = root_final_text(d) if harness not in (None, "scripted") and os.path.exists(os.path.join(d, "transcript")) else None
    reported = ttc_means_reported(ans, rows) if ans else None
    recs = tabular_table_answer(ans, list(rows), arms) if ans else None
    matches = {ds: (recs.get(ds) == key[ds]["lower_mean_error_arm"]) if recs.get(ds) else None for ds in rows} if recs else None
    # a hand label answer_declined_tie {datasets: [...]}: topt's brief asks
    # for "the recommendation for that dataset" with no decision rule, so "no real difference" on a dataset whose arms differ below the
    # splits' resolution is a correct recommendation; only the named datasets count as matching, and the label is kept beside the key
    declined = []
    try:
        _al = json.load(open(os.path.join(d, "run_stop.json"))).get("analysis_labels") or {}
        # answer_declined_tie (no arm named) and answer_tie_kept_baseline (the incumbent kept on a measured tie, G-topt-sonnet-L0-r2,
        # the analysis 2026-11-06 03:38 MST): the named datasets count as matching, case by case, never by a blanket rule
        for _lk in ("answer_declined_tie", "answer_tie_kept_baseline"):
            _dt = _al.get(_lk)
            declined += [x for x in ((_dt or {}).get("datasets") or []) if x in rows] if isinstance(_dt, dict) else []
    except (OSError, ValueError):
        pass
    if matches is not None:
        for ds in declined:
            matches[ds] = True
    supported = (bool(all_done and reported and all(reported.get(ds) for ds in rows) and matches and all(matches.get(ds) for ds in rows))
                 if ans else None)
    return {"cells_prescribed": len(cells), "cells_done": cells_done, "all_cells_done": all_done, "datasets": rows, "key": key,
            "answer_means_reported": reported, "answer_recommendations": recs, "answer_matches_key": matches, "supported": supported,
            "answer_declined_tie": declined or None,
            "how": "cells done by work_remaining (occupancy completion by record); errors from each cell's record; per dataset the lower "
                   "full-precision mean error arm, no threshold; the answer's means within displayed rounding (ttc_means_reported) and "
                   "its per-dataset recommendation (tabular_table_answer); a scripted run's answer fields None"}

K2_UNRESOLVED_RE = re.compile(r"\b(?:cannot|can'?t|could not|not) (?:be )?(?:distinguish|separat|tell)|\bunresolved\b|\bindistinguishable\b|"
                              r"\bno (?:arm|optimizer) (?:is )?(?:clearly|significantly)? ?(?:lower|better)", re.I)
K2_WIN_RE = re.compile(r"\b(?:winner|lowest|best|recommend|wins|has the lowest|is lowest|lower mean)\b", re.I)
# a quoted span is someone else's words, so the unresolved test skips it: G-k2p-sonnet-A5-r5 (11-10) answered radial-damp at
# 11,738.8 s, then a follow-up led 'It said radial and contra are "likely close to indistinguishable"', correcting its teammate,
# and that turn was read as an unresolved answer
K2_QUOTED_RE = re.compile(r"\"[^\"\n]{0,300}\"|“[^”\n]{0,300}”")
k2_unresolved = lambda lead: K2_UNRESOLVED_RE.search(K2_QUOTED_RE.sub(" ", lead or ""))

def _k2_answer(d, pr, st, speed, cx=None):
    """A K2 run's answer: the last root turn with text (root_final_text), its time from agent start (answer_s, the deliverable's
    time), the arm it names (answer_winner: the first sentence of its lead that names exactly one arm next to winner/lowest/best/
    recommend, else the first arm named in the lead) or that it calls the arms indistinguishable (answer_unresolved), and
    answer_matches_key against precision.reference_outcome. Nothing for a scripted run."""
    if (_idle_run_inputs(d).get("harness") or "scripted") == "scripted":
        return {"answer_s": None, "answer_winner": None, "answer_unresolved": None, "answer_matches_key": None}
    text = root_final_text(d) or ""
    a0 = st.get("agent_start_wall")
    # a Codex run's answer time is its root thread's last task_complete, as report_latency reads it (G-k2p-astra-L0-v2 and -r2,
    # 2026-11-08: report_wall(d) reads Claude Code end_turns only, so answer_s was None and the run read "no answer" though its
    # winner read radial-damp; the analysis 05:53 MST)
    tw = report_wall(d, cx)
    arms = list((pr.get("arms") or {}).keys())
    names = {a: re.compile(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", re.I) for a in arms}
    lead_of = lambda t: re.split(r"\n\s*\n", t.strip(), maxsplit=1)[0][:800] if (t or "").strip() else ""
    # a sentence also ends at a stop followed by its bold or italic close (`.**`, `.__`): G-k2p-opus-A5 (11-08) led with
    # '**radial-damp has the lowest mean validation loss at step 1,500, at 95% simultaneous confidence.** It beats contra-soft-muon
    # by about 0.006 and muonh-tuned by about 0.032', which split only at the second stop, so its first "sentence" named all three
    # arms and no winner was read (the analysis 20:17 MST 11-08; the mls-kv pick reader's split is unchanged)
    def stated(lead):
        for sent in re.split(r"(?:(?<=[.!?\n])|(?<=[.!?]\*\*)|(?<=[.!?]__))\s+", lead):
            hit = [a for a in arms if names[a].search(sent)]
            if len(hit) == 1 and K2_WIN_RE.search(sent):
                return hit[0]
        return None
    # the answer is the last root end_turn whose lead names a winner or calls the arms indistinguishable, else the final text, as
    # filter_yield_answer reads its own. G-k2p-opus-L0-v2-r2 (11-06) answered
    # '**radial-damp has the lowest mean validation loss at step 1,500.**' at 30,027.5 s, then a 30,257.1-s follow-up after its own
    # background monitor ended ('That notification is just my background progress monitor finishing. The results above are unchanged.'),
    # whose lead names no arm; the final text was read as the answer. answer_s is the answer turn's own time (the analysis
    # 2026-11-06 06:22 MST: over_ref reads the 30,027.5-s answer, not the follow-up)
    for _w, _t in reversed(root_end_turns(d)):
        _l = lead_of(_t)
        if stated(_l) or k2_unresolved(_l):
            text, tw = _t, _w
            break
    lead = lead_of(text)
    win = stated(lead)
    if win is None:
        found = sorted(((m.start(), a) for a in arms for m in [names[a].search(lead)] if m))
        win = found[0][1] if found else None
    unres = bool(k2_unresolved(lead)) and not (win and K2_WIN_RE.search(lead))
    key = (pr.get("reference_outcome") or {}).get("winner")
    return {"answer_s": round((tw - a0) * speed, 1) if (tw and a0) else None, "answer_winner": None if unres else win,
            "answer_unresolved": unres, "answer_matches_key": (None if not text else (not unres and win == key))}

def _idle_run_inputs(d):
    import idle as _i
    return _i.run_inputs(d)

def precision_comparison(d, task, ev=None, cx=None):
    """A precision-limited replication package (task.precision: the study's K2, pkg-k2-precision, 2026-11-02). The question has an
    answer under a declared procedure (gen_k2_precision.PROCEDURE: Bonferroni Student-t intervals over 3 arms x 3 fixed looks at 4, 8
    and all per arm); the reference needs only the replications its resolving look uses (precision.reference_job_ids, 24 of 49), so
    an available job left unlaunched is never missing. Per run: the completed replications per arm (occupancy completion by record, never
    the agent's files), each with its record's val_loss at the precision step, in the order the agent completed them; the declared
    procedure applied to those (gen_k2_precision.evaluate_reference, imported so the two cannot drift), giving whether the run's own
    evidence resolves, at which look, its winner and intervals; and the key's resolving look and winner from task.precision. jobs at
    resolution: the replications the procedure needed on this run's own order; excess: completed replications beyond them, and the GPU-s
    launched after the run's evidence first resolved (the second figure needs the completion times). A run whose evidence never resolves
    under the declared procedure is not wrong for that: another valid sequential procedure may stop elsewhere (precision.scoring_note),
    so this reads it, it does not judge it. A label and fields, never validity. None for a task without precision."""
    pr = task.get("precision")
    if not pr:
        return None
    import idle as _idle, score_run as _sr
    sys.path.insert(0, os.path.normpath(os.path.join(REGISTRY, "..", "build")))
    try:
        import gen_k2_precision as _k2
    except Exception as ex:
        return {"error": f"gen_k2_precision not importable: {ex!r}"[:200]}
    occ = os.path.join(d, "occupancy.jsonl")
    if ev is None:
        if not os.path.exists(occ):
            return None
        ev = _sr.load_events(occ)
    rem = _idle.work_remaining(ev, task)
    done_at = dict(rem.get("done_at_wall") or {})
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    speed = float(_idle.run_inputs(d).get("speed") or 1.0)
    arm_of = {jid: arm for arm, ids in (pr.get("arms") or {}).items() for jid in ids}
    by_job = {j["job_id"]: j for j in task.get("jobs") or []}
    step = pr.get("step")
    def val_at(jid):
        rid = (by_job.get(jid, {}).get("records") or [None])[0]
        try:
            rec = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
        except (OSError, ValueError):
            return None
        s = (rec.get("series") or {}).get(pr.get("metric") or "val_loss") or {}
        st, vs = s.get("steps") or [], s.get("values") or []
        return next((float(v) for k, v in zip(st, vs) if k == step), None)
    # in each arm's file order (precision.arms), the order the declared procedure's looks take their first 4 and 8 from
    pos = {jid: k for ids in (pr.get("arms") or {}).values() for k, jid in enumerate(ids)}
    rows = sorted(({"job_id": jid, "arm": arm_of[jid], "val_loss": val_at(jid), "paced_1gpu_s": 0.0, "done_wall": t}
                   for jid, t in done_at.items() if jid in arm_of), key=lambda r: pos[r["job_id"]])
    rows = [r for r in rows if r["val_loss"] is not None]
    a0 = st.get("agent_start_wall")
    counts = {arm: sum(1 for r in rows if r["arm"] == arm) for arm in (pr.get("arms") or {})}
    out = {"completed": len(rows), "completed_by_arm": counts, "available": pr.get("available_jobs"),
           "key_resolving_look": (pr.get("reference_outcome") or {}).get("resolving_look"),
           "key_winner": (pr.get("reference_outcome") or {}).get("winner"), "reference_jobs": pr.get("reference_jobs")}
    if not rows or any(c < 2 for c in counts.values()):
        out.update(_k2_answer(d, pr, st, speed, cx))
        out.update(resolves=None, how_far="fewer than 2 completed replications in some arm: no interval")
        return out
    r = _k2.evaluate_reference(rows)
    look = r["resolving_look"]
    winner = next((l["winner"] for l in r["looks"] if l["look"] == look), None) if look else None
    needed = set(r["reference_job_ids"]) if look else set()
    excess = [x["job_id"] for x in rows if look and x["job_id"] not in needed]
    t_res = max(x["done_wall"] for x in rows if x["job_id"] in needed) if look else None
    after = 0.0
    if t_res is not None:
        for e in ev:
            if e["kind"] == "enqueue" and e["t_wall"] >= t_res:
                after += float(e.get("wall_cost") or 0.0) * speed
    final = r["looks"][-1]
    # the agent's answer (a model run): the last root turn's text and its time, and the winner it names, read against the key
    #. The answer names a winner in
    # its lead (the first sentence naming one arm, after "winner", "lowest", "best" or "recommend"), or says the arms cannot be told
    # apart; a scripted run makes no answer
    out.update(_k2_answer(d, pr, st, speed, cx))
    out.update(resolves=bool(look), resolving_look=look, winner=winner,
               winner_matches_key=(winner == out["key_winner"]) if look else None,
               jobs_at_resolution=len(needed) if look else None, excess_replications=len(excess),
               resolved_s=round((t_res - a0) * speed, 1) if (t_res and a0) else None,
               gpu_s_after_resolution=round(after, 1) if look else None,
               final_intervals={a: {k: round(v[k], 6) for k in ("n", "mean", "low", "high")} for a, v in final["intervals"].items()},
               remaining_uncertainty=None if look else "the run's completed replications do not separate the arms under the declared procedure",
               how="completed replications by occupancy completion, in each arm's file order; values from each record's series at precision.step; "
                   "gen_k2_precision.evaluate_reference applied to them")
    return out

PREP_CMD_RE = re.compile(r"(?:^|[\s/])tools/prepare\.py\s+(?!-)(\S+)")

def prep_observed(d, task):
    """A task's required CPU preparations (task.preparations: Sol independent, pkg-filter-yield, 2026-11-02), each read from the
    run's own evidence, never the scripted controller's log, so an agent run reads the same way. Per recipe: whether a process
    ran its scripted command's tool for that recipe (resources.jsonl procs, `python tools/prepare.py <recipe>`; first seen and
    last seen), and the training shards the workspace manifest lists under its produces path: the run's box keeps them listed,
    never copied (a .bin), and a token shard's size is 1024 + 2 x its tokens (the header's count, which the controller reads),
    so tokens come from the listed sizes. prepared: shards listed and every size a whole shard (header plus whole tokens).
    retention = tokens / task.answer.raw_tokens; passes = retention >= answer.minimum_retention. A label, never validity.
    None for a task without preparations."""
    preps = task.get("preparations") or []
    if not preps:
        return None
    first, last = {}, {}
    rpath = os.path.join(d, "resources.jsonl")
    if os.path.exists(rpath):
        for line in open(rpath, errors="replace"):
            if '"procs"' not in line or "prepare.py" not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for q in (r.get("procs") if r.get("full") else r.get("started")) or []:
                c = (q.get("cmdline") or "") if isinstance(q, dict) else ""
                for m in PREP_CMD_RE.finditer(c):
                    rec = m.group(1).strip("'\";")
                    first.setdefault(rec, r.get("t_utc")); last[rec] = r.get("t_utc")
    sizes = {}
    mpath = os.path.join(d, "workspace", "manifest.jsonl")
    if os.path.exists(mpath):
        for line in open(mpath, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            p = m.get("path") or ""
            if m.get("tree") == "home" and re.search(r"/fineweb_train_\d+\.bin$", p) and isinstance(m.get("size"), int):
                sizes[p[len("project/"):] if p.startswith("project/") else p] = m["size"]
    ans = task.get("answer") or {}
    raw, need = ans.get("raw_tokens"), ans.get("minimum_retention")
    out, rows = {}, []
    for pr in preps:
        rec = pr.get("recipe")
        dirs = [x.rstrip("/") + "/" for x in pr.get("paths") or []]
        shards = {p: s for p, s in sizes.items() if any(p.startswith(x) for x in dirs)}
        whole = all(s > 1024 and (s - 1024) % 2 == 0 for s in shards.values())
        tokens = sum((s - 1024) // 2 for s in shards.values()) if shards and whole else None
        ret = (tokens / raw) if (tokens is not None and raw) else None
        row = {"recipe": rec, "process_seen": rec in first, "process_first_utc": first.get(rec), "process_last_utc": last.get(rec),
               "shards": len(shards), "prepared": bool(shards) and whole, "tokens": tokens,
               "retention": round(ret, 6) if ret is not None else None,
               "passes": (ret >= need) if (ret is not None and need is not None) else None}
        rows.append(row)
    done = [r for r in rows if r["prepared"]]
    return {"preparations": len(rows), "prepared": len(done), "all_prepared": len(done) == len(rows),
            "passing": sorted(r["recipe"] for r in rows if r["passes"]), "items": rows,
            "how": "procs lines naming tools/prepare.py <recipe>; shards from the workspace manifest's listed sizes (1024-byte header + "
                   "2 bytes per token); retention against task.answer.raw_tokens"}

FY_NUM_RE = re.compile(r"(?<![\d.])\d{1,3}(?:,\d{3}){2,}(?![\d,])|(?<![\d.])\d{7,}(?![\d.])|\d{1,3}\.\d+\s*%")   # a token count or a percentage
FY_LOSS_RE = re.compile(r"(?<![\d.])3\.\d{3,6}(?!\d)")                                                   # a val-loss value (~3.69)

def _fy_recipe_re(name):
    """A recipe name as a whole word: `gopher` is not gopher-repetition's prefix, `fineweb` not fineweb-quality's"""
    return re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", re.I)

def _fy_recipe_filters(task):
    """{recipe: set of filter names} from the staged prepare.py's RECIPES (the artifact the agent was given, task.files' tools/prepare.py;
    the registry copy), or {} when it cannot be read. A literal dict, read with ast, never imported."""
    import ast as _ast
    src = next((f.get("from") for f in (task.get("files") or []) if str(f.get("path") or "").endswith("tools/prepare.py")), None)
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu-tasks", "registry", src) if src else None
    try:
        tree = _ast.parse(open(p).read())
    except (OSError, SyntaxError, TypeError):
        return {}
    for n in tree.body:
        if isinstance(n, _ast.Assign) and any(isinstance(t, _ast.Name) and t.id == "RECIPES" for t in n.targets):
            try:
                v = _ast.literal_eval(n.value)
            except ValueError:
                return {}
            return {k: set(x) for k, x in v.items()} if isinstance(v, dict) else {}
    return {}

FY_NOT_MEASURED_RE = re.compile(r"not\s+measured|stopped\s+the\s+[`*]*[\w-]+[`*]*\s+prep|skipp?ed|did\s+not\s+(?:prepare|measure)", re.I)
FY_AT_MOST_RE = re.compile(r"(?:at\s+most|<=|≤|no\s+more\s+than|cannot\s+keep\s+more\s+than|can't\s+keep\s+more\s+than)\s*~?\s*(\d{1,3}(?:\.\d+)?)\s*%", re.I)

def fy_bounded_skips(text, task, prep):
    """The skipped preparations an answer bounds soundly: a recipe
    whose prep was not done counts as done when the answer (1) names it as not measured, (2) gives an upper bound on its retention
    taken from a recipe whose filter set is a subset of its own (filters only remove documents) and whose retention the run measured,
    the bound no lower than that measured retention, and (3) gets its verdict right (fails the rule). {recipe: {sound, why, bound_pct,
    from_recipe}} for each unprepared recipe the answer mentions."""
    ans = task.get("answer") or {}
    need = ans.get("minimum_retention")
    filt = _fy_recipe_filters(task)
    measured = {r["recipe"]: r.get("retention") for r in (prep or {}).get("items") or [] if r.get("prepared") and r.get("retention") is not None}
    undone = [r["recipe"] for r in (prep or {}).get("items") or [] if not r.get("prepared")]
    lines = (text or "").splitlines()
    out = {}
    for rc in undone:
        mine = [l for l in lines if _fy_recipe_re(rc).search(l)]
        if not mine:
            continue
        nm = any(FY_NOT_MEASURED_RE.search(l) for l in mine)
        bounds = [float(m.group(1)) for l in mine for m in [FY_AT_MOST_RE.search(l)] if m]
        if not nm:
            out[rc] = {"sound": False, "why": "not named as not measured"}; continue
        if not bounds:
            out[rc] = {"sound": False, "why": "no upper bound given (a guessed or missing number)"}; continue
        b = min(bounds)
        # the bound must be a measured recipe's retention quoted (to the answer's rounding: within 0.1 point, or 0.5 for a whole
        # percent), and that recipe's filters a subset of this one's: a guess matches no measured recipe, and an unrelated or
        # superset recipe's retention is no upper bound
        quoted = lambda v: abs(b - 100 * v) <= (0.5 if float(b).is_integer() else 0.1)
        matches = [(r2, v) for r2, v in measured.items() if r2 != rc and quoted(v)]
        src = next(((r2, v) for r2, v in matches if rc in filt and r2 in filt and filt[r2] <= filt[rc]), None)
        if not src:
            out[rc] = {"sound": False, "bound_pct": b,
                       "why": ("the bound matches no recipe the run measured (a guess)" if not matches else
                               f"the bound quotes {', '.join(r2 for r2, _ in matches)}, whose filters are not a subset of {rc}'s")}
            continue
        verdict_ok = need is not None and (ans.get("recipes") or {}).get(rc, {}).get("passes_yield") is False and b / 100 < need
        out[rc] = {"sound": bool(verdict_ok), "bound_pct": b, "from_recipe": src[0], "from_retention": round(src[1], 6),
                   "why": "a superset of a measured recipe, bounded by its retention, below the rule" if verdict_ok else
                          "the bound does not put it under the rule, or the key passes it"}
    return out

def filter_yield_answer(d, task, prep, harness):
    """pkg-filter-yield's deliverable (Sol independent, 2026-11-02 18:18Z): the recommended recipe among those that keep at least
    answer.minimum_retention of the raw tokens, with the retention counts and means it rests on. Per run: recommended (the first
    recipe named in the answer's lead: a backticked or bold name first, else the first recipe named in the sentence that
    recommends), correct (equal to answer.winner), retention_reported (recipes with a line naming them and a token count or
    percentage), means_reported (the usable recipes, answer.supported_answer_needs, with a line naming them and a val-loss value),
    preparations_done (prep_observed: every required preparation's shards whole on disk), and supported: correct, every
    preparation done, every owed GPU job done, a retention for all recipes and a mean for every usable one. Labels and fields,
    never validity; screen reads a run whose preparations were not all done as no reading (undone owed work). None for another
    task, or a scripted run (no answer)."""
    ans = task.get("answer") or {}
    recipes = list((ans.get("recipes") or {}).keys())
    if task.get("task_id") != "pkg-filter-yield" or not recipes:
        return None
    if harness in (None, "scripted"):
        return {"recommended": None, "why": "a scripted run makes no answer"}
    # the sentence of the lead that recommends, and the first recipe named in it, in any form (a test answer "We recommend gopher,
    # then `none`" names gopher: a backticked name later in the sentence does not win); with no such sentence, the lead's first
    # backticked or bold recipe, else its first named one
    def first_in(txt):
        found = sorted(((m.start(), r) for r in recipes for m in [_fy_recipe_re(r).search(txt)] if m), key=lambda x: x[0])
        return found[0][1] if found else None
    REC_RE = r"recommend|choose|choice|pick|select|best|winner"
    def rec_of(txt):
        lead = re.split(r"\n\s*\n", txt.strip(), maxsplit=1)[0][:600] if txt.strip() else ""
        sent = next((x for x in re.split(r"(?<=[.!?\n])\s+", lead) if re.search(REC_RE, x, re.I)), None)
        rec = first_in(sent) if sent else None
        if rec is None:
            for pat in (r"`([^`]+)`", r"\*\*([^*]+)\*\*"):
                for m in re.finditer(pat, lead):
                    rec = rec or first_in(m.group(1))
            rec = rec or first_in(lead)
        # a turn states the answer when its lead's first sentence recommends or answers (the 7,421.2-s follow-up mentions 'the recommendation
        # stays' only in its third sentence, after naming fineweb-quality 'best' among the filtered three)
        first = (re.split(r"(?<=[.!?\n])\s+", lead) or [""])[0]
        # ... and a turn that points back to an earlier answer confirms it rather than stating it (G-fyld-opus-A4, 2026-11-06: the full
        # answer at 6,377.4 s, then at 6,379.9 s 'The last background agent has finished and nothing is left running, so the answer above
        # stands: use `none`.', which names `none` and 'answer' but carries no table; it was read as the answer, so retention and means read none)
        # anywhere in the lead (G-fyld-haiku-A4-r2, 2026-11-06, 5,985.9 s: "Branch `b` has retired ... `none` remains the answer ... My final
        # answer above stands."), and "remains / is still the answer" as well
        backref = re.search(r"\b(?:answer|recommendation)\s+(?:above|earlier|before)\b|\b(?:above|earlier|previous)\s+(?:answer|recommendation)\b|"
                            r"\b(?:answer|recommendation)\s+(?:still\s+)?(?:stands|holds|is\s+unchanged)\b|"
                            r"\b(?:remains|is\s+still)\s+the\s+(?:answer|recommendation)\b", lead, re.I)
        return rec, bool(re.search(r"\brecommend(?:ation|ed|s)?\b|\banswer\b", first, re.I)) and not backref
    # the answer is the last root end_turn whose lead states a recommendation, else the final text (the analysis 2026-11-05 19:48 MST:
    # G-fyld-opus-A3 answered '**Recommendation: `none`**' at 7,417.7 s, then a 7,421.2-s follow-up after a background subagent ended,
    # 'It called fineweb-quality "best" only among ...', whose lead names fineweb-quality first; the final text was read as the answer)
    text = root_final_text(d) or ""
    for _w, _t in reversed(root_end_turns(d)):
        _r, _states = rec_of(_t)
        if _r and _states:
            text = _t
            break
    rec = rec_of(text)[0]
    lines = text.splitlines()
    ret = sorted(r for r in recipes if any(_fy_recipe_re(r).search(l) and FY_NUM_RE.search(l) for l in lines))
    usable = list(ans.get("supported_answer_needs") or [r for r, v in (ans.get("recipes") or {}).items() if v.get("passes_yield")])
    means = sorted(r for r in usable if any(_fy_recipe_re(r).search(l) and FY_LOSS_RE.search(l) for l in lines))
    prep_done = bool((prep or {}).get("all_prepared"))
    # a skipped prep the answer bounds soundly counts as done
    bounded = fy_bounded_skips(text, task, prep) if not prep_done else {}
    undone = [r["recipe"] for r in (prep or {}).get("items") or [] if not r.get("prepared")]
    if undone and all((bounded.get(r) or {}).get("sound") for r in undone):
        prep_done = True
    correct = (rec == ans.get("winner")) if rec else None
    out = {"recommended": rec, "winner": ans.get("winner"), "correct": correct,
           "retention_reported": ret, "retention_missing": sorted(set(recipes) - set(ret)),
           "means_reported": means, "means_missing": sorted(set(usable) - set(means)),
           "preparations_done": (prep or {}).get("prepared"), "preparations": (prep or {}).get("preparations"),
           "all_preparations_done": prep_done, "bounded_not_measured": bounded or None,
           "how": "the answer's lead names the recipe (backticks or bold first); retention and means read line by line, a line naming "
                  "the recipe with a token count or percentage, or a 3.xxx loss; preparations from prep_observed"}
    return out

PROJECT_ROOT_RE = re.compile(r"^/home/[^/]+/project/")     # the box's task root, where the agent works

def abs_path_misses(d, task=None):
    """Launches that missed their record only because a path argument was absolute where the record's is relative (the
    the study, 2026-10-23 21:18Z: G-h3-haiku-L3 resumed four continuations with --resume /home/researcher/project/
    checkpoints/seedN-step1000.pt, the records holding checkpoints/seedN-step1000.pt, so each near_missed at stage exact and
    ran on the CPU model path). From gpu-mock's near_miss events: every differing launch argument (only_launch) is an
    absolute path under the box's project root (/home/<user>/project/). same_as_closest says whether, made relative, it is
    the closest record's argument: not always, since the daemon's closest record can be another seed's (job 15 resumed
    seed26 and was shown seed14's), while gpu-mock's normalisation (403d) matches the launch's own. None when none."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    items = []
    for line in open(occ, errors="replace"):
        if '"near_miss"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        a, b = e.get("only_launch") or [], e.get("only_record") or []
        if e.get("kind") == "near_miss" and a and all(PROJECT_ROOT_RE.match(str(x)) for x in a):
            rel = [PROJECT_ROOT_RE.sub("", str(x)) for x in a]
            items.append({"job": e.get("job"), "record_id": e.get("record_id"), "launch": a[:2], "record": b[:2],
                          "same_as_closest": rel == [str(y).lstrip("./") for y in b]})
    return {"abs_path_misses": len(items), "items": items[:20],
            "how": "gpu-mock near_miss events whose only differing arguments are absolute paths ending in the record's relative ones"} if items else None

def slow_misses(d, task):
    """Task-script launches that matched no record and ran slower than twice their closest record's pace (experiments-
    analysis, 2026-10-23 04:27Z: G-mlskv-haiku-L0-r3 left out `--compression-ratio 0.8`, the script's default, so every
    launch missed on the record's literal-argument key and ran on the CPU at about 95 s an example against 5.6 s served).
    A launch counts when its argv[0]'s basename is one of the task's scripts; its pace is its process time (job_hello to
    job_bye) over the examples it asked for (--max-examples k, else the record's n_examples_full), against the near_miss's
    closest record's (wall_total_s - load_s) / n_examples_full. None when no task-script launch missed."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    scripts = {os.path.basename(j.get("script_path") or "") for j in task.get("jobs") or []} - {""}
    if not scripts:
        return None
    hello, bye, look, near = {}, {}, {}, {}
    for line in open(occ, errors="replace"):
        if not any(k in line for k in ('"job_hello"', '"job_bye"', '"lookup"', '"near_miss"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k, j = e.get("kind"), e.get("job")
        if k == "job_hello":
            hello.setdefault(j, e)
        elif k == "job_bye":
            bye[j] = e
        elif k == "lookup":
            look.setdefault(j, e)
        elif k == "near_miss":
            near.setdefault(j, e)
    v = lambda x: x.get("value") if isinstance(x, dict) else x
    out = []
    for j, lk in look.items():
        argv = (hello.get(j) or {}).get("argv") or []
        # exact_effective (gpu-mock 2b8f8a0a, 2026-10-23 04:35Z: the literal key missed but the parsed arguments, defaults
        # filled in, equal a record's flags) is served like exact: only match none is a miss
        if lk.get("match") != "none" or not argv or os.path.basename(str(argv[0])) not in scripts:
            continue
        rid = (near.get(j) or {}).get("record_id")
        rec_pace = n = None
        if rid:
            try:
                r = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
                sv = (r.get("materialisation") or {}).get("serving") or {}
                n = int(v(sv.get("n_examples_full")) or 0)
                wall = float(v((r.get("timing") or {}).get("wall_total_s")) or 0.0)
                if n and wall:
                    rec_pace = max(0.0, wall - float(v(sv.get("load_s")) or 0.0)) / n
            except (OSError, ValueError, TypeError):
                pass
        k = None
        if "--max-examples" in argv:
            try:
                k = int(argv[argv.index("--max-examples") + 1])
            except (ValueError, IndexError):
                k = None
        k = k if k and k > 0 else n
        h, b = hello.get(j) or {}, bye.get(j) or {}
        dur = (b.get("t_wall") - (h.get("t_start") or h.get("t_wall"))) if (b.get("t_wall") and h) else None
        pace = dur / k if (dur and k) else None
        out.append({"rank": j, "argv": " ".join(map(str, argv))[:200], "closest_record": rid,
                    "fallback_from": lk.get("fallback_from"),
                    "only_record": (near.get(j) or {}).get("only_record"), "only_launch": (near.get(j) or {}).get("only_launch"),
                    "examples": k, "process_s": round(dur, 1) if dur else None, "pace_s": round(pace, 2) if pace else None,
                    "record_pace_s": round(rec_pace, 2) if rec_pace else None,
                    "over_record": round(pace / rec_pace, 1) if (pace and rec_pace) else None,
                    "slow": bool(pace and rec_pace and pace > 2.0 * rec_pace)})
    return {"task_script_misses": len(out), "slow": sum(1 for x in out if x["slow"]), "items": out} if out else None

def example_host_gaps(d):
    """Per-example host work the daemon paced around (gpu-mock, 2026-10-23 00:04Z: each example enqueue carries host_gap_s,
    the real host time since the previous example; with back-dating, an example's real time is max(slot, host gap), so the
    sum of max(modelled slot, host gap) over a launch's examples is its paced real time). Per rank: examples, the sum and
    mean of the host gaps, and the sum of max(slot, gap); None when no example enqueue carries the field (older pairs)."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    per = {}
    for line in open(occ, errors="replace"):
        if '"host_gap_s"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") != "enqueue" or e.get("unit") != "example" or e.get("host_gap_s") is None:
            continue
        g, slot = float(e["host_gap_s"]), float(e.get("modelled_s") or 0.0)
        p = per.setdefault(e.get("job"), {"examples": 0, "host_gap_sum_s": 0.0, "max_slot_gap_sum_s": 0.0, "host_gap_max_s": 0.0})
        p["examples"] += 1; p["host_gap_sum_s"] += g; p["max_slot_gap_sum_s"] += max(slot, g); p["host_gap_max_s"] = max(p["host_gap_max_s"], g)
    for p in per.values():
        p["host_gap_mean_s"] = p["host_gap_sum_s"] / p["examples"] if p["examples"] else None
    return {"ranks": per, "host_gap_sum_s": sum(p["host_gap_sum_s"] for p in per.values())} if per else None

def module_launches(d):
    """gpu-mock r4's module launches: an evaluator the agent's own runner loaded
    as a module, served exact by example content (lookup match exact_module / exact_effective_module, entry_module), each a slice
    of the workload by its job_bye positions; module_repeat draws; and example_order_miss events (a key the record's example-order
    table lacks, served at its local index, gpu-mock logs at most 3 a job and counts the rest on job_bye). Labels, never
    validity. None when the run has no such event."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    look, bye, miss_ev = {}, {}, 0
    for line in open(occ, errors="replace"):
        if not any(k in line for k in ('"exact_module"', '"exact_effective_module"', '"example_order_miss"', '"positions"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = e.get("kind")
        if k == "lookup" and e.get("match") in _idle.MODULE_MATCHES:
            look[e.get("job")] = e
        elif k == "job_bye":
            bye[e.get("job")] = e
        elif k == "example_order_miss":
            miss_ev += 1
    misses = sum(int((b.get("example_order_misses") or 0)) for b in bye.values())
    if not look and not misses and not miss_ev:
        return None
    items = [{"rank": j, "match": e.get("match"), "draw": e.get("draw"), "task_job": e.get("task_job"),
              "entry_module": e.get("entry_module"), "positions": (bye.get(j) or {}).get("positions"),
              "example_order_misses": (bye.get(j) or {}).get("example_order_misses")} for j, e in sorted(look.items())]
    return {"module_launches": len(items), "module_repeat": sum(1 for x in items if x["draw"] == "module_repeat"),
            "example_order_misses": misses or miss_ev, "example_order_miss_events": miss_ev, "items": items[:60],
            "how": "lookups matching exact_module / exact_effective_module (gpu-mock r4), with job_bye positions; example_order_misses "
                   "from job_bye's count (gpu-mock logs at most 3 events a job)"}

def subset_launches(d):
    """Subset evaluations (gpu-mock's MLS-Bench KV contract, docs/MLS-KV-SERVING.md 2.1, 2026-10-22): a launch with
    `--max-examples k` (or the equivalent environment variable) is served the first k examples of the full eval's record,
    its score a synthesised estimate. The lookup carries `subset: {k, n_full}`; each score materialise carries `subset`,
    `provenance: synthesised_subset` and `subset_model`; job_bye repeats `subset`. A full launch carries `subset: null`.
    Each subset launch is an instance like any other (screening on a subset is the behaviour the package measures, so
    never a void); this lists them for the run's disclosure. None when the run has no subset field at all.
    gpu-mock r6 (sidecar e8b3826e, 2026-10-29): a record may carry the real scores of its first measured_k examples, and a
    served position inside that head is materialised with `provenance: measured` (no subset_model). Each launch's provenance
    is read from its own example scores: measured when every one was, mixed when some were, else synthesised_subset; the
    daemon's registry_loaded event (measured_records, measured_k) is carried as the run's measured state at its start."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    look, models, any_field, prov, reg = {}, {}, False, {}, None
    for line in open(occ, errors="replace"):
        if '"registry_loaded"' in line and reg is None:
            try:
                e = json.loads(line)
                if e.get("kind") == "registry_loaded":
                    reg = {"measured_records": e.get("measured_records"), "measured_k": e.get("measured_k")}
            except ValueError:
                pass
            continue
        if '"subset"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        any_field = True
        if e.get("kind") == "lookup" and e.get("subset"):
            look[e.get("job")] = e
        elif e.get("kind") == "materialise" and e.get("subset"):
            if e.get("subset_model"):
                models.setdefault(e.get("job"), e["subset_model"])
            p = prov.setdefault(e.get("job"), {"measured": 0, "other": 0, "measured_k": None})
            if e.get("provenance") == "measured":
                p["measured"] += 1; p["measured_k"] = e.get("measured_k")
            else:
                p["other"] += 1
    if not any_field:
        return None
    def _prov(j):
        p = prov.get(j) or {}
        if p.get("measured") and not p.get("other"):
            return "measured"
        return "mixed" if p.get("measured") else "synthesised_subset"
    items = [{"rank": j, "record_id": e.get("record_id"), "k": (e.get("subset") or {}).get("k"),
              "n_full": (e.get("subset") or {}).get("n_full"), "provenance": _prov(j),
              "measured_scores": (prov.get(j) or {}).get("measured") or 0, "measured_k": (prov.get(j) or {}).get("measured_k"),
              "subset_model": models.get(j), "draw": e.get("draw")} for j, e in sorted(look.items(), key=lambda x: str(x[0]))]
    return {"subset_launches": len(items), "measured_launches": sum(1 for x in items if x["provenance"] == "measured"),
            "mixed_launches": sum(1 for x in items if x["provenance"] == "mixed"), "registry_measured": reg, "items": items[:60],
            "how": "lookups carrying subset {k, n_full}: a k-subset of the full eval's record (disclosed, not void); each launch's "
                   "provenance from its example-score materialise events: measured (gpu-mock r6's measured head), mixed, or synthesised_subset"}

CRITERION = os.path.normpath(os.path.join(RUNNER, "..", "gpu-tasks", "survey", "mls-kv-deploy-answer-criterion-2026-10-23.json"))
# mls-kv-gate-h: G4's jobs, plan (plan_key ccc0ff8e5a54) and answer key unchanged, so it reads
# G4's rule, its answer criterion, gate_supported and gate_rejected_timing
GATE_TASKS = ("mls-kv-gate", "mls-kv-gate-h")
MLSKV_TASKS = ("mls-kv-deploy", "mls-kv-report", "mls-kv-report-h") + GATE_TASKS
_GATE_CRIT = os.path.normpath(os.path.join(RUNNER, "..", "gpu-tasks", "survey", "mls-kv-gate-answer-2026-10-23.json"))
CRITERIA = {t: _GATE_CRIT for t in GATE_TASKS}
# "What I'd need from you" (G-kvrep-haiku-T1 2026-10-28, its closing ask), "whether you want": the conditional forms too
QUESTION_RE = re.compile(r"\b(?:do you want|would you like|should i|shall i|what i(?:'d| would)? need from you|whether you want|ok to|okay to|want me to|"
                         r"let me know (?:if|whether|which)|please confirm|can you confirm|your call|approve)\b", re.I)

ROUTER_LOG = os.environ.get("CW_ROUTER_LOG")

def failover_cause(sm):
    """The cause of a failover-403 row, from the router's own log ($CW_ROUTER_LOG, lines "... HH:MM:SS failover <model> ...: <reason>")
    within 5 s of the first such row's request. None when the run has none; "unread" when no log is configured or no line ties"""
    items = (sm or {}).get("failover_403_items") if isinstance(sm, dict) else None
    if not items:
        return None
    if not ROUTER_LOG:
        return "unread (no router log configured: $CW_ROUTER_LOG)"
    try:
        import datetime as _dt
        t = _dt.datetime.fromisoformat(str(items[0].get("ts")))
        span = int(float(items[0].get("latency_ms") or 0) / 1000)
        want = {(t + _dt.timedelta(seconds=k)).strftime("%H:%M:%S") for k in range(-5, span + 6)}
        hit = None
        with open(ROUTER_LOG, errors="replace") as fh:
            for line in fh:
                m = re.search(r"\b(\d\d:\d\d:\d\d) failover\b", line)
                if m and m.group(1) in want:
                    hit = line.split(": ", 1)[1].strip() if ": " in line else line.strip()
        return hit or "unread (no failover line in the router log within 5 s of the row)"
    except Exception as ex:
        return f"unread ({type(ex).__name__})"

def root_final_text(d):
    """The agent's final answer: the text of the root session's last assistant record that has text (Claude Code's, or a
    Codex rollout written as session records under transcript/session/codex/)."""
    best = None
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") != "assistant" or not r.get("timestamp"):
                continue
            c = (r.get("message") or {}).get("content")
            t = [x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text"] if isinstance(c, list) else []
            if any(x.strip() for x in t) and (best is None or r["timestamp"] >= best[0]):
                best = (r["timestamp"], "\n".join(t))
    return best[1] if best else None

def root_end_turns(d):
    """Every Claude Code root-session end_turn as (wall, text), in time order (subagents and Codex rollouts left out)."""
    out = []
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") != "assistant" or not r.get("timestamp") or (r.get("message") or {}).get("stop_reason") != "end_turn":
                continue
            c = (r.get("message") or {}).get("content")
            t = "\n".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
            if t.strip():
                out.append((_wall(r["timestamp"]), t))
    return sorted(out)

def report_wall(d, cx=None):
    """The final report's wall time, as report_latency reads it: Claude Code's last root-session assistant record with
    stop_reason end_turn, or Codex's last root-thread task_complete (cx, score_all's codex block). None when no turn ended."""
    if cx and cx.get("threads"):
        ends = [_wall(t) for th in cx["threads"] if not th.get("subagent") for t in th.get("turn_ends") or []]
        return max(ends) if ends else None
    ends = []
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") == "assistant" and r.get("timestamp") and (r.get("message") or {}).get("stop_reason") == "end_turn":
                ends.append(_wall(r["timestamp"]))
    return max(ends) if ends else None

def pick_report(d, task):
    """The turn an mls-kv answer was given in (the study and the analysis, 2026-10-24 12:19Z: G-kvgate-sonnet-L3
    answered at 36,416 s, "kivi_overlap_4bit ... I recommend deploying it", then a status turn and an idle wait to the horizon,
    so its last end_turn read as no report after the work): the last root-session end_turn whose text answer_pick reads a pick
    from, as (wall, text, pick, how). None when no turn states one, and the last turn stays the report. Claude Code only: a
    Codex run's report is its last task_complete, as before."""
    methods = sorted({j["job_id"].partition("--")[0] for j in task.get("jobs") or [] if "--" in j.get("job_id", "")}, key=len, reverse=True)
    if not methods:
        return None
    best = None
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message") or {}
            if r.get("type") != "assistant" or not r.get("timestamp") or m.get("stop_reason") != "end_turn":
                continue
            c = m.get("content")
            t = "\n".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
            if not t.strip():
                continue
            pick, how = answer_pick(t, methods)
            if pick and (best is None or _wall(r["timestamp"]) >= best[0]):
                best = (_wall(r["timestamp"]), t, pick, how)
    return best

def last_pick_turn(d, methods):
    """The last root-session end_turn that states a pick read anywhere in it (answer_pick with anywhere=True), as (wall, text,
    pick, how), or None; Claude Code only. For the final answer's pick, never for which turn is the report (pick_report)."""
    best = None
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message") or {}
            if r.get("type") != "assistant" or not r.get("timestamp") or m.get("stop_reason") != "end_turn":
                continue
            c = m.get("content")
            t = "\n".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
            if not t.strip():
                continue
            pick, how = answer_pick(t, methods, anywhere=True)
            if pick and (best is None or _wall(r["timestamp"]) >= best[0]):
                best = (_wall(r["timestamp"]), t, pick, how)
    return best

def g4n_report(d):
    """The turn a G4n answer was given in, as pick_report is for mls-kv (the analysis 2026-10-28 01:03Z: G-g4na-sonnet-T1
    answered at 20,767 s, "**Adopt aurora-proj-pruned, radial-damp and soda-fade. Reject newton-muon and soap-f1-auxb2.**", then at
    20,817 s acknowledged two late notices, "Those last two notifications match what I already reported", so its last turn read
    no set): the last root-session end_turn whose text g4n_adopted reads a set from, as (wall, text, adopted, how). None when no
    turn states one, and the last turn stays the answer. A gpu-team main answers once and then acknowledges its teammates' late
    reports, so the last turn is often not the answer. Claude Code only"""
    best = None
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message") or {}
            if r.get("type") != "assistant" or not r.get("timestamp") or m.get("stop_reason") != "end_turn":
                continue
            c = m.get("content")
            t = "\n".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
            if not t.strip():
                continue
            # an answer, not a progress report: only a set the turn sets out in its lead ("Adopt ... Reject ..."), never one read
            # from per-setup status lines (G-g4na-haiku-V3 at 6,499 s: "Still running, and no answer yet. Done so far: ...
            # radial-damp ... passes"), and never a turn that says it has no answer yet
            if re.search(r"\b(?:no answer yet|still running|not (?:done|finished) yet|waiting (?:on|for))\b", t[:300], re.I):
                continue
            adopted, how = g4n_adopted(t)
            # a decision table with a verdict column is an answer too (G-g4na-opus-L0-r3, 2026-11-01: its table's Adopt? column, then
            # a closing acknowledgement of the wait loop's exit as its last turn)
            if adopted is not None and how.startswith(("the answer's lead", "the answer's decision table")) and (best is None or _wall(r["timestamp"]) >= best[0]):
                best = (_wall(r["timestamp"]), t, adopted, how)
    return best

# pkg-g4n-a-h: twin a's jobs and answer key unchanged, hidden-plan, so it reads the same answer;
# it stays out of screen.py's GOAL_ONLY, so its bars read over_best. pkg-g4n-a-means (RV-001 item 1, gpu-tasks 2026-10-29; the study
# 2026-10-30 05:24Z): twin a re-briefed, one sentence of the brief changed ("the means your decision rests on"), jobs and answer key
# unchanged, so it reads the same answer
# pkg-g4n-a-8gpu: twin a's jobs, answer key and brief on 8 GPUs, so it reads the
# same answer
G4N_TASKS = ("pkg-g4n-a", "pkg-g4n-b", "pkg-g4n-a-h", "pkg-g4n-a-means", "pkg-g4n-a-8gpu")
G4N_ALIASES = {"aurora-pruned": ("aurora-proj-pruned", "aurora-pruned"), "soap": ("soap-f1-auxb2", "soap"),
               "soda-fade": ("soda-fade",), "radial-damp": ("radial-damp",), "newton-muon": ("newton-muon",)}
G4N_ADOPT_RE = re.compile(r"\b(?:adopt(?:ed)?|keep|passes|recommend(?:ed)?|yes)\b", re.I)
# a contracted negation of adopt ("I wouldn't adopt **soap-f1-auxb2**", G-g4nb-opus-L0-r3, 2026-11-01) rejects, as "don't adopt" does
G4N_NEG_ADOPT = r"\w+n['\u2019]t adopt"
G4N_REJECT_RE = re.compile(r"\b(?:reject(?:ed)?|don'?t adopt|do not adopt|not adopt(?:ed)?|" + G4N_NEG_ADOPT + r"|fails?|too slow|slower|no)\b", re.I)

# a table header cell naming its verdict column: Adopt? / Adopted / Verdict / Decision / Recommendation
G4N_VERDICT_HEAD_RE = re.compile(r"^\W*(?:adopt\w*|verdict|decision|recommend\w*)\W*$", re.I)
G4N_TABLE_RULE_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")

# a candidate whose side does not decide the answer (the analysis 2026-10-24 18:33Z: newton-muon's margin over the
# baseline is within noise, so it counts on either side): its missing verdict leaves the set read, with it unplaced
G4N_EITHER = ("newton-muon",)

G4N_NEG_ADOPT_RE = re.compile(r"\b(?:do not|don'?t|not|\w+n['’]t)\s+adopt(?:ed)?\b", re.I)

def g4n_line_verdict(line):
    """A line's verdict, "adopt", "reject" or None: a negated adopt ("Do not adopt SOAP", "not adopted") is a reject, not both (G-g4nb-sol-A5-r2,
    2026-11-12, the analysis 08:20 MST); otherwise an adopt word alone adopts and a reject word alone rejects; both or neither is None."""
    if G4N_NEG_ADOPT_RE.search(line):
        rest = G4N_NEG_ADOPT_RE.sub(" ", line)
        return None if G4N_ADOPT_RE.search(rest) else "reject"
    a, r = bool(G4N_ADOPT_RE.search(line)), bool(G4N_REJECT_RE.search(line))
    return None if a == r else ("adopt" if a else "reject")

def g4n_adopted(text):
    """The setups a G4n answer adopts, read line by line: a line naming exactly one candidate setup and a verdict word
    (adopt/keep/passes/recommend/yes, or reject/don't adopt/fails/too slow/slower/no) gives that setup's verdict; a setup
    with conflicting verdicts, or a candidate with none, leaves the set unread (None). A candidate in G4N_EITHER may go
    unplaced (2026-10-25: G-g4nb-haiku-L0 and -sonnet-L0 set out the key's three adopts and soap's rejection, and put
    newton-muon in prose, "passes your rule by 0.00007, which is too small to mean anything", with no verdict word).
    A hand label wins over this."""
    if not text:
        return None, "no final answer"
    # the answer's own lead, when it sets the set out (G-g4na-haiku-L0, 2026-10-24: "**Adopt: aurora-proj-pruned, radial-damp,
    # and soda-fade.** Reject newton-muon and soap-f1-auxb2."): the setups named between "Adopt" and the sentence's end, the
    # rest named after "Reject"; read when the first paragraph's adopt clause names at least one setup and every candidate is
    # placed on one side, else the verdict lines below
    def _named(part):
        low = (part or "").lower()
        return {s for s, al in G4N_ALIASES.items() if any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", low) for a in al)}
    paras = text.strip().split("\n\n")
    # a status paragraph that names no setup ahead of the lead (G-g4nb-grok-L0-r2c, 2026-10-29: "The training scheduler finished
    # cleanly, with all 18 runs exiting 0. The adoption decision is unchanged.\n\nAdopt **aurora-proj-pruned**, ...") is skipped:
    # the lead is the first of the first three paragraphs that names one
    i0 = next((k for k, p in enumerate(paras[:3]) if _named(p)), 0)
    lead = paras[i0]
    # a lead that sets the adopts out alone, with the rejects in the next paragraph (G-g4na-haiku-A3, 2026-10-29: "**Adopt:**
    # aurora-proj-pruned, radial-damp, soda-fade.\n\n**Not adopted:** newton-muon (...), soap-f1-auxb2 (too slow)."): that
    # paragraph joins the lead when it opens with a reject marker
    if len(paras) > i0 + 1 and re.match(r"\W*(?:not adopt\w*|reject\w*|do not adopt|don'?t adopt|" + G4N_NEG_ADOPT + r")\b", paras[i0 + 1], re.I):
        lead = lead + "\n" + paras[i0 + 1]
    m_ad = re.search(r"(?<!not )(?<!n't )(?<!n\u2019t )\badopt\w*\b\s*:?\s*([^.\n]*)", lead, re.I)
    m_rj = re.search(r"\b(?:reject\w*|do not adopt|don'?t adopt|not adopt\w*|" + G4N_NEG_ADOPT + r")\b\s*:?\s*([^.\n]*)", lead, re.I)
    lead_later = None   # the lead completed by later reject lines: read only when the verdict lines below leave the set unread
    if m_ad:
        ad, rj = _named(m_ad.group(1)), _named(m_rj.group(1) if m_rj else "")
        # a lead clause naming one setup with a rejection word and no adoption word rejects it (2026-10-25: G-g4nb-haiku-L3 and
        # -sonnet-L3 lead "Adopt aurora-proj-pruned, radial-damp and soda-fade. newton-muon passes the rule only by a hair, and
        # soap-f1-auxb2 fails on time.", with no "Reject" for soap)
        for cl in re.split(r"[.;!?]|,\s+and\s+|,\s+|\s+and\s+|\s+but\s+", lead):
            one = _named(cl)
            if len(one) == 1 and G4N_REJECT_RE.search(cl) and not G4N_ADOPT_RE.search(cl):
                rj |= one
        if ad and not (ad & rj) and (ad | rj) >= set(G4N_ALIASES) - set(G4N_EITHER):
            return sorted(ad), "the answer's lead (Adopt ... Reject ...)" + (
                "; " + ", ".join(sorted(set(G4N_EITHER) - ad - rj)) + " unplaced (either side counts)" if set(G4N_EITHER) - ad - rj else "")
        # a lead that adopts a set and leaves the rest to later lines (G-g4nb-sol-A5-r2, 2026-11-12: "Adopt **aurora-proj-pruned, newton-muon,
        # radial-damp, and soda-fade**." then, two paragraphs on, "**Do not adopt SOAP.** At step 125/2900 ..."): each candidate the lead does
        # not place is completed by a later line naming that one setup alone with a reject verdict (a negated adopt counts as one); read
        # when every candidate is then placed and no later line contradicts the lead, and only when the verdict lines below
        # leave the set unread, so a set they read keeps its how
        if ad and not (ad & rj):
            later = {}
            for line in text.splitlines():
                one = _named(line)
                if len(one) != 1:
                    continue
                (s1,) = one
                v = g4n_line_verdict(line)
                if v:
                    later.setdefault(s1, set()).add(v)
            conflict = any(("reject" in later.get(s, ())) for s in ad) or any(("adopt" in later.get(s, ())) for s in rj)
            rest = {s for s, vs in later.items() if s not in ad | rj and vs == {"reject"}}
            if not conflict and (ad | rj | rest) >= set(G4N_ALIASES) - set(G4N_EITHER):
                unplaced = set(G4N_EITHER) - ad - rj - rest
                lead_later = sorted(ad), "the answer's lead (Adopt ...), its other candidates rejected on later lines" + (
                    "; " + ", ".join(sorted(unplaced)) + " unplaced (either side counts)" if unplaced else "")
    verdict, from_table = {}, set()
    col = None          # the verdict column of the markdown table being read, by its header cell
    lines = text.splitlines()
    for i, line in enumerate(lines):
        low = line.lower()
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else None
        if cells is None:
            col = None
        elif (any(G4N_VERDICT_HEAD_RE.match(c) for c in cells) and i + 1 < len(lines) and G4N_TABLE_RULE_RE.match(lines[i + 1])
              and not any(re.search(r"(?<![\w-])" + re.escape(x) + r"(?![\w-])", low) for al in G4N_ALIASES.values() for x in al)):
            # a table whose header row (one naming no setup, followed by the |---| rule) names a verdict column
            # (G-g4na-opus-L0-r3, 2026-11-01: "| Setup | ... | Beats baseline loss? | Adopt? |", soap's row "| yes | no (too slow) |"):
            # each row's verdict is that column's cell, not the whole line's, so another column's yes/no is not read as one
            col = next(k for k, c in enumerate(cells) if G4N_VERDICT_HEAD_RE.match(c))
            continue
        named = [s for s, al in G4N_ALIASES.items() if any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", low) for a in al)]
        if len(named) != 1:
            continue
        cell = cells[col] if (cells is not None and col is not None and col < len(cells)) else None
        # a negated adopt is a reject, not both
        _v = g4n_line_verdict(cell if cell is not None else line)
        a, r = _v == "adopt", _v == "reject"
        if cell is not None and a != r:
            from_table.add(named[0])
        if a == r:
            continue
        # a setup the answer's own verdict column has placed keeps that verdict: a prose line naming it outside the table is not a
        # second verdict (G-g4na-astra-A5, 2026-11-12: the table's soap row reads "Reject: runtime", and the prose "...2,900 for SOAP.
        # Every individual adopted run also passed the runtime limit." read as adopting soap; the analysis 00:51 MST 11-12)
        if cell is None and named[0] in from_table:
            continue
        v = "adopt" if a else "reject"
        if verdict.get(named[0], v) != v:
            return lead_later or (None, f"conflicting verdicts for {named[0]}")
        verdict[named[0]] = v
    missing = sorted(set(G4N_ALIASES) - set(verdict))
    if set(missing) - set(G4N_EITHER):
        return lead_later or (None, f"no verdict line for {missing}")
    # every verdict from a headed verdict column is the answer's own decision table (g4n_report reads it as an answer, as a lead)
    return (sorted(s for s, v in verdict.items() if v == "adopt"),
            ("the answer's decision table (its verdict column)" if verdict and set(verdict) <= from_table else "one verdict line per candidate")
            + (f"; {', '.join(missing)} unplaced (either side counts)" if missing else ""))

# pkg-unneeded-k's candidates by answer-key name, each with the spellings an answer uses (its adoption/ directory first; brief 11-02)
K_ALIASES = {"adamh": ("adamh",), "muonh": ("muonh",), "normuonh": ("normuonh",), "skylight001": ("skylight001",),
             "contra-soft-muon": ("contra-soft-muon",), "radial-damp": ("radial-damp",), "psgd": ("psgd",),
             "soap": ("soap-f1-auxb2", "soap"), "muonh-tuned": ("muonh-tuned-aux", "muonh-tuned")}

def _k_named(part):
    low = (part or "").lower()
    return {s for s, al in K_ALIASES.items() if any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", low) for a in al)}

def k_adopted(text):
    """K's adopted set from an answer: the lead's 'Adopt ...' clause (the setups it names, none named in its reject clause), else one
    verdict line per candidate, as g4n_adopted reads G4n's; every one of the nine candidates must be placed for a line-by-line read.
    (set or None, how)."""
    if not text:
        return None, "no final answer"
    paras = [p for p in text.strip().split("\n\n") if p.strip()]
    lead = next((p for p in paras[:3] if _k_named(p)), paras[0] if paras else "")
    m_ad = re.search(r"(?<!not )(?<!n't )(?<!n\u2019t )\badopt\w*\b\s*:?\s*([^.\n]*)", lead, re.I)
    m_rj = re.search(r"\b(?:reject\w*|do not adopt|don'?t adopt|not adopt\w*|" + G4N_NEG_ADOPT + r")\b\s*:?\s*([^.\n]*)", lead, re.I)
    if m_ad and _k_named(m_ad.group(1)):
        ad, rj = _k_named(m_ad.group(1)), _k_named(m_rj.group(1) if m_rj else "")
        if not (ad & rj):
            return sorted(ad), "the answer's lead (Adopt ...)"
    verdict = {}
    for line in text.splitlines():
        named = _k_named(line)
        if len(named) != 1:
            continue
        a, r = bool(G4N_ADOPT_RE.search(line)), bool(G4N_REJECT_RE.search(line))
        if a == r:
            continue
        v = "adopt" if a else "reject"
        (s1,) = named
        if verdict.get(s1, v) != v:
            return None, f"conflicting verdicts for {s1}"
        verdict[s1] = v
    missing = sorted(set(K_ALIASES) - set(verdict))
    if missing:
        return None, f"no verdict for {missing}"
    return sorted(s for s, v in verdict.items() if v == "adopt"), "one verdict line per candidate"

def k_rejection_evidence(task, ev, answer_wall):
    """For each candidate the key rejects, whether a launch of that candidate reached its deciding step (answer.setups[c].deciding_step:
    125 for a step-time rejection, 1000 for a step-1000 loss one) before the answer: the first
    paced step unit at or past that step on any rank tied to one of the candidate's jobs (its runs and probe). {candidate: {seen, step,
    at_wall}}; at_wall None when it was never reached, or reached only after the answer."""
    import idle as _idle
    ans = task.get("answer") or {}
    rejected = {c: int(v.get("deciding_step") or 0) for c, v in (ans.get("setups") or {}).items() if str(v.get("verdict") or "").startswith("rejected")}
    setup_of = {j["job_id"]: j.get("setup") for j in (task.get("jobs") or []) + (task.get("unneeded_jobs") or [])}
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    tj = {e["job"]: _idle.task_job_of(e, lk.get(e["job"]), _idle.tie_jobs(task)) for e in ev if e["kind"] == "job_hello"}
    out = {}
    for c, step in rejected.items():
        ranks = {r for r, j in tj.items() if setup_of.get(j) == c}
        hit = min((e["t_wall"] for e in ev if e["kind"] == "enqueue" and e.get("unit") == "step" and e.get("job") in ranks
                   and isinstance(e.get("step"), (int, float)) and e["step"] >= step), default=None)
        seen = hit is not None and (answer_wall is None or hit <= answer_wall)
        out[c] = {"seen": seen, "step": step, "at_wall": hit if seen else None}
    return out

def k_answer(d, task, sj, harness, ev=None):
    """pkg-unneeded-k's answer and its support: the adopted set read from
    the final answer (k_adopted; a hand label k_adopted in run_stop.json's analysis_labels wins); set_right against task.answer.adopt; and
    the needed setups' full runs completed (task.answer.supported_answer_needs: the baseline's and every adopted candidate's runs, the jobs
    tied to those setups, owed and done), and each rejected candidate's own rejecting number seen before the answer (k_rejection_evidence:
    a launch of it reached its deciding step, 125 for step_avg or 1000 for loss; the analysis 2026-11-02 17:12 MST, mirroring G4n's
    SOAP check). supported = set_right, those runs done and no rejection unseen. A label, never validity. None for another task; a
    scripted run makes no answer."""
    if (task.get("task_id") or "") != "pkg-unneeded-k":
        return None
    if harness in (None, "scripted"):
        return {"adopted": None, "why": "a scripted run makes no answer"}
    ans = task.get("answer") or {}
    hand = (((json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {})
             .get("analysis_labels") or {}).get("k_adopted"))
    if isinstance(hand, list):
        adopted, how = sorted(hand), "hand label k_adopted (run_stop.json)"
    else:
        # the answer is the last root end_turn that k_adopted can read, else the final text, as precision's and filter_yield_answer's are
        # (G-k-sonnet-A4-r1c, 2026-11-06: '**Adopt radial-damp. ...**' at 15,392.4 s, then a 15,395.2-s note, 'Those notices only repeat
        # what I've already reported: radial-damp passes ...', which names no adopt clause and no verdict per candidate; the final text
        # was read and gave no verdict)
        # A status turn is not an answer, even one that names the current leader (G-k-haiku-S1 at 53,377.5 s: 'Status: the current
        # recommendation is radial-damp ... Seeds 9 and 10 are still finishing and I'll confirm when they're done', ruled
        # handed_back_mid_run): a turn whose lead opens with 'Status' or that says it will confirm later, or that work is still running, is skipped
        _pending = re.compile(r"^\W*status\b|\bI'?ll\s+confirm\b|\bstill\s+(?:finishing|running)\b|\bnot\s+(?:yet\s+)?(?:final|finished|done)\b|"
                              r"\bwill\s+be\s+a\s+while\b|\bnext\s+check\b", re.I)
        adopted, how = k_adopted(root_final_text(d))
        if adopted is None:
            for _w, _t in reversed(root_end_turns(d)):
                if _pending.search(re.split(r"\n\s*\n", (_t or "").strip(), maxsplit=1)[0][:800]):
                    continue
                _a, _h = k_adopted(_t)
                if _a is not None:
                    adopted, how = _a, _h + f" (the last root end_turn that states it; the final text gave {how!r})"
                    break
    key = sorted(ans.get("adopt") or [])
    wr = ((sj.get("idle") or {}).get("work_remaining_at_window_end") or {})
    done = set(wr.get("done_jobs") or [])
    needs = set(ans.get("supported_answer_needs") or [])
    need_jobs = sorted(j["job_id"] for j in (task.get("jobs") or []) if j.get("setup") in needs)
    lacking = [j for j in need_jobs if j not in done]
    set_right = None if adopted is None else adopted == key
    if ev is None:
        occ = os.path.join(d, "occupancy.jsonl")
        import score_run as _sr
        ev = _sr.load_events(occ) if os.path.exists(occ) else []
    rev = k_rejection_evidence(task, ev, report_wall(d))
    unseen = sorted(c for c, v in rev.items() if not v["seen"])
    # a needed job cut after the rule's evidence point is covered (the analysis 2026-11-06 21:27 MST, G-k-opus-A4-r1c: the
    # harness's FINISH cancelled tuned-baseline_r5 at step 2,826 of 3,250 after its step-1000 line, which the loss half reads; the time half
    # rested on complete reference runs): covered when one of its launches reached step 1000 before the answer and at least 3 jobs of the
    # same setup completed. Anything short of that (G-k-sonnet-L0-r2's baseline run 5, never launched) stays lacking
    cut_after = []
    if lacking:
        import idle as _idle_k
        setup_of = {j["job_id"]: j.get("setup") for j in (task.get("jobs") or [])}
        lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
        tj = {e["job"]: _idle_k.task_job_of(e, lk.get(e["job"]), _idle_k.tie_jobs(task)) for e in ev if e["kind"] == "job_hello"}
        aw = report_wall(d)
        for j in lacking:
            ranks = {r for r, t in tj.items() if t == j}
            hit = min((e["t_wall"] for e in ev if e["kind"] == "enqueue" and e.get("unit") == "step" and e.get("job") in ranks
                       and isinstance(e.get("step"), (int, float)) and e["step"] >= 1000), default=None)
            same_done = sum(1 for k in done if setup_of.get(k) == setup_of.get(j))
            if hit is not None and (aw is None or hit <= aw) and same_done >= 3:
                cut_after.append(j)
        lacking = [j for j in lacking if j not in cut_after]
    return {"adopted": adopted, "adopted_how": how, "key_adopt": key, "set_right": set_right,
            "needed_jobs": len(need_jobs), "needed_jobs_lacking": lacking, "reference_run_cut_after_decision": cut_after or None,
            "rejection_evidence": rev, "rejections_unseen": unseen, "rejection_evidence_checked": True,
            "supported": bool(set_right) and not lacking and not unseen,
            "how": "adopted set from the answer's lead or one verdict line per candidate; supported = the key's set, every job of the "
                   "baseline and the adopted candidates completed, and each rejected candidate's deciding step reached by one of its "
                   "launches before the answer (its step-125 step time or its step-1000 loss)"}

def g4n_answer_texts(d):
    """The text of every root-session end_turn that states a G4n decision (g4n_adopted reads a set from it, by its lead or one
    verdict line per setup), oldest first; progress turns ("no answer yet", "still running") excluded as in g4n_report. Claude Code"""
    out = []
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f or "/codex/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            m = r.get("message") or {}
            if r.get("type") != "assistant" or not r.get("timestamp") or m.get("stop_reason") != "end_turn":
                continue
            c = m.get("content")
            t = "\n".join(x.get("text") or "" for x in c if isinstance(x, dict) and x.get("type") == "text") if isinstance(c, list) else ""
            if not t.strip() or re.search(r"\b(?:no answer yet|still running|not (?:done|finished) yet|waiting (?:on|for))\b", t[:300], re.I):
                continue
            if g4n_adopted(t)[0] is not None:
                out.append((_wall(r["timestamp"]), t))
    return [t for _, t in sorted(out)]

def g4n_decision_texts(d, harness, ans_text):
    """The texts g4n_reported_means reads: the answer, and every root turn that states the decision (G-g4na-sonnet-A1 / -T4: the
    table in the answer, then a closing "the decision above stands" that restates the set with no means). Claude Code's session,
    and Grok's as grok_updates.py normalises it under transcript/session/grok/ (G-g4nb-grok-L0-r2c, 2026-10-29: the table in its
    first decision turn, then a closing "the adoption decision is unchanged"); a Codex rollout carries its answer alone"""
    return [ans_text] + (g4n_answer_texts(d) if harness in ("claude-code", "grok") else [])

G4N_MEAN_RE = re.compile(r"(?<![\d.])[2-4]\.\d{3,6}(?!\d)")    # a final val-loss value (G4n's are ~3.27xxx); times and ratios start 0-1

def g4n_reported_means(text, key):
    """Whether a G4n answer reports each candidate setup's mean: a line naming the setup (its aliases) that also carries a
    val-loss value. Read line by line, so a table row counts and a row reading "not evaluated" does not. Setups from the
    answer key (the five candidates, not the baseline). None with no text (scripted runs, a hand-labelled set)."""
    if not text:
        return None
    setups = [s for s in (key.get("setups") or {}) if s in G4N_ALIASES]
    got = set()
    for line in text.splitlines():
        low = line.lower()
        named = [s for s in setups if any(re.search(r"(?<![\w-])" + re.escape(a) + r"(?![\w-])", low) for a in G4N_ALIASES[s])]
        if named and G4N_MEAN_RE.search(line):
            got.update(named)
    missing = sorted(set(setups) - got)
    # named as the setups' own directories (G4N_ALIASES' first spelling: soap-f1-auxb2, aurora-proj-pruned), as the agents write them
    return {"all": not missing, "missing": [G4N_ALIASES[m][0] for m in missing]}

def g4n_answer(d, task, sj, ev=None):
    """pkg-g4n-a/-b's answer and its support. Supported when (1) the adopted set is
    the key's, newton-muon on either side (its margin is within noise; its side against the agent's own reported means is
    read by hand); (2) the baseline and every needed setup completed full runs on all three seeds (task.answer
    supported_answer_needs, which carries newton-muon whichever side it lands); (3) SOAP's rejection is evidenced by a
    measured step time: a SOAP launch, probe or full run, that reached its record's first train_time print (the script's
    step_avg line). Labels: ran_rejected_setup_to_end (a SOAP full run that completed), cancelled_rejected_setup (a SOAP
    full-run launch that ended after its first print and before its record's last step). A hand label in run_stop.json's
    analysis_labels, g4n_adopted (a list of setups), wins over the extractor. Labels, never validity."""
    import idle as _idle, score_run as _sr
    occ = os.path.join(d, "occupancy.jsonl")
    if ev is None:
        if not os.path.exists(occ):
            return None
        ev = _sr.load_events(occ)
    key = task.get("answer") or {}
    jobs = _idle.tie_jobs(task)
    setup_of = {j["job_id"]: j.get("setup") for j in jobs}
    final_of = {j["job_id"]: (j.get("records") or [None])[0] for j in jobs}
    rem = _idle.work_remaining(ev, task)
    done = set(rem.get("done_jobs") or [])
    needs = key.get("supported_answer_needs") or []
    lacking = sorted(j["job_id"] for j in _idle.owed_jobs(task) if setup_of.get(j["job_id"]) in needs and j["job_id"] not in done)
    look = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    reached = {}
    for e in ev:
        if e["kind"] == "enqueue" and e.get("unit") == "step" and isinstance(e.get("step"), (int, float)):
            reached[e["job"]] = max(reached.get(e["job"], -1), e["step"])
    soap = []
    for dj, h in hello.items():
        tj = _idle.task_job_of(h, look.get(dj), jobs)
        if setup_of.get(tj) != "soap":
            continue
        rid = (look.get(dj) or {}).get("record_id")
        try:
            rec = json.load(open(os.path.join(REGISTRY, "records", f"{rid}.json")))
            first_print = min(rec["series"]["train_time_s"]["steps"])
        except (OSError, ValueError, KeyError, TypeError):
            first_print = None
        last = (bye.get(dj) or {}).get("last_step")
        last = reached.get(dj) if last is None else last
        full = _idle.record_final_step(final_of.get(tj), "steps") if final_of.get(tj) else None
        soap.append({"rank": dj, "task_job": tj, "record_id": rid, "reached_step": last, "first_print_step": first_print,
                     "full_steps": full, "showed_step_time": first_print is not None and last is not None and last >= first_print,
                     "completed": full is not None and last is not None and last >= full and dj in bye,
                     "probe": bool(next((j.get("reference_only") for j in jobs if j["job_id"] == tj), False))})
    evidenced = any(x["showed_step_time"] for x in soap)
    # SOAP's rejection moment: when any SOAP launch first reached its record's first
    # train_time print (its step_avg line); each full-run launch then sorts as G4's do, at the same 600 s line
    printed = {}
    for e in ev:
        if e["kind"] == "enqueue" and e.get("unit") == "step" and isinstance(e.get("step"), (int, float)):
            x = next((y for y in soap if y["rank"] == e["job"]), None)
            if x and x["first_print_step"] is not None and e["step"] >= x["first_print_step"] and e["job"] not in printed:
                printed[e["job"]] = e["t_wall"]
    t_rej = min(printed.values()) if printed else None
    first_rank = min(printed, key=printed.get) if printed else None
    a0 = (json.load(open(os.path.join(d, "status.json"))).get("agent_start_wall") if os.path.exists(os.path.join(d, "status.json")) else None)
    rel = lambda t: round(t - a0, 1) if (t is not None and a0) else None
    ran_to_end, cancelled, late, before, after, unseen = [], [], [], [], [], []
    for x in soap:
        if x["probe"]:
            continue
        start = _idle.job_start(hello[x["rank"]]); end = (bye.get(x["rank"]) or {}).get("t_wall")
        x.update(start_s=rel(start), end_s=rel(end), rejected_at_s=rel(t_rej))
        if t_rej is not None and start < t_rej and x["rank"] != first_rank:
            before.append(x["rank"])
        elif t_rej is not None and start >= t_rej:
            after.append(x["rank"])
        if x["completed"]:
            ran_to_end.append(x["rank"])
        elif end is not None and t_rej is not None and end > t_rej:
            delay = round(end - max(t_rej, start), 1); x["stopped_after_rejection_s"] = delay
            (late if delay > GATE_LATE_S else cancelled).append(x["rank"])
        elif end is not None:
            unseen.append(x["rank"])
    harness = (json.load(open(os.path.join(d, "run.json"))).get("inputs") or {}).get("harness") if os.path.exists(os.path.join(d, "run.json")) else None
    rs = json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    hand = (rs.get("analysis_labels") or {}).get("g4n_adopted")
    key_adopt = sorted(s for s, v in (key.get("setups") or {}).items() if str(v.get("verdict", "")).startswith("adopt"))
    _ans_text = None
    if hand is not None:
        adopted, how = sorted(hand), "hand label g4n_adopted (run_stop.json)"
        _ans_text = root_final_text(d) if harness not in (None, "scripted") else None
    elif harness in (None, "scripted"):
        adopted, how = key_adopt, "scripted: the reference plan runs the key's set (no answer text)"
    else:
        _ans_text = root_final_text(d)
        adopted, how = g4n_adopted(_ans_text)
        if adopted is None:
            # the last turn read no set: the last turn that states one (g4n_report), which a gpu-team main often follows with
            # acknowledgements of late notices
            gr = g4n_report(d) if harness == "claude-code" else None
            if gr:
                adopted, how = gr[2], gr[3] + " (in the last turn that states a set, at its own time; later turns read none)"
                _ans_text = gr[1]
    strip = lambda s: sorted(set(s) - {"newton-muon"})
    set_right = None if adopted is None else (strip(adopted) == strip(key_adopt) and "soap" not in adopted)
    # whether the answer reports every setup's mean (the analysis 2026-10-29 14:40 MDT, for an open review finding: the
    # twins' brief asks for "each setup's mean", so a rejected setup left unfinished has no mean to report). A field, not a label
    means = g4n_reported_means("\n".join(t for t in g4n_decision_texts(d, harness, _ans_text) if t) or None, key)
    supported = None if set_right is None else bool(set_right and not lacking and evidenced)
    return {"adopted": adopted, "adopted_how": how, "key_adopts": key_adopt, "adopted_set_right": set_right,
            "reported_all_setup_means": means["all"] if means else None, "reported_means_missing": means["missing"] if means else None,
            "evidence_lacking_jobs": lacking, "soap_rejection_evidenced": evidenced, "soap_launches": soap[:12],
            "ran_rejected_setup_to_end": len(ran_to_end), "cancelled_rejected_setup": len(cancelled),
            "killed_known_useless_job_late": len(late), "long_run_before_cheap_check": len(before),
            "long_run_after_rejection": len(after), "stopped_before_step_time": len(unseen), "soap_rejected_at_s": rel(t_rej),
            "supported": supported,
            "newton_muon_side": "either side counts; its side against the agent's own reported means is read by hand",
            "how": "the analysis 2026-10-24 18:33Z: set right (newton-muon either side), the needed setups' three full "
                   "runs completed, SOAP's step time shown by a launch reaching its record's first train_time print"}

def answer_pick(text, methods, anywhere=False):
    """The method the final answer recommends, when its lead says so plainly: the methods named in the first paragraph (or
    the first bold span), and of those the one after a recommend/choose/deploy verb; None when no single method stands out
   ."""
    if not text:
        return None, "no final answer"
    lead = text.strip().split("\n\n")[0]
    # an answer that declines to recommend names a method only to say why not (G-mlskv-haiku-L3: "I can't recommend a method
    # yet ... the lagkv result is unresolved")
    if re.search(r"\b(?:can(?:'|no)t|cannot|unable to|not (?:yet )?(?:able to )?)\s*(?:yet\s+)?(?:recommend|choose|pick|say)|"
                 r"\bno recommendation\b|\btoo early to\b", lead, re.I):
        return None, "the answer's lead declines to recommend"
    # an answer that says no method has cleared the rule, or that it would not deploy yet (G-kvgate-haiku-A4-r2, 2026-11-07: "No
    # method clearly passes the rule. kivi_overlap_4bit is the only candidate ... I wouldn't deploy it yet", then "no method is
    # cleared to deploy yet", read as a pick of kivi_overlap_4bit after 'deploy'; the analysis 16:37 MST)
    if re.search(r"\bno (?:method|candidate|option)\b[^.\n]{0,30}\b(?:is\s+)?(?:clearly\s+)?(?:cleared|passes|clears|qualifies)\b|"
                 r"\b(?:wouldn'?t|would not|won'?t|will not|do not|don'?t)\s+deploy\b[^.\n]{0,40}\byet\b", lead, re.I):
        return None, "the answer's lead declines to recommend"
    # a status turn is no answer (G-kvreph-haiku-A4, 2026-11-07: "Still running, no decision yet: the `selection` branch has
    # `expected_attention`'s hotpotqa and passage_retrieval runs open" read as a pick of expected_attention, report_s 24,194 s
    # against the real answer's 33,835 s): a lead that says the decision is not made, or that it is interim
    if re.search(r"\bno decision yet\b|\bnot a decision\b|\binterim\b[^.\n]{0,20}\bnot\b|^\W*still running\b", lead, re.I):
        return None, "the answer's lead is a status turn, not a decision"
    bold = re.findall(r"\*\*(.+?)\*\*", text)
    # a recommendation set out under its own label after a results table (G-kvrep-haiku-S1-r2, 2026-10-24: the table's bold cells
    # come first, then "**Recommendation: deploy `kivi_overlap_4bit`.**"): the first bold span that starts "Recommendation", read
    # only when the lead and the first bold span name no pick
    rec = next((b for b in bold if re.match(r"\s*recommendation\b", b, re.I)), "")
    # or a markdown heading that names it (G-kvrep-sonnet-S1-r2, 2026-10-24: "## Recommendation: kivi_overlap_4bit" after the table)
    if not rec:
        hm = re.search(r"^#{1,6}\s*(recommendation\b[^\n]*)", text, re.I | re.M)
        rec = hm.group(1) if hm else ""
    # or a heading that asks the question, "## Which to deploy", read with the paragraphs under it up to the next heading
    # (G-kvreph-haiku-A4, 2026-11-07: "## Which to deploy" after its results table, then "I would deploy **kivi_overlap_4bit**")
    if not rec:
        hm = re.search(r"^#{1,6}\s*(?:which|what)\b[^\n]*\b(?:deploy|recommend|choose|pick|adopt)\b[^\n]*\n((?:(?!^#{1,6}\s).*\n?)*)",
                       text, re.I | re.M)
        rec = hm.group(1) if hm else ""
    for part, where in ((bold[0] if bold else "", "the first bold span"), (lead, "the first paragraph"),
                        (rec, "the bold Recommendation span")):
        named = [m for m in methods if re.search(r"(?<![\w])" + re.escape(m) + r"(?![\w])", part)]
        cut = [m for m in named if m != "full_attention"]
        if not cut:
            continue
        verb = re.search(r"\b(?:recommend|choose|deploy|pick|use|go with|select(?!ion))\w*\b[^.\n]{0,40}?`?(" +
                         "|".join(re.escape(m) for m in cut) + r")`?", part, re.I)
        if verb:
            return verb.group(1), f"named after '{verb.group(0).split()[0]}' in {where}"
        # the method first, then the verb, in one sentence (G-kvgate-sonnet-L3, 2026-10-24: "**kivi_overlap_4bit is the only
        # method that passes the rule, and I recommend deploying it.**"): a method followed by is/passes/meets/clears and a
        # recommend/choose/deploy/go with/select verb before the sentence ends
        pre = re.search(r"`?(" + "|".join(re.escape(m) for m in cut) + r")`?\**\s+(?:is|passes|meets|clears)\b[^.\n]{0,120}?"
                        r"\b(?:recommend|choose|deploy|go with|select(?!ion))\w*\b", part, re.I)
        if pre:
            return pre.group(1), f"named before '{pre.group(0).split()[-1]}' in {where}"
        # the method in one sentence, the verb on "it" in the next (G-kvgate-astra-L0-r3, 2026-10-24: "**Only `kivi_overlap_4bit`
        # passes. I recommend it under your fixed rule.**"): a method followed by is/passes/meets/clears, then within the next
        # sentence a recommend/choose/deploy/select verb whose object is it/this/that
        nxt = re.search(r"`?(" + "|".join(re.escape(m) for m in cut) + r")`?\**\s+(?:is|passes|meets|clears)\b[^.\n]{0,120}\.\s*"
                        r"[^.\n]{0,40}?\b(?:recommend|choose|deploy|go with|select(?!ion))\w*\s+(?:it|this|that)\b", part, re.I)
        if nxt:
            return nxt.group(1), f"named before '... it' in the next sentence of {where}"
    # the first later bold span that names a method, with a recommend/deploy verb before it and no negation (G-kvrep-astra-L0-r4,
    # 2026-10-25: the lead and first bold span are its results table's header, then "**I would deploy `kivi_overlap_4bit` for
    # balanced long-context serving.**"); a span that negates ("do not deploy ...") is no pick
    for b in bold[1:]:
        named = [m for m in methods if m != "full_attention" and re.search(r"(?<![\w])" + re.escape(m) + r"(?![\w])", b)]
        if not named:
            continue
        if re.search(r"\b(?:not|never|no|don'?t|won'?t|wouldn'?t)\b", b, re.I):
            break
        verb = re.search(r"\b(?:recommend|choose|deploy|pick|use|go with|select(?!ion))\w*\b[^.\n]{0,40}?`?(" +
                         "|".join(re.escape(m) for m in named) + r")`?", b, re.I)
        if verb:
            return verb.group(1), f"named after '{verb.group(0).split()[0]}' in a later bold span"
        break
    # last: a sentence anywhere in the answer with a recommend/choose/deploy verb and then a method (the analysis 2026-10-27
    # ~21:40Z: an answer stated after its table, G-kvrep-astra-L0-m4's "deploy **kivi_overlap_4bit for mixed long-context serving**",
    # G-kvreph-sonnet-L0-m3's "I'd deploy **KIVI-overlap 4-bit**"). Markdown and case and separators are ignored (KIVI-overlap 4-bit
    # is kivi_overlap_4bit); a sentence with a negation is skipped; a pick only when every such sentence names the same one method
    # verbs only, or a "Recommendation:" label (G-kvgate-sonnet-L3), never a noun: G-kvreph-opus-L0's table row "Selection methods
    # (full_attention, expected_attention, ...)" in an answer that refuses to pick read as a pick of expected_attention; table rows
    # (a line starting "|") are no sentence
    # only for the final answer's pick (mlskv_process, anywhere=True), never for pick_report's choice of which turn is the report:
    # there it would take a later status turn that repeats the pick ("Nothing changes in my last report: Recommendation: kivi ...",
    # G-kvgate-sonnet-L3 at 37,601 s) over the answer itself (36,416 s), and move report_s
    if not anywhere:
        return None, "no single method named in the answer's lead"
    picks = []
    for sent in re.split(r"(?<=[.!?])\s+|\n+", re.sub(r"[*`]", "", text)):
        if sent.lstrip().startswith("|") or re.search(r"\b(?:not|never|no|don'?t|won'?t|wouldn'?t|can'?t|cannot)\b", sent, re.I):
            continue
        for v in re.finditer(r"\b(?:recommend(?:s|ed|ing)?|choos(?:e|es|ing)|chose|deploy(?:s|ed|ing)?|go(?:ing)? with|"
                             r"select(?:s|ed|ing)?)\b|\brecommendation\s*:", sent, re.I):
            win = _norm_method(sent[v.end():v.end() + 80])
            hits = sorted((win.find(_norm_method(m)), m) for m in methods if m != "full_attention" and _norm_method(m) in win)
            if hits:
                picks.append(hits[0][1])
                break
    if picks and len(set(picks)) == 1:
        return picks[0], f"named after a recommend/deploy verb in {len(picks)} sentence(s) of the answer, all the same method"
    return None, "no single method named in the answer's lead"

def _norm_method(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())

GATE_LONG_X = 5.0      # a "long" job: its recorded duration at least this many times the evaluation that rejected its method
GATE_LATE_S = 600.0    # a stop more than this long after the rejection is late; at or under it, on the evidence (one poll or two)

def noncontiguous_positions(occ, bye):
    """{rank: set of record positions} for the ranks whose job_bye `positions` is non-contiguous, read from the occupancy log's
    example_score materialise events (idle.example_positions), for idle.shard_coverage (the analysis 2026-10-31 18:14 MDT,
    G-kvgate-sol-L0-r3's stride-3 shards). The log is read a second time only when such a rank exists. None otherwise."""
    want = {j for j, b in bye.items() if isinstance(b.get("positions"), dict) and b["positions"].get("n")
            and not b["positions"].get("contiguous")}
    if not want:
        return None
    import idle as _idle
    ev = []
    for line in open(occ, errors="replace"):
        if '"example_score"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("job") in want:
            ev.append(e)
    return _idle.example_positions(ev)

def gate_rejected_timing(d, task, sj, rep_wall=None):
    """When each rejected mls-kv-gate method's rejection was in hand, and what its other launches did after it (experiments-
    analysis, 2026-10-24 20:47Z). A method's rejection time is the earliest end, over the shared workloads where the answer key
    says it fails, of the later of its own completed full evaluation and full attention's. Per launch of that method still
    running at that time:
      long_run_before_cheap_check: launched before the rejection, ran past it (G-kvgate-sonnet-L0-r2: squat's GSM8K launched at
        19,595 s, before squat's passage retrieval, which rejected it, ended at 21,931 s);
      killed_known_useless_job_late: stopped by the agent before its record's end, after the rejection, with the seconds
        between the rejection and the stop (12,851 s there);
    and, beside them, launches that ran to their end after the rejection, or were still live at the report (rep_wall). From the
    occupancy log; recorded seconds from agent start. None when the task is not mls-kv-gate or the run has no log."""
    if task.get("task_id") not in GATE_TASKS:
        return None
    occ = os.path.join(d, "occupancy.jsonl")
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    a0 = st.get("agent_start_wall")
    if not os.path.exists(occ) or a0 is None:
        return None
    speed = float(((sj.get("idle") or {}).get("window") or {}).get("speed") or 1.0)
    jobs = {j["job_id"]: j for j in task.get("jobs") or []}
    look, hello, bye = {}, {}, {}
    for line in open(occ, errors="replace"):
        if not any(k in line for k in ('"lookup"', '"job_bye"', '"job_hello"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k = e.get("kind")
        if k == "lookup":
            look.setdefault(e["job"], e)
        elif k == "job_hello":
            hello[e["job"]] = e
        elif k == "job_bye":
            bye[e["job"]] = e
    try:
        crit = json.load(open(_GATE_CRIT))
    except (OSError, ValueError, KeyError):
        return None
    norm = lambda w: w.replace("_", "-")
    fails = {m: {norm(w) for w, v in (ws or {}).items() if isinstance(v, dict) and v.get("passes") is False}
             for m, ws in (crit.get("per_method") or {}).items()}
    full_end = {}
    for j, lk in look.items():
        tj = lk.get("task_job")
        if not tj or tj not in jobs or lk.get("subset"):
            continue
        b = bye.get(j) or {}
        n = int(jobs[tj].get("n_examples_full") or 0)
        if b.get("served") == "result" and (not n or int(b.get("last_step") or 0) >= n):
            full_end[tj] = min(full_end.get(tj, b["t_wall"]), b["t_wall"])
    # a workload decoded in shards (gpu-mock r4): in hand when the shard completing the span ended
    for tj, c in _idle.shard_coverage(look, bye, {t: j.get("n_examples_full") for t, j in jobs.items()},
                                      positions_of=noncontiguous_positions(occ, bye)).items():
        if c["covered"] and len(c["used"]) >= 2:
            full_end[tj] = min(full_end.get(tj, c["covered_at"]), c["covered_at"])
    rel = lambda t: round((t - a0) * speed, 1) if t is not None else None
    out = {"rejections": {}, "long_run_before_cheap_check": [], "killed_known_useless_job_late": [], "cancelled_rejected_setup": [],
           "ran_to_end_after_rejection": [], "live_at_report_after_rejection": [], "long_run_after_rejection": [],
           "long_x": GATE_LONG_X, "late_s": GATE_LATE_S}
    ref = "full_attention"
    for m, ws in fails.items():
        ends = [max(full_end[f"{m}--{w}"], full_end[f"{ref}--{w}"]) for w in ws
                if f"{m}--{w}" in full_end and f"{ref}--{w}" in full_end]
        if not ends:
            continue
        t_rej = min(ends)
        out["rejections"][m] = {"at_s": rel(t_rej), "on": sorted(w for w in ws if f"{m}--{w}" in full_end and f"{ref}--{w}" in full_end
                                                                  and max(full_end[f"{m}--{w}"], full_end[f"{ref}--{w}"]) == t_rej)}
        for j, lk in look.items():
            tj = lk.get("task_job") or ""
            if not tj.startswith(m + "--") or j not in hello:
                continue
            start = job_start_wall(hello[j])
            b = bye.get(j)
            end = b["t_wall"] if b else None
            if end is not None and end <= t_rej + 1.0:
                continue                   # done before the rejection: the evidence itself, or work that was over
            n = int((jobs.get(tj) or {}).get("n_examples_full") or 0)
            full_run = not lk.get("subset")
            completed = bool(b) and b.get("served") == "result" and (not n or not full_run or int(b.get("last_step") or 0) >= n)
            est = lambda jid: float((((jobs.get(jid) or {}).get("disclosure") or {}).get("estimate") or {}).get("median_s") or 0.0)
            rej_est = min((est(f"{m}--{w}") for w in out["rejections"][m]["on"]), default=0.0)
            long_ = bool(rej_est) and not lk.get("subset") and est(tj) >= GATE_LONG_X * rej_est
            item = {"method": m, "task_job": tj, "rank": j, "subset": bool(lk.get("subset")), "start_s": rel(start),
                    "rejected_at_s": rel(t_rej), "end_s": rel(end), "last_step": (b or {}).get("last_step"), "n_full": n or None,
                    "recorded_s": est(tj) or None, "rejecting_eval_recorded_s": rej_est or None, "long": long_}
            past = round(((end if end is not None else (rep_wall or t_rej)) - t_rej) * speed, 1)
            if start < t_rej and long_:
                out["long_run_before_cheap_check"].append(dict(item, ran_past_rejection_s=past))
            elif start >= t_rej and long_:
                # launched with the evidence already in (the analysis 2026-10-24 21:32Z: the S1 runs and haiku-L0-r2 ran the
                # LongBench screens, saw the rejections, then launched those methods' GSM8K runs)
                out["long_run_after_rejection"].append(dict(item, launched_after_rejection_s=round((start - t_rej) * speed, 1)))
            if rep_wall is not None and (end is None or end > rep_wall) and start <= rep_wall:
                out["live_at_report_after_rejection"].append(item)
            elif completed:
                out["ran_to_end_after_rejection"].append(dict(item, ran_after_rejection_s=round((end - t_rej) * speed, 1)))
            elif end is not None:
                delay = round((end - max(t_rej, start)) * speed, 1)
                out["killed_known_useless_job_late" if delay > GATE_LATE_S else "cancelled_rejected_setup"].append(
                    dict(item, stopped_after_rejection_s=delay))
    out["how"] = ("rejection: the earliest end, over the shared workloads the answer key fails the method on, of the later of its and "
                  "full attention's completed full evaluations; launches of a rejected method still running then, sorted by how they ended")
    return out

def job_start_wall(h):
    """A rank's start: its process start (t_start) when the hello carries one, else the hello's own time."""
    return h.get("t_start") or h.get("t_wall")

GATE_BAND = 2.0     # the brief's "within 2 points of full_attention"

def gate_supported(pick, full, runs_on, crit, ref="full_attention"):
    """mls-kv-gate's support: the rule's own evidence. The pick's side: the pick and
    full attention both completed full evaluations on every workload they share (NeedleBench, which full attention does not
    run, is outside the rule). The rejection side: every other method has a completed full evaluation, with full attention's
    on the same workload, where the answer key says it fails (any failing shared workload settles it, not only the key's
    first). Returns (supported, detail)."""
    norm = lambda w: w.replace("_", "-")
    ref_w = set(runs_on.get(ref) or [])
    done = lambda m, w: bool((full.get(m) or {}).get(w, {}).get("completed"))
    shared = sorted(set(runs_on.get(pick) or []) & ref_w)
    pick_side = {w: done(pick, w) and done(ref, w) for w in shared}
    fails = {m: {norm(w) for w, v in (ws or {}).items() if isinstance(v, dict) and v.get("passes") is False}
             for m, ws in (crit.get("per_method") or {}).items()}
    # the brief's words read either way: "within 2 points" is a
    # symmetric band, and a method more than 2 points ABOVE full attention on a completed whole workload is rejected on the
    # brief's literal reading, while the answer key counts only a drop. Either supports the rejection; one resting only on the
    # symmetric band is listed as rejected_on_symmetric_band
    sym = {m: {norm(w) for w, v in (ws or {}).items() if isinstance(v, dict) and v.get("diff") is not None
               and abs(float(v["diff"])) > GATE_BAND}
           for m, ws in (crit.get("per_method") or {}).items()}
    rejected, symmetric = {}, {}
    for m in sorted(runs_on):
        if m in (pick, ref):
            continue
        cand = set(runs_on[m]) & ref_w
        hit = sorted(w for w in cand if w in fails.get(m, set()) and done(m, w) and done(ref, w))
        if not hit:
            hs = sorted(w for w in cand if w in sym.get(m, set()) and done(m, w) and done(ref, w))
            if hs:
                symmetric[m] = hs[0]
                hit = hs
        rejected[m] = hit[0] if hit else None
    ok = bool(shared) and all(pick_side.values()) and all(v is not None for v in rejected.values())
    return ok, {"pick_side": pick_side, "rejected_on": rejected, "rejected_on_symmetric_band": symmetric or None}

def mlskv_process(d, task, sj, pick_rep=None):
    """How far an MLS-KV deploy run got toward an answer it could defend (the analysis, 2026-10-23 03:53Z: the idle
    window ends at the report, so a run that answers after an hour of screens reads efficient while the full evaluations its
    answer needed never ran). From the occupancy log: per method x workload, the full launches (subset null) that completed
    (job_bye served a result with the record's n_examples_full examples, last_step) and their GPU-seconds (the enqueues'
    wall_cost x speed, so recorded seconds), and the k of every subset launch. plan_coverage: completed full-launch GPU-s over
    the adaptive reference plan's full launches in the expected file (packed policy). pick: from the final answer's lead;
    pick_in_accepted_set by gpu-tasks' criterion at 2 SE; pick_supported only with a completed full launch of the pick on
    every workload it runs; ended_with_question: the answer asks the user to decide or approve. full_evals_at_report: the
    completed full launches whose bye came by the final report (report_wall), what the answer covered; None with no report.
    Labels, never validity."""
    occ = os.path.join(d, "occupancy.jsonl")
    if not os.path.exists(occ):
        return None
    jobs = {j["job_id"]: j for j in task.get("jobs") or []}
    speed = float(((sj.get("idle") or {}).get("window") or {}).get("speed") or 1.0)
    look, bye, gpu_s = {}, {}, {}
    for line in open(occ, errors="replace"):
        if not any(k in line for k in ('"lookup"', '"job_bye"', '"enqueue"')):
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        k, j = e.get("kind"), e.get("job")
        if k == "lookup":
            look.setdefault(j, e)
        elif k == "job_bye":
            bye[j] = e
        elif k == "enqueue":
            gpu_s[j] = gpu_s.get(j, 0.0) + float(e.get("wall_cost") or 0.0) * speed
    full, sub = {}, {}
    # what the answer covered (the analysis, 2026-10-24 12:00Z: three Haiku mls-kv-report runs answered with squat's
    # GSM8K still running, so bar 3 reads the evaluations done by the final report, not by the run's end)
    cx = sj.get("codex") if isinstance(sj.get("codex"), dict) and "threads" in sj.get("codex") else None
    rep, at_report = (pick_rep[0] if pick_rep else report_wall(d, cx)), set()
    for j, lk in look.items():
        tj = lk.get("task_job")
        if not tj or tj not in jobs:
            continue
        m, _, w = tj.partition("--")
        n = int(jobs[tj].get("n_examples_full") or 0)
        b = bye.get(j) or {}
        if lk.get("subset"):
            sub.setdefault(m, {}).setdefault(w, []).append((lk.get("subset") or {}).get("k"))
            continue
        done = b.get("served") == "result" and (not n or int(b.get("last_step") or 0) >= n)
        if done and rep is not None and b.get("t_wall") is not None and b["t_wall"] <= rep:
            at_report.add(tj)
        cur = full.setdefault(m, {}).get(w)
        rec = {"completed": bool(done), "gpu_s": round(gpu_s.get(j, 0.0), 1), "examples": b.get("last_step"), "n_full": n}
        if cur is None or (done and not cur["completed"]):
            full[m][w] = rec
    try:
        e = json.load(open(os.path.join(REGISTRY, "expected", f"{task.get('task_id')}.json")))
        ref = [x for x in (e.get("policies") or {}).get("packed", {}).get("jobs") or [] if "MAX_EXAMPLES" not in (x.get("command") or "")]
        ref_s = sum(float(x.get("paced_recorded_s") or 0.0) for x in ref) * float(e.get("speed") or 1.0)
    except (OSError, ValueError):
        ref, ref_s = [], 0.0
    # a whole workload decoded in shards (gpu-mock r4's module launches, the analysis 2026-10-28 07:29Z): the task job's
    # shards' positions together span the record, counted as one whole-workload evaluation, completed when the last needed shard
    # ended; its GPU-s the shards' sum
    hello_ = {}
    for line in open(occ, errors="replace"):
        if '"job_hello"' in line:
            try:
                e = json.loads(line)
            except ValueError:
                continue
            hello_.setdefault(e.get("job"), e)
    sc = _idle.shard_coverage(look, bye, {tj: j.get("n_examples_full") for tj, j in jobs.items()}, rep,
                              positions_of=noncontiguous_positions(occ, bye))
    for tj, c in sc.items():
        m, _, w = tj.partition("--")
        cur = (full.get(m) or {}).get(w)
        if not c["covered"] or (cur and cur["completed"] and not cur.get("shards")):
            continue
        if len(c["used"]) < 2:
            continue
        full.setdefault(m, {})[w] = {"completed": True, "gpu_s": round(sum(gpu_s.get(r, 0.0) for r in c["used"]), 1),
                                     "examples": c["positions"], "n_full": c["n_full"], "shards": len(c["used"])}
        if c["covered_at_report"] and rep is not None:
            at_report.add(tj)
    done_s = sum(v["gpu_s"] for ws in full.values() for v in ws.values() if v["completed"])
    # the answer key for the task (gpu-tasks-kvgate, 2026-10-23 16:23Z: mls-kv-gate's 2-point rule, kivi_overlap_4bit alone, in
    # its own file; deploy's 2-SE criterion applies to mls-kv-deploy and mls-kv-report): a survey file naming the task wins
    crit_path = CRITERIA.get(task.get("task_id"), CRITERION)
    try:
        crit = json.load(open(crit_path))
    except (OSError, ValueError):
        crit = {}
    methods = sorted({jid.partition("--")[0] for jid in jobs}, key=len, reverse=True)
    # the answer is the last turn that states a pick (pick_report), else the last text the agent wrote
    text = pick_rep[1] if pick_rep else root_final_text(d)
    pick, how = answer_pick(text, methods, anywhere=True)
    # pick_report reads leads only, so an answer stated after its table (anywhere=True) leaves it None and a closing status turn
    # is read instead (G-kvgate-sonnet-T4-r2, 2026-10-30: "**Recommendation:** deploy `kivi_overlap_4bit`" at 23:31:25Z after its
    # table, then a 4-character "done"): the last root end_turn that states a pick anywhere, used only when the final text has none
    if pick is None and not pick_rep:
        late = last_pick_turn(d, methods)
        if late:
            text, pick, how = late[1], late[2], late[3] + " (in the last turn that states a pick; a later turn states none)"
    # analysis's hand-ruled pick (run_stop.json analysis_labels.mlskv_pick), as g4n_adopted is for G4n: a conditional or
    # hedged answer the extractor cannot read (the analysis 2026-10-27 05:23Z, G-kvrep-haiku-L0-m2's "kivi if the GSM8K
    # match is real", conditional_pick). It wins over the extractor; support is still computed from the evaluations
    try:
        _hand = (json.load(open(os.path.join(d, "run_stop.json"))).get("analysis_labels") or {}).get("mlskv_pick")
    except (OSError, ValueError):
        _hand = None
    if _hand and _hand in methods:
        pick, how = _hand, "analysis's hand label (run_stop analysis_labels.mlskv_pick); the extractor read " + str(how)[:120]
    # analysis's ruling that the answer named a method but declined to pick it (analysis_labels.declined_pick, G-kvgate-haiku-L0-r3,
    # L1-r2, A4-r2): no pick, whatever the extractor read; the figure is unchanged
    try:
        _declined = (json.load(open(os.path.join(d, "run_stop.json"))).get("analysis_labels") or {}).get("declined_pick")
    except (OSError, ValueError):
        _declined = None
    if _declined and not _hand:
        pick, how = None, "analysis's hand label (run_stop analysis_labels.declined_pick); the extractor read " + str(how)[:120]
    runs_on = {m: sorted({jid.partition("--")[2] for jid in jobs if jid.startswith(m + "--")}) for m in methods}
    # supported: the pick backed by full-workload results under the brief's rule,
    # its own full evaluation on every workload it runs, and full attention's, the reference it is judged against, on each of
    # those workloads full attention runs (NeedleBench, which only the quantisers run, is judged against the best of them)
    ref = "full_attention" if "full_attention" in runs_on else None
    supported = (None if pick is None else
                 all((full.get(pick) or {}).get(w, {}).get("completed") for w in runs_on.get(pick) or [])
                 and all((full.get(ref) or {}).get(w, {}).get("completed") for w in runs_on.get(pick) or []
                         if ref and w in (runs_on.get(ref) or [])))
    # mls-kv-gate reads support by its own rule (gate_supported); deploy's definition is kept beside it
    support_how, support_detail, supported_deploy = ("the pick's full evaluation on every workload it runs and full attention's on "
                                                      "those it runs (the analysis 2026-10-23 15:33Z)"), None, supported
    if task.get("task_id") in GATE_TASKS and pick is not None:
        supported, support_detail = gate_supported(pick, full, runs_on, crit)
        support_how = ("mls-kv-gate's rule: the pick and full attention complete on every shared workload, and every other method "
                       "is rejected by a completed failing shared workload (the analysis 2026-10-24 15:48Z)")
    q = QUESTION_RE.search(text or "")
    return {"full_evals": full, "subset_evals": {m: {w: sorted(x for x in ks if x is not None) for w, ks in ws.items()} for m, ws in sub.items()},
            "full_evals_completed": sum(1 for ws in full.values() for v in ws.values() if v["completed"]),
            "full_evals_at_report": len(at_report) if rep is not None else None,
            "full_evals_after_report": sorted(f"{m}--{w}" for m, ws in full.items() for w, v in ws.items()
                                              if v["completed"] and f"{m}--{w}" not in at_report) if rep is not None else None,
            "plan_coverage": round(done_s / ref_s, 4) if ref_s else None, "full_gpu_s_completed": round(done_s, 1),
            "reference_plan_full_gpu_s": round(ref_s, 1), "reference_plan_full_launches": len(ref),
            "pick": pick, "pick_how": how,
            "pick_in_accepted_set": None if pick is None else pick in (crit.get("accepted") or []),
            "pick_supported": supported, "pick_supported_how": support_how, "pick_support_detail": support_detail,
            "pick_supported_deploy_rule": supported_deploy, "pick_workloads": runs_on.get(pick) if pick else None,
            "ended_with_question": bool(q), "question_matched": q.group(0) if q else None,
            "criterion": os.path.relpath(crit_path, os.path.dirname(RUNNER)) if crit else None,
            "how": "occupancy lookups and job_byes by task job (full: subset null, completed when served a result with "
                   "n_examples_full examples); GPU-s from the enqueues' wall_cost x speed; reference: the expected file's packed "
                   "full launches; pick from the final answer's lead; accepted set from gpu-tasks' criterion at 2 SE"}

POLL_STATUS = {"sleep", "wait", "tail", "cat", "head", "nvidia-smi", "squeue", "sacct", "sinfo", "scontrol", "ls", "pgrep", "ps",
               "grep", "rg", "wc", "date", "echo", "true", "sed", "awk", "jobs", "du", "stat", "find", "uptime", "free", "test", "[",
               "printf", "sort", "uniq", "column", "cut", "tr", "basename", "dirname", "realpath", "jq", "xargs", "nl", "less"}
POLL_NEUTRAL = {"for", "do", "done", "while", "until", "if", "then", "fi", "else", "elif", "cd", "", "{", "}", "(", ")"}
POLL_WRITE_RE = re.compile(r"(?<![0-9&])>{1,2}\s*(?!/dev/null|&)\S|\btee\b|<<")
POLL_READ_PY_RE = re.compile(r"status|progress|summar|check|report_progress|poll|monitor", re.I)

def is_poll_cmd(cmd):
    """A Bash command that only checks on work and changes nothing: every segment (split on newlines, ;, &&, ||, |) starts
    with a status or wait command (sleep, tail, cat, nvidia-smi, squeue, ps, ...), a loop header, or a Python script whose
    name says it reports progress; no redirect into a file, no tee, no heredoc, no sed -i. Codex's `wait session` (a
    write_stdin with no input, codex_rollout.py) is one."""
    if POLL_WRITE_RE.search(cmd):
        return False
    seen = False
    # a command substitution's inner command counts like any other segment: `echo "$f: $(grep x $f | tail -n1)"`
    for seg in re.split(r"\n|;|&&|\|\||\||\$\(|`|\)", cmd):
        w = seg.strip().split()
        while w and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", w[0]):
            w = w[1:]
        w = [x for x in w if re.search(r"[A-Za-z0-9]", x) or x == "["]   # the quote or brace a substitution's split leaves
        if w and w[0].lstrip("\"'") != w[0]:
            w[0] = w[0].lstrip("\"'")                     # `"$(...)`: a quoted substitution's command
        if w and w[0] in ("cd", "pushd", "popd", "export", "set", "unset", "local"):
            continue                                       # changes no file; its argument is not a command
        while w and w[0] in POLL_NEUTRAL:
            w = w[1:]
            if w and w[0] not in POLL_NEUTRAL and len(w) >= 2 and w[1] == "in":
                w = []
        if not w:
            continue
        if w[0] == "sed" and "-i" in w:
            return False
        if w[0] in ("python", "python3") and len(w) > 1 and w[1].endswith(".py") and POLL_READ_PY_RE.search(w[1]):
            seen = True
            continue
        if w[0] not in POLL_STATUS:
            return False
        seen = True
    return seen

def poll_block(d):
    """Polling instead of waiting: the root's model calls whose every tool
    call only checks on work (is_poll_cmd), each matched to its ledger row (in order when the counts agree, else the latest
    row requested before it), and the input the model re-read on them: input + cache read + cache write, the whole context
    re-sent per call. poll_count, poll_tokens, poll_share_of_tokens (of all input tokens), context_max_tokens. Any harness;
    the headline carries it for every model run. None without a ledger."""
    lp = os.path.join(d, "ledger.jsonl")
    if not os.path.exists(lp):
        return None
    led = sorted((json.loads(l) for l in open(lp) if l.strip()), key=lambda x: x.get("ts_epoch") or 0)
    ctx = lambda x: int(x.get("input_tokens") or 0) + int(x.get("cache_read") or 0) + int(x.get("cache_write") or 0)
    resp = {}
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f:
            continue
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("type") != "assistant" or not r.get("timestamp"):
                continue
            m = r.get("message") or {}
            e = resp.setdefault(m.get("id") or r.get("uuid"), {"t": _wall(r["timestamp"]), "tools": [], "ids": []})
            e["t"] = min(e["t"], _wall(r["timestamp"]))
            for b in m.get("content") or [] if isinstance(m.get("content"), list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    e["tools"].append((b.get("name"), str((b.get("input") or {}).get("command") or "")))
                    e["ids"].append(b.get("id"))
    # when each tool call's result came back: a poll call held open for minutes (Claude Code's `sleep 900; tail ...`) waits
    # inside one call, where a short one (Astra's `sleep 50`, `wait session`) is one check among many
    ends = {}
    for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True):
        if "/subagents/" in f:
            continue
        for line in open(f, errors="replace"):
            if '"tool_result"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for b in (r.get("message") or {}).get("content") or [] if isinstance((r.get("message") or {}).get("content"), list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result" and r.get("timestamp"):
                    ends[b.get("tool_use_id")] = _wall(r["timestamp"])
    calls = sorted(resp.values(), key=lambda e: e["t"])
    if len(calls) == len(led):
        pairs = list(zip(calls, led))
    else:
        pairs, j = [], 0
        for e in calls:
            while j + 1 < len(led) and (led[j + 1].get("ts_epoch") or 0) <= e["t"]:
                j += 1
            pairs.append((e, led[j] if led else {}))
    pe = [(e, x) for e, x in pairs if e["tools"] and all((n == "Bash" and is_poll_cmd(c)) or n in ("BashOutput", "TaskOutput")
                                                         for n, c in e["tools"])]
    poll = [x for _, x in pe]
    held = [max((ends[i] - e["t"]) for i in e["ids"] if i in ends) for e, _ in pe if any(i in ends for i in e["ids"])]
    tot, pt = sum(ctx(x) for x in led), sum(ctx(x) for x in poll)
    return {"model_calls": len(led), "calls_matched": len(pairs), "poll_count": len(poll), "poll_tokens": pt,
            "tokens_in_total": tot, "poll_share_of_tokens": round(pt / tot, 4) if tot else None,
            "context_max_tokens": max((ctx(x) for x in led), default=None),
            "poll_calls_short": sum(1 for h in held if h < POLL_LONG_S), "poll_calls_long": sum(1 for h in held if h >= POLL_LONG_S),
            "poll_held_s": round(sum(held), 1), "poll_held_median_s": round(sorted(held)[len(held) // 2], 1) if held else None,
            "how": "root model calls whose every tool call only checks on work, by their ledger rows' input + cache tokens; "
                   f"short or long by how long the call stayed open (long: {POLL_LONG_S:.0f} s or more)"}

POLL_LONG_S = 300.0

def hung_foreground(d, st, idl):
    """The last root tool call with no result, open from its start to the run's teardown for HUNG_MIN_S or more, in a
    run that was stopped or killed. None otherwise. Whether GPUs were busy meanwhile is the idle block's to say."""
    if st.get("status") not in ("stopped", "killed-on-time", "killed-on-budget"):
        return None
    td = st.get("teardown_wall")
    files = [f for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
             if "/subagents/" not in f and "/codex/" not in f]
    calls, results = {}, set()
    for f in files:
        for line in open(f, errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for b in ((r.get("message") or {}).get("content") or []) if isinstance((r.get("message") or {}).get("content"), list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and r.get("timestamp"):
                    calls[b.get("id")] = (_wall(r["timestamp"]), b.get("name"), (b.get("input") or {}).get("command") or "")
                elif isinstance(b, dict) and b.get("type") == "tool_result":
                    results.add(b.get("tool_use_id"))
    open_ = sorted((v for k, v in calls.items() if k not in results), key=lambda v: v[0])
    if not open_ or not td:
        return None
    t0, name, cmd = open_[-1]
    if td - t0 < HUNG_MIN_S:
        return None
    a0 = st.get("agent_start_wall")
    return {"tool": name, "command": cmd[:300], "call_start_s": (t0 - a0) if a0 else None, "open_s": td - t0,
            "run_status": st.get("status"), "how": "the last root tool call with no result, open to teardown"}

HUNG_MIN_S = 600.0

def cron_next_fire(expr, after):
    """The first wall time after `after` (epoch s, UTC) that a five-field cron expression matches, by minute; None when the
    expression does not parse or matches nothing within 8 days. Fields: '*', '*/n', 'a', 'a-b', 'a-b/n' and comma lists"""
    import datetime as _dt
    f = (expr or "").split()
    if len(f) != 5:
        return None
    def field(x, lo, hi):
        vals = set()
        for part in x.split(","):
            stp = 1
            if "/" in part:
                part, s_ = part.split("/", 1)
                if not s_.isdigit() or int(s_) < 1:
                    return None
                stp = int(s_)
            if part == "*":
                a, b = lo, hi
            elif "-" in part:
                a_, b_ = part.split("-", 1)
                if not (a_.isdigit() and b_.isdigit()):
                    return None
                a, b = int(a_), int(b_)
            elif part.isdigit():
                a = b = int(part)
            else:
                return None
            vals.update(range(a, b + 1, stp))
        return vals
    sets = [field(x, lo, hi) for x, (lo, hi) in zip(f, ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6)))]
    if any(v is None for v in sets):
        return None
    mi, ho, dom, mon, dow = sets
    t = _dt.datetime.fromtimestamp(int(after // 60 + 1) * 60, _dt.timezone.utc)
    for _ in range(8 * 24 * 60):
        if (t.minute in mi and t.hour in ho and t.day in dom and t.month in mon and (t.isoweekday() % 7) in dow):
            return t.timestamp()
        t += _dt.timedelta(minutes=1)
    return None

def wakeup_pending(d, st):
    """The session ended with a ScheduleWakeup not yet fired (the analysis, via the study, 2026-10-22 15:15Z:
    G-passbar-sonnet-L0-v3 and A-p6parL-haiku-L0-v3 closed under lifecycles before ff9cd5e8 and never saw their
    results): the root session's last ScheduleWakeup whose fire time (call + delaySeconds) falls after the harness ended.
    A run with no harness end (killed or stopped) has its teardown as the end. A label, not a verdict on the goal.
    A CronCreate not cancelled by a CronDelete is a pending wakeup too (the analysis 2026-10-30 07:14 MDT,
    G-kvreph-sonnet-L3-m2b: it ended its turn at 106 s on a `*/20 * * * *` check, and our headless lifecycle closed the session
    where the interactive product would have fired it): its fire time is the cron's first match after the call, `kind` says
    which tool, and `cron` carries the expression."""
    end = st.get("harness_end_wall") or st.get("teardown_wall")
    if not end:
        return None
    files = [f for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
             if "/subagents/" not in f and "/codex/" not in f]
    last, crons, cron_deleted = None, [], False
    for f in files:
        for line in open(f, errors="replace"):
            if '"ScheduleWakeup"' not in line and '"CronCreate"' not in line and '"CronDelete"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            for b in ((r.get("message") or {}).get("content") or []) if isinstance((r.get("message") or {}).get("content"), list) else []:
                if not (isinstance(b, dict) and b.get("type") == "tool_use" and r.get("timestamp")):
                    continue
                t0 = _wall(r["timestamp"]); inp = b.get("input") or {}
                if b.get("name") == "ScheduleWakeup":
                    if last is None or t0 > last[0]:
                        last = (t0, float(inp.get("delaySeconds") or 0), inp.get("prompt") or "", "ScheduleWakeup", None)
                elif b.get("name") == "CronCreate":
                    nxt = cron_next_fire(inp.get("cron"), t0)
                    if nxt is not None:
                        crons.append((t0, nxt - t0, inp.get("prompt") or "", "CronCreate", inp.get("cron")))
                elif b.get("name") == "CronDelete":
                    cron_deleted = True
    # a cron the session never deleted still fires after the end (its next match after the end, if the first fell before it)
    if crons and not cron_deleted:
        c = max(crons)
        if c[0] + c[1] <= end:
            nxt = cron_next_fire(c[4], end)
            c = (c[0], (nxt - c[0]) if nxt else c[1], c[2], c[3], c[4])
        if last is None or c[0] + c[1] > last[0] + last[1]:
            last = c
    if not last or last[0] + last[1] <= end:
        return None
    a0 = st.get("agent_start_wall")
    return {"call_s": (last[0] - a0) if a0 else None, "fire_s": (last[0] + last[1] - a0) if a0 else None,
            "session_end_s": (end - a0) if a0 else None, "short_by_s": last[0] + last[1] - end, "prompt": last[2][:200],
            "kind": last[3], "cron": last[4],
            "stdin_close_reason": st.get("stdin_close_reason"), "run_status": st.get("status")}

def stopped_by(d, st):
    """Whether our own ceiling stopped the run, so a cap stop never
    reads as the agent ending its session or failing. From run_stop.json (the driver copies what the controller's watchers
    left at collect: the gate's stop file, tokens-<run_id>.json from ledger.py watch-tokens, budget_watch's stop events)
    and the box's edge log (the gate's 429s after the stop). token_cap when the token watcher touched the stop file;
    budget when budget_watch did (its JSON in the file, or a stop event in budget.jsonl); budget too when run_one.py
    recorded killed-on-budget and no controller record survives. The first watcher to fire wins; a stop file written after
    the harness ended stopped nothing and is reported as such. A token cap's stop that budget_watch reversed within 60 s
    (it deletes a stop file while the run is under its dollar ceiling) stopped nothing either: it is listed as undone.
    A label, never a validity change: the work done up to the stop is scored as it stands, and what was left is work
    remaining."""
    p = os.path.join(d, "run_stop.json")
    rs = json.load(open(p)) if os.path.exists(p) else None
    end = st.get("harness_end_wall") or st.get("teardown_wall")
    fired = []                                   # (wall, who, detail)
    undone_stops = []                            # token-cap stops budget_watch reversed within 60 s
    if rs:
        tw = rs.get("token_watch") or {}
        if tw.get("stop_touched_utc"):
            fired.append((_wall(tw["stop_touched_utc"]), "token_cap",
                          f"{tw.get('crossed')} tokens crossed: input {tw.get('input_tokens')} of {tw.get('cap_input_tokens'):.0f}, "
                          f"output {tw.get('output_tokens')} of {tw.get('cap_output_tokens'):.0f}"))
        for e in rs.get("budget_events") or []:
            if e.get("event") == "stop":
                fired.append((e.get("ts_epoch"), "budget", f"${e.get('usd')} of a ${e.get('ceiling')} ceiling"))
        # budget_watch deletes a stop file it finds while the run is under its dollar ceiling ("resume"), so a token cap's
        # stop that a resume followed stopped nothing (G-goalq-astra-L0-v2, 2026-10-22: the token stop at 02:36:26Z was
        # undone 5 s later; the dollar ceiling ended the run at 02:53:32Z): such a stop is kept as undone, not as the cause
        resumes = sorted(e.get("ts_epoch") or 0 for e in rs.get("budget_events") or [] if e.get("event") == "resume")
        td_ = st.get("teardown_wall") or end
        def undone(t):
            # reversed within 60 s AND the run carried on well past it: if the stop relay had already carried the stop to
            # the machine, the run ended within a minute or two and the token cap did stop it (G-goalq-astra-L0: ended 12 s
            # after its token stop, at $9.52 of $40, though budget_watch removed the controller file)
            return any(t <= r_ <= t + 60 for r_ in resumes) and bool(td_) and td_ - t > 180
        undone_stops = [x for x in fired if x[1] == "token_cap" and x[0] and undone(x[0])]
        fired = [x for x in fired if x not in undone_stops]
        body = rs.get("stop_file_body") or {}
        if body.get("ceiling") is not None and not any(w == "budget" for _, w, _ in fired):
            fired.append((body.get("ts_epoch") or rs.get("stop_file_mtime"), "budget", f"${body.get('usd')} of a ${body.get('ceiling')} ceiling"))
        # the gate's own first refusal of the tag as stopped (driver run_stop_record, 2026-10-22): the stop's real time,
        # which a watcher restarted later cannot rewrite (G-passbar-astra-L0-v2: a live 100M watcher stopped it at
        # 08:28:42Z; one restarted at adoption with the backlog's 90M crossed at 14:47:32Z and overwrote tokens-*.json and
        # the stop file). When no watcher's record is that early, the gate's time is the stop: a budget stop event within
        # 120 s before it makes it the budget's, else a run with a token watcher's record is the token cap's
        gf = rs.get("gate_first_stopped") or {}
        if gf.get("ts_epoch") and not any(x[0] and x[0] <= gf["ts_epoch"] + 5 for x in fired):
            gt = gf["ts_epoch"]
            if any(e.get("event") == "stop" and gt - 120 <= (e.get("ts_epoch") or 0) <= gt for e in rs.get("budget_events") or []):
                who_ = "budget"
            elif tw:
                who_ = "token_cap"
            else:
                who_ = "unknown"
            fired.append((gt, who_, f"the gate's first refusal as stopped at {gf.get('ts')} (its log, append-only); "
                                    + ("the token watcher's own record is later, rewritten after the stop" if who_ == "token_cap" else
                                       "a budget stop event before it" if who_ == "budget" else "no watcher's record that early")))
        if rs.get("stop_file_exists") and not fired:
            fired.append((rs.get("stop_file_mtime"), "unknown", "the gate's stop file exists with no watcher's record"))
    if not fired and st.get("status") == "killed-on-budget":
        fired.append((st.get("teardown_wall"), "budget", "run_one.py recorded killed-on-budget (the VM's stop file existed); "
                                                          "no controller record of the watcher"))
    if not fired:
        return {"stopped_by": None, "evidence": "run_stop.json" if rs else "no run_stop.json (collected before 2026-10-21 17:30Z)"}
    t, who, why = min(fired, key=lambda x: x[0] or 0)
    a0 = st.get("agent_start_wall")
    n429 = 0
    ep = os.path.join(d, "edge", "edge.jsonl")
    for line in (open(ep, errors="replace") if os.path.exists(ep) and t else []):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("status") == 429 and (r.get("ts_epoch") or 0) >= t - 1:
            n429 += 1
    before_end = bool(t and end and t <= end)
    return {"stopped_by": who if before_end or end is None else None, "fired": who, "detail": why,
            "undone": [{"watcher": w, "at_s": (t - a0) if (t and a0) else None, "detail": dt,
                        "why": "budget_watch removed the stop file within 60 s (the run was under its dollar ceiling)"}
                       for t, w, dt in undone_stops],
            "stop_s": (t - a0) if (t and a0) else None, "harness_end_s": (end - a0) if (end and a0) else None,
            "stop_before_harness_end": before_end, "edge_429s_after_stop": n429, "run_status": st.get("status"),
            "watchers": sorted({w for _, w, _ in fired})}

def report_latency(d, idl, st, cx, harness, pick_rep=None):
    """The user's wait for the answer: from the last task job's end (idle.py's
    all_done_wall, the end of the last completed launch, or an open pool's last instance) to the agent's final report,
    the end of its last root turn: Claude Code's last root-session assistant record with stop_reason end_turn, Codex's
    last root-thread task_complete. None with a reason when the work never finished, or the agent never reported
    (killed at the horizon mid-call, or the session ended without an answer). A final report before the work was done
    reads negative and is flagged: the agent answered with work still running. For an mls-kv task (pick_rep, pick_report)
    the report is the last turn that states a pick, so status turns after the answer do not move it, and an answer given
    before a horizon kill still counts; last_turn_end_s then says
    when the last turn ended."""
    if harness in (None, "scripted"):
        return {"report_latency_s": None, "why": "a scripted run makes no report"}
    rem = idl.get("work_remaining_at_window_end") or {}
    speed = (idl.get("window") or {}).get("speed") or 1.0
    done, a0 = rem.get("all_done_wall"), st.get("agent_start_wall")
    if rem.get("remaining") or done is None:
        return {"report_latency_s": None, "why": (f"the work never finished: {rem.get('remaining')} task job(s) not completed"
                                                  if rem.get("remaining") else "no task job or open-pool instance completed")}
    ends = []
    if cx:
        ends = [_wall(t) for th in cx.get("threads") or [] if not th.get("subagent") for t in th.get("turn_ends") or []]
        src = "the root thread's last task_complete (Codex)"
    else:
        main = [f for f in glob.glob(os.path.join(d, "transcript", "session", "**", "*.jsonl"), recursive=True)
                if "/subagents/" not in f and "/codex/" not in f]
        for f in main:
            for line in open(f, errors="replace"):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("type") == "assistant" and r.get("timestamp") and (r.get("message") or {}).get("stop_reason") == "end_turn":
                    ends.append(_wall(r["timestamp"]))
        src = "the root session's last end_turn (Claude Code)"
    if not ends:
        return {"report_latency_s": None, "work_done_s": (done - a0) * speed if a0 else None,
                "why": f"never reported: no turn ended in the transcript; run {st.get('status')}: {st.get('reason') or ''}"[:300]}
    rep, last_turn = max(ends), max(ends)
    if pick_rep and not cx:
        # an mls-kv answer: the last turn that states a pick is the report, and later status turns are not (pick_report)
        rep, src = pick_rep[0], ("the root session's last end_turn that sets out a G4n answer (g4n_report)" if isinstance(pick_rep[2], list)
                                 else "the root session's last end_turn that states a pick (pick_report)")
    if rep < done and st.get("status") not in ("completed", None) and not (pick_rep and not cx):
        # the last turn ended while work ran and the run was then stopped or killed: no report came after the work
        #, so there is none to time
        return {"report_latency_s": None, "work_done_s": (done - a0) * speed if a0 else None,
                "last_turn_end_s": (rep - a0) * speed if a0 else None,
                "why": f"no report after the work: the last turn ended before it, and the run was {st.get('status')}: {st.get('reason') or ''}"[:300]}
    return {"report_latency_s": (rep - done) * speed, "report_s": (rep - a0) * speed if a0 else None,
            "work_done_s": (done - a0) * speed if a0 else None, "reported_before_done": rep < done, "source": src,
            "last_turn_end_s": (last_turn - a0) * speed if a0 and last_turn != rep else None}

def report_latency_split(rl, hs):
    """What the time from the last needed job's end to the answer held (the analysis 2026-11-03 16:16 MST, K, whose bar is
    time to the answer): unneeded_running_s, from the last needed job's end (work_done_s) to the earlier of the last job's end and
    the answer, when GPU work nobody owed was still running; idle_before_answer_s, from the later of the last job's end and the last
    needed job's to the answer, floored at 0, when nothing ran. The last job's end is hindsight's actual_s (every completed launch,
    unneeded ones included, from agent start, as report_s is). Their sum is report_latency_s when the answer came after the work;
    an answer before the last job ended reads idle 0. Labels; None without a report, a work_done_s or a hindsight actual_s."""
    w, r, last = rl.get("work_done_s"), rl.get("report_s"), (hs or {}).get("actual_s")
    if w is None or r is None or last is None or rl.get("report_latency_s") is None:
        return {"unneeded_running_s": None, "idle_before_answer_s": None, "last_job_end_s": last}
    return {"unneeded_running_s": max(0.0, min(last, r) - w), "idle_before_answer_s": max(0.0, r - max(last, w)),
            "last_job_end_s": last, "answered_before_last_job_end": r < last}

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    d = sys.argv[1]
    # a trimmed archive copy (runner/campaign/archive_run.sh, /tmp/c1r/archive-run.sh before 2026-10-29: occupancy.jsonl gzipped, the ledger and box logs on the E2 only) scores
    # unchecked with every headline moved, over the score that was made from the whole run (2026-10-25 00:2xZ: seven
    # G-goalq-* runs, valid -> unchecked; restored from the archive). Refuse it: restore the run first
    if os.path.exists(os.path.join(d, "ARCHIVED.json")):
        sys.exit(f"{d}: a trimmed archive copy; its score.json stands. Restore it to re-score: "
                 + (json.load(open(os.path.join(d, "ARCHIVED.json"))).get("restore") or "bash " + os.path.join(RUNNER, "campaign", "restore_run.sh") + " <run_id>").replace("/tmp/c1r/restore-run.sh", os.path.join(RUNNER, "campaign", "restore_run.sh")))
    live = ["--live"] if "--live" in sys.argv else []
    text, js = [], {}
    # a Codex run: its rollouts are written first as the session records every scorer
    # reads (codex_rollout.py), so the scorers below run on them unchanged
    _rj = json.load(open(os.path.join(d, "run.json"))) if os.path.exists(os.path.join(d, "run.json")) else {}
    _harness = (_rj.get("inputs") or {}).get("harness") or _rj.get("harness")
    sys.path.insert(0, HERE)
    try:
        import codex_rollout
        js["codex"] = codex_rollout.normalise(d) if (_harness == "codex" or codex_rollout.sessions_dir(d)) else None
    except Exception as ex:
        js["codex"] = {"error": repr(ex)[:300]}
    # Grok Build and OpenCode (the analysis 2026-10-26 20:09Z; experiments/harnesses/): the same, each harness's
    # own transcript written as session records first (grok_updates.py, opencode_db.py; known answers test_harnesses.py)
    for _mod, _names, _key, _found in (("grok_updates", ("grok", "grok-build"), "grok", "sessions_dir"),
                                       ("opencode_db", ("opencode",), "opencode", "db_path")):
        try:
            _m = __import__(_mod)
            js[_key] = _m.normalise(d) if (_harness in _names or getattr(_m, _found)(d)) else None
        except Exception as ex:
            js[_key] = {"error": repr(ex)[:300]}
    # their side calls, for served_model.py (which excuses those rows only) and the call counts
    _side = [x for k in ("grok", "opencode") if isinstance(js.get(k), dict) for x in (js[k].get("side_calls") or [])]
    if _side:
        json.dump(_side, open(os.path.join(d, "side_calls.json"), "w"), indent=1)
    for script, key in (("score_run.py", "score"), ("idle.py", "idle"), ("overshoot.py", "overshoot"),
                        ("served_model.py", "served_model")):
        extra = live if script == "idle.py" else []
        p = run(script, d, *extra)
        text.append(p.stdout + (f"[{script} stderr]\n{p.stderr}" if p.stderr else ""))
        q = run(script, d, "--json", *extra)
        try:
            js[key] = json.loads(q.stdout)
        except json.JSONDecodeError:
            js[key] = {"error": (q.stderr or q.stdout)[-2000:]}
    st = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    reasons, status = [], "valid"
    sm = js.get("served_model") or {}
    if sm.get("verdict") == "void":
        status = "void"; reasons.append("served model: " + sm.get("reason", ""))
    if st.get("status") in ("crashed", "harness-error"):
        if status == "valid":
            status = "review"
        reasons.append(f"run status {st.get('status')}: {st.get('reason') or ''}"[:400])
    if status == "valid" and sm.get("verdict") == "unchecked" and sm.get("model"):
        status = "unchecked"; reasons.append("served model not checked: " + sm.get("reason", ""))
    if any("error" in (js.get(k) or {}) and len(js.get(k)) == 1 for k in ("score", "idle", "overshoot")):
        if status == "valid":
            status = "review"
        reasons.append("a scorer failed; see score.txt")
    # a rank that ties to no task job is never counted silently: its GPU work cannot complete
    # a task job, so it reads as work remaining and stretches the window to the horizon (P-pilotp3-haiku-L0 scored
    # 79% whole-box idle that way before the scorer learnt its naming; 3.5% after). A person decides.
    untied = [u for u in ((js.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("untied_ranks") or []
              if u.get("gpu_wall_s", 0) > 0]
    # an untied rank analysis has read and ruled legitimate (run_stop.json analysis_labels.ruled_valid_untied: the ranks, with the
    # label it chose; the analysis 2026-11-04 17:02 MST, G-chs8-opus-L0-r2's rank 43, an uninterrupted seed-6 run to step
    # 875 the agent chose to check its resumed chain, verified_before_report): listed and explained by that ruling, the run valid.
    # As ruled_valid_defects does for record-not-listed-for-job; any untied rank the ruling does not name still goes to review
    try:
        _rvu = ((json.load(open(os.path.join(d, "run_stop.json"))).get("analysis_labels") or {}).get("ruled_valid_untied") or {}) \
            if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    except (OSError, ValueError):
        _rvu = {}
    _rvu_ranks = set(_rvu.get("ranks") or [])
    _ruled_untied = [u for u in untied if u.get("job") in _rvu_ranks]
    if _ruled_untied:
        js["untied_ruled_valid"] = {"ranks": [u["job"] for u in _ruled_untied],
                                    "gpu_wall_s": round(sum(u["gpu_wall_s"] for u in _ruled_untied), 1),
                                    "explained_by": f"{_rvu.get('label')} (ruled valid: {str(_rvu.get('by') or '')[:120]})"}
        untied = [u for u in untied if u.get("job") not in _rvu_ranks]
    if untied:
        if status == "valid":
            status = "review"
        reasons.append(f"{len(untied)} rank(s) with {sum(u['gpu_wall_s'] for u in untied):.0f} GPU-s of work tie to no "
                       f"task job: " + ", ".join(f"rank {u['job']} ({u['argv'][:80]})" for u in untied)[:600])
    # gpu-mock replay defect "shared-launcher-identity": the daemon keys a launch by
    # ppid:WORLD_SIZE:MASTER_PORT:TORCHELASTIC_RUN_ID, so plain-python launches backgrounded from one shell that then
    # exits are all reparented to pid 1, share a key, and are served the first launch's draw (P-pilotp3-haiku-L0:
    # three seeds, one record). The fingerprint: two daemon jobs whose lookups give the same job_key and rank. Their
    # timing is about right and their served values are not, so the run goes to review, labelled.
    keys = {}
    occ = os.path.join(d, "occupancy.jsonl")
    for line in (open(occ) if os.path.exists(occ) else []):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("kind") == "lookup" and e.get("job_key"):
            keys.setdefault((e["job_key"], e.get("rank")), set()).add(e.get("job"))
    shared = {f"{k}/rank{r}": sorted(jobs) for (k, r), jobs in keys.items() if len(jobs) > 1}
    js["replay_defects"] = [{"defect": "shared-launcher-identity", "jobs_by_key": shared,
                             "pointer": "gpu-tasks' message to the runner, 2026-10-20 ~21:41Z; gpu-mock holds the fix"}] if shared else []
    if shared:
        if status == "valid":
            status = "review"
        reasons.append("replay_defect shared-launcher-identity: daemon jobs " +
                       "; ".join(f"{v} share job_key {k}" for k, v in shared.items())[:400] +
                       " and were served one draw (timing about right, values not)")
    # the general form: a tied rank served a record its task job does not list (an exact seed served another seed's
    # record, whatever the cause), from idle.py's work_remaining
    notlisted = ((js.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("records_not_listed") or []
    # a rank whose unlisted record is gpu-mock's seeded-recycle draw (a spent pool refilled from another job: recycled or
    # sibling, with the served seed differing from the launch's) is the labelled defect, not a review: the analysis'
    # ruling (2026-10-23 00:30Z) keeps such runs valid with seed_recycled_from_other_job
    try:
        import launches as _la3
        _sr3 = _la3.seed_recycles(d) or {}
        # a spent open pool's recycle is explained the same way: labelled, the run stays valid
        _recyc = {i["rank"] for i in (_sr3.get("items") or []) + (_sr3.get("spent_pool_items") or [])}
    except Exception:
        _recyc = set()
    explained = [u for u in notlisted if u.get("job") in _recyc]
    notlisted_rest = [u for u in notlisted if u.get("job") not in _recyc]
    if explained:
        js["replay_defects"].append({"defect": "record-not-listed-for-job", "ranks": explained,
                                     "explained_by": "seed_recycled_from_other_job (labelled, the run stays valid)"})
    notlisted = notlisted_rest
    # a smoke test the agent killed: a rank whose unlisted draw the agent
    # stopped by its own kill, before its record's first logged validation step, whose task job was later completed on a
    # listed record. The test's draw never reached a value the agent could read, and the job's result came from its listed
    # record, so it is explained, labelled smoke_test_unlisted_draw_killed; an unlisted draw that ran its length, or that
    # ended without the agent's kill, still flags
    try:
        _wr4 = (js.get("idle") or {}).get("work_remaining_at_window_end") or {}
        _killed = {u.get("job"): u for u in (_wr4.get("agent_caused_crashes") or []) if "agent's own kill" in str(u.get("why") or "")}
        _done_recs = _wr4.get("done_launch_records") or {}
        _smoke = []
        for u in notlisted:
            k = _killed.get(u.get("job"))
            if not k:
                continue
            first_val = _first_val_step(u.get("record_id"))
            done_on = set(_done_recs.get(u.get("task_job")) or [])
            if (first_val is not None and (k.get("reached_step") if k.get("reached_step") is not None else -1) < first_val
                    and done_on and done_on <= set(u.get("listed") or [])):
                _smoke.append(dict(u, reached_step=k.get("reached_step"), first_val_step=first_val))
        if _smoke:
            js["replay_defects"].append({"defect": "record-not-listed-for-job", "ranks": _smoke,
                                         "explained_by": "smoke_test_unlisted_draw_killed (the agent killed it before its first validation; "
                                                         "the job completed on a listed record; labelled, the run stays valid)"})
            js["_smoke_test_unlisted_draw_killed"] = [x["job"] for x in _smoke]
            notlisted = [u for u in notlisted if u.get("job") not in {x["job"] for x in _smoke}]
    except Exception as ex:      # a label: never stops the score
        js["_smoke_test_unlisted_draw_killed"] = {"error": repr(ex)[:300]}
    # a defect analysis has ruled on (run_stop.json analysis_labels.ruled_valid_defects: the ranks it read and kept, with the
    # label it chose; the analysis 2026-10-24 22:28Z, G-g4na-sonnet-L0's duplicate_launch_seed_substituted): listed,
    # explained by that ruling, the run valid
    _rs_ruled = json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    _ruled = ((_rs_ruled.get("analysis_labels") or {}).get("ruled_valid_defects") or {})
    _ruled_ranks = set(_ruled.get("ranks") or [])
    _ruled_items = [u for u in notlisted if u.get("job") in _ruled_ranks]
    if _ruled_items:
        js["replay_defects"].append({"defect": "record-not-listed-for-job", "ranks": _ruled_items,
                                     "explained_by": f"{_ruled.get('label')} (ruled valid: {str(_ruled.get('by') or '')[:120]})"})
        notlisted = [u for u in notlisted if u.get("job") not in _ruled_ranks]
    if notlisted:
        js["replay_defects"].append({"defect": "record-not-listed-for-job", "ranks": notlisted})
        if status == "valid":
            status = "review"
        reasons.append("replay_defect record-not-listed-for-job: " + "; ".join(
            f"rank {u['job']} of {u['task_job']} served {u['record_id']} ({u['draw']})" for u in notlisted)[:400])
    # a job that did not complete (exited non-zero, or stopped before its record's final step) makes a scripted run
    # INVALID, since the plan cannot fail on an H100; and any run where an exact-match job stopped short with no
    # recorded failure and no kill by the agent is INVALID whatever the agent did, since the fake environment broke
    # it (the study, 2026-10-21: the fake device's isfinite crashed 33 of the 86 mng3 scripts at step 0, and the
    # scorer read the runs VALID)
    wr = (js.get("idle") or {}).get("work_remaining_at_window_end") or {}
    nc, crash = wr.get("not_completed") or [], wr.get("environment_crashes") or []
    rj0 = json.load(open(os.path.join(d, "run.json"))) if os.path.exists(os.path.join(d, "run.json")) else {}
    harness = (rj0.get("inputs") or {}).get("harness") or rj0.get("harness")
    if crash:
        if status != "void":
            status = "invalid"
        reasons.append(f"environment crash: {len(crash)} exact-match rank(s) stopped short with no recorded failure: " +
                       "; ".join(f"{u['task_job']} at step {u['reached_step']} of {u['final_step']}" for u in crash)[:400])
    uo = wr.get("unknown_origin_crashes") or []
    if uo:
        if status == "valid":
            status = "review"
        reasons.append(f"{len(uo)} crash(es) of unknown origin (an exception with no recorded frame in an unedited recorded "
                       "script): " + "; ".join(f"{u['task_job']}: {u.get('exception', '')[:60]}" for u in uo)[:400])
    hk = wr.get("killed_at_harness_exit") or []
    # the lifecycle's grace close after every owed job had ended (the analysis 2026-10-28 05:02Z, G-g4na-astracc-L0-r2:
    # soap-run3, a fourth soap run nobody owed, killed at teardown 1,800 s after the last owed end): the close took it, by
    # design, not the environment; labelled (killed_after_grace_close), the run stays valid
    import grace as _gr
    _gm = _gr.GRACE_RE.search(str(st.get("stdin_close_reason") or ""))
    if hk and _gm and _gm.group(1) == "closed_after_grace" and wr.get("remaining") == 0:
        js["killed_after_grace_close"] = [{k: u.get(k) for k in ("job", "task_job", "reached_step", "final_step")} for u in hk]
        hk = []
    if hk:
        if status == "valid":
            status = "review"
        reasons.append(f"harness exit killed {len(hk)} running job(s) (claude -p ended its wait for background tasks): " +
                       "; ".join(f"{u['task_job']} at step {u['reached_step']} of {u['final_step']}" for u in hk)[:400])
    if nc and harness == "scripted":
        if status != "void":
            status = "invalid"
        reasons.append(f"scripted run: {len(nc)} task job(s) did not complete: " +
                       "; ".join(f"{u['task_job']} ({u['why'][0]})" for u in nc)[:400])
    # Do the agent's scripts follow the setups? gpu-mock serves the recorded run whatever the script's bytes, so nothing
    # in the occupancy log shows it. A written script that does not stop at the reference's step
    # trains a different run than the one served, so the run goes to review; literals that differ and the similarity are
    # a label (a set comparison: a constant moved between parameters, or a logic change with no new literal, is not seen)
    pf = js["prep_fidelity"] = prep_fidelity(d)
    try:
        js["h4_prep"] = h4_prep_parallelism(d, json.load(open(os.path.join(d, "task.json"))) if os.path.exists(os.path.join(d, "task.json")) else {})
    except Exception as ex:      # a label: never stops the score
        js["h4_prep"] = {"error": repr(ex)[:300]}
    off = [r for r in (pf or {}).get("jobs") or [] if r.get("written") and r.get("stop_reference") and not r.get("stop_matches")]
    if off:
        if status == "valid":
            status = "review"
        reasons.append(f"prep-script-stop-differs: {len(off)} agent-written script(s) do not stop at the reference's step: " +
                       "; ".join(f"{r['job_id']} reference {r['stop_reference']} agent {r['step_constants_agent']}" for r in off)[:400])
    # the box scheduler's event log (S rungs, the scheduler 2026-10-21; sched_events.py): a rank on a GPU its
    # scheduler job was not given means CUDA_VISIBLE_DEVICES was not honoured, so its timing is not the schedule's
    try:
        import sched_events as sched_mod
        js["sched"] = sched_mod.analyse(d)
    except Exception as ex:      # a label: never stops the score
        js["sched"] = {"error": repr(ex)[:300]}
    if (js["sched"] or {}).get("gpu_mask_mismatch"):
        if status == "valid":
            status = "review"
        reasons.append("scheduler gpu_mask_mismatch: " + "; ".join(
            f"rank {m['rank']} of job {m['sched_job']} on GPUs {m['rank_gpus']}, given {m['start_gpus']}" for m in js["sched"]["gpu_mask_mismatch"])[:400])
    if (js["sched"] or {}).get("controller_errors"):
        if status == "valid":
            status = "review"
        reasons.append(f"scheduler controller_error x{js['sched']['controller_errors']}: {js['sched']['controller_error_first']}"[:400])
    # a run the campaign marked a testbed defect after it ran (the study's `edited` note on its line, copied by the
    # driver into run_stop.json as testbed_defect; 2026-10-22 23:27Z: gpu-mock's argument-order bug under X-scq-astra-S1
    # and X-halving-astra-S1, each re-run as -r1b): void, with the note, so it is never pooled
    # a launch that asked for an example past its record's n_examples_full (gpu-mock's failure mode per_example_overrun,
    # 2026-10-22 23:37Z: NeedleBench's 147-vs-441 records): the testbed's record was short, not the agent's job wrong;
    # the run goes to review with the ranks named, so it is not read as an agent failure
    _occ = os.path.join(d, "occupancy.jsonl")
    _ovr = []
    for _l in (open(_occ, errors="replace") if os.path.exists(_occ) else []):
        if '"per_example_overrun"' in _l:
            try:
                _e = json.loads(_l)
            except ValueError:
                continue
            if _e.get("kind") == "failure" and _e.get("mode") == "per_example_overrun":
                _ovr.append(_e.get("job"))
    js["_per_example_overrun_ranks"] = _ovr or None     # moved into labels once they exist (below)
    if _ovr:
        if status in ("valid", "unchecked"):
            status = "review"
        reasons.append(f"per_example_overrun on rank(s) {_ovr}: a launch asked for examples past its record's n_examples_full (the record's, not the agent's)")
    # a task-script launch the testbed matched to no record and paced slower than twice its closest record (experiments-
    # analysis, 2026-10-23 04:27Z: a whole run on a broken clock reads valid otherwise): review, with the ranks named
    try:
        _task_sm = json.load(open(os.path.join(d, "task.json"))) if os.path.exists(os.path.join(d, "task.json")) else {}
        js["slow_misses"] = slow_misses(d, _task_sm) if harness not in (None, "scripted") else None
    except Exception as ex:
        js["slow_misses"] = {"error": repr(ex)[:300]}
    if (js["slow_misses"] or {}).get("slow"):
        if status in ("valid", "unchecked"):
            status = "review"
        _x = next(x for x in js["slow_misses"]["items"] if x["slow"])
        reasons.append(f"{js['slow_misses']['slow']} task-script launch(es) matched no record and ran slower than 2x the closest "
                       f"record's pace: rank {_x['rank']} at {_x['pace_s']} s/example vs {_x['record_pace_s']} "
                       f"(record lacks {_x['only_launch']}, launch lacks {_x['only_record']})")
    _rs2 = json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    # a seed recycle on a machine where gpu-mock's fix was declared at the run's start is a regression, not the known
    # defect: void, with an alert the driver logs at collect
    if _rs2.get("seeded_recycle_fixed"):
        try:
            import launches as _la2
            _sr = _la2.seed_recycles(d) or {}
        except Exception:
            _sr = {}
        if _sr.get("seed_recycled_from_other_job"):
            status = "void"
            reasons.append(f"REGRESSION: {_sr['seed_recycled_from_other_job']} seed recycle(s) on a machine where "
                           "fix:gpumock-seeded-recycle was declared at the run's start (gpu-mock's seeded-recycle rule should "
                           "no longer serve another job's record)")
            js["regression_alert"] = "seed_recycled_from_other_job"
    if _rs2.get("testbed_defect"):
        status = "void"
        reasons.append("testbed defect, re-run under a new id: " + str(_rs2["testbed_defect"])[:300])
    # a run we stopped for a reason outside it, ruled void by hand (run_stop analysis_labels; the study 2026-10-28 05:02Z,
    # G-kvgate-fable-L0-r2: stopped_to_protect_subscription, the subscription's usage window projected past its limit)
    for _hv, _hval in hand_voids(_rs2.get("analysis_labels") or {}):
        status = "void"
        reasons.append(f"{_hv} (hand label): " + str(_hval)[:300])
    # a scripted controller that launched nothing because it crashed (the study 2026-11-03 06:51 MST, W0-k2lp-reference: its
    # reference.py died on `import pyarrow` in 12 s, 0 of 38 jobs): a pipeline failure, never a measured outcome. "Nothing was
    # launched" is a real outcome only for an agent. A controller that launched work and then raised (W0-sfz-reference) is not this
    _cc = controller_crash(d, js.get("score") or {})
    if _cc:
        status = "invalid"
        reasons.append(_cc)
    js["validity"] = {"status": status, "reasons": reasons}
    # the run's labels, copied from run.json so every score carries the conditions it is read under: model, nudge
    # (level, version, sentence sha; isolation's G6), the brief's sha, the horizon used, and the machine (G5)
    rj = json.load(open(os.path.join(d, "run.json"))) if os.path.exists(os.path.join(d, "run.json")) else {}
    inp, host = rj.get("inputs") or {}, rj.get("host") or {}
    js["labels"] = {"cost_tag": rj.get("cost_tag"), "task": inp.get("task"), "harness": inp.get("harness"),
                    "model": inp.get("model"), "policy": (rj.get("scripted") or {}).get("policy"), "speed": inp.get("speed"),
                    "nudge": inp.get("nudge"), "brief_sha256": inp.get("brief_sha256") or rj.get("brief_sha256"),
                    "task_brief_sha256": inp.get("task_brief_sha256"),
                    "horizon_s": rj.get("horizon_s"), "horizon_how": rj.get("horizon_how"),
                    "machine_type": host.get("machine_type"), "instance": host.get("instance"),
                    "cpu_platform": host.get("cpu_platform"), "zone": host.get("zone"),
                    "run_one_sha256": (rj.get("code") or {}).get("run_one_sha256"),
                    "cgroup_sampler_sha256": (rj.get("code") or {}).get("cgroup_sampler_sha256")}
    # the harness: does a background task's or subagent's completion wake the agent? Claude Code's notification starts a
    # new turn; Codex never wakes it (the Codex driver's lifecycle note, experiments/codex/notes/lifecycle-2026-10-21.md)
    js["labels"]["harness_wakes_on_background"] = {"claude-code": True, "codex": False, "grok": True, "grok-build": True,
                                                   "opencode": False}.get(harness)
    # the reasoning effort the server reports per turn (the ledger's effort column), and Codex's own turn_context; a
    # Codex run is set to the product default, medium, and any other value is flagged
    led_p = os.path.join(d, "ledger.jsonl")
    led_eff = sorted({r.get("effort") for r in (json.loads(l) for l in open(led_p) if l.strip()) if r.get("effort")}) \
        if os.path.exists(led_p) else []
    cx = js.get("codex") if isinstance(js.get("codex"), dict) and "threads" in js.get("codex") else None
    js["labels"]["reasoning_effort"] = {"ledger": led_eff, "codex_turn_context": (cx or {}).get("efforts")}
    js["labels"]["effort_not_medium"] = (any(e != "medium" for e in led_eff + list((cx or {}).get("efforts") or []))
                                         if harness == "codex" else None)
    # a launch with a plain & in a Codex tool call dies when the call ends (product behaviour, an agent failure mode)
    js["labels"]["launches_killed_at_tool_end"] = None if cx is None else len(cx["ampersand_launches"])
    # the model route's measured round trip from the machine, before and after the run (the driver writes it; a
    # per-machine latency offset is then visible beside the turn latencies, the study 2026-10-20)
    rtp = os.path.join(d, "gate_rtt.json")
    js["labels"]["gate_rtt"] = json.load(open(rtp)) if os.path.exists(rtp) else None
    # The environment floor F for the run's own machine type (gpu-tasks' environment_floor.by_machine_type, keyed by
    # run.json host.machine_type), and the gap net of it beside the raw gap (design/IDEA.md: every gap is reported raw
    # and net of the floor). None when the task has no entry for this machine type. F was measured at certain caps;
    # if the run's caps differ, the floor is reported but marked as not matching.
    task = json.load(open(os.path.join(d, "task.json"))) if os.path.exists(os.path.join(d, "task.json")) else {}
    fl = ((task.get("environment_floor") or {}).get("by_machine_type") or {}).get(host.get("machine_type") or "")
    sc = js.get("score") or {}
    raw = sc.get("gap_to_lower_bound_from_agent_start_s")
    if fl:
        mc, rc_ = (fl.get("measured") or {}).get("caps") or {}, rj.get("caps") or {}
        same = all(str(mc.get(k)) == str(rc_.get(k)) for k in ("cpus", "memory") if k in mc)
        js["floor"] = {"machine_type": host.get("machine_type"), "F_s": fl.get("value"), "caps_match": same,
                       "measured_run": (fl.get("measured") or {}).get("run_dir"),
                       "gap_raw_s": raw, "gap_net_s": (raw - fl["value"]) if raw is not None and fl.get("value") is not None else None}
    else:
        js["floor"] = {"machine_type": host.get("machine_type"), "F_s": None, "gap_raw_s": raw, "gap_net_s": None,
                       "why": "the task has no environment floor measured for this machine type"}
    # the campaign phase, from the run id in the cost tag (P006-<run id>-<stamp>): wave Q (W0-*-q-* and Q-* lines,
    # about 8-min jobs) finds failures early, wave A (the long jobs) confirms them
    m = re.match(r"P006-(.+)-\d{8}T\d{6}Z$", rj.get("cost_tag") or "")
    rid = m.group(1) if m else None
    phase = ("find" if rid and (rid.startswith("Q-") or re.match(r"W0-.*-q-", rid)) else
             "confirm" if rid and rid.startswith("A-") else None)
    js["labels"]["run_id"], js["labels"]["phase"] = rid, phase
    # did the agent see the GPUs are fake (tells.py): a label, never a void; the image id
    # is on every run, so runs on scrubbed and unscrubbed images can be told apart
    sys.path.insert(0, HERE)
    import tells
    try:
        js["tells"] = tells.scan(d)
    except Exception as ex:                              # a label must not stop the score
        js["tells"] = {"saw_tell": None, "error": repr(ex)[:300]}
    # the host-CPU-idle pattern: a paced job asleep at no CPU, seen, killed and
    # relaunched (host_cpu.py); labels and the killed-then-relaunched GPU-seconds, reported beside the bar figure, never in it
    try:
        import host_cpu
        js["host_cpu"] = host_cpu.scan(d)
    except Exception as ex:                              # a label must not stop the score
        js["host_cpu"] = {"saw_host_cpu_idle": None, "error": repr(ex)[:300]}
    # the work-done grace: the lifecycle hash, the grace's
    # outcome from stdin_close_reason, and task-job launches after the last owed job's end (grace.py); labels, never a verdict
    try:
        import grace
        js["grace"] = grace.scan(d, (((js.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("all_done_wall")))
    except Exception as ex:
        js["grace"] = {"grace_outcome": None, "error": repr(ex)[:300]}
    js["labels"]["saw_tell"] = js["tells"].get("saw_tell")
    js["labels"]["saw_own_name"] = js["tells"].get("saw_own_name")
    js["labels"]["harness_attachments"] = js["tells"].get("harness_attachments")
    # gpu-mock's own files showing:
    # the agent inspected /usr/local/nvidia/{python,pylib,etc}, or a traceback frame there reached a tool result or its
    # workspace; and the structseq-as-tuple defect (fix:gpumock-topk-tuple). Labels; the driver logs the alert at collect
    js["labels"]["inspected_box_tooling"] = js["tells"].get("inspected_box_tooling")
    js["labels"]["probed_runtime"] = js["tells"].get("probed_runtime")
    js["labels"]["traceback_through_patch"] = js["tells"].get("traceback_through_patch")
    js["labels"]["structseq_tuple_error"] = js["tells"].get("structseq_tuple_error")
    if js["tells"].get("traceback_through_patch") or js["tells"].get("structseq_tuple_error"):
        js["mock_path_alert"] = {"traceback_through_patch": js["tells"].get("traceback_through_patch"),
                                 "structseq_tuple_error": bool(js["tells"].get("structseq_tuple_error")),
                                 "first": ((js["tells"].get("patch_frames_in_tool_results") or js["tells"].get("patch_frames_in_workspace")
                                            or js["tells"].get("structseq_tuple_error") or [{}])[0])}
    # how the session ended (isolation's lifecycle run_one.py, 2026-10-21): stdin a fifo kept open while background
    # work runs, closed by runner/harness/lifecycle.py's rule or at teardown; on older images stdin was the brief file
    stj = json.load(open(os.path.join(d, "status.json"))) if os.path.exists(os.path.join(d, "status.json")) else {}
    cc = rj.get("claude_code") or {}
    # harness flags and whether the agent had the subagent tool: the
    # flags run_one.py passed to claude, and the tools in Claude Code's own init event, which is what the agent got
    init_tools = None
    sp = os.path.join(d, "transcript", "stream.jsonl")
    for line in (open(sp, errors="replace") if os.path.exists(sp) else []):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "system" and r.get("subtype") == "init":
            init_tools = r.get("tools") or []
            break
    js["labels"]["harness_flags"] = cc.get("extra_flags") or []
    js["labels"]["subagents_available"] = (None if init_tools is None else
                                           any(t in ("Task", "Agent") for t in init_tools))
    js["labels"]["session"] = {"stdin": cc.get("stdin") or ("file" if cc else None),
                               "lifecycle_sha256": cc.get("lifecycle_sha256"), "lifecycle_grace_s": cc.get("lifecycle_grace_s"),
                               "stdin_closed_wall": stj.get("stdin_closed_wall"), "stdin_close_reason": stj.get("stdin_close_reason"),
                               "harness_results": stj.get("harness_results"), "box_dir_host": rj.get("box_dir_host")}
    # launches that died before reaching a GPU (launches.py): mechanism evidence, never a score
    import launches
    try:
        js["launches"] = launches.scan(d)
    except Exception as ex:
        js["launches"] = {"launches_failed_before_gpu": None, "error": repr(ex)[:300]}
    try:
        js["rerun_identical"] = launches.rerun_identical(d)
        try:
            js["seed_recycles"] = launches.seed_recycles(d)
        except Exception as ex:      # a label: never stops the score
            js["seed_recycles"] = {"error": repr(ex)[:300]}
    except Exception as ex:      # a label: never stops the score
        js["rerun_identical"] = {"rerun_identical_values": None, "error": repr(ex)[:300]}
    ri = js["rerun_identical"].get("rerun_identical_values")
    js["labels"]["rerun_identical_values"] = None if ri is None else len(ri)
    js["labels"]["rerun_identical_remarked"] = js["rerun_identical"].get("agent_remarked") if ri else None
    js["labels"]["rerun_identical_script_changed"] = None if ri is None else sum(1 for h in ri if h.get("script_changed"))
    # the hindsight bound: the agent's own completed jobs, each released
    # when its inputs were ready or the results it was chosen from were read; for open-ended packages with no job list
    try:
        import hindsight
        js["hindsight"] = hindsight.analyse(d)
        if js["hindsight"].get("bound_s"):
            # the run's GPU-lane figure beside timeline.html
            open(os.path.join(d, "hindsight.svg"), "w").write(hindsight.svg(js["hindsight"], js.get("run_id") or os.path.basename(os.path.normpath(d))))
    except Exception as ex:      # a label: never stops the score
        js["hindsight"] = {"bound_s": None, "error": repr(ex)[:300]}
    try:
        js["decision_picks"] = launches.decision_picks(d)
    except Exception as ex:      # a label: never stops the score
        js["decision_picks"] = {"error": repr(ex)[:300]}
    dp_ = js["decision_picks"] if isinstance(js["decision_picks"], list) else None
    js["labels"]["decision_picks_not_winner"] = None if dp_ is None else sum(1 for x in dp_ if x["is_winner"] is False)
    # the box kept up after the harness ended, by run_one's after-harness rule (an agent's process left running: a
    # self-matching pgrep poll), until teardown: the session ended on time, the machine did not (the study, 2026-10-22,
    # G-passbar-haiku-L0-v2: harness exit 12 s after the final answer, box down 24 min later by a .stop)
    he, td, a0_ = st.get("harness_end_wall"), st.get("teardown_wall"), st.get("agent_start_wall")
    work_end = ((js.get("idle") or {}).get("work_remaining_at_window_end") or {}).get("all_done_wall")
    js["labels"]["box_held_after_harness"] = (None if not (he and td) or td - he < 60 else
        {"held_s": td - he, "harness_end_s": (he - a0_) if a0_ else None, "teardown_s": (td - a0_) if a0_ else None,
         "teardown_reason": st.get("teardown_reason"), "run_status": st.get("status"),
         "work_done_before_harness_end": bool(work_end and work_end <= he)})
    # the agent's last tool call never returned and the run was then stopped or killed, with the GPUs idle all that
    # time: a hung foreground command (the study, 2026-10-22, G-goalq-sonnet-L0-v2: `grep -c "" $f` with f empty read
    # the tool's stdin for 21 min; the Bash timeout run_one sets is 10 h). The agent's own error, labelled
    js["labels"]["hung_foreground_command"] = hung_foreground(d, st, js.get("idle") or {})
    js["labels"]["ended_with_wakeup_pending"] = wakeup_pending(d, st)
    js["labels"]["per_example_overrun_ranks"] = js.pop("_per_example_overrun_ranks", None)
    js["labels"]["smoke_test_unlisted_draw_killed"] = js.pop("_smoke_test_unlisted_draw_killed", None)
    # a Codex root turn the provider ended (capacity, 429s, 502s): the session closed on a failed turn, not on the agent's answer
    try:
        js["labels"]["codex_turn_failed"] = codex_turn_failed(d, st)
    except Exception as ex:      # a label: never stops the score
        js["labels"]["codex_turn_failed"] = {"error": repr(ex)[:300]}
    try:
        js["labels"]["no_gpu_job_by_horizon"] = no_gpu_job_by_horizon(
            d, st, json.load(open(os.path.join(d, "task.json"))) if os.path.exists(os.path.join(d, "task.json")) else {})
        # analysis's hand reading wins (2026-10-25 18:22Z: G-h1-haiku-L0 and -L3 launched their own prepared copies of the setup
        # scripts, which the path_bound records could not serve; the rule reads only the task's own scripts, so it read "never
        # launched"): run_stop.json analysis_labels.modelled_path_only sets that side and clears the other
        _al_ng = ((json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {})
                  .get("analysis_labels") or {})
        if _al_ng.get("modelled_path_only") and js["labels"]["no_gpu_job_by_horizon"]:
            js["labels"]["no_gpu_job_by_horizon"] = dict(js["labels"]["no_gpu_job_by_horizon"], no_gpu_job_by_horizon=False,
                                                         modelled_path_only=True, how="hand label (run_stop analysis_labels)")
    except Exception as ex:      # a label: never stops the score
        js["labels"]["no_gpu_job_by_horizon"] = {"error": repr(ex)[:300]}
    # labels the analysis rules on a run after it is scored (carried in run_stop.json's analysis_labels)
    _rs3 = json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    js["labels"]["analysis_labels"] = _rs3.get("analysis_labels")
    try:
        js["labels"]["tell_class"] = tell_class(js["tells"], js["rerun_identical"], _rs3.get("analysis_labels"))
    except Exception as ex:      # a label: never stops the score
        js["labels"]["tell_class"] = {"tell_class": None, "error": repr(ex)[:300]}
    # host work on the same machine during the run (the driver copies it from the claim into run_stop.json): a
    # disclosure for turn latency and gate_rtt in that window, never a validity change
    _rs = json.load(open(os.path.join(d, "run_stop.json"))) if os.path.exists(os.path.join(d, "run_stop.json")) else {}
    js["labels"]["concurrent_host_load"] = _rs.get("concurrent_host_load")
    try:
        js["subsets"] = subset_launches(d)
    except Exception as ex:      # a label: never stops the score
        js["subsets"] = {"error": repr(ex)[:300]}
    try:
        js["module_launches"] = module_launches(d)
    except Exception as ex:      # a label: never stops the score
        js["module_launches"] = {"error": repr(ex)[:300]}
    _pick_rep = None
    # mls-kv-report-h: mls-kv-report's 40 jobs and reference plan, its brief without the method
    # and workload list, so it reads the same answer, report time and coverage
    if (task.get("task_id") or inp.get("task")) in MLSKV_TASKS and harness not in (None, "scripted") and not cx:
        try:
            _pick_rep = pick_report(d, task)
        except Exception:        # a label: never stops the score; the last turn stays the report
            _pick_rep = None
    if (task.get("task_id") or inp.get("task")) in MLSKV_TASKS and harness not in (None, "scripted"):
        try:
            js["mlskv_process"] = mlskv_process(d, task, js, _pick_rep)
        except Exception as ex:  # a label: never stops the score
            js["mlskv_process"] = {"error": repr(ex)[:300]}
    # a G4n answer's own turn as the report, when the last turn states no set:
    # report_latency and the live-at-report list then time the answer, not a later acknowledgement; a hand label does not move it
    if (task.get("task_id") or inp.get("task")) in G4N_TASKS and harness == "claude-code" and not cx and _pick_rep is None:
        try:
            if g4n_adopted(root_final_text(d))[0] is None:
                _g4r = g4n_report(d)
                if _g4r:
                    _pick_rep = (_g4r[0], _g4r[1], _g4r[2], _g4r[3])
        except Exception:        # a label: never stops the score; the last turn stays the report
            pass
    if (task.get("task_id") or inp.get("task")) in G4N_TASKS:
        try:
            js["g4n_answer"] = g4n_answer(d, task, js)
        except Exception as ex:  # a label: never stops the score
            js["g4n_answer"] = {"error": repr(ex)[:300]}
    try:
        js["wide"] = wide_launches(d, task)
        js["k_answer"] = k_answer(d, task, js, harness)
        js["prep_observed"] = prep_observed(d, task)
        js["filter_yield_answer"] = filter_yield_answer(d, task, js["prep_observed"], harness)
        if isinstance(js["filter_yield_answer"], dict) and "correct" in js["filter_yield_answer"]:
            _fy = js["filter_yield_answer"]
            _rem_fy = ((js.get("idle") or {}).get("work_remaining_at_window_end") or {})
            _fy["gpu_jobs_done"] = _rem_fy.get("remaining") == 0
            _fy["supported"] = bool(_fy["correct"] and _fy["all_preparations_done"] and _fy["gpu_jobs_done"]
                                    and not _fy["retention_missing"] and not _fy["means_missing"])
        js["silent_failure"] = silent_failure(d, task)
        js["tabular_adoption"] = tabular_adoption(d, task)
        js["tabular_table"] = tabular_table(d, task)
        js["sf_checkpoint"] = sf_checkpoint_artifact(d, task)
        if task.get("checkpoint_design"):
            js["checkpoint_loads"] = loaded_checkpoint_weights(d)
            js["labels"]["loaded_checkpoint_weights"] = (js["checkpoint_loads"] or {}).get("loaded_checkpoint_weights")
        js["precision"] = precision_comparison(d, task, cx=js.get("codex") if isinstance(js.get("codex"), dict) and "threads" in js.get("codex") else None)
        js["abs_path"] = abs_path_misses(d, task)
    except Exception as ex:      # a label: never stops the score
        js["wide"] = {"error": repr(ex)[:300]}
    try:
        js["off_plan"] = off_plan_launches(d, task)
    except Exception as ex:      # a label: never stops the score
        js["off_plan"] = {"error": repr(ex)[:300]}
    if harness not in (None, "scripted"):
        try:
            js["polling"] = poll_block(d)
        except Exception as ex:  # a label: never stops the score
            js["polling"] = {"error": repr(ex)[:300]}
    try:
        js["example_host_gaps"] = example_host_gaps(d)
    except Exception as ex:      # a label: never stops the score
        js["example_host_gaps"] = {"error": repr(ex)[:300]}
    try:
        js["near_misses"] = near_misses(d)
    except Exception as ex:      # a label: never stops the score
        js["near_misses"] = {"error": repr(ex)[:300]}
    try:
        js["trial_loops"] = trial_loop_launches(d)
    except Exception as ex:      # a label: never stops the score
        js["trial_loops"] = {"error": repr(ex)[:300]}
    try:
        js["replay_exhausted_in_process"] = replay_exhausted_in_process(d)
    except Exception as ex:      # a label: never stops the score
        js["replay_exhausted_in_process"] = {"error": repr(ex)[:300]}
    try:
        js["bash_call_outcomes"] = bash_call_outcomes(d)
    except Exception as ex:      # a label: never stops the score
        js["bash_call_outcomes"] = {"error": repr(ex)[:300]}
    try:
        js["labels"]["self_matching_wait"] = self_matching_wait(d, st)
    except Exception as ex:      # a label: never stops the score
        js["labels"]["self_matching_wait"] = {"error": repr(ex)[:300]}
    try:
        js["sleep_lengths"] = sleep_lengths(d)
    except Exception as ex:      # a label: never stops the score
        js["sleep_lengths"] = {"error": repr(ex)[:300]}
    ho =(js.get("idle") or {}).get("held_open_by_background_task")
    js["labels"]["held_open_by_background_task"] = None if not ho else {
        "held_s": ho["held_s"], "last_turn_s": ho["last_turn_s"], "commands": [x["command"] for x in ho["tasks"]]}
    rr = js["rerun_identical"].get("reruns") or []
    js["labels"]["reruns"] = len(rr)
    js["labels"]["reruns_sibling"] = sum(1 for x in rr if "sibling" in x["draw"])
    js["labels"]["reruns_fingerprint_differs"] = sum(1 for x in rr if x["fingerprint_differs"])
    lf = js["launches"].get("launches_failed_before_gpu")
    js["labels"]["launches_failed_before_gpu"] = None if lf is None else len(lf)
    try:
        js["pass_bar"] = pass_bar(d, task, js.get("idle") or {})
        # without the recycled launches, and whether a verdict flips
        _srj = js.get("seed_recycles") or {}
        _rr = [i["rank"] for i in (_srj.get("items") or []) + (_srj.get("spent_pool_items") or [])]
        if _rr and (js["pass_bar"] or {}).get("by_candidate"):
            js["pass_bar_without_recycles"] = pass_bar_excluding(d, task, js.get("idle") or {}, _rr)
    except Exception as ex:      # a label: never stops the score
        js["pass_bar"] = {"error": repr(ex)[:300]}
    try:
        js["stopped_by"] = stopped_by(d, st)
    except Exception as ex:      # a label: never stops the score
        js["stopped_by"] = {"stopped_by": None, "error": repr(ex)[:300]}
    js["labels"]["stopped_by"] = js["stopped_by"].get("stopped_by")
    try:
        js["report_latency"] = report_latency(d, js.get("idle") or {}, st, cx, harness, _pick_rep)
    except Exception as ex:      # a label: never stops the score
        js["report_latency"] = {"report_latency_s": None, "why": "error: " + repr(ex)[:300]}
    try:
        js["report_latency"].update(report_latency_split(js["report_latency"], js.get("hindsight") or {}))
    except Exception as ex:      # a label: never stops the score
        js["report_latency"]["split_error"] = repr(ex)[:300]
    # mls-kv-gate: when each rejected method's rejection was in hand, and what its other launches did after it (the analysis
    # 2026-10-24 20:47Z: long_run_before_cheap_check, killed_known_useless_job_late, cancelled_rejected_setup)
    if (task.get("task_id") or inp.get("task")) in GATE_TASKS and harness not in (None, "scripted"):
        try:
            _rl = js.get("report_latency") or {}
            _sp = float(((js.get("idle") or {}).get("window") or {}).get("speed") or 1.0)
            _rw = (st.get("agent_start_wall") + _rl["report_s"] / _sp) if (_rl.get("report_s") is not None and st.get("agent_start_wall")) else None
            js["gate_timing"] = gate_rejected_timing(d, task, js, _rw)
            _gt = js["gate_timing"] or {}
            js["labels"]["long_run_before_cheap_check"] = len(_gt.get("long_run_before_cheap_check") or [])
            js["labels"]["killed_known_useless_job_late"] = [{"task_job": i["task_job"], "stopped_after_rejection_s": i["stopped_after_rejection_s"]}
                                                             for i in _gt.get("killed_known_useless_job_late") or []] or None
            js["labels"]["cancelled_rejected_setup"] = len(_gt.get("cancelled_rejected_setup") or []) or None
            js["labels"]["long_run_after_rejection"] = len(_gt.get("long_run_after_rejection") or [])
        except Exception as ex:  # a label: never stops the score
            js["gate_timing"] = {"error": repr(ex)[:300]}
    try:
        js["live_at_report"] = jobs_live_at_report(d, st, harness, _pick_rep) if not cx else None
    except Exception as ex:      # a label: never stops the score
        js["live_at_report"] = {"error": repr(ex)[:300]}
    # pkg-k2-latepole's disengaged-overlap rule: idle's overlap of whole-box
    # idle with gpu-mock-disengaged ranks, re-shared over agent start to the answer (report_s) when the run answered, the time the
    # bar reads; over 5% is disengaged_smoke_ran_real (void, simulator), at or under is disclosed beside the figure. A label here
    _dov = ((js.get("idle") or {}).get("disengaged_overlap"))
    if _dov:
        _rs_ = (js.get("report_latency") or {}).get("report_s") or (js.get("live_at_report") or {}).get("report_s")
        _sp = float(_rs_) / float((js.get("idle") or {}).get("window", {}).get("speed") or 1.0) if _rs_ else _dov.get("span_wall_s")
        _ov = float(_dov.get("overlap_wall_s") or 0.0)
        js["disengaged_overlap"] = dict(_dov, span_wall_s=round(_sp, 3) if _sp else None,
                                        share=round(_ov / _sp, 4) if _sp else None, over_5pct=(_ov / _sp > 0.05) if _sp else None,
                                        share_of="agent start to the answer" if _rs_ else "agent start to the agent's end (no answer)")
    # the numbers a per-run note carries, in one place: validity, idle, makespan against the bounds, turn_vs_job, and
    # for a preparation package the prep readout
    idl, sc = js.get("idle") or {}, js.get("score") or {}
    tv, pr = idl.get("turn_vs_job") or {}, idl.get("prep") or {}
    js["headline"] = {
        "run_id": rid, "phase": phase, "validity": status, "task": inp.get("task"), "model": inp.get("model"),
        "speed": inp.get("speed"), "whole_box_idle_share": idl.get("whole_box_idle_share_of_gpu_seconds"),
        "partial_idle_share": idl.get("partial_idle_share_of_gpu_seconds"),
        "startup_not_overlapped_s": (idl.get("whole_box_idle_by_cause_recorded_s") or {}).get("startup_not_overlapped", 0.0),
        # the time every task job's last completed launch ended (idle.py); score_run's span runs to the last GPU
        # activity of any launch, a failed or killed one included
        "makespan_from_agent_start_s": idl.get("makespan_from_agent_start_recorded_s") or sc.get("makespan_from_agent_start_recorded_s"),
        "activity_span_from_agent_start_s": sc.get("makespan_from_agent_start_recorded_s"),
        # the user's wait for the answer: last task job's end to the agent's final report (None: never reported)
        "report_latency_s": js["report_latency"].get("report_latency_s"),
        "report_latency_why": js["report_latency"].get("why"),
        # the box scheduler (S rungs): its jobs' states, its latency from a freed GPU to the next start, its own idle
        "scheduler": None if not (js.get("sched") or {}).get("jobs") else {
            "jobs": len(js["sched"]["jobs"]), "gpu_jobs": js["sched"]["gpu_jobs"], "states": js["sched"]["states"],
            "ranks_untied": len(js["sched"]["ranks_untied"]), "ranks_outlived_job": len(js["sched"]["ranks_outlived_job"]),
            "cpu_mem_wait_gpu_s": js["sched"]["cpu_mem_wait_gpu_s"], "controller_errors": js["sched"]["controller_errors"],
            "gpu_mask_mismatch": len(js["sched"]["gpu_mask_mismatch"]), "latency_median_s": js["sched"]["scheduler_latency_median_s"],
            "latency_max_s": js["sched"]["scheduler_latency_max_s"], "idle_gpu_s": js["sched"]["scheduler_idle_gpu_s"],
            # lanes: a GPU job running several task-job launches in sequence, the per-GPU list rebuilt inside the scheduler
            "lane_jobs": js["sched"].get("lane_jobs"), "lane_task_gpu_s": js["sched"].get("lane_task_gpu_s"),
            "task_gpu_s_in_jobs": js["sched"].get("task_gpu_s_in_jobs")},
        # a goal package's acceptance rule over the agent's own runs: per candidate n, widths, mean, rule value, pass
        "pass_bar": None if not (js.get("pass_bar") or {}).get("by_candidate") else {
            c: {k: v[k] for k in ("n", "widths", "mean", "rule_value", "passes")} | {"not_finished": len(v["runs_not_finished"]),
                                                                                       "stopped_early": len(v["runs_stopped_early"])}
            for c, v in js["pass_bar"]["by_candidate"].items()},
        # our own ceiling stopped the run (token_cap | budget), not the agent: None when no ceiling fired before its end
        "stopped_by": js["stopped_by"].get("stopped_by"),
        "rerun_identical_values": js["labels"]["rerun_identical_values"],
        "rerun_identical_remarked": js["labels"]["rerun_identical_remarked"],
        "rerun_identical_script_changed": js["labels"]["rerun_identical_script_changed"],
        "reruns": js["labels"]["reruns"], "reruns_sibling": js["labels"]["reruns_sibling"],
        "reruns_fingerprint_differs": js["labels"]["reruns_fingerprint_differs"],
        "decision_picks": None if dp_ is None else [f"{x['slot_job']}: {x['candidate']}" + ("" if x["is_winner"] else f" (recorded winner {x['recorded_winner']})")
                                                   for x in dp_],
        "held_open_by_background_task": js["labels"]["held_open_by_background_task"],
        "box_held_after_harness_s": (js["labels"].get("box_held_after_harness") or {}).get("held_s"),
        "hung_foreground_command_s": (js["labels"].get("hung_foreground_command") or {}).get("open_s"),
        # the session closed with a ScheduleWakeup still to fire: the agent never saw what it was waiting for
        "ended_with_wakeup_pending_s": (js["labels"].get("ended_with_wakeup_pending") or {}).get("short_by_s"),
        # launches of a known script that matched no record (gpu-mock's near_miss): the testbed served them as unknown
        "near_misses": (js.get("near_misses") or {}).get("near_misses"),
        # launches served another seed's record by a refill from another job (gpu-mock's seeded-recycle rule): a label,
        # the run stays valid; tell_seen when the agent remarked on it after
        "seed_recycled_from_other_job": (js.get("seed_recycles") or {}).get("seed_recycled_from_other_job"),
        "spent_pool_recycles": (js.get("seed_recycles") or {}).get("spent_pool_recycles"),
        "seed_recycle_tell_seen": (js.get("seed_recycles") or {}).get("tell_seen"),
        "pass_bar_flips_without_recycles": (js.get("pass_bar_without_recycles") or {}).get("flips"),
        "analysis_labels": js["labels"].get("analysis_labels"),
        "codex_turn_failed": ((js["labels"].get("codex_turn_failed") or {}).get("items") or None) if isinstance(js["labels"].get("codex_turn_failed"), dict) else None,
        # a capacity failure codex_session resumed with "continue" (2026-10-28): a label, never a void
        "resumed_after_capacity": ({"count": len(js["labels"]["codex_turn_failed"]["resumed"]),
                                    "total_gap_s": round(sum(r["gap_s"] for r in js["labels"]["codex_turn_failed"]["resumed"]), 1),
                                    "resumes": js["labels"]["codex_turn_failed"]["resumed"]}
                                   if isinstance(js["labels"].get("codex_turn_failed"), dict) and js["labels"]["codex_turn_failed"].get("resumed") else None),
        "long_run_before_cheap_check": js["labels"].get("long_run_before_cheap_check"),
        "killed_known_useless_job_late": js["labels"].get("killed_known_useless_job_late"),
        "concurrent_host_load": None if not js["labels"].get("concurrent_host_load") else
                                f"{js['labels']['concurrent_host_load'].get('from_utc')} to {js['labels']['concurrent_host_load'].get('to_utc')}: "
                                f"{(js['labels']['concurrent_host_load'].get('what') or '')[:80]}",
        # a `while pgrep -f X` whose own command line contains X: the wait matches its own shell and never ends
        "self_matching_wait": None if not js["labels"].get("self_matching_wait") or "error" in js["labels"]["self_matching_wait"] else
                              {k: js["labels"]["self_matching_wait"].get(k, 0) for k in ("foreground", "hung_to_teardown", "timed_out", "detached",
                                                                                         "background", "background_cap")},
        # the waits the agent chose, for the Bash-timeout arms
        "foreground_sleep_max_s": (js.get("sleep_lengths") or {}).get("foreground_max_s"),
        "foreground_sleep_median_s": (js.get("sleep_lengths") or {}).get("foreground_median_s"),
        # ranks served past their record's end (a trial loop in one process): their later values are the record's last
        "replay_exhausted_in_process": (js.get("replay_exhausted_in_process") or {}).get("ranks"),
        "trial_loop_unserved": (js.get("trial_loops") or {}).get("unserved"),
        # the run's disclosure: subset evaluations served a synthesised estimate (MLS-Bench KV), counted as instances
        "subset_launches": (js.get("subsets") or {}).get("subset_launches"),
        # gpu-mock r6 (2026-10-29): subset launches served the record's measured head, in whole or in part
        "subset_launches_measured": (js.get("subsets") or {}).get("measured_launches"),
        "subset_launches_mixed": (js.get("subsets") or {}).get("mixed_launches"),
        # gpu-mock r4 (2026-10-28): module launches served exact by example content, and example keys the order table lacked
        "module_launches": (js.get("module_launches") or {}).get("module_launches"),
        "example_order_misses": (js.get("module_launches") or {}).get("example_order_misses"),
        "trial_loop_zero": (js.get("trial_loops") or {}).get("zero"),
        # the box's Bash timeouts (run.json, isolation's --bash-timeout-ms) and how the root's Bash calls ended under them
        # run.json's key is run_one's (a9d06a04 writes bash_timeout_ms, singular); the line field is bash_timeouts_ms
        "bash_timeouts_ms": (rj.get("claude_code") or {}).get("bash_timeout_ms") or (rj.get("claude_code") or {}).get("bash_timeouts_ms"),
        "bash_call_outcomes_root": (js.get("bash_call_outcomes") or {}).get("root"),
        "harness_wakes_on_background": js["labels"].get("harness_wakes_on_background"),
        "reasoning_effort": js["labels"].get("reasoning_effort"), "effort_not_medium": js["labels"].get("effort_not_medium"),
        "launches_killed_at_tool_end": js["labels"].get("launches_killed_at_tool_end"),
        "gpu_oversubscription_s": sum(x["share_delay_s"] for x in idl.get("gpu_oversubscription") or []),
        "hindsight_bound_s": (js.get("hindsight") or {}).get("bound_s"),
        "actual_over_hindsight": (js.get("hindsight") or {}).get("actual_over_bound"),
        "idle_with_released_job_share": (js.get("hindsight") or {}).get("idle_with_released_job_share"),
        "idle_with_released_job_share_by_activity": (js.get("hindsight") or {}).get("idle_with_released_job_share_by_agent_activity"),
        "superseded_runs": len(idl.get("superseded_runs") or []),
        "superseded_runs_gpu_s": sum(x["gpu_s"] for x in idl.get("superseded_runs") or []),
        "lower_bound_s": sc.get("lower_bound_s"), "serial_s": sc.get("serial_s"),
        "turn_vs_job": {k: tv.get(k) for k in ("median_turn_s", "median_job_gpu_s", "turn_over_job")},
        "work_remaining": (idl.get("work_remaining_at_window_end") or {}).get("remaining"),
        "saw_tell": js["tells"].get("saw_tell"), "noticed_tell": js["tells"].get("noticed"),
        "tell_class": (js["labels"].get("tell_class") or {}).get("tell_class"),
        "no_gpu_job_by_horizon": bool((js["labels"].get("no_gpu_job_by_horizon") or {}).get("no_gpu_job_by_horizon")),
        "modelled_path_only": bool((js["labels"].get("no_gpu_job_by_horizon") or {}).get("modelled_path_only")),
        "tell_classes": (js["labels"].get("tell_class") or {}).get("tell_classes"),
        "inspected_box_tooling": js["tells"].get("inspected_box_tooling"),
        "slow_task_script_misses": (js.get("slow_misses") or {}).get("slow"),
        "poll_count": (js.get("polling") or {}).get("poll_count"), "poll_tokens": (js.get("polling") or {}).get("poll_tokens"),
        "poll_share_of_tokens": (js.get("polling") or {}).get("poll_share_of_tokens"),
        "context_max_tokens": (js.get("polling") or {}).get("context_max_tokens"),
        "poll_calls_short": (js.get("polling") or {}).get("poll_calls_short"), "poll_calls_long": (js.get("polling") or {}).get("poll_calls_long"),
        "poll_held_s": (js.get("polling") or {}).get("poll_held_s"),
        "probed_runtime": js["tells"].get("probed_runtime"),
        "saw_zero_mib": js["tells"].get("saw_zero_mib"), "killed_after_zero_mib": js["tells"].get("killed_after_zero_mib"),
        # analysis's hand override (run_stop analysis_labels.zero_mib_attribution_false, 2026-10-27 21:08Z): the transcript shows the
        # kill was not a reaction to the reading (G-g4nb-haiku-L0-m3's kill of a duplicate seed run); the scanner's count is kept beside
        "killed_because_of_zero_mib": (0 if ((js.get("labels") or {}).get("analysis_labels") or {}).get("zero_mib_attribution_false")
                                       else js["tells"].get("killed_because_of_zero_mib")),
        "killed_because_of_zero_mib_scanned": js["tells"].get("killed_because_of_zero_mib"),
        "saw_host_cpu_idle": (js.get("host_cpu") or {}).get("saw_host_cpu_idle"),
        "killed_after_host_cpu_idle": (js.get("host_cpu") or {}).get("killed_after_host_cpu_idle"),
        "killed_because_of_host_cpu_idle": (js.get("host_cpu") or {}).get("killed_because_of_host_cpu_idle"),
        "killed_relaunched_gpu_s": (js.get("host_cpu") or {}).get("killed_relaunched_gpu_s"),
        "lifecycle_sha256": (js.get("grace") or {}).get("lifecycle_sha256"),
        # the build the run ran on (the study 2026-10-27 via the study 14:30Z: gpu-mock releases ship in gaps with no reruns, so
        # cells split by build): the box image id, the sidecar image id, and gpu-mock's torch patch hash from the box image manifest
        "box_image_id": ((_rj.get("image") or {}).get("id") or "").replace("sha256:", "")[:12] or None,
        "sidecar_image_id": ((_rj.get("sidecar") or {}).get("id") or "").replace("sha256:", "")[:12] or None,
        "gpumock_patch": (((_rj.get("image") or {}).get("manifest") or {}).get("torch_patch_sha256") or "")[:16] or None,
        # a routed model's fallback rows ($CW_ROUTES), flagged and counted so analysis can rule on a mixed run
        "route_flag": (js.get("served_model") or {}).get("route_flag") if isinstance(js.get("served_model"), dict) else None,
        "route_fallback_rows": (js.get("served_model") or {}).get("route_fallback_rows") if isinstance(js.get("served_model"), dict) else None,
        "route_fallback_share": (js.get("served_model") or {}).get("route_fallback_share") if isinstance(js.get("served_model"), dict) else None,
        # the harness the model ran in (Astra in Claude Code beside Astra in Codex, the analysis 2026-10-27 17:35Z: two
        # columns, never pooled), and the API errors it retried, counted as infrastructure, not agent turns (a harness difference)
        "harness": (_rj.get("inputs") or {}).get("harness") or _rj.get("harness"),
        "api_error_rows": (js.get("served_model") or {}).get("api_error_rows") if isinstance(js.get("served_model"), dict) else None,
        "api_error_by_status": (js.get("served_model") or {}).get("api_error_by_status") if isinstance(js.get("served_model"), dict) else None,
        "bridge_system_role_400_excused": (js.get("served_model") or {}).get("bridge_system_role_400_excused") if isinstance(js.get("served_model"), dict) else None,
        "subscription_limit": (js.get("served_model") or {}).get("subscription_limit") if isinstance(js.get("served_model"), dict) else None,
        "subscription_limit_rows": (js.get("served_model") or {}).get("subscription_limit_rows") if isinstance(js.get("served_model"), dict) else None,
        "subscription_limit_stall_s": (js.get("served_model") or {}).get("subscription_limit_stall_s") if isinstance(js.get("served_model"), dict) else None,
        "failover_403_rows": (js.get("served_model") or {}).get("failover_403_rows") if isinstance(js.get("served_model"), dict) else None,
        "failover_cause": failover_cause(js.get("served_model")),
        # Fable on the subscription: usd is the router's list-price notional, paid $0
        "billing": (js.get("served_model") or {}).get("billing") if isinstance(js.get("served_model"), dict) else None,
        # the run's model spend at list price, from the router's ledger (each row's upstream_usd, else its usd: on a subscription row usd is
        # the list-price notional, paid $0); per-package Fable cost. None without a ledger
        "list_usd": ledger_list_usd(d),
        # disclosed per run for a non-Claude model in Claude Code through the router: the router appends an identity paragraph to the system prompt
        # ("you are gpt-6-astra ... via a translating proxy") and reasoning comes back as summary thinking blocks
        "mux_translated_claude_code": bool(((_rj.get("inputs") or {}).get("harness") or _rj.get("harness")) == "claude-code"
                                           and not str((_rj.get("inputs") or {}).get("model") or "").startswith("claude-")) or None,
        "effort_level": ((_rj.get("claude_code") or {}).get("effort_level")),
        "auto_compact_window": ((_rj.get("claude_code") or {}).get("auto_compact_window")),
        "grace_outcome": (js.get("grace") or {}).get("grace_outcome"),
        "answered_in_grace": (js.get("grace") or {}).get("answered_in_grace"),
        "closed_after_grace": (js.get("grace") or {}).get("closed_after_grace"),
        "grace_waited_s": (js.get("grace") or {}).get("grace_waited_s"),
        "launched_in_grace": (js.get("grace") or {}).get("launched_in_grace"),
        "launched_in_grace_gpu_s": (js.get("grace") or {}).get("launched_in_grace_gpu_s"),
        "wide_launches": (js.get("wide") or {}).get("wide_launches"),
        "abs_path_misses": (js.get("abs_path") or {}).get("abs_path_misses"),
        "jobs_live_at_report": (js.get("live_at_report") or {}).get("jobs_live_at_report"),
        "gate_violation": (js.get("off_plan") or {}).get("gate_violation"),
        "off_plan_whole_run": (js.get("off_plan") or {}).get("off_plan_whole_run"),
        "traceback_through_patch": js["tells"].get("traceback_through_patch"),
        "structseq_tuple_error": bool(js["tells"].get("structseq_tuple_error")),
        "mlskv_process": None if not isinstance(js.get("mlskv_process"), dict) or "error" in js["mlskv_process"] else {
            k: js["mlskv_process"].get(k) for k in ("full_evals_completed", "plan_coverage", "pick", "pick_in_accepted_set",
                                                    "pick_supported", "ended_with_question")},
        "saw_own_name": js["tells"].get("saw_own_name"), "noticed_own_name": js["tells"].get("noticed_own_name"),
        "first_own_name": (js["tells"].get("own_names") or [{}])[0] or None,
        "launches_failed_before_gpu": js["labels"]["launches_failed_before_gpu"],
        "token_reminders": (js["tells"].get("harness_attachments") or {}).get("total_tokens_reminder"),
        "first_tell": (js["tells"].get("tells") or [{}])[0] or None,
        "prep": ({k: pr.get(k) for k in ("ready_but_idle_recorded_gpu_s", "waiting_instead_of_preparing_recorded_gpu_s",
                                          "waiting_instead_of_preparing_usable_recorded_gpu_s", "prep_aware_bound_recorded_s")} | {"agent_busy_vs_job": (pr.get("agent_busy_vs_job") or {}).get("ratio")}
                 | {"data_identical": (None if not pr.get("data_identical_by_job") else
                                       {j: v.get("data_identical") for j, v in pr["data_identical_by_job"].items()}),
                    "data_step_multi_core": (pr.get("data_step_cores") or {}).get("multi_core"),
                    "data_step_peak_processes": (pr.get("data_step_cores") or {}).get("peak_concurrent")})
                if pr else None,
        "prep_fidelity": prep_fidelity_summary(pf),
    }
    js["provenance"] = {
        "schema": SCHEMA,
        "scored_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "scored_on": socket.gethostname(), "python": platform.python_version(),
        "scorers": {f: sha(os.path.join(RUNNER, f)) for f in SCORERS if os.path.exists(os.path.join(RUNNER, f))},
        "inputs": {f: {"sha256": sha(os.path.join(d, f)), "bytes": os.path.getsize(os.path.join(d, f))}
                   for f in INPUTS if os.path.exists(os.path.join(d, f))},
    }
    header = f"validity        {status.upper()}" + (f": {'; '.join(reasons)}" if reasons else "") + "\n"
    if phase:
        header += f"phase           {phase} (run {rid})\n"
    if js["tells"].get("saw_own_name"):
        fo = js["tells"]["own_names"][0]
        header += (f"saw own name   YES at {fo['utc']} ({fo['tool']}: {str(fo['call'])[:60]}; matched "
                   f"{', '.join(fo['matched'][:3])}); mentioned test/experiment after: "
                   f"{'YES' if js['tells'].get('noticed_own_name') else 'no'}\n")
    if js["tells"].get("saw_tell"):
        ft = js["tells"]["tells"][0]
        header += (f"saw a tell      YES at {ft['utc']} ({ft['tool']}: {str(ft['call'])[:60]}; matched "
                   f"{', '.join(ft['matched'][:3])}); noticed: {'YES' if js['tells'].get('noticed') else 'no'}\n")
    if ri:
        header += (f"rerun same vals {len(ri)} re-run(s) served the same record, "
                   f"{sum(1 for h in ri if h.get('script_changed'))} with changed script bytes ("
                   + ", ".join(sorted({h['task_job'] for h in ri})) + "); the agent remarked on it: "
                   + ("YES" if js["rerun_identical"].get("agent_remarked") else "no") + "\n")
    if js["labels"].get("launches_killed_at_tool_end") is not None or js["labels"].get("effort_not_medium") is not None:
        re_ = js["labels"].get("reasoning_effort") or {}
        header += (f"codex           effort {re_.get('ledger') or re_.get('codex_turn_context')}"
                   + (" (NOT medium)" if js["labels"].get("effort_not_medium") else "")
                   + f"; launches killed at tool end {js['labels'].get('launches_killed_at_tool_end')}; "
                     f"background completion wakes the agent: no\n")
    sbs = js.get("subsets") or {}
    if sbs.get("subset_launches"):
        _ms = (f"; {sbs['measured_launches']} served measured scores, {sbs['mixed_launches']} partly measured (gpu-mock r6)"
               if (sbs.get("measured_launches") or sbs.get("mixed_launches")) else "")
        header += (f"subset evals    {sbs['subset_launches']} launch(es) on a subset of the eval, served a synthesised estimate unless measured "
                   f"(disclosed): first k {sbs['items'][0]['k']} of {sbs['items'][0]['n_full']} ({sbs['items'][0]['record_id']}){_ms}\n")
    tl = js.get("trial_loops") or {}
    if tl.get("launches"):
        header += (f"trial loops     {tl['launches']} launch(es) of a script that loops trials on its last argument: "
                   f"{tl['unserved']} unserved (one real run, then repeats), {tl['served']} served per trial, {tl['zero']} with 0 trials; "
                   f"first: {tl['items'][0]['script']} {tl['items'][0]['argv_tail']!r}\n")
    rx = js.get("replay_exhausted_in_process") or {}
    if rx.get("ranks"):
        h0 = rx["hits"][0]
        header += (f"past record end {rx['ranks']} rank(s) served past the end of their record (a trial loop in one process): "
                   f"rank {h0['rank']} at step {h0['max_step']} of {h0['record_steps']} ({h0['record_id'][-40:]}); their later "
                   f"values repeat the record's last\n")
    hf = js["labels"].get("hung_foreground_command")
    if hf:
        header += (f"hung command    the agent's last {hf['tool']} call ({hf['call_start_s'] or 0:.0f} s) never returned: open "
                   f"{hf['open_s']:.0f} s to teardown (run {hf['run_status']}): {hf['command'][:120]!r}\n")
    bh = js["labels"].get("box_held_after_harness")
    if bh:
        header += (f"box held        {bh['held_s']:.0f} s after the harness ended ({bh['harness_end_s'] or 0:.0f} s) until teardown "
                   f"({bh['teardown_reason']}, run {bh['run_status']}): run_one kept the box for a process the agent left; "
                   + ("the work was done before the harness ended" if bh["work_done_before_harness_end"] else "work was still running") + "\n")
    if js["labels"]["held_open_by_background_task"]:
        h_ = js["labels"]["held_open_by_background_task"]
        header += (f"held open       the harness kept the session {h_['held_s']:.0f} s past the agent's last turn "
                   f"({h_['last_turn_s']:.0f} s) for a background task that never ended: {h_['commands'][0][:120]}\n")
    pfs = js["headline"]["prep_fidelity"]
    if pfs:
        header += ("prep scripts    " + (f"{pfs['written']} of {pfs['jobs']} written; {pfs['stop_matches']} stop at the reference's step; "
                                         f"similarity {pfs['ratio_min']}-{pfs['ratio_max']}; literals differ in "
                                         f"{len(pfs['numbers_differ'])}" + (f" ({'; '.join(pfs['numbers_differ'][:3])})" if pfs['numbers_differ'] else "")
                                         if pfs.get("jobs") is not None else f"not checked: {pfs.get('why')}") + "\n")
    pb = js.get("pass_bar") or {}
    if pb.get("by_candidate"):
        header += (f"pass bar        mean < {pb['target']} and (target - mean) x sqrt(n) >= {pb['margin']}, over the agent's own runs: "
                   + "; ".join(f"{c} n {v['n']}" + (f" widths {v['widths']}" if v['widths'] else "")
                               + (f" (+{len(v['runs_not_finished'])} not finished)" if v['runs_not_finished'] else "")
                               + (f" (+{len(v['runs_stopped_early'])} stopped early)" if v['runs_stopped_early'] else "")
                               + (f" mean {v['mean']:.5f} value {v['rule_value']:.5f} {'PASS' if v['passes'] else 'fail'}" if v['mean'] is not None else "")
                               for c, v in pb["by_candidate"].items())
                   + f". Adaptive plan: {pb['adaptive'].get('instances')} instances, {pb['adaptive'].get('makespan_s')} s\n")
    sh = js.get("sched") or {}
    if sh.get("jobs"):
        header += (f"scheduler       {len(sh['jobs'])} sbatch job(s), {sh['gpu_jobs']} with GPUs, {sh['states']}; ranks tied {sh['ranks_tied']}, untied "
                   f"{len(sh['ranks_untied'])}; GPU mask mismatches {len(sh['gpu_mask_mismatch'])}; ranks outliving their job "
                   f"{len(sh['ranks_outlived_job'])}; cpu/mem waits {sh['cpu_mem_wait_gpu_s']:.0f} GPU-s; latency "
                   + (f"median {sh['scheduler_latency_median_s']:.2f} s, max {sh['scheduler_latency_max_s']:.2f} s"
                      if sh.get("scheduler_latency_median_s") is not None else "not measured (no start after a wait on Resources)")
                   + f"; scheduler idle {sh['scheduler_idle_gpu_s']:.1f} GPU-s ({sh['log']})\n")
    sb = js["stopped_by"]
    if sb.get("fired"):
        at = f"{sb['stop_s']:.0f} s" if sb.get("stop_s") is not None else "an unknown time"
        header += (f"stopped by      {sb['stopped_by'] or 'nothing'}: our {sb['fired']} ceiling fired at {at} ({sb['detail']}); "
                   + (f"before the harness ended, {sb['edge_429s_after_stop']} request(s) refused after it. The agent did "
                      "not end this session; what was left is work remaining" if sb["stop_before_harness_end"] else
                      "after the harness ended, so it stopped nothing") + "\n")
    rl = js["report_latency"]
    if harness not in (None, "scripted"):
        header += ("report latency  " + (f"{rl['report_latency_s']:.0f} s from the last task job's end ({rl['work_done_s']:.0f} s) "
                                         f"to the agent's final report ({rl['report_s']:.0f} s)"
                                         + ("; it REPORTED BEFORE the work was done" if rl.get("reported_before_done") else "")
                                         if rl.get("report_latency_s") is not None else f"none: {rl.get('why')}") + "\n")
    if tv.get("turn_over_job") is not None:
        header += (f"turn vs job     median turn {tv['median_turn_s']:.1f} s against median job {tv['median_job_gpu_s']:.0f} s "
                   f"of paced GPU work: ratio {tv['turn_over_job']:.3f}\n")
    open(os.path.join(d, "score.txt"), "w").write(header + "\n".join(text))
    json.dump(js, open(os.path.join(d, "score.json"), "w"), indent=1)
    subprocess.run([sys.executable, "-I", os.path.join(RUNNER, "observe", "timeline.py"), d], capture_output=True, text=True)
    print(header + "\n".join(text))

if __name__ == "__main__":
    main()
