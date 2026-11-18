"""Split a run's idle GPU time into whole-box idle and partial idle, and attribute each
whole-box-idle interval to what the agent was doing (design/IDEA.md, "Idle time").

usage: python3 -I idle.py <run_dir> [--horizon-s <recorded seconds from agent start>] [--json]

Inputs, all from the run directory:
  occupancy.jsonl            gpu-mock's daemon log (busy time rebuilt as in score_run.py)
  status.json                agent_start_wall, and end_utc / agent_end_wall if present
  task.json                  the task; its stats give the default deadline
  transcript/stream.jsonl    Claude Code's --output-format stream-json stdout, or
  transcript/session/**/*.jsonl  the native session transcript (preferred when present)

The window runs from the agent's start to the last job's end, or to the task's horizon if work
remains (design/IDEA.md, "No deadline in the brief", the study 2026-10-20). The horizon is the
task's `horizon_s` (recorded seconds, owned by gpu-tasks, never shown to the agent), the run's hard
stop. A task without one falls back to its `deadline_s` (the earlier design, when the brief stated
it), then to 1.5 x its lower bound. Which one was used is printed beside every number, and runs under
a stated deadline are never pooled with runs under the horizon.

The window is cut at the last job's completion when every task job completed before the
deadline (the task's work_remaining rule): idle capacity after all work is done is not lost.
The deadline is in recorded seconds and is divided by the run's speed knob.

Whole-box idle means no GPU is busy. Partial idle means at least one GPU is busy and at least
one is not. Both are given as a share of GPU-seconds in the window (gpus x window), and
whole-box idle also as a share of wall time.

Attribution of a whole-box-idle instant, first match wins:
  before_first_launch   before the first job's process started (job_hello's t_start, else its time)
  startup_not_overlapped  a launched job is in its start-up, no GPU is busy, and an overlapping
                        schedule could have hidden it: the job was launchable while another job
                        held the GPUs. The agent's. startup_split
                        has the exact rule, agreed with gpu-tasks' expected_scores.py
  job_host_phase        a launched job's process is alive but no GPU is busy, and no schedule could
                        have hidden it: a first launch, a job whose inputs did not exist yet, in-job
                        gaps and exit tails. Real CPU work in the box, the environment's
                        (the study, 2026-10-20, design/IDEA.md "Idle time")
  route_down            the inference route was failing (the edge forwarder logged 502/503/504 or
                        an upstream error): the route's time, not the agent's
  after_agent_ended     after the agent's session ended
  after_job_failure     after a gpu-mock "failure" event and before the agent's next tool call
  model_turn            inside a model turn: from the end of the previous tool result (or the
                        previous user/system record) to the last block of the next assistant response
  wait_or_poll          inside a tool call whose command is a sleep, wait, poll or nvidia-smi
  non_gpu_tool          inside any other tool call
  unattributed          none of the above; reported, never silently dropped
Times in the transcript are UTC ISO-8601; occupancy times are converted from t_wall.
"""
import argparse, glob, json, os, re, shlex, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep as prep_mod
import score_run

WAIT_RE = re.compile(r"(^|[;&|\s(])(sleep|wait|watch|timeout\s+\d+\s+(tail|sleep)|tail\s+-f|nvidia-smi|gpustat|nvtop|ps\b|pgrep|squeue|jobs\b)", re.I)

def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def read_jsonl(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out

def transcript_records(run_dir):
    """Timestamped records, from the native session files if present, else stream-json."""
    files = sorted(glob.glob(os.path.join(run_dir, "transcript", "session", "**", "*.jsonl"), recursive=True))
    main = [f for f in files if "/subagents/" not in f]
    recs = []
    for f in main or []:
        recs += [r for r in read_jsonl(f) if r.get("timestamp")]
    if not recs:
        s = os.path.join(run_dir, "transcript", "stream.jsonl")
        if os.path.exists(s):
            recs = [r for r in read_jsonl(s) if r.get("timestamp")]
    recs.sort(key=lambda r: ts(r["timestamp"]))
    return recs

KILL_CALLS = []      # (start, end) of tool calls whose full command sends a kill, filled by activity()
# an agent's own `timeout N <cmd>` ends <cmd> N seconds after the call starts (G-g4nb-grok-L0-r2c, 2026-10-29: two probes under
# `timeout 100` and `timeout 150` ran 100.1 s and 150.1 s and were read as environment crashes, voiding the run). Each one adds a
# kill window at start + N, so a rank that stops then with no exception is the agent's stop. A timeout around a wait (tail, sleep,
# watch) kills no task job and adds none
TIMEOUT_RE = re.compile(r"(?:^|[;&|\s(])timeout\s+(?:(?:-[sk]|--signal|--kill-after)\s+\S+\s+|--?[-\w]+(?:=\S+)?\s+)*(\d+(?:\.\d+)?)([smhd]?)\s+(?!(?:tail|sleep|watch|cat|grep)\b)\S")
TIMEOUT_UNIT = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}
TIMEOUT_CALLS = []   # (start + N - 3, start + N + 30) per timeout-wrapped command, filled by activity()
# an agent's own STOP_STEP=N (an early-stop knob a recorded script reads from its environment: its loop breaks at step N) ends the
# script cleanly at step N, the agent's stop as a `timeout N` is (G-k-opus-A5-r5, the analysis 22:38 MST 11-10: its pace check
# `STOP_STEP=250 python -u adoption/soap-f1-auxb2/train_gpt_clean.py --seed 6`, a gpu-harness job, ended at step 250 of 2900 after
# 704.8 s and was read as an environment crash, voiding the run). The setting pairs with the first .py script after it on its line
STOP_STEP_RE = re.compile(r"(?<![\w$])STOP_STEP=[\"']?(\d+)")
PY_TOKEN_RE = re.compile(r"[\w./-]+\.py\b")

def subagent_kill_calls(run_dir):
    """(start, end, agent) of every kill a subagent or teammate sent (its own session file under subagents/), with that
    file's timestamps. A gpu-team run cancels jobs from its `gpus` teammate by design (the study and the analysis
    2026-10-27 00:34Z: G-g4na-haiku-T1's `kill $(cat .../soap-s0.log.pid)` at 19:43:12Z, in subagents/agent-a79b...jsonl,
    was read as an environment crash because only the main session was searched). Used for the kill rules only; idle and
    turn accounting stay the main session's"""
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "transcript", "session", "**", "subagents", "*.jsonl"), recursive=True)):
        agent = os.path.basename(f)[:-6]
        calls, results = {}, {}
        for r in read_jsonl(f):
            if not r.get("timestamp"):
                continue
            t = ts(r["timestamp"])
            msg = r.get("message") or {}
            for b in (msg.get("content") or []) if isinstance(msg, dict) and isinstance(msg.get("content"), list) else []:
                if not isinstance(b, dict):
                    continue
                if r.get("type") == "assistant" and b.get("type") == "tool_use":
                    calls[b["id"]] = (t, b.get("input") or {})
                elif b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = t
        for tid, (t0, inp) in calls.items():
            cmd = inp.get("command", "") if isinstance(inp, dict) else ""
            if KILL_RE.search(cmd or ""):
                t1 = results.get(tid, t0)
                out.append((t0, max(t0, t1), agent))
    return out

def harness_stop_calls(run_dir, before_s=2.0, after_s=30.0):
    """(start, end, job) of every job the gpu-harness skill stopped (the agent's own harness, solution's A1: `gpuh` cancels a
    rejected candidate's running job and kills its process group). Its record is captured with the workspace:
    home/.gpuh/runs/<stamp>/timeline.jsonl, a row with state "stopping" at the moment it sends the signal. The agent
    chose this by running the harness, so it counts as the agent's kill, exactly as a teammate's `kill` does
    (subagent_kill_calls). G-g4na-haiku-A1 and G-g4na-sonnet-A1, 2026-10-28: soap-run1 stopped 0.1 s after the harness's
    "soap-f1-auxb2 rejected" was read as an environment crash and voided both runs. The window runs from just before the
    stop to after_s past it, the time the harness allows a process group to exit."""
    out = []
    for f in sorted(glob.glob(os.path.join(run_dir, "workspace", "home", ".gpuh", "runs", "*", "timeline.jsonl"))):
        for r in read_jsonl(f):
            if str(r.get("state") or "").lower() == "stopping" and isinstance(r.get("t"), (int, float)):
                out.append((r["t"] - before_s, r["t"] + after_s, r.get("job")))
    return out

def own_stop_steps(run_dir, recs):
    """(t_from, t_to, script, N, source) per command that sets STOP_STEP=N for a .py script (STOP_STEP_RE): the agent's tool calls, in
    the main session and every subagent or teammate session, for a rank started at or after the call (launched by it, or from a plan
    it wrote, t_to None); and the gpu-harness board's jobs, for a rank started within 10 s of the job's started_at"""
    out = []
    def add(cmd, t_from, t_to, source):
        for line in str(cmd or "").splitlines():
            for m in STOP_STEP_RE.finditer(line):
                py = PY_TOKEN_RE.search(line, m.end())
                if py:
                    out.append((t_from, t_to, py.group(0), int(m.group(1)), source))
    sessions = [("main", recs)] + [(os.path.basename(f)[:-6], [r for r in read_jsonl(f) if r.get("timestamp")]) for f in
                                   sorted(glob.glob(os.path.join(run_dir, "transcript", "session", "**", "subagents", "*.jsonl"), recursive=True))]
    for who, rs in sessions:
        for r in rs:
            msg = r.get("message") or {}
            for b in (msg.get("content") or []) if isinstance(msg, dict) and isinstance(msg.get("content"), list) else []:
                if isinstance(b, dict) and r.get("type") == "assistant" and b.get("type") == "tool_use":
                    inp = b.get("input") or {}
                    cmd = inp.get("command", "") if isinstance(inp, dict) else ""
                    if "STOP_STEP=" in (cmd or ""):
                        add(cmd, ts(r["timestamp"]), None, f"tool call ({who})")
    ws = os.path.join(run_dir, "workspace", "home")
    boards = set(glob.glob(os.path.join(ws, ".gpuh", "runs", "*", "board.json")))
    for depth in range(1, 5):
        boards |= set(glob.glob(os.path.join(ws, *(["*"] * depth), "board.json")))
    for f in sorted(boards):
        try:
            jobs = (json.load(open(f)) or {}).get("jobs") or []
        except (ValueError, OSError, AttributeError):
            continue
        for j in jobs if isinstance(jobs, list) else []:
            if isinstance(j, dict) and isinstance(j.get("started_at"), (int, float)) and "STOP_STEP=" in str(j.get("command") or ""):
                add(j["command"], j["started_at"] - 10.0, j["started_at"] + 10.0, f"gpu-harness job {j.get('id')}")
    return out

def own_stop_at(own_stops, hello_e, reached_step):
    """N when a rank that reached step N was started by a command setting STOP_STEP=N for its script (own_stop_steps), else None.
    The scripts match on their trailing path components (a relative launch and an absolute one name the same file)"""
    tok = next((t for t in (hello_e.get("argv") or []) if isinstance(t, str) and t.endswith(".py")), None)
    if not tok or not isinstance(reached_step, (int, float)):
        return None
    start = job_start(hello_e)
    a = [p for p in os.path.normpath(tok).split("/") if p not in ("", ".")]
    for t_from, t_to, script, n, _src in own_stops:
        b = [p for p in os.path.normpath(script).split("/") if p not in ("", ".")]
        k = min(len(a), len(b))
        if (n == reached_step and k and a[-k:] == b[-k:] and start >= t_from - 1.0
                and (t_to is None or start <= t_to)):
            return n
    return None

def activity(recs, open_until=None):
    """Intervals (start, end, kind, detail) for model turns and tool calls. A tool call with no result
    is dropped, unless open_until is given: then it runs to open_until. That is the case when the box
    was stopped with the call in flight (killed-on-time, killed-on-budget, stopped), and run_one.py
    stops the box with `docker stop`, so the harness writes no result for that call."""
    calls, results = {}, {}
    KILL_CALLS.clear(); TIMEOUT_CALLS.clear()
    last_input_end = None
    turns = []                     # (start, end) per API response
    resp_last = {}                 # message.id -> last block time
    resp_first_input = {}          # message.id -> time the turn could start
    for r in recs:
        t = ts(r["timestamp"])
        typ = r.get("type")
        if typ == "assistant":
            msg = r.get("message") or {}
            mid = msg.get("id") or r.get("uuid")
            resp_last[mid] = t
            resp_first_input.setdefault(mid, last_input_end if last_input_end is not None else t)
            for b in msg.get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    calls[b["id"]] = (t, b.get("name"), b.get("input") or {})
        elif typ in ("user", "system", "queue-operation", "attachment"):
            msg = r.get("message") or {}
            content = msg.get("content") if isinstance(msg, dict) else None
            for b in content or [] if isinstance(content, list) else []:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    results[b.get("tool_use_id")] = (t, bool(b.get("is_error")))
            last_input_end = t
    for mid, end in resp_last.items():
        turns.append((resp_first_input[mid], end, "model_turn", mid))
    tools = []
    for tid, (t0, name, inp) in calls.items():
        t1 = results.get(tid, (None,))[0]
        if t1 is None and open_until is not None and open_until > t0:
            t1 = open_until
        if t1 is None or t1 < t0:
            continue
        cmd = inp.get("command", "") if isinstance(inp, dict) else ""
        bg = isinstance(inp, dict) and inp.get("run_in_background")
        kind = "wait_or_poll" if (name == "Bash" and not bg and WAIT_RE.search(cmd or "")) else "non_gpu_tool"
        tools.append((t0, t1, kind, f"{name}: {(cmd or '')[:80]}"))
        if KILL_RE.search(cmd or ""):
            KILL_CALLS.append((t0, t1))     # the whole command is read for a kill, not the 80 characters kept above
        for m in TIMEOUT_RE.finditer(cmd or ""):
            n = float(m.group(1)) * TIMEOUT_UNIT[m.group(2)]
            TIMEOUT_CALLS.append((t0 + n - 3.0, t0 + n + 30.0))
    session_end = max((ts(r["timestamp"]) for r in recs), default=None)
    return turns, tools, session_end

def whole_and_partial(ivs, gpus, w0, w1):
    """Sweep the per-GPU busy intervals over [w0, w1): returns whole-box-idle intervals and
    partial-idle GPU-seconds."""
    pts = []
    for l in ivs.values():
        for a, b in l:
            a, b = max(a, w0), min(b, w1)
            if b > a:
                pts += [(a, +1), (b, -1)]
    pts.sort()
    whole, partial_gs = [], 0.0
    busy, t = 0, w0
    for x, d in pts + [(w1, 0)]:
        if x > t:
            if busy == 0:
                whole.append((t, x))
            elif busy < gpus:
                partial_gs += (gpus - busy) * (x - t)
        busy += d
        t = max(t, x)
    return whole, partial_gs

def unmatched_ran_real(ev):
    """Ranks gpu-mock found no record for (lookup match none after a near_miss: the launch's arguments differ from every record's, e.g.
    G-k2lp-sonnet-L0's `--max-cases 0` against records made at 50 / 100 cases), which ran the agent's launch for real on the CPU device
    with no paced unit, where an H100 takes minutes (the analysis 2026-11-03 22:00 MST: the 14:09 rule extends to them,
    unmatched_launch_ran_real). {rank: near_miss event}"""
    nm = {e["job"]: e for e in ev if e.get("kind") == "near_miss"}
    enq = {e["job"] for e in ev if e.get("kind") == "enqueue"}
    return {e["job"]: nm[e["job"]] for e in ev if e.get("kind") == "lookup" and e.get("match") == "none"
            and e.get("fallback_from") == "near_miss" and e["job"] in nm and e["job"] not in enq}

def disengaged_overlap(ev, whole, span, speed):
    """The whole-box idle that overlaps any rank gpu-mock disengaged (replay_disengaged: it ran the agent's own launch for real,
    at CPU speed, where an H100 takes seconds), as a share of `span` (the analysis 2026-11-03 14:09 and 14:12 MST,
    pkg-k2-latepole: over 5% voids as disengaged_smoke_ran_real; at or under, valid with the overlap disclosed). The span is the
    time the bar reads: agent start to the agent's end here, and score_all re-shares it over agent start to the answer when the
    run has one, else to harness end. Never the horizon window, which would put every run under 5%. Unmatched ranks (unmatched_ran_real)
    count with disengaged ones. A rank's span is its process start to its bye
    (the last bye or idle end if it never said bye). Wall seconds; None when no rank was disengaged."""
    disg = {e["job"] for e in ev if e.get("kind") == "replay_disengaged"}
    unm = set(unmatched_ran_real(ev))
    dis = disg | unm
    if not dis:
        return None
    hello = {e["job"]: job_start(e) for e in ev if e.get("kind") == "job_hello" and e["job"] in dis}
    bye = {e["job"]: e["t_wall"] for e in ev if e.get("kind") == "job_bye" and e["job"] in dis}
    end = max([b for a, b in whole] + list(bye.values()) + list(hello.values()) + [0.0])
    spans = sorted((hello[j], bye.get(j, end)) for j in hello)
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    ov = sum(max(0.0, min(b, d) - max(a, c)) for a, b in whole for c, d in merged)
    return {"ranks": len(dis), "disengaged_ranks": len(disg), "unmatched_ranks": len(unm), "overlap_wall_s": round(ov, 3), "overlap_recorded_s": round(ov * speed, 3),
            "span_wall_s": round(span, 3) if span else None, "share": round(ov / span, 4) if span else None,
            "over_5pct": (ov / span > 0.05) if span else None, "share_of": "agent start to the agent's end",
            "how": "whole-box idle intersected with the union of the process-start-to-bye spans of disengaged ranks and of unmatched ranks "
                   "(a near_miss then lookup none, no paced unit: run for real), over the span"}

def job_start(e):
    """When a job's process started. Since 2026-10-20 gpu-mock logs job_hello at the job's first
    CUDA use and sends the process's own start as t_start; before that, job_hello was the start.
    t_start is in the daemon host's wall clock, as t_wall is."""
    t = e.get("t_start")
    return float(t) if isinstance(t, (int, float)) else e["t_wall"]

def route_down_intervals(run_dir):
    """When the inference route was failing, from the edge forwarder's per-request log
    (<run_dir>/forwarder/edge.jsonl, isolation's forwarder): each request that got a 502, a 503 or
    504, or an `upstream` error, from its arrival to its end. Overlapping and touching failures merge,
    and a gap of up to 30 s between failures is bridged, because the harness retries with backoff
   .
    Idle time in these intervals belongs to the route, not to the agent."""
    p = os.path.join(run_dir, "forwarder", "edge.jsonl")
    if not os.path.exists(p):
        return []
    fails = []
    for line in open(p, errors="replace"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("refused") or r.get("event"):
            continue
        st = r.get("status")
        err = str(r.get("error") or "")
        if st in (502, 503, 504) or err.startswith("upstream"):
            t0 = float(r.get("ts_epoch") or 0)
            fails.append((t0, t0 + float(r.get("total_ms") or 0) / 1000.0))
    fails.sort()
    out = []
    for a, b in fails:
        if out and a - out[-1][1] <= 30.0:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [tuple(x) for x in out]

def launcher_intervals(run_dir):
    """When a launched job's launcher was alive, from isolation's process records (resources.jsonl
    procs lines): a launcher (accelerate launch, torchrun, or the shell an agent runs a job through)
    starts before the job's ranks say hello and exits after the last says bye, and the GPUs are idle
    while it alone runs. Matching is on the cmdline of a process whose cmdline launches a training
    script; a rank's own process is already covered by alive_intervals. Returns [(start, end)] in
    wall time. A run without process records returns [], and the scorer then falls back to the ranks."""
    p = os.path.join(run_dir, "resources.jsonl")
    if not os.path.exists(p):
        return []
    started, exited, last = {}, {}, None
    def is_launcher(cmd):
        c = cmd or ""
        return (".py" in c) and any(k in c for k in ("accelerate launch", "torchrun", "torch.distributed.run", "sh -c"))
    for line in open(p, errors="replace"):
        try:
            l = json.loads(line)
        except json.JSONDecodeError:
            continue
        if l.get("kind") != "procs":
            continue
        try:
            t = ts(l["t_utc"])
        except (KeyError, ValueError):
            continue
        last = t
        procs = (l.get("procs") if l.get("full") else l.get("started")) or []
        live = set()
        for q in procs:
            if isinstance(q, dict) and q.get("pid") is not None:
                pid = int(q["pid"]); live.add(pid)
                if is_launcher(q.get("cmdline")) and pid not in started:
                    started[pid] = t
        for pid in (l.get("exited") or []):
            if int(pid) in started and int(pid) not in exited:
                exited[int(pid)] = t
        if l.get("full"):
            for pid in list(started):
                if pid not in live and pid not in exited:
                    exited[pid] = t
    return [(t0, exited.get(pid, last)) for pid, t0 in started.items() if exited.get(pid, last) is not None]

# a modelled job counts as busy on its GPU (the analysis 2026-10-28 04:41Z; the before-and-after table approved 06:32Z, no
# bar figure moved): P006_MODELLED_AS_BUSY=0 reproduces the reading before it, for a before-and-after table only
MODELLED_AS_BUSY = os.environ.get("P006_MODELLED_AS_BUSY", "1") != "0"
LAUNCH_WINDOW_S = 600   # a launching call starts at most this long before its job's process (a `sleep N; ... python x.py` chain)
LAUNCH_FALLBACK_S = 120 # ... and a call that does not name the job's script (a wrapper .sh, `python -`), this long
PROC_WINDOW_S = 15      # a launching shell in the process records is first seen within this of its job's start
_CVD_RE = re.compile(r"CUDA_VISIBLE_DEVICES=(\"[^\"]*\"|'[^']*'|[^\s;&|)]+)")
_LOOP_RE = re.compile(r"\bfor\s+(\w+)\s+in\s+([^;\n]+?)\s*;?\s*do\b(.*?)\bdone\b", re.S)

def _loop_values(vals):
    """A for-loop's word list: literal words, `$(seq a b)` and `{a..b}` expanded."""
    m = re.fullmatch(r"\$\(\s*seq\s+(\d+)\s+(\d+)\s*\)", vals.strip())
    if m:
        return [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
    m = re.fullmatch(r"\{(\d+)\.\.(\d+)\}", vals.strip())
    if m:
        return [str(i) for i in range(int(m.group(1)), int(m.group(2)) + 1)]
    return vals.split()

def gpu_assignments(cmd):
    """The (CUDA_VISIBLE_DEVICES value, the .py/.sh files named after it up to the next value) pairs a shell command sets, in launch order,
    a for-loop's body once per value (`for g in 0 1; do (CUDA_VISIBLE_DEVICES=$g python x.py) & done`); a value is a list of
    GPU indices, or None when it is not literal digits (an unexpanded variable)."""
    out, spans = [], []
    def pair(body, at, var=None, val=None, skip=()):
        for c in _CVD_RE.finditer(body):
            if any(a <= c.start() < b for a, b in skip):
                continue
            x = c.group(1).strip("'\"")
            if var and x in (f"${var}", f"${{{var}}}"):
                x = val
            nxt = _CVD_RE.search(body, c.end())
            seg = body[c.end():nxt.start() if nxt else len(body)]
            gs = [int(g) for g in x.split(",")] if re.fullmatch(r"\d+(,\d+)*", x or "") else None
            out.append((at + c.start(), gs, {os.path.basename(m_.group(1)) for m_ in re.finditer(r"([\w./-]+\.(?:py|sh))\b", seg)}))
    for m in _LOOP_RE.finditer(cmd):
        body = m.group(3)
        # a counter the body steps (`g=$((g+1))`, `((g++))`, `g=$((g + 1))`) from a literal set before the loop
        ctr = {c.group(1) for c in re.finditer(r"\b(\w+)=\$\(\(\s*(\w+)\s*\+\s*1\s*\)\)", body) if c.group(1) == c.group(2)} | \
              set(re.findall(r"\(\(\s*(\w+)\s*\+\+\s*\)\)", body))
        init = {}
        for c_ in ctr:
            ms = list(re.finditer(rf"(?:^|[;&\s(]){c_}=(\d+)\b", cmd[:m.start()]))
            if ms:
                init[c_] = int(ms[-1].group(1))
        for i, v in enumerate(_loop_values(m.group(2))):
            sub = body
            for c_, n0 in init.items():
                sub = re.sub(rf"\$\{{?{c_}\}}?(?![\w])", str(n0 + i), sub)
            pair(sub, m.start(3), m.group(1), v)
        spans.append((m.start(), m.end()))
    pair(cmd, 0, skip=spans)
    return [(g, s) for _, g, s in sorted(out, key=lambda z: z[0])]

def modelled_job_busy(run_dir, ev, gpus=None, end_wall=None):
    """Modelled-path jobs as GPU busy (the analysis 2026-10-28 04:41Z, G-kvreph-opus-L0-r2: five jobs, 24,459 s, read
    as idle because a modelled job writes no busy events). A modelled job is a rank whose lookup matched no record (match none)
    and that was never enqueued; it is busy on its GPU from its process start (job_hello's t_start) to its bye, else the run's
    end. Its GPU: job_hello carries gpu -1 and no environment, and the box's process records carry cmdlines, not environments, so
    it is read from the launch: the latest Bash call (main or subagent) up to LAUNCH_WINDOW_S before the job's start that sets
    CUDA_VISIBLE_DEVICES ahead of one of the job's scripts (a .py/.sh in its argv; a wrapper such as the agent's own wrap.py names
    the task script after itself), gpu_assignments giving literal values and a for-loop's values in order. The jobs one call
    started take its values in pid order (fork order), each the first unused value naming one of its scripts; which of two
    identical launches had which GPU is not recorded, and moves only which lane carries which interval. A job with no such call,
    or a value that is not literal digits, counts on no GPU and is listed as unassigned. Returns {gpu: [[start, end]]}, the items
    and the wall seconds on a GPU and on none."""
    hello, look, bye, enq = {}, {}, {}, set()
    for e in ev:
        k = e.get("kind")
        if k == "job_hello":
            hello.setdefault(e["job"], e)
        elif k == "lookup":
            look.setdefault(e["job"], e)
        elif k == "job_bye":
            bye[e["job"]] = e["t_wall"]
        elif k == "enqueue":
            enq.add(e["job"])
    mod = [j for j, lk in look.items() if lk.get("match") == "none" and j not in enq and j in hello]
    if not mod:
        return {"by_gpu": {}, "items": [], "modelled_job_gpu_wall_s": 0.0, "unassigned_wall_s": 0.0}
    calls = []
    for f in glob.glob(os.path.join(run_dir, "transcript", "session", "**", "*.jsonl"), recursive=True):
        for r in read_jsonl(f):
            if r.get("type") != "assistant" or not r.get("timestamp"):
                continue
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") == "Bash":
                    calls.append((ts(r["timestamp"]), (b.get("input") or {}).get("command") or ""))
    calls.sort(key=lambda c: c[0])
    merged = []
    for c in calls:
        if merged and c[0] - merged[-1][0] <= 1.0:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + c[1])
        else:
            merged.append(c)
    calls = merged
    asg_of = {}
    groups, items = {}, {}
    for j in mod:
        h = hello[j]
        t = job_start(h)
        scripts = {os.path.basename(str(a)) for a in h.get("argv") or [] if str(a).endswith((".py", ".sh"))}
        call = fallback = None
        for c in reversed(calls):
            if c[0] > t + 2:
                continue
            if c[0] < t - LAUNCH_WINDOW_S:
                break
            if c not in asg_of:
                asg_of[c] = gpu_assignments(c[1])
            if any(scripts & sc for _, sc in asg_of[c]):
                call = c
                break
            if fallback is None and c[0] >= t - LAUNCH_FALLBACK_S and any(g for g, _ in asg_of[c]):
                fallback = c
        call = call or fallback
        end = bye.get(j) or end_wall
        items[j] = {"rank": j, "pid": h.get("pid"), "start_wall": t, "end_wall": end, "scripts": sorted(scripts),
                    "argv": " ".join(map(str, h.get("argv") or []))[:160], "call_wall": call[0] if call else None,
                    "call": call[1][:300] if call else None, "gpus": None,
                    "gpu_how": "no launching call" if not call else "no literal value for it in its launching call"}
        if call:
            groups.setdefault(call, []).append(j)
    for call, js in groups.items():
        # in fork order, each job takes the call's first unused value whose segment names one of its scripts; the jobs left
        # (a wrapper .sh, a heredoc `python -`, a rewritten argv) take the unused values in order, and a call that sets a single
        # value gives it to every job it started
        allv = asg_of[call]
        free = list(allv)
        js = sorted(js, key=lambda j: (hello[j].get("pid") or 0, job_start(hello[j])))
        left = []
        for j in js:
            sc = set(items[j]["scripts"])
            k = next((i for i, (_, s_) in enumerate(free) if sc & s_), None)
            if k is None:
                left.append(j)
                continue
            items[j]["gpus"], items[j]["gpu_how"] = free.pop(k)[0], "the value set ahead of its script"
        one = {tuple(g) for g, _ in allv if g}
        for j in left:
            if free:
                items[j]["gpus"], items[j]["gpu_how"] = free.pop(0)[0], "the call's next unused value, in fork order"
            elif len(one) == 1:
                items[j]["gpus"], items[j]["gpu_how"] = list(next(iter(one))), "the call's only value"
    # a job no transcript call launched: the box's process records, a shell that set CUDA_VISIBLE_DEVICES first seen within
    # PROC_WINDOW_S of the job's start (records come every few seconds); in pid order on both sides (the box's pid namespace and
    # the host's allocate in the same order), a shell naming one of the job's scripts first
    rest = sorted((it for it in items.values() if not it["gpus"]), key=lambda it: (it["pid"] or 0, it["start_wall"]))
    if rest:
        procs = _cvd_processes(run_dir)
        used = set()
        for it in rest:
            near = [x for x in procs if abs(x[0] - it["start_wall"]) <= PROC_WINDOW_S and x[1] not in used]
            pick = next((x for x in near if any(set(it["scripts"]) & sc for _, sc in x[2])), near[0] if near else None)
            if pick:
                used.add(pick[1])
                it["gpus"] = next(g for g, _ in pick[2] if g)
                it["gpu_how"] = f"the box's process record of its launching shell (host pid {pick[1]})"
    for it in items.values():
        g = it["gpus"]
        if g and gpus and any(x >= int(gpus) for x in g):
            it["gpus"], it["gpu_how"] = None, "names a GPU the box does not have"
    by_gpu, on_s, none_s = {}, 0.0, 0.0
    for it in items.values():
        d = (it["end_wall"] - it["start_wall"]) if it["end_wall"] else 0.0
        if it["gpus"]:
            for g in it["gpus"]:
                by_gpu.setdefault(g, []).append([it["start_wall"], it["end_wall"]])
                on_s += d
        else:
            none_s += d
    return {"by_gpu": {g: _union(v) for g, v in by_gpu.items()}, "items": sorted(items.values(), key=lambda x: x["start_wall"]),
            "modelled_job_gpu_wall_s": on_s, "unassigned_wall_s": none_s}

def _cvd_processes(run_dir):
    """Processes whose cmdline sets a literal CUDA_VISIBLE_DEVICES, from isolation's process records (resources.jsonl procs
    lines): (first seen, host pid, the gpu_assignments of the cmdline). A Codex launch from its code-mode script
    (`tools.exec_command({cmd: `CUDA_VISIBLE_DEVICES=${i+1} ...`})`, G-kvgate-astra-L0) is no Bash call in the transcript, but its
    shell is a process here."""
    p = os.path.join(run_dir, "resources.jsonl")
    out, seen = [], set()
    if not os.path.exists(p):
        return out
    for line in open(p, errors="replace"):
        if '"procs"' not in line or "CUDA_VISIBLE_DEVICES=" not in line:
            continue
        try:
            l = json.loads(line)
            t = ts(l["t_utc"])
        except (ValueError, KeyError):
            continue
        for q in (l.get("procs") or []) + (l.get("started") or []):
            if isinstance(q, dict) and q.get("pid") not in seen and "CUDA_VISIBLE_DEVICES=" in (q.get("cmdline") or ""):
                asg = gpu_assignments(q["cmdline"])
                if any(g for g, _ in asg):
                    seen.add(q["pid"])
                    out.append((t, int(q["pid"]), asg))
    return sorted(out)

def with_modelled_busy(run_dir, ev, ivs, gpus, st):
    """Merge modelled_job_busy's intervals into ivs (wall time, per GPU) in place when MODELLED_AS_BUSY; returns its result,
    with the GPU-seconds it added (the part not already busy)."""
    end = st.get("teardown_wall") or st.get("harness_end_wall") or max((e["t_wall"] for e in ev), default=None)
    try:
        mb = modelled_job_busy(run_dir, ev, gpus, end)
    except Exception as ex:          # a label: never stops the score
        return {"error": repr(ex)[:300], "counted_as_busy": False}
    added = 0.0
    if MODELLED_AS_BUSY:
        for g, l in mb["by_gpu"].items():
            before = sum(b - a for a, b in _union(ivs.get(g, [])))
            ivs[g] = [list(x) for x in _union((ivs.get(g) or []) + l)]
            added += sum(b - a for a, b in ivs[g]) - before
    mb["counted_as_busy"] = MODELLED_AS_BUSY
    mb["added_busy_wall_s"] = added
    return mb

def alive_intervals(ev):
    """When each daemon job's process was alive: its start (t_start) to job_bye (or the last event, if the
    log has no bye). Whole-box idle while any launched job is alive and none is busy is the job's
    host phase (start-up, data loading, saving), not the agent's."""
    hello = {e["job"]: job_start(e) for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e["t_wall"] for e in ev if e["kind"] == "job_bye"}
    last = max((e["t_wall"] for e in ev), default=None)
    return [(t0, bye.get(j, last)) for j, t0 in hello.items() if bye.get(j, last) is not None]

def _union(ivs):
    out = []
    for a, b in sorted((a, b) for a, b in ivs if b > a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out

def _overlap(ivs, a, b):
    return sum(max(0.0, min(e, b) - max(s, a)) for s, e in ivs)

def _complement_sorted(u, lo, hi):
    """_complement on an interval list already unioned (sorted, disjoint): only the part that meets [lo, hi) is walked,
    found by bisection (a G run's lane holds about 10,000 intervals, and startup_split asks once per job per GPU)."""
    import bisect
    i = max(0, bisect.bisect_left(u, [lo, lo]) - 1)
    out, t = [], lo
    for s, e in u[i:]:
        if s >= hi:
            break
        if e <= t:
            continue
        if s > t:
            out.append([t, min(s, hi)])
        t = max(t, e)
    if t < hi:
        out.append([t, hi])
    return out

def _complement(ivs, lo, hi):
    out, t = [], lo
    for s, e in _union(ivs):
        if e <= t or s >= hi:
            continue
        if s > t:
            out.append([t, min(s, hi)])
        t = max(t, e)
    if t < hi:
        out.append([t, hi])
    return out

def _intersect(A, B):
    """A ∩ B by a two-pointer merge of the two unioned lists: linear, where the pairwise loop it replaces was
    quadratic (a G run's whole-box idle and busy lanes hold about 10,000 intervals each)."""
    out, A, B = [], _union(A), _union(B)
    i = j = 0
    while i < len(A) and j < len(B):
        lo, hi = max(A[i][0], B[j][0]), min(A[i][1], B[j][1])
        if hi > lo + 1e-9:
            out.append([lo, hi])
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1
    return _union(out)

def _minus(A, B):
    Bu = _union(B)             # once, not once per interval of A (a G run's whole-box idle has about 10,000)
    return [iv for s, e in _union(A) for iv in _complement_sorted(Bu, s, e)]

def _latest(ivs, amount):
    out, need = [], amount
    for s, e in reversed(_union(ivs)):
        if need <= 1e-9:
            break
        take = min(need, e - s)
        out.append([e - take, e])
        need -= take
    return _union(out)

def _hideable(a, p, r, H):
    """max over x in [r, a] of |H ∩ [x, x + p)| - |H ∩ [a, a + p)|, floored at 0 (gpu-tasks' hideable_startup)."""
    if r is None or r >= a - 1e-9 or p <= 1e-9:
        return 0.0
    cands = {r, a} | {x for s, e in H for edge in (s, e) for x in (edge, edge - p) if r <= x <= a}
    return max(0.0, max(_overlap(H, x, x + p) for x in cands) - _overlap(H, a, a + p))

def startup_split(ev, task, whole, agent_start, run_dir=None, lanes=None):
    """Split host-phase idle into job_host_phase (the environment's) and startup_not_overlapped (the
    agent's): design/IDEA.md, "job host phase", the study 2026-10-20. The definition is gpu-tasks'
    startup_split in build/expected_scores.py, agreed line for line, with f_j the latest first unit:
      S_j = [a_j, f_j): a_j the job's alive start (its launcher's start where the box recorded one,
            else its first rank's t_start); f_j the LATEST first-unit start over its ranks, when its GPU
            work is under way everywhere. In a real run rank 0 runs one unit early and then waits at the
            first collective, so the earliest unit would cut the start-up short.
      held span of a job = [f_j, its last unit's end over all ranks); in-job gaps count as held.
      H_j, the cover = the union of every other job's held span, and [a_k, k's last unit end) for every
            job k launched strictly before j: j's start-up could have run beside k's own start-up too,
            whose idle is already the environment's. Not to k's exit: launched earlier, j's GPU work would
            have filled k's exit tail. An earlier job that never reached the GPUs covers [a_k, its exit).
            A job launched with or after j covers it only by its held span, so two late jobs launched
            together are both charged and neither excuses the other.
      r_j = agent start, or the latest job_bye among j's `after` dependencies; a dependency that never
            exited means j was never launchable early, and A_j = 0.
      A_j = max over x in [r_j, a_j] of |H_j ∩ [x, x + p_j)| - |H_j ∩ S_j|, p_j = f_j - a_j, floored at 0
            and capped at S_j's whole-box idle. The overlap is piecewise linear in x, so the maximum is at
            r_j, a_j, or where x or x + p_j meets an edge of H_j.
      the latest A_j seconds of S_j's whole-box idle are startup_not_overlapped; the rest job_host_phase.
    S_j's whole-box idle starts no earlier than the first launch: before it, idle is before_first_launch.
    Per job the report gives the start-up, its whole-box idle and A_j, not a per-job job_host_phase: where
    two start-ups share idle the earlier-launched job carries the agent's share, so the environment's
    share exists only in the cause totals.
    A task job is the set of ranks task_job_of maps to it (one launch per task job). Ranks that cannot be
    mapped are their own job with A = 0, counted in "unmapped", which leans toward the environment.
    Partial idle, per GPU g of j (the study 2026-10-20, "Partial idle also carries causes"; gpu-tasks'
    definition): the cover C_j^g is C_j restricted to jobs on GPU g, plus the whole-box idle inside C_j
    (idle already accounted in the whole-box totals). A_j^g = min(hideable over C_j^g, GPU g's idle in
    S_j); the latest A_j^g seconds of GPU g's idle in S_j are the agent's on that lane, the whole-box
    instants among them are left to the whole-box rule, and the partial ones are startup_not_overlapped
    in GPU-seconds. lanes is {gpu: busy intervals}; without it the per-lane share is not computed.
    Returns (agent_parts [(s, e)], per_job {task job: {...}}, lane_parts {gpu: [(s, e)]}), wall seconds."""
    jobs = (task or {}).get("jobs") or []
    after = {j["job_id"]: list(j.get("after") or []) for j in jobs}
    s0 = next((e for e in ev if e["kind"] == "daemon_start"), ev[0])
    off = s0["t_wall"] - s0["t_mono"]
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    bye = {e["job"]: e["t_wall"] for e in ev if e["kind"] == "job_bye"}
    first, last, cur, gpus_of = {}, {}, {}, {}
    for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
        st = max(cur.get(g, t), t); cur[g] = st + c
        first.setdefault(e["job"], st); last[e["job"]] = st + c
        gpus_of.setdefault(e["job"], set()).add(g)
    groups = {}
    for dj, h in hello.items():
        tj = task_job_of(h, lookup.get(dj), jobs) or f"unmapped-{dj}"
        groups.setdefault(tj, []).append(dj)
    launchers = launcher_intervals(run_dir) if run_dir else []
    rank_start = {tj: min(job_start(hello[d]) for d in djs) for tj, djs in groups.items()}
    t0 = min(rank_start.values(), default=None)       # the first launch
    info, pre_gpu = {}, {}
    for tj, djs in groups.items():
        if not all(d in first for d in djs):
            # never reached the GPUs: no start-up to split, but it covers later launches while alive
            pre_gpu[tj] = (rank_start[tj], max((bye.get(d) or rank_start[tj]) for d in djs))
            continue
        a = rank_start[tj]
        # The launcher that started this job's ranks: alive when its first rank started, and started
        # after every earlier job's first rank, so a shell that chains `a; b` is not taken as b's start.
        prev = max((t for k, t in rank_start.items() if k != tj and t < a), default=float("-inf"))
        ls = [s for s, e in launchers if prev < s <= a <= e]
        a = min(ls) if ls else a
        f = max(first[d] for d in djs)
        held = (f, max(last[d] for d in djs))
        done = all(d in bye for d in djs)
        info[tj] = {"a": a, "f": f, "held": held, "bye": max(bye[d] for d in djs) if done else None,
                    "exit": max(bye.get(d, last[d]) for d in djs),
                    "gpus": set().union(*(gpus_of.get(d, set()) for d in djs))}
    whole_u = _union(whole)
    lanes_u = {}               # each GPU's busy lane, unioned once
    parts, per, lane_parts = [], {}, {}
    def cover(tj, a, on_gpu=None):
        keep = lambda o: on_gpu is None or on_gpu in o["gpus"]
        return _union([o["held"] for k, o in info.items() if k != tj and keep(o)] +
                      [(o["a"], o["held"][1]) for k, o in info.items() if k != tj and o["a"] < a and keep(o)] +
                      ([iv for k, iv in pre_gpu.items() if iv[0] < a] if on_gpu is None else []))
    for tj, j in info.items():
        a, f = j["a"], j["f"]
        p = f - a
        H = cover(tj, a)
        deps = after.get(tj)
        if deps is None:
            r = None                                   # unmapped: not provably launchable early
        elif any(d not in info or info[d]["bye"] is None for d in deps):
            r = None
        else:
            r = max([agent_start] + [info[d]["bye"] for d in deps])
        act = _overlap(H, a, f)
        best = act
        if r is not None and r < a and p > 0:
            cands = {r, a}
            for s, e in H:
                for edge in (s, e):
                    for x in (edge, edge - p):
                        if r <= x <= a:
                            cands.add(x)
            best = max(_overlap(H, x, x + p) for x in cands)
        lo = max(a, t0) if t0 is not None else a
        idle_in = [[max(s, lo), min(e, f)] for s, e in whole_u if min(e, f) > max(s, lo)]
        idle_s = sum(e - s for s, e in idle_in)
        A = min(max(0.0, best - act), idle_s)
        need = A
        for s, e in reversed(idle_in):
            if need <= 1e-9:
                break
            take = min(need, e - s)
            parts.append((e - take, e))
            need -= take
        lane_s = part_gs = 0.0
        if lanes is not None and f > lo + 1e-9:
            env_whole = _intersect(whole_u, H)
            for g in sorted(j["gpus"]):
                Hg = _union(cover(tj, a, on_gpu=g) + env_whole)
                if g not in lanes_u:
                    lanes_u[g] = _union(lanes.get(g, []))
                idle_g = _complement_sorted(lanes_u[g], lo, f)
                Ag = min(sum(e - s for s, e in idle_g), _hideable(a, p, r, Hg))
                Pg = _minus(_latest(idle_g, Ag), whole_u)
                lane_parts.setdefault(g, []).extend(Pg)
                lane_s += Ag
                part_gs += sum(e - s for s, e in Pg)
        per[tj] = {"startup_s": p, "startup_whole_box_idle_s": idle_s, "startup_not_overlapped_s": A,
                   "partial_startup_not_overlapped_gpu_s": part_gs, "lane_hideable_s": lane_s, "gpus": len(j["gpus"]),
                   "launchable_from_wall": r, "alive_from_wall": a,
                   "unmapped": tj.startswith("unmapped-")}
    return parts, per, {g: _union(v) for g, v in lane_parts.items()}

def partial_attribution(lanes, gpus, w0, w1, lane_parts, lane_alive, launches, task, done_at, agent_end, turns, tools,
                        failures=(), route_down=()):
    """Causes of partial idle, per GPU lane, in GPU-seconds, summing to the partial idle reported beside
    them (the study 2026-10-20, "Partial idle also carries causes"; gpu-tasks' partial_attribution,
    agreed). A lane-instant in [w0, w1) is partial idle when its GPU is idle and another is busy. First
    match wins:
      startup_not_overlapped  inside the agent part of a start-up on that lane (startup_split's lane parts)
      job_host_phase          a job on that GPU is alive: its start-up, an in-job gap or its exit tail
      route_down              the inference route was failing
      no_work_left            no task job is both unlaunched and launchable, so nothing could run there
      after_agent_ended, after_job_failure, model_turn, wait_or_poll, non_gpu_tool, unattributed
    launches: {task job: launch wall time}; done_at: {task job: exit wall time} for jobs that ended."""
    jobs = (task or {}).get("jobs") or []
    anybusy = _intersect(_union([iv for l in lanes.values() for iv in l]), [[w0, w1]])
    acts = sorted(turns + tools)
    def work_left(m):
        for j in jobs:
            L = launches.get(j["job_id"])
            if L is not None and L <= m:
                continue
            if all(d in done_at and done_at[d] <= m for d in j.get("after") or []):
                return True
        return False
    cause = {}
    events = set(launches.values()) | set(done_at.values())
    import bisect
    # "is m inside any of these intervals": a bisection on the unioned list, not a scan (a G run has about 10,000
    # idle pieces per lane and thousands of turns and tool calls; the scan made this quadratic, 3 minutes on c3-11)
    def member(ivs):
        u = _union(ivs); starts = [a for a, _ in u]
        def inside(m):
            i = bisect.bisect_right(starts, m) - 1
            return i >= 0 and m < u[i][1]
        return inside
    in_route = member(route_down)
    # activities by start, for "the model turn, else the first activity, covering m" in the original order
    acts_by_start = sorted(acts, key=lambda x: x[0]); act_starts = [x[0] for x in acts_by_start]
    max_len = max((t1 - t0 for t0, t1, *_ in acts), default=0.0)
    def covering(m):
        """the activities with t0 <= m < t1, in acts' original (sorted) order"""
        hi = bisect.bisect_right(act_starts, m); lo = bisect.bisect_left(act_starts, m - max_len)
        return sorted((x for x in acts_by_start[lo:hi] if x[0] <= m < x[1]), key=acts.index) if hi > lo else []
    for g in range(gpus):
        idle = _intersect(_complement(lanes.get(g, []), w0, w1), anybusy)
        if not idle:
            continue
        agent = lane_parts.get(g, [])
        alive = lane_alive.get(g, [])
        in_agent, in_alive = member(agent), member(alive)
        cuts_all = sorted(events | {x for x in (agent_end, *failures) if x is not None} |
                          {x for iv in alive for x in iv} | {x for iv in agent for x in iv} |
                          {x for iv in route_down for x in iv} | {x for s_, e_, *_ in acts for x in (s_, e_)})
        for s_, e_ in idle:
            cuts = [s_] + [c for c in cuts_all[bisect.bisect_right(cuts_all, s_):bisect.bisect_left(cuts_all, e_)]] + [e_]
            for x, y in zip(cuts, cuts[1:]):
                if y <= x:
                    continue
                m = (x + y) / 2
                if in_agent(m):
                    k = "startup_not_overlapped"
                elif in_alive(m):
                    k = "job_host_phase"
                elif in_route(m):
                    k = "route_down"
                elif not work_left(m):
                    k = "no_work_left"
                elif agent_end is not None and m > agent_end:
                    k = "after_agent_ended"
                else:
                    fail = [f_ for f_ in failures if f_ <= m]
                    nxt = min((t0 for t0, t1, kk, _ in tools if fail and t0 >= max(fail)), default=None) if fail else None
                    if fail and (nxt is None or m < nxt):
                        k = "after_job_failure"
                    else:
                        cov = covering(m)
                        k = next((kk for t0, t1, kk, _ in cov if kk == "model_turn"), None) or \
                            next((kk for t0, t1, kk, _ in cov), None) or "unattributed"
                cause[k] = cause.get(k, 0.0) + (y - x)
    return cause

def attribute(whole, first_launch, agent_end, failures, turns, tools, alive=(), route_down=(), agent_parts=()):
    acts = sorted(turns + tools)
    out = {}
    def add(k, s):
        out[k] = out.get(k, 0.0) + s
    for a, b in whole:
        cuts = {a, b}
        for p in (first_launch, agent_end, *[f for f in failures]):
            if p is not None and a < p < b:
                cuts.add(p)
        for s, e, *_ in list(acts) + [(s, e) for s, e in alive] + [(s, e) for s, e in route_down] + [(s, e) for s, e in agent_parts]:
            for p in (s, e):
                if a < p < b:
                    cuts.add(p)
        cuts = sorted(cuts)
        for x, y in zip(cuts, cuts[1:]):
            m = (x + y) / 2
            if first_launch is not None and m < first_launch:
                add("before_first_launch", y - x); continue
            if any(s <= m < e for s, e in alive):
                # host phase an overlapping schedule would have hidden is the agent's (startup_split)
                add("startup_not_overlapped" if any(s <= m < e for s, e in agent_parts) else "job_host_phase", y - x)
                continue
            if any(s <= m < e for s, e in route_down):
                add("route_down", y - x); continue
            if agent_end is not None and m > agent_end:
                add("after_agent_ended", y - x); continue
            fail = [f for f in failures if f <= m]
            if fail:
                nxt = min((s for s, e, k, _ in tools if s >= max(fail)), default=None)
                if nxt is None or m < nxt:
                    add("after_job_failure", y - x); continue
            k = next((k for s, e, k, _ in acts if s <= m < e and k == "model_turn"), None) \
                or next((k for s, e, k, _ in acts if s <= m < e), None)
            add(k or "unattributed", y - x)
    return out

def _output_dir(argv):
    for i, a in enumerate(argv or []):
        if a.startswith("--output_dir="):
            return a.split("=", 1)[1]
        if a == "--output_dir" and i + 1 < len(argv):
            return argv[i + 1]
    return None

def _script_args(cmd):
    """The script and its options in a command line (a string or an argv list): the first token
    ending in .py, then --key=value, --key value and bare --flag after it, with - in keys read as _."""
    toks = shlex.split(cmd) if isinstance(cmd, str) else list(cmd or [])
    i = next((k for k, t in enumerate(toks) if t.endswith(".py")), None)
    if i is None:
        return None, {}
    opts, rest, k = {}, toks[i + 1:], 0
    while k < len(rest):
        if rest[k].startswith("--"):
            key, eq, val = rest[k][2:].partition("=")
            key = key.replace("-", "_")
            if eq:
                opts[key] = val
            elif k + 1 < len(rest) and not rest[k + 1].startswith("--"):
                opts[key] = rest[k + 1]; k += 1
            else:
                opts[key] = True
        k += 1
    return os.path.normpath(toks[i]), opts

def _same_script(a, b):
    """Two script paths name the same file when the shorter is the tail of the longer, so a launch
    from inside the script's directory, or by absolute path, still matches the task's relative one."""
    pa, pb = a.split(os.sep), b.split(os.sep)
    n = min(len(pa), len(pb))
    return pa[-n:] == pb[-n:]

def _job_by_arguments(argv, jobs):
    """The one task job whose launch line this command matches: the same script, and the same value
    for every option on which the task's launches of that script differ (the seed, in a seed fan-out).
    The output dir is left out: it is the agent's to choose, and the output-dir rule read it first."""
    script, opts = _script_args(argv)
    if not script:
        return None
    cands = []
    for j in jobs:
        s, o = _script_args(j.get("launch") or "")
        if s and _same_script(script, s):
            cands.append((j["job_id"], o, s))
    if len({s for _, _, s in cands}) > 1:
        return None  # a bare "train.py" run from inside one of several directories: which file is unknown
    cands = [(jid, o) for jid, o, _ in cands]
    keys = {k for _, o in cands for k in o} - {"output_dir"}
    differ = [k for k in keys if len({str(o.get(k)) for _, o in cands}) > 1]
    hits = [jid for jid, o in cands if all(str(opts.get(k)) == str(o.get(k)) for k in differ)]
    return hits[0] if len(hits) == 1 else None

def listed_records(job, decisions=None):
    """Every record gpu-mock may serve a task job without a defect: its records, its
    rerun_siblings, and for a decision slot (records null, bind {decision, slot}) its candidate_pool, its recorded
    winner's records and its decision's candidates' confirm_records."""
    out = list(job.get("records") or []) + list(job.get("rerun_siblings") or [])
    if job.get("bind"):
        out += list(job.get("candidate_pool") or []) + list((job.get("recorded_winner") or {}).get("records") or [])
        did = (job.get("bind") or {}).get("decision")
        for d in decisions or []:
            if d.get("decision_id") == did:
                out += [r for c in d.get("candidates") or [] for r in c.get("confirm_records") or []]
    return out

# H1-style decisions: a decision whose candidates carry confirm_by_seed names, for each served
# continuation record, its candidate and seed. The slot is that candidate's, in the order the run first continued each
# candidate; the run is the seed's place in the candidate's seeds. gpu-mock's daemon binds every slot launch to the decision's
# first free slot with no bind.run, so its task_job reads runs 2 and 3 as run 1 and four continuations as superseded.
# The order comes from the run's own lookups: each lookup event carries t_wall, so a record's first-continued time is known
# from the lookups seen so far in this run (kept per occupancy file by the lookup's own t_wall, never across runs).
def _confirm_index(jobs):
    """{record: (decision, candidate, run)} from the decisions the slot jobs bind, read off the task's jobs' `decision_meta`
    when the loader attached it, else empty (the tie then falls back to the daemon's bind)."""
    out = {}
    for dec in DECISIONS_OF.get(id(jobs)) or []:
        for c in dec.get("candidates") or []:
            seeds = [int(x) for x in (c.get("seeds") or [])]
            for sd, v in (c.get("confirm_by_seed") or {}).items():
                if isinstance(v, dict) and v.get("record") and int(sd) in seeds:
                    out[v["record"]] = (dec["decision_id"], c["variant"], seeds.index(int(sd)) + 1)
    return out

DECISIONS_OF = {}       # id(task["jobs"]) -> task["decisions"], registered by attach_decisions
FIRST_CONT = {}         # id(task["jobs"]) -> {decision: {candidate: earliest t_wall of a served continuation record}}

def attach_decisions(task, lookups):
    """Register a task's decisions and each candidate's first continuation time in this run, keyed by the task's own job list
    (the object every caller passes as `jobs`), so a scorer scoring many runs never mixes one run's order into another."""
    jobs = (task or {}).get("jobs") or []
    DECISIONS_OF[id(jobs)] = list((task or {}).get("decisions") or [])
    idx = _confirm_index(jobs)
    first = {}
    for e in lookups:
        if e.get("record_id") in idx:
            dec, cand, _ = idx[e["record_id"]]
            t = e.get("t_wall") or 0.0
            if cand not in first.setdefault(dec, {}) or t < first[dec][cand]:
                first[dec][cand] = t
    FIRST_CONT[id(jobs)] = first

def _slot_by_seed(lookup, jobs):
    """The slot job a continuation record ties to by candidate and seed; None when the task's decisions were not attached,
    carry no confirm_by_seed, or the record is not one."""
    idx = _confirm_index(jobs)
    rid = (lookup or {}).get("record_id")
    if rid not in idx:
        return None
    dec, cand, run = idx[rid]
    first = (FIRST_CONT.get(id(jobs)) or {}).get(dec) or {}
    if cand not in first:
        return None
    slot = sorted(first, key=lambda c: first[c]).index(cand) + 1
    hits = [j["job_id"] for j in jobs if (j.get("bind") or {}).get("decision") == dec and
            (j.get("bind") or {}).get("slot") == slot and (j.get("bind") or {}).get("run") == run]
    return hits[0] if len(hits) == 1 else None

def _slot_of(lookup, jobs):
    """A decision slot's rank, by the record it was served (sc-q, halving-q: a confirm launch runs the same script and
    seed as its candidate's screen, so its arguments name the screen; only the record, a later segment from the
    slot's pool, tells them apart). The slots of one decision share a pool, so: the slot the lookup names, if it
    names one; else the slot whose recorded winner is that record; else none (untied: review, never a silent guess)."""
    # a decision with confirm_by_seed (H1): the served record's candidate and seed name the slot and run, whatever the daemon
    # bound (its bind has no run)
    by_seed = _slot_by_seed(lookup, jobs)
    if by_seed:
        return by_seed
    # gpu-mock binds a slot launch itself from its daemon of 2026-10-21 (the lowest slot of the decision not yet bound,
    # in launch order) and names it: lookup.bind {decision, slot} and lookup.task_job. Taken as given for a slot.
    if isinstance((lookup or {}).get("bind"), dict) and (lookup or {}).get("task_job") in {j["job_id"] for j in jobs}:
        return lookup["task_job"]
    rid = (lookup or {}).get("record_id")
    if not rid:
        return None
    slots = [j for j in jobs if j.get("bind") and rid in listed_records(j)]
    if not slots:
        return None
    b = (lookup or {}).get("bind") or {}
    dec, slot = b.get("decision", (lookup or {}).get("decision")), b.get("slot", (lookup or {}).get("slot"))
    if slot is not None:
        named = [j for j in slots if (j["bind"].get("slot") == slot and (dec is None or j["bind"].get("decision") == dec))]
        if len(named) == 1:
            return named[0]["job_id"]
    if len(slots) == 1:
        return slots[0]["job_id"]
    won = [j for j in slots if rid in ((j.get("recorded_winner") or {}).get("records") or [])]
    return won[0]["job_id"] if len(won) == 1 else None

def held_open(run_dir, recs):
    """Was the session held open after the agent's last turn by background tasks that never ended? From the box's
    harness stream (box/harness.stdout): the last `result` came while background tasks were alive and no assistant
    event followed it. Returns the tasks (id, command) and the last turn's time (the transcript's last assistant
    record), or None."""
    p = os.path.join(run_dir, "box", "harness.stdout")
    if not os.path.exists(p):
        return None
    alive, started, after_result, at_result = {}, {}, False, None
    for line in open(p, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t, sub = r.get("type"), r.get("subtype")
        if t == "system" and sub == "task_started":
            started[r.get("task_id")] = r.get("description")
        elif t == "system" and sub == "background_tasks_changed":
            alive = {x.get("task_id"): x.get("description") for x in r.get("tasks") or []}
        elif t == "result":
            after_result, at_result = True, dict(alive)
        elif t == "assistant":
            after_result = False
    if not after_result or not at_result:
        return None
    last = max((ts(r["timestamp"]) for r in recs if r.get("type") == "assistant" and r.get("timestamp")), default=None)
    if last is None:
        return None
    return {"last_turn_wall": last, "tasks": [{"task_id": k, "command": (started.get(k) or v or "")[:300]} for k, v in at_result.items()]}

def tie_jobs(task):
    """The jobs a rank may tie to: the task's jobs, plus its unneeded_jobs (gpu-tasks' G4n twins, 2026-10-24: the rejected
    setup's full runs, served exactly when launched but owed by no deliverable). The task's own list object when it has
    none, so attach_decisions' registration by id(task["jobs"]) still holds."""
    jobs = (task or {}).get("jobs") or []
    extra = (task or {}).get("unneeded_jobs") or []
    return jobs + [dict(j, unneeded=True) for j in extra] if extra else jobs

def owed_jobs(task):
    """The task jobs the deliverable needs: every job but a reference_only one (G4n's soap-probe: a script only the scripted
    reference writes, its look at a setup an agent takes by launching the setup itself; the study 2026-10-24 14:02Z)."""
    return [j for j in (task or {}).get("jobs") or [] if not j.get("reference_only")]

# gpu-mock next/ r4 (2026-10-28 07:29Z, fix:gpumock-module-entry): an evaluator loaded as a module by the agent's own runner is served
# exact (match exact_module / exact_effective_module) by example content, possibly a slice of the workload; job_bye carries
# `positions` {n, first, last, contiguous, prefix}. A repeated module launch of one record is draw module_repeat, another shard
EXACT_MATCHES = ("exact", "exact_effective", "exact_module", "exact_effective_module")
MODULE_MATCHES = ("exact_module", "exact_effective_module")
REPEAT_DRAWS = ("deterministic_rerun", "module_repeat")

def shard_coverage(look, bye, n_of, rep_wall=None, positions_of=None):
    """Whole-workload evaluations made of shards: per task job, the record positions its
    clean, served, unsubsetted launches decoded (job_bye `positions`, first..last when contiguous), and whether together they span
    0..n-1 of the record's n_examples_full. A single launch that covered n already counts as a whole evaluation by its last_step;
    this adds the case of several launches, none whole, whose union is. look/bye: {rank: event}; n_of: {task job: n};
    rep_wall: only shards that ended by then count toward `covered_at_report`. Returns {task job: {"covered": bool,
    "covered_at": wall of the shard that completed the span, "covered_at_report": bool, "shards": [...], "positions": n}}.
    positions_of: {rank: set of record positions it decoded}, from the occupancy log's example_score materialise events
    (example_positions below); a non-contiguous shard (gpu-mock r4's job_bye `positions` has no list, only n/first/last) is unioned
    from it, exactly (the analysis 2026-10-31 18:14 MDT, G-kvgate-sol-L0-r3: three stride-3 shards of kivi's GSM8K whose union
    is all 1,319). Without it, a non-contiguous shard is skipped, as before."""
    by = {}
    for j, lk in look.items():
        tj = lk.get("task_job")
        b = bye.get(j) or {}
        p = b.get("positions")
        if not tj or lk.get("subset") or b.get("served") != "result" or not isinstance(p, dict) or not p.get("n"):
            continue
        if p.get("contiguous"):
            pos = set(range(int(p["first"]), int(p["last"]) + 1))
        elif positions_of and positions_of.get(j):
            pos = set(positions_of[j])     # a non-contiguous set, unioned exactly from its own per-example events
        else:
            continue                       # a non-contiguous set with no per-example events: it cannot be unioned exactly
        by.setdefault(tj, []).append((b.get("t_wall"), pos, j))
    out = {}
    for tj, xs in by.items():
        n = int(n_of.get(tj) or 0)
        if not n:
            continue
        xs.sort(key=lambda x: x[0] or 0)
        cov, at, at_rep = set(), None, False
        for t, pos, j in xs:
            cov |= {x for x in pos if 0 <= x < n}
            if at is None and len(cov) >= n:
                at = t
        at_rep = at is not None and (rep_wall is None or (at or 0) <= rep_wall)
        out[tj] = {"covered": at is not None, "covered_at": at, "covered_at_report": at_rep, "positions": len(cov),
                   "n_full": n, "shards": [{"rank": j, "first": min(pos), "end": max(pos) + 1, "n": len(pos),
                                            "contiguous": max(pos) - min(pos) + 1 == len(pos), "t_wall": t} for t, pos, j in xs],
                   # the shards that had ended when the span closed: the whole evaluation's own work (later repeats excluded)
                   "used": [j for t, pos, j in xs if at is not None and (t or 0) <= at]}
    return out

def example_positions(ev):
    """{rank: set of record positions} from the occupancy log's example_score materialise events: `position`, else `example`
    (example 0's event carries position null; G-kvgate-sol-L0-r3, 2026-10-31). For shard_coverage's non-contiguous shards."""
    out = {}
    for e in ev:
        if e.get("kind") == "materialise" and e.get("role") == "example_score":
            x = e.get("position") if isinstance(e.get("position"), int) else e.get("example")
            if isinstance(x, int):
                out.setdefault(e.get("job"), set()).add(x)
    return out

def task_job_of(hello, lookup, jobs):
    """Which task job a daemon job (one rank) belongs to. First by the launch's --output_dir:
    a run whose launches each have their own output dir (every scripted plan, and any agent that
    isolated its outputs) maps exactly, whatever record the draw gave it. Then by the launch's
    arguments (_job_by_arguments): an agent that named its own output dirs ("...-seed2" for job
    "sft-seed2") still passed the option that tells the task's jobs apart. A seed-substituted job
    draws a record listed under another job, so the record is only the last fallback."""
    od = _output_dir(hello.get("argv"))
    if od:
        base = os.path.basename(od.rstrip("/"))
        # the output dir names the job when it IS the job id or ends in "-<job id>" (plan.py's form);
        # the longest matching id wins, so "sft-seed1" never claims "sft-seed11"
        hits = [j["job_id"] for j in jobs if base == j["job_id"] or base.endswith("-" + j["job_id"])]
        if hits:
            best = max(hits, key=len)
            if sum(1 for h in hits if len(h) == len(best)) == 1:
                return best
    if any(j.get("bind") for j in jobs) and lookup:
        # a slot launch (the daemon's bind, or a record from a decision slot's listed set) ties to that slot before the
        # arguments, which name the screen. The daemon's task_job is not used for ordinary jobs: it follows the record,
        # and a seed-substituted rank draws a record listed under another job
        if isinstance(lookup.get("bind"), dict) or (lookup.get("record_id") and
                                                    any(j.get("bind") and lookup["record_id"] in listed_records(j) for j in jobs)):
            return _slot_of(lookup, jobs)
    by_args = _job_by_arguments(hello.get("argv"), jobs)
    if by_args:
        return by_args
    if lookup and lookup.get("record_id"):
        # a rerun may be served one of its job's rerun_siblings (gpu-mock's `sibling` draw, 2026-10-21); gpu-tasks'
        # validator keeps every sibling under one job
        hits = [j["job_id"] for j in jobs if lookup["record_id"] in listed_records(j)]
        if len(hits) == 1:
            return hits[0]
    return None

RECORDS = os.environ.get("P006_RECORDS_DIR") or os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gpu-tasks", "registry", "records"))
_FINAL = {}

def record_final_step(rid, by="series"):
    """A record's last step. by="steps": its `steps`, the optimizer steps gpu-mock paces (read against the enqueue
    events' `step`). by="series": the last step of its longest series (val_loss, train_time_s), else its `steps`
    (read against materialise events, which exist only where the script reads a value). None when the record is not
    in the registry."""
    if rid not in _FINAL:
        try:
            r = json.load(open(os.path.join(RECORDS, rid + ".json")))
        except (OSError, ValueError):
            _FINAL[rid] = None
            return None
        lasts = [v["steps"][-1] for v in (r.get("series") or {}).values() if isinstance(v, dict) and v.get("steps")]
        st = int(r["steps"]) if r.get("steps") else None
        # a whole_job record (the TTC tabular cells) has no steps or series: gpu-mock paces its n_units, enqueued with unit
        # whole_job and step 1..n_units, so its last unit is its final step either way
        sv = (r.get("materialisation") or {}).get("serving") or {}
        if st is None and not lasts and sv.get("mode") == "whole_job" and sv.get("n_units"):
            st = int(sv["n_units"])
        _FINAL[rid] = {"series": max(lasts) if lasts else st, "steps": st}
        # a per_example record (the MLS-KV evaluations) has no steps or series either: gpu-mock paces each example as one unit
        # (unit example, step 1..k), so it is checked against what the launch asked for, never by a final step (per_example_n)
        n_full = sv.get("n_examples_full")
        n_full = n_full.get("value") if isinstance(n_full, dict) else n_full
        if sv.get("mode") == "per_example" and n_full:
            _FINAL[rid]["per_example_n"] = int(n_full)
    return (_FINAL[rid] or {}).get(by)

def per_example_n(rid):
    """A per_example record's n_examples_full, None for any other record or one not in the registry."""
    record_final_step(rid)
    return (_FINAL.get(rid) or {}).get("per_example_n")

# Crashes a real H100 run would also have: the package scripts' own guard when a data file the agent should have
# prepared is missing or half-written (gpu-tasks, 2026-10-21: "data file <path> is missing or incomplete", exit 1).
# Such a crash is the agent's (it launched too early), never the environment's.
REAL_CRASH_RE = re.compile(r"data file .* is missing or incomplete", re.I)

# scancel too: under the box scheduler an agent stops its own job with scancel, which signals the job's processes (G-h3-haiku-S1,
# 2026-10-24: `scancel 104348 104349 104350 104351` at 9,653 s ended two exact-match ranks at steps 384 and 75, and the run was
# read as an environment crash)
# `kill -0 <pid>` is a liveness test, never a kill: a worker
# loop's `while kill -0 "$pid"` must not open a kill window that reads a rank ending inside it as the agent's kill
KILL_RE = re.compile(r"(^|[;&|\s(])(kill(?!\s+-0\b)|pkill|killall|scancel)\b"
                     # the same from Python: os.kill / os.killpg / signal.SIGINT|SIGTERM|SIGKILL sent to a pid (a Codex agent
                     # interrupted its own duplicated run that way, G-goalq-astra-L0-v2 at 02:52:53Z; the ranks' KeyboardInterrupt
                     # was then read as an environment crash)
                     r"|\bos\.kill(pg)?\s*\(|\bsignal\.SIG(INT|TERM|KILL)\b")

def scripted_exit_failures(recs):
    """Task-job labels whose process exited non-zero, from the scripted agent's own records: a background launch's
    <task-notification> with <status>failed</status>, or a foreground run's tool_result marked is_error. A model
    run's harness reports no exit codes this way, so it returns nothing there."""
    desc, failed = {}, set()
    for r in recs:
        if not r.get("scripted"):
            continue
        c = (r.get("message") or {}).get("content")
        if isinstance(c, str) and "<task-notification>" in c and "<status>failed</status>" in c:
            m = re.search(r"<task-id>([^<]+)</task-id>", c)
            if m:
                failed.add(m.group(1))
        for b in c if isinstance(c, list) else []:
            if b.get("type") == "tool_use":
                desc[b.get("id")] = (b.get("input") or {}).get("description")
            if b.get("type") == "tool_result" and b.get("is_error") and desc.get(b.get("tool_use_id")):
                failed.add(desc[b["tool_use_id"]])
    return failed

def agent_written(run_dir):
    """Files under project/ the agent wrote or changed, from the workspace manifest: every listed file except the
    staged inputs it left as they were (why "staged input, unchanged")."""
    p = os.path.join(run_dir, "workspace", "manifest.jsonl")
    out = set()
    if os.path.exists(p):
        for line in open(p, errors="replace"):
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if (m.get("path") or "").startswith("project/") and m.get("why") != "staged input, unchanged":
                out.add(m["path"])
    return out

def _script_rel(argv):
    """The rank's script as a manifest path (project/...), from its argv's first .py token."""
    tok = next((t for t in (argv or []) if isinstance(t, str) and t.endswith(".py")), None)
    if not tok:
        return None
    tok = tok.split("/home/researcher/", 1)[-1] if tok.startswith("/") else "project/" + os.path.normpath(tok)
    return os.path.normpath(tok)

def width_ties(hello, lookup, jobs, bye=None, bad=(), reached=None):
    """{rank: task job} for a task whose jobs carry required_width (pkg-astra-widths; Astra independent, 2026-11-02). A launch is
    its ranks sharing a job_key (else a rank alone), its width their count. Its candidate rows are those whose launch line names
    the same script and the same values of the options the line carries, the output dir excepted (unseeded rows share all of
    them: muonh-tuned's 5 at width 1 and 5 at 2). In start order, a launch ties to the first row of its width not yet taken,
    preferring one that lists its record; a launch that completed (every rank byed at its record's last step, none failed or
    raised) takes that row, so a failed launch and its relaunch tie to one row. A launch whose width no open row of its script
    has (all of that width taken, or none of that width at all) ties to the first open candidate of another width, not taking
    it, so the wrong width is seen at completion. A launch with no candidate is left to task_job_of."""
    bye, reached = bye or {}, reached or {}
    def completed(ds):
        rid = next(((lookup.get(d) or {}).get("record_id") for d in ds if (lookup.get(d) or {}).get("record_id")), None)
        fin = record_final_step(rid, "steps") if rid else None
        return (all(d in bye for d in ds) and not (set(ds) & set(bad))
                and fin is not None and all(reached.get(d, -1) >= fin for d in ds))
    groups = {}
    for d in hello:
        groups.setdefault((lookup.get(d) or {}).get("job_key") or f"rank-{d}", []).append(d)
    order = sorted(groups.items(), key=lambda kv: min(job_start(hello[d]) for d in kv[1]))
    taken, out = set(), {}
    for key, ds in order:
        script, opts = _script_args(hello[ds[0]].get("argv"))
        if not script:
            continue
        cands = []
        for j in jobs:
            s, o = _script_args(j.get("launch") or "")
            if s and _same_script(script, s) and all(str(opts.get(k)) == str(v) for k, v in o.items() if k != "output_dir"):
                cands.append(j)
        if not cands:
            continue
        rid = next(((lookup.get(d) or {}).get("record_id") for d in ds if (lookup.get(d) or {}).get("record_id")), None)
        same_w = [j for j in cands if int(j.get("required_width") or 0) == len(ds)]
        open_w = [j for j in same_w if j["job_id"] not in taken]
        listed = [j for j in open_w if rid and rid in (j.get("records") or [])]
        if open_w:
            pick = (listed or open_w)[0]
            if completed(ds):
                taken.add(pick["job_id"])
        else:
            pick = next((j for j in cands if j["job_id"] not in taken), cands[0])
        for d in ds:
            out[d] = pick["job_id"]
    return out

def pool_ties(hello, lookup, jobs, bye=None, bad=(), reached=None, skip=()):
    """{rank: task job} for interchangeable rows: rows with the same launch
    line and the same record pool, drawn without replacement (obs-latepole's 5 muonh and 5 normuonh at 1,000 steps), which neither an
    output dir, an argument nor the record (listed under all of them) tells apart, so task_job_of ties none. Only ranks task_job_of leaves
    untied, and not in `skip` (width_ties' own), are tied here, so no rank that ties today moves. A launch is its ranks sharing a job_key
    (else a rank alone). In start order, a launch ties to the group's first open row; a launch that completed (every rank byed at its
    record's last step, none failed or raised) takes that row only when its record is one the group has not yet been credited with: a
    rerun served the same record ties to the same row and credits nothing new, so a run is never credited with more rows done than
    distinct records it was served. A failed launch ties without taking its row, so it and its relaunch share one"""
    bye, reached = bye or {}, reached or {}
    groups = {}
    for j in jobs:
        recs = tuple(sorted(j.get("records") or []))
        if j.get("launch") and recs:
            groups.setdefault((j["launch"].strip(), recs), []).append(j)
    pooled = {k: v for k, v in groups.items() if len(v) > 1}
    if not pooled:
        return {}
    def completed(ds, rid):
        fin = record_final_step(rid, "steps") if rid else None
        return (all(d in bye for d in ds) and not (set(ds) & set(bad))
                and fin is not None and all(reached.get(d, -1) >= fin for d in ds))
    launches = {}
    for d in hello:
        if d in skip or task_job_of(hello[d], lookup.get(d), jobs) is not None:
            continue
        launches.setdefault((lookup.get(d) or {}).get("job_key") or f"rank-{d}", []).append(d)
    order = sorted(launches.items(), key=lambda kv: min(job_start(hello[d]) for d in kv[1]))
    taken, credited, row_of_record, out = {}, {}, {}, {}
    for key, ds in order:
        rid = next(((lookup.get(d) or {}).get("record_id") for d in ds if (lookup.get(d) or {}).get("record_id")), None)
        script, opts = _script_args(hello[ds[0]].get("argv"))
        def _match(g):
            s2, o2 = _script_args(g[0])
            return (rid in g[1] and script and s2 and _same_script(script, s2)
                    and all(str(opts.get(k)) == str(v) for k, v in o2.items() if k != "output_dir"))
        grp = next((g for g in pooled if _match(g)), None)
        if grp is None:
            continue
        rows = pooled[grp]
        if rid in row_of_record.setdefault(grp, {}):
            pick = row_of_record[grp][rid]           # a rerun served a record already credited: the same row, nothing new
        else:
            open_rows = [r for r in rows if r["job_id"] not in taken.setdefault(grp, set())]
            if not open_rows:
                continue                               # every row taken: more launches than rows, left untied
            pick = open_rows[0]["job_id"]
            if completed(ds, rid):
                taken[grp].add(pick)
                row_of_record[grp][rid] = pick
        for d in ds:
            out[d] = pick
    return out

def work_remaining(ev, task, deadline_wall=None, exit_failed=(), kills=(), window_end=None, harness_end=None,
                   agent_files=(), controller_end=None, own_stops=()):
    """The task's work_remaining rule (jobs_not_completed): a task job is done when every rank launched for it said
    job_bye with no failure event, reached its record's final step (the last step gpu-mock served it, from the
    occupancy log's materialise events), and did not exit non-zero (scripted runs, which record exit codes).
    A rank whose record is not in the registry (or that has none) cannot be checked and is listed as unverified. A rank served a
    per_example record (no final step) is checked against the examples it asked for and listed in per_example_ranks (experiments-
    analysis 2026-11-03 06:48 MST); one short of it is no completed instance of an open pool (16:16 MST), never a crash, and one
    that still counted toward a fixed job's completed launch is listed in per_example_short_counted_complete. Daemon jobs are ranks, tied to
    task jobs by task_job_of. Ranks that cannot be tied to one task job are counted separately.
    environment_crashes: exact-match ranks (the recorded script and arguments, which completed on the recorded
    hardware) that ended before their final step, with no recorded failure, no kill by the agent around their end,
    and before the window's end. An H100 would have run them to the end, so the fake environment broke them
    (the study, 2026-10-21: the fake device's isfinite crashed 33 of the 86 mng3 scripts at step 0).
    cancelled_by_controller (controller_end: a scripted controller run's exit, run.json scripted.policy controller): a rank that
    stopped short with no exception of its own and that the controller stopped, never the environment (the analysis
    2026-11-02 12:22 MST, W0-sfz-reference): relaunched as the same task job within 10 s of its end (the controller's stop after
    reading its log, the six zero-LR first-update stops), or ended within 5 s of the controller's exit and within 10 s after another
    rank's exception (its teardown after that error). The controller's own signals are not tool calls, so these are the only trace."""
    # ranks tie to the task's jobs and its unneeded_jobs (tie_jobs); only owed_jobs are owed (the study 2026-10-24 14:02Z,
    # G4n): a run whose unneeded or reference-only work never ran is done when the owed jobs are, as on any fixed list
    owed = {j["job_id"] for j in owed_jobs(task)}
    unneeded = {j["job_id"] for j in (task or {}).get("unneeded_jobs") or []}
    jobs = tie_jobs(task)
    if not jobs:
        return {"remaining": None, "task_jobs": None, "done": None, "unmapped_ranks": None, "untied_ranks": None,
                "records_not_listed": None, "not_completed": None, "failed_launches": None, "environment_crashes": None,
                "agent_caused_crashes": None, "killed_at_harness_exit": None, "unknown_origin_crashes": None, "cancelled_by_controller": None,
                "briefed_failures": None,
                "wrong_width_launches": None, "broken_launches": None,
                "unverified_ranks": None, "disengaged_ranks": None, "per_example_ranks": None,
                "per_example_short_counted_complete": None, "done_at_wall": {}, "completed_launch_wall_s": {}, "done_launch_wall": {},
                "completed_launch_solo_s": {}, "share_delay_by_rank_s": {},
                "superseded_runs": [], "open_pool": False, "open_instances": [], "open_unfinished": [],
                "all_done_wall": None}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    failed = {e["job"] for e in ev if e["kind"] == "failure"}
    ranks = {}
    untied, wrong = [], []
    work = {}
    # a rank's wait behind another job on its GPU: gpu-mock time-slices a shared GPU (sharers > 1), so a unit
    # enqueued while shared waits for the other job's units; the gap from such a unit's end to the rank's next unit is
    # sharing delay (Q-prepP6-haiku-L2: two jobs per GPU on two GPUs, contra's 399 s of work spread over 883 s)
    share_delay, prev_unit, rank_gpus = {}, {}, {}
    for e in ev:
        if e["kind"] == "enqueue":
            rank_gpus.setdefault(e["job"], set()).add(e["gpu"])
            work[e["job"]] = work.get(e["job"], 0.0) + float(e.get("wall_cost", 0.0))
            pu = prev_unit.get(e["job"])
            if pu and (pu[2] or 1) > 1:
                share_delay[e["job"]] = share_delay.get(e["job"], 0.0) + max(0.0, e["t_wall"] - pu[0] - pu[1])
            prev_unit[e["job"]] = (e["t_wall"], float(e.get("wall_cost", 0.0)), e.get("sharers"))
    # a task whose jobs carry required_width (pkg-astra-widths: the width is a factor of the report; Astra independent and the
    # the study, 2026-11-02): a launch is its ranks sharing the daemon's job_key, and its width is their count. The rows of one
    # script and seed (unseeded muonh-tuned: 10 rows, 5 at width 1 and 5 at 2) tie by width, each launch to the next undone row of
    # its width in task order; a launch at a width no row of its script has stays tied by task_job_of and is listed below
    if any(j.get("required_width") for j in jobs):
        _reached = {}
        for e in ev:
            if e["kind"] == "enqueue" and e.get("unit") in ("step", "whole_job") and isinstance(e.get("step"), (int, float)):
                _reached[e["job"]] = max(_reached.get(e["job"], -1), e["step"])
        width_tie = width_ties(hello, lookup, jobs, bye, failed | {e["job"] for e in ev if e["kind"] == "exception"}, _reached)
    else:
        width_tie = {}
    # interchangeable rows (same launch, same record pool: pkg-obs-latepole's muonh and normuonh), for ranks task_job_of leaves untied
    _reached_p = {}
    for e in ev:
        if e["kind"] == "enqueue" and e.get("unit") in ("step", "whole_job") and isinstance(e.get("step"), (int, float)):
            _reached_p[e["job"]] = max(_reached_p.get(e["job"], -1), e["step"])
    pool_tie = pool_ties(hello, lookup, jobs, bye, failed | {e["job"] for e in ev if e["kind"] == "exception"}, _reached_p, skip=width_tie)
    for dj in hello:
        tj = width_tie.get(dj) or task_job_of(hello[dj], lookup.get(dj), jobs) or pool_tie.get(dj)
        if tj is None:
            # never counted silently: an untied rank's GPU work cannot complete a task
            # job, so it reads as work remaining and stretches the window; score_all sends the run to review
            argv = hello[dj].get("argv")
            untied.append({"job": dj, "gpu_wall_s": round(work.get(dj, 0.0), 3),
                           "record_id": (lookup.get(dj) or {}).get("record_id"),
                           "argv": (" ".join(argv) if isinstance(argv, list) else str(argv or ""))[:400]})
            continue
        ranks.setdefault(tj, []).append(dj)
        rid = (lookup.get(dj) or {}).get("record_id")
        # a sibling served to a rerun is replication, not a defect (another real run of the same configuration), and a
        # decision slot is served from its pool
        listed = next((listed_records(j, (task or {}).get("decisions")) for j in jobs if j["job_id"] == tj), [])
        if rid and listed and rid not in listed:
            # gpu-mock served this rank a record its task job does not list (P-pilotp6-haiku-L0: seed 1 was
            # served seed 3's record after a substitute took seed 1's); timing about right, values not
            wrong.append({"job": dj, "task_job": tj, "record_id": rid, "draw": lookup[dj].get("draw"), "listed": listed})
    # How far a rank got. From gpu-mock's revision of 2026-10-21 every enqueue carries `step` and `unit` ("step" or
    # "eval"): the max step of its step units against the record's `steps` holds whatever the script reads. Before
    # it, the last materialise event against the record's last series step: right for every script that validates
    # or logs at its end (all mng3 and TRL runs so far), and blind to one that saves and stops silently.
    paced, read, ex_paced = {}, {}, {}
    for e in ev:
        # a whole_job record's units (unit whole_job, step 1..n_units) are its steps (record_final_step reads n_units)
        if e["kind"] == "enqueue" and e.get("unit") in ("step", "whole_job") and isinstance(e.get("step"), (int, float)):
            paced[e["job"]] = max(paced.get(e["job"], -1), e["step"])
        if e["kind"] == "materialise" and isinstance(e.get("step"), (int, float)):
            read[e["job"]] = max(read.get(e["job"], -1), e["step"])
        # a per_example record's examples (unit example, step 1..k): how many the rank was paced (per_example_ranks)
        if e["kind"] == "enqueue" and e.get("unit") == "example" and isinstance(e.get("step"), (int, float)):
            ex_paced[e["job"]] = max(ex_paced.get(e["job"], 0), int(e["step"]))
    has_step_units = any(e["kind"] == "enqueue" and "step" in e and "unit" in e for e in ev)
    reached = paced if has_step_units else read
    exc = {e["job"]: e for e in ev if e["kind"] == "exception"}
    # a rank gpu-mock disengaged (replay_disengaged: a whole_job identity check refused the launch, e.g. env DLM_MAX_EXAMPLES='2'
    # against a record made with '0'; G-k2lp-opus-L0, 2026-11-03) ran the agent's own launch for real, not the record: it serves
    # no record units, so judging it against the record's final step reads a clean exit as a crash. Listed apart, never short
    disengaged = {e["job"]: e for e in ev if e["kind"] == "replay_disengaged"}
    short, unverified, crashes, agent_crashes, harness_kills, unknown = {}, [], [], [], [], []
    disengaged_ranks, per_example = [], {}
    open_ids = {j["job_id"] for j in jobs if j.get("draw") == "open"}
    cancelled, briefed = [], []
    exc_times = sorted((e["t_wall"], e["job"]) for e in ev if e["kind"] == "exception")
    # a job whose failure the brief sets: its real
    # exception, matched by type, message and innermost frame, is a briefed failure, never the environment's crash nor the agent's
    fab = {j["job_id"]: j["fails_as_briefed"] for j in jobs if isinstance(j.get("fails_as_briefed"), dict)}
    def briefed_match(tj, e):
        f = fab.get(tj)
        if not f or not e:
            return False
        x = f.get("exception") or {}
        suffix = f.get("frame_file_suffix") or x.get("frame_file_suffix")
        return (str(e.get("etype") or "") == str(x.get("type") or "") and bool(x.get("message_re"))
                and re.search(x["message_re"], str(e.get("message") or "")) is not None
                and (not suffix or str(e.get("frame_file") or "").endswith(suffix)))
    for tj, djs in ranks.items():
        for d in djs:
            rid = (lookup.get(d) or {}).get("record_id")
            if d in disengaged and d not in exc:
                disengaged_ranks.append({"job": d, "task_job": tj, "record_id": rid, "reason": disengaged[d].get("reason"),
                                         "detail": str(disengaged[d].get("detail") or "")[:200]})
                continue
            final = record_final_step(rid, "steps" if has_step_units else "series") if rid else None
            n_pe = per_example_n(rid) if rid and final is None else None
            if n_pe:
                # a per_example record: checked against the examples the launch asked for, a --max-examples k (k 0 or >= n is the
                # whole record), a module launch's own shard (job_bye positions n, gpu-mock r4), else n_examples_full; complete when
                # it said job_bye with no failure or exception, paced that many examples, and its bye's last_step agrees
                lk, b = lookup.get(d) or {}, bye.get(d) or {}
                sub = lk.get("subset") or b.get("subset")
                k = int(sub.get("k") or 0) if isinstance(sub, dict) else 0
                pos = b.get("positions") if isinstance(b.get("positions"), dict) else {}
                module = lk.get("match") in MODULE_MATCHES
                asked = (min(k, n_pe) if k else int(pos["n"]) if module and pos.get("n") else None if module else n_pe)
                got = ex_paced.get(d, 0)
                last = b.get("last_step")
                why = (f"raised {exc[d].get('etype')}" if d in exc else "a recorded failure" if d in failed
                       else "no job_bye" if d not in bye else "a module launch with no positions" if asked is None
                       else f"paced {got} of {asked} examples" if got < asked
                       else f"its job_bye's last_step {last} is not the {got} examples paced" if isinstance(last, int) and last != got
                       else None)
                # what ended a short rank that said job_bye: the agent's own kill or timeout, the harness's exit, the window's end
                t_b = b.get("t_wall")
                ended_by = (None if why is None or d in exc or t_b is None else
                            "the agent's kill" if any(a - 2 <= t_b <= z + 2 for a, z in kills) else
                            "the harness's exit" if harness_end is not None and abs(t_b - harness_end) <= 5.0 else
                            "the window's end" if window_end is not None and t_b >= window_end - 1.0 else "not seen")
                per_example[d] = {"job": d, "task_job": tj, "record_id": rid, "n_examples_full": n_pe,
                                  "asked": asked, "asked_from": ("subset k" if k else "module shard" if module else "n_examples_full"),
                                  "examples_paced": got, "complete": why is None, "why_short": why, "ended_by": ended_by}
                if why is not None and d not in exc and tj in open_ids:
                    # an open pool's instance that fell short of what it asked for did not do the job, whatever its clean bye
                    #; never a crash
                    short[d] = {"job": d, "task_job": tj, "reached_step": got, "final_step": asked}
                if d in exc:
                    # as for an unverified rank: an exception is a crash whatever the record
                    short[d] = {"job": d, "task_job": tj, "reached_step": reached.get(d, -1), "final_step": None,
                                "exception": f"{exc[d].get('etype')}: {exc[d].get('message')}"[:300]}
                continue
            if final is None:
                unverified.append({"job": d, "task_job": tj, "record_id": rid,
                                   "why": "no record id" if not rid else "record not in the registry"})
                # an exception is a crash whatever the record: without a final step to check it against, the rank still
                # did not complete (an open pool's instance that raised, the runner 2026-10-21)
                if d in exc:
                    short[d] = {"job": d, "task_job": tj, "reached_step": reached.get(d, -1), "final_step": None,
                                "exception": f"{exc[d].get('etype')}: {exc[d].get('message')}"[:300]}
                continue
            if (reached.get(d, -1) < final and d in bye) or d in exc:
                # a rank that served no step at all crashed before its first one: reached_step -1
                short[d] = {"job": d, "task_job": tj, "reached_step": reached.get(d, -1), "final_step": final}
                if d in exc:
                    short[d]["exception"] = f"{exc[d].get('etype')}: {exc[d].get('message')}"[:300]
                t_end = bye[d]["t_wall"] if d in bye else exc[d]["t_wall"]
                killed = any(a - 2 <= t_end <= b + 2 for a, b in kills)
                # the brief's own failure (fails_as_briefed): before every other reading, so an agent that later edits the script to
                # fix it (the manifest's `edited` is the run's end state) does not turn the attempt before the edit into its own crash
                # the agent's own STOP_STEP=N (own_stop_steps): a rank started by a command setting it for its script that ends cleanly
                # at step N stopped where the agent asked, never the environment's crash
                _ss = own_stop_at(own_stops, hello[d], reached.get(d)) if d not in exc and d not in failed and d in hello else None
                if _ss is not None:
                    agent_crashes.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                              why=f"stopped at its own STOP_STEP={_ss} (the agent's early stop)", relaunched_after_s=None))
                    continue
                if d in exc and briefed_match(tj, exc[d]):
                    briefed.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                        why=f"the brief's {(fab[tj].get('mode') or 'failure')} at {fab[tj].get('at') or 'its step'}: "
                                            f"{exc[d].get('etype')}: {str(exc[d].get('message'))[:120]}"))
                    continue
                # an uncaught exception in an exact-match job (the recorded script and arguments, which ran to the end
                # on the recorded hardware) is the environment's; so is an early bye with neither a failure the
                # record replays, nor an exception, nor a kill by the agent, before any stop or horizon
                real = d in exc and REAL_CRASH_RE.search(str(exc[d].get("message") or ""))
                # A rank that stopped short within a few seconds of the harness's exit was killed by it: `claude -p`
                # waits a bounded time for background tasks after the agent's last message, then exits and kills them
                # and the jobs they launched (Q-prepP6-haiku-L0: six subagents, three jobs killed at 703.5 s). An
                # interactive session would have kept them, so it is the measuring harness's, labelled for review.
                if harness_end is not None and abs(t_end - harness_end) <= 5.0 and d not in exc and d not in failed:
                    harness_kills.append(dict(short[d], record_id=rid, ended_s_after_start=round(t_end - job_start(hello[d]), 3)))
                    continue
                if controller_end is not None and d not in exc and d not in failed:
                    # the relaunch's process can start a moment before the stopped rank's bye lands (W0-sfz-reference: contra s1's
                    # relaunch started 0.01 s before rank 4's bye), so 2 s either side of the end, a later start up to 10 s after
                    relaunch = min((job_start(hello[o]) for o in djs if o != d and job_start(hello[o]) > job_start(hello[d])
                                    and t_end - 2.0 <= job_start(hello[o]) <= t_end + 10.0), default=None)
                    err = next((j for t, j in exc_times if j != d and t_end - 10.0 <= t <= t_end), None)
                    if relaunch is not None or (err is not None and abs(t_end - controller_end) <= 5.0):
                        cancelled.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                              why=("stopped and relaunched by the controller (relaunched %+.1f s from its end)" % (relaunch - t_end)
                                                   if relaunch is not None else
                                                   f"the controller's teardown after rank {err}'s exception (ended {controller_end - t_end:.1f} s before its exit)")))
                        continue
                # A path-bound match serves the record for a script the agent wrote (the preparation packages): an
                # exception there is the agent's code failing (Q-prepP6-haiku-L0-v2: "NameError: name 'AdamW' is not
                # defined" in its own soda-fade script), which a real run would also hit. Only an exact match, the
                # recorded script and arguments that ran to the end on the recorded hardware, can be the environment's.
                match = (lookup.get(d) or {}).get("match")
                # an exact match whose script the agent edited (the workspace manifest lists it as changed) runs the
                # agent's code too: an exception there is the agent's
                edited = _script_rel(hello[d].get("argv")) in set(agent_files)
                # From gpu-mock's next images the exception carries its innermost frame (frame_file). A frame in our
                # mock (the patch, the vendored FakeGPU, all under /usr/local/nvidia/) is the environment's whatever
                # the script; otherwise an agent-written or edited script's exception is the agent's (a torch frame
                # there is torch rejecting the agent's own call, as a GPU would); an unedited recorded script's is the
                # environment's. With no frame, an unedited script's exception is of unknown origin (gpu-mock).
                frame = str((exc.get(d) or {}).get("frame_file") or "")
                ours = frame.startswith("/usr/local/nvidia/")
                if killed and str((exc.get(d) or {}).get("etype") or "") in ("KeyboardInterrupt", "SystemExit", ""):
                    # the agent's own kill (a kill, pkill or os.kill whose call spans this rank's end): a signal's
                    # KeyboardInterrupt, or a silent exit, is the agent stopping its job, never the environment's crash
                    # nor one of unknown origin (G-goalq-astra-L0-v2: it SIGINTed a run it judged a duplicate, 2026-10-22)
                    agent_crashes.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                              why="stopped by the agent's own kill", relaunched_after_s=None))
                    continue
                if d in exc and not real and not ours and not (match == "path" or edited) and not frame:
                    unknown.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                        why="an exception with no recorded frame in an unedited recorded script"))
                    continue
                if real or (d in exc and not ours and (match == "path" or edited)):
                    later = [job_start(hello[o]) for o in djs if o != d and job_start(hello[o]) > t_end]
                    agent_crashes.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                              why=("the data guard: launched before its data was ready" if real else
                                                   "an exception in a script the agent wrote (path-bound match)" if match == "path"
                                                   else "an exception in a script the agent edited (workspace manifest)"),
                                              relaunched_after_s=round(min(later) - t_end, 3) if later else None))
                elif ((match in EXACT_MATCHES or ours) and d not in failed
                        and (d in exc or (not killed and (window_end is None or t_end < window_end - 1.0)))):
                    crashes.append(dict(short[d], record_id=rid, ended_after_s=round(t_end - job_start(hello[d]), 3),
                                        read_from="enqueue step units" if has_step_units else "materialise events"))
    exit_failed = set(exit_failed)
    done, last, not_completed, failed_launches = set(), None, [], []
    wrong_width, broken_completed = [], []
    # an open pool (gpu-tasks' goal packages, draw "open"): every completed launch is an instance, none supersedes
    # another, and the job is done once instances.min have completed (min 0: nothing is owed)
    open_jobs = {j["job_id"]: (j.get("instances") or {}) for j in jobs if j.get("draw") == "open"}
    open_instances, open_unfinished = [], []
    done_at, completed_run, done_launch, superseded, solo_run, done_records = {}, {}, {}, [], {}, {}
    pe_counted, done_key = [], {}
    # a shard of a job whose record its shards cover exactly (shard_coverage: every position 0..n-1, e.g. G-k2lp-astra-L0-v2's 27 contiguous
    # squat gsm8k shards, 1,319 of 1,319) did its part of the whole evaluation: read complete, not short against n_examples_full
    # (the analysis 2026-11-04 07:26 MST; a label, work_remaining's done rule is shard_coverage's own, unchanged)
    _covered = {}
    if per_example:
        _n_of = {}
        for _d, _u in per_example.items():
            _n_of[_u["task_job"]] = _u["n_examples_full"]
        for _tj, _c in shard_coverage(lookup, bye, _n_of, None, example_positions(ev)).items():
            if not _c["covered"]:
                continue
            # only shards the covering needs: a rank whose positions the other used shards already cover is a superseded attempt (a killed
            # first launch, then a full relaunch: G-k2lp-haiku-L0's squat gsm8k 0-420, then 0-1,318), never a shard of the evaluation
            _n = _c["n_full"]; _sh = [x for x in _c["shards"] if x["rank"] in set(_c["used"])]
            _need = set()
            for x in _sh:
                rest = set()
                for y in _sh:
                    if y["rank"] != x["rank"]:
                        rest |= set(range(y["first"], y["end"]))
                if len({p for p in rest if 0 <= p < _n}) < _n:
                    _need.add(x["rank"])
            _covered[_tj] = _need
    def _shard_complete(u, cov):
        if not u["complete"] and u["task_job"] in cov and u["job"] in cov[u["task_job"]] and not str(u.get("why_short") or "").startswith("raised"):
            return dict(u, complete=True, why_short=None, shard_of_covered_job=True)
        return u
    for tj, djs in ranks.items():
        # A task job is done when one of its launches completed: every rank of that launch ended cleanly by the
        # deadline, at its final step, with no recorded failure or exception. An agent that relaunches a crashed job
        # and gets it through has done the job; the crashed launch is listed under failed_launches. The ranks of one
        # launch share the daemon's job_key (torchrun's or the launcher's identity); a rank without one is its own.
        # A scripted run never relaunches, so a scripted job whose process exited non-zero is not done.
        launches = {}
        for d in djs:
            launches.setdefault((lookup.get(d) or {}).get("job_key") or f"rank-{d}", []).append(d)
        why_rank = lambda d: (f"rank {d} stopped at step {short[d]['reached_step']} of {short[d]['final_step']}" +
                              (f" raising {short[d]['exception']}" if short[d].get("exception") else "") if d in short
                              else f"rank {d} has a recorded failure" if d in failed
                              else f"rank {d} had not ended by the deadline")
        ok_ends, bad = [], []
        need_w = next((int(j["required_width"]) for j in jobs if j["job_id"] == tj and j.get("required_width")), None)
        # a silent-failure job (pkg-sf-zero-lr: Astra, 2026-11-02): only a launch served one of its successful_records trains the
        # model; one served a broken record (the zero-LR curve) exits zero at its last step and completes nothing
        sf = next(((j.get("silent_failure") or {}) for j in jobs if j["job_id"] == tj and j.get("silent_failure")), None)
        good = set((sf or {}).get("successful_records") or [])
        for key, ds in launches.items():
            served = {(lookup.get(d) or {}).get("record_id") for d in ds} - {None}
            if sf is not None and not (served & good):
                broken_completed.append({"task_job": tj, "ranks": sorted(ds), "record_id": next(iter(served), None),
                                         "start_wall": min(job_start(hello[d]) for d in ds),
                                         "end_wall": max((bye[d]["t_wall"] for d in ds if d in bye), default=None),
                                         "ran_to_end": all(d in bye and d not in short and d not in failed for d in ds),
                                         "broken_record": bool(served & set(sf.get("broken_records") or []))})
                bad.append(f"rank{'s' if len(ds) > 1 else ''} {', '.join(map(str, sorted(ds)))} served a broken record (no trained model)")
                continue
            if need_w is not None and len(ds) != need_w:
                # the wrong width: its ranks may all have run to the end, but the row's width is part of the deliverable, so it
                # completes nothing (Astra independent, 2026-11-02); its observed times are kept
                wrong_width.append({"task_job": tj, "required_width": need_w, "launched_width": len(ds), "ranks": sorted(ds),
                                    "record_id": next(((lookup.get(d) or {}).get("record_id") for d in ds if (lookup.get(d) or {}).get("record_id")), None),
                                    "start_wall": min(job_start(hello[d]) for d in ds),
                                    "end_wall": max((bye[d]["t_wall"] for d in ds if d in bye), default=None),
                                    "ran_to_end": all(d in bye and d not in short and d not in failed for d in ds)})
                bad.append(f"launched at width {len(ds)}, its row requires {need_w}")
                continue
            # a per-example rank short of the examples it asked for completes nothing unless it is a needed shard of a covering set (the
            # shard_of_covered_job rule, 11-04): the analysis 2026-11-05 21:51 MST, G-k2lp-haiku-A3's squat gsm8k, killed at 829 of
            # 1,319 and its relaunch at 83, read as done; an open pool's short instance is already in `short` above
            pe_short = [d for d in ds if d in per_example and not per_example[d]["complete"]
                        and not (tj in _covered and d in _covered[tj])]
            if pe_short:
                bad += [f"rank {d} {per_example[d]['why_short']}" for d in pe_short]
                continue
            if all(d in bye and d not in short and d not in failed and
                   (deadline_wall is None or bye[d]["t_wall"] <= deadline_wall) for d in ds):
                ok_ends.append((max(bye[d]["t_wall"] for d in ds), min(job_start(hello[d]) for d in ds), key))
                pe_counted += [dict(per_example[d], launch=key) for d in ds if d in per_example and not per_example[d]["complete"]]
            else:
                bad += [why_rank(d) for d in ds if d in short or d in failed or d not in bye or
                        (deadline_wall is not None and bye[d]["t_wall"] > deadline_wall)]
        if tj in exit_failed:
            ok_ends = []
            bad.insert(0, "its process exited non-zero")
        if tj in open_jobs:
            # the launches that did not finish their schedule, each with the step it reached and why: a goal's rule counts
            # only finished runs, and lists these beside them
            ok_keys = {k for _, _, k in ok_ends}
            env_d = {u["job"] for u in crashes}; agent_d = {u["job"] for u in agent_crashes}
            hk_d = {u["job"] for u in harness_kills}; unk_d = {u["job"] for u in unknown}
            for key, ds in launches.items():
                if key in ok_keys:
                    continue
                t_end = max((bye[d]["t_wall"] for d in ds if d in bye), default=None)
                cause = ("environment" if env_d & set(ds) else "agent" if agent_d & set(ds) else
                         "harness exit" if hk_d & set(ds) else "unknown origin" if unk_d & set(ds) else
                         "killed by the agent" if t_end is not None and any(a - 2 <= t_end <= b + 2 for a, b in kills) else
                         "recorded failure" if set(ds) & failed else
                         "exception (record not in the registry, so not classified)" if set(ds) & set(exc) else
                         "still running at the window's end" if any(d not in bye for d in ds) else "stopped short")
                open_unfinished.append({"task_job": tj, "start_wall": min(job_start(hello[d]) for d in ds), "end_wall": t_end,
                                        "rank_ids": sorted(ds), "reached_step": max((short.get(d) or {}).get("reached_step", -1) for d in ds),
                                        "final_step": max(((short.get(d) or {}).get("final_step") or -1) for d in ds),
                                        "cause": cause, "exception": next((short[d].get("exception") for d in ds if d in short and short[d].get("exception")), None)})
            for e1, s0, k in sorted(ok_ends, key=lambda x: x[1]):
                open_instances.append({"task_job": tj, "start_wall": s0, "end_wall": e1, "ranks": len(launches[k]),
                                       "rank_ids": sorted(launches[k]),
                                       # the instance's width (the GPUs its ranks ran on) and the record it was served
                                       "gpus": sorted({g for d in launches[k] for g in (rank_gpus.get(d) or ())}),
                                       "record_id": next(((lookup.get(d) or {}).get("record_id") for d in launches[k]
                                                          if (lookup.get(d) or {}).get("record_id")), None),
                                       "gpu_wall_s": sum(work.get(d, 0.0) for d in launches[k]),
                                       "solo_s": (e1 - s0) - max((share_delay.get(d, 0.0) for d in launches[k]), default=0.0)})
            if len(ok_ends) >= int(open_jobs[tj].get("min") or 0):
                done.add(tj)
            if ok_ends:
                t, t_start, k_done = max(ok_ends)
                last = t if last is None else max(last, t)
                done_at[tj], completed_run[tj], done_launch[tj] = t, t - t_start, t_start
                done_key[tj] = k_done
                solo_run[tj] = (t - t_start) - max((share_delay.get(d, 0.0) for d in launches[k_done]), default=0.0)
            if bad:
                failed_launches.append({"task_job": tj, "why": bad})
            continue
        if ok_ends:
            done.add(tj)
            # The job is done at its LAST completed launch, and its run time is that launch's own (the study,
            # 2026-10-21): an agent that re-prepares and re-runs a job it had completed (Q-prepP6-haiku-L0-nosub fixed its
            # assembler, regenerated all six scripts and re-ran them) is not done until the re-run is, since on a GPU the
            # earlier run's results came from the code it replaced. Earlier completed launches are superseded runs, listed
            # with their GPU work. A failed launch never sets the run time (Q-prepP10-haiku-L0's ema seeds).
            t, t_start, k_done = max(ok_ends)
            last = t if last is None else max(last, t)
            done_at[tj], completed_run[tj], done_launch[tj] = t, t - t_start, t_start
            done_key[tj] = k_done
            # the records the completing launch was served (the smoke-test rule in score_all asks whether a job's
            # unlisted early draw was followed by a completion on a listed record)
            done_records[tj] = sorted({(lookup.get(d) or {}).get("record_id") for d in launches[k_done]} - {None})
            # its solo run time: the wall run less the longest sharing delay of its ranks (the bounds use this, since
            # a job that shared a GPU ran stretched)
            solo_run[tj] = (t - t_start) - max((share_delay.get(d, 0.0) for d in launches[k_done]), default=0.0)
            for e1, s0, k in ok_ends:
                if k != k_done:
                    superseded.append({"task_job": tj, "start_wall": s0, "end_wall": e1, "ranks": len(launches[k]),
                                       "gpu_wall_s": sum(work.get(d, 0.0) for d in launches[k])})
            if bad:
                failed_launches.append({"task_job": tj, "why": bad})
        elif bad:
            not_completed.append({"task_job": tj, "why": bad})
    # an open job never launched still owes nothing when its min is 0
    for tj, inst in open_jobs.items():
        if tj not in ranks and int(inst.get("min") or 0) == 0:
            done.add(tj)
    # unneeded work (G4n's rejected setup's full runs): tied, checked like any rank, owed by nothing; its GPU work is listed
    unneeded_work = [{"task_job": tj, "ranks": sorted(ranks[tj]), "gpu_wall_s": round(sum(work.get(d, 0.0) for d in ranks[tj]), 3),
                      "completed": tj in done} for tj in sorted(unneeded) if tj in ranks]
    owed_done = {tj for tj in done if tj in owed}
    last = (max((done_at[tj] for tj in owed if tj in done_at), default=None) if owed <= done else None)
    return {"remaining": len(owed) - len(owed_done), "task_jobs": len(owed), "done": len(owed_done),
            "unneeded_work": unneeded_work, "reference_only_jobs": sorted(j["job_id"] for j in (task or {}).get("jobs") or []
                                                                        if j.get("reference_only")),
            "open_pool": bool(open_jobs), "open_instances": open_instances, "open_unfinished": open_unfinished,
            "unmapped_ranks": len(untied), "untied_ranks": untied, "records_not_listed": wrong,
            "wrong_width_launches": wrong_width, "broken_launches": broken_completed,
            "not_completed": not_completed, "failed_launches": failed_launches, "done_jobs": sorted(done),
            "environment_crashes": crashes, "agent_caused_crashes": agent_crashes, "killed_at_harness_exit": harness_kills,
            "unknown_origin_crashes": unknown, "cancelled_by_controller": cancelled, "briefed_failures": briefed,
            "unverified_ranks": unverified, "disengaged_ranks": disengaged_ranks,
            "per_example_ranks": [_shard_complete(per_example[d], _covered) for d in sorted(per_example)],
            # a short per_example rank in a launch the rule above counted complete, and whether that launch set its job's done time: a
            # fixed job's only, since an open pool's short instance completes nothing (none in the catalogue on 2026-11-03)
            "per_example_short_counted_complete": [dict(u, open_instance=u["task_job"] in open_jobs,
                                                        sets_done=done_key.get(u["task_job"]) == u["launch"]) for u in pe_counted],
            "done_at_wall": done_at, "completed_launch_wall_s": completed_run, "done_launch_wall": done_launch,
            "done_launch_records": done_records,
            "completed_launch_solo_s": solo_run, "share_delay_by_rank_s": share_delay,
            "superseded_runs": superseded,
            "all_done_wall": last}

class SpeedUnknown(Exception):
    pass

def run_speed(run_dir, ev=None):
    """The run's speed knob. It is a required input: read from run.json's
    inputs, cross-checked against the daemon's own daemon_start. A run with no recorded speed is
    refused rather than assumed to be 1, because a sped-up run is a smoke test, never a result."""
    rs = run_inputs(run_dir).get("speed")
    ds = next((e.get("speed") for e in (ev or []) if e.get("kind") == "daemon_start"), None)
    if rs is None and ds is None:
        raise SpeedUnknown(f"{run_dir}: no speed in run.json or the occupancy log")
    if rs is not None and ds is not None and abs(float(rs) - float(ds)) > 1e-9:
        raise SpeedUnknown(f"{run_dir}: run.json says speed {rs}, the daemon ran at {ds}")
    return float(rs if rs is not None else ds)

SMOKE_NOTE = ("SMOKE TEST: this run ran at speed {speed:g}, above 1. Its numbers check that the pipeline works; "
              "they are not a result and are never pooled with speed-1 runs (the study, 2026-10-20).")

def run_inputs(run_dir):
    """A run's inputs from run.json: task, harness, model, policy, seed, speed. isolation's run_one.py
    nests them under "inputs" with the policy under "scripted"; the bench writes them flat. Missing
    fields are None."""
    p = os.path.join(run_dir, "run.json")
    if not os.path.exists(p):
        return {}
    try:
        r = json.load(open(p))
    except (ValueError, OSError):
        return {}
    flat = dict(r.get("inputs") or {})
    for k in ("task", "harness", "model", "policy", "seed", "speed"):
        if flat.get(k) is None and r.get(k) is not None:
            flat[k] = r[k]
    if flat.get("policy") is None and isinstance(r.get("scripted"), dict):
        flat["policy"] = r["scripted"].get("policy")
    return flat

class RunNotFinished(Exception):
    pass

FINAL_STATUSES = {"completed", "killed-on-budget", "killed-on-time", "stopped", "crashed", "harness-error"}

def finished(run_dir):
    """A run directory is scoreable once it is over: its status.json carries an end time (run_one.py
    writes end_utc at teardown, and the bench does too) or a final status (gpu-tasks' synthetic
    directories carry status "completed" and no end time). A live run has neither and is not a
    result."""
    p = os.path.join(run_dir, "status.json")
    if not os.path.exists(p):
        return False
    try:
        st = json.load(open(p))
    except (ValueError, OSError):
        return False
    return bool(st.get("end_utc")) or st.get("status") in FINAL_STATUSES

def turn_vs_job(ev, task, turns):
    """The agent's median turn latency beside the median job length, and their ratio (the study, 2026-10-20, for
    the study's job-length concern: whether the jobs were too short for the agent is then a number in every run).
    A turn is one model response, from the end of its input (the previous tool result or user record) to its last
    block: activity()'s model turns, from the transcript. A job's length is its paced GPU duration as the box
    replayed it: the wall cost of one rank's units, summed, per launched task job (replay paces it from the job's
    recorded run). Medians over turns and over launched task jobs; None where there are none."""
    import statistics
    jobs_l = (task or {}).get("jobs") or []
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    per_gpu = {}
    for e in ev:
        if e["kind"] == "enqueue":
            k = (e["job"], e["gpu"])
            per_gpu[k] = per_gpu.get(k, 0.0) + float(e.get("wall_cost", 0.0))
    per_rank = {}                       # a rank's paced time on its own GPU (a daemon job spanning GPUs counts once)
    for (dj, g), v in per_gpu.items():
        per_rank[dj] = max(per_rank.get(dj, 0.0), v)
    by_tj = {}
    for dj, h in hello.items():
        tj = task_job_of(h, lk.get(dj), tie_jobs({"jobs": jobs_l, "unneeded_jobs": (task or {}).get("unneeded_jobs")})) or f"unmapped-{dj}"
        if dj in per_rank:
            by_tj[tj] = max(by_tj.get(tj, 0.0), per_rank[dj])   # one rank's paced GPU time; ranks pace alike
    tl = [b - a for a, b, *_ in turns if b >= a]
    mt = statistics.median(tl) if tl else None
    mj = statistics.median(by_tj.values()) if by_tj else None
    return {"median_turn_s": mt, "turns": len(tl), "median_job_gpu_s": mj, "jobs": len(by_tj),
            "turn_over_job": (mt / mj) if (mt is not None and mj) else None,
            "how": "turns: model responses from the transcript, input end to last block; jobs: paced GPU seconds of one "
                   "rank per launched task job, from the occupancy log"}

def analyse(run_dir, deadline_s=None, allow_live=False):
    if not allow_live and not finished(run_dir):
        raise RunNotFinished(f"{run_dir}: status.json has no end time and no final status, so the run has not finished; "
                             "pass --live to score it as it stands, labelled as live")
    ev = score_run.load_events(os.path.join(run_dir, "occupancy.jsonl"))
    task = json.load(open(os.path.join(run_dir, "task.json"))) if os.path.exists(os.path.join(run_dir, "task.json")) else {}
    attach_decisions(task, [e for e in ev if e["kind"] == "lookup"])   # H1's slot tie by candidate and seed
    st = json.load(open(os.path.join(run_dir, "status.json"))) if os.path.exists(os.path.join(run_dir, "status.json")) else {}
    start_ev = next((e for e in ev if e["kind"] == "daemon_start"), ev[0])
    gpus = start_ev.get("gpus") or task.get("box", {}).get("gpus")
    speed = run_speed(run_dir, ev)
    # work on t_wall throughout so the transcript's UTC times line up
    off = start_ev["t_wall"] - start_ev["t_mono"]
    ivs = {g: [[a + off, b + off] for a, b in l] for g, l in score_run.busy_intervals(ev).items()}
    # modelled jobs as busy on their GPU: merged into the lanes every reading below uses
    mb = with_modelled_busy(run_dir, ev, ivs, gpus, st)
    hello = [e for e in ev if e["kind"] == "job_hello"]
    first_launch = min((job_start(e) for e in hello), default=None)
    # gpu-mock logs an injected or recorded failure as kind "failure" (gpu_daemon.py, the
    # failure branch); job_bye carries no exit code, so a failure event is the only signal.
    failures = [e["t_wall"] for e in ev if e["kind"] == "failure"]
    recs = transcript_records(run_dir)
    # When the harness never exited (killed-on-time, killed-on-budget, stopped), run_one.py stopped the box
    # at teardown_wall, mid-call: the agent ended there, and a call left without a result ran until then
    # (gpu-tasks, 2026-10-20, from run_one.py's monitor and finish). No real run has been killed so yet.
    killed_at = st.get("teardown_wall") if not st.get("agent_end_wall") and not st.get("harness_end_wall") else None
    turns, tools, session_end = activity(recs, open_until=killed_at)
    agent_start = st.get("agent_start_wall") or (ts(recs[0]["timestamp"]) if recs else first_launch)
    agent_end = (st.get("agent_end_wall") or st.get("harness_end_wall") or killed_at) or session_end
    # A session the harness held open after the agent's final answer, because a background task never ended
    # (Q-prepP6-sonnet-L0-nosub: `while pgrep -f "quick/.*train_gpt"` matched its own shell): the agent ended with
    # its last turn, not at teardown
    held = held_open(run_dir, recs) if killed_at else None
    if held:
        agent_end = held["last_turn_wall"]
    lb = (task.get("stats") or {}).get("lower_bound_makespan_s")
    if deadline_s is not None:
        how = "horizon given on the command line"
    elif task.get("horizon_s"):
        deadline_s, how = float(task["horizon_s"]), "the task's horizon_s"
    elif task.get("deadline_s"):
        deadline_s, how = float(task["deadline_s"]), "the task's deadline_s (earlier design: stated in the brief)"
    elif lb:
        deadline_s, how = 1.5 * lb, "default horizon, 1.5 x the task's lower bound (the task has no horizon_s)"
    else:
        how = "no horizon and no lower bound: window ends at the last busy instant"
    w0 = agent_start
    w1 = agent_start + deadline_s / speed if deadline_s else max(b for l in ivs.values() for a, b in l)
    sub_kills = subagent_kill_calls(run_dir)
    # and the agent's own gpu-harness stopping a rejected candidate's job (solution's A1, 2026-10-28): its timeline's "stopping" rows
    kills = ([(a, b) for a, b, k, det in tools if KILL_RE.search(det or "")] + list(KILL_CALLS) + [(a, b) for a, b, _ag in sub_kills]
             + [(a, b) for a, b, _j in harness_stop_calls(run_dir)] + list(TIMEOUT_CALLS))
    # a rank still running when the box was stopped or killed was cut by the stop, not broken by the environment
    stop_at = st.get("teardown_wall") if st.get("status") in ("stopped", "killed-on-time", "killed-on-budget") else None
    ends = [x for x in (agent_start + deadline_s / speed if deadline_s else None, stop_at) if x is not None]
    # a harness can kill its agent's jobs only in a model run (claude -p's bounded background wait); a scripted agent
    # exits because its jobs did, so its end near a job's is the effect, not the cause
    model_run = run_inputs(run_dir).get("harness") not in (None, "scripted")
    # a scripted controller (plan.py's controller policy: the task's scripted_reference) stops its own ranks without a tool call
    ctl = (not model_run) and run_inputs(run_dir).get("policy") in ("controller", "filter_yield")
    rem = work_remaining(ev, task, w1 if deadline_s else None, scripted_exit_failures(recs), kills,
                         min(ends) if ends else None, st.get("harness_end_wall") if model_run else None,
                         agent_written(run_dir), st.get("harness_end_wall") if ctl else None,
                         own_stop_steps(run_dir, recs))
    pool_end = None
    if rem.get("open_pool"):
        # an open pool (goal packages): no job list says when the work is done; the agent's final report does, so the
        # window runs from its start to its end (capped at the horizon), the time after its last run included. A report
        # made before the work was done does not end it: the window then runs to the last completed instance's end
        #. An
        # instance the teardown cut is not work done, as in report_latency (G-passbar-astra-L0's was cut 0.1 s after its report)
        last_done = rem.get("all_done_wall")
        pool_end = agent_end if agent_end is None or last_done is None else max(agent_end, last_done)
        if pool_end is not None and pool_end < w1:
            w1, how = pool_end, how + ("; an open pool: the window ends at the agent's final report" if pool_end == agent_end else
                                       "; an open pool reported before its work was done: the window ends at the last instance's end")
    elif rem["remaining"] == 0 and rem["all_done_wall"] is not None and rem["all_done_wall"] < w1:
        # Every job completed before the deadline: capacity after that is not wasted, so the
        # window ends when the last job did (task work_remaining rule, jobs_not_completed).
        w1, how = rem["all_done_wall"], how + "; cut at the last job's completion, since no work remained"
    # The window never extends past the deadline. GPU work still queued or running when the deadline
    # falls is outside the run: a job killed there, or one the agent left running, is counted as
    # work remaining, and its busy time stops at the deadline (convention agreed with gpu-tasks,
    # 2026-10-20). Without a deadline, the window ends at the last busy instant.
    last_busy = max((b for l in ivs.values() for a, b in l), default=w0)
    past_deadline = max(0.0, last_busy - w1) if deadline_s else 0.0
    if not deadline_s and last_busy > w1:
        w1, how = last_busy, how + "; extended to the last busy instant"
    whole, partial_gs = whole_and_partial(ivs, gpus, w0, w1)
    win = w1 - w0
    whole_s = sum(b - a for a, b in whole)
    rdown = route_down_intervals(run_dir)
    route_down_s = sum(b - a for a, b in rdown)
    # a job is alive from its launcher's start to its launcher's exit when the box recorded them,
    # and from its first rank's start to its last rank's bye in any case
    alive = alive_intervals(ev) + launcher_intervals(run_dir)
    agent_parts, split, lane_parts = startup_split(ev, task, whole, agent_start, run_dir, lanes=ivs)
    attr = attribute(whole, first_launch, agent_end, failures, turns, tools, alive, rdown, agent_parts)
    # partial idle's causes: per lane, the jobs on that GPU and when each was alive (its launcher's start, or
    # its first rank's, to its last rank's bye), and per task job its launch and exit
    jobs_l = task.get("jobs") or []
    lk = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    byes = {e["job"]: e["t_wall"] for e in ev if e["kind"] == "job_bye"}
    jobs_t = tie_jobs(task) if task.get("unneeded_jobs") else jobs_l
    rank_tj = {e["job"]: (task_job_of(e, lk.get(e["job"]), jobs_t) or f"unmapped-{e['job']}") for e in hello}
    lane_alive = {}
    for e in ev:
        if e["kind"] == "enqueue" and e["job"] in rank_tj:
            tj = rank_tj[e["job"]]
            v = split.get(tj)
            a0 = v["alive_from_wall"] if v else job_start(next(h for h in hello if h["job"] == e["job"]))
            ex = max((byes.get(d) or w1) for d, t in rank_tj.items() if t == tj)
            if (a0, ex) not in lane_alive.setdefault(e["gpu"], []):
                lane_alive[e["gpu"]].append((a0, ex))
    launches = {}
    for d, tj in rank_tj.items():
        t = (split.get(tj) or {}).get("alive_from_wall") or job_start(next(h for h in hello if h["job"] == d))
        launches[tj] = min(launches.get(tj, t), t)
    done_at = {}
    for tj in set(rank_tj.values()):
        ds = [d for d, t in rank_tj.items() if t == tj]
        if all(d in byes for d in ds):
            done_at[tj] = max(byes[d] for d in ds)
    # a completed job is done at its last completed launch; its run time is that launch's own (work_remaining)
    done_at.update(rem.get("done_at_wall") or {})
    pattr = partial_attribution(ivs, gpus, w0, w1, lane_parts, lane_alive, launches, task, done_at, agent_end,
                                turns, tools, failures, rdown)
    # preparation packages (prep.py): per-job prep, ready-but-idle, waiting instead of preparing, the prep-aware bound
    # a job's width: the GPUs one launch of it used (a task job run several times, an open pool's instances, uses many
    # GPUs in all but no more than the widest launch at once)
    per_rank = {}
    for e in ev:
        if e["kind"] == "enqueue" and e["job"] in rank_tj:
            per_rank.setdefault(e["job"], set()).add(e["gpu"])
    key_of = {e["job"]: e.get("job_key") or f"rank-{e['job']}" for e in ev if e["kind"] == "lookup"}
    per_launch = {}
    for d, gs in per_rank.items():
        per_launch.setdefault((rank_tj[d], key_of.get(d, f"rank-{d}")), set()).update(gs)
    gpus_by_job = {}
    for (tj, _), gs in per_launch.items():
        gpus_by_job[tj] = min(int(gpus or len(gs)), max(gpus_by_job.get(tj, 0), len(gs)))
    # run times only for jobs that completed (work_remaining's done set): a killed or crashed job's launch-to-exit
    # is not its run time, and the prep-aware bound then says which jobs it lacks
    # each completed job's run time alone on its GPUs: the bounds place jobs as they would run unshared
    run_s = dict(rem.get("completed_launch_solo_s") or rem.get("completed_launch_wall_s") or {})
    # for preparation, a job's launch is the launch that completed it, so a re-preparation before a re-run is prep and
    # the prep end is the last preparation before that launch
    # (a job whose first launch completed it keeps its launch time from the launcher's start)
    prep_launches = dict(launches)
    first_rank = {}
    for e in hello:
        tj = rank_tj.get(e["job"])
        first_rank[tj] = min(first_rank.get(tj, job_start(e)), job_start(e))
    for tj, t_done in (rem.get("done_launch_wall") or {}).items():
        if tj in first_rank and t_done > first_rank[tj] + 1.0:
            prep_launches[tj] = t_done
    prep_r = prep_mod.analyse(run_dir, task, recs, ivs, gpus, w0, w1, speed, prep_launches, done_at, gpus_by_job, run_s,
                              open_until=killed_at, wait_re=WAIT_RE)
    return {
        # modelled-path jobs counted busy on their GPU: recorded seconds, and each job
        "modelled_job_gpu_s": (mb.get("modelled_job_gpu_wall_s") or 0.0) * speed if "error" not in mb else None,
        "modelled_job_busy": {"counted_as_busy": mb.get("counted_as_busy"), "error": mb.get("error"),
                              "added_busy_recorded_s": (mb.get("added_busy_wall_s") or 0.0) * speed,
                              "unassigned_recorded_s": (mb.get("unassigned_wall_s") or 0.0) * speed,
                              "jobs": [{"rank": x["rank"], "gpus": x["gpus"], "start_s": (x["start_wall"] - w0) * speed,
                                        "end_s": ((x["end_wall"] - w0) * speed) if x["end_wall"] else None, "argv": x["argv"],
                                        "gpu_from": x.get("gpu_how")}
                                       for x in (mb.get("items") or [])],
                              "how": "a rank with lookup match none and no enqueue, busy from its process start to its bye on the GPU "
                                     "its launching Bash call's CUDA_VISIBLE_DEVICES names (idle.modelled_job_busy)"},
        "window": {"from": "agent start", "to": how, "wall_s": win, "recorded_s": win * speed,
                   "speed": speed, "gpus": gpus},
        "smoke_test": speed != 1.0,
        "disengaged_overlap": disengaged_overlap(ev, whole, (agent_end - w0) if agent_end else win, speed),
        "whole_box_idle_share_of_gpu_seconds": whole_s * gpus / (gpus * win) if win else None,
        "whole_box_idle_share_of_wall": whole_s / win if win else None,
        "partial_idle_share_of_gpu_seconds": partial_gs / (gpus * win) if win else None,
        "time_to_first_launch_recorded_s": (first_launch - w0) * speed if first_launch else None,
        # two readings, kept apart: the cause after_agent_ended (nothing of the agent's or a job's
        # was running), and all whole-box idle after the agent's end, which also counts a job's
        # host phase that happens after the agent left
        "idle_after_agent_stopped_recorded_s": attr.get("after_agent_ended", 0.0) * speed,
        "whole_box_idle_after_agent_end_recorded_s": (sum(max(0.0, b - max(a, agent_end)) for a, b in whole) * speed
                                                      if agent_end is not None else None),
        "whole_box_idle_by_cause_recorded_s": {k: v * speed for k, v in sorted(attr.items(), key=lambda kv: -kv[1])},
        "partial_idle_recorded_gpu_s": partial_gs * speed,
        "partial_idle_by_cause_recorded_gpu_s": {k: v * speed for k, v in sorted(pattr.items(), key=lambda kv: -kv[1])},
        "startup_split_by_job_recorded_s": {tj: {"startup_s": v["startup_s"] * speed,
                                                 "whole_box_idle_in_startup_s": v["startup_whole_box_idle_s"] * speed,
                                                 "startup_not_overlapped": v["startup_not_overlapped_s"] * speed,
                                                 "partial_startup_not_overlapped_gpu_s": v["partial_startup_not_overlapped_gpu_s"] * speed,
                                                 "lane_hideable_s": v["lane_hideable_s"] * speed, "gpus": v["gpus"],
                                                 "launchable_from_s": None if v["launchable_from_wall"] is None
                                                 else (v["launchable_from_wall"] - w0) * speed,
                                                 "unmapped": v["unmapped"]}
                                            for tj, v in sorted(split.items(), key=lambda kv: kv[1]["alive_from_wall"])},
        "work_remaining_at_window_end": rem,
        "route_down_wall_s": route_down_s,
        "status": st.get("status"), "status_reason": st.get("reason"),
        "harness_result": st.get("harness_result"),
        "gpu_work_past_deadline_recorded_s": past_deadline * speed,
        "transcript_records": len(recs), "model_turns": len(turns), "tool_calls": len(tools),
        "turn_vs_job": turn_vs_job(ev, task, turns),
        "prep": prep_r,
        "makespan_from_agent_start_recorded_s": ((pool_end - w0) * speed if rem.get("open_pool") and pool_end else
                                                 (rem["all_done_wall"] - w0) * speed if rem["all_done_wall"] else None),
        # an open pool: its instances, and the last one's end beside the report
        "open_pool": {"instances": [dict(x, start_s=(x["start_wall"] - w0) * speed, end_s=(x["end_wall"] - w0) * speed,
                                         gpu_s=x["gpu_wall_s"] * speed, width=len(x.get("gpus") or ()) or None)
                                    for x in rem.get("open_instances") or []],
                      "unfinished": [dict(x, start_s=(x["start_wall"] - w0) * speed,
                                          end_s=None if x["end_wall"] is None else (x["end_wall"] - w0) * speed)
                                     for x in rem.get("open_unfinished") or []],
                      "last_instance_end_s": (rem["all_done_wall"] - w0) * speed if rem.get("all_done_wall") else None,
                      "report_s": (agent_end - w0) * speed if agent_end else None} if rem.get("open_pool") else None,
        # completed launches of a job that a later completed launch superseded, with their GPU work (the enqueued
        # wall cost of their ranks): GPU time the job needed again
        # jobs that shared a GPU with another (gpu-mock's sharers > 1) and the time each waited behind it
        "gpu_oversubscription": sorted(({"task_job": rank_tj.get(d), "rank": d, "share_delay_s": v * speed}
                                        for d, v in (rem.get("share_delay_by_rank_s") or {}).items() if v > 1.0),
                                       key=lambda x: -x["share_delay_s"]),
        "superseded_runs": [{"task_job": x["task_job"], "start_s": (x["start_wall"] - w0) * speed,
                             "end_s": (x["end_wall"] - w0) * speed, "ranks": x["ranks"], "gpu_s": x["gpu_wall_s"] * speed}
                            for x in rem.get("superseded_runs") or []],
        "held_open_by_background_task": None if not held else dict(held, last_turn_s=(held["last_turn_wall"] - w0) * speed,
                                                                    held_s=(killed_at - held["last_turn_wall"]) * speed),
        "agent_end_source": ("the agent's last turn: the harness held the session open to teardown for a background task "
                             "that never ended (held_open_by_background_task)" if held else
                             "status.json agent_end_wall" if st.get("agent_end_wall") else
                             "status.json harness_end_wall" if st.get("harness_end_wall") else
                             "status.json teardown_wall (the box was stopped with the harness running)" if killed_at else
                             "last transcript record"),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--deadline-s", "--horizon-s", dest="deadline_s", type=float, help="the horizon, recorded seconds from the agent's start")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--live", action="store_true", help="score a run that has not finished, labelled as live")
    a = ap.parse_args()
    try:
        r = analyse(a.run_dir, a.deadline_s, allow_live=a.live)
    except (RunNotFinished, SpeedUnknown) as e:
        sys.exit(str(e))
    r["live"] = not finished(a.run_dir)
    if a.json:
        print(json.dumps(r, indent=1)); return
    w = r["window"]
    hr = r.get("harness_result") or {}
    if hr.get("is_error"):
        # Claude Code's result record says subtype "success" even when is_error is true; isolation
        # keys on is_error (2026-10-20), and so does this line
        print(f"HARNESS ERROR: {r.get('status_reason') or (hr.get('result') or '')[:200]}")
    if r.get("smoke_test"):
        print(SMOKE_NOTE.format(speed=w["speed"]))
    if r.get("live"):
        print("LIVE: this run has not finished; every number below is provisional")
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    print(f"window          {w['recorded_s']:.0f} s recorded ({w['wall_s']:.1f} s wall), agent start to {w['to']}")
    print(f"whole-box idle  {pct(r['whole_box_idle_share_of_gpu_seconds'])} of GPU-seconds ({pct(r['whole_box_idle_share_of_wall'])} of wall time)")
    print(f"partial idle    {pct(r['partial_idle_share_of_gpu_seconds'])} of GPU-seconds ({r['partial_idle_recorded_gpu_s']:.0f} GPU-s)")
    for k, v in r["partial_idle_by_cause_recorded_gpu_s"].items():
        print(f"  {k:<22} {v:8.0f} GPU-s")
    t = r["time_to_first_launch_recorded_s"]
    print(f"first launch    {'never' if t is None else f'{t:.0f} s'} after the agent started")
    wa = r["whole_box_idle_after_agent_end_recorded_s"]
    print(f"after stop      {r['idle_after_agent_stopped_recorded_s']:.0f} s idle after the agent ended its session ({r['agent_end_source']})"
          + (f"; {wa:.0f} s whole-box idle after its end, including job host phases" if wa is not None and abs(wa - r['idle_after_agent_stopped_recorded_s']) > 0.5 else ""))
    for k, v in r["whole_box_idle_by_cause_recorded_s"].items():
        print(f"  {k:<20} {v:8.0f} s")
    if r.get("route_down_wall_s"):
        print(f"route down      {r['route_down_wall_s']:.0f} s wall when the inference route was failing (edge forwarder log)")
    wr = r["work_remaining_at_window_end"]
    if wr["remaining"] is None:
        print("work remaining  unknown (task.json lists no jobs)")
    else:
        print(f"work remaining  {wr['remaining']} of {wr['task_jobs']} task jobs not completed" +
              (f"; {wr['unmapped_ranks']} ranks tie to no task job "
               f"({sum(u['gpu_wall_s'] for u in wr['untied_ranks']):.0f} GPU-s of work), so the run goes to review"
               if wr["unmapped_ranks"] else ""))
        for u in wr["untied_ranks"] or []:
            print(f"  untied rank {u['job']}: {u['gpu_wall_s']:.0f} GPU-s, record {u['record_id']}: {u['argv'][:160]}")
        for u in wr.get("not_completed") or []:
            print(f"  not completed  {u['task_job']}: {'; '.join(u['why'])[:200]}")
        for u in wr.get("failed_launches") or []:
            print(f"  relaunched     {u['task_job']}: done on a later launch after {'; '.join(u['why'])[:160]}")
        ov = r.get("gpu_oversubscription") or []
        if ov:
            print(f"  shared GPUs    {len(ov)} job(s) waited behind another job on their GPU, {sum(x['share_delay_s'] for x in ov):.0f} s in all: "
                  + ", ".join(f"{x['task_job']} {x['share_delay_s']:.0f} s" for x in ov)[:200])
        rep = r.get("superseded_runs") or []
        if rep:
            print(f"  superseded     {len(rep)} completed launch(es) re-run later, {sum(x['gpu_s'] for x in rep):.0f} GPU-s: " +
                  ", ".join(f"{x['task_job']} {x['start_s']:.0f}-{x['end_s']:.0f} s" for x in rep)[:200])
        for u in wr.get("agent_caused_crashes") or []:
            print(f"  agent's crash  rank {u['job']} of {u['task_job']}: {u.get('exception', '')[:120]} ({u.get('why')}; a "
                  f"real H100 run would crash too)" + (f", relaunched {u['relaunched_after_s']:.0f} s later"
                                                       if u.get("relaunched_after_s") is not None else ""))
        for u in wr.get("killed_at_harness_exit") or []:
            print(f"  HARNESS KILL   rank {u['job']} of {u['task_job']} stopped at step {u['reached_step']} of {u['final_step']} "
                  f"as the harness exited: claude -p ended and took its background jobs with it")
        for u in wr.get("environment_crashes") or []:
            print(f"  ENVIRONMENT CRASH rank {u['job']} of {u['task_job']}: an exact-match job stopped at step "
                  f"{u['reached_step']} of {u['final_step']}, {u['ended_after_s']:.1f} s after its start, with no "
                  f"recorded failure and no kill: an H100 would have run it to the end")
        for u in wr["records_not_listed"] or []:
            print(f"  rank {u['job']} of {u['task_job']} was served {u['record_id']} ({u['draw']}), "
                  f"which the task job does not list")
    print(f"transcript      {r['transcript_records']} records, {r['model_turns']} model turns, {r['tool_calls']} tool calls")
    tv = r.get("turn_vs_job") or {}
    if tv.get("median_turn_s") is not None and tv.get("median_job_gpu_s"):
        print(f"turn vs job     median turn {tv['median_turn_s']:.1f} s ({tv['turns']} turns), median job {tv['median_job_gpu_s']:.0f} s "
              f"of paced GPU work ({tv['jobs']} jobs): ratio {tv['turn_over_job']:.3f}")
    pr = r.get("prep")
    if pr:
        f = lambda x: "-" if x is None else f"{x:.0f}"
        print(f"prep            {pr['jobs_needing_prep']} jobs need preparation; recorded s from agent start" +
              (f"; {len(pr['prep_unobserved_jobs'])} launched with no observed preparation (left out of the sweeps)"
               if pr.get("prep_unobserved_jobs") else ""))
        print(f"  ready but idle               {pr['ready_but_idle_recorded_gpu_s']:8.0f} GPU-s "
              f"({pr['ready_but_idle_usable_recorded_gpu_s']:.0f} usable by the ready jobs; "
              f"{pr['ready_but_idle_whole_box_recorded_s']:.0f} s whole-box)")
        print(f"  waiting instead of preparing {pr['waiting_instead_of_preparing_recorded_gpu_s']:8.0f} GPU-s "
              f"({pr['waiting_instead_of_preparing_usable_recorded_gpu_s']:.0f} usable by the unprepared jobs; "
              f"{pr['waiting_instead_of_preparing_whole_box_recorded_s']:.0f} s whole-box)")
        print(f"  makespan {f(r.get('makespan_from_agent_start_recorded_s'))} s from agent start; prep-aware bound "
              f"{f(pr['prep_aware_bound_recorded_s'])} s" + ("" if pr["prep_aware_bound_recorded_s"] is not None
                                                             else f" ({pr['prep_aware_bound_why']})"))
        ab = pr["agent_busy_vs_job"]
        if ab["ratio"] is not None:
            print(f"  agent busy vs job            median prep busy {ab['median_prep_busy_s']:.0f} s per job, "
                  f"median job {ab['median_job_s']:.0f} s: ratio {ab['ratio']:.3f}")
        print(f"  {'job':<28} {'prep start':>10} {'prep end':>9} {'ready':>7} {'launch':>7} {'done':>7} {'wait':>6} {'busy':>6}  end set by")
        for jid, v in pr["by_job_recorded_s"].items():
            print(f"  {jid[:28]:<28} {f(v['prep_start']):>10} {f(v['prep_end']):>9} {f(v['ready']):>7} {f(v['launch']):>7} "
                  f"{f(v['done']):>7} {f(v['ready_wait']):>6} {f(v['prep_busy']):>6}  {v['prep_end_set_by'] or '-'}")

if __name__ == "__main__":
    main()
