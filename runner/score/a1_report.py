# the runner: the gpu-harness (A1) additions to the team report, beside team_report.py.
# Reads a collected run on the controller: the harness's own record (workspace/home/.gpuh/runs/<stamp>/ board.json, events.jsonl,
# timeline.jsonl, captured by run_one's workspace capture), the occupancy log, and the transcript.
#   a. main's peak context (team-report's 2) and every compact_boundary (stream system events)
#   b. the harness's cancels, and (b2) its holds (A3): each GPU it kept free, for how long, and what it released to;
#      the cancels: each job the timeline shows STOPPING/CANCELLED, its daemon rank (board pid = job_hello pid, else the
#      rank started on that GPU closest to the job's start), and whether that rank's bye came within CANCEL_OK_S of the stop
#   c. launches outside the harness: task-job ranks (job_hello) whose pid is not a harness job's and that no harness job's start
#      matches on its GPU within LAUNCH_TOL_S; and the agent's own sleeps >= 60 s and polling loops (team-report's 4)
#   d. G1 only: the longest job's start against the first end of its method's calibration (the harness's first finished job of
#      that method), and the makespan against LPT with known lengths from g1_replay.py's lpt()
# usage: python3 -I a1_report.py <run_id> [--json]
# (in the repository from 2026-10-29: the /tmp/c1r/a1-report.py copy was lost when the controller's reboot cleared /tmp; recovered from the runner's session transcript)
import glob, json, os, re, subprocess, sys, datetime
rid = sys.argv[1]; D = os.path.expanduser(f"~/computewise-runs/{rid}")
EXP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CANCEL_OK_S, LAUNCH_TOL_S = 30.0, 10.0
J = lambda p: json.load(open(p)) if os.path.exists(p) else {}
st = J(D + "/status.json"); a0 = st.get("agent_start_wall") or 0
task = J(D + "/task.json"); out = {"run": rid, "task": task.get("task_id")}

# the harness record
runs = sorted(glob.glob(D + "/workspace/home/.gpuh/runs/*/"))
out["gpuh_runs"] = [os.path.basename(r.rstrip("/")) for r in runs]
board, tl, ev = {}, [], []
if runs:
    R = runs[-1]
    board = J(R + "board.json")
    tl = [json.loads(l) for l in open(R + "timeline.jsonl")] if os.path.exists(R + "timeline.jsonl") else []
    ev = [json.loads(l) for l in open(R + "events.jsonl")] if os.path.exists(R + "events.jsonl") else []
out["gpuh_record"] = {"board": bool(board), "timeline_rows": len(tl), "events": len(ev)} if runs else "missing: no workspace/home/.gpuh/runs/"
bjobs = (board.get("jobs") or {}) if isinstance(board.get("jobs"), dict) else {j.get("id"): j for j in board.get("jobs") or []}

# occupancy
hello, bye = {}, {}
for l in open(D + "/occupancy.jsonl", errors="replace"):
    if '"job_hello"' in l or '"job_bye"' in l:
        e = json.loads(l); (hello if e["kind"] == "job_hello" else bye)[e["job"]] = e
gpu_of = {}
for l in open(D + "/occupancy.jsonl", errors="replace"):
    if '"enqueue"' in l:
        e = json.loads(l)
        if e.get("gpu", -1) >= 0: gpu_of.setdefault(e["job"], e["gpu"])
start = lambda r: (hello[r].get("t_start") or hello[r]["t_wall"])
pid_rank = {h.get("pid"): r for r, h in hello.items()}

# per harness job: every start (RUNNING row) and its rank. A job the harness retried has one RUNNING row per attempt, and only
# the attempt that reached the GPU has a daemon rank (G-kvrep-sonnet-A3, 2026-10-29: every first attempt failed in seconds on a
# wrong workdir, rc 127, before the daemon saw it, and the retries are the real launches; matching each job's first RUNNING row
# only read all 40 retries as launches outside the harness). A rank matched by any attempt is the harness's
runs_of = {}
for x in tl:
    if x.get("state") in ("running", "RUNNING"):
        runs_of.setdefault(x["job"], []).append(x)
first = {j: v[0] for j, v in runs_of.items()}
def _match(f):
    p = f.get("pid")
    if p in pid_rank:
        return pid_rank[p], "pid"
    c = [r for r in hello if gpu_of.get(r) == f.get("gpu") and abs(start(r) - f["t"]) <= LAUNCH_TOL_S]
    return (min(c, key=lambda r: abs(start(r) - f["t"])), "gpu and start") if c else (None, "no rank matched")
def rank_of(jid, before=None):
    """The daemon rank of a harness job: the board's pid, else the attempt in force at `before` (the last RUNNING row at or
    before it; a cancel stops the attempt then running), else the latest attempt that matches a rank."""
    p = (bjobs.get(jid) or {}).get("pid")
    if p in pid_rank:
        return pid_rank[p], "pid"
    rows = runs_of.get(jid) or []
    if not rows:
        return None, "never started"
    if before is not None:
        rows = [x for x in rows if x["t"] <= before] or rows
    for f in reversed(rows):
        r, how = _match(f)
        if r is not None:
            return r, how
    return None, "no rank matched"
harness_ranks = {}
for jid, rows in runs_of.items():
    for f in rows:
        r, how = _match(f)
        if r is not None:
            harness_ranks[r] = jid

# b. cancels
canc = []
for x in tl:
    if x.get("state") in ("stopping", "STOPPING"):
        r, how = rank_of(x["job"], before=x["t"])
        b = bye.get(r) if r is not None else None
        dt = (b["t_wall"] - x["t"]) if b else None
        canc.append({"job": x["job"], "at_s": round(x["t"] - a0), "why": x.get("why"), "rank": r, "matched_by": how,
                     "rank_ended_after_s": round(dt, 1) if dt is not None else None, "stopped": dt is not None and dt <= CANCEL_OK_S})
dropped = sum(1 for x in tl if x.get("state") in ("cancelled", "CANCELLED"))
out["b_cancels"] = {"stopping": len(canc), "stopped_within_%ds" % CANCEL_OK_S: sum(1 for c in canc if c["stopped"]),
                    "not_stopped": [c for c in canc if not c["stopped"]], "dropped_before_start": dropped}

# b2. holds: a timeline row in state `hold` keeps a GPU free for a check whose judgement gates the
# longest chain; it lasts until the next row on that GPU that is not a hold, and that row's state and job say what it released to
holds = []
for h in (x for x in tl if x.get("state") == "hold"):
    nxt = next((x for x in tl if x["t"] > h["t"] and x.get("gpu") == h.get("gpu") and x.get("state") != "hold"), None)
    holds.append({"gpu": h.get("gpu"), "at_s": round(h["t"] - a0), "for_job": h.get("job"),
                  "held_s": round(nxt["t"] - h["t"], 1) if nxt else None, "released_to": (nxt or {}).get("state"),
                  "next_job": (nxt or {}).get("job")})
_hs = [x["held_s"] for x in holds if x["held_s"] is not None]
out["b_holds"] = {"holds": len(holds), "held_s_total": round(sum(_hs), 1), "held_s_max": max(_hs) if _hs else None,
                  "unreleased": sum(1 for x in holds if x["held_s"] is None), "items": holds[:15]}

# c. launches outside the harness
outside = [{"rank": r, "start_s": round(start(r) - a0), "gpu": gpu_of.get(r), "argv": " ".join(map(str, h.get("argv") or []))[:90]}
           for r, h in sorted(hello.items()) if r not in harness_ranks and gpu_of.get(r) is not None]
out["c_outside_harness"] = {"ranks": len(outside), "items": outside[:15]}
tr = subprocess.run([sys.executable, "-I", os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_report.py"), rid, "--json"], capture_output=True, text=True)
try:
    t = json.loads(tr.stdout)
    out["a_main_peak_context_tokens"] = t["2_main_load"]["main_peak_context_tokens"]
    out["c_waiting"] = {k: v for k, v in t["4_waiting"].items() if any(v.values())} or "no sleeps >= 60 s, no polls"
except (ValueError, KeyError):
    out["a_main_peak_context_tokens"] = None
comp = []
for l in open(D + "/transcript/stream.jsonl", errors="replace"):
    if "compact_boundary" in l:
        try: x = json.loads(l)
        except ValueError: continue
        if x.get("subtype") == "compact_boundary" or x.get("type") == "compact_boundary":
            comp.append({"parent": x.get("parent_tool_use_id"), "meta": x.get("compact_metadata")})
out["a_compact_boundaries"] = comp

# d. G1
if task.get("task_id") == "mls-kv-report" and first:
    rec_len = {}
    for j in task.get("jobs") or []:
        rec_len[j["job_id"]] = ((j.get("disclosure") or {}).get("estimate") or {}).get("median_s")
    # the harness names a job <candidate>_<workload> (single underscores; its timeline carries the candidate), the task
    # <method>--<workload-with-hyphens>: map by the row's own candidate field, not by splitting the name
    cand = {x["job"]: x.get("candidate") for x in tl if x.get("candidate")}
    def tj_of(jid):
        c = cand.get(jid)
        if not c or not jid.startswith(c + "_"):
            return None
        return f"{c}--{jid[len(c) + 1:].replace('_', '-')}"
    lj = max(first, key=lambda j: rec_len.get(tj_of(j)) or 0)
    meth = cand.get(lj)
    ends = sorted(x["t"] for x in tl if cand.get(x["job"]) == meth and x["job"] != lj and x.get("state") in ("done", "DONE"))
    sys.path.insert(0, os.path.join(EXP, "solution", "skills", "gpu-harness"))
    lpt = subprocess.run([sys.executable, "-I", os.path.join(EXP, "solution", "probes", "harness-replay", "g1_replay.py"), EXP],
                         capture_output=True, text=True).stdout
    m = re.search(r"LPT with known lengths ([\d.]+) h", lpt)
    sc = J(D + "/score.json"); ms = (sc.get("hindsight") or {}).get("actual_s")
    # the attempt that reached the GPU, not a first attempt that failed before the daemon saw it (G-kvrep-sonnet-A3: rc 127 at 54 s,
    # the real launch later); the last RUNNING row that matches a daemon rank, else the first
    _lr = next((f for f in reversed(runs_of.get(lj) or []) if _match(f)[0] is not None), first[lj])
    out["d_g1"] = {"longest_job": lj, "longest_started_s": round(_lr["t"] - a0),
                   "longest_attempts": len(runs_of.get(lj) or []),
                   "first_end_of_its_method_s": round(ends[0] - a0) if ends else None,
                   "lpt_known_h": float(m.group(1)) if m else None, "makespan_h": round(ms / 3600, 2) if ms else None,
                   "makespan_over_lpt": round(ms / 3600 / float(m.group(1)), 3) if (ms and m) else None}
print(json.dumps(out, indent=1, default=str) if "--json" in sys.argv else json.dumps(out, default=str))
