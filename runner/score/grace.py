"""The work-done grace: when every owed task job is done and
the agent's own background task holds the session, stdin stays open up to 1,800 s for the agent's next turn. Labels, never a
verdict: screen pools runs before and after the grace by default and splits them on request (--split grace).

usage: python3 -I grace.py <run_dir> [--json]

lifecycle_sha256       run.json claude_code.lifecycle_sha256 (first 16 hex digits; None for a harness with no lifecycle)
grace_outcome          from status.json stdin_close_reason's `[work_done_grace outcome=<o> waited_s=<s> grace_s=<s>]`:
                       answered_in_grace (a turn came within the grace), closed_after_grace (it ran out),
                       task_ended_no_answer (the background task ended with no turn after it); None when no grace ran
answered_in_grace / closed_after_grace   booleans of the same; grace_waited_s, grace_s
launched_in_grace      task-job launches (a job_hello the scorer ties to a task job, owed or not) that started after the last
                       owed job's end and before the session's end (stdin_closed_wall, else harness_end_wall, else teardown):
                       the work the grace window let the agent start. Each with its task job, start (s from agent start) and
                       GPU-seconds (its enqueued unit costs, times speed). They count in the run's figures like any launch;
                       this only says how often the window mattered. 0 when nothing launched; None when the run has no
                       occupancy log or no owed job finished.
"""
import argparse, gzip, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import idle

GRACE_RE = re.compile(r"\[work_done_grace outcome=(\w+) waited_s=([\d.]+) grace_s=([\d.]+)\]")

def _load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return {}

def scan(run_dir, all_done_wall=None):
    """all_done_wall: the scorer's own last-owed-job end (idle.work_remaining's all_done_wall); score_all passes it. Standalone,
    it is read from the run's score.json, else computed by work_remaining on the whole occupancy log."""
    rj, st = _load(os.path.join(run_dir, "run.json")), _load(os.path.join(run_dir, "status.json"))
    lc = ((rj.get("claude_code") or {}).get("lifecycle_sha256") or "")[:16] or None
    m = GRACE_RE.search(str(st.get("stdin_close_reason") or ""))
    out = {"lifecycle_sha256": lc, "grace_outcome": m.group(1) if m else None,
           "answered_in_grace": (m.group(1) == "answered_in_grace") if m else False,
           "closed_after_grace": (m.group(1) == "closed_after_grace") if m else False,
           "grace_waited_s": float(m.group(2)) if m else None, "grace_s": float(m.group(3)) if m else None,
           "launched_in_grace": None, "launched_in_grace_gpu_s": None, "launched_in_grace_items": []}
    p = os.path.join(run_dir, "occupancy.jsonl")
    occ = p if os.path.exists(p) else (p + ".gz" if os.path.exists(p + ".gz") else None)
    task = _load(os.path.join(run_dir, "task.json"))
    if not occ or not idle.owed_jobs(task):
        return out
    done = all_done_wall
    if done is None:
        done = ((_load(os.path.join(run_dir, "score.json")).get("idle") or {}).get("work_remaining_at_window_end") or {}).get("all_done_wall")
    if done is None:
        import score_run
        try:
            if occ.endswith(".gz"):
                ev = [json.loads(l) for l in gzip.open(occ, "rt", errors="replace") if l.strip().startswith("{")]
                ev = score_run.split_trials(ev)
            else:
                ev = score_run.load_events(occ)
            done = idle.work_remaining(ev, task).get("all_done_wall")
        except Exception:
            done = None
    if done is None:
        return out
    hello, look, cost = {}, {}, {}
    with (gzip.open(occ, "rt", errors="replace") if occ.endswith(".gz") else open(occ, errors="replace")) as fh:
        for line in fh:
            if not any(k in line for k in ('"job_hello"', '"lookup"', '"enqueue"')):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            k = e.get("kind")
            if k == "job_hello":
                hello.setdefault(e["job"], e)
            elif k == "lookup":
                look.setdefault(e["job"], e)
            elif k == "enqueue":
                cost[e["job"]] = cost.get(e["job"], 0.0) + float(e.get("wall_cost") or 0.0)
    jobs = idle.tie_jobs(task)
    idle.attach_decisions(task, list(look.values()))
    tj = {r: idle.task_job_of(h, look.get(r), jobs) for r, h in hello.items()}
    end = st.get("stdin_closed_wall") or st.get("harness_end_wall") or st.get("teardown_wall")
    a0 = st.get("agent_start_wall")
    speed = float(idle.run_inputs(run_dir).get("speed") or 1.0)
    items = [{"rank": r, "task_job": tj[r], "start_s": round((idle.job_start(h) - a0) * speed, 1) if a0 else None,
              "gpu_s": round(cost.get(r, 0.0) * speed, 1)}
             for r, h in sorted(hello.items()) if tj.get(r) and idle.job_start(h) > done and (end is None or idle.job_start(h) <= end)]
    out.update(launched_in_grace=len(items), launched_in_grace_gpu_s=round(sum(i["gpu_s"] for i in items), 1),
               launched_in_grace_items=items[:20], last_owed_end_s=round((done - a0) * speed, 1) if a0 else None,
               session_end_s=round((end - a0) * speed, 1) if (a0 and end) else None)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = scan(a.run_dir)
    if a.json:
        print(json.dumps(r, indent=1))
        return
    print(f"lifecycle {r['lifecycle_sha256']}  grace {r['grace_outcome']} (waited {r['grace_waited_s']} of {r['grace_s']} s)  "
          f"launched in grace {r['launched_in_grace']} ({r['launched_in_grace_gpu_s']} GPU-s)")

if __name__ == "__main__":
    sys.exit(main())
