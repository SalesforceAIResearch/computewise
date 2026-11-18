"""`gpuh`: the one command agents call.

    gpuh start PLAN.json [--gpus 0,1,2,3] [--dir RUN_DIR]   validate the plan, start the harness, return at once
    gpuh wait --for AGENT [--timeout S]                     block until AGENT has an event that needs it; print them
    gpuh status [-v]                                        GPUs, candidates, and (with -v) the queue and its reasons
    gpuh pass CHECK [CHECK ...]                             a check passed: open the work gated on it
    gpuh reject CAND --why TEXT [--evidence JOB,...]        rule a candidate out: its whole subtree is cancelled
    gpuh decide CAND --verdict V --why TEXT [--evidence ..]  record a candidate's verdict
    gpuh add FRAGMENT.json                                  add jobs (and candidates) to the running plan
    gpuh cancel JOB [JOB ...] [--why TEXT]                  stop or drop jobs
    gpuh retry JOB [--command CMD]                          run a failed, died or cancelled job again
    gpuh estimate JOB SECONDS                               correct a waiting job's prior length
    gpuh results                                            every finished job's result line, by candidate
    gpuh log JOB [-n N]                                     the last lines of a job's log
    gpuh events [--for AGENT] [--all]                       the event log
    gpuh export [--html PATH] [--json]                      the run's timeline
    gpuh finish [--why TEXT]                                stop what is left and end every wait
    gpuh check PLAN.json                                    validate a plan without starting anything

Every agent names itself with --as NAME (or GPUH_AGENT); commands it sends are attributed to it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

from . import __version__
from . import plan as P
from .daemon import TICK_S
from .events import duration
from .launcher import detect_gpus
from .ledger import Store, current_run, gpuh_home, set_current

REMIND_OPEN_S = (120.0, 600.0, 1800.0, 3600.0)  # a decision left open is shown again after these, in turn


def _last(shown: dict, k: str, default: float) -> float:
    v = shown.get(k)
    return v[0] if isinstance(v, list) else (v if isinstance(v, (int, float)) else default)


def _times(shown: dict, k: str) -> int:
    v = shown.get(k)
    return int(v[1]) if isinstance(v, list) else 0
MIN_WAIT_S = 4.0       # a wait never returns sooner: the agent that started it has time to end its turn, so the
                       # wait's completion arrives as a fresh notification that starts a new turn, never inside the old one
from .timeline import export_html, export_json, results, status


def _store(args: argparse.Namespace) -> Store:
    d = getattr(args, "dir", None) or current_run()
    if not d or not os.path.exists(os.path.join(d, "plan.json")):
        sys.exit("no gpuh run: start one with `gpuh start PLAN.json`")
    return Store(d)


def _me(args: argparse.Namespace) -> str:
    return getattr(args, "as_", None) or os.environ.get("GPUH_AGENT") or "main"


def _send(args: argparse.Namespace, cmd: dict) -> int:
    st = _store(args)
    if not st.daemon_pid():
        sys.exit("the harness daemon is not running (see daemon.log in " + st.dir + ")")
    cid = st.submit(dict(cmd, **{"from": _me(args)}))
    rep = st.await_reply(cid, timeout=30 + 2 * TICK_S)
    if rep is None:
        print("sent; no reply yet (the daemon may be busy)")
        return 0
    print(("" if rep["ok"] else "refused: ") + rep["text"])
    me = _me(args)
    if me not in st.live_waiters():
        print(f"(no wait is running for {me}: start `gpuh --as {me} wait --for {me}` in the background before you end "
              f"your turn, or nothing will wake you)")
    return 0 if rep["ok"] else 2


def _csv(v: Optional[str]) -> List[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


# commands ---------------------------------------------------------------------------------------------------------

def cmd_check(args: argparse.Namespace) -> int:
    try:
        board, policy, warnings = P.load(args.plan)
    except (P.PlanError, OSError) as e:
        print(f"plan refused: {e}")
        return 2
    print(f"plan ok: {len(board.candidates)} candidates, {len(board.jobs)} jobs "
          f"({sum(1 for j in board.jobs.values() if j.role.value == 'check')} checks)")
    for w in warnings:
        print(f"  note: {w}")
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    gpus = [int(g) for g in _csv(args.gpus)] if args.gpus else None
    try:
        board, policy, warnings = P.load(args.plan, gpus=gpus)
    except (P.PlanError, OSError) as e:
        print(f"plan refused: {e}")
        return 2
    if not board.gpus:
        board.gpus = detect_gpus()
    if not board.gpus:
        print("no GPUs: give --gpus 0,1,2,3 or list them in the plan")
        return 2
    run_dir = args.dir or os.path.join(gpuh_home(), "runs", time.strftime("%Y%m%dT%H%M%S"))
    if os.path.exists(os.path.join(run_dir, "board.json")) and not args.resume:
        print(f"{run_dir} already holds a run; pass --resume to continue it, or a new --dir")
        return 2
    st = Store(run_dir)
    with open(args.plan) as fh:
        plan_d = json.load(fh)
    plan_d["gpus"] = board.gpus
    plan_d["workdir"] = board.workdir
    with open(st.path("plan.json"), "w") as fh:
        json.dump(plan_d, fh, indent=1)
    board.started_at = time.time()
    if not args.resume:
        st.save_board(board)
    set_current(run_dir)
    log = open(st.path("daemon.log"), "a")
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.Popen([sys.executable, "-m", "gpuharness.daemon", run_dir], cwd=pkg_root, env=env,
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(50):
        if st.daemon_pid():
            break
        time.sleep(0.1)
    print(f"harness up (pid {proc.pid}): {len(board.candidates)} candidates, {len(board.jobs)} jobs on GPUs "
          f"{','.join(map(str, board.gpus))}; run dir {run_dir}")
    for w in warnings:
        print(f"  note: {w}")
    branches = sorted({c.branch for c in board.candidates.values()} - {"main"})
    print("next: each branch runs `gpuh wait --for <its name>` as a background task"
          + (f" ({', '.join(branches)})" if branches else "") + "; you run `gpuh wait --for main` the same way")
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    st = _store(args)
    agent = args.for_ or _me(args)
    marker = st.register_waiter(agent)
    started = time.time()
    deadline = started + args.timeout if args.timeout else None
    t0 = 0.0
    try:
        with open(st.path("plan.json")) as fh:
            t0 = json.load(fh).get("_t0", 0.0)
    except (OSError, ValueError):
        pass
    try:
        while True:
            pend = st.pending(agent)
            wake = [e for e in pend if e.wake]
            daemon_up = st.daemon_pid() is not None
            # a decision only this agent can make, not shown to it for a while (missed, or left undone): show it again
            now = time.time()
            shown = st.shown(agent)          # key -> [last shown, times shown]
            items = st.read_open(agent)
            fresh = [k for k, _ in items if k not in shown]
            if fresh:   # first seen now: its own event brought it; remind only if it is still open later
                for k in fresh:
                    shown[k] = [now, 0]
                st.set_shown(agent, shown)
            stale = [(k, txt) for k, txt in items
                     if now - _last(shown, k, now) > REMIND_OPEN_S[min(_times(shown, k), len(REMIND_OPEN_S) - 1)]]
            if (wake or stale or (not daemon_up and pend)) and time.time() - started >= MIN_WAIT_S:
                for e in pend:
                    print(e.line(t0))
                for k, txt in stale:
                    print(f"[open] STILL OPEN {txt}")
                if pend:
                    st.set_cursor(agent, max(e.seq for e in pend))
                for k, _ in stale:
                    shown[k] = [now, _times(shown, k) + 1]
                st.set_shown(agent, shown)
                return 0
            if not daemon_up:
                print("the harness is not running: nothing more will happen (gpuh status for the last state)")
                return 3
            if deadline and time.time() > deadline:
                for e in pend:
                    print(e.line(t0))
                if pend:
                    st.set_cursor(agent, max(e.seq for e in pend))
                print(f"no event for {agent} in {duration(args.timeout)}; wait again")
                return 0
            time.sleep(1.0)
    finally:
        try:
            os.remove(marker)
        except OSError:
            pass


SILENT_AFTER_S = 600.0   # a running job whose log has not grown for this long is reported as silent, with its health
CPU_SAMPLE_S = 3.0       # how long `gpuh status` watches a silent job's processes


def silent_health(st: Store, board, now: float) -> List[str]:
    """For each running job whose log has said nothing for SILENT_AFTER_S: is it working or stuck?

    The harness reads what the machine shows: the job's process group is alive (its state says so), and whether those
    processes are using CPU right now. A GPU job's host side always uses some CPU (its launch loop, data loading);
    one that has stopped using any is not making progress. Many evaluations print nothing between loading the model
    and their final line, so silence alone says nothing."""
    from .launcher import cpu_seconds
    silent = []
    for j in board.jobs.values():
        if j.state.value != "running" or not j.pid or not j.log:
            continue
        try:
            quiet = now - os.path.getmtime(j.log)
        except OSError:
            continue
        if quiet >= SILENT_AFTER_S:
            silent.append((j, quiet))
    if not silent:
        return []
    first = {j.id: cpu_seconds(j.pid) for j, _ in silent}
    time.sleep(CPU_SAMPLE_S)
    out = []
    for j, quiet in silent:
        a, b = first[j.id], cpu_seconds(j.pid)
        if a is None or b is None:
            verdict = "its processes could not be read"
        elif b - a > 0.05 * CPU_SAMPLE_S:
            verdict = (f"working: its processes used {b - a:.1f} CPU-s in the last {CPU_SAMPLE_S:.0f} s "
                       f"({b:.0f} CPU-s since it started)")
        else:
            verdict = (f"STUCK? its processes used no CPU in the last {CPU_SAMPLE_S:.0f} s "
                       f"({b:.0f} CPU-s since it started): check again before acting")
        exp = j.estimate_s
        out.append(f"  silent {j.id}: no new log line for {duration(quiet)}, running {duration(now - (j.started_at or now))}"
                   f"; {verdict}" + (f"; the plan estimated {duration(exp)}" if exp else ""))
    return out


def cmd_status(args: argparse.Namespace) -> int:
    st = _store(args)
    board = st.load_board()
    now = time.time()
    print(status(board, now, verbose=args.verbose))
    health = silent_health(st, board, now)
    if health:
        print("silent jobs (no log progress; a job that prints nothing until it ends is normal):")
        print("\n".join(health))
    if not st.daemon_pid():
        print("(the harness daemon is not running)")
    return 0


def cmd_results(args: argparse.Namespace) -> int:
    print(results(_store(args).load_board()))
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    st = _store(args)
    path = st.path("logs", f"{args.job}.log")
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 200_000))
            lines = fh.read().decode("utf-8", "replace").replace("\r", "\n").splitlines()
    except OSError:
        print(f"no log for {args.job}")
        return 2
    print("\n".join(lines[-args.n:]))
    try:
        board = st.load_board()
        health = silent_health(st, board, time.time()) if args.job in board.jobs else []
        mine = [h for h in health if h.startswith(f"  silent {args.job}:")]
        if mine:
            print("[gpuh]" + mine[0][len("  silent"):])
    except (OSError, ValueError):
        pass
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    st = _store(args)
    for e in st.read_events():
        if args.all or not args.for_ or e.to in (args.for_, "all"):
            print(f"{e.seq:5d} -> {e.to:12s} {e.line()}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    st = _store(args)
    board = st.load_board()
    if args.json:
        print(export_json(st, board))
        return 0
    path = export_html(st, board, args.html or st.path("timeline.html"))
    print(path)
    return 0


def cmd_pass(args: argparse.Namespace) -> int:
    return _send(args, {"op": "pass", "jobs": args.jobs})


def cmd_reject(args: argparse.Namespace) -> int:
    return _send(args, {"op": "reject", "candidate": args.candidate, "why": args.why, "evidence": _csv(args.evidence)})


def cmd_decide(args: argparse.Namespace) -> int:
    return _send(args, {"op": "decide", "candidate": args.candidate, "verdict": args.verdict, "why": args.why,
                        "evidence": _csv(args.evidence), "stop": args.stop})


def cmd_add(args: argparse.Namespace) -> int:
    with open(args.fragment) as fh:
        frag = json.load(fh)
    return _send(args, {"op": "add", "plan": frag})


def cmd_cancel(args: argparse.Namespace) -> int:
    # a job that only looks hung: silent, but its processes are working. Say so before it is stopped (G-k2lp-haiku-A3:
    # squat's GSM8K, 5 h into a 29,000-s run that printed nothing per example, was cancelled as "hung")
    try:
        st = _store(args)
        board = st.load_board()
        working = [h for h in silent_health(st, board, time.time())
                   if "working:" in h and any(h.startswith(f"  silent {j}:") for j in args.jobs)]
    except (OSError, ValueError):
        working = []
    if working and not args.force:
        print("not cancelled: these jobs are silent but working, not hung:")
        print("\n".join(working))
        print("A job that prints nothing until it ends is normal. To stop it anyway, give the reason: "
              "`gpuh cancel <job> --force --why '...'`.")
        return 2
    return _send(args, {"op": "cancel", "jobs": args.jobs, "why": args.why})


def cmd_retry(args: argparse.Namespace) -> int:
    return _send(args, {"op": "retry", "jobs": args.jobs, "command": args.command})


def cmd_estimate(args: argparse.Namespace) -> int:
    return _send(args, {"op": "estimate", "job": args.job, "seconds": args.seconds})


def cmd_finish(args: argparse.Namespace) -> int:
    return _send(args, {"op": "finish", "why": args.why})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="gpuh", description="GPU job harness for agents: the agent judges, the harness "
                                 "schedules, launches, watches and cancels.")
    ap.add_argument("--version", action="version", version=f"gpuh {__version__}")
    ap.add_argument("--as", dest="as_", help="the agent sending this command (default $GPUH_AGENT or main)")
    ap.add_argument("--dir", help="the run directory (default: the current run)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name: str, fn, help_: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.set_defaults(fn=fn)
        return p

    p = add("check", cmd_check, "validate a plan without starting anything"); p.add_argument("plan")
    p = add("start", cmd_start, "validate the plan and start the harness")
    p.add_argument("plan"); p.add_argument("--gpus"); p.add_argument("--resume", action="store_true")
    p = add("wait", cmd_wait, "block until an event needs this agent; print its events")
    p.add_argument("--for", dest="for_", help="the agent to wait for (default --as)")
    p.add_argument("--timeout", type=float, default=0.0, help="return after this many seconds anyway")
    p = add("status", cmd_status, "GPUs, candidates and (with -v) the queue")
    p.add_argument("-v", "--verbose", action="store_true")
    add("results", cmd_results, "every finished job's result line, by candidate")
    p = add("log", cmd_log, "the last lines of a job's log"); p.add_argument("job"); p.add_argument("-n", type=int, default=20)
    p = add("events", cmd_events, "the event log")
    p.add_argument("--for", dest="for_"); p.add_argument("--all", action="store_true")
    p = add("export", cmd_export, "the run's timeline as HTML (default) or JSON")
    p.add_argument("--html"); p.add_argument("--json", action="store_true")
    p = add("pass", cmd_pass, "a check passed: open the work gated on it"); p.add_argument("jobs", nargs="+")
    p = add("reject", cmd_reject, "rule a candidate out; its whole subtree is cancelled at once")
    p.add_argument("candidate"); p.add_argument("--why", required=True)
    p.add_argument("--evidence", help="comma-separated job ids the rejection rests on")
    p = add("decide", cmd_decide, "record a candidate's verdict")
    p.add_argument("candidate"); p.add_argument("--verdict", required=True); p.add_argument("--why", default="")
    p.add_argument("--evidence"); p.add_argument("--stop", action="store_true", help="also stop its running jobs")
    p = add("add", cmd_add, "add jobs and candidates from a plan fragment"); p.add_argument("fragment")
    p = add("cancel", cmd_cancel, "stop or drop jobs"); p.add_argument("jobs", nargs="+"); p.add_argument("--why")
    p.add_argument("--force", action="store_true", help="stop a job even if it is silent but working")
    p = add("retry", cmd_retry, "run a failed, died or cancelled job again")
    p.add_argument("jobs", nargs="+"); p.add_argument("--command")
    p = add("estimate", cmd_estimate, "correct a waiting job's prior length")
    p.add_argument("job"); p.add_argument("seconds", type=float)
    p = add("finish", cmd_finish, "stop what is left and end every wait"); p.add_argument("--why", default="")
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
