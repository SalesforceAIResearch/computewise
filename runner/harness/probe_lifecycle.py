"""The in-box check of the background-lifecycle fix (one of isolation's ship checks, 2026-10-21): a Haiku box run whose
brief asks the agent to start `sleep 25` as a background task, end its turn at once ("WAITING"), and reply "FINISHED"
when the task completes. It passes when Claude Code re-invoked the agent on the task's completion inside the box: two
`result` records in harness.stdout, a task_notification between them, the second turn's text says FINISHED, and the
harness ended after the second result.

usage, on the controller:  python3 -I probe_lifecycle.py <machine> [--task pkg-p6-par-q] [--usd 0.5] [--image <name:tag>]
                       python3 -I probe_lifecycle.py <machine> --held-open done|left [--image <name:tag>]

--held-open (the study's (b'), 2026-10-21) checks the close for a session held open by a background task that
never ends, on pkg-p3-mixed-q (three short jobs, part2 after part1). The brief has the agent run the jobs with nohup,
wait for them with short foreground polls, then start `while true; do sleep 30; done` as a background task and reply
DONE. `done` runs all three: it passes when stdin closed on "every task job completed" within 30 s of that last
answer. `left` never runs part2: it passes when stdin stayed open to teardown (work remained), as before.

It runs through run_smoke.py and run_one.py like any model run (gate, ceiling, stop relay), with --brief-file, so it
uses whatever harness mechanism the machine's run_one.py and image carry. On an image without the fix the second turn
never comes, and it fails. It prints the verdict and the evidence; the run directory stays on the machine under
/srv/runs/<tag>. A model run of about a minute, cents.
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
CAMPAIGN = os.path.join(RUNNER, "campaign")
sys.path.insert(0, CAMPAIGN)
BRIEF = ("Start `sleep 25; echo TASKDONE` as a background Bash task (run_in_background true). Then end your turn at once, "
         "replying only with the word WAITING. When that task finishes, reply only with the word FINISHED and its output.")

_LAUNCH = {
    "p1": "CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py "
          "--seed 6 --stop-at 250 > p1.log 2>&1 &",
    "r1": "CUDA_VISIBLE_DEVICES=1 nohup torchrun --standalone --nproc_per_node=1 --master_port 29511 "
          "radial-damp/radial_scale_then_radius_correct.py --seed 0 > r1.log 2>&1 &",
    "p2": "CUDA_VISIBLE_DEVICES=0 nohup torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py "
          "--seed 6 --resume checkpoints/seed6-step250.pt --stop-at 500 > p2.log 2>&1 &",
}
_WAIT = ("then wait for them by running, as a normal (foreground) Bash command, `sleep 60; pgrep -fa train | grep -v pgrep; "
         "tail -n 2 {logs}`, again and again until no torchrun process is left")
_TAIL = (" Then start `while true; do sleep 30; done` as a background Bash task (run_in_background true), and reply only "
         "with the word DONE. Do nothing else, and do not read any other file.")
WAKE_BRIEF = ("This is a short check, not a task. Call ScheduleWakeup with delaySeconds 90 and the prompt 'reply only "
              "with the word WOKE', then end your turn at once, replying only with the word WAITING. When you are woken, "
              "reply only with the word WOKE.")
HELD_BRIEF = {
    "done": ("This is a short check, not a task. In the project directory, run these two commands in one Bash call: `"
             + _LAUNCH["p1"] + "` and `" + _LAUNCH["r1"] + "`; " + _WAIT.format(logs="p1.log r1.log") + ". Then run `"
             + _LAUNCH["p2"] + "` and " + _WAIT.format(logs="p2.log") + "." + _TAIL),
    "left": ("This is a short check, not a task. In the project directory, run these two commands in one Bash call: `"
             + _LAUNCH["p1"] + "` and `" + _LAUNCH["r1"] + "`; " + _WAIT.format(logs="p1.log r1.log") + "." + _TAIL),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine")
    ap.add_argument("--task", default="pkg-p6-par-q")
    ap.add_argument("--usd", type=float, default=0.5)
    ap.add_argument("--flags", default=None, type=lambda v: [t for t in v.split(",") if t],
                    help="the harness-flags check instead (the study, 2026-10-21), comma-separated tokens, e.g. "
                         "--flags=--disallowedTools,Agent,Task: pass them to claude through "
                         "run_one.py's --claude-arg, with a one-line brief, and check that the init event's tool list "
                         "lacks Agent and Task and that run.json records the flags")
    ap.add_argument("--held-open", choices=("done", "left"), default=None)
    ap.add_argument("--wakeup", action="store_true", help="does a ScheduleWakeup fire in this mode with stdin held open? "
                                                          "(analysis, 2026-10-22) The agent schedules a 90 s wakeup and ends its "
                                                          "turn; PASS when the harness re-invokes it and it replies WOKE")
    ap.add_argument("--sidecar-image", default=None, help="the sidecar image, by full id, beside --image")
    ap.add_argument("--image", default=None, help="the box image, when the machine has more than one with the task's label "
                                                  "(run_one.py then refuses without it: the E2 during an isolation build)")
    a = ap.parse_args()
    if a.held_open:
        a.task = "pkg-p3-mixed-q"
    sys.argv = [sys.argv[0]]
    import driver
    row = driver.fleet().get(a.machine) or {}
    zone = row.get("zone")
    if row.get("status") != "RUNNING":
        sys.exit(f"{a.machine} is not running")
    tag = "P006-LIFECYCLE-PROBE-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if a.flags is not None:
        tag = tag.replace("LIFECYCLE", "FLAGS")
    if a.held_open:
        tag = tag.replace("LIFECYCLE", "HELDOPEN-" + a.held_open.upper())
    if a.wakeup:
        tag = tag.replace("LIFECYCLE", "WAKEUP")
    local = os.path.join("/tmp", tag + ".brief")
    open(local, "w").write(("Reply only with the word OK." if a.flags is not None else
                            WAKE_BRIEF if a.wakeup else
                            HELD_BRIEF[a.held_open] if a.held_open else BRIEF) + "\n")
    driver.upload(a.machine, local, f"/tmp/{tag}.brief", zone)
    os.remove(local)
    subprocess.run([sys.executable, "-I", os.path.join(RUNNER, "score", "set_ceiling.py"), tag, str(a.usd)], check=True)
    relay = subprocess.Popen([sys.executable, "-I", os.path.join(RUNNER, "score", "stop_relay.py"), tag],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=driver.env_for(a.machine, zone))
    cmd = ["/usr/bin/python3", "-I", "run_smoke.py", "--task", a.task, "--model", "claude-haiku-5-5",
           "--budget-usd", str(a.usd), "--ceiling-confirmed", "--max-hours", "0.6" if a.held_open else "0.25", "--tag", tag,
           "--brief-file", f"/tmp/{tag}.brief", "--", "--shm", "8g", "--pids", "8192", "--oom-protect", "on",
           "--after-deadline", "stop", *([f"--image={a.image}"] if a.image else []), *([f"--sidecar-image={a.sidecar_image}"] if a.sidecar_image else []), *[f"--claude-arg={f}" for f in (a.flags or [])]]
    unit = "p006-lifecycle-probe"
    driver.ssh(a.machine, f"sudo systemd-run --unit {unit} --collect --property=StandardOutput=file:/tmp/{tag}.log "
                          f"--property=StandardError=file:/tmp/{tag}.log --working-directory={driver.REMOTE_EXP}/runner "
                          + " ".join(cmd), zone)
    print(f"{tag} started on {a.machine}", flush=True)
    driver.wait_unit(a.machine, zone, unit, 3000 if a.held_open else 1800)
    try:
        relay.wait(timeout=60)
    except subprocess.TimeoutExpired:
        relay.terminate()
    rc, out = driver.ssh(a.machine, f"sudo cat /srv/runs/{tag}/box/harness.stdout; echo '@@STATUS@@'; "
                                    f"sudo cat /srv/runs/{tag}/status.json; echo '@@RUN@@'; sudo cat /srv/runs/{tag}/run.json", zone)
    stream, _, rest = out.partition("@@STATUS@@")
    st, _, runj = rest.partition("@@RUN@@")
    if a.wakeup:
        results = [json.loads(l) for l in stream.splitlines() if '"type": "result"' in l or '"type":"result"' in l]
        texts = [str(r.get("result") or "") for r in results]
        try:
            status = json.loads(st.strip() or "{}")
        except ValueError:
            status = {}
        called = '"name":"ScheduleWakeup"' in stream or '"name": "ScheduleWakeup"' in stream
        ok = called and len(results) >= 2 and any("WOKE" in t for t in texts[1:])
        print(json.dumps({"tag": tag, "machine": a.machine, "mode": "wakeup", "verdict": "PASS" if ok else "FAIL",
                          "schedule_wakeup_called": called, "results": texts[-3:], "status": status.get("status"),
                          "stdin_close_reason": status.get("stdin_close_reason")}, indent=1))
        sys.exit(0 if ok else 1)
    if a.held_open:
        try:
            status = json.loads(st.strip() or "{}")
        except ValueError:
            status = {}
        results = [json.loads(l) for l in stream.splitlines() if '"type": "result"' in l or '"type":"result"' in l]
        reason = status.get("stdin_close_reason") or ""
        closed, end = status.get("stdin_closed_wall"), status.get("harness_end_wall")
        # the last answer's time: the transcript's last assistant record, read on the machine
        rc2, last = driver.ssh(a.machine, f"sudo sh -c 'cat /srv/runs/{tag}/transcript/session/*/*.jsonl' | "
                                          "grep '\"type\":\"assistant\"' | tail -1 | python3 -c 'import sys,json,datetime; "
                                          "print(datetime.datetime.fromisoformat(json.loads(sys.stdin.read())[\"timestamp\"].replace(\"Z\",\"+00:00\")).timestamp())'", zone)
        try:
            last = float(last.strip().splitlines()[-1])
        except (ValueError, IndexError):
            last = None
        if a.held_open == "done":
            # stdin closed on the rule soon after the last answer AND the harness then ended: closing stdin alone does
            # not end a Claude Code session still holding a background task (the first probe, 15:14Z: closed at 912 s,
            # ran on to the 2,100 s horizon)
            ok = (reason.startswith("lifecycle: every task job completed") and closed is not None and last is not None
                  and 0 <= closed - last <= 30 and bool(results) and end is not None and end - closed <= 60)
        else:
            ok = reason == "teardown" and bool(results)
        print(json.dumps({"tag": tag, "machine": a.machine, "mode": a.held_open, "verdict": "PASS" if ok else "FAIL",
                          "stdin_close_reason": reason, "closed_after_last_answer_s": None if closed is None or last is None else round(closed - last, 1),
                          "status": status.get("status"), "results": [r.get("result") for r in results][-2:],
                          "harness_end_wall": end}, indent=1))
        sys.exit(0 if ok else 1)
    if a.flags is not None:
        init = next((json.loads(l) for l in stream.splitlines() if '"subtype": "init"' in l or '"subtype":"init"' in l), {})
        tools = init.get("tools") or []
        try:
            extra = (json.loads(runj.strip() or "{}").get("claude_code") or {}).get("extra_flags")
        except ValueError:
            extra = None
        results = [json.loads(l).get("result") for l in stream.splitlines() if '"type": "result"' in l or '"type":"result"' in l]
        ok = bool(tools) and "Agent" not in tools and "Task" not in tools and extra == a.flags and bool(results)
        print(json.dumps({"tag": tag, "machine": a.machine, "verdict": "PASS" if ok else "FAIL", "init_tools": len(tools),
                          "agent_in_tools": "Agent" in tools, "task_in_tools": "Task" in tools, "extra_flags": extra,
                          "results": results}, indent=1))
        sys.exit(0 if ok else 1)
    try:
        status = json.loads(st.strip() or "{}")
    except ValueError:
        status = {}
    results, texts, notif_between, seen_first = [], [], False, False
    for line in stream.splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") == "result":
            results.append(r.get("result"))
            seen_first = True
        elif r.get("type") == "system" and r.get("subtype") == "task_notification" and seen_first and len(results) == 1:
            notif_between = True
        elif r.get("type") == "assistant":
            for b in (r.get("message") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append((len(results), b.get("text", "")))
    second_says = any(n == 1 and "FINISHED" in t for n, t in texts)
    ok = len(results) >= 2 and notif_between and second_says
    print(json.dumps({"tag": tag, "machine": a.machine, "verdict": "PASS" if ok else "FAIL", "results": results,
                      "task_notification_between_results": notif_between, "second_turn_says_finished": second_says,
                      "status": {k: status.get(k) for k in status if "stdin" in k or k in ("status", "reason",
                                                                                          "harness_end_wall", "agent_start_wall")}},
                     indent=1))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
