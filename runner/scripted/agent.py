"""A scripted ComputeWise agent: executes a fixed plan of shell launches, waits and pauses, with no model.

usage (inside the box, as the agent user, cwd = the task root):
  python3 agent.py --plan plan.json --transcript <session dir>/scripted.jsonl

It writes a transcript in the shape of Claude Code's session JSONL, so the runner's scorer
reads it unchanged: an `assistant` record with one `tool_use` block per action, and a `user`
record with its `tool_result` when the action returns. Each record carries "scripted": true,
so a scripted run can never be taken for a model's.

Plan schema (written on the host by plan.py; nothing in it is hidden-side):
  {"policy": "...", "steps": [step, ...]}
  step := {"op": "think",  "seconds": S}                     # a pause, logged as a model turn
        | {"op": "run",    "cmd": "...", "label": "..."}     # foreground shell command, waits for it
        | {"op": "launch", "cmd": "...", "label": "..."}     # background shell command (a lane),
                                                             #   returns at once
        | {"op": "wait",   "labels": [...] | "all"}          # block until those launches exit
        | {"op": "launch_at", "at_s": S, "cmd": "...", "label": "..."}  # background launch at S s after start
        | {"op": "pool", "gpus": G, "jobs": [{"cmd", "label", "width"}], "stop_at_first_completion": bool}
                                                             # list scheduling over G GPUs (run_pool)
        | {"op": "shell", "cmd": "...", "label": "...", "expect_s": S}   # a foreground command that is no task job
                                                             #   (a shared data step); runs like `run`
        | {"op": "write", "path": P, "content": C, "seconds": S, "label": L}
                                                             # prepare a job: S s of the agent's time, then write
                                                             #   C to P (a Write tool call), as a model would
        | {"op": "prep_pool", "gpus": G, "jobs": [{"cmd", "label", "width", "after", "write": write-step | null}]}
                                                             # prepare jobs one by one and launch each prepared job
                                                             #   as GPUs free (run_prep_pool)
        | {"op": "exit"}                                      # end the session now; launches keep running
A `launch` runs its command with `sh -c`, in its own process group, stdout and stderr to
logs/<label>.log under the cwd. A chain of jobs on one GPU is one launch of a `cmd1; cmd2` line.

Standard library only, Python 3.8+.
"""
import argparse, json, os, subprocess, sys, time, uuid
from datetime import datetime, timezone

def _job_oom_score():
    """Run in each launched child before exec: its jobs become killable again. The harness may
    run with oom_score_adj -1000 so a box-level OOM never kills the agent itself (isolation,
    2026-10-20); a child inherits that, so reset it to 0 or an OOM freezes the box instead of
    killing a job. Lowering is privileged, raising is not, so this needs no capability."""
    try:
        with open("/proc/self/oom_score_adj", "w") as fh:
            fh.write("0")
    except OSError:
        pass

T0 = None                       # the agent's start, wall clock; launch_at times are relative to it

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

class Transcript:
    def __init__(self, path, session_id):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.fh = open(path, "a", buffering=1)
        self.sid = session_id
        self.n = 0
    def write(self, rec):
        rec.update({"timestamp": now_iso(), "sessionId": self.sid, "scripted": True, "uuid": str(uuid.uuid4())})
        self.fh.write(json.dumps(rec) + "\n")
    def turn(self, text):
        self.n += 1
        self.write({"type": "assistant", "message": {"id": f"scripted-msg-{self.n}", "role": "assistant",
                                                       "content": [{"type": "text", "text": text}]}})
    def call(self, name, inp):
        self.n += 1
        tid = f"scripted-tool-{self.n}"
        self.write({"type": "assistant", "message": {"id": f"scripted-msg-{self.n}", "role": "assistant",
                                                       "content": [{"type": "tool_use", "id": tid, "name": name, "input": inp}]}})
        return tid
    def result(self, tid, content, is_error=False):
        self.write({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": content, "is_error": is_error}]}})

def run_pool(step, tr, procs):
    """List scheduling over a fixed GPU pool: launch the first job in order whose `after` jobs have
    exited cleanly, as soon as `width` GPUs are free, on the lowest-numbered free ones, then block
    until any job exits. A job's
    completion is written the way Claude Code records a background task finishing."""
    free = list(range(int(step["gpus"])))
    queue = list(step["jobs"])
    running = {}
    stop_first = bool(step.get("stop_at_first_completion"))
    exited = {}                                   # label -> exit code, for `after` dependencies
    def ready(j):
        return all(exited.get(d) == 0 for d in j.get("after") or [])
    while queue or running:
        while True:
            # the first job in order whose `after` jobs have all exited cleanly; it waits until enough GPUs are free
            j = next((x for x in queue if ready(x)), None)
            if j is None or len(free) < int(j["width"]):
                break
            queue.remove(j)
            gpus = [free.pop(0) for _ in range(int(j["width"]))]
            cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} {j['cmd']}"
            tid = tr.call("Bash", {"command": cmd, "description": j["label"], "run_in_background": True})
            log = open(os.path.join("logs", f"{j['label']}.log"), "ab")
            p = subprocess.Popen(["sh", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=_job_oom_score)
            running[p.pid] = (p, gpus, j["label"], tid)
            procs[j["label"]] = (p, tid)
            tr.result(tid, f"started {j['label']} on GPUs {gpus}")
        if not running:
            break                                 # nothing running and nothing ready: a dependency failed
        pid, status = os.waitpid(-1, 0)
        if pid not in running:
            continue
        p, gpus, label, tid = running.pop(pid)
        p.returncode = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8
        exited[label] = p.returncode
        tr.write({"type": "user", "message": {"role": "user", "content":
                  f"<task-notification><task-id>{label}</task-id><tool-use-id>{tid}</tool-use-id>"
                  f"<status>{'completed' if p.returncode == 0 else 'failed'}</status>"
                  f"<summary>exit {p.returncode}</summary></task-notification>"}})
        free = sorted(free + gpus)
        if stop_first:
            return

def do_write(step, tr):
    """A preparation: the agent spends `seconds` on it (as a model would writing the file), then writes the file."""
    time.sleep(float(step.get("seconds") or 0.0))
    tid = tr.call("Write", {"file_path": step["path"], "content": step["content"]})
    d = os.path.dirname(step["path"])
    if d:
        os.makedirs(d, exist_ok=True)
    with open(step["path"], "w") as fh:
        fh.write(step["content"])
    tr.result(tid, f"wrote {step['path']} ({len(step['content'])} bytes)")

def do_prep(step, tr):
    """A pool job's preparation: a write (a code preparation), a sequence of a write and a foreground data command (a data
    preparation, cpu-tasks' H4), or none (a sibling the group's command already prepared)."""
    if step["op"] == "write":
        do_write(step, tr)
    elif step["op"] == "seq":
        for st in step["steps"]:
            if st["op"] == "write":
                do_write(st, tr)
            elif st["op"] == "shell":
                tid = tr.call("Bash", {"command": st["cmd"], "description": st.get("label", "")})
                with open(os.path.join("logs", f"{st.get('label', 'prep')}.log"), "ab") as log:
                    rc = subprocess.call(["sh", "-c", st["cmd"]], stdout=log, stderr=subprocess.STDOUT, preexec_fn=_job_oom_score)
                tr.result(tid, f"exit {rc}", is_error=rc != 0)

def run_prep_pool(step, tr, procs):
    """Prepare jobs one after another and launch each prepared job as soon as `width` GPUs are free and its `after`
    jobs have exited cleanly, on the lowest-numbered free GPUs. Between preparations the agent reaps finished jobs and
    launches what it can; while it prepares, it launches nothing (one agent, one thing at a time). When nothing is
    left to prepare and nothing can launch, it blocks until a job exits."""
    free = list(range(int(step["gpus"])))
    unprepared = [j for j in step["jobs"] if j.get("write")]
    prepared = [j for j in step["jobs"] if not j.get("write")]
    running, exited = {}, {}
    def note_exit(pid, status):
        nonlocal free
        p, gpus, label, tid = running.pop(pid)
        p.returncode = os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8
        exited[label] = p.returncode
        tr.write({"type": "user", "message": {"role": "user", "content":
                  f"<task-notification><task-id>{label}</task-id><tool-use-id>{tid}</tool-use-id>"
                  f"<status>{'completed' if p.returncode == 0 else 'failed'}</status>"
                  f"<summary>exit {p.returncode}</summary></task-notification>"}})
        free = sorted(free + gpus)
    def reap():
        while running:
            pid, status = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                return
            if pid in running:
                note_exit(pid, status)
    while unprepared or prepared or running:
        reap()
        launched = False
        for j in list(prepared):
            if all(exited.get(d) == 0 for d in j.get("after") or []) and len(free) >= int(j["width"]):
                prepared.remove(j)
                gpus = [free.pop(0) for _ in range(int(j["width"]))]
                cmd = f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} {j['cmd']}"
                tid = tr.call("Bash", {"command": cmd, "description": j["label"], "run_in_background": True})
                log = open(os.path.join("logs", f"{j['label']}.log"), "ab")
                p = subprocess.Popen(["sh", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     preexec_fn=_job_oom_score)
                running[p.pid] = (p, gpus, j["label"], tid)
                procs[j["label"]] = (p, tid)
                tr.result(tid, f"started {j['label']} on GPUs {gpus}")
                launched = True
                break
        if launched:
            continue
        if unprepared:
            j = unprepared.pop(0)
            do_prep(j["write"], tr)
            prepared.append(j)
            continue
        if not running:
            break                                 # prepared jobs whose dependencies failed: nothing more can run
        pid, status = os.waitpid(-1, 0)
        if pid in running:
            note_exit(pid, status)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--session-id", default=None)
    a = ap.parse_args()
    global T0
    T0 = time.time()
    plan = json.load(open(a.plan))
    tr = Transcript(a.transcript, a.session_id or str(uuid.uuid4()))
    tr.write({"type": "user", "message": {"role": "user", "content": f"scripted policy {plan.get('policy')}"}})
    os.makedirs("logs", exist_ok=True)
    procs = {}
    for step in plan["steps"]:
        op = step["op"]
        if op == "think":
            time.sleep(float(step["seconds"]))
            tr.turn(f"think {step['seconds']} s")
        elif op in ("run", "shell"):          # shell: a foreground command that is not a task job (a data step)
            tid = tr.call("Bash", {"command": step["cmd"], "description": step.get("label", "")})
            with open(os.path.join("logs", f"{step.get('label', 'run')}.log"), "ab") as log:
                rc = subprocess.call(["sh", "-c", step["cmd"]], stdout=log, stderr=subprocess.STDOUT, preexec_fn=_job_oom_score)
            tr.result(tid, f"exit {rc}", is_error=rc != 0)
        elif op == "launch_at":
            # packed_overlap: launch in the background at a set time after the agent started, so
            # this job's CPU start-up overlaps the previous job's GPU work (plan.overlap_lanes)
            delay = T0 + float(step["at_s"]) - time.time()
            if delay > 0:
                time.sleep(delay)
            label = step["label"]
            tid = tr.call("Bash", {"command": step["cmd"], "description": label, "run_in_background": True})
            log = open(os.path.join("logs", f"{label}.log"), "ab")
            procs[label] = (subprocess.Popen(["sh", "-c", step["cmd"]], stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, preexec_fn=_job_oom_score), tid)
            tr.result(tid, f"started {label}")
        elif op == "launch":
            label = step["label"]
            tid = tr.call("Bash", {"command": step["cmd"], "description": label, "run_in_background": True})
            log = open(os.path.join("logs", f"{label}.log"), "ab")
            procs[label] = (subprocess.Popen(["sh", "-c", step["cmd"]], stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True, preexec_fn=_job_oom_score), tid)
            tr.result(tid, f"started {label}")
        elif op == "wait":
            labels = list(procs) if step["labels"] == "all" else step["labels"]
            tid = tr.call("Bash", {"command": "wait " + " ".join(labels), "description": "wait for " + ",".join(labels)})
            rcs = {l: procs[l][0].wait() for l in labels if l in procs}
            tr.result(tid, json.dumps(rcs), is_error=any(rc != 0 for rc in rcs.values()))
        elif op == "pool":
            run_pool(step, tr, procs)
        elif op == "write":
            do_write(step, tr)
        elif op == "prep_pool":
            run_prep_pool(step, tr, procs)
        elif op == "exit":
            break
        else:
            sys.exit(f"unknown op {op!r}")
    tr.turn("done")
    # Hand the box the launched process groups, so its teardown rule can wait for them: a job
    # that has not yet reached the daemon is still alive as a process. The real box judges this
    # from the processes in its cgroup; the bench harness reads this file.
    try:
        with open(os.environ.get("SCRIPTED_PGIDS", os.path.join("logs", ".pgids")), "w") as fh:
            fh.write("\n".join(str(p.pid) for p, _ in procs.values()))
    except OSError:
        pass
    # Leave background launches running: the box's teardown rule keeps them going until the
    # deadline or until every launched job ends. Exit without waiting for them.
    os._exit(0)

if __name__ == "__main__":
    main()
