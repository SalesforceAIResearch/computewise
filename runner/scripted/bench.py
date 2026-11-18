"""Bench test of the runner's tooling on the VM, with no box: gpu-mock's real daemon, the scripted
agent, and stand-in jobs, producing a real run directory for the scorer and the timeline.

usage, on the VM:  python3 -I bench.py --out /srv/runs/bench-<name> [--policy packed] [--speed 1]

The bench task is synthetic and small, so a run takes about a minute: one-GPU jobs of 10-20 s of paced
GPU work, each with 2 s of host time, on a 4-GPU daemon, deadline 90 s. --shape fit has four jobs
that all fit at once; --shape queue has six, so two must wait for a free GPU. Its
task.json is written into the run directory beside the occupancy log, so score_run.py, idle.py and
the timeline read it the same way they read a real run. This proves the tooling against real
daemon output. It does not prove the box, the image, or replay against real records; C1 does that
through isolation's run_one.py.
"""
import argparse, json, os, shlex, signal, subprocess, sys, time, uuid

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", ".."))
DAEMON = os.path.join(EXP, "gpu-mock", "src", "daemon", "gpu_daemon.py")
STANDIN = os.path.join(HERE, "standin_job.py")

BENCH_JOBS = {
    # four one-GPU jobs that all fit at once: packing is trivial, serial is not
    "fit": (("long-a", 20), ("long-b", 20), ("short-a", 10), ("short-b", 10)),
    # six one-GPU jobs on four GPUs: two must wait, so stopping early leaves work unrun
    "queue": (("long-a", 20), ("long-b", 20), ("mid-a", 15), ("mid-b", 15), ("short-a", 10), ("short-b", 10)),
}

def bench_task(shape="fit"):
    jobs = []
    for jid, g in BENCH_JOBS[shape]:
        jobs.append({"job_id": jid, "records": [], "gpu_options": [1], "after": [],
                     "launch": f"python3 -I {STANDIN} --output_dir=out --gpu-s {g} --host-s 2",
                     "bench_gpu_s": g, "bench_host_s": 2})
    work = sum(j["bench_gpu_s"] + j["bench_host_s"] for j in jobs)
    longest = max(j["bench_gpu_s"] + j["bench_host_s"] for j in jobs)
    return {"task_id": f"bench-{shape}", "task_class": "fan_out_required", "box": {"gpus": 4},
            "deadline_s": 90, "jobs": jobs,
            "stats": {"lower_bound_makespan_s": max(longest, work / 4), "serial_makespan_s": work,
                      "lower_bound_how": "bench: max(longest job, total work / 4 GPUs), host time included",
                      "serial_how": "bench: sum of host + GPU over the jobs"}}

def bench_plan(task, policy, think_s):
    """The bench's own plans, in the same step language as plan.py, without the registry."""
    def cmd(j, gpu=None):
        c = j["launch"].replace("--output_dir=out", f"--output_dir=out-{j['job_id']}")
        return f"CUDA_VISIBLE_DEVICES={gpu} {c}" if gpu is not None else c
    jobs = task["jobs"]
    steps = []
    if policy == "serial":
        for j in jobs:
            steps += [{"op": "think", "seconds": think_s}, {"op": "run", "cmd": cmd(j, 0), "label": j["job_id"]}]
    elif policy in ("packed", "delayed", "quit_at_first_completion"):
        if policy == "delayed":
            steps.append({"op": "think", "seconds": 15})
        order = sorted(jobs, key=lambda j: -j["bench_gpu_s"])
        steps.append({"op": "pool", "gpus": 4, "stop_at_first_completion": policy == "quit_at_first_completion",
                      "jobs": [{"cmd": cmd(j), "label": j["job_id"], "width": 1} for j in order]})
        if policy == "quit_at_first_completion":
            steps.append({"op": "exit"})
    elif policy == "packed_overlap":
        # the same overlap rule as plan.py, over the bench's synthetic jobs; host terms from the
        # stand-in's known 2 s start-up
        sys.path.insert(0, HERE)
        import plan as _plan
        ht = {"standin_job.py": {"hello_s": 0.0, "pre_s": float(jobs[0]["bench_host_s"]), "post_s": 0.0}}
        for at, j, g in _plan.overlap_lanes(jobs, 4, ht):
            steps.append({"op": "launch_at", "at_s": round(at, 3), "cmd": _plan.job_cmd(j, g), "label": j["job_id"]})
        steps.append({"op": "wait", "labels": "all"})
        return {"policy": policy, "task_id": task["task_id"], "think_s": think_s, "host_terms": ht, "steps": steps}
    elif policy == "quit_while_running":
        steps += [{"op": "launch", "cmd": cmd(jobs[0], 0), "label": jobs[0]["job_id"]}, {"op": "exit"}]
    else:
        sys.exit(f"unknown bench policy {policy!r}")
    return {"policy": policy, "task_id": task["task_id"], "think_s": think_s, "steps": steps}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="packed")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--think-s", type=float, default=3.0)
    ap.add_argument("--shape", default="fit", choices=sorted(BENCH_JOBS))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=False)
    task = bench_task(a.shape)
    plan = bench_plan(task, a.policy, a.think_s)
    json.dump(task, open(os.path.join(out, "task.json"), "w"), indent=1)
    json.dump(plan, open(os.path.join(out, "plan.json"), "w"), indent=1)
    sock = f"/tmp/bench-{uuid.uuid4().hex[:8]}.sock"
    work = os.path.join(out, "workspace"); os.makedirs(work)
    env = dict(os.environ, GPU_MOCK_SOCKET=sock)
    d = subprocess.Popen([sys.executable, "-I", DAEMON, "--socket", sock, "--gpus", "4", "--speed", str(a.speed),
                          "--occupancy", os.path.join(out, "occupancy.jsonl")],
                         stdout=open(os.path.join(out, "daemon.log"), "w"), stderr=subprocess.STDOUT)
    for _ in range(100):
        if os.path.exists(sock):
            break
        time.sleep(0.05)
    sid = str(uuid.uuid4())
    start = time.time()
    st = {"status": "running", "agent_start_wall": start, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
          "bench": True}
    json.dump(st, open(os.path.join(out, "status.json"), "w"))
    json.dump({"task": task["task_id"], "harness": "scripted", "model": None, "policy": a.policy, "speed": a.speed,
               "session_id": sid, "bench": True}, open(os.path.join(out, "run.json"), "w"), indent=1)
    rc = subprocess.call([sys.executable, "-I", os.path.join(HERE, "agent.py"), "--plan", os.path.join(out, "plan.json"),
                          "--transcript", os.path.join(out, "transcript", "session", "proj", "scripted.jsonl"),
                          "--session-id", sid], cwd=work, env=env)
    agent_end = time.time()
    # teardown rule: keep the daemon up until the deadline or until no job is alive
    deadline = start + task["deadline_s"] / a.speed
    pgids = [int(x) for x in (open(os.path.join(work, "logs", ".pgids")).read().split()
                              if os.path.exists(os.path.join(work, "logs", ".pgids")) else [])]
    def alive():
        """Any process left in a launched job's process group. Judged from processes, not from the
        occupancy log, as the real box does: a job that has not yet said hello is still running."""
        live = []
        for g in pgids:
            try:
                os.killpg(g, 0)
                live.append(g)
            except ProcessLookupError:
                pass
            except PermissionError:
                live.append(g)
        return live
    while time.time() < deadline and alive():
        time.sleep(0.5)
    d.send_signal(signal.SIGTERM)
    try:
        d.wait(5)
    except subprocess.TimeoutExpired:
        d.kill()
    st.update({"status": "completed" if rc == 0 else "harness-error", "agent_end_wall": agent_end,
               "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "harness_exit_code": rc})
    json.dump(st, open(os.path.join(out, "status.json"), "w"))
    print(f"bench {a.policy}: {out}  agent exit {rc}  agent ran {agent_end - start:.1f} s")

if __name__ == "__main__":
    main()
