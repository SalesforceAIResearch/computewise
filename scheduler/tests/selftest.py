"""Known-answer tests for the box scheduler, run anywhere (no root, no GPUs): starts slurmctld on a temp socket
and spool, drives it through the real command shims, and checks each behaviour. usage: python3 selftest.py"""
import os, re, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
tmp = tempfile.mkdtemp(prefix="sched-test-")
env = dict(os.environ, SLURMCTLD_SOCKET=os.path.join(tmp, "ctld.sock"), SLURM_LIBDIR=os.path.join(ROOT, "slurm"),
           SLURMSTEPD=os.path.join(ROOT, "bin", "slurmstepd"), PATH=os.path.join(ROOT, "bin") + ":" + os.environ["PATH"])
work = os.path.join(tmp, "work"); os.makedirs(work)
spool = os.path.join(tmp, "spool")
ctl = subprocess.Popen([os.path.join(ROOT, "bin", "slurmctld"), "-D", "--gpus", "4", "--cpus", "8", "--mem-mb", "16000",
                        "--node-name", "gpu-node-01", "--spool", spool, "--socket", env["SLURMCTLD_SOCKET"],
                        "--first-job-id", "1000", "--active-file", os.path.join(tmp, "active")], env=env)
fails = []


def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=work, env=env, capture_output=True, text=True)
    if check and r.returncode != 0 and "--wait" not in cmd and "-W" not in cmd:
        print("CMD FAILED:", cmd, r.stdout, r.stderr)
    return r


def ok(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else "  :: " + str(detail)))
    if not cond:
        fails.append(name)


def jid(out):
    m = re.search(r"(\d+)", out)
    return m.group(1) if m else None


def state(j):
    return sh(f"sacct -j {j} -X -n -P --format=State").stdout.strip()


def wait_state(j, want, t=20):
    end = time.time() + t
    while time.time() < end:
        s = state(j)
        if s.split()[0:1] and s.split()[0] in want:
            return s
        time.sleep(0.3)
    return state(j)


def read_when(name, t=10):
    """A job's output file, once it has content: RUNNING in squeue comes before the job's first line is written."""
    path = os.path.join(work, name)
    end = time.time() + t
    while time.time() < end:
        if os.path.exists(path) and open(path).read().strip():
            break
        time.sleep(0.1)
    return open(path).read().strip() if os.path.exists(path) else ""


for _ in range(50):
    if os.path.exists(env["SLURMCTLD_SOCKET"]):
        break
    time.sleep(0.1)
try:
    # 1. a simple job: submit message, output file, COMPLETED, env
    r = sh("sbatch --gres=gpu:1 -J hello --wrap 'echo CVD=$CUDA_VISIBLE_DEVICES JOB=$SLURM_JOB_ID; sleep 1'")
    ok("sbatch prints 'Submitted batch job N'", re.fullmatch(r"Submitted batch job \d+\n", r.stdout), r.stdout + r.stderr)
    j1 = jid(r.stdout)
    s = wait_state(j1, ("COMPLETED", "FAILED"))
    ok("simple job COMPLETED", s == "COMPLETED", s)
    out = open(os.path.join(work, f"slurm-{j1}.out")).read() if os.path.exists(os.path.join(work, f"slurm-{j1}.out")) else ""
    ok("default output slurm-<id>.out with CUDA_VISIBLE_DEVICES and SLURM_JOB_ID", f"CVD=0 JOB={j1}" in out, out)
    ec = sh(f"sacct -j {j1} -X -n -P --format=JobID,State,ExitCode").stdout.strip()
    ok("sacct parsable row", ec == f"{j1}|COMPLETED|0:0", ec)
    rows = sh(f"sacct -j {j1} -n -P --format=JobID,State").stdout.split()
    ok("sacct shows .batch and .extern steps without -X", f"{j1}.batch|COMPLETED" in rows and f"{j1}.extern|COMPLETED" in rows, rows)

    # 2. five 1-GPU jobs on 4 GPUs: four run on distinct GPUs, the fifth pends on Resources and starts when one ends
    ids = [jid(sh(f"sbatch --parsable --gres=gpu:1 --wrap 'echo $CUDA_VISIBLE_DEVICES > gpu.$SLURM_JOB_ID; sleep {d}'").stdout)
           for d in (3, 6, 6, 6, 2)]
    time.sleep(1.0)
    q = sh("squeue -h -o '%i %t %R'").stdout
    pend = [l for l in q.splitlines() if " PD " in l]
    ok("fifth job pends with (Resources)", len(pend) == 1 and "(Resources)" in pend[0] and ids[4] in pend[0], q)
    # the box's end-of-run rule and the session lifecycle read this count from the host: pending jobs count too
    act_now = open(os.path.join(tmp, "active")).read().strip() if os.path.exists(os.path.join(tmp, "active")) else "missing"
    ok("active file counts running and pending jobs (4 running + 1 pending = 5)", act_now == "5", act_now)
    run_g = sorted(read_when(f"gpu.{i}") for i in ids[:4])
    ok("four running jobs got GPUs 0,1,2,3", run_g == ["0", "1", "2", "3"], run_g)
    wait_state(ids[0], ("COMPLETED",))
    t0 = time.time()
    wait_state(ids[4], ("RUNNING", "COMPLETED"))
    ok("pending job starts within 2 s of a GPU freeing", time.time() - t0 < 2.0, time.time() - t0)
    g5 = read_when(f"gpu.{ids[4]}")
    ok("it got the freed GPU 0", g5 == "0", g5)

    # 3. dependencies: afterok chain; afterok on a failure is cancelled (DependencyNeverSatisfied)
    a = jid(sh("sbatch --parsable --wrap 'sleep 3; exit 0'").stdout)
    b = jid(sh(f"sbatch --parsable --dependency=afterok:{a} --wrap 'echo B'").stdout)
    f_ = jid(sh("sbatch --parsable --wrap 'exit 3'").stdout)
    c = jid(sh(f"sbatch --parsable -d afterok:{f_} --wrap 'echo never'").stdout)
    d = jid(sh(f"sbatch --parsable -d afternotok:{f_} --wrap 'echo onfail'").stdout)
    q = sh(f"squeue -h -j {b} -o '%t %R'").stdout.strip()
    ok("dependent job pends with (Dependency)", q == "PD (Dependency)", q)
    ok("afterok chain completes", wait_state(b, ("COMPLETED",)) == "COMPLETED")
    sb = wait_state(c, ("CANCELLED",))
    ok("afterok on a failed job -> CANCELLED", sb.startswith("CANCELLED"), sb)
    reason = sh(f"sacct -j {c} -X -n -P --format=Reason").stdout.strip()
    ok("its reason is DependencyNeverSatisfied", reason == "DependencyNeverSatisfied", reason)
    ok("afternotok runs after a failure", wait_state(d, ("COMPLETED",)) == "COMPLETED")
    ok("failed job FAILED with exit code 3:0", sh(f"sacct -j {f_} -X -n -P --format=State,ExitCode").stdout.strip() == "FAILED|3:0")
    r = sh("sbatch --dependency=afterok:999999 --wrap true", check=False)
    ok("dependency on an unknown job is rejected", r.returncode != 0 and "Job dependency problem" in r.stderr, r.stderr)

    # 4. sbatch --wait returns the job's exit code
    r = sh("sbatch --wait --wrap 'sleep 1; exit 5'", check=False)
    ok("sbatch --wait exits with the job's exit code", r.returncode == 5 and "Submitted batch job" in r.stdout, (r.returncode, r.stdout))
    r = sh("sbatch -W --wrap 'sleep 1'", check=False)
    ok("sbatch -W exits 0 on success", r.returncode == 0)

    # 5. arrays with a throttle
    r = sh("sbatch --parsable --array=0-5%2 --gres=gpu:1 --wrap 'echo task $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_JOB_ID; sleep 2'")
    arr = r.stdout.strip()
    time.sleep(1.0)
    q = sh(f"squeue -h -j {arr} -o '%i %t %R'").stdout
    running = [l for l in q.splitlines() if " R " in l]
    ok("array throttle %2: two tasks running", len(running) == 2, q)
    ok("pending array tasks merged as [2-5%2] with JobArrayTaskLimit", f"{arr}_[2-5%2] PD (JobArrayTaskLimit)" in q, q)
    for i in range(6):
        wait_state(f"{arr}_{i}", ("COMPLETED",), t=30)
    out0 = open(os.path.join(work, f"slurm-{arr}_0.out")).read().strip()
    ok("array output slurm-%A_%a.out with task id", out0 == f"task 0 of {arr}", out0)
    ok("sacct shows all 6 array tasks", len(sh(f"sacct -j {arr} -X -n -P --format=JobID").stdout.split()) == 6)

    # 6. time limit
    t = jid(sh("sbatch --parsable --time=0:02 --wrap 'sleep 30'").stdout)
    st = wait_state(t, ("TIMEOUT",), t=15)
    ok("--time exceeded -> TIMEOUT", st == "TIMEOUT", st)

    # 7. scancel pending and running; hold/release
    big = jid(sh("sbatch --parsable --gres=gpu:4 --wrap 'sleep 20'").stdout)
    p = jid(sh("sbatch --parsable --gres=gpu:1 --wrap 'sleep 1'").stdout)
    time.sleep(0.8)
    ok("job behind a 4-GPU job pends", "PD" in sh(f"squeue -h -j {p} -o %t").stdout)
    sh(f"scancel {p}")
    ok("scancel a pending job -> CANCELLED", wait_state(p, ("CANCELLED",)).startswith("CANCELLED"))
    sh(f"scancel {big}")
    ok("scancel a running job -> CANCELLED", wait_state(big, ("CANCELLED",)).startswith("CANCELLED"))
    h = jid(sh("sbatch --parsable -H --wrap true").stdout)
    time.sleep(0.6)
    ok("held job pends (JobHeldUser)", sh(f"squeue -h -j {h} -o %R").stdout.strip() == "(JobHeldUser)")
    sh(f"scontrol release {h}")
    ok("released job completes", wait_state(h, ("COMPLETED",)) == "COMPLETED")

    # 8. #SBATCH directives in a script, script args, -o pattern, --chdir
    open(os.path.join(work, "job.sh"), "w").write("#!/bin/bash\n#SBATCH -J scripted\n#SBATCH --gres=gpu:2\n"
                                                    "#SBATCH -o logs/%x-%j.log\necho args=$1,$2 gpus=$CUDA_VISIBLE_DEVICES\n")
    os.makedirs(os.path.join(work, "logs"), exist_ok=True)
    s = jid(sh("sbatch --parsable job.sh foo bar").stdout)
    wait_state(s, ("COMPLETED",))
    lp = os.path.join(work, "logs", f"scripted-{s}.log")
    txt = open(lp).read().strip() if os.path.exists(lp) else ""
    ok("#SBATCH directives, -o %x-%j and script args", txt.startswith("args=foo,bar gpus=") and len(txt.split("gpus=")[1].split(",")) == 2, txt)

    # 9. formats
    q = sh("sbatch --parsable --gres=gpu:1 -J fmt --wrap 'sleep 3'")
    time.sleep(0.6)
    out = sh("squeue").stdout.splitlines()
    ok("squeue default header", out and out[0].split() == ["JOBID", "PARTITION", "NAME", "USER", "ST", "TIME", "NODES", "NODELIST(REASON)"], out[:1])
    ok("squeue running row shows node name", any("gpu-node-01" in l and " R " in l for l in out[1:]), out)
    si = sh("sinfo").stdout.splitlines()
    ok("sinfo default", si[0].split() == ["PARTITION", "AVAIL", "TIMELIMIT", "NODES", "STATE", "NODELIST"] and si[1].startswith("gpu*"), si)
    sc = sh(f"scontrol show job {jid(q.stdout)}").stdout
    ok("scontrol show job has JobState and TresPerNode", "JobState=RUNNING" in sc and "TresPerNode=gres/gpu:1" in sc, sc[:200])
    sa = sh("sacct -X --format=JobID,JobName,State,ExitCode").stdout.splitlines()
    ok("sacct table header and dashes", sa[0].split() == ["JobID", "JobName", "State", "ExitCode"] and set(sa[1].replace(" ", "")) == {"-"}, sa[:2])
    r = sh("sbatch --gres=gpu:5 --wrap true", check=False)
    ok("more GPUs than the node has is rejected", r.returncode != 0 and "Requested node configuration is not available" in r.stderr, r.stderr)

    # 10. srun in a job and as an allocation
    r = sh("srun --gres=gpu:1 bash -c 'echo $CUDA_VISIBLE_DEVICES $SLURM_JOB_ID'")
    ok("srun allocates a GPU and runs the command", r.returncode == 0 and len(r.stdout.split()) == 2, (r.stdout, r.stderr))
    # an srun killed outright (SIGKILL: no chance to report) frees its allocation; a cancelled srun stops its command
    sp = subprocess.Popen("exec srun --gres=gpu:1 -J held1 sleep 60", shell=True, cwd=work, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        held = sh("squeue -h -n held1 -o '%i %t'").stdout.split()
        if held[1:2] == ["R"]:
            break
        time.sleep(0.2)
    sp.kill(); sp.wait()
    st = wait_state(held[0], ("CANCELLED", "FAILED", "COMPLETED"), t=5) if held else "none"
    ok("a killed srun's job ends and frees its GPUs", held[1:2] == ["R"] and st.startswith("CANCELLED"), (held, st))
    sp = subprocess.Popen("exec srun --gres=gpu:1 -J cxl sleep 60", shell=True, cwd=work, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        cj, _, st = sh("squeue -h -n cxl -o '%i %t'").stdout.strip().partition(" ")
        if st == "R":
            break
        time.sleep(0.2)
    sh(f"scancel {cj}")
    try:
        sp.wait(timeout=5)
        stopped = True
    except subprocess.TimeoutExpired:
        sp.kill(); stopped = False
    ok("scancel of an srun job stops srun's command", stopped and state(cj).startswith("CANCELLED"), (stopped, state(cj)))

    # srun inside a batch job runs the step in place, on the job's GPUs; options clusters' scripts carry are accepted
    r = sh("sbatch --parsable --gres=gpu:2 --cpus-per-gpu=3 --time-min=5 --hint=nomultithread -J instep "
           "--wrap 'srun bash -c \"echo step \\$CUDA_VISIBLE_DEVICES \\$SLURM_STEP_ID\"'")
    ij = r.stdout.strip()
    st = wait_state(ij, ("COMPLETED", "FAILED"))
    out = read_when(f"slurm-{ij}.out")
    ok("srun inside a job runs its step on the job's GPUs", st == "COMPLETED" and re.fullmatch(r"step \d,\d 0", out), (st, out, r.stderr))
    ok("--cpus-per-gpu sets the job's CPUs", sh(f"sacct -j {ij} -X -n -P --format=AllocCPUS").stdout.strip() == "6",
       sh(f"sacct -j {ij} -X -n -P --format=AllocCPUS").stdout)

    # 11. the event log
    ev = [l for l in open(os.path.join(spool, "sched-events.jsonl"))]
    kinds = set(__import__("json").loads(l)["ev"] for l in ev)
    ok("event log has submit, eligible, start, end, pending_reason", {"submit", "eligible", "start", "end", "pending_reason"} <= kinds, kinds)
    time.sleep(4)
    act = open(os.path.join(tmp, "active")).read().strip() if os.path.exists(os.path.join(tmp, "active")) else "missing"
    ok("active file reads 0 once the queue is empty", act == "0", act)
    # a daemonized start (no -D) returns only once its socket answers, so an sbatch right after it cannot miss it
    s2 = os.path.join(tmp, "d.sock")
    e2 = dict(env, SLURMCTLD_SOCKET=s2)
    r = subprocess.run([os.path.join(ROOT, "bin", "slurmctld"), "--spool", os.path.join(tmp, "spool2"),
                        "--socket", s2, "--first-job-id", "9000"], env=e2, capture_output=True, text=True, timeout=20)
    q = subprocess.run("sinfo -h -o %G; scontrol shutdown", shell=True, env=e2, capture_output=True, text=True)
    ev2 = [__import__("json").loads(l) for l in open(os.path.join(tmp, "spool2", "sched-events.jsonl"))]
    ok("a spool that runs scripts gives no controller_error at start", not [e for e in ev2 if e["ev"] == "controller_error"], ev2[:3])
    ok("daemonized slurmctld is ready when it returns", r.returncode == 0 and q.returncode == 0 and "gpu:" in q.stdout,
       (r.returncode, q.stdout, q.stderr))
    # the node's GPU type comes from slurm.conf's gres line when nothing overrides it, as SLURM takes it from the config
    conf = os.path.join(tmp, "a100.conf")
    open(conf, "w").write("GresTypes=gpu\nNodeName=DEFAULT Gres=gpu:a100:4 State=UNKNOWN\n")
    s3 = os.path.join(tmp, "t.sock")
    e3 = {k: v for k, v in dict(env, SLURMCTLD_SOCKET=s3, SLURM_CONF=conf).items() if k != "SLURM_NODE_GPU_TYPE"}
    r = subprocess.run([os.path.join(ROOT, "bin", "slurmctld"), "--spool", os.path.join(tmp, "spool3"), "--gpus", "4",
                        "--socket", s3, "--first-job-id", "9500"], env=e3, capture_output=True, text=True, timeout=20)
    q = subprocess.run("sinfo -h -o %G; scontrol shutdown", shell=True, env=e3, capture_output=True, text=True)
    ok("the GPU type comes from slurm.conf's gres line", q.stdout.strip() == "gpu:a100:4", (r.returncode, q.stdout, q.stderr))
    # an 8-GPU node: eight one-GPU jobs start together and a ninth waits for Resources, each on its own GPU
    s4 = os.path.join(tmp, "g8.sock")
    e4 = dict(env, SLURMCTLD_SOCKET=s4)
    r = subprocess.run([os.path.join(ROOT, "bin", "slurmctld"), "--spool", os.path.join(tmp, "spool4"), "--gpus", "8",
                        "--cpus", "16", "--mem-mb", "32000", "--socket", s4, "--first-job-id", "9800"], env=e4,
                       capture_output=True, text=True, timeout=20)
    q = subprocess.run("sinfo -h -o %G", shell=True, env=e4, capture_output=True, text=True)
    ok("an 8-GPU node shows gpu:h100:8", q.stdout.strip() == "gpu:h100:8", (r.returncode, q.stdout, q.stderr))
    subprocess.run("for i in 1 2 3 4 5 6 7 8 9; do sbatch -Q --gres=gpu:1 -J g8 --wrap 'sleep 30'; done", shell=True,
                   cwd=work, env=e4, capture_output=True, text=True)
    st8 = ""
    for _ in range(40):
        st8 = subprocess.run("squeue -h -o '%t %r %b'", shell=True, env=e4, capture_output=True, text=True).stdout
        if st8.count("R ") >= 8:
            break
        time.sleep(0.25)
    lines8 = [l.split() for l in st8.strip().splitlines()]
    ok("8-GPU node: eight run, one waits for Resources",
       sum(1 for l in lines8 if l[0] == "R") == 8 and sum(1 for l in lines8 if l[0] == "PD" and l[1] == "Resources") == 1, st8)
    ev8 = [__import__("json").loads(l) for l in open(os.path.join(tmp, "spool4", "sched-events.jsonl"))]
    g8 = sorted(g for e in ev8 if e["ev"] == "start" for g in e.get("gpus") or [])
    ok("8-GPU node: the eight started jobs hold GPUs 0-7, one each", g8 == list(range(8)), g8)
    subprocess.run("scancel -u $(id -un); sleep 1; scontrol shutdown", shell=True, env=e4, capture_output=True, text=True)
    r = sh("scontrol shutdown")
    time.sleep(0.6)
    ok("scontrol shutdown stops the controller", ctl.poll() is not None, ctl.poll())
except Exception as e:                     # a check that raises is a failure, not the end of the suite with a pass
    import traceback
    traceback.print_exc()
    fails.append(f"raised {e!r}")
finally:
    ctl.terminate()
    ctl.wait()
    print(f"\n{len(fails)} failed" if fails else "\nall passed")
    shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if fails else 0)
