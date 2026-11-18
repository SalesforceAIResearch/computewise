# usage: python3 -I isolation/tests/idle_after_harness_known_answer.py <experiments dir> [<collected run dir>...]
# Known answer for idle_after_harness's scope, on the real scorer and real task files, with fake time, cgroup and /proc.
import importlib.util, json, os, shutil, sys, tempfile
EXP = sys.argv[1]
spec = importlib.util.spec_from_file_location("ro", os.path.join(EXP, "isolation", "run_one.py")); ro = importlib.util.module_from_spec(spec); spec.loader.exec_module(ro)
clock = [1000.0]; usec = [0]; cg = tempfile.mkdtemp()
ro.time.time = lambda: clock[0]; ro.cgroup_path = lambda cid: cg
def tick(dt):
    clock[0] += dt; open(os.path.join(cg, "cpu.stat"), "w").write(f"usage_usec {usec[0]}\n")
real_open = open
NS = {4242: 31, 4343: 207}   # host pid -> box pid; 4343 stands for a live rank whose job_hello pid is 207
def fake_open(path, *a, **k):
    import io
    if isinstance(path, str) and path.startswith("/proc/") and path.endswith("/cmdline"):
        return io.BytesIO(CMD.encode())
    if isinstance(path, str) and path.startswith("/proc/") and path.endswith("/status"):
        return io.StringIO(f"Name:\tx\nNSpid:\t{path.split('/')[2]}\t{NS.get(int(path.split('/')[2]), 1)}\n")
    return real_open(path, *a, **k)
ro.open = fake_open
def case(label, task, cmd):
    global CMD; CMD = cmd
    out = tempfile.mkdtemp()
    shutil.copy(os.path.join(EXP, "gpu-tasks", "registry", "tasks", task + ".json"), os.path.join(out, "task.json"))
    open(os.path.join(out, "occupancy.jsonl"), "w").close()   # no job ever ran: a fixed list has work remaining
    state, status = {"cid": "x"}, {"harness_end_wall": clock[0]}
    fired = None
    for i in range(400):
        tick(1.0)
        if ro.idle_after_harness(state, status, out, [4242]):
            fired = i; break
    print(f"{label}: ends at {fired} s; work={status.get('held_after_harness_work')} launch_like={status.get('held_after_harness_launch_like')} held_s={status.get('held_after_harness_s')}")
case("open pool, idle poll", "goal-pass-bar", "bash -c while pgrep -f x; do sleep 20; done; torchrun a.py")
case("fixed list, work remaining, launch-like queue", "pkg-p6-par-q", "bash -c sleep 1500; torchrun --nproc_per_node=1 contra-muon-to-soft-muon.py --seed 0")
case("fixed list, work remaining, plain sleep", "pkg-p6-par-q", "sleep 99999")
def case_run(label, rundir, cmd):
    global CMD; CMD = cmd
    out = tempfile.mkdtemp()
    for f in ("task.json", "occupancy.jsonl"):
        shutil.copy(os.path.join(rundir, f), os.path.join(out, f))
    state, status = {"cid": "x"}, {"harness_end_wall": clock[0]}
    fired = None
    for i in range(400):
        tick(1.0)
        if ro.idle_after_harness(state, status, out, [4242]):
            fired = i; break
    print(f"{label}: ends at {fired} s; work={status.get('held_after_harness_work')} launch_like={status.get('held_after_harness_launch_like')}")
for r in sys.argv[2:]:
    case_run(f"collected {os.path.basename(r.rstrip('/'))}, launch-like queue", r, "bash -c sleep 1500; torchrun contra-muon-to-soft-muon.py --seed 0")

def case_leak(label, procs):
    global CMD; CMD = "bash -c while pgrep -f x; do sleep 20; done"
    out = tempfile.mkdtemp()
    shutil.copy(os.path.join(EXP, "gpu-tasks", "registry", "tasks", "goal-pass-bar.json"), os.path.join(out, "task.json"))
    open(os.path.join(out, "occupancy.jsonl"), "w").write(json.dumps({"kind": "job_hello", "job": 9, "pid": 207}) + "\n")
    state, status = {"cid": "x"}, {"harness_end_wall": clock[0]}
    fired = None
    for i in range(400):
        tick(1.0)
        if ro.idle_after_harness(state, status, out, procs):
            fired = i; break
    print(f"{label}: ends at {fired} s; leaked={status.get('after_harness_leaked_jobs')}")
case_leak("job_hello without job_bye, its rank gone (leaked connection)", [4242])
case_leak("job_hello without job_bye, its rank alive", [4242, 4343])
