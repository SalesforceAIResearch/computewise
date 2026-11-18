"""Stress check for the controller's wake path (2026-11-04, G-k-haiku-S1-r2's scancel hang): arrays whose tasks fail at
once (exit 127, as an exec of a missing command does), throttled to the node's GPUs, then scancel of each array while it
is still failing. Every scancel must return, and the controller must keep answering. Before the fix the controller's
main thread could deadlock in its SIGCHLD handler on a burst of job exits.
usage: python3 stress_fastfail.py [rounds] [tasks per array]   (exit 1 on a hang)
  In a box image (installed commands, a root controller, jobs owned by another uid as in a run):
  SCHED_CTLD=/usr/sbin/slurmctld SCHED_BIN=/usr/bin SLURMSTEPD=/usr/sbin/slurmstepd SCHED_LIBDIR=<library dir>
  SCHED_SUBMIT_AS=1000:1000 python3 -I stress_fastfail.py"""
import os, shutil, subprocess, sys, tempfile, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("SCHED_ROOT") or os.path.dirname(HERE)
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 40
tasks = int(sys.argv[2]) if len(sys.argv) > 2 else 200
tmp = tempfile.mkdtemp(prefix="sched-stress-")
os.chmod(tmp, 0o1777)
env = dict(os.environ, SLURMCTLD_SOCKET=os.path.join(tmp, "ctld.sock"),
           SLURM_LIBDIR=os.environ.get("SCHED_LIBDIR") or os.path.join(ROOT, "slurm"),
           SLURMSTEPD=os.environ.get("SLURMSTEPD") or os.path.join(ROOT, "bin", "slurmstepd"),
           PATH=(os.environ.get("SCHED_BIN") or os.path.join(ROOT, "bin")) + ":" + os.environ["PATH"])
AS = os.environ.get("SCHED_SUBMIT_AS")
pre = (f"setpriv --reuid={AS.split(':')[0]} --regid={AS.split(':')[1]} --clear-groups " if AS else "")
work = os.path.join(tmp, "work"); os.makedirs(work)
ctl = subprocess.Popen([os.environ.get("SCHED_CTLD") or os.path.join(ROOT, "bin", "slurmctld"), "-D", "--gpus", "4", "--cpus", "8", "--mem-mb", "16000",
                        "--spool", os.path.join(tmp, "spool"), "--socket", env["SLURMCTLD_SOCKET"],
                        "--first-job-id", "1000"], env=env)
hung = None
try:
    for _ in range(50):
        if subprocess.run("sinfo -h", shell=True, env=env, capture_output=True).returncode == 0:
            break
        time.sleep(0.1)
    t0 = time.time()
    for k in range(rounds):
        r = subprocess.run(f"{pre}sbatch --parsable --gres=gpu:1 --array=1-{tasks}%4 -o /dev/null --wrap 'exec nosuchcmd'",
                           shell=True, cwd=work, env=env, capture_output=True, text=True, timeout=20)
        aid = r.stdout.strip().split(";")[0]
        time.sleep(0.05 * (k % 7))                     # cancel at different points of the failing burst
        try:
            c = subprocess.run(f"{pre}scancel {aid}", shell=True, env=env, capture_output=True, text=True, timeout=15)
            q = subprocess.run("squeue -h", shell=True, env=env, capture_output=True, text=True, timeout=15)
        except subprocess.TimeoutExpired as e:
            hung = f"round {k + 1}: `{e.cmd}` did not return in {e.timeout:.0f} s"
            break
        if c.returncode != 0 or q.returncode != 0:
            hung = f"round {k + 1}: scancel rc {c.returncode} {c.stderr.strip()}; squeue rc {q.returncode}"
            break
    print(f"{'HANG ' + hung if hung else 'ok'}: {k + 1 if hung else rounds} rounds of {tasks} fast-failing tasks, "
          f"{time.time() - t0:.1f} s, library {env['SLURM_LIBDIR']}{', jobs as ' + AS if AS else ''}")
finally:
    ctl.kill()
    ctl.wait()
    shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if hung else 0)
