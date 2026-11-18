"""Start a job in its own process group, check it from the machine, stop the whole group.

Every job runs as

    bash -c '<command>'; echo $? > rc     in a new session (its own process group), with
    CUDA_VISIBLE_DEVICES set to its one GPU (empty for a CPU-only job), stdout and stderr to its log.

so three things hold that prose instructions could not keep:

  * The pid the harness records leads the job's process group. A job that is a wrapper script around the real
    program (`bash scripts/eval.sh` -> `python eval.py`) is stopped as a whole: the group is signalled, not the pid.
  * Liveness is read from the machine, not from notes. A job is alive while any process of its group is alive.
  * The exit code is written by the job's own shell to a file, so the harness learns how a job ended even if the
    harness itself was restarted. A job that vanished without writing it was killed from outside: DIED.

Stopping is TERM to the group, a grace period, then KILL to the group, and the GPU is released only once nothing of
the group is left. Descendants that moved to a new group are found through the process table and signalled too.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

KILL_GRACE_S = 10.0


@dataclass
class Started:
    pid: int
    log: str
    rc_path: str
    started_at: float


def start(job_id: str, command: str, gpu: Optional[int], cwd: str, logdir: str, env: Optional[Dict[str, str]] = None,
          gpu_env: str = "CUDA_VISIBLE_DEVICES") -> Started:
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, f"{job_id}.log")
    rc_path = os.path.join(logdir, f"{job_id}.rc")
    for p in (rc_path,):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
    full_env = dict(os.environ)
    full_env.update(env or {})
    full_env[gpu_env] = "" if gpu is None else str(gpu)   # a CPU-only job sees no GPU at all
    full_env["GPUH_JOB"] = job_id
    # the command runs in a child shell, so an `exit` or `exec` inside it can't skip the line that records its exit code
    script = (f"bash -c {shlex.quote(command)}\nrc=$?\n"
              f"echo $rc > {shlex.quote(rc_path)}.tmp && mv {shlex.quote(rc_path)}.tmp {shlex.quote(rc_path)}\nexit $rc")
    with open(log, "ab", buffering=0) as out, open(os.devnull, "rb") as devnull:
        out.write(f"[gpuh] start {job_id} {'cpu' if gpu is None else f'gpu={gpu}'} at "
                  f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\n".encode())
        proc = subprocess.Popen(["bash", "-c", script], cwd=cwd, env=full_env, stdin=devnull, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    return Started(pid=proc.pid, log=log, rc_path=rc_path, started_at=time.time())


def _pgid_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie is not alive for our purposes
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split(") ", 1)[1][0] != "Z"
    except (OSError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout
        return bool(out.strip()) and not out.strip().startswith("Z")
    except (OSError, subprocess.SubprocessError):
        return True


def descendants(pid: int) -> List[int]:
    """Every live descendant of pid in the process table, whatever group it moved to."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: Dict[int, List[int]] = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    seen: List[int] = []
    stack = list(children.get(pid, []))
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.append(p)
        stack.extend(children.get(p, []))
    return seen


def cpu_seconds(pgid: int) -> Optional[float]:
    """CPU seconds the job's process group (and descendants that left it) have used so far, from the process table.

    A job whose log is silent but whose processes keep using CPU is working: a GPU job's host side always does (the
    framework's launch loop, data loading). None if the table cannot be read."""
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,pgid=,time="], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    rows, children = {}, {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) != 4 or not (parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit()):
            continue
        rows[int(parts[0])] = (int(parts[2]), _clock(parts[3]))
        children.setdefault(int(parts[1]), []).append(int(parts[0]))
    pids = {p for p, (g, _) in rows.items() if g == pgid}
    stack = list(pids)
    while stack:                                   # descendants that moved to another group still count
        for c in children.get(stack.pop(), []):
            if c not in pids:
                pids.add(c); stack.append(c)
    return sum(rows[p][1] for p in pids if p in rows) if pids else None


def _clock(t: str) -> float:
    """`ps -o time` ([[dd-]hh:]mm:ss[.cc]) to seconds."""
    days = 0
    if "-" in t:
        d, t = t.split("-", 1)
        days = int(d) if d.isdigit() else 0
    parts = t.split(":")
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return 0.0
    while len(nums) < 3:
        nums.insert(0, 0.0)
    return days * 86400 + nums[-3] * 3600 + nums[-2] * 60 + nums[-1]


def group_members(pgid: int) -> List[int]:
    try:
        out = subprocess.run(["ps", "-A", "-o", "pid=,pgid="], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    pids = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) == pgid and parts[0].isdigit():
            pids.append(int(parts[0]))
    return pids


def alive(pid: int) -> bool:
    """True while the job's process group has any live member."""
    if _pid_alive(pid):
        return True
    return any(_pid_alive(p) for p in group_members(pid))


def reap(pid: int) -> None:
    """Collect the job's exit status if it is our child, so it does not linger as a zombie."""
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    except OSError:
        pass


def exit_code(rc_path: str) -> Optional[int]:
    try:
        with open(rc_path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def signal_group(pid: int, sig: int) -> None:
    """Signal the job's process group and every descendant that left it."""
    for p in descendants(pid):
        try:
            os.kill(p, sig)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def terminate(pid: int) -> None:
    signal_group(pid, signal.SIGTERM)


def kill(pid: int) -> None:
    signal_group(pid, signal.SIGKILL)


def gpu_processes(gpu_query: str = "nvidia-smi") -> Optional[Dict[int, List[int]]]:
    """GPU index -> pids using it, from nvidia-smi, or None when it can't be read. Used for status views only."""
    try:
        uuid_out = subprocess.run([gpu_query, "--query-gpu=index,uuid", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10)
        app_out = subprocess.run([gpu_query, "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    if uuid_out.returncode != 0 or app_out.returncode != 0:
        return None
    idx = {}
    for line in uuid_out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            idx[parts[1]] = int(parts[0])
    used: Dict[int, List[int]] = {i: [] for i in idx.values()}
    for line in app_out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] in idx and parts[1].isdigit():
            used[idx[parts[0]]].append(int(parts[1]))
    return used


def detect_gpus(gpu_query: str = "nvidia-smi") -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip().isdigit()]
    try:
        out = subprocess.run([gpu_query, "-L"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [i for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]
