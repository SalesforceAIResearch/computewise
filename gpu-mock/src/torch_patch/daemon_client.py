"""Client the torch patch uses to talk to the gpu-mock daemon.

Newline-delimited JSON over AF_UNIX. One connection per job process, held for
the life of the process. All methods are synchronous request/response; the
daemon never blocks on us (sync returns a sleep duration we sleep locally), so a
slow client cannot stall other jobs.

This module has no torch dependency so it can be tested on its own.
"""

from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Optional

DEFAULT_SOCKET = "/var/run/nvidia-persistenced/socket"


def _genuine(name: str):
    """The genuine time function `name`: virtual_clock's capture when the patch
    is loaded (this module may be imported after it rebinds `time`), else the
    time module's own. No torch import here."""
    import sys
    vc = sys.modules.get("virtual_clock")
    real = getattr(vc, "_REAL", None) if vc is not None else None
    return (real or {}).get(name) or getattr(time, name)


def process_start_time() -> Optional[float]:
    """When this process started, as wall time, from /proc on Linux. A job says
    hello at its first CUDA use, not at interpreter start, so the daemon logs
    this with job_hello to keep the job's real beginning (the runner measures
    host time before the first enqueue from it)."""
    try:
        with open("/proc/self/stat") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])          # field 22, after pid and comm
        with open("/proc/uptime") as fh:
            uptime = float(fh.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        return time.time() - uptime + start_ticks / hz
    except (OSError, ValueError, IndexError):
        return None


def job_key(env: Optional[dict] = None, pid: Optional[int] = None,
            ppid: Optional[int] = None, t_start: Optional[float] = None) -> str:
    """The identity under which the daemon makes ONE draw per job and hands it
    to every rank of that job (lookup caches by it).
    - A launched multi-rank job (torchrun, accelerate, deepspeed): its ranks
      share the launcher as parent and a run id, so the key is the parent pid,
      the world size, the master port and the run id, as before.
    - A launcher-less process is a job of its own, so the key is its own pid
      and start time. The old key for it was the parent pid plus three empty
      fields. Three `nohup python sft.py --seed N &` launched from one shell
      that then exited were all reparented to pid 1, shared the key `1:1::`,
      and were served the first one's draw, so every seed printed the same
      losses. The start time keeps a reused pid from
      colliding with an earlier job's key within one daemon."""
    e = os.environ if env is None else env
    ws = (e.get("WORLD_SIZE") or "").strip()
    run_id = (e.get("TORCHELASTIC_RUN_ID") or "").strip()
    launched = bool(run_id) or (ws.isdigit() and int(ws) > 1)
    if launched:
        return f"{os.getppid() if ppid is None else ppid}:{ws or '1'}:{e.get('MASTER_PORT', '')}:{run_id}"
    ts = process_start_time() if t_start is None else t_start
    return f"pid:{os.getpid() if pid is None else pid}:{'' if ts is None else round(float(ts), 3)}"


def visible_physical(cvd: Optional[str], count: int) -> list:
    """The physical GPU indices CUDA_VISIBLE_DEVICES exposes, in order, parsed as
    a real driver does: unset means all, empty means none, and the list stops at
    the first entry that is not an integer in range. The one parser for this
    variable: cuda_hooks._fix_visible_devices counts devices with it."""
    if cvd is None:
        return list(range(count))
    out = []
    for tok in (cvd.split(",") if cvd.strip() else []):
        tok = tok.strip()
        if not tok.lstrip("-").isdigit():
            break
        idx = int(tok)
        if idx < 0 or idx >= count:
            break
        out.append(idx)
    return out


def physical_gpu_index(local: int, cvd: Optional[str], count: int) -> int:
    """The daemon's (physical) index of the GPU a process works on, given the
    process's LOCAL device index. Under CUDA_VISIBLE_DEVICES=2,3 a process's
    cuda:0 is physical GPU 2. A launcher that isolates each rank
    (CUDA_VISIBLE_DEVICES=k with LOCAL_RANK=k) leaves one visible device, which
    is that rank's. Never returns an index the daemon does not have."""
    vis = visible_physical(cvd, count)
    if vis:
        if 0 <= local < len(vis):
            return vis[local]
        if len(vis) == 1:
            return vis[0]
    return local % count if count > 0 else 0


class DaemonClient:
    def __init__(self, socket_path: Optional[str] = None,
                 connect_timeout: float = 5.0):
        # Explicit path wins; otherwise env var; otherwise the default node path.
        self.socket_path = (socket_path
                            or os.environ.get("NVRT_SOCKET")
                            or DEFAULT_SOCKET)
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(connect_timeout)
        self._sock.connect(self.socket_path)
        self._sock.settimeout(None)
        self._f = self._sock.makefile("rwb", buffering=0)
        self.job_id: Optional[int] = None

    def _rpc(self, obj: dict) -> dict:
        self._f.write((json.dumps(obj, separators=(",", ":")) + "\n").encode())
        line = self._f.readline()
        if not line:
            raise ConnectionError("daemon closed connection")
        return json.loads(line.decode("utf-8"))

    # -- lifecycle --------------------------------------------------------

    def hello(self, argv: Optional[list] = None, cwd: Optional[str] = None) -> int:
        import sys
        resp = self._rpc({
            "cmd": "hello",
            "pid": os.getpid(),
            "argv": argv if argv is not None else sys.argv,
            "cwd": cwd if cwd is not None else os.getcwd(),
            "t_start": process_start_time(),
        })
        self.job_id = resp["job_id"]
        # the speed knob is the daemon's (its --speed): the box carries no
        # variable for it, so the clock learns it here (env-name scrub, 2026-10-21)
        try:
            if resp.get("speed") is not None:
                import virtual_clock as vc
                vc.set_speed(float(resp["speed"]))
        except Exception:
            pass
        return self.job_id

    def lookup(self, fingerprint: Optional[str] = None,
               entry_hash: Optional[str] = None,
               argv: Optional[list] = None) -> dict:
        return self._rpc({"cmd": "lookup", "fingerprint": fingerprint,
                          "entry_hash": entry_hash, "argv": argv})

    def bye(self) -> None:
        try:
            self._rpc({"cmd": "bye"})
        except (ConnectionError, OSError):
            pass

    # -- pacing -----------------------------------------------------------

    def enqueue(self, gpu: int, op: str, modelled_s: float,
                mem_delta: int = 0, step: Optional[int] = None,
                unit: Optional[str] = None, lr: Optional[list] = None,
                startup_s: Optional[float] = None, startup_real_s: Optional[float] = None,
                host_gap_s: Optional[float] = None, oom_aware: bool = False,
                key: Optional[str] = None) -> dict:
        """`step` (the optimizer step this unit belongs to) and `unit`
        ("step" | "eval") are logged on the enqueue event, so a reader can tell
        from the occupancy log alone how far a job got (the runner's scorer,
        2026-10-21), without depending on what the script chose to read."""
        req = {"cmd": "enqueue", "gpu": gpu, "op": op,
               "modelled_s": modelled_s, "mem_delta": mem_delta}
        if step is not None:
            req["step"] = int(step)
        if unit is not None:
            req["unit"] = unit
        if startup_s is not None:
            # a per_example launch's start-up on the box's clock (the pacing identity's
            # term) and in real seconds, sent once with its first example
            req["startup_s"] = float(startup_s)
        if startup_real_s is not None:
            req["startup_real_s"] = float(startup_real_s)
        if host_gap_s is not None:
            # real host seconds since the previous example unit drained; the daemon
            # back-dates this unit by it (the record's wall holds host time once)
            req["host_gap_s"] = float(host_gap_s)
        if lr:
            req["lr"] = list(lr)      # the learning rate per param group at this step
        if oom_aware:
            req["oom_aware"] = True   # this patch raises on an oversubscribed GPU: the daemon may log the failure
        if key:
            req["key"] = key          # a per_example unit's example content key (the record position it decodes)
        return self._rpc(req)

    def exception(self, etype: str, message: str, step: int,
                  frame: Optional[dict] = None) -> None:
        """An uncaught exception is ending this process: logged by the daemon
        as an `exception` event, so a job that ends early carries its cause.
        `frame` is the innermost traceback frame ({file, line, func})."""
        req = {"cmd": "exception", "etype": str(etype)[:80],
               "message": str(message)[:300], "step": int(step)}
        if frame:
            req["frame_file"] = str(frame.get("file", ""))[:300]
            req["frame_line"] = int(frame.get("line") or 0)
            req["frame_func"] = str(frame.get("func", ""))[:120]
        try:
            self._rpc(req)
        except (ConnectionError, OSError):
            pass

    def hold(self, gpu: int, bytes: int, oom_aware: bool = False) -> dict:
        """Hold (negative: release) device memory on a GPU without queueing
        work — the record's weights from engagement."""
        req = {"cmd": "hold", "gpu": gpu, "bytes": int(bytes)}
        if oom_aware:
            req["oom_aware"] = True
        return self._rpc(req)

    def sync_wait_s(self, gpu: int = 0) -> float:
        """Ask how long until the GPU cursor passes, WITHOUT sleeping. The
        replay client sleeps with the genuine sleep: under the patched clock,
        time.sleep only advances virtual time and would not pace wall time."""
        resp = self._rpc({"cmd": "sync", "gpu": gpu})
        return float(resp.get("sleep_s", 0.0))

    def sync(self, gpu: int = 0) -> None:
        """Block the caller until the GPU cursor has passed. The daemon tells us
        how long; we wait locally so the daemon stays responsive, busy and
        yielding as a CUDA host thread does (replay._gpu_wait has the reason),
        on the genuine monotonic clock."""
        wait_s = self.sync_wait_s(gpu)
        if wait_s <= 0:
            return
        if os.environ.get("NVRT_WAIT") == "sleep":
            _genuine("sleep")(wait_s)
            return
        mono = _genuine("monotonic")
        end = mono() + wait_s
        while mono() < end:
            os.sched_yield()

    def lookup_v2(self, **fields) -> dict:
        """Registry lookup with the RECORD-FORMAT identity fields (job_key,
        rank, world_size, entry_hash, argv, env, fingerprint, stage)."""
        return self._rpc({"cmd": "lookup", **fields})

    # -- values and failures ---------------------------------------------

    def materialise(self, role: str, step: int, kind: str = "scalar", key: Optional[str] = None) -> Any:
        req = {"cmd": "materialise", "role": role, "step": step, "kind": kind}
        if key:
            req["key"] = key
        resp = self._rpc(req)
        return resp.get("value")

    def materialise_many(self, roles: list, step: int, pick: str = "covering") -> dict:
        """Every named series at one step in one round trip: {role: value|None}."""
        resp = self._rpc({"cmd": "materialise", "roles": list(roles), "step": step,
                          "pick": pick})
        return resp.get("values") or {}

    def pending_failure(self, step: int, phase: str = "training",
                        where: Optional[str] = None) -> Optional[dict]:
        req = {"cmd": "fail?", "step": step, "phase": phase}
        if where:
            req["where"] = where          # forward | backward | step: the seam asking
        resp = self._rpc(req)
        return resp.get("failure")

    def snapshot(self) -> dict:
        return self._rpc({"cmd": "snapshot"}).get("snapshot", {})

    def close(self) -> None:
        try:
            self._f.close()
            self._sock.close()
        except OSError:
            pass
