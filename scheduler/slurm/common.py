"""Shared pieces of the box's SLURM-compatible scheduler: time formats, option values, and the client side of the
controller socket. Formats follow SLURM 23.11's commands."""
import json
import os
import re
import socket
import time

VERSION = "23.11.4"
SOCKET = os.environ.get("SLURMCTLD_SOCKET", "/run/slurm/slurmctld.sock")
TERMINAL = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")
STATE_CODE = {"PENDING": "PD", "RUNNING": "R", "COMPLETING": "CG", "COMPLETED": "CD", "FAILED": "F",
              "CANCELLED": "CA", "TIMEOUT": "TO", "OUT_OF_MEMORY": "OOM", "NODE_FAIL": "NF"}


class ControllerError(Exception):
    pass


def request(op, timeout=None, **kw):
    """One request to the controller; returns its reply dict, or raises ControllerError on a connect failure."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    if timeout:
        s.settimeout(timeout)
    try:
        s.connect(SOCKET)
    except OSError:
        raise ControllerError("Unable to contact slurm controller (connect failure)")
    try:
        s.sendall((json.dumps(dict(kw, op=op)) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    if not buf:
        raise ControllerError("Unable to contact slurm controller (connect failure)")
    return json.loads(buf)


def parse_time_limit(v):
    """SLURM --time: minutes, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS; 0, UNLIMITED or INFINITE mean none.
    Returns seconds or None for unlimited; raises ValueError."""
    v = str(v).strip()
    if v.upper() in ("UNLIMITED", "INFINITE", "-1", "0"):
        return None
    days = 0
    if "-" in v:
        d, v = v.split("-", 1)
        days = int(d)
        parts = [int(p) for p in v.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, s = parts[:3]
        return days * 86400 + h * 3600 + m * 60 + s
    parts = [int(p) for p in v.split(":")]
    if len(parts) == 1:
        return parts[0] * 60
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise ValueError(v)


def fmt_limit(sec):
    if sec is None:
        return "UNLIMITED"
    return fmt_elapsed(sec)


def fmt_elapsed(sec):
    """sacct Elapsed / Timelimit: [D-]HH:MM:SS."""
    sec = int(max(0, sec or 0))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    return (f"{d}-" if d else "") + f"{h:02d}:{m:02d}:{s:02d}"


def fmt_squeue_time(sec):
    """squeue TIME: M:SS, H:MM:SS or D-HH:MM:SS."""
    sec = int(max(0, sec or 0))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}-{h:02d}:{m:02d}:{s:02d}"
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def fmt_ts(t):
    if not t:
        return "Unknown"
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t))


def parse_mem(v):
    """--mem style value to MB: 4000, 4G, 512M, 1T; 0 means all of the node."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([KMGT]?)B?", str(v).strip().upper())
    if not m:
        raise ValueError(v)
    n = float(m.group(1))
    return int(n * {"": 1, "K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[m.group(2)])


def fmt_mem(mb):
    if mb >= 1024 * 1024 and mb % (1024 * 1024) == 0:
        return f"{mb // (1024 * 1024)}T"
    if mb >= 1024 and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def parse_array(spec):
    """--array: '0-9', '1,3,5', '0-15:4', with an optional '%N' throttle. Returns (indices, throttle)."""
    throttle = None
    if "%" in spec:
        spec, t = spec.split("%", 1)
        throttle = int(t)
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if ":" in part:
            part, st = part.split(":", 1)
            step = int(st)
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1, step))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(spec)
    return sorted(set(out)), throttle


def compress_indices(idx):
    """[0,1,2,5,7,8] -> '0-2,5,7-8' (squeue's array range form)."""
    idx = sorted(idx)
    runs, start, prev = [], None, None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    if start is not None:
        runs.append((start, prev))
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in runs)


def job_display_id(j):
    if j.get("array_job_id") is not None:
        return f"{j['array_job_id']}_{j['array_task_id']}"
    return str(j["id"])
