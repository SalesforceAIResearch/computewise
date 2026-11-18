#!/usr/bin/env python3
# Samples one container's cgroup v2 files from outside the box, as JSON lines, for the run
# directory's resources.jsonl. It runs on the Docker host as root, beside the run command, so the
# agent never sees it or its output. Standard library only.
#
#   python3 -I cgroup_sampler.py --cgroup /sys/fs/cgroup/system.slice/docker-<id>.scope \
#       --out <run-dir>/resources.jsonl [--interval 1] [--disk <host mount of the box's disk>] \
#       [--container-id <id>] [--wait 120]
#
# The cgroup path depends on Docker's cgroup driver: system.slice/docker-<id>.scope under
# systemd (the GCP host), docker/<id> under cgroupfs (Colima).
#
# Lines, by kind:
#   limits   once, first: the caps as the kernel holds them, and what disk_used_bytes measures.
#   sample   every --interval seconds, and at once whenever memory.events changes (it is
#            checked four times a second), so a short memory spike that ends in a kill is not
#            missed between samples. memory.peak is a lifetime maximum on kernels before 6.12,
#            where it cannot be reset per interval. Counters are cumulative; consumers difference
#            them. Host-wide pressure and host meminfo (MemFree exact, MemAvailable heuristic) ride along, so a run made while other work
#            loaded the host can be flagged.
#   procs    the box's processes (host pid, ppid, comm, cmdline, rss, oom_score_adj): every 60 s the
#            full list (full: true); in between, whenever the set of pids changes, only the processes
#            that started and the pids that exited (full: false). Together they map a killed pid to
#            its job.
#   oom_kill one per kernel out-of-memory kill of a process in this container, read from
#            /dev/kmsg: which process died, its anon RSS and its oom_score_adj, and the kill's
#            scope: box (this container's limit), host (the whole host ran out: an environment
#            failure, not the agent's), or ancestor (a cgroup above the box hit its limit).
#            memory.events says that a kill happened, not which process died or why.
#   end      when the cgroup disappears; kmsg_overruns counts kernel-log records lost to overrun.
import argparse
import datetime
import errno
import json
import os
import re
import shutil
import time

MEMSTAT_KEYS = ("anon", "file", "shmem", "kernel", "sock", "pgmajfault", "pgscan", "pgsteal",
                "workingset_refault_anon", "workingset_refault_file")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def scalar(path):
    v = read(path)
    if v is None:
        return None
    v = v.strip()
    return int(v) if v.isdigit() else v


def keyed(path, keys=None):
    out = {}
    for line in (read(path) or "").splitlines():
        k, _, v = line.partition(" ")
        if (keys is None or k in keys) and v.strip().lstrip("-").isdigit():
            out[k] = int(v)
    return out


def pressure(path):
    out = {}
    for line in (read(path) or "").splitlines():
        kind, *fields = line.split()
        d = dict(f.split("=", 1) for f in fields if "=" in f)
        if "avg10" in d and "total" in d:
            out[kind] = {"avg10": float(d["avg10"]), "total_us": int(d["total"])}
    return out


def io_totals(cg):
    tot = {"rbytes": 0, "wbytes": 0, "rios": 0, "wios": 0}
    for line in (read(os.path.join(cg, "io.stat")) or "").splitlines():
        for f in line.split()[1:]:
            k, _, v = f.partition("=")
            if k in tot and v.isdigit():
                tot[k] += int(v)
    return tot


# MemFree is nr_free_pages, an exact counter, unlike the MemAvailable heuristic. Recording the
# category fields too lets a reader recompute "used pages in no meminfo category"
# (MemTotal - [MemFree + Buffers + Cached + AnonPages + Slab + KernelStack + PageTables + Percpu + VmallocUsed]),
# which is where the 2026-10-20 out-of-box drop lived. See isolation/notes/out-of-box-memory-2026-10-20.md.
_MEMINFO_FIELDS = ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "AnonPages",
                   "Shmem", "Slab", "SReclaimable", "SUnreclaim", "KReclaimable", "KernelStack",
                   "PageTables", "Percpu", "VmallocUsed", "Mlocked", "SwapTotal", "SwapFree")


def host_meminfo():
    m = {}
    for line in (read("/proc/meminfo") or "").splitlines():
        k, _, v = line.partition(":")
        if k in _MEMINFO_FIELDS:
            m[k + "_kb"] = int(v.split()[0])
    return m


def sample(cg, disk, reason):
    j = lambda n: os.path.join(cg, n)
    cpu = keyed(j("cpu.stat"))
    rec = {
        "kind": "sample", "t_utc": now(), "t_wall": time.time(), "reason": reason,
        "cpu_usage_usec": cpu.get("usage_usec"), "cpu_user_usec": cpu.get("user_usec"),
        "cpu_system_usec": cpu.get("system_usec"), "cpu_nr_throttled": cpu.get("nr_throttled"),
        "cpu_throttled_usec": cpu.get("throttled_usec"),
        "memory_current": scalar(j("memory.current")), "memory_peak": scalar(j("memory.peak")),
        "memory_swap_current": scalar(j("memory.swap.current")),
        "memory_events": keyed(j("memory.events")),
        "memory_stat": keyed(j("memory.stat"), MEMSTAT_KEYS),
        "io": io_totals(cg), "pids_current": scalar(j("pids.current")),
        "pressure": {r: pressure(j(f"{r}.pressure")) for r in ("cpu", "memory", "io")},
        "host_pressure": {r: pressure(f"/proc/pressure/{r}") for r in ("cpu", "memory", "io")},
        "host_mem": host_meminfo(),
    }
    if disk:
        try:
            u = shutil.disk_usage(disk)
            rec["disk_used_bytes"], rec["disk_total_bytes"] = u.used, u.total
        except OSError:
            pass
    return rec


def procs(cg, only=None):
    out = []
    for p in (read(os.path.join(cg, "cgroup.procs")) or "").split():
        if only is not None and int(p) not in only:
            continue
        st = read(f"/proc/{p}/status") or ""
        f = dict((l.split(":", 1)[0], l.split(":", 1)[1].strip()) for l in st.splitlines() if ":" in l)
        cmd = (read(f"/proc/{p}/cmdline") or "").replace("\0", " ").strip()
        out.append({"pid": int(p), "ppid": int(f.get("PPid", "0") or 0), "comm": f.get("Name"),
                    "cmdline": cmd[:400], "rss_kb": int((f.get("VmRSS", "0 kB").split() or ["0"])[0]),
                    "oom_score_adj": scalar(f"/proc/{p}/oom_score_adj")})
    return out


class Kmsg:
    """Non-blocking reader of /dev/kmsg, from now on, keeping OOM kills of this container's processes.

    The kernel reports each kill as two records: an `oom-kill:` context line, then `Killed process`.
    The context line names the cgroup whose limit was hit (`oom_memcg=`) or says `global_oom` when the
    whole host ran out, and always names the victim's cgroup (`task_memcg=`). A kill counts as this
    box's when the victim's cgroup is this container's; its scope is `box` when this container's own
    limit was hit, `host` for a host-wide OOM, and `ancestor` when a cgroup above the box hit its
    limit. A host-scope kill is an environment failure, not the agent's overshoot, and memory.events
    alone cannot tell it apart, because the kernel raises oom_kill against the victim's cgroup either
    way. `overruns` counts reads that found the ring buffer had overrun this reader, since a context
    line lost to an overrun means its kill is dropped."""
    KILLED = re.compile(r"Killed process (\d+) \(([^)]*)\) total-vm:(\d+)kB, anon-rss:(\d+)kB.*?oom_score_adj:(-?\d+)")
    FIELD = {k: re.compile(k + r"=([^,]+)") for k in ("oom_memcg", "task_memcg", "pid", "task", "constraint")}

    def __init__(self, cid):
        self.cid, self.fd, self.pending, self.overruns = cid, None, None, 0
        try:
            self.fd = os.open("/dev/kmsg", os.O_RDONLY | os.O_NONBLOCK)
            os.lseek(self.fd, 0, os.SEEK_END)
        except OSError:
            self.fd = None

    def feed(self, msg, ts_us=None):
        """Takes one kmsg record's message text; returns an oom_kill record, or None."""
        if "oom-kill:" in msg:
            f = {k: (r.search(msg).group(1) if r.search(msg) else None) for k, r in self.FIELD.items()}
            if f["task_memcg"] and self.cid in f["task_memcg"]:
                if f["oom_memcg"] and self.cid in f["oom_memcg"]:
                    scope = "box"
                elif "global_oom" in msg:
                    scope = "host"
                else:
                    scope = "ancestor"
                self.pending = {"scope": scope, "oom_memcg": f["oom_memcg"], "constraint": f["constraint"],
                                "pid": int(f["pid"]) if f["pid"] and f["pid"].isdigit() else None}
            else:
                self.pending = None
            return None
        m = self.KILLED.search(msg)
        if not (m and self.pending):
            return None
        ctx, self.pending = self.pending, None
        if ctx["pid"] is not None and ctx["pid"] != int(m.group(1)):
            return None
        return {"kind": "oom_kill", "t_utc": now(), "kernel_ts_us": ts_us, "scope": ctx["scope"],
                "oom_memcg": ctx["oom_memcg"], "constraint": ctx["constraint"],
                "pid": int(m.group(1)), "comm": m.group(2), "total_vm_kb": int(m.group(3)),
                "anon_rss_kb": int(m.group(4)), "oom_score_adj": int(m.group(5)), "line": msg.strip()[:400]}

    def poll(self):
        out = []
        while self.fd is not None:
            try:
                raw = os.read(self.fd, 8192).decode("utf-8", "replace")
            except OSError as e:
                if e.errno == errno.EAGAIN:
                    break
                if e.errno == errno.EPIPE:  # the ring buffer overran us; the next read resumes
                    self.overruns += 1
                    self.pending = None
                    continue
                break
            hdr, _, msg = raw.partition(";")
            parts = hdr.split(",")
            r = self.feed(msg, int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None)
            if r:
                out.append(r)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cgroup", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--disk")
    ap.add_argument("--container-id", default="")
    ap.add_argument("--wait", type=float, default=120.0)
    a = ap.parse_args()

    deadline = time.monotonic() + a.wait
    while not os.path.isdir(a.cgroup):
        if time.monotonic() > deadline:
            raise SystemExit(f"cgroup never appeared: {a.cgroup}")
        time.sleep(0.2)

    j = lambda n: os.path.join(a.cgroup, n)
    kmsg = Kmsg(a.container_id) if a.container_id else None
    with open(a.out, "a", buffering=1) as out:
        w = lambda r: out.write(json.dumps(r) + "\n")
        w({"kind": "limits", "t_utc": now(), "cgroup": a.cgroup,
           "cpu_max": scalar(j("cpu.max")), "cpuset_cpus": scalar(j("cpuset.cpus.effective")),
           "memory_max": scalar(j("memory.max")), "memory_high": scalar(j("memory.high")),
           "memory_swap_max": scalar(j("memory.swap.max")), "pids_max": scalar(j("pids.max")),
           "disk_source": "host-side statvfs of the run's loop-mounted ext4 image" if a.disk else None,
           "disk_path": a.disk, "interval_s": a.interval, "kmsg": bool(kmsg and kmsg.fd is not None)})
        next_sample = time.monotonic()
        next_procs = 0.0
        last_events = read(j("memory.events"))
        last_pids = None
        known = {}
        while os.path.isdir(a.cgroup):
            t = time.monotonic()
            ev = read(j("memory.events"))
            reason = None
            if t >= next_sample:
                reason = "interval"
                next_sample += a.interval
                if next_sample < t:
                    next_sample = t + a.interval
            if ev is not None and ev != last_events:
                reason = "memory_events"
                last_events = ev
            if reason:
                w(sample(a.cgroup, a.disk, reason))
                pids = read(j("cgroup.procs"))
                if t >= next_procs:
                    known = {p["pid"]: p for p in procs(a.cgroup)}
                    w({"kind": "procs", "t_utc": now(), "full": True, "procs": list(known.values())})
                    last_pids, next_procs = pids, t + 60.0
                elif pids != last_pids:
                    cur = {int(x) for x in (pids or "").split()}
                    started = procs(a.cgroup, only=cur - set(known))
                    exited = sorted(set(known) - cur)
                    for p in exited:
                        known.pop(p, None)
                    for p in started:
                        known[p["pid"]] = p
                    if started or exited:
                        w({"kind": "procs", "t_utc": now(), "full": False, "started": started, "exited": exited})
                    last_pids = pids
            if kmsg:
                for r in kmsg.poll():
                    w(r)
            time.sleep(0.25)
        if kmsg:
            for r in kmsg.poll():
                w(r)
        w({"kind": "end", "t_utc": now(), "kmsg_overruns": kmsg.overruns if kmsg else None})


if __name__ == "__main__":
    main()
