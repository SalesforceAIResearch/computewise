"""The box's single-node SLURM-compatible controller: one process that holds the queue, allocates the node's GPUs,
CPUs and memory, honours dependencies, arrays and time limits, runs each batch job's script as its owner, and
writes an event log of every job transition. The CLIs (sbatch, squeue, sacct, scancel, sinfo, srun, scontrol) talk
to it over a unix socket.

It schedules in priority (submit) order with first-fit backfill, on every submit, exit and cancel and on a 1-s
tick. A pending job's reason follows SLURM's: Dependency, Resources (the highest-priority job blocked on
resources), Priority (jobs behind it), JobHeldUser, JobArrayTaskLimit, DependencyNeverSatisfied. With
DependencyParameters=kill_invalid_depend, a job whose dependency can no longer be satisfied is cancelled.

usage: ctld.py [-D] [--gpus N] [--cpus N] [--mem-mb N] [--spool DIR] [--socket PATH] [--first-job-id N]
"""
import argparse
import hashlib
import json
import os
import pwd
import select
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

KILL_WAIT_S = 30
STEPD = os.environ.get("SLURMSTEPD") or os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "slurmstepd")
TICK_S = 1.0


class Wake:
    """The scheduling loop's wake-up, which the SIGCHLD handler sets: a self-pipe, so setting it takes no lock.
    threading.Event's set() takes a non-reentrant lock, and a handler that set it while the main thread was already
    inside set() waited on its own thread forever, holding the controller lock (2026-11-04: an array whose tasks failed
    at once, then a scancel that never returned)."""

    def __init__(self):
        self.r, self.w = os.pipe()
        os.set_blocking(self.r, False)
        os.set_blocking(self.w, False)

    def set(self):
        try:
            os.write(self.w, b"\0")
        except BlockingIOError:                 # the pipe is full: a wake-up is already pending
            pass

    def wait(self, timeout):
        select.select([self.r], [], [], timeout)

    def clear(self):
        try:
            while os.read(self.r, 4096):
                pass
        except BlockingIOError:
            pass


class Controller:
    def __init__(self, a):
        self.lock = threading.RLock()
        self.wake = Wake()
        self.jobs = {}                      # id -> job dict
        self.next_id = a.first_job_id
        self.node = a.node_name
        self.gpus = a.gpus
        self.cpus = a.cpus
        self.mem = a.mem_mb
        self.gpu_type = a.gpu_type
        self.partition = a.partition
        self.spool = a.spool
        self.def_mem_per_cpu = max(1, self.mem // max(1, self.cpus))
        self.procs = {}                     # id -> Popen
        self.kill_at = {}                   # id -> (time to SIGKILL, final state)
        self.pass_no = 0
        self.active_file = a.active_file
        self.last_active = None
        os.makedirs(self.spool, exist_ok=True)
        os.chmod(self.spool, 0o755)
        ev = a.event_log or os.path.join(self.spool, "sched-events.jsonl")
        os.makedirs(os.path.dirname(ev), exist_ok=True)
        self.events = open(ev, "a", buffering=1)
        self.event("controller_start", node=self.node, gpus=self.gpus, cpus=self.cpus, mem_mb=self.mem,
                   first_job_id=self.next_id, version=common.VERSION, pid=os.getpid())
        bad = spool_exec_error(self.spool)
        if bad:                             # every batch job would fail at its exec: say so once, at the start
            self.event("controller_error", error=f"spool {self.spool}: job scripts cannot be executed ({bad})")

    # ---------------------------------------------------------------- records
    def event(self, ev, **kw):
        rec = dict(t=round(time.time(), 3), ev=ev, **kw)
        self.events.write(json.dumps(rec, sort_keys=True) + "\n")

    def jev(self, ev, j, **kw):
        self.event(ev, job=j["id"], display=common.job_display_id(j), array_job_id=j.get("array_job_id"),
                   array_task_id=j.get("array_task_id"), name=j["name"], pass_no=self.pass_no, **kw)

    # ---------------------------------------------------------------- allocation state
    def used(self):
        g, c, m = set(), 0, 0
        for j in self.jobs.values():
            if j["state"] in ("RUNNING", "COMPLETING"):
                g.update(j["gpu_ids"])
                c += j["cpus"]
                m += j["mem_mb"]
        return g, c, m

    def node_state(self):
        g, c, _ = self.used()
        if not g and not c:
            return "idle"
        if len(g) >= self.gpus or c >= self.cpus:
            return "allocated"
        return "mixed"

    # ---------------------------------------------------------------- submit
    def submit(self, r):
        n_gpus = int(r.get("gpus") or 0)
        cpus = int(r.get("cpus") or 1)
        mem = r.get("mem_mb")
        if mem is None:
            mem = self.def_mem_per_cpu * cpus
        mem = int(mem) or self.mem
        if r.get("exclusive"):
            n_gpus, cpus, mem = max(n_gpus, self.gpus), self.cpus, self.mem
        part = r.get("partition") or self.partition
        if part != self.partition:
            return {"error": "invalid partition specified: %s" % part, "fatal": True}
        if n_gpus > self.gpus or cpus > self.cpus or mem > self.mem:
            return {"error": "Batch job submission failed: Requested node configuration is not available"}
        deps = r.get("dependency")
        if deps:
            for clause in deps["clauses"]:
                for jid in clause["ids"]:
                    if not self.lookup(jid):
                        return {"error": "Batch job submission failed: Job dependency problem"}
        indices = r.get("array") or None
        base_id = self.next_id
        created = []
        now = time.time()
        for k, idx in enumerate(indices if indices else [None]):
            jid = self.next_id
            self.next_id += 1
            j = dict(id=jid, name=r.get("name") or "sbatch", user=r.get("user") or "root", uid=int(r.get("uid") or 0),
                     gid=int(r.get("gid") or 0), submit=now, eligible=None, start=None, end=None,
                     state="PENDING", reason="None", gpus=n_gpus, gpu_ids=[], cpus=cpus, mem_mb=mem,
                     time_limit=r.get("time_limit"), partition=part, workdir=r.get("workdir") or "/",
                     stdout=r.get("stdout"), stderr=r.get("stderr"), open_mode=r.get("open_mode") or "truncate",
                     env=r.get("env") or {}, script=r.get("script") or "", argv=r.get("argv") or [],
                     dependency=deps, dependency_str=r.get("dependency_str") or "", held=bool(r.get("hold")),
                     priority=1000000 - jid, exit_code=None, signal=None, cancelled_by=None, pid=None,
                     array_job_id=base_id if indices else None, array_task_id=idx,
                     array_spec=r.get("array_spec"), array_throttle=r.get("array_throttle"), srun_pid=r.get("srun_pid"),
                     array_size=len(indices) if indices else None, comment=r.get("comment") or "",
                     account=r.get("account") or "", qos=r.get("qos") or "normal", kind=r.get("kind") or "batch",
                     nice=int(r.get("nice") or 0), command=r.get("command") or "")
            self.jobs[jid] = j
            created.append(j)
            sha = hashlib.sha256(j["script"].encode()).hexdigest() if j["script"] else None
            self.jev("submit", j, gpus_req=n_gpus, cpus=cpus, mem_mb=mem, time_limit=j["time_limit"],
                     dependency=j["dependency_str"], workdir=j["workdir"], argv=j["argv"], script_sha256=sha,
                     hold=j["held"], kind=j["kind"], user=j["user"])
        self.wake.set()
        return {"job_id": base_id, "ids": [j["id"] for j in created]}

    def lookup(self, spec):
        """A job reference: '123' (a job, or a whole array), '123_4' (one array task). Returns a list of jobs."""
        spec = str(spec).strip()
        if "_" in spec:
            a, t = spec.split("_", 1)
            try:
                a, t = int(a), int(t)
            except ValueError:
                return []
            return [j for j in self.jobs.values() if j.get("array_job_id") == a and j.get("array_task_id") == t]
        try:
            n = int(spec)
        except ValueError:
            return []
        arr = [j for j in self.jobs.values() if j.get("array_job_id") == n]
        if arr:
            return arr
        j = self.jobs.get(n)
        return [j] if j else []

    # ---------------------------------------------------------------- dependencies
    def dep_status(self, j):
        """'ok' when satisfied, 'wait' while it may still be, 'never' when it cannot be."""
        d = j.get("dependency")
        if not d:
            return "ok"
        results = []
        for clause in d["clauses"]:
            typ = clause["type"]
            if typ == "singleton":
                others = [o for o in self.jobs.values() if o["id"] != j["id"] and o["name"] == j["name"]
                          and o["user"] == j["user"] and o["id"] < j["id"] and o["state"] not in common.TERMINAL]
                results.append("wait" if others else "ok")
                continue
            st = []
            for ref in clause["ids"]:
                deps = self.lookup(ref)
                if not deps:
                    st.append("ok" if typ in ("afterany", "after") else "never")
                    continue
                for o in deps:
                    s = o["state"]
                    if typ == "after":
                        st.append("ok" if (o["start"] or s in common.TERMINAL) else "wait")
                    elif typ == "afterany":
                        st.append("ok" if s in common.TERMINAL else "wait")
                    elif typ == "afterok":
                        if s not in common.TERMINAL:
                            st.append("wait")
                        else:
                            st.append("ok" if s == "COMPLETED" and o["exit_code"] == 0 else "never")
                    elif typ == "afternotok":
                        if s not in common.TERMINAL:
                            st.append("wait")
                        else:
                            st.append("never" if s == "COMPLETED" and o["exit_code"] == 0 else "ok")
                    else:
                        st.append("ok")
            results.append("never" if "never" in st else ("wait" if "wait" in st else "ok"))
        if d.get("any"):
            if "ok" in results:
                return "ok"
            return "never" if all(x == "never" for x in results) else "wait"
        if "never" in results:
            return "never"
        return "wait" if "wait" in results else "ok"

    # ---------------------------------------------------------------- scheduling
    def set_reason(self, j, reason):
        if j["reason"] != reason:
            j["reason"] = reason
            self.jev("pending_reason", j, reason=reason)

    def schedule(self):
        with self.lock:
            self.pass_no += 1
            now = time.time()
            self.reap(now)
            free_g = [g for g in range(self.gpus) if g not in self.used()[0]]
            _, used_c, used_m = self.used()
            free_c, free_m = self.cpus - used_c, self.mem - used_m
            blocked = False
            for j in sorted((x for x in self.jobs.values() if x["state"] == "PENDING"),
                            key=lambda x: (-x["priority"] + x["nice"], x["id"])):
                ds = self.dep_status(j)
                if ds == "never":
                    self.finish(j, "CANCELLED", exit_code=0, sig=0, reason="DependencyNeverSatisfied")
                    continue
                if ds == "wait":
                    self.set_reason(j, "Dependency")
                    continue
                if j["eligible"] is None:
                    j["eligible"] = now
                    self.jev("eligible", j)
                if j["held"]:
                    self.set_reason(j, "JobHeldUser")
                    continue
                if j.get("array_throttle"):
                    running = sum(1 for o in self.jobs.values() if o.get("array_job_id") == j["array_job_id"]
                                  and o["state"] in ("RUNNING", "COMPLETING"))
                    if running >= j["array_throttle"]:
                        self.set_reason(j, "JobArrayTaskLimit")
                        continue
                if j["gpus"] <= len(free_g) and j["cpus"] <= free_c and j["mem_mb"] <= free_m:
                    ids, free_g = free_g[:j["gpus"]], free_g[j["gpus"]:]
                    free_c -= j["cpus"]
                    free_m -= j["mem_mb"]
                    self.start(j, ids, now)
                else:
                    self.set_reason(j, "Priority" if blocked else "Resources")
                    blocked = True
            self.write_active()

    def write_active(self):
        """The number of jobs not yet ended, for the box's end-of-run rule: the run is not over while the queue holds
        work, even when no job process exists yet (a pending job)."""
        if not self.active_file:
            return
        n = sum(1 for j in self.jobs.values() if j["state"] not in common.TERMINAL)
        if n == self.last_active:
            return
        tmp = self.active_file + ".tmp"
        with open(tmp, "w") as f:
            f.write("%d\n" % n)
        os.replace(tmp, self.active_file)
        self.last_active = n

    def start(self, j, gpu_ids, now):
        j["gpu_ids"] = gpu_ids
        j["start"] = now
        j["state"] = "RUNNING"
        j["reason"] = "None"
        if j["kind"] == "interactive":
            j["pid"] = j.get("srun_pid")
            self.jev("start", j, gpus=gpu_ids, cpus=j["cpus"], mem_mb=j["mem_mb"], pid=j["pid"])
            return
        try:
            p = self.launch(j)
        except Exception as e:                 # the job could not be started (an output path that cannot be opened)
            self.jev("start", j, gpus=gpu_ids, cpus=j["cpus"], mem_mb=j["mem_mb"], launch_error=str(e))
            self.finish(j, "FAILED", exit_code=1, sig=0, reason="launch failed")
            return
        self.procs[j["id"]] = p
        j["pid"] = p.pid
        self.jev("start", j, gpus=gpu_ids, cpus=j["cpus"], mem_mb=j["mem_mb"], pid=p.pid)

    def job_env(self, j):
        # the submitter's own job variables (an sbatch from inside a job) go; its site settings stay
        env = {k: v for k, v in (j["env"] or {}).items() if not k.startswith("SLURM_") or k in ("SLURM_LIBDIR", "SLURM_CONF")}
        env.pop("CUDA_VISIBLE_DEVICES", None)
        gl = ",".join(str(g) for g in j["gpu_ids"])
        env.update({
            "SLURM_JOB_ID": str(j["id"]), "SLURM_JOBID": str(j["id"]), "SLURM_JOB_NAME": j["name"],
            "SLURM_JOB_USER": j["user"], "SLURM_JOB_UID": str(j["uid"]), "SLURM_JOB_PARTITION": j["partition"],
            "SLURM_JOB_NODELIST": self.node, "SLURM_NODELIST": self.node, "SLURM_JOB_NUM_NODES": "1",
            "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_NPROCS": "1", "SLURM_CPUS_ON_NODE": str(j["cpus"]),
            "SLURM_CPUS_PER_TASK": str(j["cpus"]), "SLURM_JOB_CPUS_PER_NODE": str(j["cpus"]),
            "SLURM_MEM_PER_NODE": str(j["mem_mb"]), "SLURM_SUBMIT_DIR": j["workdir"], "SLURM_SUBMIT_HOST": self.node,
            "SLURMD_NODENAME": self.node, "SLURM_CLUSTER_NAME": "cluster", "SLURM_TASK_PID": "",
            "SLURM_CONF": "/etc/slurm/slurm.conf", "SLURM_JOB_QOS": j["qos"], "SLURM_GTIDS": "0", "SLURM_PROCID": "0",
            "SLURM_LOCALID": "0", "SLURM_NODEID": "0", "SLURM_TOPOLOGY_ADDR": self.node,
            "ENVIRONMENT": "BATCH", "PWD": j["workdir"],
        })
        env["CUDA_VISIBLE_DEVICES"] = gl
        if j["gpu_ids"]:
            env.update({"SLURM_JOB_GPUS": gl, "SLURM_GPUS_ON_NODE": str(len(j["gpu_ids"])), "GPU_DEVICE_ORDINAL": gl,
                        "SLURM_TRES_PER_NODE": "gres/gpu:%d" % len(j["gpu_ids"])})
        if j.get("array_job_id") is not None:
            sib = sorted(o["array_task_id"] for o in self.jobs.values() if o.get("array_job_id") == j["array_job_id"])
            env.update({"SLURM_ARRAY_JOB_ID": str(j["array_job_id"]), "SLURM_ARRAY_TASK_ID": str(j["array_task_id"]),
                        "SLURM_ARRAY_TASK_COUNT": str(len(sib)), "SLURM_ARRAY_TASK_MIN": str(min(sib)),
                        "SLURM_ARRAY_TASK_MAX": str(max(sib)), "SLURM_ARRAY_TASK_STEP": "1"})
        if j["time_limit"]:
            env["SLURM_JOB_END_TIME"] = str(int(j["start"] + j["time_limit"]))
        env["SLURM_JOB_START_TIME"] = str(int(j["start"]))
        return env

    def out_path(self, j, pattern, default):
        p = pattern or default
        a = j.get("array_job_id")
        repl = {"%j": str(j["id"]), "%J": str(j["id"]), "%A": str(a if a is not None else j["id"]),
                "%a": str(j["array_task_id"] if a is not None else 4294967294), "%x": j["name"], "%u": j["user"],
                "%N": self.node, "%n": "0", "%t": "0", "%s": "batch", "%%": "\0"}
        for k, v in repl.items():
            p = p.replace(k, v)
        p = p.replace("\0", "%")
        return p if os.path.isabs(p) else os.path.join(j["workdir"], p)

    def launch(self, j):
        """Write the job's script and step file into the spool (root-owned, world-readable), then start slurmstepd as
        the job's owner. The step opens the job's output files itself, as the owner (the box drops CAP_CHOWN and
        CAP_DAC_OVERRIDE, so root could neither create them in the owner's directory nor hand them over)."""
        jd = os.path.join(self.spool, "job%05d" % j["id"])
        os.makedirs(jd, exist_ok=True)
        os.chmod(jd, 0o755)
        script = os.path.join(jd, "slurm_script")
        with open(script, "w") as f:
            f.write(j["script"])
        os.chmod(script, 0o755)
        default = "slurm-%A_%a.out" if j.get("array_job_id") is not None else "slurm-%j.out"
        out = self.out_path(j, j["stdout"], default)
        err = self.out_path(j, j["stderr"], None) if j["stderr"] else out
        step = dict(script=script, argv=list(j["argv"] or []), workdir=j["workdir"], stdout=out, stderr=err,
                    open_mode=j["open_mode"], env=self.job_env(j), job_id=j["id"])
        sp = os.path.join(jd, "step.json")
        with open(sp, "w") as f:
            json.dump(step, f)
        os.chmod(sp, 0o644)
        kw = {}
        if os.geteuid() == 0 and j["uid"] != 0:
            kw = dict(user=j["uid"], group=j["gid"], extra_groups=[])
        return subprocess.Popen([STEPD, sp], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True, close_fds=True, **kw)

    # ---------------------------------------------------------------- ends
    def reap(self, now):
        for j in self.jobs.values():
            if j["kind"] != "interactive" or j["state"] != "RUNNING" or not j.get("pid"):
                continue
            if not pid_alive(j["pid"]):               # srun was killed before it could report: the allocation goes
                self.finish(j, self.kill_at.pop(j["id"], (None, "CANCELLED"))[1], exit_code=0, sig=9,
                            reason="srun exited")
            elif j["time_limit"] and now - j["start"] > j["time_limit"] and j["id"] not in self.kill_at:
                self.signal_job(j, signal.SIGTERM, group=False)
                self.kill_at[j["id"]] = (now + KILL_WAIT_S, "TIMEOUT")
                self.jev("timeout", j)
        for jid, p in list(self.procs.items()):
            j = self.jobs[jid]
            rc = p.poll()
            if rc is None:
                if j["time_limit"] and now - j["start"] > j["time_limit"] and jid not in self.kill_at:
                    self.signal_job(j, signal.SIGTERM)
                    self.kill_at[jid] = (now + KILL_WAIT_S, "TIMEOUT")
                    self.jev("timeout", j)
                ka = self.kill_at.get(jid)
                if ka and now >= ka[0]:
                    self.signal_job(j, signal.SIGKILL)
                continue
            del self.procs[jid]
            self.signal_job(j, signal.SIGKILL)          # the batch script ended: its leftover processes go with it
            final = self.kill_at.pop(jid, (None, None))[1]
            sig = -rc if rc < 0 else 0
            code = rc if rc >= 0 else 0
            if final:
                state = final
            else:
                state = "COMPLETED" if rc == 0 else "FAILED"
            self.finish(j, state, exit_code=code, sig=sig)

    def signal_job(self, j, sig, group=True):
        """Signal the job's process group. A root controller without CAP_KILL (the box drops every capability but
        MKNOD, SETUID and SETGID) may not signal another user's processes, so it signals as the job's owner, from a
        child that switches to that uid first, as slurmstepd does on a compute node."""
        if not j.get("pid"):
            return
        try:
            if os.geteuid() == 0 and j["uid"] != 0:
                kill_as(j["uid"], j["gid"], j["pid"], sig, group)
            else:
                (os.killpg if group else os.kill)(j["pid"], sig)
        except ProcessLookupError:
            pass
        except OSError as e:
            self.event("controller_error", job=j["id"], error=f"signal {int(sig)} to job {j['id']}: {e}")

    def finish(self, j, state, exit_code=0, sig=0, reason=None, by=None):
        j["state"] = state
        j["end"] = time.time()
        j["exit_code"] = exit_code
        j["signal"] = sig
        if reason:
            j["reason"] = reason
        if by is not None:
            j["cancelled_by"] = by
        self.jev("end", j, state=state, exit_code=exit_code, signal=sig, reason=reason, gpus=j["gpu_ids"],
                 cancelled_by=by)
        self.wake.set()

    # ---------------------------------------------------------------- requests
    def cancel(self, r):
        targets = []
        for ref in r.get("ids") or []:
            js = self.lookup(ref)
            if not js:
                return {"error": "Kill job error on job id %s: Invalid job id specified" % ref}
            targets.extend(js)
        if not r.get("ids"):
            targets = list(self.jobs.values())
        out = []
        caller = r.get("uid")
        if r.get("ids") and caller not in (None, 0, os.geteuid()):
            for j in targets:
                if j["uid"] != caller:
                    return {"error": "Kill job error on job id %s: Access/permission denied" % common.job_display_id(j)}
        for j in targets:
            if j["state"] in common.TERMINAL:
                continue
            if caller not in (None, 0, os.geteuid()) and j["uid"] != caller:
                continue
            if r.get("target_user") and j["user"] != r["target_user"]:
                continue
            if r.get("name") and j["name"] != r["name"]:
                continue
            if r.get("states") and common.STATE_CODE.get(j["state"]) not in r["states"] and j["state"] not in r["states"]:
                continue
            sig = r.get("signal")
            if sig and j["state"] == "RUNNING":
                self.signal_job(j, int(sig))
                self.jev("signal", j, signal=int(sig))
                continue
            self.jev("cancel_request", j, by=r.get("uid"))
            if j["state"] == "PENDING":
                self.finish(j, "CANCELLED", exit_code=0, sig=0, by=r.get("uid"))
            elif j["kind"] == "interactive":            # srun forwards the signal to its command and reports back
                self.signal_job(j, signal.SIGTERM, group=False)
                self.finish(j, "CANCELLED", exit_code=0, sig=15, by=r.get("uid"))
            else:
                j["cancelled_by"] = r.get("uid")
                self.signal_job(j, signal.SIGTERM)
                self.kill_at[j["id"]] = (time.time() + KILL_WAIT_S, "CANCELLED")
            out.append(j["id"])
        self.wake.set()
        return {"cancelled": out}

    def hold(self, r, held):
        out = []
        for ref in r.get("ids") or []:
            js = self.lookup(ref)
            if not js:
                return {"error": "Invalid job id specified"}
            for j in js:
                if r.get("uid") not in (None, 0, os.geteuid()) and j["uid"] != r["uid"]:
                    return {"error": "Access/permission denied"}
                if j["state"] == "PENDING":
                    j["held"] = held
                    if not held and j["reason"] == "JobHeldUser":
                        j["reason"] = "None"
                    self.jev("hold" if held else "release", j)
                    out.append(j["id"])
        self.wake.set()
        return {"ok": out}

    def interactive_done(self, r):
        j = self.jobs.get(int(r["job_id"]))
        if j and j["state"] == "RUNNING":
            rc = int(r.get("rc", 0))
            sig = int(r.get("signal", 0))
            self.finish(j, "COMPLETED" if rc == 0 and not sig else "FAILED", exit_code=rc, sig=sig)
        return {"ok": True}

    def public(self, j):
        d = {k: v for k, v in j.items() if k not in ("env", "script")}
        d["dependency_state"] = self.dep_status(j) if j["state"] == "PENDING" else None
        return d

    def handle(self, r):
        op = r.get("op")
        with self.lock:
            if op == "ping":
                return {"ok": True, "version": common.VERSION}
            if op == "submit":
                rep = self.submit(r)
                self.schedule()
                return rep
            if op == "jobs":
                return {"jobs": [self.public(j) for j in sorted(self.jobs.values(), key=lambda x: x["id"])],
                        "now": time.time()}
            if op == "cancel":
                return self.cancel(r)
            if op == "hold":
                return self.hold(r, True)
            if op == "release":
                return self.hold(r, False)
            if op == "interactive_done":
                return self.interactive_done(r)
            if op == "shutdown":
                if int(r.get("uid", -1)) not in (0, os.geteuid()):
                    return {"error": "Access/permission denied"}
                self.event("controller_stop", requested_by=r.get("uid"))
                threading.Timer(0.2, lambda: os._exit(0)).start()
                return {"ok": True}
            if op == "node":
                g, c, m = self.used()
                return {"node": self.node, "gpus": self.gpus, "gpu_type": self.gpu_type, "cpus": self.cpus,
                        "mem_mb": self.mem, "alloc_gpus": sorted(g), "alloc_cpus": c, "alloc_mem_mb": m,
                        "state": self.node_state(), "partition": self.partition, "version": common.VERSION}
        return {"error": "unknown request"}

    def loop(self):
        while True:
            self.wake.wait(TICK_S)
            self.wake.clear()
            try:
                self.schedule()
            except Exception as e:            # keep scheduling; record the fault
                self.event("controller_error", error=repr(e))


def serve(ctl, path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)

    class H(socketserver.StreamRequestHandler):
        def handle(self):
            line = self.rfile.readline()
            try:
                r = json.loads(line)
                peer = peer_ids(self.request)
                if peer is not None:          # who the caller is comes from the kernel, never from the request
                    r["uid"], r["gid"] = peer
                    try:
                        r["user"] = pwd.getpwuid(peer[0]).pw_name
                    except KeyError:
                        r["user"] = str(peer[0])
                rep = ctl.handle(r)
            except Exception as e:
                rep = {"error": "slurm controller error: %r" % (e,)}
            self.wfile.write((json.dumps(rep) + "\n").encode())

    class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    srv = S(path, H)
    os.chmod(path, 0o666)
    return srv


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                         # exists, owned by someone this process may not signal
        return True
    return True


def spool_exec_error(spool):
    """None if a script written to the spool can be executed; else why not (a noexec mount, most often)."""
    t = os.path.join(spool, ".exec-check")
    try:
        with open(t, "w") as f:
            f.write("#!/bin/sh\nexit 0\n")
        os.chmod(t, 0o755)
        rc = subprocess.run([t], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=10).returncode
        return None if rc == 0 else f"exit {rc}"
    except OSError as e:
        return e.strerror or str(e)
    except subprocess.TimeoutExpired:
        return "timed out"
    finally:
        try:
            os.remove(t)
        except OSError:
            pass


def kill_as(uid, gid, pgid, sig, group=True):
    """killpg(pgid, sig) (or kill, with group=False) from a forked child running as uid/gid; raises the child's error
    (ProcessLookupError when the target is gone)."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        try:
            os.close(r)
            os.setgroups([])
            os.setgid(gid)
            os.setuid(uid)
            (os.killpg if group else os.kill)(pgid, sig)
            os.write(w, b"0")
        except OSError as e:
            os.write(w, str(e.errno).encode())
        finally:
            os._exit(0)
    os.close(w)
    res = os.read(r, 16)
    os.close(r)
    os.waitpid(pid, 0)
    if res and res != b"0":
        err = int(res)
        raise (ProcessLookupError if err == 3 else OSError)(err, os.strerror(err))


def peer_ids(sock):
    """The connecting process's (uid, gid) from SO_PEERCRED (Linux); None where the platform has no such option, which
    is only the controller's self-test, where caller and controller are the same user."""
    opt = getattr(socket, "SO_PEERCRED", None)
    if opt is None:
        return None
    _pid, uid, gid = struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, opt, struct.calcsize("3i")))
    return uid, gid


def gpu_devices():
    """The GPUs this node has, as SLURM's gres.conf File=/dev/nvidia[0-N] names them: the count of /dev/nvidia<N>."""
    try:
        return sum(1 for n in os.listdir("/dev") if n.startswith("nvidia") and n[6:].isdigit())
    except OSError:
        return 0


def conf_gpu_type(path=None):
    """The GPU type slurm.conf's node line names (Gres=gpu:<type>:<n>), as SLURM takes it from the config; None if the
    file or the field is missing (a staged or test install with no /etc/slurm)."""
    try:
        for line in open(path or os.environ.get("SLURM_CONF") or "/etc/slurm/slurm.conf"):
            if line.lstrip().startswith("NodeName"):
                for f in line.split():
                    if f.startswith("Gres=gpu:") and f.count(":") >= 2:
                        return f.split(":")[1] or None
    except OSError:
        pass
    return None


def cpus_available():
    """The box's CPU limit: cgroup v2 cpu.max quota/period, else the affinity mask, else os.cpu_count()."""
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) // int(per)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def set_comm(name):
    """Name the process in /proc/<pid>/comm (what top and ps -e show), as the real daemon is named."""
    try:
        import ctypes
        ctypes.CDLL(None).prctl(15, name.encode()[:15], 0, 0, 0)
    except Exception:
        pass


def mem_total_mb():
    try:
        lim = open("/sys/fs/cgroup/memory.max").read().strip()
        if lim != "max":
            return int(lim) // (1 << 20)
    except OSError:
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 64 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-D", action="store_true", help="run in the foreground")
    ap.add_argument("--gpus", type=int, default=None,
                    help="default: $SLURM_NODE_GPUS, else the node's /dev/nvidia<N> devices (gres.conf File=), else 4")
    ap.add_argument("--gpu-type", default=None,
                    help="default: $SLURM_NODE_GPU_TYPE, else slurm.conf's Gres=gpu:<type>:, else h100")
    ap.add_argument("--cpus", type=int, default=None)
    ap.add_argument("--mem-mb", type=int, default=None)
    ap.add_argument("--node-name", default=socket.gethostname().split(".")[0])
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--spool", default=os.environ.get("SLURM_SPOOL", "/var/spool/slurmd"))
    ap.add_argument("--socket", default=common.SOCKET)
    ap.add_argument("--first-job-id", type=int, default=None)
    ap.add_argument("--active-file", default=os.environ.get("SLURM_ACTIVE_FILE"),
                    help="write the number of jobs not yet ended here on every change (the box's end-of-run rule)")
    ap.add_argument("--event-log", default=os.environ.get("SLURM_EVENT_LOG"),
                    help="where the job event log goes (default: <spool>/sched-events.jsonl)")
    a = ap.parse_args()
    if a.gpus is None:
        a.gpus = int(os.environ.get("SLURM_NODE_GPUS") or 0) or gpu_devices() or 4
    if a.gpu_type is None:
        a.gpu_type = os.environ.get("SLURM_NODE_GPU_TYPE") or conf_gpu_type() or "h100"
    if a.cpus is None:
        a.cpus = cpus_available()
    set_comm("slurmctld")
    if a.mem_mb is None:
        a.mem_mb = max(1024, mem_total_mb() - 2048)
    if a.first_job_id is None:
        a.first_job_id = 100000 + int(time.time()) % 800000
    ready = None
    if not a.D:                               # daemonize as slurmctld does without -D: detach from the caller entirely
        rfd, ready = os.pipe()
        if os.fork():                         # the caller returns once the socket accepts requests, so a submit
            os.close(ready)                   # straight after `slurmctld` cannot miss the controller
            ok = os.read(rfd, 1)
            os._exit(0 if ok == b"1" else 1)
        os.close(rfd)
        os.setsid()
        if os.fork():
            os._exit(0)
        fd = os.open(os.devnull, os.O_RDWR)
        for k in (0, 1, 2):
            os.dup2(fd, k)
    ctl = Controller(a)
    srv = serve(ctl, a.socket)
    if ready is not None:
        os.write(ready, b"1")
        os.close(ready)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    def stop(*_):
        try:                                  # a stop that lands inside another event write raises here: exit anyway
            ctl.event("controller_stop")
        except Exception:
            pass
        os._exit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGCHLD, lambda *_: ctl.wake.set())      # a job's exit is reaped at once, not on the tick
    ctl.loop()


if __name__ == "__main__":
    main()
