"""The scheduler's commands: sbatch, squeue, sacct, scancel, sinfo, srun, scontrol. One module, dispatched on the
name it is run as (bin/<command> are one-line shims). Options and output formats follow SLURM 23.11."""
import json
import os
import pwd
import shlex
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402
from common import ControllerError, request  # noqa: E402


def die(prog, msg, code=1):
    sys.stderr.write(f"{prog}: error: {msg}\n")
    sys.exit(code)


def me():
    uid = os.getuid()
    try:
        name = pwd.getpwuid(uid).pw_name
    except KeyError:
        name = str(uid)
    return uid, os.getgid(), name


# ===================================================================== option parsing
# (long, short, takes_value); values may come as --opt=v, --opt v, -o v or -ov
SBATCH_OPTS = [
    ("job-name", "J", True), ("output", "o", True), ("error", "e", True), ("time", "t", True),
    ("cpus-per-task", "c", True), ("mem", None, True), ("mem-per-cpu", None, True), ("mem-per-gpu", None, True),
    ("gres", None, True), ("gpus", "G", True), ("gpus-per-node", None, True), ("gpus-per-task", None, True),
    ("ntasks", "n", True), ("ntasks-per-node", None, True), ("nodes", "N", True), ("partition", "p", True),
    ("dependency", "d", True), ("array", "a", True), ("wrap", None, True), ("parsable", None, False),
    ("wait", "W", False), ("hold", "H", False), ("chdir", "D", True), ("export", None, True),
    ("open-mode", None, True), ("exclusive", None, False), ("comment", None, True), ("account", "A", True),
    ("qos", "q", True), ("nice", None, True), ("mail-type", None, True), ("mail-user", None, True),
    ("requeue", None, False), ("no-requeue", None, False), ("signal", None, True), ("quiet", "Q", False),
    ("verbose", "v", False), ("constraint", "C", True), ("nodelist", "w", True), ("exclude", "x", True),
    ("reservation", None, True), ("begin", "b", True), ("deadline", None, True), ("input", "i", True),
    ("gpu-bind", None, True), ("cpu-bind", None, True), ("distribution", "m", True), ("threads-per-core", None, True),
    ("kill-on-invalid-dep", None, True), ("propagate", None, True), ("uid", None, True), ("gid", None, True),
    ("test-only", None, False), ("help", "h", False), ("usage", "u", False), ("version", "V", False),
    ("get-user-env", None, False), ("tmp", None, True), ("licenses", "L", True), ("cluster", "M", True),
    ("container", None, True), ("tres-per-task", None, True), ("ignore-pbs", None, False), ("mincpus", None, True),
    # accepted and recorded, with no effect on a single node: what a cluster's job scripts carry by habit
    ("time-min", None, True), ("hint", None, True), ("mem-bind", None, True), ("gres-flags", None, True),
    ("cpus-per-gpu", None, True), ("ntasks-per-gpu", None, True), ("oversubscribe", "s", False),
    ("overcommit", "O", False), ("no-kill", "k", False), ("priority", None, True), ("wckey", None, True),
    ("export-file", None, True), ("nodefile", "F", True), ("sockets-per-node", None, True),
    ("cores-per-socket", None, True), ("ntasks-per-core", None, True), ("profile", None, True),
    ("acctg-freq", None, True), ("prefer", None, True), ("spread-job", None, False), ("use-min-nodes", None, False),
    ("gpu-freq", None, True), ("switches", None, True), ("delay-boot", None, True), ("reboot", None, False),
    ("core-spec", "S", True), ("thread-spec", None, True), ("power", None, True), ("mcs-label", None, True),
    ("network", None, True), ("tres-bind", None, True), ("gpus-per-socket", None, True),
    ("extra-node-info", "B", True), ("clusters", None, True), ("x11", None, False), ("burst-buffer", None, True),
    ("bbf", None, True), ("cluster-constraint", None, True), ("requeue-exit", None, True),
]
SRUN_EXTRA = [("pty", None, False), ("label", "l", False), ("unbuffered", "u", False), ("overlap", None, False),
              ("exact", None, False), ("mpi", None, True), ("immediate", "I", False), ("kill-on-bad-exit", "K", False)]


def parse_opts(prog, argv, table, stop_at_positional=True):
    longs = {l: (l, v) for l, s, v in table}
    shorts = {s: (l, v) for l, s, v in table if s}
    opts, pos, i = {}, [], 0
    while i < len(argv):
        a = argv[i]
        if a == "--":
            pos.extend(argv[i + 1:])
            break
        if a.startswith("--") and len(a) > 2:
            name, eq, val = a[2:].partition("=")
            if name not in longs:
                die(prog, f"unrecognized option '{a}'\nTry \"{prog} --help\" for more information")
            key, takes = longs[name]
            if takes:
                if not eq:
                    i += 1
                    if i >= len(argv):
                        die(prog, f"option '--{name}' requires an argument")
                    val = argv[i]
                opts[key] = val
            else:
                opts[key] = True
        elif a.startswith("-") and len(a) > 1:
            c = a[1]
            if c not in shorts:
                die(prog, f"invalid option -- '{c}'\nTry \"{prog} --help\" for more information")
            key, takes = shorts[c]
            if takes:
                val = a[2:]
                if not val:
                    i += 1
                    if i >= len(argv):
                        die(prog, f"option requires an argument -- '{c}'")
                    val = argv[i]
                opts[key] = val
            else:
                opts[key] = True
                for c2 in a[2:]:                 # bundled flags like -Wv
                    if c2 in shorts and not shorts[c2][1]:
                        opts[shorts[c2][0]] = True
        else:
            if stop_at_positional:
                pos.extend(argv[i:])
                break
            pos.append(a)
        i += 1
    return opts, pos


def script_directives(text):
    """#SBATCH lines, from after the shebang up to the first command line, as an argv list."""
    out = []
    for line in text.splitlines()[1:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#SBATCH"):
            rest = s[len("#SBATCH"):].split("#", 1)[0] if " #" in s else s[len("#SBATCH"):]
            try:
                out.extend(shlex.split(rest))
            except ValueError:
                pass
            continue
        if s.startswith("#"):
            continue
        break
    return out


def parse_gpus(opts, prog):
    """GPU count from --gres=gpu[:type]:N[,..], --gpus/-G [type:]N, --gpus-per-node, --gpus-per-task (x ntasks)."""
    n = 0
    if opts.get("gres"):
        for part in str(opts["gres"]).split(","):
            bits = part.strip().split(":")
            if bits[0] != "gpu":
                if bits[0] in ("none", ""):
                    continue
                die(prog, "Invalid generic resource (gres) specification")
            try:
                n += int(bits[-1]) if len(bits) > 1 and bits[-1].isdigit() else 1
            except ValueError:
                die(prog, "Invalid generic resource (gres) specification")
    for key in ("gpus", "gpus-per-node"):
        if opts.get(key):
            v = str(opts[key]).split(":")[-1]
            if not v.isdigit():
                die(prog, f"Invalid --{key} specification")
            n = max(n, int(v))
    if opts.get("gpus-per-task"):
        v = str(opts["gpus-per-task"]).split(":")[-1]
        if not v.isdigit():
            die(prog, "Invalid --gpus-per-task specification")
        n = max(n, int(v) * int(opts.get("ntasks") or 1))
    if opts.get("tres-per-task"):
        for part in str(opts["tres-per-task"]).split(","):
            if part.startswith("gres/gpu") or part.startswith("gres:gpu"):
                v = part.split(":")[-1].split("=")[-1]
                if v.isdigit():
                    n = max(n, int(v) * int(opts.get("ntasks") or 1))
    return n


def parse_dependency(spec, prog):
    """'afterok:1:2,afterany:3' (AND), 'afterok:1?afterok:2' (OR), 'singleton'."""
    spec = str(spec).strip()
    any_ = "?" in spec
    clauses = []
    for part in spec.replace("?", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if part == "singleton":
            clauses.append({"type": "singleton", "ids": []})
            continue
        if ":" not in part:
            die(prog, "Batch job submission failed: Job dependency problem")
        typ, rest = part.split(":", 1)
        if typ not in ("after", "afterany", "afterok", "afternotok", "aftercorr", "afterburstbuffer"):
            die(prog, "Batch job submission failed: Job dependency problem")
        ids = [x.split("+")[0] for x in rest.split(":") if x]
        if not ids:
            die(prog, "Batch job submission failed: Job dependency problem")
        clauses.append({"type": {"aftercorr": "afterok", "afterburstbuffer": "afterany"}.get(typ, typ), "ids": ids})
    return {"clauses": clauses, "any": any_}


def build_request(prog, opts, script, name_default, argv=(), kind="batch", command=""):
    uid, gid, user = me()
    if opts.get("nodes") and str(opts["nodes"]).split("-")[0] not in ("1",):
        die(prog, "Batch job submission failed: Node count specification invalid")
    cpus = int(opts.get("cpus-per-task") or 1) * int(opts.get("ntasks") or opts.get("ntasks-per-node") or 1)
    if opts.get("cpus-per-gpu"):
        cpus = max(cpus, int(opts["cpus-per-gpu"]) * max(1, parse_gpus(opts, prog)))
    if opts.get("mincpus"):
        cpus = max(cpus, int(opts["mincpus"]))
    mem = None
    try:
        if opts.get("mem") is not None:
            mem = common.parse_mem(opts["mem"])
        elif opts.get("mem-per-cpu"):
            mem = common.parse_mem(opts["mem-per-cpu"]) * cpus
        elif opts.get("mem-per-gpu"):
            mem = common.parse_mem(opts["mem-per-gpu"]) * max(1, parse_gpus(opts, prog))
    except ValueError:
        die(prog, "Invalid --mem specification")
    tl = None
    if opts.get("time"):
        try:
            tl = common.parse_time_limit(opts["time"])
        except (ValueError, TypeError):
            die(prog, "Invalid time limit specification")
    arr, thr = None, None
    if opts.get("array"):
        try:
            arr, thr = common.parse_array(opts["array"])
        except ValueError:
            die(prog, "Invalid job array specification")
    dep = parse_dependency(opts["dependency"], prog) if opts.get("dependency") else None
    workdir = os.path.abspath(opts.get("chdir") or os.getcwd())
    export = str(opts.get("export") or "ALL")
    if export.upper().startswith("NONE"):
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": os.environ.get("HOME", "/"), "USER": user}
        rest = export.split(",")[1:]
    else:
        env = dict(os.environ)
        rest = export.split(",")[1:] if export.upper().startswith("ALL") else export.split(",")
    for kv in rest:
        if "=" in kv:
            k, v = kv.split("=", 1)
            env[k] = v
        elif kv and kv in os.environ:
            env[kv] = os.environ[kv]
    return dict(name=opts.get("job-name") or name_default, user=user, uid=uid, gid=gid,
                gpus=parse_gpus(opts, prog), cpus=cpus, mem_mb=mem, time_limit=tl, partition=opts.get("partition"),
                workdir=workdir, stdout=opts.get("output"), stderr=opts.get("error"),
                open_mode=opts.get("open-mode"), env=env, script=script, argv=list(argv), dependency=dep,
                dependency_str=opts.get("dependency") or "", hold=bool(opts.get("hold")), array=arr,
                array_spec=opts.get("array"), array_throttle=thr, comment=opts.get("comment"),
                account=opts.get("account"), qos=opts.get("qos"), exclusive=bool(opts.get("exclusive")),
                nice=opts.get("nice"), kind=kind, command=command or " ".join([prog] + list(sys.argv[1:])))


def wait_jobs(ids, poll=1.0):
    """Block until every job is terminal; returns the jobs."""
    ids = set(ids)
    while True:
        try:
            jobs = [j for j in request("jobs")["jobs"] if j["id"] in ids]
        except ControllerError:
            time.sleep(poll)
            continue
        if jobs and all(j["state"] in common.TERMINAL for j in jobs):
            return jobs
        time.sleep(poll)


def wait_exit_code(jobs):
    rc = 0
    for j in jobs:
        if j["state"] == "COMPLETED":
            rc = max(rc, j["exit_code"] or 0)
        elif j["exit_code"]:
            rc = max(rc, j["exit_code"])
        else:
            rc = max(rc, 1)
    return rc


# ===================================================================== sbatch
def sbatch(argv):
    prog = "sbatch"
    opts, pos = parse_opts(prog, argv, SBATCH_OPTS)
    if opts.get("help") or opts.get("usage"):
        print("Usage: sbatch [OPTIONS(0)...] [ : [OPTIONS(N)...]] script(0) [args(0)...]")
        return 0
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    if opts.get("wrap") is not None:
        if pos:
            die(prog, "Script arguments not permitted with --wrap option")
        script = "#!/bin/sh\n# This script was created by sbatch --wrap.\n\n%s\n" % opts["wrap"]
        name, args = "wrap", []
    else:
        if pos:
            path = pos[0]
            try:
                script = open(path).read()
            except OSError:
                die(prog, f"Unable to open file {path}")
            name, args = os.path.basename(path), pos[1:]
        else:
            if sys.stdin.isatty():
                die(prog, "No batch script specified")
            script = sys.stdin.read()
            name, args = "sbatch", []
        if not script.startswith("#!"):
            die(prog, "This does not look like a batch script.  The first\nsbatch: error: line must start with #! "
                      "followed by the path to an interpreter.\nsbatch: error: For instance: #!/bin/sh")
        directives = script_directives(script)
        if directives:
            dopts, _ = parse_opts(prog, directives, SBATCH_OPTS, stop_at_positional=False)
            dopts.update(opts)
            opts = dopts
    if opts.get("test-only"):
        print(f"sbatch: Job would start at {common.fmt_ts(time.time())} using 1 processors on nodes "
              f"{request('node')['node']} in partition {request('node')['partition']}", file=sys.stderr)
        return 0
    req = build_request(prog, opts, script, name, args)
    try:
        rep = request("submit", **req)
    except ControllerError as e:
        die(prog, f"Batch job submission failed: {e}")
    if rep.get("error"):
        die(prog, rep["error"].replace("Batch job submission failed: ", "Batch job submission failed: ", 1))
    jid = rep["job_id"]
    if opts.get("parsable"):
        print(jid)
    elif not opts.get("quiet"):
        print(f"Submitted batch job {jid}")
    sys.stdout.flush()
    if opts.get("wait"):
        return wait_exit_code(wait_jobs(rep["ids"]))
    return 0


# ===================================================================== squeue
SQ_DEFAULT = "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
SQ_LONG = "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
SQ_HEAD = {"i": "JOBID", "A": "JOBID", "F": "ARRAY_JOB_ID", "K": "ARRAY_TASK_ID", "P": "PARTITION", "j": "NAME",
           "u": "USER", "U": "UID", "t": "ST", "T": "STATE", "M": "TIME", "l": "TIME_LIMIT", "L": "TIME_LEFT",
           "D": "NODES", "R": "NODELIST(REASON)", "r": "REASON", "N": "NODELIST", "C": "CPUS", "b": "TRES_PER_NODE",
           "S": "START_TIME", "V": "SUBMIT_TIME", "e": "END_TIME", "Q": "PRIORITY", "p": "PRIORITY", "a": "ACCOUNT",
           "o": "COMMAND", "Z": "WORK_DIR", "E": "DEPENDENCY", "m": "MIN_MEMORY", "q": "QOS", "y": "NICE",
           "k": "COMMENT", "g": "GROUP", "x": "EXC_NODES", "f": "FEATURES", "B": "EXEC_HOST", "c": "MIN_CPUS",
           "z": "S:C:T", "h": "OVER_SUBSCRIBE", "n": "REQ_NODES", "w": "WCKEY", "O": "CONTIGUOUS", "v": "RESERVATION",
           "H": "SOCKETS", "I": "CORES", "J": "THREADS", "s": "NODELIST", "G": "GID", "X": "CORE_SPEC",
           "Y": "SCHEDNODES", "d": "MIN_TMP_DISK"}
SQ_LONGNAMES = {"jobid": "i", "jobarrayid": "i", "arrayjobid": "F", "arraytaskid": "K", "partition": "P",
                "name": "j", "username": "u", "userid": "U", "user": "u", "statecompact": "t", "state": "T",
                "timeused": "M", "timelimit": "l", "timeleft": "L", "numnodes": "D", "reasonlist": "R",
                "reason": "r", "nodelist": "N", "numcpus": "C", "tres-per-node": "b", "gres": "b", "starttime": "S",
                "submittime": "V", "endtime": "e", "priority": "Q", "account": "a", "command": "o", "workdir": "Z",
                "dependency": "E", "minmemory": "m", "qos": "q", "nice": "y", "comment": "k", "prioritylong": "Q"}


def sq_fields(j, node, now):
    def tl(x):
        return common.fmt_limit(x) if x else "UNLIMITED"
    run = (now - j["start"]) if j["state"] in ("RUNNING", "COMPLETING") and j["start"] else 0
    if j["state"] == "PENDING":
        reason = j["reason"] if j["reason"] != "None" else "None"
        rl = f"({reason})"
    else:
        rl = node
    jid = j.get("_display") or common.job_display_id(j)
    gpu = f"gres/gpu:{j['gpus']}" if j["gpus"] else "N/A"
    dep = j.get("dependency_str") or "(null)"
    if j["state"] == "PENDING" and j.get("dependency_state") == "ok":
        dep = "(null)"
    return {"i": jid, "A": str(j.get("array_job_id") or j["id"]), "F": str(j.get("array_job_id") or j["id"]),
            "K": str(j["array_task_id"]) if j.get("array_task_id") is not None else "N/A",
            "P": j["partition"], "j": j["name"], "u": j["user"], "U": str(j["uid"]),
            "t": common.STATE_CODE.get(j["state"], j["state"][:2]), "T": j["state"],
            "M": common.fmt_squeue_time(run), "l": tl(j["time_limit"]),
            "L": common.fmt_squeue_time(j["time_limit"] - run) if j["time_limit"] else "UNLIMITED",
            "D": "1", "R": rl, "r": j["reason"] if j["state"] == "PENDING" else "None",
            "N": node if j["state"] != "PENDING" else "", "C": str(j["cpus"]), "b": gpu,
            "S": common.fmt_ts(j["start"]) if j["start"] else "N/A", "V": common.fmt_ts(j["submit"]),
            "e": common.fmt_ts(j["start"] + j["time_limit"]) if j["start"] and j["time_limit"] else "N/A",
            "Q": str(j["priority"]), "p": "%.8f" % (j["priority"] / 4294967295.0), "a": j.get("account") or "(null)",
            "o": j.get("command") or "", "Z": j["workdir"], "E": dep, "m": common.fmt_mem(j["mem_mb"]),
            "q": j.get("qos") or "normal", "y": str(j.get("nice") or 0), "k": j.get("comment") or "(null)",
            "g": j["user"], "G": str(j.get("gid", 0)), "x": "", "f": "(null)", "B": node if j["start"] else "n/a",
            "c": str(j["cpus"]), "z": "*:*:*", "h": "OK", "n": "", "w": "(null)", "O": "0", "v": "(null)",
            "H": "*", "I": "*", "J": "*", "s": node if j["start"] else "", "X": "N/A", "Y": "(null)", "d": "0"}


def render(fmt, rows, header):
    """SLURM %-format: %[.][width]X; '.' right-justifies; width truncates."""
    import re
    toks = re.findall(r"%(\.?)(\d*)([A-Za-z%])|([^%]+)", fmt)
    out = []
    if header:
        out.append(fmt_line(toks, SQ_HEAD_OR(header)))
    for r in rows:
        out.append(fmt_line(toks, r))
    return "\n".join(out)


def SQ_HEAD_OR(h):
    return h


def fmt_line(toks, vals):
    s = []
    for dot, width, code, lit in toks:
        if lit:
            s.append(lit)
            continue
        if code == "%":
            s.append("%")
            continue
        v = str(vals.get(code, ""))
        if width:
            w = int(width)
            v = v[:w]
            v = v.rjust(w) if dot else v.ljust(w)
        s.append(v)
    return "".join(s).rstrip()


def long_format(spec):
    """-O/--Format 'JobID:12,State,TimeUsed:.10' -> a %-format string."""
    parts = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, w = item.partition(":")
        code = SQ_LONGNAMES.get(name.lower())
        if not code:
            die("squeue", f"Invalid job format specification: {name}")
        right = w.startswith(".")
        w = w.lstrip(".") or "20"
        parts.append("%" + ("." if right else "") + w + code)
    return " ".join(parts)


def squeue(argv):
    prog = "squeue"
    table = [("format", "o", True), ("Format", "O", True), ("noheader", "h", False), ("jobs", "j", True),
             ("user", "u", True), ("me", None, False), ("states", "t", True), ("long", "l", False),
             ("array", "r", False), ("partition", "p", True), ("name", "n", True), ("iterate", "i", True),
             ("start", None, False), ("sort", "S", True), ("all", "a", False), ("verbose", "v", False),
             ("help", None, False), ("usage", None, False), ("version", "V", False), ("steps", "s", False),
             ("nodelist", "w", True), ("account", "A", True), ("qos", "q", True), ("json", None, False),
             ("noconvert", None, False), ("local", None, False), ("federation", None, False), ("clusters", "M", True)]
    opts, pos = parse_opts(prog, argv, table, stop_at_positional=False)
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    if pos and not opts.get("jobs"):
        die(prog, "Unrecognized option: " + pos[0])
    iterate = int(opts["iterate"]) if opts.get("iterate") else None
    while True:
        rc = squeue_once(prog, opts)
        if not iterate or rc:
            return rc
        print()
        time.sleep(iterate)


def squeue_once(prog, opts):
    try:
        rep = request("jobs")
        node = request("node")["node"]
    except ControllerError as e:
        die(prog, f"slurm_load_jobs error: {e}")
    now = rep["now"]
    jobs = [j for j in rep["jobs"] if j["state"] in ("PENDING", "RUNNING", "COMPLETING")]
    if opts.get("jobs"):
        want = set(x.strip() for x in str(opts["jobs"]).split(",") if x.strip())
        jobs = [j for j in jobs if str(j["id"]) in want or str(j.get("array_job_id")) in want
                or common.job_display_id(j) in want]
    if opts.get("me"):
        jobs = [j for j in jobs if j["uid"] == os.getuid()]
    if opts.get("user"):
        users = set(str(opts["user"]).split(","))
        jobs = [j for j in jobs if j["user"] in users or str(j["uid"]) in users]
    if opts.get("name"):
        names = set(str(opts["name"]).split(","))
        jobs = [j for j in jobs if j["name"] in names]
    if opts.get("partition"):
        jobs = [j for j in jobs if j["partition"] in str(opts["partition"]).split(",")]
    if opts.get("states"):
        st = set(s.strip().upper() for s in str(opts["states"]).split(","))
        if "ALL" not in st:
            jobs = [j for j in jobs if j["state"] in st or common.STATE_CODE.get(j["state"]) in st]
    if opts.get("json"):
        print(json.dumps({"jobs": jobs}, indent=2))
        return 0
    if not opts.get("array"):                      # pending array tasks that have not started share one line
        merged, seen = [], {}
        for j in jobs:
            a = j.get("array_job_id")
            if a is not None and j["state"] == "PENDING":
                key = (a, j["reason"])
                if key in seen:
                    seen[key]["_tasks"].append(j["array_task_id"])
                    continue
                j = dict(j)
                j["_tasks"] = [j["array_task_id"]]
                seen[key] = j
            merged.append(j)
        for j in merged:
            if "_tasks" in j:
                rng = common.compress_indices(j["_tasks"])
                thr = f"%{j['array_throttle']}" if j.get("array_throttle") else ""
                j["_display"] = f"{j['array_job_id']}_" + (f"[{rng}{thr}]" if len(j["_tasks"]) > 1 or thr
                                                           else str(j["_tasks"][0]))
        jobs = merged
    order = {"RUNNING": 0, "COMPLETING": 0, "PENDING": 1}
    jobs.sort(key=lambda j: (order.get(j["state"], 2), -j["priority"], j["id"]) if j["state"] == "PENDING"
              else (order.get(j["state"], 2), j["id"]))
    if opts.get("Format"):
        fmt = long_format(opts["Format"])
    else:
        fmt = opts.get("format") or os.environ.get("SQUEUE_FORMAT") or (SQ_LONG if opts.get("long") else SQ_DEFAULT)
    if opts.get("long") and not opts.get("noheader"):
        print(time.strftime("%a %b %d %H:%M:%S %Y"))
    rows = [sq_fields(j, node, now) for j in jobs]
    text = render(fmt, rows, None if opts.get("noheader") else SQ_HEAD)
    if text:
        print(text)
    return 0


# ===================================================================== sacct
SA_DEFAULT = "JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode"
SA_WIDTH = {"jobid": 12, "jobidraw": 12, "jobname": 10, "partition": 10, "account": 10, "alloccpus": 10,
            "ncpus": 10, "state": 10, "exitcode": 8, "submit": 19, "eligible": 19, "start": 19, "end": 19,
            "elapsed": 10, "elapsedraw": 10, "timelimit": 10, "timelimitraw": 10, "nodelist": 15, "alloctres": 10,
            "reqtres": 10, "user": 9, "uid": 6, "group": 9, "gid": 6, "workdir": 10, "reason": 20, "priority": 10,
            "reqmem": 10, "nnodes": 8, "derivedexitcode": 10, "comment": 10, "cputime": 10, "cputimeraw": 10,
            "qos": 10, "allocnodes": 10, "reqcpus": 8, "submitline": 20, "maxrss": 10, "averss": 10, "planned": 10,
            "plannedcpu": 10, "reserved": 10, "cluster": 10, "maxvmsize": 10, "reqnodes": 8, "constraints": 19,
            "flags": 20, "jobname%": 10, "allocgres": 12, "reqgres": 12, "suspended": 10, "systemcpu": 10,
            "usercpu": 10, "totalcpu": 10, "ntasks": 8, "layout": 10, "consumedenergy": 14, "state%": 10,
            "dbindex": 10, "nodelist%": 15, "mcslabel": 10, "container": 10, "tresusageintot": 10}


def sa_rows(j, node, now, steps):
    end = j["end"] or (now if j["state"] == "RUNNING" else None)
    elapsed = int((end - j["start"])) if j["start"] and end else 0
    state = j["state"]
    if state == "CANCELLED" and j.get("cancelled_by") is not None:
        state = f"CANCELLED by {j['cancelled_by']}"
    ec = f"{j['exit_code'] or 0}:{j['signal'] or 0}" if j["state"] in common.TERMINAL else "0:0"
    tres = f"billing={j['cpus']},cpu={j['cpus']},mem={common.fmt_mem(j['mem_mb'])},node=1"
    if j["gpus"]:
        tres += f",gres/gpu={j['gpus']}"
    alloc = tres if j["start"] else ""
    planned = int((j["start"] or now) - (j["eligible"] or j["submit"])) if (j["eligible"] or j["start"]) else 0
    base = {"jobid": common.job_display_id(j), "jobidraw": str(j["id"]), "jobname": j["name"],
            "partition": j["partition"], "account": j.get("account") or "", "alloccpus": str(j["cpus"] if j["start"] else 0),
            "ncpus": str(j["cpus"]), "state": state, "exitcode": ec, "submit": common.fmt_ts(j["submit"]),
            "eligible": common.fmt_ts(j["eligible"]), "start": common.fmt_ts(j["start"]),
            "end": common.fmt_ts(j["end"]), "elapsed": common.fmt_elapsed(elapsed), "elapsedraw": str(elapsed),
            "timelimit": common.fmt_limit(j["time_limit"]) if j["time_limit"] else "UNLIMITED",
            "timelimitraw": str((j["time_limit"] or 0) // 60) if j["time_limit"] else "UNLIMITED",
            "nodelist": node if j["start"] else "None assigned", "alloctres": alloc, "reqtres": tres,
            "user": j["user"], "uid": str(j["uid"]), "group": j["user"], "gid": str(j.get("gid", 0)),
            "workdir": j["workdir"], "reason": j["reason"], "priority": str(j["priority"]),
            "reqmem": common.fmt_mem(j["mem_mb"]), "nnodes": "1", "derivedexitcode": "0:0",
            "comment": j.get("comment") or "", "cputime": common.fmt_elapsed(elapsed * j["cpus"]),
            "cputimeraw": str(elapsed * j["cpus"]), "qos": j.get("qos") or "normal",
            "allocnodes": "1" if j["start"] else "0", "reqcpus": str(j["cpus"]), "submitline": j.get("command") or "",
            "maxrss": "", "averss": "", "planned": common.fmt_elapsed(planned), "plannedcpu": "",
            "reserved": common.fmt_elapsed(planned), "cluster": "cluster", "maxvmsize": "", "reqnodes": "1",
            "constraints": "", "flags": "SchedMain", "allocgres": f"gpu:{j['gpus']}" if j["gpus"] and j["start"] else "",
            "reqgres": f"gpu:{j['gpus']}" if j["gpus"] else "", "suspended": "00:00:00", "systemcpu": "",
            "usercpu": "", "totalcpu": "", "ntasks": "", "layout": "", "consumedenergy": "0", "dbindex": str(j["id"]),
            "mcslabel": "", "container": "", "tresusageintot": ""}
    rows = [base]
    if steps and j["start"] and j["kind"] == "batch":
        for suffix, name in ((".batch", "batch"), (".extern", "extern")):
            r = dict(base)
            r.update({"jobid": base["jobid"] + suffix, "jobidraw": str(j["id"]) + suffix, "jobname": name,
                      "partition": "", "user": "", "timelimit": "", "timelimitraw": "", "reqmem": "",
                      "submitline": "", "reason": "", "priority": "", "qos": "", "comment": "", "flags": ""})
            if suffix == ".extern":
                r["exitcode"] = "0:0"
                if j["state"] in common.TERMINAL:
                    r["state"] = "COMPLETED" if j["state"] not in ("CANCELLED",) else "CANCELLED"
            elif j["state"] in ("TIMEOUT",) or str(state).startswith("CANCELLED"):
                r["state"] = "CANCELLED"
            elif j["state"] == "RUNNING":
                r["state"] = "RUNNING"
            r["alloctres"] = f"cpu={j['cpus']},mem={common.fmt_mem(j['mem_mb'])},node=1" + (
                f",gres/gpu={j['gpus']}" if j["gpus"] else "")
            rows.append(r)
    return rows


def sacct(argv):
    prog = "sacct"
    table = [("jobs", "j", True), ("format", "o", True), ("parsable", "p", False), ("parsable2", "P", False),
             ("noheader", "n", False), ("allocations", "X", False), ("starttime", "S", True), ("endtime", "E", True),
             ("state", "s", True), ("user", "u", True), ("name", None, True), ("brief", "b", False),
             ("long", "l", False), ("allusers", "a", False), ("units", None, True), ("delimiter", None, True),
             ("helpformat", "e", False), ("help", "h", False), ("version", "V", False), ("duplicates", "D", False),
             ("partition", "r", True), ("noconvert", None, False), ("json", None, False), ("array", None, False),
             ("verbose", "v", False), ("clusters", "M", True), ("accounts", "A", True), ("truncate", "T", False)]
    opts, pos = parse_opts(prog, argv, table, stop_at_positional=False)
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    if opts.get("helpformat"):
        print("Account AllocCPUS AllocNodes AllocTRES Cluster Comment CPUTime CPUTimeRAW DerivedExitCode Elapsed "
              "ElapsedRaw Eligible End ExitCode GID Group JobID JobIDRaw JobName NCPUS NNodes NodeList Partition "
              "Planned Priority QOS Reason ReqCPUS ReqMem ReqTRES Reserved Start State Submit SubmitLine Timelimit "
              "TimelimitRaw UID User WorkDir")
        return 0
    try:
        rep = request("jobs")
        node = request("node")["node"]
    except ControllerError as e:
        die(prog, f"{e}")
    now = rep["now"]
    jobs = rep["jobs"]
    if opts.get("jobs"):
        want = set(x.strip().split(".")[0] for x in str(opts["jobs"]).split(",") if x.strip())
        jobs = [j for j in jobs if str(j["id"]) in want or str(j.get("array_job_id")) in want
                or common.job_display_id(j) in want]
    else:
        if not opts.get("allusers") and os.getuid() != 0:
            jobs = [j for j in jobs if j["uid"] == os.getuid()]
        start = time.time() - (time.time() % 86400) if not opts.get("starttime") else None
        if start is not None:
            jobs = [j for j in jobs if (j["end"] or now) >= start - 86400 * 0]
    if opts.get("user"):
        users = set(str(opts["user"]).split(","))
        jobs = [j for j in jobs if j["user"] in users or str(j["uid"]) in users]
    if opts.get("name"):
        jobs = [j for j in jobs if j["name"] in str(opts["name"]).split(",")]
    if opts.get("state"):
        st = set(s.strip().upper() for s in str(opts["state"]).split(","))
        jobs = [j for j in jobs if j["state"] in st or common.STATE_CODE.get(j["state"]) in st]
    fields = opts.get("format") or os.environ.get("SACCT_FORMAT") or (
        "JobID,State,ExitCode" if opts.get("brief") else SA_DEFAULT)
    cols = []
    for f in fields.split(","):
        f = f.strip()
        if not f:
            continue
        name, _, w = f.partition("%")
        key = name.lower()
        if key not in SA_WIDTH:
            die(prog, f"Invalid field requested: \"{name}\"")
        cols.append((name, key, int(w) if w.isdigit() else SA_WIDTH[key]))
    rows = []
    for j in sorted(jobs, key=lambda x: x["id"]):
        rows.extend(sa_rows(j, node, now, steps=not opts.get("allocations")))
    if opts.get("json"):
        print(json.dumps({"jobs": jobs}, indent=2))
        return 0
    p2 = opts.get("parsable2")
    p1 = opts.get("parsable")
    delim = opts.get("delimiter") or "|"
    out = []
    if p1 or p2:
        if not opts.get("noheader"):
            out.append(delim.join(n for n, _, _ in cols) + (delim if p1 else ""))
        for r in rows:
            out.append(delim.join(str(r.get(k, "")) for _, k, _ in cols) + (delim if p1 else ""))
    else:
        if not opts.get("noheader"):
            out.append(" ".join(n[:w].rjust(w) if False else n[:w].ljust(w) for n, _, w in cols).rstrip())
            out.append(" ".join("-" * w for _, _, w in cols))
        for r in rows:
            cells = []
            for _, k, w in cols:
                v = str(r.get(k, ""))
                if len(v) > w:
                    v = v[:w - 1] + "+"
                cells.append(v.rjust(w) if k in ("alloccpus", "ncpus", "exitcode", "elapsedraw", "uid", "gid",
                                                 "priority", "nnodes", "reqcpus", "cputimeraw", "allocnodes")
                             else v.ljust(w))
            out.append(" ".join(cells).rstrip())
    if out:
        print("\n".join(out))
    return 0


# ===================================================================== scancel
def scancel(argv):
    prog = "scancel"
    table = [("user", "u", True), ("me", None, False), ("name", "n", True), ("state", "t", True),
             ("signal", "s", True), ("batch", "b", False), ("full", "f", False), ("quiet", "Q", False),
             ("verbose", "v", False), ("interactive", "i", False), ("partition", "p", True), ("help", "h", False),
             ("version", "V", False), ("account", "A", True), ("qos", "q", True), ("hurry", "H", False)]
    opts, pos = parse_opts(prog, argv, table, stop_at_positional=False)
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    if not pos and not (opts.get("user") or opts.get("me") or opts.get("name") or opts.get("state")):
        die(prog, "No job identification provided")
    sig = None
    if opts.get("signal"):
        s = str(opts["signal"]).upper().replace("SIG", "")
        sig = int(s) if s.isdigit() else getattr(signal, "SIG" + s, None)
        if sig is None:
            die(prog, f"Invalid signal name: {opts['signal']}")
        sig = int(sig)
    user = None
    if opts.get("me"):
        user = me()[2]
    elif opts.get("user"):
        user = opts["user"]
    states = [s.strip().upper() for s in str(opts["state"]).split(",")] if opts.get("state") else None
    ids = []
    for p in pos:
        ids.extend(x for x in p.split(",") if x)
    try:
        rep = request("cancel", ids=ids, target_user=user, name=opts.get("name"), states=states, signal=sig, uid=os.getuid())
    except ControllerError as e:
        die(prog, f"{e}")
    if rep.get("error"):
        sys.stderr.write(f"scancel: error: {rep['error']}\n")
        return 1
    return 0


# ===================================================================== sinfo
SI_DEFAULT = "%9P %.5a %.10l %.6D %.6t %N"


def sinfo(argv):
    prog = "sinfo"
    table = [("format", "o", True), ("noheader", "h", False), ("Node", "N", False), ("long", "l", False),
             ("summarize", "s", False), ("partition", "p", True), ("states", "t", True), ("all", "a", False),
             ("help", None, False), ("version", "V", False), ("responding", "r", False), ("dead", "d", False),
             ("exact", "e", False), ("iterate", "i", True), ("verbose", "v", False), ("Format", "O", True),
             ("json", None, False), ("noconvert", None, False), ("list-reasons", "R", False)]
    opts, pos = parse_opts(prog, argv, table, stop_at_positional=False)
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    try:
        n = request("node")
    except ControllerError as e:
        die(prog, f"{e}")
    st = n["state"]
    a = len(n["alloc_gpus"])
    vals = {"P": n["partition"] + "*", "R": n["partition"], "a": "up", "l": "infinite", "D": "1", "t": st[:6],
            "T": st, "N": n["node"], "n": n["node"], "G": f"gpu:{n['gpu_type']}:{n['gpus']}",
            "C": f"{n['alloc_cpus']}/{n['cpus'] - n['alloc_cpus']}/0/{n['cpus']}", "c": str(n["cpus"]),
            "m": str(n["mem_mb"]), "e": str(n["mem_mb"] - n["alloc_mem_mb"]), "F": "0/1/0/1" if st != "idle" else "0/1/0/1",
            "A": "1/0" if st != "idle" else "0/1", "f": "(null)", "E": "none", "s": "unlimited", "X": "1", "Y": "1",
            "Z": "1", "z": "1:22:2", "O": "0.00", "d": "0", "w": "1", "h": "NO", "g": "all", "b": f"gpu:{n['gpu_type']}:{a}"}
    head = {"P": "PARTITION", "R": "PARTITION", "a": "AVAIL", "l": "TIMELIMIT", "D": "NODES", "t": "STATE",
            "T": "STATE", "N": "NODELIST", "n": "HOSTNAMES", "G": "GRES", "C": "CPUS(A/I/O/T)", "c": "CPUS",
            "m": "MEMORY", "e": "FREE_MEM", "F": "NODES(A/I/O/T)", "A": "NODES(A/I)", "f": "AVAIL_FEATURES",
            "E": "REASON", "s": "JOB_SIZE", "X": "SOCKETS", "Y": "CORES", "Z": "THREADS", "z": "S:C:T", "O": "CPU_LOAD",
            "d": "TMP_DISK", "w": "WEIGHT", "h": "OVERSUBSCRIBE", "g": "GROUPS", "b": "GRES_USED"}
    if opts.get("format"):
        fmt = opts["format"]
    elif opts.get("Node") and opts.get("long"):
        fmt = "%N %.6D %.9P %.11T %.4c %.8z %.6m %.8d %.6w %.8f %20E"
    elif opts.get("Node"):
        fmt = "%N %.6D %.9P %.6t"
    elif opts.get("long"):
        fmt = "%9P %.5a %.10l %.10s %.4r %.8h %.10g %.6D %.11T %N"
        vals["r"] = "no"
        head["r"] = "ROOT"
    elif opts.get("summarize"):
        fmt = "%9P %.5a %.10l %.16F %N"
    else:
        fmt = SI_DEFAULT
    import re
    toks = re.findall(r"%(\.?)(\d*)([A-Za-z%])|([^%]+)", fmt)
    lines = []
    if opts.get("long") and not opts.get("noheader"):
        lines.append(time.strftime("%a %b %d %H:%M:%S %Y"))
    if not opts.get("noheader"):
        lines.append(fmt_line(toks, head))
    lines.append(fmt_line(toks, vals))
    print("\n".join(lines))
    return 0


# ===================================================================== scontrol
def scontrol(argv):
    prog = "scontrol"
    args = [a for a in argv if not a.startswith("-")]
    flags = [a for a in argv if a.startswith("-")]
    oneliner = "-o" in flags or "--oneliner" in flags
    if not args:
        die(prog, "no command given")
    cmd = args[0].lower()
    try:
        if cmd == "show" and len(args) >= 2:
            what = args[1].lower()
            if what in ("job", "jobs"):
                return show_jobs(args[2] if len(args) > 2 else None, oneliner)
            if what in ("node", "nodes"):
                return show_node(oneliner)
            if what in ("partition", "partitions"):
                return show_partition(oneliner)
            if what == "config":
                n = request("node")
                print(f"Configuration data as of {common.fmt_ts(time.time())}\nClusterName             = cluster\n"
                      f"SLURM_VERSION           = {common.VERSION}\nSelectType              = select/cons_tres\n"
                      f"GresTypes               = gpu\nSchedulerType           = sched/backfill\n"
                      f"DependencyParameters    = kill_invalid_depend\nSlurmctldHost[0]        = {n['node']}")
                return 0
            die(prog, f"invalid entity: {args[1]} for keyword: show")
        if cmd in ("hold", "uhold", "release"):
            ids = ",".join(args[1:]).split(",")
            rep = request("hold" if cmd != "release" else "release", ids=[i for i in ids if i])
            if rep.get("error"):
                die(prog, rep["error"])
            return 0
        if cmd == "shutdown":
            rep = request("shutdown", uid=os.getuid())
            if rep.get("error"):
                die(prog, rep["error"])
            return 0
        if cmd == "ping":
            r = request("ping")
            n = request("node")
            print(f"Slurmctld(primary) at {n['node']} is UP" if r.get("ok") else "Slurmctld(primary) is DOWN")
            return 0
        if cmd in ("version", "-v"):
            print("slurm " + common.VERSION)
            return 0
    except ControllerError as e:
        die(prog, f"{e}")
    die(prog, f"Invalid command: {args[0]}")


def show_jobs(spec, oneliner):
    rep = request("jobs")
    node = request("node")["node"]
    now = rep["now"]
    jobs = rep["jobs"]
    if spec:
        jobs = [j for j in jobs if str(j["id"]) == spec or str(j.get("array_job_id")) == spec
                or common.job_display_id(j) == spec]
        if not jobs:
            sys.stderr.write("slurm_load_jobs error: Invalid job id specified\n")
            return 1
    if not jobs:
        print("No jobs in the system")
        return 0
    blocks = []
    for j in jobs:
        run = int(((j["end"] or now) - j["start"])) if j["start"] else 0
        ec = f"{j['exit_code'] or 0}:{j['signal'] or 0}"
        tl = common.fmt_limit(j["time_limit"]) if j["time_limit"] else "UNLIMITED"
        tres = f"cpu={j['cpus']},mem={common.fmt_mem(j['mem_mb'])},node=1,billing={j['cpus']}" + (
            f",gres/gpu={j['gpus']}" if j["gpus"] else "")
        arr = (f" ArrayJobId={j['array_job_id']} ArrayTaskId={j['array_task_id']}"
               if j.get("array_job_id") is not None else "")
        fields = [
            f"JobId={j['id']}{arr} JobName={j['name']}",
            f"UserId={j['user']}({j['uid']}) GroupId={j['user']}({j.get('gid', 0)}) MCS_label=N/A",
            f"Priority={j['priority']} Nice={j.get('nice') or 0} Account={j.get('account') or '(null)'} "
            f"QOS={j.get('qos') or 'normal'}",
            f"JobState={j['state']} Reason={j['reason']} Dependency={j.get('dependency_str') or '(null)'}",
            f"Requeue=1 Restarts=0 BatchFlag={1 if j['kind'] == 'batch' else 0} Reboot=0 ExitCode={ec}",
            f"RunTime={common.fmt_elapsed(run)} TimeLimit={tl} TimeMin=N/A",
            f"SubmitTime={common.fmt_ts(j['submit'])} EligibleTime={common.fmt_ts(j['eligible'])}",
            f"AccrueTime={common.fmt_ts(j['eligible'])}",
            f"StartTime={common.fmt_ts(j['start'])} EndTime={common.fmt_ts(j['end'])} Deadline=N/A",
            f"Partition={j['partition']} AllocNode:Sid={node}:1",
            f"NodeList={node if j['start'] else '(null)'} BatchHost={node if j['start'] else '(null)'}",
            f"NumNodes=1 NumCPUs={j['cpus']} NumTasks=1 CPUs/Task={j['cpus']} ReqB:S:C:T=0:0:*:*",
            f"ReqTRES={tres}", f"AllocTRES={tres if j['start'] else '(null)'}",
            f"MinCPUsNode={j['cpus']} MinMemoryNode={common.fmt_mem(j['mem_mb'])} MinTmpDiskNode=0",
            f"Command={j.get('command') or '(null)'}", f"WorkDir={j['workdir']}",
            f"StdErr={j.get('stderr') or j.get('stdout') or os.path.join(j['workdir'], 'slurm-%d.out' % j['id'])}",
            f"StdOut={j.get('stdout') or os.path.join(j['workdir'], 'slurm-%d.out' % j['id'])}",
        ]
        if j["gpus"]:
            fields.append(f"TresPerNode=gres/gpu:{j['gpus']}")
        blocks.append(" ".join(fields) if oneliner else "\n   ".join(fields))
    print(("\n" if oneliner else "\n\n").join(blocks) + ("" if oneliner else "\n"))
    return 0


def show_node(oneliner):
    n = request("node")
    f = [f"NodeName={n['node']} Arch=x86_64 CoresPerSocket={max(1, n['cpus'] // 2)}",
         f"CPUAlloc={n['alloc_cpus']} CPUEfctv={n['cpus']} CPUTot={n['cpus']} CPULoad=0.00",
         "AvailableFeatures=(null)", "ActiveFeatures=(null)", f"Gres=gpu:{n['gpu_type']}:{n['gpus']}",
         f"NodeAddr={n['node']} NodeHostName={n['node']} Version={n['version'][:5]}",
         f"RealMemory={n['mem_mb']} AllocMem={n['alloc_mem_mb']} FreeMem={n['mem_mb'] - n['alloc_mem_mb']} Sockets=1 Boards=1",
         f"State={n['state'].upper()} ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A",
         f"Partitions={n['partition']}",
         f"CfgTRES=cpu={n['cpus']},mem={n['mem_mb']}M,billing={n['cpus']},gres/gpu={n['gpus']}",
         f"AllocTRES=cpu={n['alloc_cpus']},mem={n['alloc_mem_mb']}M,gres/gpu={len(n['alloc_gpus'])}"]
    print(" ".join(f) if oneliner else "\n   ".join(f))
    return 0


def show_partition(oneliner):
    n = request("node")
    f = [f"PartitionName={n['partition']}", "AllowGroups=ALL AllowAccounts=ALL AllowQos=ALL",
         "Default=YES QoS=N/A", "DefaultTime=NONE DisableRootJobs=NO ExclusiveUser=NO GraceTime=0 Hidden=NO",
         "MaxNodes=UNLIMITED MaxTime=UNLIMITED MinNodes=0 LLN=NO MaxCPUsPerNode=UNLIMITED",
         f"Nodes={n['node']}", "State=UP OverSubscribe=NO",
         f"TotalCPUs={n['cpus']} TotalNodes=1 SelectTypeParameters=NONE",
         f"TRES=cpu={n['cpus']},mem={n['mem_mb']}M,node=1,billing={n['cpus']},gres/gpu={n['gpus']}"]
    print(" ".join(f) if oneliner else "\n   ".join(f))
    return 0


# ===================================================================== srun
def srun(argv):
    prog = "srun"
    opts, pos = parse_opts(prog, argv, SBATCH_OPTS + SRUN_EXTRA)
    if opts.get("version"):
        print("slurm " + common.VERSION)
        return 0
    if not pos:
        die(prog, "No command given to execute.")
    if os.environ.get("SLURM_JOB_ID"):              # a step inside an existing allocation: run it in place
        env = dict(os.environ)
        env["SLURM_STEP_ID"] = "0"
        try:
            return subprocess.call(pos, env=env)
        except FileNotFoundError:
            die(prog, f"execve(): {pos[0]}: No such file or directory", 2)
    req = build_request(prog, opts, "", opts.get("job-name") or os.path.basename(pos[0]), kind="interactive",
                        command=" ".join(pos))
    req["srun_pid"] = os.getpid()
    try:
        rep = request("submit", **req)
    except ControllerError as e:
        die(prog, f"Unable to allocate resources: {e}")
    if rep.get("error"):
        die(prog, "Unable to allocate resources: " + rep["error"].replace("Batch job submission failed: ", ""))
    jid = rep["job_id"]
    announced = False
    while True:
        j = [x for x in request("jobs")["jobs"] if x["id"] == jid][0]
        if j["state"] == "RUNNING":
            break
        if j["state"] in common.TERMINAL:
            sys.stderr.write(f"srun: error: Unable to allocate resources: Job {jid} was cancelled\n")
            return 1
        if not announced:
            sys.stderr.write(f"srun: job {jid} queued and waiting for resources\n")
            announced = True
        time.sleep(0.5)
    if announced:
        sys.stderr.write(f"srun: job {jid} has been allocated resources\n")
    env = dict(req["env"])
    gl = ",".join(str(g) for g in j["gpu_ids"])
    env.update({"SLURM_JOB_ID": str(jid), "SLURM_JOBID": str(jid), "SLURM_STEP_ID": "0", "SLURM_JOB_NAME": j["name"],
                "SLURM_NNODES": "1", "SLURM_NTASKS": "1", "SLURM_CPUS_ON_NODE": str(j["cpus"]),
                "SLURM_JOB_NODELIST": request("node")["node"], "CUDA_VISIBLE_DEVICES": gl, "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0"})
    if j["gpu_ids"]:
        env["SLURM_JOB_GPUS"] = gl
    if not gl:
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = ""
    rc, sig = 1, 0
    child = {}

    def forward(signum, _frame):                    # scancel or the time limit: pass it on to the task, as srun does
        if child.get("p"):
            child["p"].send_signal(signum)
    signal.signal(signal.SIGTERM, forward)
    try:
        p = child["p"] = subprocess.Popen(pos, env=env)
        try:
            rc = p.wait()
        except KeyboardInterrupt:
            p.send_signal(signal.SIGINT)
            rc = p.wait()
        if rc < 0:
            sig, rc = -rc, 0
    except FileNotFoundError:
        sys.stderr.write(f"slurmstepd: error: execve(): {pos[0]}: No such file or directory\n")
        sys.stderr.write(f"srun: error: {request('node')['node']}: task 0: Exited with exit code 2\n")
        rc = 2
    finally:
        try:
            request("interactive_done", job_id=jid, rc=rc, signal=sig)
        except ControllerError:
            pass
    if rc:
        sys.stderr.write(f"srun: error: {env['SLURM_JOB_NODELIST']}: task 0: Exited with exit code {rc}\n")
    return rc if not sig else 128 + sig


COMMANDS = {"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scancel": scancel, "sinfo": sinfo,
            "scontrol": scontrol, "srun": srun}


def main():
    name = os.path.basename(sys.argv[0])
    if name not in COMMANDS and len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        name = sys.argv.pop(1)
    fn = COMMANDS.get(name)
    if not fn:
        sys.stderr.write("usage: one of " + ", ".join(COMMANDS) + "\n")
        return 2
    try:
        return fn(sys.argv[1:]) or 0
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
