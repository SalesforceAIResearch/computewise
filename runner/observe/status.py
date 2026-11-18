"""Live status of the ComputeWise host: the whole machine first, then every container as its share of it.

usage, on the VM (needs sudo for docker):   sudo python3 -I status.py [--runs /srv/runs] [--json]
from the controller:                            bash experiments/gcp/host/vm.sh ssh 'sudo python3 -I /opt/computewise/runner/observe/status.py'

Why the whole host first: on 2026-10-19 the controller overloaded and had to be rebooted although every
session stayed inside its own caps, because nobody summed them. So this
shows, before any single box:
  - the host's CPU busy %, memory used and pressure, and disk use on every mounted data path;
  - the sum of the caps of all running containers against the host (CPU, memory), which is what an
    admission check must keep under the host minus a reserve;
  - every running container, ComputeWise or not, with its live CPU, memory and its caps, each as a share
    of the host;
  - for each live ComputeWise run directory under --runs: task, harness, model, how long it has run, GPU
    jobs alive, and the latest per-GPU busy state from gpu-mock's occupancy log.

Reads only: /proc, /sys/fs/cgroup, `docker ps` and `docker inspect`, and the run directories. It
never changes anything. Standard library only.
"""
import argparse, glob, json, os, shutil, subprocess, sys, time

CG = "/sys/fs/cgroup"

def read(path, default=None):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return default

def kv(path):
    out = {}
    for line in (read(path, "") or "").splitlines():
        p = line.split()
        if len(p) >= 2:
            try:
                out[p[0]] = int(p[1])
            except ValueError:
                pass
    return out

def pressure(path):
    """`some` and `full` avg10 from a PSI file."""
    out = {}
    for line in (read(path, "") or "").splitlines():
        p = line.split()
        if p and p[0] in ("some", "full"):
            out[p[0]] = float(dict(x.split("=") for x in p[1:]).get("avg10", 0.0))
    return out

def host_cpu_sample():
    f = read("/proc/stat").splitlines()[0].split()[1:]
    vals = list(map(int, f))
    idle = vals[3] + vals[4]
    return sum(vals), idle

def meminfo():
    m = {}
    for line in read("/proc/meminfo").splitlines():
        k, v = line.split(":", 1)
        m[k] = int(v.split()[0]) * 1024
    return m

def docker(*args):
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""

def containers():
    ids = docker("ps", "-q", "--no-trunc").split()
    if not ids:
        return []
    info = json.loads(docker("inspect", *ids) or "[]")
    out = []
    for c in info:
        hc = c.get("HostConfig", {})
        cid = c["Id"]
        scope = os.path.join(CG, "system.slice", f"docker-{cid}.scope")
        cpu_max = (read(os.path.join(scope, "cpu.max"), "max 100000") or "max 100000").split()
        cpus_cap = None if cpu_max[0] == "max" else int(cpu_max[0]) / int(cpu_max[1])
        if hc.get("CpusetCpus"):
            n = 0
            for part in hc["CpusetCpus"].split(","):
                a, _, b = part.partition("-")
                n += (int(b) - int(a) + 1) if b else 1
            cpus_cap = min(cpus_cap, n) if cpus_cap else n
        mem_max = read(os.path.join(scope, "memory.max"), "max")
        out.append({
            "name": c["Name"].lstrip("/"), "id": cid[:12], "image": c["Config"]["Image"],
            "labels": c["Config"].get("Labels") or {},
            "scope": scope, "cpus_cap": cpus_cap, "cpuset": hc.get("CpusetCpus") or "",
            "mem_cap": None if mem_max in (None, "max") else int(mem_max),
            "mem_now": int(read(os.path.join(scope, "memory.current"), "0") or 0),
            "mem_peak": int(read(os.path.join(scope, "memory.peak"), "0") or 0),
            "oom_kill": kv(os.path.join(scope, "memory.events")).get("oom_kill", 0),
            "cpu_usec": kv(os.path.join(scope, "cpu.stat")).get("usage_usec", 0),
            "mem_pressure": pressure(os.path.join(scope, "memory.pressure")),
            "started": c["State"].get("StartedAt", ""),
        })
    return out

def runs(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        st = os.path.join(d, "status.json")
        rj = os.path.join(d, "run.json")
        if not os.path.isdir(d) or (os.path.exists(st) and json.load(open(st)).get("end_utc")):
            continue                                # finished runs are not live
        if not os.path.exists(rj):
            continue                                # not a run directory (old probes and replays, a logs dir)
        try:
            r = json.load(open(rj)) if os.path.exists(rj) else {}
        except ValueError:
            r = {}
        run = dict(r.get("inputs") or {}, **{k: r[k] for k in ("task", "harness", "model") if r.get(k) is not None})
        occ = os.path.join(d, "occupancy.jsonl")
        alive, busy_until, last = set(), {}, None
        if os.path.exists(occ):
            for line in open(occ, errors="replace"):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last = e.get("t_wall", last)
                k = e.get("kind")
                if k == "job_hello":
                    alive.add(e["job"])
                elif k == "job_bye":
                    alive.discard(e["job"])
                elif k == "enqueue":
                    g = e["gpu"]
                    start = max(busy_until.get(g, 0.0), e["t_mono"])
                    busy_until[g] = start + float(e.get("wall_cost", 0.0))
        out.append({"dir": d, "task": run.get("task"), "harness": run.get("harness"), "model": run.get("model"),
                    "jobs_alive": len(alive), "occupancy_events_last_wall": last,
                    "gpu_busy_until_mono": busy_until})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/srv/runs")
    ap.add_argument("--interval", type=float, default=1.0, help="CPU sampling interval, seconds")
    ap.add_argument("--reserve-cpus", type=float, default=2.0)
    ap.add_argument("--reserve-mem-gib", type=float, default=8.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    ncpu = os.cpu_count()
    t0, i0 = host_cpu_sample()
    cs0 = containers()
    time.sleep(a.interval)
    t1, i1 = host_cpu_sample()
    cs1 = {c["id"]: c for c in containers()}
    host_busy = 1.0 - (i1 - i0) / max(1, t1 - t0)
    m = meminfo()
    mem_total, mem_avail = m["MemTotal"], m["MemAvailable"]
    disks = {}
    for p in ("/", "/srv"):
        if os.path.exists(p):
            u = shutil.disk_usage(p)
            disks[p] = {"total": u.total, "used": u.used, "free": u.free}
    rows = []
    for c in cs0:
        c1 = cs1.get(c["id"])
        if not c1:
            continue
        cores = (c1["cpu_usec"] - c["cpu_usec"]) / 1e6 / a.interval
        rows.append({**c1, "cpu_cores_now": cores})
    cap_cpu = sum(r["cpus_cap"] or ncpu for r in rows)
    cap_mem = sum(r["mem_cap"] or mem_total for r in rows)
    uncapped = [r["name"] for r in rows if r["cpus_cap"] is None or r["mem_cap"] is None]
    res = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {"cpus": ncpu, "cpu_busy_share": host_busy, "mem_total": mem_total,
                 "mem_used": mem_total - mem_avail, "pressure_cpu": pressure("/proc/pressure/cpu"),
                 "pressure_mem": pressure("/proc/pressure/memory"), "pressure_io": pressure("/proc/pressure/io"),
                 "disks": disks},
        "caps_sum": {"cpus": cap_cpu, "mem": cap_mem, "cpus_share_of_host": cap_cpu / ncpu,
                     "mem_share_of_host": cap_mem / mem_total, "uncapped": uncapped,
                     "admission_headroom_cpus": ncpu - a.reserve_cpus - cap_cpu,
                     "admission_headroom_mem": mem_total - a.reserve_mem_gib * 2**30 - cap_mem},
        "containers": rows, "runs": runs(a.runs),
    }
    if a.json:
        print(json.dumps(res, indent=1, default=str)); return
    G = 2**30
    h = res["host"]; cs = res["caps_sum"]
    print(f"host  {res['time_utc']}   CPU {100*h['cpu_busy_share']:5.1f}% of {ncpu}   "
          f"mem {h['mem_used']/G:6.1f} / {h['mem_total']/G:.1f} GiB ({100*h['mem_used']/h['mem_total']:.0f}%)   "
          f"PSI cpu {h['pressure_cpu'].get('some',0):.1f} mem {h['pressure_mem'].get('some',0):.1f}/{h['pressure_mem'].get('full',0):.1f} io {h['pressure_io'].get('some',0):.1f}")
    for p, d in disks.items():
        print(f"      disk {p:<5} {d['used']/G:7.1f} / {d['total']/G:.1f} GiB ({100*d['used']/d['total']:.0f}%)")
    print(f"caps  sum over {len(rows)} containers: CPU {cs['cpus']:.1f} ({100*cs['cpus_share_of_host']:.0f}% of host)   "
          f"mem {cs['mem']/G:.1f} GiB ({100*cs['mem_share_of_host']:.0f}% of host)   "
          f"headroom after reserve: {cs['admission_headroom_cpus']:.1f} CPU, {cs['admission_headroom_mem']/G:.1f} GiB"
          + (f"   UNCAPPED: {', '.join(uncapped)}" if uncapped else ""))
    for r in sorted(rows, key=lambda r: -r["mem_now"]):
        capc = f"{r['cpus_cap']:.1f}" if r["cpus_cap"] else "none"
        capm = f"{r['mem_cap']/G:.1f}G" if r["mem_cap"] else "none"
        print(f"  {r['name'][:28]:<28} CPU {r['cpu_cores_now']:5.2f} ({100*r['cpu_cores_now']/ncpu:4.1f}% host, cap {capc}{' set '+r['cpuset'] if r['cpuset'] else ''})   "
              f"mem {r['mem_now']/G:5.2f}G ({100*r['mem_now']/mem_total:4.1f}% host, cap {capm}, peak {r['mem_peak']/G:.2f}G)"
              + (f"   OOM-KILLS {r['oom_kill']}" if r["oom_kill"] else "")
              + (f"   memPSI {r['mem_pressure'].get('some',0):.1f}" if r['mem_pressure'].get('some') else ""))
    now_mono = time.monotonic()
    for r in res["runs"]:
        busy = sorted(g for g, u in r["gpu_busy_until_mono"].items() if u > now_mono)
        print(f"run   {os.path.basename(r['dir'])}   {r['task']} / {r['harness']} / {r['model']}   "
              f"jobs alive {r['jobs_alive']}   GPUs busy now {busy if busy else 'none'}")
    if not res["runs"]:
        print("run   (no live run directories under %s)" % a.runs)

if __name__ == "__main__":
    main()
