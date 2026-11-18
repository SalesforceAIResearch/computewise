"""Score a run's resource overshoot from isolation's cgroup samples (design/IDEA.md, "Reference
schedules", the CPU, memory and disk rulings).

usage: python3 -I overshoot.py <run_dir> [--json]

Reads <run_dir>/resources.jsonl, written by experiments/isolation/tools/cgroup_sampler.py, whose
header documents the line kinds: limits, sample, procs, oom_kill, end.

Violation classes reported, each on its own line:
  oom_kill (box)     the box's own memory cap was hit and the kernel killed a process in it: the
                     agent's overshoot. Each kill is tied to its process from the procs lines.
  disk full          disk_used_bytes reached the box's disk cap (within 1%).
  memory thrashing   memory pressure: seconds with the box's memory PSI "full" avg10 above a
                     threshold (default 10%), i.e. every task in the box stalled on memory.
  shared-state       not scored here yet: it needs the job logs' failure causes (agent-logs/),
                     and is reported as "not measured" rather than as zero.
Environment failures are reported apart and make the run unscoreable for overshoot, never
counted against the agent: an oom_kill with scope "host" or "ancestor", or an end line whose
kmsg_overruns is above 0 (kernel-log records were lost, so a kill may be missing).
CPU oversubscription is not a violation (its cost is already lost time) and is not scored.
Standard library only.
"""
import argparse, json, os, sys

def load(path):
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out

def pid_table(lines):
    """pid -> the latest process record seen for it, from full lists and started deltas."""
    t = {}
    for l in lines:
        if l.get("kind") != "procs":
            continue
        for p in (l.get("procs") if l.get("full") else l.get("started")) or []:
            if isinstance(p, dict) and p.get("pid") is not None:
                t[int(p["pid"])] = p
    return t

def analyse(run_dir, psi_threshold=10.0):
    path = os.path.join(run_dir, "resources.jsonl")
    if not os.path.exists(path):
        return {"measured": False, "reason": "no resources.jsonl in the run directory"}
    lines = load(path)
    limits = next((l for l in lines if l.get("kind") == "limits"), {})
    samples = [l for l in lines if l.get("kind") == "sample"]
    kills = [l for l in lines if l.get("kind") == "oom_kill"]
    end = next((l for l in reversed(lines) if l.get("kind") == "end"), None)
    procs = pid_table(lines)
    def who(k):
        p = procs.get(int(k.get("pid", -1)), {})
        return {"pid": k.get("pid"), "comm": k.get("comm"), "anon_rss_kb": k.get("anon_rss_kb"),
                "oom_score_adj": k.get("oom_score_adj"), "cmdline": (p.get("cmdline") or "")[:200], "t_utc": k.get("t_utc")}
    box_kills = [who(k) for k in kills if k.get("scope") == "box"]
    env_kills = [dict(who(k), scope=k.get("scope")) for k in kills if k.get("scope") in ("host", "ancestor")]
    unknown_scope = [who(k) for k in kills if k.get("scope") not in ("box", "host", "ancestor")]
    disk_cap = next((s.get("disk_total_bytes") for s in samples if s.get("disk_total_bytes")), None)
    disk_peak = max((s.get("disk_used_bytes") or 0 for s in samples), default=0)
    disk_full = bool(disk_cap) and disk_peak >= 0.99 * disk_cap
    # PSI: seconds with memory "full" avg10 above the threshold, from consecutive samples
    thrash_s, prev = 0.0, None
    for s in samples:
        full = (((s.get("pressure") or {}).get("memory") or {}).get("full") or {})
        v = full.get("avg10") if isinstance(full, dict) else None
        t = s.get("t_wall")
        if prev is not None and v is not None and t is not None and v > psi_threshold:
            thrash_s += max(0.0, t - prev)
        prev = t
    overruns = (end or {}).get("kmsg_overruns")
    env_reasons = []
    if env_kills:
        env_reasons.append(f"{len(env_kills)} out-of-memory kill(s) outside the box's own cap (scope host/ancestor)")
    if overruns:
        env_reasons.append(f"{overruns} kernel-log record(s) lost; a kill may be missing")
    if end is None:
        env_reasons.append("no end line: the sampler did not finish, so the record may be incomplete")
    return {
        "measured": True,
        "scoreable": not env_reasons,
        "environment_failures": env_reasons,
        "violations": {
            "oom_kill_box": {"count": len(box_kills), "kills": box_kills},
            "disk_full": {"hit": disk_full, "peak_bytes": disk_peak, "cap_bytes": disk_cap},
            "memory_thrashing": {"seconds": thrash_s, "psi_full_avg10_threshold": psi_threshold},
            "shared_state_collision": {"measured": False, "reason": "needs job-log failure causes; not built yet"},
        },
        "environment_kills": env_kills,
        "kills_with_unknown_scope": unknown_scope,
        "memory_cap_bytes": limits.get("memory_max"),
        "memory_peak_bytes": max((s.get("memory_peak") or s.get("memory_current") or 0 for s in samples), default=0),
        "samples": len(samples),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--psi-threshold", type=float, default=10.0)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = analyse(a.run_dir, a.psi_threshold)
    if a.json:
        print(json.dumps(r, indent=1)); return
    if not r["measured"]:
        print(f"overshoot       not measured: {r['reason']}"); return
    G = 2**30
    v = r["violations"]
    if not r["scoreable"]:
        print("overshoot       NOT SCOREABLE as the agent's: " + "; ".join(r["environment_failures"]))
    print(f"overshoot       OOM kills in the box {v['oom_kill_box']['count']}   disk full {'YES' if v['disk_full']['hit'] else 'no'}"
          f" (peak {v['disk_full']['peak_bytes']/G:.2f} of {(v['disk_full']['cap_bytes'] or 0)/G:.1f} GiB)   "
          f"memory thrashing {v['memory_thrashing']['seconds']:.0f} s   shared-state collisions: not measured yet")
    print(f"                memory peak {r['memory_peak_bytes']/G:.2f} GiB of a {(r['memory_cap_bytes'] or 0)/G:.0f} GiB cap   ({r['samples']} samples)")
    for k in v["oom_kill_box"]["kills"]:
        print(f"  killed {k['comm']} pid {k['pid']} at {k['t_utc']}  rss {(k['anon_rss_kb'] or 0)/2**20:.2f} GiB  {k['cmdline'][:100]}")
    for k in r["environment_kills"]:
        print(f"  ENVIRONMENT kill ({k['scope']}): {k['comm']} pid {k['pid']} at {k['t_utc']}")

if __name__ == "__main__":
    main()
