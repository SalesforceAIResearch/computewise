"""Counterfactual for Fix B on a collected run.
Each job's per_example units are replayed on its GPU twice: under the daemon's
whole-unit FIFO cursor (what 3ec0 does) and under an alternative sharing law:
processor sharing by default (each job with a unit in service drains at 1/N of
the GPU, which is Fix B), or with --retained R a retained-rate law (each of N >= 2
sharers keeps R of its solo rate; 2026-11-04, from the H200 pair sp1). Held fixed
from the log: every job's first request time (the agent's launch), each unit's
GPU cost (wall_cost), each host gap, each pause the job's process took beyond
its GPU wait (a stopped or descheduled process), and the number of units the
job ran (a killed job stops at the same unit). The FIFO replay must reproduce
the log; that is the check that the model is the daemon's."""
import json, sys, collections

def load(run):
    t0 = None; ev = []; bye = {}
    for l in open(f"{run}/occupancy.jsonl"):
        if '"daemon_start"' in l and t0 is None: t0 = json.loads(l)["t_mono"]
        elif '"enqueue"' in l:
            d = json.loads(l)
            if d.get("kind") == "enqueue" and d.get("unit") == "example":
                ev.append((d["t_mono"], d["job"], d["gpu"], float(d.get("wall_cost") or 0.0), float(d.get("host_gap_s") or 0.0)))
        elif '"job_bye"' in l:
            d = json.loads(l); bye[d["job"]] = d["t_mono"]
    ev.sort()
    jobs = collections.defaultdict(lambda: {"r": [], "c": [], "h": [], "cur": []})
    cursor = collections.defaultdict(lambda: -1e18)
    for t, j, g, c, h in ev:        # the daemon's own cursor, open loop on the logged requests
        base = max(cursor[g], t - h); cursor[g] = base + c
        J = jobs[j]; J["gpu"] = g; J["r"].append(t); J["c"].append(c); J["h"].append(h); J["cur"].append(cursor[g])
    for j, J in jobs.items():
        n = len(J["r"]); J["bye"] = bye.get(j)
        # observed drain of unit k: the next request less its host gap; the last unit: its cursor
        J["d_obs"] = [J["r"][k + 1] - J["h"][k + 1] for k in range(n - 1)] + [max(J["cur"][-1], J["r"][-1])]
        J["pause"] = [max(0.0, J["d_obs"][k] - max(J["cur"][k], J["r"][k])) for k in range(n)]
    return t0, dict(jobs)

def share_rate(n, mode, retained=None, cap=None):
    """Each serving job's drain rate (fraction of its solo rate) with n jobs in service.
    ps: processor sharing, 1/n (the GPU's throughput is fixed and divided).
    rr: a retained-rate law: each of n >= 2 sharers keeps
        `retained` of its solo rate, optionally capped at cap/n (total throughput
        cap x one job's). Measured only at n = 2: two batch-1 KV decodes on one
        H200 kept 0.943 and 0.938 of solo (pair sp1, the study
        2026-11-04); n > 2 is an extrapolation, and the cap is the knob for it."""
    if n <= 1:
        return 1.0
    if mode == "rr":
        r = float(retained)
        return min(r, float(cap) / n) if cap else r
    return 1.0 / n

def simulate(jobs, mode, retained=None, cap=None):
    """Closed loop. Unit 0 activates at its logged request; unit k+1 at
    drain_k + pause_k, requested h_{k+1} later; drain_k = max(service end, request)."""
    out = {}
    by_gpu = collections.defaultdict(list)
    for j, J in jobs.items(): by_gpu[J["gpu"]].append(j)
    for g, js in by_gpu.items():
        st = {j: {"k": 0, "act": jobs[j]["r"][0], "req": jobs[j]["r"][0], "rem": jobs[j]["c"][0], "end": None, "d": []} for j in js}
        if mode == "fifo":
            cursor = -1e18
            while True:
                live = [j for j in js if st[j]["k"] < len(jobs[j]["r"])]
                if not live: break
                j = min(live, key=lambda x: st[x]["req"])     # the daemon serves in request order
                S = st[j]; J = jobs[j]; k = S["k"]
                start = max(cursor, S["act"]); end = start + J["c"][k]; cursor = end
                dk = max(end, S["req"]); S["d"].append(dk); S["k"] += 1
                if S["k"] < len(J["r"]):
                    S["act"] = dk + J["pause"][k]; S["req"] = S["act"] + J["h"][S["k"]]
        else:
            t = min(S["act"] for S in st.values())
            while True:
                live = [j for j in js if st[j]["k"] < len(jobs[j]["r"])]
                if not live: break
                serving = [j for j in live if st[j]["end"] is None and st[j]["act"] <= t + 1e-9]
                n = len(serving)
                rate = share_rate(n, mode, retained, cap)
                nxt = []
                if n: nxt.append(t + min(st[j]["rem"] for j in serving) / rate)
                nxt += [st[j]["act"] for j in live if st[j]["end"] is None and st[j]["act"] > t + 1e-9]
                nxt += [st[j]["req"] for j in live if st[j]["end"] is not None]
                tn = max(t, min(nxt))
                for j in serving: st[j]["rem"] -= (tn - t) * rate
                t = tn
                for j in live:
                    S = st[j]; J = jobs[j]
                    if S["end"] is None and S["act"] <= t + 1e-9 and S["rem"] <= 1e-9:
                        S["end"] = t
                    if S["end"] is not None and S["req"] <= t + 1e-9:
                        k = S["k"]; dk = max(S["end"], S["req"]); S["d"].append(dk); S["k"] += 1; S["end"] = None
                        if S["k"] < len(J["r"]):
                            S["act"] = dk + J["pause"][k]; S["req"] = S["act"] + J["h"][S["k"]]; S["rem"] = J["c"][S["k"]]
        for j in js: out[j] = st[j]["d"]
    return out

def anchor_pauses(jobs, mode, min_pause=5.0, rounds=6, retained=None, cap=None):
    """A pause the agent imposed (SIGSTOP, a dispatcher holding a job) ends at the
    wall time it ended in the log, not after a fixed length: re-solve each pause
    longer than min_pause so the unit after it resumes at its logged time."""
    import copy
    J2 = copy.deepcopy(jobs)
    for _ in range(rounds):
        sim = simulate(J2, mode, retained, cap)
        for j, J in J2.items():
            for k in range(len(J["r"]) - 1):
                if jobs[j]["pause"][k] > min_pause:
                    J["pause"][k] = max(0.0, jobs[j]["d_obs"][k] - sim[j][k])
    return simulate(J2, mode, retained, cap)

def dispatch_pauses(jobs, mode, min_pause=5.0, rounds=10, retained=None, cap=None):
    """A pause the agent's own dispatcher imposed and lifted when another job on
    the same GPU ended (G-kvreph-astra-L0's review/gpu_dispatch.py: one active
    evaluator per GPU, the rest SIGSTOPped, resumed as a slot frees): each such
    pause ends when its trigger, the job on that GPU whose logged end most
    closely precedes the logged resume, ends in the replay, plus the logged lag."""
    import copy
    J2 = copy.deepcopy(jobs)
    trig = {}
    for j, J in jobs.items():
        for k in range(len(J["r"]) - 1):
            if J["pause"][k] > min_pause:
                res = J["d_obs"][k]
                cands = [(res - jobs[t]["d_obs"][-1], t) for t in jobs if t != j and jobs[t]["gpu"] == J["gpu"] and 0 <= res - jobs[t]["d_obs"][-1] < 30]
                if cands:
                    lag, t = min(cands); trig[(j, k)] = (t, lag)
    for _ in range(rounds):
        sim = simulate(J2, mode, retained, cap)
        for (j, k), (t, lag) in trig.items():
            J2[j]["pause"][k] = max(0.0, sim[t][-1] + lag - sim[j][k])
    n_paused = sum(1 for J in jobs.values() for k in range(len(J["r"]) - 1) if J["pause"][k] > min_pause)
    print(f"   agent pauses: {n_paused}, {len(trig)} tied to a same-GPU job's end within 30 s")
    return simulate(J2, mode, retained, cap)

def main(run, verbose=True, anchored=False, dispatch=False, retained=None, cap=None):
    t0, jobs = load(run)
    law = "rr" if retained is not None else "ps"
    if dispatch:
        f = dispatch_pauses(jobs, "fifo"); p = dispatch_pauses(jobs, law, retained=retained, cap=cap)
    elif anchored:
        f = anchor_pauses(jobs, "fifo"); p = anchor_pauses(jobs, law, retained=retained, cap=cap)
    else:
        f = simulate(jobs, "fifo"); p = simulate(jobs, law, retained, cap)
    # a unit the agent paused after its drain: the log's drain includes the pause, the replay's does not
    err = sorted(abs(f[j][k] - jobs[j]["d_obs"][k]) for j in jobs for k in range(len(f[j])) if not (k < len(jobs[j]["pause"]) - 1 and jobs[j]["pause"][k] > 5.0))
    first = min(J["r"][0] for J in jobs.values())
    fe = max(v[-1] for v in f.values()); pe = max(v[-1] for v in p.values()); oe = max(J["d_obs"][-1] for J in jobs.values())
    rows = []
    for j, J in jobs.items():
        df, dp = f[j][-1] - J["r"][0], p[j][-1] - J["r"][0]
        rows.append((j, J["gpu"], len(J["r"]), round(J["c"][0], 3), round(df), round(dp), round((dp - df) / max(1.0, df) * 100, 1)))
    moved10 = sum(1 for r in rows if abs(r[6]) > 10); moved25 = sum(1 for r in rows if abs(r[6]) > 25)
    sum_f = sum(r[4] for r in rows); sum_p = sum(r[5] for r in rows)
    how = " (agent pauses lifted when the same-GPU job ahead ends)" if dispatch else (" (agent pauses anchored to their logged ends)" if anchored else "")
    alt = "fair share" if law == "ps" else (f"retained {retained}" + (f" cap {cap}" if cap else ""))
    print(f"== {run}{how}: {len(jobs)} per_example jobs, {sum(len(J['r']) for J in jobs.values())} units")
    print(f"   FIFO replay vs the log, every drain: median {err[len(err)//2]:.2f} s, p99 {err[int(len(err)*.99)]:.2f} s, max {err[-1]:.1f} s")
    print(f"   work end from the first launch: log {oe - first:.0f} s, FIFO {fe - first:.0f} s, {alt} {pe - first:.0f} s ({(pe - fe) / (fe - first) * 100:+.2f}%)")
    print(f"   sum of job durations: FIFO {sum_f:.0f} s, {alt} {sum_p:.0f} s ({(sum_p - sum_f) / sum_f * 100:+.1f}%)")
    print(f"   jobs whose own duration moves >10%: {moved10} of {len(rows)}, >25%: {moved25}")
    if verbose:
        for r in sorted(rows, key=lambda r: -abs(r[5] - r[4]))[:5]:
            print(f"     job {r[0]} gpu {r[1]}: {r[2]} units of {r[3]} s, FIFO {r[4]} s -> {alt} {r[5]} s ({r[6]:+}%)")
    return {"jobs": len(rows), "moved10": moved10, "moved25": moved25, "end_fifo": fe - first, "end_ps": pe - first,
            "err_p99": err[int(len(err)*.99)], "law": law, "retained": retained, "cap": cap,
            "per_job": [{"job": r[0], "gpu": r[1], "units": r[2], "fifo_s": r[4], "alt_s": r[5], "delta_pct": r[6]} for r in rows]}

if __name__ == "__main__":
    # usage: python3 fairshare_counterfactual.py [--anchor-pauses | --dispatch-pauses]
    #            [--retained R [--cap C]] [--json OUT] <run dir>...
    # Default law: processor sharing (1/n per sharer). --retained R: each of n >= 2
    # sharers keeps R of its solo rate (sp1: 0.94), optionally capped at C/n.
    def opt(name, cast):
        if name in sys.argv:
            i = sys.argv.index(name); return cast(sys.argv[i + 1])
        return None
    retained, cap, out = opt("--retained", float), opt("--cap", float), opt("--json", str)
    skip = {sys.argv.index(n) + 1 for n in ("--retained", "--cap", "--json") if n in sys.argv}
    args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and i not in skip]
    res = {}
    for run in args:
        res[run] = main(run, anchored="--anchor-pauses" in sys.argv, dispatch="--dispatch-pauses" in sys.argv,
                        retained=retained, cap=cap)
    if out:
        json.dump(res, open(out, "w"), indent=1)
