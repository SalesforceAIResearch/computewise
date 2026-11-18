"""Measure each job's host terms from a finished scripted run, in gpu-tasks' expected_scores.py
--host shape, so both expected-value paths can predict real runs, not only the ideal one.

usage: python3 -I host_terms.py <run_dir> [--out host.json] [--per-job]

Output, per script (keyed by the launch's script basename, e.g. "dpo.py", or by its path in the task tree where
two scripts of the run share a basename), each a median over that script's ranks in the run:
  {"dpo.py": {"hello_s": .., "pre_s": .., "gaps": [[after_step, s], ..], "post_s": .., "n": ..}}
  hello_s  launch to the process's start: the agent's launch tool call to job_hello's t_start
  pre_s    process start to its first paced unit (imports, model load, data loading). For a
           multi-rank job this is its SLOWEST rank's figure, since the job's GPU work waits for it;
           pre_s_ranks lists every rank's, so a lagging rank (e.g. a cache miss) is visible;
           pre_s_first is the median over cold jobs and pre_s_rest over warm ones: a job is cold when no job
           of its script had reached its first paced unit by its launch (the rule the simulators apply)
  gaps     GPU-idle seconds after a step, where the job's GPU queue drained between two paced
           units by more than --gap-min-s (checkpoint saves, eval, logging that blocks)
  post_s   the last paced unit's end to job_bye (final save, exit); the slowest rank's, per job
All in wall seconds: host time is real CPU work, which the speed knob does not shorten (gpu-tasks'
host terms and expect.py read them so). The launch time comes from the transcript's tool_use
record; a job whose launch cannot be found there gets hello_s 0 and is flagged.

--per-job also writes every rank's terms, so the spread can be seen; the C1 tolerance is taken
from that spread.
"""
import argparse, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "score"))
import idle, score_run

def script_of(argv):
    """The rank's script as a normalised relative path; keys collapse to the basename below unless two scripts of the
    run share one (gpu-tasks' derive_host_terms.py rule, agreed 2026-10-20)."""
    for a in argv or []:
        if a.endswith(".py"):
            return os.path.normpath(a)
    return os.path.normpath((argv or ["?"])[0])

REGISTRY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "gpu-tasks", "registry")

def shard_steps(rec):
    """Optimizer steps (1-based) before which a modded-nanogpt job reads its next training shard, after the first: the
    loader reloads when its position plus one global batch plus one token reaches the shard's token count, and step s
    (0-based) draws the batch for optimizer step s+1. Written out from gpu-tasks' definition (common.shard_load_steps,
    2026-10-20), not imported. Empty for a record with no shard profile (TRL records)."""
    sb = ((rec.get("host") or {}).get("train_data") or {}).get("shard_bytes")
    b = (rec.get("launch") or {}).get("global_batch_tokens")
    b = b.get("value") if isinstance(b, dict) else b
    if not sb or not b:
        return set()
    n, pos, out = (int(sb) - 1024) // 2, 0, set()
    for st in range(int(rec["steps"])):
        if pos + b + 1 >= n:
            if st > 0:
                out.add(st + 1)
            pos = 0
        pos += b
    return out

def per_event(rec, u, ready):
    """In-run host time per gap class, from one rank's units u [(start, end, seq, modelled_s)]: each class's mean gap
    per occurrence, wall seconds, counting only gaps after the job's ready time. Classes (gpu-tasks' common.gap_classes,
    agreed 2026-10-20): `once` beside a one-off unit (start-up, a first step of its own length), `val_after` after a
    validation, `val_before` before one, `shard` before the optimizer step that reads the next shard, `step` otherwise.
    Units are classified by their modelled seconds: the commonest value is a step; the value that recurs as often as
    the record validates is a validation; anything else is a one-off. None when that doesn't fit the record."""
    import collections
    shards = shard_steps(rec)
    if not shards:
        return None
    vl = ((rec.get("series") or {}).get("val_loss") or {})
    n_val = len(vl.get("steps") or []) if isinstance(vl, dict) else 0
    ms = [round(float(x[3] or 0.0), 9) for x in u]
    cnt = collections.Counter(ms)
    step_v = cnt.most_common(1)[0][0]
    val_v = next((v for v, c in cnt.most_common() if v != step_v and c == n_val), None)
    if val_v is None or cnt[step_v] < int(rec["steps"]) - 1:
        return None
    kinds = ["val" if m == val_v else "step" if m == step_v else "other" for m in ms]
    tot, num = collections.Counter(), collections.Counter()
    step = 1 if kinds[0] == "step" else 0
    for i in range(1, len(u)):
        prev, cur = kinds[i - 1], kinds[i]
        if cur == "step":
            step += 1
        if prev == "other" or cur == "other":
            c = "once"
        elif prev == "val":
            c = "val_after"
        elif cur == "val":
            c = "val_before"
        elif cur == "step" and step in shards:
            c = "shard"
        else:
            c = "step"
        if u[i - 1][1] < ready - 1e-9:
            continue
        num[c] += 1
        tot[c] += max(0.0, u[i][0] - u[i - 1][1])
    return {f"{c}_s": (tot[c] / num[c] if num[c] else None) for c in ("once", "val_after", "val_before", "shard", "step")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-job", action="store_true")
    ap.add_argument("--gap-min-s", type=float, default=0.5, help="smallest GPU-idle gap to report, wall seconds")
    a = ap.parse_args()
    d = a.run_dir
    if not idle.finished(d):
        sys.exit(f"{d}: run has not finished")
    ev = score_run.load_events(os.path.join(d, "occupancy.jsonl"))
    speed = float(idle.run_inputs(d).get("speed") or next((e.get("speed") for e in ev if e["kind"] == "daemon_start"), 1.0) or 1.0)
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    lookup = {e["job"]: e for e in ev if e["kind"] == "lookup"}
    task = json.load(open(os.path.join(d, "task.json")))
    # paced units per job, on the daemon's wall clock, rebuilt the way the scorer does
    s0 = next((e for e in ev if e["kind"] == "daemon_start"), ev[0])
    off = s0["t_wall"] - s0["t_mono"]
    units, cursor = {}, {}
    for e in sorted((e for e in ev if e["kind"] == "enqueue"), key=lambda e: e["t_mono"]):
        g, t, c = e["gpu"], e["t_mono"] + off, float(e.get("wall_cost", 0.0))
        start = max(cursor.get(g, t), t); cursor[g] = start + c
        units.setdefault(e["job"], []).append((start, start + c, e.get("seq"), e.get("modelled_s")))
    # launches from the transcript: each tool_use's time, keyed by the task job its command names
    launches = {}
    for r in idle.transcript_records(d):
        if r.get("type") != "assistant":
            continue
        for b in (r.get("message") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                inp = b.get("input") or {}
                cmd, desc = inp.get("command", ""), inp.get("description", "")
                for j in task["jobs"]:
                    # the scripted agent labels each launch with its job id; else the job's isolated output dir;
                    # else its own launch line (tasks whose launch lines have no output dir, the mng3 packages)
                    if desc == j["job_id"] or f"-{j['job_id']}" in cmd or f"={j['job_id']}" in cmd or j["launch"] in cmd:
                        launches.setdefault(j["job_id"], idle.ts(r["timestamp"]))
    per, flags = [], []
    for jid, h in sorted(hello.items()):
        u = units.get(jid) or []
        if not u or jid not in bye:
            flags.append(f"daemon job {jid}: no paced units or no bye; skipped")
            continue
        tj = idle.task_job_of(h, lookup.get(jid), task["jobs"])
        start = idle.job_start(h)
        launched = launches.get(tj)
        if launched is None:
            flags.append(f"daemon job {jid} ({tj}): launch not found in the transcript; hello_s set to 0")
        gaps = []
        for i in range(1, len(u)):
            gap = u[i][0] - u[i - 1][1]
            if gap > a.gap_min_s:
                gaps.append([i, gap])
        per.append({"daemon_job": jid, "task_job": tj, "script": script_of(h.get("argv")),
                    "rank_of": lookup.get(jid, {}).get("rank"),
                    "hello_s": max(0.0, (start - launched)) if launched is not None else 0.0,
                    "launched_wall": launched, "start_wall": start,
                    "pre_s": (u[0][0] - start), "last_unit_end": u[-1][1], "bye_wall": bye[jid]["t_wall"],
                    "gaps": gaps, "units": len(u)})
    # key by basename unless two different scripts of the run share it (contra-soft-muon/ and soda-fade/ both hold
    # contra-muon-to-soft-muon.py); those keep their path, which plan.terms_of and expected_scores.py try first
    paths = {}
    for p in per:
        paths.setdefault(os.path.basename(p["script"]), set()).add(p["script"])
    for p in per:
        if len(paths[os.path.basename(p["script"])]) == 1:
            p["script"] = os.path.basename(p["script"])
    out = {}
    for sc in sorted({p["script"] for p in per}):
        ps = [p for p in per if p["script"] == sc]
        # A multi-rank job's GPU work starts only when its slowest rank is ready, and it ends when its
        # last rank says bye. So a job's pre_s and post_s are the MAX over its ranks; the script's
        # term is then the median over its task jobs. hello_s (launch to process start) is per rank,
        # and ranks start together, so it stays the median over ranks.
        by_job = {}
        for p in ps:
            by_job.setdefault(p["task_job"], []).append(p)
        job_pre = [max(r["pre_s"] for r in rs) for rs in by_job.values()]
        # a job's post_s: its last job_bye minus its last unit end, each a maximum over its ranks, the span from the
        # end of its GPU work to its release of the GPUs (gpu-tasks' definition, agreed 2026-10-20)
        job_post = [max(r["bye_wall"] for r in rs) - max(r["last_unit_end"] for r in rs) for rs in by_job.values()]
        # Cold or warm by the rule both simulators apply (plan.overlap_lanes, expect.simulate, gpu-tasks'
        # expected_scores and derive_host_terms.py, agreed 2026-10-20): a job is cold when no job of its script had
        # reached its first paced unit by its launch. Jobs of one script launched together are all cold, as packed
        # launches them. A job's launch is its launch tool call (else its first rank's start); it is ready when its
        # slowest rank reaches its first paced unit.
        launch = {tj: (min(r["launched_wall"] for r in rs) if all(r["launched_wall"] is not None for r in rs)
                       else min(r["start_wall"] for r in rs)) for tj, rs in by_job.items()}
        ready = {tj: max(r["start_wall"] + r["pre_s"] for r in rs) for tj, rs in by_job.items()}
        first_ready = min(ready.values())
        cold = [tj for tj in by_job if launch[tj] < first_ready - 1e-6]
        warm = [tj for tj in by_job if launch[tj] >= first_ready - 1e-6]
        jpre = {tj: max(r["pre_s"] for r in rs) for tj, rs in by_job.items()}
        first_tj = min(launch, key=launch.get) if launch else None
        # a cold/warm split needs at least two jobs of the script (gpu-tasks' derive_host_terms.py does the same)
        two = len(by_job) >= 2
        first_pre = statistics.median(jpre[tj] for tj in cold) if (cold and two) else None
        rest_pre = [jpre[tj] for tj in warm] if two else []
        med = lambda k: statistics.median(p[k] for p in ps)
        # a gap after a given step is kept when most ranks show it; its size is the median
        steps = {}
        for p in ps:
            for st, g in p["gaps"]:
                steps.setdefault(st, []).append(g)
        pes = []
        for tj, rs in by_job.items():
            slow = max(rs, key=lambda r: r["last_unit_end"])
            rid = (lookup.get(slow["daemon_job"]) or {}).get("record_id")
            rp = os.path.join(REGISTRY, "records", f"{rid}.json") if rid else None
            if rp and os.path.exists(rp):
                pe = per_event(json.load(open(rp)), units[slow["daemon_job"]], max(r["start_wall"] + r["pre_s"] for r in rs))
                if pe:
                    pes.append(pe)
        pe_med = ({k: (statistics.median(x[k] for x in pes if x[k] is not None) if any(x[k] is not None for x in pes) else None)
                   for k in ("once_s", "val_after_s", "val_before_s", "shard_s", "step_s")} if pes else None)
        out[sc] = {"hello_s": med("hello_s"), "pre_s": statistics.median(job_pre), "post_s": statistics.median(job_post),
                   **({"per_event": pe_med} if pe_med else {}),
                   "pre_s_first": first_pre, "pre_s_rest": statistics.median(rest_pre) if rest_pre else None,
                   "first_job": first_tj, "cold_jobs": len(cold) if two else None, "warm_jobs": len(warm) if two else None,
                   "pre_s_ranks": sorted(round(p["pre_s"], 1) for p in ps),
                   "ranks_per_job": statistics.median(len(rs) for rs in by_job.values()),
                   "gaps": [[st, statistics.median(gs)] for st, gs in sorted(steps.items()) if len(gs) * 2 > len(ps)],
                   "n": len(ps),
                   "spread_s": {"hello_s": max(p["hello_s"] for p in ps) - min(p["hello_s"] for p in ps),
                                "pre_s": max(p["pre_s"] for p in ps) - min(p["pre_s"] for p in ps),
                                "post_s": max(job_post) - min(job_post)}}
    res = {"run_dir": os.path.abspath(d), "speed": speed, "terms": out, "flags": flags}
    if a.per_job:
        res["per_job"] = per
    s = json.dumps(res, indent=1)
    if a.out:
        open(a.out, "w").write(s + "\n")
    print(s)

if __name__ == "__main__":
    main()
