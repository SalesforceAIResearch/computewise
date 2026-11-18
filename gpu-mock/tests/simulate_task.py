#!/usr/bin/env python3
"""Dry-run a task file through the daemon's registry: what an agent's launches
would draw, with the task loaded, before any pilot burns a machine.

For every job it plays an agent that launches the job's script with its own
seeds: the first recorded seeds, an unrecorded one, a repeat, and (open pools)
on until the pool is spent; then one launch of an EDITED script (other entry
hash, same path). Reports each draw's match/draw labels, the record, whether a
serving unit repeats before the pool is spent, and the task_job attribution.
    python3 -I tests/simulate_task.py <registry dir> <task.json> [--max-launches N]
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "daemon"))
import gpu_daemon as gd  # noqa: E402

reg_dir, task_path = Path(sys.argv[1]), Path(sys.argv[2])
max_launches = int(sys.argv[sys.argv.index("--max-launches") + 1]) if "--max-launches" in sys.argv else 40
task = json.loads(task_path.read_text())
reg = gd.Registry(reg_dir / "records" if (reg_dir / "records").is_dir() else reg_dir)
n_sib = reg.load_task(task_path)
print(f"task {task.get('task_id')}: {len(task.get('jobs') or [])} jobs, {n_sib} sibling lists, {len(reg._by_id)} records in the registry")
problems = 0
for job in task.get("jobs") or []:
    jid = job.get("job_id"); recs = job.get("records") or []
    sp = job.get("script_path") or ""
    cands = [reg._by_id.get(r) for r in recs]
    missing = [r for r, c in zip(recs, cands) if c is None]
    cands = [c for c in cands if c]
    not_ready = [reg._rid(c) for c in cands if not (c.get("readiness") or {}).get("replay", True)]
    eh = {c["identity"].get("entry_hash") for c in cands}
    seeds = sorted({reg._seed(c) for c in cands if reg._seed(c) is not None})
    print(f"\n== job {jid}: draw={job.get('draw')} instances={json.dumps(job.get('instances', {}).get('max'))} script={sp}")
    print(f"   {len(recs)} listed, {len(missing)} missing, {len(not_ready)} not replay-ready, entry hashes {len(eh)}, recorded seeds {seeds[:12]}{'…' if len(seeds) > 12 else ''}")
    if missing: problems += 1; print(f"   MISSING: {missing[:5]}")
    if not_ready: print(f"   not ready: {not_ready[:5]}")
    if not cands or len(eh) != 1:
        pb = sum(1 for c in cands if c["identity"].get("match_policy") == "path_bound")
        if len(eh) > 1: print(f"   {len(eh)} entry hashes among the listed records ({pb} path-bound): an unedited launch matches its hash exactly, an edited one matches by path")
        if not cands: continue
    entry = next(iter({c["identity"].get("entry_hash") for c in cands if c["identity"].get("match_policy") != "path_bound"} or eh))
    script = Path(sp).name if sp else "train.py"
    script_abs = f"/home/researcher/project/{sp}" if sp else None
    # the agent's launches: recorded seeds low to high, an unrecorded one, a repeat, then on until spent
    agent_seeds = [s for s in (0, 1, 2) ] + [42, 0] + [s for s in seeds if s not in (0, 1, 2)]
    served_units, rows, launched = [], [], 0
    for k, seed in enumerate(agent_seeds[:max_launches]):
        argv = [script, "--seed", str(seed)]
        rec, mode, info = reg.lookup(None, entry, argv, job_key=f"{jid}-L{k}", stage="exact", script_path=script_abs)
        rid = reg._rid(rec) if rec else None
        unit = reg._unit(rec) if rec else None
        repeat = unit in served_units and info.get("draw") not in ("recycled", "reused", "sibling")
        if repeat: problems += 1
        served_units.append(unit)
        rows.append((seed, mode, info.get("match"), info.get("draw"), rid, info.get("task_job"), "REPEAT!" if repeat else ""))
        launched += 1
        if info.get("draw") in ("recycled", "reused") and k >= 5:
            break
    for r in rows:
        print("   seed %-3s %-6s %-20s %-18s %-70s job=%-18s %s" % r)
    spent = sum(1 for r in rows if r[3] in ("recycled", "reused"))
    print(f"   {launched} launches, {len(set(u for u in served_units if u))} distinct units, {spent} after the pool was spent; task_job attributed on {sum(1 for r in rows if r[5] == jid)}/{launched}")
    # edited scripts: another entry hash, the same path; the units they draw must
    # not repeat the unedited launches' units while the pool has unused ones
    for k, seed in enumerate((7, 0, 99)):
        rec, mode, info = reg.lookup(None, "e" * 64, [script, "--seed", str(seed)], job_key=f"{jid}-edited{k}", stage="exact", script_path=script_abs)
        unit = reg._unit(rec) if rec else None
        repeat = unit in served_units and info.get("draw") not in ("recycled", "reused", "sibling")
        if repeat: problems += 1
        served_units.append(unit)
        print(f"   edited script seed {seed}: mode={mode} match={info.get('match')} draw={info.get('draw')} record={reg._rid(rec) if rec else None} unit={unit} {'REPEAT!' if repeat else ''}")
        if mode == "model":
            print("   -> an edited candidate script has no record: modelled path (real CPU compute)")
print(f"\n{'OK' if problems == 0 else 'PROBLEMS: %d' % problems}")
sys.exit(1 if problems else 0)
