"""Sibling runs for gpu-mock's rerun design (b): for each job of each task, the other recorded runs of the same setup that could
be served when the job is run again or its script departs from the record, without serving any run, or any part of one, that
another job of the task may be served (gpu-tasks, 2026-10-21; the study's question to gpu-mock, 07:15 UTC).

usage: python3 -I build/siblings.py <registry> [--out FILE] [--tasks a,b,...]

Design (b), as the study put it: when the daemon's fingerprint of an agent's script departs from its record (or the agent
reruns a job), serve a different record of the same setup, so that changed code gives changed values without inventing data.
This script derives what that would draw from. It changes no record and no task.

Per job:
- exact_seed jobs (a prefix, prepared, data-checked or full run of one seed): the siblings are the other recorded runs of the
  same script at the same stop step, as records of the job's own kind where they exist (a `-prepq` job gets `-prepq` records,
  a `-dcheck` job `-dcheck` ones), else the prefix records themselves. A run is left out if its served unit (common.served_unit)
  overlaps that of any record any job of the task lists, so no sibling repeats a value the task can show elsewhere (the
  `model_runs` rule). The remaining pool of a group is dealt round-robin to the task's jobs of that group, in job order, so no
  two jobs share a sibling.
- without_replacement jobs (an unseeded script): the job already lists its group's whole pool and the daemon draws one; a
  rerun can draw another from the same list. Their count is the pool less one.
- segments: none. A resumed segment loads its predecessor's checkpoint, so a sibling for one segment would need the whole chain
  served from the sibling run; that is a chain-level design question, not a list.

Per setup it also reports the spread of the served validation loss at the stop step across the pool (`spread`), and, where
every record is a rerun of identical code, that same spread as `rerun_spread`: newton-muon's one script fixes no seed, so its
15 recorded runs are reruns of identical code with fresh randomness, and their spread at step 250 is what a real rerun of
identical code varies by on those GPUs. (An earlier version called these "reruns of one effective seed" because all 15 start
at one step-0 loss; that loss is ln(vocab), the value of a zeroed vocab head under any seed, so it was the right set for the
wrong reason.)
"""
import argparse
import collections
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FAMILY_SUFFIXES = ("-dcheck", "-prepqseed", "-prepq", "-prepseed", "-prep")


def family(rid):
    """The record kind a job's record belongs to, as its id's suffix ('' for a plain prefix or full record)."""
    return next((s for s in FAMILY_SUFFIXES if rid.endswith(s)), "")


def base_unit(r, recs):
    """The prefix (or full) record behind a derived record: the run and stop step the sibling pool is drawn from."""
    rid = r["identity"]["record_id"]
    fam = family(rid)
    return recs.get(rid[:len(rid) - len(fam)] if fam else rid)


def pool_key(r):
    """Records that are runs of one script to one stop step: their parent group and stop step (a prefix), else their own
    replicate group (a full run)."""
    d = r["identity"].get("derived_from") or {}
    if d.get("kind") == "prefix":
        return ("prefix", d["parent_replicate_group"], d["stop_after_step"])
    return ("full", r["identity"]["replicate_group"], r.get("steps"))


def final_val(r):
    v = r["series"].get("val_loss") or {}
    return v["values"][-1] if v.get("values") else None


def sibling_pools(recs):
    """Every replay-ready modded-nanogpt prefix or full record, by pool_key: what siblings are drawn from."""
    pools = collections.defaultdict(list)
    for r in recs.values():
        d = r["identity"].get("derived_from") or {}
        if (d.get("kind") in (None, "prefix") and r["readiness"]["replay"] and r["identity"]["record_id"].startswith("mng3-")
                and not family(r["identity"]["record_id"])):
            pools[pool_key(r)].append(r)
    return pools


def parent_of(r):
    """The recorded run a prefix or full record serves: its parent's id, or its own."""
    d = r["identity"].get("derived_from") or {}
    return d.get("record") or r["identity"]["record_id"]


def task_siblings(task, recs, pools):
    """job_id -> its sibling record ids in serving order, for the task's exact-seed modded-nanogpt jobs that are not segments
    (the rule in this module's docstring), and job_id -> how. Jobs without a list are absent from the first map.

    Jobs whose pools draw on one script's recorded runs (a P20-tail's contra prefixes and its long contra job, say) share those
    runs: the free runs (no record any job of the task lists comes from them) are dealt round-robin over all such jobs, and
    each job gets the run as a record of its own pool (its stop step, its kind)."""
    used = {C.served_unit(recs[x])[0] for j in task["jobs"] for x in (j.get("records") or []) if x in recs}
    fams, out, how = collections.OrderedDict(), {}, {}
    for j in task["jobs"]:
        rs = [recs[x] for x in (j.get("records") or []) if x in recs]
        if (j.get("draw") != "exact_seed" or not rs or not rs[0]["identity"]["record_id"].startswith("mng3-")
                or any((x["identity"].get("derived_from") or {}).get("kind") == "segment" for x in rs)):
            continue
        b = base_unit(rs[0], recs)
        if b is None:
            continue
        key = pool_key(b)
        by_parent = {parent_of(r): r for r in pools[key]}
        script_group = recs[parent_of(b)]["identity"]["replicate_group"]
        fams.setdefault(script_group, []).append((j["job_id"], family(rs[0]["identity"]["record_id"]), by_parent))
        out[j["job_id"]] = []
    for g, members in fams.items():
        runs = sorted({p for _, _, bp in members for p in bp} - used)
        i = 0
        for p in runs:
            for _ in range(len(members)):            # the next job in turn that has this run at its own stop step
                jid, fam, bp = members[i % len(members)]
                i += 1
                if p in bp:
                    rid = bp[p]["identity"]["record_id"] + fam
                    out[jid].append(rid if rid in recs else bp[p]["identity"]["record_id"])
                    break
        for jid, _, bp in members:
            how[jid] = (f"{len(out[jid])} other runs of the same script at the job's stop step, none from a run any job of "
                        f"the task lists; {len(runs)} such runs free of {len({p for _, _, b in members for p in b})}, dealt "
                        f"to {len(members)} job(s)")
    return out, how


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("registry")
    ap.add_argument("--out")
    ap.add_argument("--tasks")
    a = ap.parse_args()
    recs = C.load_records(os.path.join(a.registry, "records"))
    pools = sibling_pools(recs)
    tdir = os.path.join(a.registry, "tasks")
    names = a.tasks.split(",") if a.tasks else sorted(f[:-5] for f in os.listdir(tdir) if f.endswith(".json"))
    out = {"schema": "p006-gpu-siblings/0.1", "derived_by": "build/siblings.py (gpu-tasks)", "tasks": {}, "setups": {}}
    used_keys = set()
    for tid in names:
        task = json.load(open(os.path.join(tdir, tid + ".json")))
        if not any(x.startswith("mng3-") for j in task["jobs"] for x in j.get("records") or []):
            continue
        sib, how = task_siblings(task, recs, pools)
        rows = []
        for j in task["jobs"]:
            rs = [recs[x] for x in (j.get("records") or []) if x in recs]
            row = {"job_id": j["job_id"], "draw": j.get("draw"), "records": j.get("records") or [],
                   "siblings": sib.get(j["job_id"], [])}
            if j["job_id"] in sib:
                row["how"] = how[j["job_id"]]
                used_keys.add(pool_key(base_unit(rs[0], recs)))
            elif any((x["identity"].get("derived_from") or {}).get("kind") == "segment" for x in rs):
                row["how"] = "a segment: a sibling needs the whole chain from the sibling run"
            elif j.get("draw") == "without_replacement":
                row.update({"own_pool_rest": max(len(rs) - 1, 0),
                            "how": "unseeded: the job lists its pool, and a rerun can draw another from it"})
            else:
                row["how"] = "no sibling list: not an exact-seed modded-nanogpt job"
            rows.append(row)
        out["tasks"][tid] = {"jobs": rows, "min_siblings_exact_seed": min((len(v) for v in sib.values()), default=None)}
    for key in sorted(used_keys | {k for k in pools if k[0] == "prefix"}, key=str):
        rs = pools[key]
        vals = [v for v in (final_val(r) for r in rs) if v is not None]
        seeds = {r["identity"]["seed"]["value"] for r in rs}
        # Reruns of identical code: one script that fixes no seed, so every run draws fresh randomness. Not "one identical
        # first validation loss": a script that zeroes its vocab head starts every run at ln(vocab) = 10.82584 whatever its
        # seed (newton-muon does; muonh-tuned, also unseeded, does not and starts at 10.95-11.07), so that test proves a
        # zeroed head, not a shared seed.
        rerun = seeds == {None} and len(rs) > 1 and len({r["identity"]["entry_hash"] for r in rs}) == 1
        out["setups"]["|".join(map(str, key))] = {
            "records": len(rs), "seeds": sorted(s for s in seeds if s is not None), "unseeded": None in seeds,
            "final_val_min": min(vals) if vals else None, "final_val_max": max(vals) if vals else None,
            "spread": round(max(vals) - min(vals), 4) if len(vals) > 1 else None,
            "stdev": round(statistics.stdev(vals), 4) if len(vals) > 2 else None,
            "rerun_spread": round(max(vals) - min(vals), 4) if rerun else None,
            "rerun_how": ("one script's bytes, fixing no seed: every record is a rerun of identical code with fresh "
                          "randomness, so the spread is run-to-run variation" if rerun else None),
        }
    text = json.dumps(out, indent=1) + "\n"
    if a.out:
        C.write_text(a.out, text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
