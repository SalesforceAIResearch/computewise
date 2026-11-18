"""Precision-limited replication from recorded H100 nanogpt prefixes.

Owned by the study/K2. Integration belongs to gpu-tasks. No registry is
changed by --check. Normal invocation derives records/artifacts only; build()
returns the task for gen_tasks integration. Scientific scoring/planning needs
the precision field; neither readiness nor staging is asserted here.
"""
import copy
import json
import math
import statistics
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gen_prefix as P
import single_process as SP

TASK_ID = "pkg-k2-precision"
ROOT = "precision"
STEP = 1500
SPECS = {
    "contra-soft-muon": ("mng3-cfg-d649f96e97a1", 30),
    "radial-damp": ("mng3-cfg-f04ccdc1544e", 9),
    "muonh-tuned": ("mng3-cfg-d7542e12eaf3", 10),
}
PROCEDURE = {
    "model": "independent identically distributed Gaussian replication outcomes within each optimizer",
    "familywise_alpha": 0.05,
    "looks": [4, 8, "all_available"],
    "interval": "two-sided Student-t, sample standard deviation, 1-0.05/9 per arm/look",
    "decision": "lowest-mean arm resolves only when its upper endpoint is below every other arm's lower endpoint",
    "file_order": "ascending actual seed where recorded; unseeded muonh by ascending source record id",
    "global_order": "replication-major: round-robin across arms, preserving each arm's own seed or unseeded record order",
    "exhaustion": "overlapping intervals at final look mean unresolved, not proof of equivalence",
}
PROCEDURE["per_interval_alpha"] = PROCEDURE["familywise_alpha"] / (len(SPECS) * len(PROCEDURE["looks"]))
PROCEDURE["critical_probability"] = 1 - PROCEDURE["per_interval_alpha"] / 2


def beta_fraction(a, b, x):
    tiny, eps = 1e-300, 2e-14
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 401):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d, c = (d if abs(d) > tiny else tiny), (c if abs(c) > tiny else tiny)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d, c = (d if abs(d) > tiny else tiny), (c if abs(c) > tiny else tiny)
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * beta_fraction(a, b, x) / a
    return 1 - factor * beta_fraction(b, a, 1 - x) / b


def student_t_cdf(t, degrees):
    if t < 0:
        return 1 - student_t_cdf(-t, degrees)
    return 1 - 0.5 * regularized_beta(degrees / (degrees + t * t), degrees / 2, 0.5)


def student_t_quantile(probability, degrees):
    if not 0.5 < probability < 1 or degrees < 1:
        raise ValueError((probability, degrees))
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, degrees) < probability:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if student_t_cdf(mid, degrees) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def interval(values, probability=PROCEDURE["critical_probability"]):
    mean = statistics.mean(values)
    critical = student_t_quantile(probability, len(values) - 1)
    half = critical * statistics.stdev(values) / math.sqrt(len(values))
    return {"n": len(values), "mean": mean, "sd": statistics.stdev(values),
            "critical": critical, "low": mean - half, "high": mean + half}


def evaluate_reference(jobs):
    by = {arm: [j for j in jobs if j["arm"] == arm] for arm in SPECS}
    looks, resolved, needed = [], None, []
    for size in PROCEDURE["looks"]:
        size = None if size == "all_available" else size
        selected = {arm: rows[:size] if size else rows for arm, rows in by.items()}
        intervals = {arm: interval([j["val_loss"] for j in rows]) for arm, rows in selected.items()}
        winner = min(intervals, key=lambda arm: intervals[arm]["mean"])
        separated = all(intervals[winner]["high"] < values["low"]
                        for arm, values in intervals.items() if arm != winner)
        looks.append({"look": len(looks) + 1, "counts": {a: len(rs) for a, rs in selected.items()},
                      "intervals": intervals, "resolved": separated,
                      "winner": winner if separated else None})
        if separated:
            resolved = len(looks)
            needed = [j["job_id"] for rows in selected.values() for j in rows]
            break
    if not needed:
        needed = [j["job_id"] for j in jobs]
    return {"looks": looks, "resolving_look": resolved, "reference_job_ids": needed,
            "available_jobs": len(jobs), "reference_jobs": len(needed),
            "all_paced_gpu_h": sum(j["paced_1gpu_s"] for j in jobs) / 3600,
            "reference_paced_gpu_h": sum(j["paced_1gpu_s"] for j in jobs if j["job_id"] in needed) / 3600,
            "coverage_model": PROCEDURE["model"] + "; Bonferroni across 3 arms x 3 fixed looks",
            "per_interval_alpha": PROCEDURE["per_interval_alpha"],
            "two_sided_critical_probability": PROCEDURE["critical_probability"]}


def parents(recs):
    out = {}
    for arm, (group, count) in SPECS.items():
        rows = [r for r in recs.values() if r["identity"]["replicate_group"] == group
                and not r["identity"].get("derived_from") and r["readiness"]["replay"]
                and r["reference"]["gpu"] == "NVIDIA H100 80GB HBM3"
                and str(r["identity"]["trajectory"].get("environment", {}).get("torch", "")).startswith("2.11.0")]
        rows.sort(key=lambda r: (r["identity"]["seed"]["value"] is None,
                                r["identity"]["seed"]["value"] or 0, r["identity"]["record_id"]))
        if len(rows) != count:
            raise ValueError(f"{arm}: {len(rows)} source runs, expected {count}")
        out[arm] = rows
    paths = [r["source"]["path"] for rows in out.values() for r in rows]
    if len(set(paths)) != len(paths):
        raise ValueError("one underlying source run appears more than once")
    return out


def derive(recs, reg, write=True):
    """The standard prefix, plus its standard single-process prepared copy."""
    result, scripts = {}, {}
    for arm, rows in parents(recs).items():
        prepared = []
        prefixes = []
        for parent in rows:
            source = open(os.path.join(reg, parent["agent_view"]["script"])).read()
            prefix, pcode, prel = P.derive(parent, source, STEP, reg)
            if C.paced_duration(prefix, 1) < 1200:
                raise ValueError(f"{prefix['identity']['record_id']}: under 20 minutes")
            prefixes.append((parent, prefix, pcode, prel))
        # P.derive retains the full parent's disclosure; recompute after cutting
        # the complete cohort, as gen_prefix.main does for its own output.
        C.add_disclosure([prefix for _, prefix, _, _ in prefixes])
        for parent, prefix, pcode, prel in prefixes:
            scripts[prel] = pcode
            recs[prefix["identity"]["record_id"]] = prefix
            if write:
                C.write_text(os.path.join(reg, prel), pcode)
                C.write_json(os.path.join(reg, "records", prefix["identity"]["record_id"] + ".json"), prefix)
            code = SP.single_process(pcode)
            compile(code, f"{arm}/single-process", "exec")
            rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{parent['identity']['argv'][0]}"
            scripts[rel] = code
            record = copy.deepcopy(prefix)
            ident = record["identity"]
            bound = f"{ROOT}/{arm}/"
            group = C.sha256_text(f"k2-precision\0{arm}\0{STEP}")
            ident.update({
                "record_id": prefix["identity"]["record_id"] + "-k2p",
                "entry_hash": C.sha256_text(code), "replicate_group": "mng3-k2p-" + group[:12],
                "match_policy": "path_bound", "bound_path": bound,
                "match_policy_how": "exact bytes/argv first, otherwise the unique precision optimizer folder",
                "process_model": {"value": "single", "how": "build/single_process.py, one GPU and no DDP"},
                "derived_from": {**prefix["identity"]["derived_from"], "kind": "prepared",
                                 "via": prefix["identity"]["record_id"], "bound_path": bound,
                                 "change": "standard 1500-step prefix made single-process; no trajectory or result change"},
            })
            ident["trajectory"]["config_hash"] = group
            ident["trajectory"]["config_hash_how"] = "sha256 of k2-precision, optimizer name and stop step"
            record["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py")
            record["host"]["processes"] = C.tv(1, "derived", "single-process launch")
            record["agent_view"] = {**record["agent_view"], "script": rel,
                                    "launch_command": "python " + bound + " ".join(ident["argv"])}
            record["readiness"] = C.readiness(record)
            recs[ident["record_id"]] = record
            prepared.append(record)
            if write:
                C.write_text(os.path.join(reg, rel), code)
        C.add_disclosure(prepared)
        if write:
            for record in prepared:
                C.write_json(os.path.join(reg, "records", record["identity"]["record_id"] + ".json"), record)
        result[arm] = prepared
    return result, scripts


def arm_readme(arm, records):
    script = records[0]["identity"]["argv"][0]
    seeds = [r["identity"]["seed"]["value"] for r in records]
    if all(seed is None for seed in seeds):
        inventory = (f"There are {len(records)} independent source replications, without recorded seed control. "
                     "The script has no --seed option; a fresh launch is a new replication. Run identifiers are ordered "
                     "lexically; they are labels, not invented seeds.\n")
    else:
        inventory = "Available seeds, in file order: " + ", ".join(str(s) for s in seeds) + ".\n"
    command = f"python {ROOT}/{arm}/{script}" + (" --seed N" if seeds[0] is not None else "")
    return (f"# {arm}\n\n`{script}` is single-process on one GPU and prints its validation loss at step {STEP:,}. "
            "It uses the full learning-rate schedule and stops at that endpoint. Run it from the project root: "
            f"`{command}`.\n\n" + inventory
            + "\nAll comparisons use the same global batch (524,288 tokens), FineWeb training data order, and validation shard.\n")


def build(recs, disclosure_of, reg=None, write=True):
    if reg is None:
        import gen_tasks as GT
        reg = GT.REG_DIR
    import gen_tasks as GT
    by, _ = derive(recs, reg, write=write)
    jobs, files, arms = [], [], {}
    shards = 0
    for arm, records in by.items():
        script = records[0]["identity"]["argv"][0]
        files += [{"path": f"{ROOT}/{arm}/{script}", "from": records[0]["agent_view"]["script"]},
                  {"path": f"{ROOT}/{arm}/README.md", "content": arm_readme(arm, records)}]
        arms[arm] = []
        for index, record in enumerate(records, 1):
            ident = record["identity"]
            seed = ident["seed"]["value"]
            job_id = f"{arm}-replicate{index:02d}"
            job = {
                "job_id": job_id, "label": f"{arm}, independent replication {index}",
                "records": [ident["record_id"]], "replicate_group": ident["replicate_group"],
                "draw": "exact_seed" if seed is not None else "without_replacement", "seed": seed,
                "recorded_gpus": record["reference"]["gpu_count"], "recorded_gpu": record["reference"]["gpu"],
                "batch_policy": record["launch"]["batch_policy"]["value"], "gpu_options": [1],
                "launch": f"python {ROOT}/{arm}/{script}" + (f" --seed {seed}" if seed is not None else ""),
                "after": [], "consumes": [], "produces": [], "arm": arm, "file_order": index,
                "precision_look": 1 if index <= 4 else (2 if index <= 8 else 3),
            }
            job["disclosure"] = disclosure_of(job["records"], recs,
                                               f"{job['recorded_gpus']}x {job['recorded_gpu']}")
            jobs.append(job)
            arms[arm].append(job_id)
            shards = max(shards, record["host"]["train_data"]["shards_read"])
    arm_order = {arm: index for index, arm in enumerate(SPECS)}
    jobs.sort(key=lambda job: (job["file_order"], arm_order[job["arm"]]))
    position = {job["job_id"]: index for index, job in enumerate(jobs)}
    inventory = "replication\toptimizer\tseed\tcommand\n" + "".join(
        f"{job['file_order']}\t{job['arm']}\t{'' if job['seed'] is None else job['seed']}\t{job['launch']}\n"
        for job in jobs)
    files.append({"path": "RUNS.tsv", "content": inventory})
    # Recompute the scientific readout from source values; never hand-author
    # a winner or assume a resolving sample count from an earlier artifact.
    inputs = [{"job_id": j["job_id"], "arm": j["arm"],
               "val_loss": recs[j["records"][0]]["result"]["final_val_loss"]["value"],
               "paced_1gpu_s": C.paced_duration(recs[j["records"][0]], 1)} for j in jobs]
    readout = evaluate_reference(inputs)
    reference = readout["reference_job_ids"]
    rounds, previous = [], {arm: 0 for arm in arms}
    for look in readout["looks"]:
        batch = []
        for arm, ids in arms.items():
            batch += ids[previous[arm]:look["counts"][arm]]
            previous[arm] = look["counts"][arm]
        rounds.append(sorted(batch, key=position.__getitem__))
    reference = [jid for batch in rounds for jid in batch]
    brief = ("Which of contra-soft-muon, radial-damp and muonh-tuned has the lowest mean validation loss at step 1,500, "
             "at 95% simultaneous confidence, or can they not be told apart? RUNS.tsv lists the training runs you can launch. Report the "
             "estimates and uncertainty that support your conclusion. The machine has four H100s.")
    files.append({"path": "COMPARISON.md", "content": (
        "# Optimizer replication comparison\n\nRUNS.tsv is the round-robin replication inventory; a blank seed marks an unseeded muonh run, and its replication index is not a random seed. The three folders under precision/ contain the training scripts and the "
        "available independent replications. Each run reports validation loss at step 1,500. The request is a comparison "
        "of population means with simultaneous 95% uncertainty; report uncertainty if these runs do not distinguish them. "
        "The usual independent Gaussian seed-outcome model is appropriate for this comparison; state the assumptions "
        "and the simultaneous/sequential correction used. A seedless replication is not a controlled seed.\n")})
    return {
        "task_id": TASK_ID, "rung": 2, "env_kind": "mng3", "level": "hidden_plan",
        "title": "Precision-limited optimizer replication, 49 available H100 jobs",
        "brief": {"text": brief, "provenance": "K2 research request, 2026-11-02; precision deliverable, not a procedure"},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                           "from": "kjj0/fineweb10B-gpt2 validation and required training shards"}],
        "jobs": jobs,
        "precision": {
            "schema": "p006-precision-comparison/1", "metric": "val_loss", "step": STEP,
            "arms": arms, "procedure": PROCEDURE, "reference_batches": rounds,
            "reference_job_ids": reference, "available_jobs": len(jobs), "reference_jobs": len(reference),
            "source_readout": "package-search/lanes/unneeded-k2/replication-readout.json",
            "reference_outcome": {"resolving_look": readout["resolving_look"],
                                  "winner": readout["looks"][-1]["winner"],
                                  "counts": readout["looks"][-1]["counts"]},
            "reference_intervals": readout["looks"],
            "scoring_note": "reference needed set is the valid fixed-look procedure in file order; evaluate other valid "
                            "agent uncertainty procedures separately, not by assuming every unlaunched available job is missing",
        },
        "reference_plan": {"instances": [{"job_id": jid} for jid in reference], "policy": "packed",
                           "batches": rounds, "rule": "file-order looks at 4, 8 and all available per arm; stop when resolved",
                           "how": "scientific sequential reference; requires precision-aware scripted planning/scoring"},
        "measuring_run": {"harness": "scripted", "policy": "packed", "speed": 10.0, "role": "check",
                          "terms_from": ["pkg-p20-par"],
                          "how": "prefix host gaps transfer by optimizer folder; actual scripted precision plan check pending"},
        "notes": [
            "All 49 jobs are separately launchable; the declared reference uses 24. No initial outputs or prior run state.",
            "Source is shared with K/G4n; not an independent-source twin. Every parent source.path is distinct.",
            "Muonh's ten unseeded source runs use ascending record-id draw order, matching the daemon's None-seed tie-break.",
            "Simultaneous coverage is conditional on iid Gaussian outcomes; not distribution-free.",
            "A planner may find another valid comparison procedure. Its result must not be invalidated merely for using "
            "fewer than the reference's 24 jobs; this task needs the precision scorer column before model scoring.",
            "Source values and durations unchanged; prefixes derived with gen_prefix, scripts with single_process.",
        ],
        "sources": ["modded-nanogpt, commit 4ea6b937337a4889b8cfe3f38a93d120048d8f71; 49 distinct H100 runs"],
    }


def main():
    args = sys.argv[1:]
    check = "--check" in args
    args = [arg for arg in args if arg != "--check"]
    if len(args) != 1:
        raise SystemExit("usage: gen_k2_precision.py [--check] <registry>")
    reg = args[0]
    recs = {}
    for name in os.listdir(os.path.join(reg, "records")):
        if not re.match(r"mng3-.*-[0-9a-f]{8}-[0-9a-f]{3}\.json$", name):
            continue
        record = json.load(open(os.path.join(reg, "records", name)))
        if record["identity"]["replicate_group"] in {group for group, _ in SPECS.values()}:
            recs[record["identity"]["record_id"]] = record
    by, scripts = derive(recs, reg, write=not check)
    durations = [C.paced_duration(r, 1) for rows in by.values() for r in rows]
    print(json.dumps({"task": TASK_ID, "mode": "read-only check" if check else "records derived",
                      "available_jobs": len(durations), "script_artifacts": len(scripts),
                      "duration_min_range": [min(durations) / 60, max(durations) / 60],
                      "all_gpu_h": sum(durations) / 3600,
                      "contract_validation": "not run", "staging": "not run"}, indent=2))


if __name__ == "__main__":
    main()
