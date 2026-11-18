"""Independent Astra: a distributed-width reproducibility matrix.

This builder reads existing original records and writes no records or artifacts.
The required widths belong to the requested comparison, not an invented memory
constraint. Width-dependent timings are gpu-mock conversions, not independently
measured scaling curves. gpu-tasks owns integration via finish_one_task.py.

For a local, unfinished proposal and arithmetic report (no registry mutation):
  python3 -I build/gen_astra_widths.py REGISTRY OUTPUT_DIRECTORY
"""
import argparse
import collections
import copy
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gen_tasks as GT

TASK_ID = "pkg-astra-widths"
ROOT = "width_study"
CARD = "NVIDIA H100 80GB HBM3"
SETUPS = (
    ("contra-soft-muon", "20260509_contra_soft_muon", "contra-muon-to-soft-muon.py", (1, 2, 4)),
    ("radial-damp", "20260511_dampen_radial_gradient_component", "radial_scale_then_radius_correct.py", (1, 2, 4)),
    ("muonh-tuned", "20260611_muonh_tuned_aux_3250", "train_gpt_simple_muonh.py", (1, 2)),
)


def pool(recs, submission, script):
    out = [r for r in recs.values()
           if not r["identity"].get("derived_from")
           and r["source"].get("submission") == submission
           and r["identity"]["argv"][0] == script
           and r["reference"]["gpu"] == CARD
           and str(r["identity"]["trajectory"]["environment"].get("torch", "")).startswith("2.11.0")
           and r["readiness"]["replay"] and GT.drawable(r)]
    out.sort(key=lambda r: (r["identity"]["seed"]["value"] is None,
                            r["identity"]["seed"]["value"] or 0, r["identity"]["record_id"]))
    if not out or len({r["identity"]["replicate_group"] for r in out}) != 1:
        raise ValueError(f"{submission}: expected a single nonempty original-record group")
    if len({r["source"]["path"] for r in out}) != len(out):
        raise ValueError(f"{submission}: duplicate underlying recordings")
    return out


def schedule(rows, gpus, order, backfill=True):
    """Nonpreemptive rigid jobs; each duration is the median converted record time.

    Proxy ordering reads only steps and width; actual duration is used by the
    simulation clock, not by that policy. No agent behavior is fitted here.
    """
    pending = list(order)
    active = []
    trace = []
    now = 0.0
    while pending or active:
        free = gpus - sum(w for _, w, _ in active)
        launched = False
        for ix in list(pending):
            r = rows[ix]
            if r["width"] <= free:
                pending.remove(ix)
                active.append((now + r["duration_s"], r["width"], ix))
                trace.append({"job_id": r["job_id"], "start_s": now,
                              "end_s": now + r["duration_s"], "width": r["width"]})
                free -= r["width"]
                launched = True
            elif not backfill:
                break
        if not active:
            raise ValueError("No feasible launch")
        if not launched or not pending or free == 0 or all(rows[i]["width"] > free for i in pending):
            now = min(t for t, _, _ in active)
            active = [a for a in active if a[0] > now + 1e-8]
    return now, trace


def figures(jobs, recs, gpus):
    rows = []
    for j in jobs:
        ds = [C.paced_duration(recs[x], j["gpu_options"][0]) for x in j["records"]]
        rows.append({"job_id": j["job_id"], "width": j["gpu_options"][0],
                     "steps": recs[j["records"][0]]["steps"], "duration_s": C.median(ds),
                     "min_duration_s": min(ds), "max_duration_s": max(ds)})
    # This optimistic bound applies to every draw, including the unseeded pool.
    lb = max(sum(r["min_duration_s"] * r["width"] for r in rows) / gpus,
             max(r["min_duration_s"] for r in rows))
    policies = {}
    for name, order, backfill in (
        ("file_order_backfill", list(range(len(rows))), True),
        ("file_order_head_blocking", list(range(len(rows))), False),
        ("code_steps_per_width", sorted(range(len(rows)), key=lambda i: (-rows[i]["steps"] / rows[i]["width"], i)), True),
        ("width_first", sorted(range(len(rows)), key=lambda i: (-rows[i]["width"], -rows[i]["steps"], i)), True),
        ("duration_informed", sorted(range(len(rows)), key=lambda i: (-rows[i]["duration_s"], i)), True),
    ):
        end, trace = schedule(rows, gpus, order, backfill)
        policies[name] = {"makespan_s": end, "over_optimistic_lower_bound": end / lb, "schedule": trace}
    return {"jobs": len(rows), "gpus": gpus, "width_counts": dict(collections.Counter(r["width"] for r in rows)),
            "optimistic_lower_bound_s": lb, "min_job_s": min(r["min_duration_s"] for r in rows),
            "max_job_s": max(r["max_duration_s"] for r in rows), "policies": policies, "jobs_table": rows,
            "scope": "Ideal GPU pacing only; no host startup, model behavior or measured agent result."
                     " Unseeded jobs use pool medians; lower bound uses minima."}


def build(recs, disclosure_of, reg=None, gpus=8):
    if gpus not in (4, 8):
        raise ValueError("gpus must be 4 or 8")
    jobs, files, seen_paths = [], [], set()
    manifest = ["setup,width,replicate,seed,script"]
    launch_lines = []
    shards = 0
    sources = set()
    for setup, submission, script, widths in SETUPS:
        rs = pool(recs, submission, script)
        seeded = GT.seed_mechanism(rs[0]) == "argv"
        if not seeded and GT.seed_mechanism(rs[0]) != "none":
            raise ValueError("Unsupported seed mechanism")
        if len(rs) % len(widths):
            raise ValueError(f"{setup}: uneven replication across requested widths")
        n = len(rs) // len(widths)
        path = f"{ROOT}/{setup}/{script}"
        if len({r["identity"]["entry_hash"] for r in rs}) != 1:
            raise ValueError(f"{setup}: scripts differ")
        files.append({"path": path, "from": rs[0]["agent_view"]["script"]})
        for windex, width in enumerate(widths):
            for rep in range(1, n + 1):
                r = rs[windex * n + rep - 1]
                seed = r["identity"]["seed"]["value"] if seeded else None
                jid = f"{setup}-w{width}-r{rep:02d}"
                line = f"torchrun --standalone --nproc_per_node={width} {path}"
                if seeded:
                    line += f" --seed {seed}"
                # Unseeded launches share the full pool: each launch draws once,
                # rather than falsely claiming the script selects a given record.
                j = GT.job_from_group(jid, f"{setup}, {width} GPUs, replicate {rep}",
                                      r["identity"]["replicate_group"], rs, seed=seed,
                                      gpu_options=[width], launch=line,
                                      setup=setup, required_width=width, replicate=rep)
                ds = [C.paced_duration(recs[x], width) for x in j["records"]]
                if any(d is None or d < 1200 for d in ds):
                    raise ValueError(f"{jid}: a drawable job is under 20 minutes or unconvertible")
                jobs.append(j)
                manifest.append(f"{setup},{width},{rep},{'' if seed is None else seed},{path}")
                launch_lines.append(line)
                shards = max(shards, r["host"]["train_data"]["shards_read"])
                sources.add(r["source"]["path"])
        if any(r["source"]["path"] in seen_paths for r in rs):
            raise ValueError("A recorded run occurs in two setups")
        seen_paths.update(r["source"]["path"] for r in rs)
    problems = C.repeated_records({"jobs": jobs}, recs)
    if problems:
        raise ValueError(problems)
    files.extend([
        {"path": "study.csv", "content": "\n".join(manifest) + "\n"},
        {"path": "commands.txt", "content": "\n".join(launch_lines) + "\n"},
        {"path": "STUDY.md", "content": (
            "# Distributed optimizer comparison\n\n"
            "The rows of study.csv specify the report's independent replicates. Width is the number of GPUs assigned "
            "to that replicate and is part of the requested comparison. A row at one width does not replace a row "
            "at another width. commands.txt gives each row's unmodified training invocation.\n\n"
            "The scripts retain their standard global batch and learning-rate schedule; gradient accumulation "
            "accounts for the worker count. Seeded setups use separate seeds across widths. muonh-tuned takes no "
            "seed argument; its rows are independent replicates. No statistical equivalence claim is required: "
            "report the observations and the limitations of the replicate counts. Each run writes its own log.\n")},
        {"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
         "from": f"kjj0/fineweb10B-gpt2: validation shard and first {shards} training shards"},
    ])
    return {"task_id": TASK_ID + ("-4gpu" if gpus == 4 else ""), "rung": 1, "env_kind": "mng3",
            "box": copy.deepcopy(GT.BOX8 if gpus == 8 else GT.BOX), "level": "given_plan", "set": "development",
            "title": "Optimizer reproducibility across distributed worker counts",
            "brief": {"text": (
                "We are comparing the validation behavior of our optimizer implementations across distributed worker "
                "counts. study.csv specifies the optimizer, GPU width and independent replicates for the comparison; "
                "the training scripts are in width_study/. We need a report with each row's final validation loss, "
                "the mean and spread for each optimizer-width pair, and a discussion of whether the optimizer ranking "
                "changes across widths. Include the actual GPU width and the supporting log for every row. "
                f"The machine has {'eight' if gpus == 8 else 'four'} H100s."),
                "provenance": "Independent Astra package design, 2026-11-02. Widths are an experimental factor in the "
                              "deliverable; no time limit, fake memory requirement or artificial delay."},
            "files": files, "jobs": jobs, "measuring_run": GT.measuring_run(jobs),
            "sources": ["modded-nanogpt track 3, pinned source records already in the registry; H100 SXM, torch 2.11.0"],
            "notes": [
                "Each seeded cell has a distinct original recording. Unseeded muonh-tuned draws without replacement "
                "from its complete original pool. No prefix, duplicated source run, changed script or invented curve.",
                "Width-dependent times use gpu-mock convert(); these are not independently recorded multiwidth "
                "scaling measurements. The package studies agent resource orchestration, not physical scaling fidelity.",
                "Required width follows the report's experimental factor, not memory capacity. The daemon permits "
                "other widths. Scoring must check actual launch width and mark wrong-width rows incomplete rather "
                "than crediting an incorrect report. Runner must verify this before interpreting completion.",
                "All original scripts remain distributed. CUDA_VISIBLE_DEVICES placement and torchrun rendezvous "
                "must be tested in the normal VM staging/smoke path. No local staged or cluster result is asserted.",
                "A scheduler may remove this mechanism. That would be a measured counterexample to the broader "
                "claim, not a reason to omit the screen. This is not declared a qualifying package.",
            ],
            "astra_design": figures(jobs, recs, gpus)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("registry")
    p.add_argument("output")
    p.add_argument("--gpus", type=int, default=8)
    a = p.parse_args()
    recs = C.load_records(os.path.join(a.registry, "records"))
    task = build(recs, GT.disclosure_of, a.registry, gpus=a.gpus)
    os.makedirs(a.output, exist_ok=True)
    for suffix, obj in (("proposal", task), ("arithmetic", task["astra_design"])):
        C.write_json(os.path.join(a.output, task["task_id"] + "." + suffix + ".json"), obj)
    d = task["astra_design"]
    print(json.dumps({k: v for k, v in d.items() if k not in ("jobs_table", "policies")}, indent=2))
    for name, f in d["policies"].items():
        print(name, round(f["makespan_s"], 2), round(f["over_optimistic_lower_bound"], 6))


if __name__ == "__main__":
    main()
