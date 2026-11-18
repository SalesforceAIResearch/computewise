"""SF: a zero-effective-learning-rate configuration, with a real flag repair.

No registry writes on import. `--describe OUTPUT` derives in memory and writes
only a job list. The registry owner runs `gen_sf_zero_lr.py REGISTRY` to install
records/scripts, then calls build from gen_tasks. No training runs here.

The clean trajectories are recorded H100 runs, cut at a validation step and
made single-process with existing transformations. The zero-LR trajectory and
its copied timing are explicitly modelled. With the zero-initialized output
projection and every optimizer LR zero, logits remain zero: CE=log(50304).
The script actually multiplies all optimizer rates; this is not a mock flag.
"""
import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gen_prefix as GP
import single_process as SP

TASK_ID = "pkg-sf-zero-lr"
KIND = "silent_zero_lr"
ROOT = "sf_train"
FLAG = "--lr-scale"
VOCAB = 50304
FLAT_LOSS = math.log(VOCAB)
# Every eligible source run in these groups, never copies counted as new runs.
SPECS = [
    ("contra-soft-muon", "mng3-cfg-d649f96e97a1", 2000, 0),
    ("radial-damp", "mng3-cfg-f04ccdc1544e", 1750, 1),
    ("circuit-muon", "mng3-cfg-c736a13dc03f", 1250, 1),
]
REG_DEFAULT = Path(__file__).resolve().parent.parent / "registry"


def script_change(prefix_code):
    code = SP.single_process(prefix_code)
    seed_line = 'parser.add_argument("--seed", type=int, default=0)'
    if code.count(seed_line) != 1 or FLAG in code:
        raise ValueError("need one seed argument and no pre-existing LR-scale flag")
    code = code.replace(seed_line, seed_line + '\nparser.add_argument("--lr-scale", type=float, default=1.0, '
                        'help="multiplier on every optimizer learning rate")')
    point = "    set_hparams(step)\n"
    if code.count(point) != 1:
        raise ValueError("need exactly one loop call of set_hparams(step)")
    # Circuit-muon's EMA wrapper owns no LR. Follow the same inner optimizer
    # collection as its scheduler, while leaving opt.step() on the wrapper.
    lr_optimizers = "optimizers"
    if "optimizers = [EMA_Nesterov(" in code:
        lr_optimizers = "optimizers[0].inner_optimizer"
        scheduler = next(n for n in ast.parse(code).body
                         if isinstance(n, ast.FunctionDef) and n.name == "set_hparams")
        if not any(isinstance(n, ast.For) and ast.unparse(n.iter) == lr_optimizers
                   for n in ast.walk(scheduler)):
            raise ValueError("EMA scheduler no longer updates the expected inner optimizers")
    code = code.replace(point, point + '''    for opt in LR_OPTIMIZERS:
        for group in opt.param_groups:
            group["lr"] *= args.lr_scale
'''.replace("LR_OPTIMIZERS", lr_optimizers))
    update = "    for opt in optimizers:\n        opt.step()\n"
    if code.count(update) != 1:
        raise ValueError("need one optimizer-update loop for post-update diagnostics")
    code = code.replace(update, update + '''
    if step == 0 or (step + 1) % val_regular_interval == 0:
        effective_lrs = [float(group["lr"]) for opt in LR_OPTIMIZERS for group in opt.param_groups]
        print0(f"step:{step+1} lr_scale:{args.lr_scale:g} effective_lrs:{effective_lrs}", console=True)
'''.replace("LR_OPTIMIZERS", lr_optimizers))
    # This is the mathematical basis of the modelled flat curve, not an
    # arbitrary chosen score. Reject a source whose output initialization differs.
    if 'GPT(vocab_size=50304,' not in code or 'if "proj" in name:\n        p.data.zero_()' not in code:
        raise ValueError("uniform-output initialization no longer established")
    ast.parse(code)
    return code


def select(recs):
    out = []
    for setup, group, stop, scale in SPECS:
        parents = [r for r in recs.values()
                   if r["identity"]["replicate_group"] == group
                   and not r["identity"].get("derived_from")
                   and r["reference"]["gpu"] == "NVIDIA H100 80GB HBM3"
                   and str(r["identity"].get("trajectory", {}).get("environment", {}).get("torch", "")).startswith("2.11.0")
                   and r["readiness"]["replay"] and "--seed" in r["identity"]["argv"]]
        for parent in parents:
            out.append((setup, stop, scale, parent))
    out.sort(key=lambda x: (x[3]["identity"]["seed"]["value"], x[0]))
    paths = [x[3]["source"]["path"] for x in out]
    if len(out) < 40 or len(paths) != len(set(paths)):
        raise ValueError(f"need >=40 distinct real runs; selected {len(out)}, distinct {len(set(paths))}")
    return out


def derive(parent, setup, stop, scale, reg):
    parent_code = (Path(reg) / parent["agent_view"]["script"]).read_text()
    r, prefix_code, _ = GP.derive(parent, parent_code, stop, str(reg))
    code = script_change(prefix_code)
    ident = r["identity"]
    pid = parent["identity"]["record_id"]
    rid = f"{pid}-sf{stop}-lr{scale}"
    base = ident["argv"][0]
    rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{base}"
    key = C.sha256_text(code + f"\0lr_scale={scale}")
    ident.update({
        "record_id": rid, "entry_hash": C.sha256_text(code),
        "entry_hash_how": "sha256 of prefix + single_process + the actual LR multiplier and LR logging",
        "argv": ident["argv"] + [FLAG, str(scale)], "argv_exempt": [],
        "match_policy": "exact_only", "replicate_group": "mng3-sf-" + key[:12],
        "match_policy_how": "exact script bytes and argv, including --lr-scale; a different LR never selects this curve",
        "process_model": {"value": "single", "how": "build/single_process.py"},
        "derived_from": {"kind": KIND, "record": pid, "steps": [1, stop],
                         "stop_after_step": stop, "parent_steps": parent["steps"],
                         "parent_replicate_group": parent["identity"]["replicate_group"],
                         "lr_scale": scale, "setup": setup,
                         "change": "prefix; single process; multiply every optimizer-group LR by --lr-scale; log actual LRs",
                         "how": "clean curve is recorded; zero-LR curve and zero-LR pacing are modelled and disclosed"},
    })
    # Daemon _unit uses record id for this custom derived kind. Each LR config
    # is a different computation, even when the seed and timing anchor agree.
    ident.pop("served_unit", None)
    ident.pop("bound_path", None)
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = "sha256 of transformed script and lr_scale; each scale has a distinct group"
    ident["trajectory"]["lr_schedule"] = dict(ident["trajectory"]["lr_schedule"],
        identity=ident["trajectory"]["lr_schedule"]["identity"] + f":scale={scale}",
        source=ident["trajectory"]["lr_schedule"]["source"] + f"; every effective group LR multiplied by {scale}")
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py")
    r["host"]["processes"] = C.tv(1, "derived", "single-process script")
    seed = ident["seed"]["value"]
    r["agent_view"].update(script=rel,
        launch_command=f"python {ROOT}/{setup}/{base} --seed {seed} {FLAG} {scale}")
    r["failures"] = []
    if scale == 0:
        why = ("synthesised, not a measured failed run: every group LR is zero and the output projection starts at zero, "
               "so logits stay uniform over 50304 tokens and cross-entropy is ln(50304). No timing is slowed")
        for role in ["val_loss", "train_loss"]:
            if role in r["series"]:
                r["series"][role].update(values=[FLAT_LOSS] * len(r["series"][role]["steps"]),
                                          provenance="modelled", how=why)
        if "train_time_s" in r["series"]:
            r["series"]["train_time_s"].update(provenance="modelled",
                how="synthesised zero-LR run uses its clean timing anchor's recorded train clock unchanged")
        for k, v in r["timing"].items():
            if isinstance(v, dict) and v.get("value") is not None:
                v.update(provenance="modelled", how="zero-LR pacing assumed equal to the clean prefix; " + v.get("how", ""))
        r["result"]["final_val_loss"] = C.tv(FLAT_LOSS, "modelled", why)
        if "reached_target" in r["result"]:
            r["result"]["reached_target"] = C.tv(False, "derived", "uniform untrained loss exceeds the recorded target")
        r["outcome"] = {"status": "completed", "provenance": "modelled",
                        "how": "script has no zero-LR exit; finite loss and unchanged step count; no real zero-LR run collected"}
        r["source"]["derived"] = why + "; source.path identifies the clean timing anchor, not a recorded zero-LR run"
    else:
        r["source"]["derived"] += "; single-process script with LR scale 1, preserving its recorded trajectory"
    r["readiness"] = C.readiness(r)
    return r, code, rel


def derive_all(recs, reg, write=False):
    records, files, rows = [], {}, []
    for setup, stop, launch_scale, parent in select(recs):
        clean, code, rel = derive(parent, setup, stop, 1, reg)
        flat, _, _ = derive(parent, setup, stop, 0, reg)
        # Both scales are supported for every setup; the fault is a property of
        # LR=0, never of a task id, setup name or which lines were misconfigured.
        records.extend([clean, flat])
        files[rel] = code
        early, _, _ = GP.derive(parent, (Path(reg) / parent["agent_view"]["script"]).read_text(), 125, str(reg))
        first_update = copy.deepcopy(clean)
        first_update["steps"] = 1
        first_update["timing"]["eval_s"] = C.tv(0, "derived", "no validation before the first update")
        first_update["series"]["val_loss"] = {"steps": [], "values": []}
        duration = C.paced_duration(clean, 1)
        if duration < 1200:
            raise ValueError("selected job falls below twenty minutes on one H100")
        rows.append({"job_id": setup + "-s" + str(clean["identity"]["seed"]["value"]),
                     "setup": setup, "seed": clean["identity"]["seed"]["value"],
                     "source_path": parent["source"]["path"], "parent_record": parent["identity"]["record_id"],
                     "clean_record": clean["identity"]["record_id"], "flat_record": flat["identity"]["record_id"],
                     "stop_step": stop, "duration_s": duration,
                     "evidence_step": 125, "evidence_s": C.paced_duration(early, 1),
                     "evidence_policy": "first_eval (secondary reference)",
                     "first_update_evidence_step": 1,
                     "first_update_evidence_s": C.paced_duration(first_update, 1),
                     "initial_scale": launch_scale, "broken_as_briefed": launch_scale == 0,
                     "launch": (flat if launch_scale == 0 else clean)["agent_view"]["launch_command"],
                     "fixed_launch": clean["agent_view"]["launch_command"],
                     "clean_final_val_loss": clean["result"]["final_val_loss"]["value"]})
    C.add_disclosure(records)
    if write:
        for rel, code in files.items():
            C.write_text(str(Path(reg) / rel), code)
        for r in records:
            C.write_json(str(Path(reg) / "records" / (r["identity"]["record_id"] + ".json")), r)
    return records, files, rows


def check_record(r, reg):
    d = r["identity"]["derived_from"]
    parent = json.loads((Path(reg) / "records" / (d["record"] + ".json")).read_text())
    want, code, _ = derive(parent, d["setup"], d["stop_after_step"], d["lr_scale"], reg)
    errs = []
    for key in ["identity", "series", "timing", "steps", "planned_steps", "result", "outcome", "failures"]:
        if r[key] != want[key]:
            errs.append(f"silent_zero_lr {key} differs from its declared derivation")
    path = Path(reg) / r["agent_view"]["script"]
    if not path.exists() or path.read_text() != code:
        errs.append("silent_zero_lr script bytes differ from the parent transformations")
    return errs


def build(recs, disclosure_of, reg=None):
    import gen_tasks as GT
    reg = reg or GT.REG_DIR
    reference_source = Path(__file__).resolve().parent.parent.parent / "package-search/lanes/silent-failure/reference_zero_lr.py"
    reference_code = reference_source.read_text()
    reference_rel = f"artifacts/sf/{C.sha256_text(reference_code)[:12]}/reference_zero_lr.py"
    C.write_text(str(Path(reg) / reference_rel), reference_code)
    records, scripts, rows = derive_all(recs, reg, write=True)
    recs.update({r["identity"]["record_id"]: r for r in records})
    jobs, files = [], []
    for row in rows:
        r = recs[row["clean_record"]]
        ident = r["identity"]
        jobs.append({"job_id": row["job_id"], "label": row["job_id"],
                     "records": [row["clean_record"], row["flat_record"]],
                     "replicate_group": ident["replicate_group"], "draw": "exact_seed", "seed": row["seed"],
                     "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                     "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                     "launch": row["launch"], "after": [], "consumes": [], "produces": [],
                     "setup": row["setup"], "rerun_siblings": [],
                     "silent_failure": {"broken_as_briefed": row["broken_as_briefed"],
                                        "successful_records": [row["clean_record"]],
                                        "broken_records": [row["flat_record"]],
                                        "fixed_launch": row["fixed_launch"],
                                        "evidence_step": row["evidence_step"], "evidence_s": row["evidence_s"],
                                        "primary_evidence": "effective_lrs at first completed optimizer update",
                                        "first_update_evidence_step": row["first_update_evidence_step"],
                                        "first_update_evidence_s": row["first_update_evidence_s"],
                                        "cause": "effective optimizer learning rates are zero",
                                        "fix": "restore --lr-scale 1; the recorded optimizer rates are then used"}})
    for setup, _, _, _ in SPECS:
        r = next(r for r in records if r["identity"]["derived_from"]["setup"] == setup)
        files.append({"path": f"{ROOT}/{setup}/{r['identity']['argv'][0]}", "from": r["agent_view"]["script"]})
    files.append({"path": ROOT + "/runs.txt", "content": "# one single-process GPU run per line; seed-major sweep\n" +
                  "".join(row["launch"] + "\n" for row in rows)})
    shards = max(r["host"]["train_data"]["shards_read"] for r in records)
    files.append({"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                  "from": f"kjj0/fineweb10B-gpt2: validation and first {shards} training shards"})
    return {"task_id": TASK_ID, "rung": 1, "env_kind": "mng3",
            "title": "Small-GPT optimizer comparison with a zero-learning-rate misconfiguration",
            "brief": {"text": (f"We're comparing small-GPT optimizer recipes. {ROOT}/runs.txt lists the training runs to launch and their seeds, "
                               "and each recipe's folder contains its training script. Produce a table of the last reported "
                               "validation loss for each trained run and explain any configuration repairs. The machine has four H100s."),
                      "provenance": "SF lane, 2026-11-02; goal, no procedure or deadline, GPU count stated"},
            "files": files, "jobs": jobs, "measuring_run": GT.measuring_run(jobs),
            "scripted_reference": {"path": "sf_reference/reference_zero_lr.py", "from": reference_rel,
                                   "how": "brief-only first-update observer; staged only for scripted W0, never in model workspaces"},
            "silent_failure_design": {"successful_records_by_job": {x["job_id"]: x["clean_record"] for x in rows},
                                      "reference": "package-search/lanes/silent-failure/sim_zero_lr.py",
                                      "how": "An exit-zero flat run does not satisfy a trained-model job. Score the clean record only."},
            "candidates_stats": rows,
            "sources": ["modded-nanogpt track 3; each source_path in candidates_stats; H100 SXM, torch 2.11.0"],
            "notes": ["One real cause, zero effective LR; no inference that this survives an agent that repairs a whole queue.",
                      "The --lr-scale flag and the zero values in the run list are our modifications. Every optimizer group is actually multiplied.",
                      "Clean curves and base hardware are recorded. Prefix and one-GPU conversion use existing builders. Zero-LR curves and copied timing are modelled, not observed failed runs.",
                      "Both configuration records belong to one logical job. Only clean completion is useful work; stock expected_scores is not the SF reference.",
                      "No path-bound matching, LR exemption, sleep, slowdown, failure exception, or changed finite-loss guard."]}


def post_finish(task):
    """Owner calls this after gen_tasks.finish, before writing the task.

    The normal serial horizon counts one attempt per logical job. Here a
    serial agent can finish every bad attempt and repair it, so include those
    attempts too. This changes neither recorded timing nor the score bound.
    """
    rows = task["candidates_stats"]
    extra = sum(x["duration_s"] for x in rows if x["broken_as_briefed"])
    n_bad = sum(x["broken_as_briefed"] for x in rows)
    terms = task["horizon_terms"]
    extra_setup = terms["startup_total_s"] * n_bad / len(rows)
    task["horizon_s"] = C.colleague_round(1.5 * (terms["serial_makespan_s"] +
                                               terms["startup_total_s"] + extra + extra_setup))
    terms.update(sf_broken_serial_s=extra, sf_extra_startup_allowance_s=extra_setup)
    task["horizon_how"] = ("override: colleague_round(1.5 x (clean serial makespan + clean start-ups + "
                           "every initially broken full attempt + proportional extra start-up allowance)). "
                           "A serial agent may inspect only exits and still repair every listed job within the window. "
                           "No deadline is shown to the agent; durations and scoring bounds are unchanged.")
    task["work_remaining"]["how"] = ("A logical job remains until one of its silent_failure.successful_records completes. "
                                     "Exit-zero completion of a broken_records configuration is not successful training. "
                                     "Runner's SF scorer implements this distinction.")
    return task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("registry", nargs="?", type=Path, default=REG_DEFAULT)
    p.add_argument("--describe", type=Path)
    a = p.parse_args()
    recs = C.load_records(str(a.registry / "records"))
    records, scripts, rows = derive_all(recs, a.registry, write=a.describe is None)
    if a.describe:
        a.describe.write_text(json.dumps({"task_id": TASK_ID, "gpus": 4, "jobs": rows,
                                         "timing": "paced H100 seconds, speed 1, host start-up excluded"}, indent=2) + "\n")
    print(f"{TASK_ID}: {len(rows)} distinct jobs; {len(records)} configuration records; {len(scripts)} scripts; "
          f"{sum(x['broken_as_briefed'] for x in rows)} initially zero-LR")


if __name__ == "__main__":
    main()
