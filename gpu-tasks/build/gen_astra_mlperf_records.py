"""Build the independent Astra MLPerf v4.1 LoRA candidate records.

Historical builder retained after the study's 2026-11-03 stop on new
packages. This candidate never became task-ready and is not integrated.
Disposition: package-search/astra-independent/SEARCH-STOP-2026-11-03.md.

Original usage: run on a fleet VM with --source-root pointing to the acquired independent lane
and --output pointing to an isolated candidate registry. No task or shared
registry is changed. Source scripts are copied, never imported or executed.
Unresolved serving and initialization facts stay in materialisation.blockers;
common.readiness computes the resulting status.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C


BLOCKERS = [
    "NeMo requires at least r8.8 (357333e2): the reduced TP4 loop reports the intended metrics, but width-8/DP2 serving and the real warmup/full-validation timing ratio remain untested; evidence is in package-search/astra-independent/mlperf-progress/runtime-probes/e2-r88-1",
    "The original 70B NeMo checkpoint restoration and real training/validation arrays have not been staged or served; reduced-model synthetic probe assets are not replacements",
    "Launch reconstruction is partial: FP8, communication, warmup and remaining source environment settings are not yet bound to each trial",
    "Only entry-hash and Hydra argv matching have been established; arbitrary identity.env matching and env-only callback consumers remain unresolved",
]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(run, job, source, entry, script_rel):
    """Convert observed values and explicitly identify the uniform pacing model."""
    width = run["hardware"]["gpu_count"]
    params, timing = run["parameters"], run["timing_components"]
    seed, steps = params["seed"], run["steps"]
    if run["status"] != "success" or width not in (4, 8):
        raise ValueError(f"Unexpected outcome or width: {run['path']}")
    if run["path"] != job["source_path"] or run["sha256"] != job["source_sha256"]:
        raise ValueError(f"Proposal/source mismatch: {run['path']}")
    if job["derived_environment"]["SEED"] != str(seed):
        raise ValueError(f"Dataset/model seed mismatch: {run['path']}")
    argv, env = job["derived_argv"], job["derived_environment"]
    cfg_basis = {"entry_hash": entry, "argv": argv, "env": env,
                 "source_commit": source["source_tree_sha"], "source_path": run["path"]}
    cfg_hash = digest(json.dumps(cfg_basis, sort_keys=True).encode())
    signature = digest(json.dumps({k: run[k] for k in
        ("parameters", "run_duration_s", "evaluations", "train_loss")}, sort_keys=True).encode())
    if run.get("trajectory_signature_sha256", signature) != signature:
        raise ValueError(f"Trajectory signature mismatch: {run['path']}")
    rid = f"mlperf41-lora-w{width}-s{seed}-{digest(run['path'].encode())[:10]}"
    vals = run["series"]["val_loss"]
    hit_index = next(i for i, value in enumerate(vals["values"]) if value <= .95)
    hit_step = vals["steps"][hit_index]
    if hit_step != job["record_target_step"]:
        raise ValueError(f"Target step mismatch: {run['path']}")
    steady = timing["mean_step_s"]
    eval_total = timing["evaluation_elapsed_s"]
    paced_target = hit_step * steady + (hit_index + 1) * eval_total / len(vals["steps"])
    launch = shlex.join(["env", *(f"{k}={v}" for k, v in sorted(env.items())),
                         "torchrun", "--standalone", f"--nproc_per_node={width}", *argv])
    rec = {
        "schema": C.SCHEMA_RECORD,
        "identity": {
            "record_id": rid, "kind": "train", "entry_hash": entry,
            "entry_hash_how": "SHA-256 of the unchanged submitted train.py; not proof of bytes at the original launch",
            "argv": argv, "argv_provenance": "derived", "argv_how": job["argv_how"],
            "argv_exempt": [], "env": env, "env_exempt": [],
            "env_provenance": "derived", "env_how": job["environment_how"],
            "seed": C.tv(seed, "recorded", "MLLOG seed event"),
            "replicate_group": "mlperf41-lora-exact-" + cfg_hash[:16],
            "match_policy": "exact_only",
            "match_policy_how": "One source trajectory per full Hydra identity including width and model.seed; no seed pool or cross-width interpolation",
            "trajectory": {
                "config_hash": cfg_hash,
                "config_hash_how": "SHA-256 of entry_hash, derived argv/env, pinned source commit and original log path; seed remains part of this exact-record identity",
                "environment": {"source_framework": run["hardware"]["framework"],
                                "nemo_commit": "4068955b5add5a52a493c10be12437de3f4ca361",
                                "original_dependency_lock": None},
                "lr_schedule": {"identity": f"source CosineAnnealing, peak={params['opt_base_learning_rate']}, configured_steps={params['opt_learning_rate_training_steps']}",
                                "source": "Submitted YAML plus MLLOG hyperparameters; no learning-rate trace extracted", "trace": None},
                "data": {"files": ["/data/train.npy", "/data/validation.npy"],
                         "how": "Submitted source YAML; actual arrays and original checkpoint not acquired"},
                "global_batch_sequences": params["global_batch_size"],
                "precision": C.absent("Remaining per-trial FP8/BF16 launch reconstruction is unresolved"),
            },
            "fingerprint": None, "fingerprint_version": None, "fingerprint_inputs": None,
            "fingerprint_how": "Exact entry+argv candidate; no structural match claimed for the NeMo/Megatron optimizer path",
        },
        "mode": "replay",
        "source": {
            "name": "MLPerf Training v4.1 Llama-2-70B LoRA", "repo": "https://github.com/mlcommons/training_results_v4.1",
            "commit": source["source_tree_sha"], "path": run["path"], "url": run["url"], "sha256": run["sha256"],
            "trajectory_signature_sha256": signature,
            "licence": "code: Apache-2.0 in train.py header; result-log license not separately established here",
            "read_on": "2026-11-02", "script_how": "Submitted source at the pinned results revision, with byte-identical train.py and callbacks across the three selected vendors; exact original environment is not attested",
        },
        "reference": {
            "gpu": run["hardware"]["gpu"], "gpu_provenance": "inferred",
            "gpu_how": "Submitted system descriptor linked by the result directory, not telemetry inside this log",
            "gpu_count": width, "gpu_count_provenance": "inferred",
            "gpu_count_how": "Submitted system descriptor for this result directory",
            "interconnect": None, "precision": None,
        },
        "launch": {
            "launcher": "torchrun --standalone", "launcher_provenance": "derived",
            "world_size": C.tv(width, "inferred", "Submitted system descriptor"),
            "per_device_microbatch": C.tv(1, "inferred", "Submitted TP4 configuration MBS=1"),
            "accumulation_steps": C.tv(params["gradient_accumulation_steps"], "recorded", "MLLOG gradient_accumulation_steps"),
            "global_batch_sequences": C.tv(params["global_batch_size"], "recorded", "MLLOG global_batch_size"),
            "batch_policy": C.tv("fixed_per_device", "modelled", "Conservative serving restriction: only the recorded width is supported. This disables cross-width conversion; it does not claim a measured NeMo scaling law"),
            "gpu_counts_supported": [width],
            "parallelism": {"tensor": 4, "pipeline": 1, "context": 1, "data": width // 4,
                            "how": "Submitted configurations; global batch = microbatch x accumulation x data-parallel groups"},
        },
        "steps": steps, "planned_steps": params["opt_learning_rate_training_steps"],
        "planned_steps_how": "MLLOG opt_learning_rate_training_steps; original callback stopped earlier at its quality target",
        "per_step_gpu_s": steady,
        "timing": {
            "pacing": "per_step",
            "train_time_s": C.tv(timing["train_elapsed_s"], "derived", "run_stop minus run_start, less all logged eval_start to eval_stop intervals"),
            "steady_step_s": C.tv(steady, "modelled", "Uniform pacing: non-eval timed region / completed optimizer steps. Mean is arithmetic; constant per-step assignment is an approximation"),
            "first_step_s": C.absent("No separate first optimizer-step duration extracted"),
            "startup_in_train_s": C.absent("No separate startup inside the timed region extracted; not asserted zero"),
            "setup_s": C.absent("Logged init_start to run_start is retained in observed only; actual initialization executes on the host"),
            "eval_s": C.tv(eval_total, "derived", "Sum of logged eval_start to eval_stop intervals; current serving divides this equally over the sparse val_loss points"),
            "checkpoint_s": C.absent("No checkpoint timing; submitted YAML disables saving"),
            "wall_total_s": C.tv(run["run_duration_s"], "derived", "MLLOG run_stop minus run_start; excludes initialization"),
        },
        "memory": {"peak_allocated_bytes": None, "peak": C.absent("The source log has no peak allocated-memory measurement")},
        "series": run["series"],
        "observed": {"source_evaluation_intervals": run["evaluations"],
                     "init_to_run_s": run["init_to_run_s"],
                     "max_abs_uniform_pacing_eval_endpoint_error_s": timing["max_abs_constant_pacing_checkpoint_error_s"],
                     "how": "Exact source intervals retained for comparison; uniform pacing is proposed, not an observed per-step trace"},
        "materialisation": {
            "reads": ["NeMo inference_epoch_end reduces/broadcasts loss, then Lightning logs val_loss; original MetricsLogger emits MLLOG and stops below 0.925"],
            "needs_series": ["train_loss", "val_loss"], "blockers": list(BLOCKERS),
            "serving": {"mode": "per_op", "how": "Proposed original NeMo loop with step pacing and sparse tensor-value serving; not whole_job and no recorded stdout replay"},
        },
        "result": {role: C.tv(s["values"][-1], "recorded", f"Last MLLOG {role} value at optimizer step {s['steps'][-1]}") for role, s in run["series"].items()},
        "outcome": {"status": "completed", "provenance": "recorded", "how": "MLLOG run_stop metadata.status = success"},
        "failures": [],
        "artefacts": {"checkpoint_saves": C.tv(0, "inferred", "Submitted trainer.enable_checkpointing=False and save_nemo_on_validation_end=False"),
                      "final_model_save": C.tv(False, "inferred", "Submitted entry ends after trainer.fit; no final save"),
                      "checkpoint_bytes": C.absent("No saved checkpoint"),
                      "log_bytes": C.absent("Copied source log bytes differ from the agent's eventual output")},
        "host": {"process_count": C.tv(width, "derived", "One proposed torchrun rank per requested GPU"),
                 "process_rss_peak_bytes": C.absent("No recorded RSS series"),
                 "how": "No host startup or environment-floor measurement exists for this candidate"},
        "agent_view": {"script": script_rel, "launch_command": launch,
                       "inputs": ["/ckpt", "/data/train.npy", "/data/validation.npy"],
                       "recognisability": "Public NeMo fine-tuning source; no prior output or source trajectory is placed in the task workspace"},
    }
    rec["readiness"] = C.readiness(rec)
    contract = {"job_id": job["job_id"], "records": [rid], "required_width": width, "gpu_options": [width],
                "progress_target": job["progress_target"], "record_target_step": hit_step,
                "source_target_s": job["first_sufficient_s"], "uniform_paced_target_s": paced_target,
                "uniform_full_paced_s": C.paced_duration(rec, width), "launch": launch,
                "timing_scope": "Candidate timing model, no measured host startup or probe outcome"}
    return rec, contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.source_root.resolve(), args.output.resolve()
    if out.parts[-3:] == ("experiments", "gpu-tasks", "registry"):
        parser.error("Use an isolated candidate registry; gpu-tasks owns canonical integration")
    source = json.loads((root / "sources/mlperf-4.1/lora-trajectories.json").read_text())
    proposal = json.loads((root / "mlperf-progress/proposal.json").read_text())
    jobs = {j["source_path"]: j for j in proposal["jobs"]}
    if len(jobs) != len(source["runs"]) or set(jobs) != {r["path"] for r in source["runs"]}:
        raise ValueError("Proposal and source run sets differ")
    src = root / "sources/mlperf-4.1/raw/Dell/benchmarks/llama2_70b_lora/implementations/nemo"
    entry = digest((src / "train.py").read_bytes())
    script_rel = f"artifacts/mlperf41-lora/scripts/{entry[:12]}/train.py"
    artifacts = out / script_rel
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    companions = {}
    for path in [src / "train.py", src / "custom_callbacks.py", *sorted((src / "conf").rglob("*"))]:
        if path.is_file():
            rel = path.relative_to(src)
            dest = artifacts.parent / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            companions[str(rel)] = digest(path.read_bytes())
    (out / "tasks").mkdir(exist_ok=True)
    records, contracts, seen, signatures = [], [], set(), set()
    for run in source["runs"]:
        raw = root / "sources/mlperf-4.1/raw" / run["path"]
        if digest(raw.read_bytes()) != run["sha256"]:
            raise ValueError(f"Raw log hash mismatch: {run['path']}")
        rec, contract = record(run, jobs[run["path"]], source, entry, script_rel)
        key = (entry, tuple(rec["identity"]["argv"]))
        if key in seen:
            raise ValueError("Two source trajectories would share one exact argv key")
        seen.add(key)
        signature = rec["source"]["trajectory_signature_sha256"]
        if signature in signatures:
            raise ValueError("Two candidate records would duplicate a source trajectory")
        signatures.add(signature)
        records.append(rec)
        contracts.append(contract)
    C.add_disclosure(records)
    for rec in records:
        dump(out / "records" / (rec["identity"]["record_id"] + ".json"), rec)
    total = sum(j["uniform_paced_target_s"] * j["required_width"] for j in contracts)
    summary = {"status": "source_records_generated; replay_unresolved; no_registry_task",
               "source_revision": source["source_tree_sha"], "record_count": len(records),
               "width_counts": dict(collections.Counter(j["required_width"] for j in contracts)),
               "replay_ready": sum(r["readiness"]["replay"] for r in records),
               "blockers": BLOCKERS, "entry_sha256": entry, "companion_sha256": companions,
               "completion_rule": "first_delivered_target",
               "sufficient_work_lower_bound_s": max(total / 8, max(j["uniform_paced_target_s"] for j in contracts)),
               "bound_scope": "Optimistic eight-GPU bound under proposed uniform timing, before any actual probe/host overhead; not the source-endpoint bound",
               "jobs": contracts}
    dump(out / "candidate.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "record_count", "width_counts", "replay_ready", "sufficient_work_lower_bound_s")}, indent=2))


if __name__ == "__main__":
    main()
