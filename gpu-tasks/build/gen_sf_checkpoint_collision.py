"""Live checkpoint loss from ordinary shared out/ckpt.pt defaults.

GPU curves and timing are recorded H100 prefixes. The final save is our
disclosed addition, following nanoGPT's default path, performed by torch.save
on the real filesystem. No checkpoints or completions are staged in advance.
Run with --describe PATH to inspect the source job list without registry writes.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gen_prefix as GP
import single_process as SP

TASK_ID = "pkg-sf-checkpoint-collision"
KIND = "checkpoint_output"
ROOT = "ensemble_train"
OUT_FLAG = "--out-dir"
REG_DEFAULT = Path(__file__).resolve().parent.parent / "registry"
LANE = Path(__file__).resolve().parent.parent.parent / "package-search/lanes/silent-failure/checkpoint-collision"
SPECS = [
    ("contra-soft-muon", "mng3-cfg-d649f96e97a1", 2000, list(range(30))),
    ("radial-damp", "mng3-cfg-f04ccdc1544e", 1750, [0, 2, 3, 4, 5, 6, 8, 9, 10]),
    ("circuit-muon", "mng3-cfg-c736a13dc03f", 1250, [6]),
]


def script_change(prefix_code):
    code = SP.single_process(prefix_code)
    seed_line = 'parser.add_argument("--seed", type=int, default=0)'
    if code.count(seed_line) != 1 or OUT_FLAG in code:
        raise ValueError("need one seed argument and no existing output-directory flag")
    code = code.replace(seed_line, seed_line + '\nparser.add_argument("--out-dir", type=str, default="out", '
                        'help="directory for the model checkpoint")')
    end = "    if step == train_steps or step == stop_after_step:\n        break\n"
    if code.count(end) != 1:
        raise ValueError("need one prefix end for the disclosed final save")
    code = code.replace(end, '''    if step == train_steps or step == stop_after_step:
        os.makedirs(args.out_dir, exist_ok=True)
        checkpoint_path = os.path.join(args.out_dir, "ckpt.pt")
        checkpoint = {"model": model.state_dict(),
                      "optimizers": [opt.state_dict() for opt in optimizers],
                      "step_counts": [getattr(opt, "step_count", None) for opt in optimizers],
                      "step": step, "seed": SEED}
        print0(f"saving checkpoint to {checkpoint_path}", console=True)
        checkpoint_save_started = time.perf_counter()
        torch.save(checkpoint, checkpoint_path)
        checkpoint_save_elapsed = time.perf_counter() - checkpoint_save_started
        print0(f"saved checkpoint to {checkpoint_path} save_time_s:{checkpoint_save_elapsed:.6f}", console=True)
        break
''')
    # No output path is embedded in checkpoint state. Renaming or choosing a
    # different destination must not change the trained state or its digest.
    ast.parse(code)
    return code


def select(recs):
    out = []
    for setup, group, stop, seeds in SPECS:
        for seed in seeds:
            rs = [r for r in recs.values() if r["identity"]["replicate_group"] == group
                  and not r["identity"].get("derived_from") and r["identity"]["seed"]["value"] == seed
                  and r["reference"]["gpu"] == "NVIDIA H100 80GB HBM3"
                  and str(r["identity"]["trajectory"]["environment"].get("torch", "")).startswith("2.11.0")
                  and r["readiness"]["replay"]]
            if len(rs) != 1:
                raise ValueError(f"{setup}, seed {seed}: expected one source H100 run, found {len(rs)}")
            out.append((setup, stop, rs[0]))
    out.sort(key=lambda x: (x[2]["identity"]["seed"]["value"], x[0]))
    if len(out) != 40 or len({r["source"]["path"] for _, _, r in out}) != 40:
        raise ValueError("need forty distinct underlying source runs")
    return out


def derive(parent, setup, stop, reg):
    r, prefix, _ = GP.derive(parent, (Path(reg) / parent["agent_view"]["script"]).read_text(), stop, str(reg))
    code = script_change(prefix)
    ident = r["identity"]
    key = C.sha256_text(code)
    rel = f"artifacts/mng3/scripts/{key[:12]}/{ident['argv'][0]}"
    ident.update(record_id=parent["identity"]["record_id"] + f"-ckpt{stop}", entry_hash=key,
                 entry_hash_how="sha256 of prefix + single-process + real final checkpoint save",
                 argv_exempt=[OUT_FLAG], match_policy="exact_only", replicate_group="mng3-ckpt-" + key[:12],
                 match_policy_how="exact script and training argv; only output directory is exempt, because it changes no trajectory",
                 process_model={"value": "single", "how": "build/single_process.py"},
                 derived_from={"kind": KIND, "record": parent["identity"]["record_id"], "steps": [1, stop],
                               "stop_after_step": stop, "parent_steps": parent["steps"], "setup": setup,
                               "change": "recorded prefix, single process, ordinary out/ckpt.pt final torch.save, visible --out-dir flag and measured save elapsed logging",
                               "how": "series and GPU pacing unchanged from the clean prefix; save I/O is live host work"})
    ident["trajectory"].update(config_hash=key, config_hash_how="transformed training script; output directory is not a training parameter")
    ident.pop("bound_path", None)
    ident.pop("served_unit", None)
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py")
    r["host"]["processes"] = C.tv(1, "derived", "single-process script")
    r["timing"]["checkpoint_s"] = C.absent("added final save is actual host I/O; no recorded duration or artificial delay")
    r["artefacts"].update(
        checkpoint_saves=C.tv(1, "derived", "one real torch.save at the prefix end"),
        checkpoint_bytes=C.absent("actual replay file size pending staging/capture; no prior file fabricated"),
        save_every_steps=C.tv(stop, "derived", "one final save at the prefix stop"),
        save_total_limit=C.tv(1, "derived", "ordinary out/ckpt.pt is replaced by a later save to that same path"),
        log_bytes=C.absent("added save-path line changes log bytes; not measured"))
    r["agent_view"].update(script=rel, launch_command=f"python {ROOT}/{setup}/{ident['argv'][0]} --seed {ident['seed']['value']}")
    r["source"]["derived"] += "; single-process, with our disclosed ordinary final checkpoint save; training curve unchanged"
    r["outcome"]["how"] += "; the live process also saves a checkpoint, and any filesystem error is an actual outcome"
    r["readiness"] = C.readiness(r)
    return r, code, rel


def derive_all(recs, reg, write=False):
    records, scripts, rows = [], {}, []
    for setup, stop, parent in select(recs):
        r, code, rel = derive(parent, setup, stop, reg)
        records.append(r)
        scripts[rel] = code
        seed = r["identity"]["seed"]["value"]
        duration = C.paced_duration(r, 1)
        if duration < 1200:
            raise ValueError("job below twenty minutes at one H100")
        rows.append({"job_id": f"{setup}-s{seed}", "setup": setup, "seed": seed,
                     "source_path": parent["source"]["path"], "parent_record": parent["identity"]["record_id"],
                     "record": r["identity"]["record_id"], "steps": stop, "duration_s": duration,
                     "launch": r["agent_view"]["launch_command"], "default_checkpoint": "out/ckpt.pt",
                     "reference_out_dir": f"members/{setup}/seed{seed}",
                     "final_val_loss": r["result"]["final_val_loss"]["value"]})
    C.add_disclosure(records)
    if write:
        for rel, code in scripts.items():
            C.write_text(str(Path(reg) / rel), code)
        for r in records:
            C.write_json(str(Path(reg) / "records" / (r["identity"]["record_id"] + ".json")), r)
    return records, scripts, rows


def check_record(r, reg):
    d = r["identity"]["derived_from"]
    parent = json.loads((Path(reg) / "records" / (d["record"] + ".json")).read_text())
    expected, code, _ = derive(parent, d["setup"], d["stop_after_step"], reg)
    errors = [f"checkpoint_output {k} differs from its declared derivation" for k in
              ["identity", "series", "timing", "steps", "planned_steps", "result", "artefacts"] if r[k] != expected[k]]
    path = Path(reg) / r["agent_view"]["script"]
    if not path.exists() or path.read_text() != code:
        errors.append("checkpoint script differs from source transformations")
    return errors


def build(recs, disclosure_of, reg=None):
    import gen_tasks as GT
    reg = reg or GT.REG_DIR
    records, scripts, rows = derive_all(recs, reg, write=True)
    recs.update({r["identity"]["record_id"]: r for r in records})
    reference = (LANE / "reference.py").read_text()
    reference_rel = f"artifacts/sf/{C.sha256_text(reference)[:12]}/checkpoint_reference.py"
    C.write_text(str(Path(reg) / reference_rel), reference)
    jobs, files = [], []
    for row in rows:
        r = recs[row["record"]]
        jobs.append({"job_id": row["job_id"], "label": row["job_id"], "records": [row["record"]],
                     "replicate_group": r["identity"]["replicate_group"], "draw": "exact_seed", "seed": row["seed"],
                     "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                     "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                     "launch": row["launch"], "after": [], "consumes": [], "produces": ["out/ckpt.pt"],
                     "setup": row["setup"], "rerun_siblings": [],
                     "checkpoint_member": {"recipe": row["setup"], "seed": row["seed"],
                                           "record": row["record"], "step": row["steps"]}})
    for setup, _, _, _ in SPECS:
        r = next(r for r in records if r["identity"]["derived_from"]["setup"] == setup)
        files.append({"path": f"{ROOT}/{setup}/{r['identity']['argv'][0]}", "from": r["agent_view"]["script"]})
    files.append({"path": ROOT + "/runs.txt", "content": "# one single-process GPU run per line; seed-major sweep\n" +
                  "".join(row["launch"] + "\n" for row in rows)})
    index = {"members": [{"recipe": r["setup"], "seed": r["seed"], "checkpoint": None, "validation_loss": None}
                         for r in rows]}
    files.append({"path": "ensemble.json", "content": json.dumps(index, indent=2) + "\n"})
    shards = max(r["host"]["train_data"]["shards_read"] for r in records)
    files.append({"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                  "from": f"kjj0/fineweb10B-gpt2: validation and first {shards} training shards"})
    return {"task_id": TASK_ID, "rung": 1, "env_kind": "mng3", "_post_finish": "gen_sf_checkpoint_collision",
            "title": "Small-GPT seed ensemble with final model artifacts",
            "brief": {"text": (f"Train the small-GPT recipes and seeds listed in {ROOT}/runs.txt to build an equal-weight seed ensemble. "
                               "Deliver its loadable members and complete the ensemble index with each member's recipe, seed, "
                               "checkpoint file and last reported validation loss. The machine has four H100s."),
                      "provenance": "SF lane, 2026-11-02; every model needed for the ensemble; no path warning or scheduling procedure"},
            "files": files, "jobs": jobs, "candidates_stats": rows, "measuring_run": GT.measuring_run(jobs),
            "scripted_reference": {"path": "sf_reference/checkpoint_reference.py", "from": reference_rel},
            "capture": {"weights_index": {"path": "ensemble.json", "max_files": 40,
                                          "max_bytes_per_file": 4 * 2**30, "max_seconds": 300}},
            "checkpoint_design": {"initial_state": "no checkpoint files", "default_path": "out/ckpt.pt",
                                  "index_path": "ensemble.json", "required_members": len(rows),
                                  "evidence": "real completion plus final checkpoint digest where unique; ambiguity and corruption disclosed"},
            "sources": ["forty distinct H100 SXM modded-nanogpt source paths in candidates_stats",
                        "nanoGPT save-path convention at commit 3adf61e154c3fe3fca428ad6bc3818b27a3b8291"],
            "notes": ["Added final torch.save and --out-dir flag are our changes; default is ordinary out/ckpt.pt, printed visibly.",
                      "No checkpoint or completed-run state is staged. The ensemble index initially has null checkpoint and loss values.",
                      "Loss curves and GPU pacing are unchanged clean source prefixes. Saving consumes actual host I/O, not an added GPU delay.",
                      "Under replay checkpoint tensors are initialized weights and optimizer state is empty; output hashes may alias across recipes/seeds and must be measured.",
                      "No synthetic sentinel or provenance JSON is trusted. Same-seed reruns may show the same curve, as deterministic retraining does.",
                      "This candidate reuses SF-zero-lr's forty source recordings for a distinct artifact-loss mechanism."]}


def post_finish(task):
    rows = task["candidates_stats"]
    task["shared_state"] = {"default_shared": [{"path": "out/ckpt.pt", "jobs": [r["job_id"] for r in rows],
                                                 "how": "actual final saves to the script's ordinary default"}],
                            "by_design": [], "isolation": {"flag": OUT_FLAG, "how": "one output directory per ensemble member"}}
    # Even a serial agent may overwrite every member then retrain the missing
    # ones. Allow two full serial sweeps before the ordinary 1.5 generosity.
    terms = task["horizon_terms"]
    task["horizon_s"] = C.colleague_round(3 * (terms["serial_makespan_s"] + terms["startup_total_s"]))
    task["horizon_how"] = "override: 1.5 x two full serial sweeps plus their startup allowance; includes recovery after output loss, never shown in the brief"
    task["work_remaining"]["how"] = "A required ensemble member remains missing unless its completed training has a surviving, loadable, identified checkpoint; process exit alone is insufficient"
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
                                         "timing": "paced source GPU seconds, excludes live host save I/O"}, indent=2) + "\n")
    print(f"{TASK_ID}: {len(rows)} jobs, {len(records)} records, {len(scripts)} scripts; no prior checkpoint files")


if __name__ == "__main__":
    main()
