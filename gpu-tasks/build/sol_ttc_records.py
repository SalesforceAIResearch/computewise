"""Generate whole-job replay records from the pinned TTC CSV rows.

Usage: python3 -I generate_records.py <output-registry>
This does not edit the canonical registry. Readiness is computed from the
source fields and the retained representative real-entry replay evidence.
"""
import argparse
import ast
import functools
import hashlib
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parents[1] / "survey" / "sol-tabular-ttc"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common as C
import sol_ttc_candidates as SOURCE

SCRIPT = "artifacts/sol-tabular/single_cell.py"
SEAM = "tabarena.benchmark.experiment.experiment_runner:ExperimentRunner._run"


@functools.lru_cache(maxsize=1)
def memory_model():
    path = HERE / "memory-model.json"
    raw = path.read_bytes()
    model = json.loads(raw)
    for source in model["sources"]:
        if C.sha256_bytes((HERE / source["path"]).read_bytes()) != source["sha256"]:
            raise ValueError("TTC memory source differs: " + source["path"])
    return model, C.sha256_bytes(raw)


def memory_estimate(dataset, arm):
    """Static allocation proxy; no source peak or phase trace was measured."""
    model, digest = memory_model()
    shape = model["datasets"][dataset]
    a, p = model["architecture_defaults"], model["assumptions"]
    rows, features = shape["rows_total"], shape["features"]
    embed, cls = a["embed_dim"], a["cls_tokens"]
    width = embed * cls
    act, inp, slots = p["activation_bytes"], p["input_bytes"], p["live_tensor_slots"]
    weights = model["weights"]["regression" if shape["problem_type"] == "regression" else "classification"]
    weight_bytes = weights["serialized_bytes"]
    context = p["context_allocator_bytes"]
    # Circular grouping preserves F columns, each holding g values + g indicators.
    grouped = inp * rows * features * (2 * a["feature_group_size"])
    inducing = act * a["dist_blocks"] * features * a["inducing_points"] * embed
    column = act * slots * min(rows, a["row_chunk"]) * (features + cls) * embed
    icl = act * slots * rows * width
    inference = weight_bytes + context + grouped + inducing + max(column, icl)
    components = {"weights_proxy": weight_bytes, "context_allocator": context,
                  "inference_grouped_input": grouped, "inducing_hidden": inducing,
                  "column_live": column, "icl_live": icl, "inference_total": inference}
    adapted = arm == "tabpfn3_diagscale_aggregation_96"
    if adapted:
        n = min(shape["train_rows_ceil"], p["finetune_max_rows"])
        views = p["finetune_views"]
        delta_params = a["dist_blocks"] * embed + a["icl_layers"] * width
        ft_input = inp * views * n * features * (2 * a["feature_group_size"])
        # One hidden-width checkpoint per block, for both differentiable views.
        saved = act * views * n * ((a["dist_blocks"] + a["column_blocks"])
                                  * (features + cls) * embed + a["icl_layers"] * width)
        current = act * slots * n * (features + cls) * embed
        state = p["diagscale_state_bytes_per_parameter"] * delta_params
        training = weight_bytes + context + ft_input + saved + current + state
        inference += 4 * delta_params  # FP32 adapter parameters in the inference model.
        components.update({"finetune_rows": n, "finetune_views": views,
                           "finetune_grouped_input": ft_input, "saved_checkpoints": saved,
                           "finetune_live": current, "diagscale_parameters": delta_params,
                           "diagscale_state": state, "finetune_total": training,
                           "adapted_inference_total": inference})
        peak = max(inference, training)
    else:
        peak = inference
    how = (
        f"MODELLED static working-set proxy, not a measured source peak. {dataset}: "
        f"N={rows} full raw rows, F={features}; nominal train rows rounded up to {shape['train_rows_ceil']}. "
        f"TabPFN v3 source defaults E={embed}, CLS={cls}, D={width}, ICL layers={a['icl_layers']}, "
        f"distribution/column blocks={a['dist_blocks']}/{a['column_blocks']}, inducing points={a['inducing_points']}. "
        f"Weights proxy={weight_bytes} B, published serialized {weights['filename']} size at HF "
        f"{model['hf_metadata_revision']}; not a device measurement or the original run's checkpoint digest. "
        f"Assume {act} B activations, {inp} B grouped inputs, {slots} hidden-width live slots and "
        f"{context} B context/allocator slack. Inference = weights + slack + {grouped} B grouped input "
        f"(4*N*F*6) + {inducing} B inducing hidden (2*3*F*128*128) + "
        f"max({column} B column work [2*8*min(N,2048)*(F+4)*128], "
        f"{icl} B ICL work [2*8*N*512]). "
        + (f"DiagScale: max(adapted inference {inference} B, training {training} B); "
           f"training = weights + slack + {ft_input} B grouped inputs for {views} views at "
           f"T=min(ceil(train rows),10000)={n} + {saved} B saved layer checkpoints "
           f"[2*2*T*(6*(F+4)*128+24*512)] + {current} B live work "
           f"[2*8*T*(F+4)*128] + {state} B for {delta_params} diagonal parameters "
           "and gradients/Adam moments (16 B/parameter). " if adapted else "")
        + f"Peak={peak} B. The 96 configurations and 2 validation/8 test views do not multiply "
        "resident inference weights: configurations and single-device forwards are sequential. "
        "Default architecture, raw feature width, mixed precision, live slots and activation checkpointing "
        "are explicit proxy assumptions; checkpoint-internal configs, preprocessing/search differences, "
        "transient attention/compiler workspaces and allocator peaks were not measured. "
        f"Inputs/arithmetic: survey/sol-tabular-ttc/memory-model.json sha256 {digest}."
    )
    return {"peak_allocated_bytes": peak, "peak": C.tv(peak, "derived", how)}, components


def replay_evidence(code):
    path = HERE / "replay-proof.json"
    if not path.exists():
        return None, ["actual TTC entry replay evidence is absent"]
    proof = json.loads(path.read_text())
    observed = proof["observed"]
    amendment = None
    if proof["entry_hash"] != C.sha256_text(code):
        amendment_path = HERE / "replay-docstring-amendment.json"
        if amendment_path.exists():
            change = json.loads(amendment_path.read_text())
            old, new = change["original_docstring_literal"], change["revised_docstring_literal"]
            restored = old + code[len(new):] if code.startswith(new) else ""
            if (change["original_entry_hash"] == proof["entry_hash"]
                    and change["revised_entry_hash"] == C.sha256_text(code)
                    and change["original_replay_proof_sha256"] == C.sha256_bytes(path.read_bytes())
                    and C.sha256_text(restored) == proof["entry_hash"]):
                before, after = ast.parse(restored), ast.parse(code)
                if (isinstance(before.body[0], ast.Expr) and isinstance(after.body[0], ast.Expr)
                        and isinstance(before.body[0].value, ast.Constant)
                        and isinstance(after.body[0].value, ast.Constant)
                        and isinstance(before.body[0].value.value, str)
                        and isinstance(after.body[0].value.value, str)):
                    before.body.pop(0)
                    after.body.pop(0)
                    if ast.dump(before) == ast.dump(after):
                        amendment = {"source": "survey/sol-tabular-ttc/replay-docstring-amendment.json",
                                     "sha256": C.sha256_bytes(amendment_path.read_bytes()),
                                     "revised_entry_hash": change["revised_entry_hash"],
                                     "scope": change["scope"]}
    if (proof["entry_hash"] != C.sha256_text(code) and amendment is None) or proof["seam"] != SEAM:
        return None, ["task adapter or seam differs from the tested real entry"]
    units = observed["whole_job_units"]
    if (observed["match"] != "exact" or observed["served"] != "result"
            or not observed["is_whole_job"] or not observed["result_matches_record"]
            or not units or units[0] != units[1] or observed["enqueue_count"] != units[0]):
        return None, ["retained real-entry evidence does not show complete exact replay"]
    evidence = {"source": "survey/sol-tabular-ttc/replay-proof.json", "sha256": C.sha256_bytes(path.read_bytes()),
            "entry_hash": proof["entry_hash"], "tested_record_id": proof["tested_record_id"],
            "runtime_hashes_reported": proof["runtime_hashes_reported"], "scope": proof["scope"]}
    if amendment is not None:
        evidence["docstring_amendment"] = amendment
    return evidence, []


def phase_units(fit, predict):
    total = fit + predict
    durations = []
    for phase in (fit, predict):
        if phase:
            n = max(1, math.ceil(phase / 60))
            durations.extend([phase / n] * n)
    shares = [d / total for d in durations]
    shares[-1] += 1 - sum(shares)
    return shares


def make_record(cell, acquisition, code):
    key = json.dumps([cell["arm"], cell["dataset"], cell["split"]], separators=(",", ":"))
    chash = C.sha256_text(key)
    rid = "sol-ttc-" + C.slug(cell["arm"]) + "-" + C.slug(cell["dataset"]) + f"-split{cell['split']}"
    argv = ["single_cell.py", "--dataset", cell["dataset"], "--split", str(cell["split"]), "--arm", cell["arm"]]
    metrics = {"metric": cell["metric"], "metric_error": cell["metric_error"],
               "time_train_s": cell["fit_s"], "time_infer_s": cell["predict_s"]}
    evidence, missing = replay_evidence(code)
    shares = phase_units(cell["fit_s"], cell["predict_s"])
    record = {
        "schema": C.SCHEMA_RECORD,
        "identity": {"record_id": rid, "kind": "eval", "entry_hash": C.sha256_text(code),
                     "argv": argv, "argv_provenance": "derived", "argv_how": "one prescribed source cell per task-owned launcher process",
                     "argv_exempt": ["--output"], "argv_exempt_how": "result file path only",
                     "env": {}, "seed": C.absent("split identity is explicit; source arm config fixes its own seed"),
                     "replicate_group": "sol-ttc-" + chash[:12], "match_policy": "exact_only",
                     "trajectory": {"config_hash": chash, "config_hash_how": "sha256 of [arm, dataset, split]",
                                    "environment": {"ttc_commit": acquisition["code_commit"],
                                                    "tabarena_commit": acquisition["tabarena_commit"]},
                                    "lr_schedule": {"identity": None, "source": None, "trace": None},
                                    "data": {"dataset": cell["dataset"], "split": cell["split"]},
                                    "precision": C.absent("not recorded per source CSV row")},
                     "fingerprint": None, "fingerprint_inputs": None, "consumes": []},
        "mode": "replay",
        "source": {"name": "Test-Time Compute for Tabular Foundation Models", "repo": "https://github.com/" + acquisition["code_repo"],
                   "commit": acquisition["code_commit"], "results_repo": "https://huggingface.co/datasets/" + acquisition["hf_repo"],
                   "results_revision": acquisition["hf_revision"],
                   "path": cell["source_csv"] + "#row=" + str(cell["source_csv_line"]),
                   "csv_sha256": cell["source_csv_sha256"], "licence": "results CC-BY-4.0; code Apache-2.0",
                   "source_row": cell["source_row"]},
        "reference": {"gpu": "NVIDIA H100 80GB HBM3", "gpu_provenance": "inferred",
                      "gpu_how": "source README explicitly states one 80 GB H100 per run; CSV has no per-row device log",
                      "gpu_count": 1, "gpu_count_provenance": "inferred", "interconnect": None, "precision": None},
        "launch": {"launcher": "python", "world_size": C.tv(1, "derived", "one source cell per single-GPU process"),
                   "per_device_microbatch": C.absent(), "accumulation_steps": C.absent(),
                   "global_batch_tokens": C.absent(), "batch_policy": C.absent("no width conversion"), "alternatives": {}},
        "steps": None, "planned_steps": None, "per_step_gpu_s": None,
        "timing": {"pacing": "total", "wall_total_s": C.tv(cell["fit_predict_s"], "derived",
                   "source time_train_s + time_infer_s. Original _run also computes CPU metric_error: that unrecorded CPU scoring is skipped and unpaced. Real CPU input/split loading stays outside the seam."),
                   "setup_s": C.absent("data/model setup not separately measured in source; per-job writable cache copy and data loading run for real outside the seam"),
                   "eval_s": C.absent(), "checkpoint_s": C.absent()},
        "memory": memory_estimate(cell["dataset"], cell["arm"])[0],
        "series": {},
        "materialisation": {"reads": ["final metric dict"], "needs_series": [], "blockers": missing,
                            "replay_evidence": evidence,
                            "serving": {"mode": "whole_job", "seam": SEAM, "n_units": len(shares),
                                        "unit": "subdivision of a measured fit or prediction phase", "unit_shares": shares,
                                        "result_key": "final_metrics", "result_kind": "metric dict", "result_shape": "dict",
                                        "how": "Original pinned TabArena function; final fields from source CSV. Within-phase equal subdivisions are modelled, each at most 60 s, preserving measured phase totals. No intermediate scores or prediction arrays are invented."}},
        "result": {"final_metrics": C.tv(metrics, "recorded", "source CSV row, numeric timing/metric columns parsed as floats")},
        "outcome": {"status": "completed", "provenance": "inferred", "how": "non-imputed source result row with finite measured times and error"},
        "failures": [], "artefacts": {"produces": []},
        "host": {"weights_bytes": C.absent(), "data_bytes": C.absent()},
        "agent_view": {"script": SCRIPT, "launch_command": "python " + " ".join(argv) + " --output results/cell.json",
                       "inputs": [cell["dataset"] + " prescribed split " + str(cell["split"])],
                       "recognisability": "standard tabular ML experiment code; source code and dataset names visible"},
    }
    record["readiness"] = C.readiness(record)
    return record


def derive(output_registry):
    output_registry = pathlib.Path(output_registry)
    by, _ = SOURCE.load()
    acquisition = json.loads((HERE / "source/acquisition.json").read_text())
    code = (HERE / "single_cell.py").read_text()
    C.write_text(str(output_registry / SCRIPT), code)
    records = []
    for tid, datasets in SOURCE.PACKAGES.items():
        proposal = SOURCE.candidate(tid, datasets, by, acquisition)
        for cell in proposal["cells"]:
            record = make_record(cell, acquisition, code)
            records.append(record)
    C.add_disclosure(records)
    for record in records:
        C.write_json(str(output_registry / "records" / (record["identity"]["record_id"] + ".json")), record)
    print(json.dumps({"source_cells": len(records), "replay_ready": sum(r["readiness"]["replay"] for r in records),
                      "script_sha256": hashlib.sha256(code.encode()).hexdigest(), "output": str(output_registry)}))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output_registry", type=pathlib.Path)
    derive(ap.parse_args().output_registry)


if __name__ == "__main__":
    main()
