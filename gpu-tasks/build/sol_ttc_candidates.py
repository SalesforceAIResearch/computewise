"""Build source-backed TTC package specifications; never execute source code.

These are candidate specifications, not canonical registry tasks or a claim that
the simulator adapter/environment has been staged. Every observation is copied
from a pinned, non-imputed CSV row. Runtime is measured fit + prediction only.
"""
import collections
import csv
import hashlib
import json
import math
import pathlib
import statistics

HERE = pathlib.Path(__file__).resolve().parents[1] / "survey" / "sol-tabular-ttc"
ARMS = ("tabpfn3_aggregation_96", "tabpfn3_diagscale_aggregation_96")
PACKAGES = {
    "pkg-sol-tabular-science": ("Bioresponse", "QSAR-TID-11", "physiochemical_protein"),
    "pkg-sol-tabular-operations": ("APSFailure", "GiveMeSomeCredit", "bank-marketing",
                                   "credit_card_clients_default", "customer_satisfaction_in_airline"),
}
MIN_RELATIVE_ERROR_REDUCTION = 0.02


def load():
    by = {}
    summary = {}
    for p in sorted((HERE / "source/hf/raw").glob("*/results_per_split.csv")):
        arm = p.parent.name
        raw = p.read_bytes()
        rows = list(csv.DictReader(raw.decode().splitlines()))
        clean = []
        for line, row in enumerate(rows, 2):
            if row["imputed"].lower() != "false":
                continue
            fit, pred, metric = (float(row[k]) for k in ("time_train_s", "time_infer_s", "metric_error"))
            if not all(math.isfinite(x) for x in (fit, pred, metric)) or min(fit, pred) < 0 or fit + pred <= 0:
                raise ValueError(f"invalid measured cell at {p}:{line}")
            key = (arm, row["dataset"], int(row["fold"]))
            if key in by:
                raise ValueError(f"duplicate measured cell {key}")
            item = {"arm": arm, "dataset": row["dataset"], "split": int(row["fold"]),
                    "metric": row["metric"], "metric_error": metric, "problem_type": row["problem_type"],
                    "fit_s": fit, "predict_s": pred, "fit_predict_s": fit + pred,
                    "source_csv": str(p.relative_to(HERE)), "source_csv_sha256": hashlib.sha256(raw).hexdigest(),
                    "source_csv_line": line, "source_row": row}
            by[key] = item
            clean.append(item)
        long = [r for r in clean if r["fit_predict_s"] >= 1200]
        summary[arm] = {"rows": len(rows), "non_imputed_rows": len(clean), "at_least_20_minutes": len(long),
                        "long_datasets": dict(sorted(collections.Counter(r["dataset"] for r in long).items()))}
    return by, summary


def candidate(tid, datasets, by, acquisition):
    cells = []
    for dataset in datasets:
        splits = [set(split for arm, name, split in by if arm == a and name == dataset) for a in ARMS]
        if splits[0] != splits[1] or not splits[0]:
            raise ValueError(f"unpaired source cells for {dataset}")
        for split in sorted(splits[0]):
            for arm in ARMS:
                cell = by[arm, dataset, split]
                if cell["fit_predict_s"] < 1200:
                    raise ValueError(f"{tid}: source cell below 20 minute floor: {arm, dataset, split}")
                cells.append(cell)
    result = {}
    for dataset in datasets:
        result[dataset] = {}
        for arm in ARMS:
            own = [c for c in cells if c["dataset"] == dataset and c["arm"] == arm]
            result[dataset][arm] = {"n": len(own), "metric": own[0]["metric"],
                                   "mean_metric_error": statistics.mean(c["metric_error"] for c in own),
                                   "mean_fit_predict_s": statistics.mean(c["fit_predict_s"] for c in own)}
        result[dataset]["lower_mean_error_arm"] = min(ARMS, key=lambda a: result[dataset][a]["mean_metric_error"])
    durations = [c["fit_predict_s"] for c in cells]
    return {"schema": "p006-sol-source-package/1", "task_id": tid,
            "state": "source-backed specification; whole_job adapter and environment not yet staged",
            "box": {"gpus": 4, "gpu": "H100 80 GB"},
            "brief": "We are comparing our 96-configuration tabular prediction ensemble with the version that also adapts "
                     "the backbone using DiagScale. For each of the scientific assay datasets, we need the paired results "
                     "over the prescribed splits: each cell's prediction error, each version's mean error, and the "
                     "recommendation for that dataset. Use the dataset's own error metric; errors from different datasets "
                     "have different units. The machine has four H100s.",
            "datasets": list(datasets), "arms": list(ARMS), "cells": cells,
            "summary": {"gpu_jobs": len(cells), "min_s": min(durations), "median_s": statistics.median(durations),
                        "max_s": max(durations), "gpu_hours": sum(durations) / 3600,
                        "gpu_work_lower_bound_s": sum(durations) / 4,
                        "basis": "measured fit + prediction; excludes data loading, CPU scoring and process launch"},
            "answer_from_source": result,
            "provenance": {"acquisition": acquisition,
                           "hardware_basis": "source authors explicitly state one 80 GB H100 per run; not a per-row device log",
                           "hardware_provenance": "inferred", "card_conversion": "none",
                           "results_license": "CC-BY-4.0", "code_license": "Apache-2.0"},
            "adapter": {"mode": "whole_job", "seam": "tabarena.benchmark.experiment.experiment_runner:ExperimentRunner._run",
                        "entry_script": "single_cell.py", "unit": "subdivision of a measured fit or prediction phase",
                        "unit_shares": "split each phase into equal subunits of at most 60 seconds; preserve both measured phase totals",
                        "result_key": "final_metrics", "result_shape": "dict",
                        "source_split_semantics": "CSV fold is a flattened split index; use TabArena task_grid split, not fold directly",
                        "first_cuda_use": "inside _run; gpu-mock was asked for import-time seam installation in r8",
                        "cpu_scoring": "included in original _run seam, but not separately timed in source and not paced; disclosed omission",
                        "result_fields": "recorded final metric and fit/predict times; no prediction arrays fabricated"},
            "remaining_work": [
                "Stage and box-test single_cell.py against pinned TTC and TabArena source.",
                "Install original _run seam before its first call (model first reaches CUDA inside that function).",
                "Keep real CPU data loading and split preparation outside the seam; disclose the skipped, untimed CPU scoring within _run.",
                "Resolve TabPFN-3 weight access and pinned patched-library environment if real model initialization is required outside the seam.",
                "Generate canonical records/tasks and run W0 and agent screens after r8 is on a box.",
                "Measure agent behaviour: source diversity and job count alone do not establish any waste bar."]}


def adoption_spec(proposal):
    """Trace a brief-only policy using published cost profiles, never unseen scores.

    The simulator here needs durations and outcomes to evaluate the trace. The
    policy uses durations from the profile the agent also receives and observes
    each dataset's scores only after both arms' prescribed splits complete.
    """
    p = json.loads(json.dumps(proposal))
    p["task_id"] += "-adoption"
    p["brief"] = ("We are deciding whether to add DiagScale adaptation to our 96-configuration tabular prediction ensemble. "
                  "Our adoption criterion requires at least a 2% reduction in mean prediction error on every dataset in "
                  "datasets.json, using each dataset's nine prescribed splits and its own error metric. Should we adopt "
                  "the adapted version? Report the paired means your decision rests on and your recommendation. "
                  "The machine has four H100s.")
    order = sorted(p["datasets"], key=lambda d: (sum(c["fit_predict_s"] for c in p["cells"] if c["dataset"] == d), d))
    required = []
    visited = []
    rejected_on = None
    ref_s = 0.0
    lower_bounds = {}
    for d in order:
        own = [c for c in p["cells"] if c["dataset"] == d]
        durations = [c["fit_predict_s"] for c in own]
        # All eighteen observations are needed to report the two nine-split means.
        baseline = p["answer_from_source"][d][ARMS[0]]["mean_metric_error"]
        adapted = p["answer_from_source"][d][ARMS[1]]["mean_metric_error"]
        if adapted > baseline * (1 - MIN_RELATIVE_ERROR_REDUCTION):
            # Nonpreemptive, one-GPU cells: at least one lane holds ceil(n/4)
            # complete jobs. Its load is at least that many shortest durations.
            cardinality_bound = sum(sorted(durations)[:math.ceil(len(durations) / 4)])
            lower_bounds[d] = max(sum(durations) / 4, max(durations), cardinality_bound)
    for d in order:
        own = sorted((c for c in p["cells"] if c["dataset"] == d), key=lambda c: (-c["fit_predict_s"], c["arm"], c["split"]))
        lanes = [0.0] * 4
        for cell in own:
            lane = lanes.index(min(lanes))
            lanes[lane] += cell["fit_predict_s"]
        ref_s += max(lanes)
        required.extend((c["arm"], c["dataset"], c["split"]) for c in own)
        visited.append(d)
        baseline = p["answer_from_source"][d][ARMS[0]]["mean_metric_error"]
        adapted = p["answer_from_source"][d][ARMS[1]]["mean_metric_error"]
        if adapted > baseline * (1 - MIN_RELATIVE_ERROR_REDUCTION):
            rejected_on = d
            break
    p["adoption_rule"] = {"minimum_relative_error_reduction": MIN_RELATIVE_ERROR_REDUCTION,
                          "reject_proof": "both complete nine-split means on any one dataset fail the criterion",
                          "accept_proof": "both complete nine-split means on every dataset satisfy the criterion",
                          "alternative_rejection_proofs_are_valid": True,
                          "selection_disclosure": "criterion constructed during source inspection, not preregistered"}
    lb = min(lower_bounds.values()) if lower_bounds else p["summary"]["gpu_work_lower_bound_s"]
    p["reference_trace"] = {"dataset_order": order, "datasets_observed": visited,
                            "required_source_keys": required, "reject_on": rejected_on,
                            "fit_predict_only_makespan_s": ref_s, "rejection_proof_lower_bound_s": lb,
                            "reference_over_relaxed_bound": ref_s / lb,
                            "policy": "lowest published paired cost first; complete its paired prescribed means using longest-profile-first queue; stop on observed failure",
                            "duration_profile_available_to_agent": True,
                            "basis": "arithmetic using published cost profiles and source outcomes, not an agent run; CPU loading, launch and scoring absent"}
    p["reference_trace"]["lower_bound_basis"] = "minimum over valid failing-dataset proofs of max(total work/4, longest cell, sum of ceil(n/4) shortest cells); nonpreemptive one-GPU cells"
    p["remaining_work"].append("Scorer must accept any valid failing-dataset proof, not require the reference's chosen dataset or call alternative evidence unneeded.")
    return p


def main(output_directory=None):
    output = pathlib.Path(output_directory) if output_directory else HERE
    output.mkdir(parents=True, exist_ok=True)
    by, summary = load()
    acquisition = json.loads((HERE / "source/acquisition.json").read_text())
    (output / "inventory.json").write_text(json.dumps(summary, indent=2) + "\n")
    all_keys = set()
    for tid, datasets in PACKAGES.items():
        out = candidate(tid, datasets, by, acquisition)
        keys = {(c["arm"], c["dataset"], c["split"]) for c in out["cells"]}
        if all_keys & keys:
            raise ValueError("the specifications share recorded cells")
        all_keys |= keys
        if tid.endswith("operations"):
            out["brief"] = out["brief"].replace("scientific assay datasets", "operational prediction datasets")
        (output / (tid + ".json")).write_text(json.dumps(out, indent=2) + "\n")
        print(tid, json.dumps(out["summary"]))
        adoption = adoption_spec(out)
        (output / (adoption["task_id"] + ".json")).write_text(json.dumps(adoption, indent=2) + "\n")
        print(adoption["task_id"], json.dumps(adoption["reference_trace"]))


if __name__ == "__main__":
    main()
