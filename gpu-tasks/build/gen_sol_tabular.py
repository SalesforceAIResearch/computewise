"""Independent Sol's source-backed tabular adoption twins.

main derives records and artifacts into a scratch registry. build returns one
unfinished task, for finish_one_task/gen_tasks. Shared registry files are owned
by gpu-tasks. The retained representative real entry passed on native FakeGPU
with r8.1 source mounts; installed full-box runs remain separate measurements.
"""
import argparse
import csv
import json
import math
import pathlib
import shlex
import sys

HERE = pathlib.Path(__file__).resolve().parent
SOURCE_DIR = HERE.parent / "survey" / "sol-tabular-ttc"
sys.path.insert(0, str(HERE))
import common as C
import sol_ttc_candidates as S
import sol_ttc_records as R

ARTIFACT_ROOT = "artifacts/sol-tabular"
CONSTRAINTS = ARTIFACT_ROOT + "/environment/constraints.txt"
REFERENCE = ARTIFACT_ROOT + "/adoption_reference.py"
PACKAGES = ["torch==2.9.1", "autogluon==1.5.0"]


def job_id(cell):
    return C.slug(cell["dataset"]) + "--" + C.slug(cell["arm"]) + "--split" + str(cell["split"])


def write_support(reg):
    C.write_text(str(pathlib.Path(reg) / CONSTRAINTS), "\n".join(PACKAGES) + "\n")
    C.write_text(str(pathlib.Path(reg) / REFERENCE), (SOURCE_DIR / "adoption_reference.py").read_text())


def check_record(record, reg):
    """Validate source correspondence, measured phases and the actual script key."""
    if not record["identity"]["record_id"].startswith("sol-ttc-"):
        return []
    errors = []
    row = record["source"]["source_row"]
    expected_time = float(row["time_train_s"]) + float(row["time_infer_s"])
    if row["imputed"].lower() != "false":
        errors.append("imputed source cell")
    if expected_time < 1200 or not math.isclose(record["timing"]["wall_total_s"]["value"], expected_time):
        errors.append("duration differs from measured fit + predict or is under 20 minutes")
    metrics = record["result"]["final_metrics"]["value"]
    for key in ("metric_error", "time_train_s", "time_infer_s"):
        if metrics[key] != float(row[key]):
            errors.append(key + " differs from source CSV")
    serving = record["materialisation"]["serving"]
    shares = serving["unit_shares"]
    if serving["seam"] != R.SEAM or serving["n_units"] != len(shares) or not math.isclose(sum(shares), 1):
        errors.append("invalid original-library seam/phase units")
    if any(x <= 0 or x * expected_time > 60.000001 for x in shares):
        errors.append("unit is nonpositive or longer than 60 seconds")
    script = pathlib.Path(reg) / record["agent_view"]["script"]
    if not script.exists() or C.sha256_bytes(script.read_bytes()) != record["identity"]["entry_hash"]:
        errors.append("script bytes differ from entry_hash")
    csv_path = SOURCE_DIR / record["source"]["path"].split("#row=")[0]
    if C.sha256_bytes(csv_path.read_bytes()) != record["source"]["csv_sha256"]:
        errors.append("source CSV differs from pinned digest")
    line = int(record["source"]["path"].split("#row=")[1])
    source_rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    if line < 2 or line - 2 >= len(source_rows) or source_rows[line - 2] != row:
        errors.append("retained source row does not match the addressed CSV line")
    if metrics["metric"] != row["metric"] or shares != R.phase_units(float(row["time_train_s"]), float(row["time_infer_s"])):
        errors.append("metric identity or phase subdivision differs from the source")
    argv = record["identity"]["argv"]
    expected_memory, _ = R.memory_estimate(row["dataset"], argv[argv.index("--arm") + 1])
    if record["memory"] != expected_memory:
        errors.append("modelled memory differs from pinned architecture/shape arithmetic")
    if record["readiness"]["replay"]:
        evidence, missing = R.replay_evidence(script.read_text()) if script.exists() else (None, ["script absent"])
        if missing or record["materialisation"].get("replay_evidence") != evidence:
            errors.append("replay readiness is not bound to the retained real-entry evidence")
    return errors


def derive(reg):
    records = R.derive(reg)
    write_support(reg)
    for record in records:
        errors = check_record(record, reg)
        if errors:
            raise ValueError(record["identity"]["record_id"] + ": " + "; ".join(errors))
    print("TTC source correspondence: every derived record checked against retained CSV and script bytes")


def build(recs, disclosure_of, reg=None, twin="science", adoption=True):
    if twin not in ("science", "operations") or reg is None:
        raise ValueError("twin must be science/operations and registry is required")
    tid = "pkg-sol-tabular-" + twin
    by, _ = S.load()
    acquisition = json.loads((SOURCE_DIR / "source/acquisition.json").read_text())
    proposal = S.candidate(tid, S.PACKAGES[tid], by, acquisition)
    if twin == "operations":
        proposal["brief"] = proposal["brief"].replace("scientific assay datasets", "operational prediction datasets")
    adoption_spec = S.adoption_spec(proposal)
    if adoption:
        tid += "-adoption"
    code = (SOURCE_DIR / "single_cell.py").read_text()
    jobs, files, profiles, cells, record_ids = [], [], [], [], []
    for cell in proposal["cells"]:
        rid = R.make_record(cell, acquisition, code)["identity"]["record_id"]
        record = recs[rid]
        if errors := check_record(record, reg):
            raise ValueError(rid + ": " + "; ".join(errors))
        jid = job_id(cell)
        script, result = "jobs/" + jid + ".sh", "results/" + jid + ".json"
        argv = ["python", "tools/single_cell.py", *record["identity"]["argv"][1:], "--output", result]
        artifact = ARTIFACT_ROOT + "/jobs/" + jid + ".sh"
        C.write_text(str(pathlib.Path(reg) / artifact), "#!/bin/sh\nset -eu\nexec " + shlex.join(argv) + "\n")
        files.append({"path": script, "from": artifact})
        jobs.append({"job_id": jid, "label": f"{cell['dataset']}, {cell['arm']}, prescribed split {cell['split']}",
                     "replicate_group": record["identity"]["replicate_group"], "records": [rid],
                     "draw": "open" if adoption else "without_replacement", "seed": None,
                     "gpu_options": [1], "recorded_gpus": 1, "recorded_gpu": record["reference"]["gpu"],
                     "batch_policy": None, "launch": "bash " + script if adoption else shlex.join(argv),
                     "after": [], "consumes": [], "produces": [result]})
        identity = {"job_id": jid, "dataset": cell["dataset"], "arm": cell["arm"], "split": cell["split"],
                    "script": script, "result": result}
        cells.append(identity)
        profiles.append({**identity, "fit_s": cell["fit_s"], "predict_s": cell["predict_s"],
                         "fit_predict_s": cell["fit_predict_s"]})
        record_ids.append(rid)
    if len({recs[r]["source"]["path"] for r in record_ids}) != len(jobs):
        raise ValueError("package repeats an underlying source cell")
    manifest = {"datasets": proposal["datasets"], "arms": list(S.ARMS), "splits": sorted({c["split"] for c in cells})}
    if adoption:
        manifest["minimum_relative_error_reduction"] = S.MIN_RELATIVE_ERROR_REDUCTION
    # Include the adapter bytes so its agent-visible text participates in the
    # task hash; a shared artifact path alone would keep the old staging key.
    files += [{"path": "tools/single_cell.py", "content": code},
              {"path": "datasets.json", "content": json.dumps(manifest, indent=2) + "\n"},
              {"path": "cost_profiles.json", "content": json.dumps({"cells": profiles,
                "provenance": "Published fit and prediction measurements for these configurations on one 80GB H100; excludes process startup and data loading. Supplied equally to agent and reference."}, indent=2) + "\n"},
              {"path": "README.md", "content": "# Paired tabular prediction study\n\n"
                "datasets.json lists the datasets, the two versions and the prescribed split IDs. Each jobs/*.sh launch runs one "
                "dataset/split/version on one GPU and writes its metric_error, metric and fit/prediction times to its own results/*.json. "
                "Split IDs are the source study's flattened fold/repeat indices. Nine splits define each version's mean error on a dataset; "
                "lower error is better. Error units differ by dataset. The original code and raw OpenML inputs are installed locally. "
                "cost_profiles.json contains the published fit/prediction cost profiles.\n"}]
    write_support(reg)
    task = {"task_id": tid, "rung": 3 if adoption else 1, "goal_level": adoption,
            "set": "development", "set_how": "Independent Sol search; disjoint scientific and operational source-cell cohorts",
            "title": "Paired tabular adaptation adoption" if adoption else "Paired tabular adaptation results table",
            "brief": {"text": adoption_spec["brief"] if adoption else proposal["brief"],
                      "provenance": "Independent Sol: goal and dataset-specific error rule; no schedule or deadline; criterion constructed during source inspection, disclosed"},
            "files": files, "jobs": jobs,
            "output_isolation": {"flag": "--output", "exempt": "each wrapper already writes its unique results/<cell>.json; --output is argv-exempt"},
            "environment": {"python": C.tv("3.11", "inferred", "built from python:3.11-slim-bookworm; resolved exact version in image"),
                "install": {"packages": PACKAGES, "constraints": CONSTRAINTS,
                            "how": "Dockerfile.probe and build_environment.py build pinned patched source stack; source imports passed on c3-15 2026-11-02. Includes AutoGluon1.5.0 and torch2.9.1 CPU."},
                "hub": [], "hub_bytes": 0, "env": {"OPENML_CACHE_DIR": "/opt/cache/openml", "HF_HUB_OFFLINE": "1"},
                "verified": "source imports and all eight datasets' nine input splits loaded on c3-15; real805f Bioresponse baseline entry served82/82 units and recorded metrics on native FakeGPU/torch2.9.1 with r8.1 source mounts, uid1000 and read-only original cache, actual adapter copy, no cache shim or TabPFN3 weights. Representative fixture only; installed full-box task runs are separate. Evidence: survey/sol-tabular-ttc/replay-proof.json",
                "image_label": {"value": "sol-tabular-ttc-r8", "how": "experiment label retained; requires tested r8.1 patch4c17e9d5 and sidecar2da8801c. Native-image entry probe passed; full-box build/staging tracked by isolation and gpu-tasks"}},
            "sources": ["https://github.com/" + acquisition["code_repo"] + "@" + acquisition["code_commit"],
                        "https://huggingface.co/datasets/" + acquisition["hf_repo"] + "@" + acquisition["hf_revision"]],
            "recorded_outcome": proposal["answer_from_source"],
            "tabular_adoption": {**manifest, "cells": cells,
                "rule": "adapted_mean_error <= baseline_mean_error * (1-minimum_relative_error_reduction) on every dataset",
                "reject_proof": "all prescribed cells for both arms of any failing dataset completed and their paired means reported",
                "accept_proof": "all prescribed cells for both arms of every dataset completed and each paired mean passes",
                "alternative_proofs": "every valid failing-dataset proof is accepted; reference dataset is not required",
                "avoidable_work": "paced GPU work after the earliest completed valid rejection proof, including speculative tails; establishing that first proof is legitimate whichever failing dataset it uses. Pre-proof speculative work reported separately."},
            "notes": ["Original single-H100 source cells, non-imputed, each at least20 minutes fit+predict; no card or width conversion.",
                      "Hardware inferred from explicit author statement; not a per-row device log.",
                      "Original _run seam also performs CPU scoring; its unrecorded time is skipped and unpaced. Input loading and splitting run for real outside seam.",
                      "Complete table controls share source cells with adoption variant, but twins share no source cells.",
                      "Potential job tally exceeds40; reference's rejection proof requires18. No claim that all potential jobs are necessary."]}
    if adoption:
        keys = {tuple(k) for k in adoption_spec["reference_trace"]["required_source_keys"]}
        selected = [j for j, c in zip(jobs, proposal["cells"]) if (c["arm"], c["dataset"], c["split"]) in keys]
        task["reference_plan"] = {"instances": [{"job_id": j["job_id"], "count": 1, "width": 1} for j in selected],
            "policy": "controller", "how": adoption_spec["reference_trace"]["policy"]}
        task["scripted_reference"] = {"path": "reference/tabular_adoption.py", "from": REFERENCE,
            "how": "scripted-only observed-result controller; reads same manifests/profiles as agent, no registry or source score tables"}
        task["reference_arithmetic"] = adoption_spec["reference_trace"]
        task["horizon_override"] = {"minimum_s": C.colleague_round(1.5 * sum(c["fit_predict_s"] for c in proposal["cells"])),
            "how": "full potential pool serial GPU time times1.5, rounded; allows even an agent that produces the entire table serially to finish. Ordinary reference-based terms and unmeasured startup remain disclosed. No fabricated host measurements."}
    else:
        task.pop("tabular_adoption")
        # These launches already name different result files. Generic scripted
        # policies must keep those paths, rather than add --output_dir isolation.
        task["output_isolation"] = None
        task["notes"][-1] = "All listed source cells are required by the full-table brief; this is a matched control for the adoption decision, not a new source cohort."
        task["tabular_table"] = {**manifest, "cells": cells,
            "completion_rule": "every prescribed cell completed; report each cell's error and each arm's per-dataset mean",
            "comparison": "lower mean error is better within each dataset; error units differ between datasets",
            "work_scope": "all listed cells are owed; an early adoption rejection never makes table work unneeded"}
    return task


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("registry")
    derive(ap.parse_args().registry)
