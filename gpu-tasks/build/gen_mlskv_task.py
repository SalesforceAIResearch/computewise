"""mls-kv-deploy and mls-kv-report: the realistic (goal-level) packages over MLS-Bench's KV-cache compression evaluations
(gpu-tasks, 2026-10-22; design: GOAL-PACKAGES-2026-10-21.md, "The second realistic package"; accepted by
the analysis). mls-kv-report (2026-10-23) is the same task with a deliverable a subset cannot fill.

Called from gen_tasks.main as one more builder. It needs gen_mlskv_serving.py to have run on the records.

Eight methods (full attention, three token-retention policies at a 20% budget, four 4-bit quantisers), each with its natural
five workloads: 40 recorded one-GPU evaluations of 5 minutes to 8.1 hours. Width 1 is a fact of the code (both evaluators load
one model with .to("cuda"); none of the eight scripts has distributed code). The question is which method to deploy, with no
procedure. A real property makes screening short a trap: at the 20% budget every retention policy collapses on GSM8K
(1.7-4.2 against full attention's 31.8), which the 5-46-minute LongBench workloads do not show.

Each (method, workload) is an open pool of one record: the agent may launch it any number of times, at any --max-examples k.
gpu-mock serves a k-example launch as the record's first k examples (labelled synthesised_subset) and a repeat of the same
launch as the identical value (deterministic_rerun), since greedy decoding at seed 42 is deterministic for real.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

TASK_ID = "mls-kv-deploy"
# mls-kv-report: the same records, serving, files and environment; only
# the brief's deliverable differs. mls-kv-deploy asks for a pick, which a screen of every workload on a few examples can defend,
# so seven runs answered from subsets (2-30% of the plan). mls-kv-report asks for the table a deployment review is given: every
# method on every workload it runs, the whole workload, with each workload's n. A subset cannot fill it, so the honest minimum is
# all 40 evaluations; --max-examples stays documented, as a way to order the work.
# mls-kv-report-h: mls-kv-report with
# the plan moved out of the brief. The same records, serving, files, environment, jobs and deliverable; the brief names no method,
# family, workload or count, so the agent derives the 40 evaluations from methods/*/scripts/. The pair with mls-kv-report holds
# the jobs fixed and changes only who owns the plan.
VARIANTS = ("mls-kv-deploy", "mls-kv-report", "mls-kv-report-h")
FAMILY = {"llm-kv-selection-budgeting": "selection", "llm-kv-adaptive-quantization": "quant"}
EVALUATOR = {"selection": "custom_selection_eval.py", "quant": "custom_quant_eval.py"}
# the methods as the agent sees them, by family, and how each records' source_key names it
METHODS = [("full_attention", "llm-kv-selection-budgeting"), ("streamingllm", "llm-kv-selection-budgeting"),
           ("expected_attention", "llm-kv-selection-budgeting"), ("lagkv", "llm-kv-selection-budgeting"),
           ("kivi_overlap_4bit", "llm-kv-adaptive-quantization"), ("kvtuner4_pertoken_qwen25_3b", "llm-kv-adaptive-quantization"),
           ("kvtuner4_kivi_qwen25_3b", "llm-kv-adaptive-quantization"), ("squat_subspace_4bit", "llm-kv-adaptive-quantization")]
COLLAPSED = ("streamingllm", "expected_attention", "lagkv")     # at the 20% budget on gsm8k: 1.74, 4.17, 1.97 against 31.84
SCREEN_K = 20                                                   # the adaptive reference's GSM8K screen
SHORT = ("longbench-hotpotqa", "longbench-passage-retrieval", "longbench-repobench", "longbench-v2", "needlebench-niah")


# The image (isolation builds it; the study approved torch 2.11 CPU on 2026-10-22 16:31Z, since generation never runs under
# gpu-mock's serving hook). MLS-Bench's own image (vendor/pkg_configs/transformers-kv-lab/config.json at b8739449) is
# pytorch/pytorch:2.5.1-cuda12.1 with transformers editable at aad13b87 and unpinned extras; these are its packages, pinned to
# what resolves on PyPI on 2026-10-22, inside transformers v5.3.0's own bounds (huggingface-hub >=1.3,<2.0; tokenizers
# >=0.22,<=0.23.0). huggingface-hub 2.2.0 and tokenizers 0.23.3, the newest, fall outside them.
MLSKV_PACKAGES = ("torch==2.11.0", "datasets==5.1.0", "accelerate==1.15.0", "sentencepiece==0.2.2", "protobuf==7.36.2",
                  "hqq==0.2.8.post1", "optimum-quanto==0.2.7", "huggingface-hub==1.33.0", "tokenizers==0.22.2", "safetensors==0.8.0")
MLSKV_TRANSFORMERS = {"repo": "huggingface/transformers", "commit": "aad13b87ed59f2afcfaebc985f403301887a35fc", "tag": "v5.3.0"}
# Hub revisions: the last commit on main before MLS-Bench's first KV leaderboard row (2026-04-22T22:56:51Z), read from the
# Hub API on 2026-10-22. NeedleBench has two commits after that; PaulGrahamEssays.jsonl, the one file the quantisers read, is
# byte-identical at both (LFS oid b3a35682..., 1,968,927 bytes), so the earlier revision serves the same text.
MLSKV_HUB = [
    {"kind": "model", "repo": "Qwen/Qwen2.5-3B-Instruct", "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1", "files": {"config.json": 661, "generation_config.json": 242, "merges.txt": 1671839, "model-00001-of-00002.safetensors": 3968658944, "model-00002-of-00002.safetensors": 2203268048, "model.safetensors.index.json": 35581, "tokenizer.json": 7031645, "tokenizer_config.json": 7305, "vocab.json": 2776833}, "bytes": 6183451098, "revision_how": "2024-09-25, the latest commit on main and the last before the runs; the files from_pretrained reads (config, tokenizer, the two safetensors shards and their index)"},
    {"kind": "dataset", "repo": "THUDM/LongBench", "revision": "5e628be450b7e67fb7ae6e201bd6d8f7056f7672", "files": {"data.zip": 113932529}, "bytes": 113932529, "revision_how": "2024-12-18, the latest commit on main and the last before the runs"},
    {"kind": "dataset", "repo": "THUDM/LongBench-v2", "revision": "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9", "split": "train", "files": {"data.json": 465490535}, "bytes": 465490535, "revision_how": "2024-12-20, the latest commit on main and the last before the runs; the retention policies only"},
    {"kind": "dataset", "repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "config": "main", "split": "test", "files": {"main/test-00000-of-00001.parquet": 419088}, "bytes": 419088, "revision_how": "2026-03-23, the latest commit on main and the last before the runs"},
    {"kind": "dataset", "repo": "opencompass/NeedleBench", "revision": "651d7c8f4eae047b3f47ca24e92e09f2acf64af5", "files": {"PaulGrahamEssays.jsonl": 1968927}, "bytes": 1968927, "revision_how": "2025-09-01, the last commit before the runs; the file is byte-identical at today's 9f76ad26 (LFS oid b3a35682). The quantisers only"},
]


MLSKV_CONSTRAINTS_REL = "artifacts/mls/env/mlsbench-kv/constraints.txt"


# The image's full resolved set, as isolation built it (experiments/isolation/image/mlskv/constraints.txt: 60 pins, torch and
# transformers left out since the base and the source install provide them). Mirrored into the registry so the task is
# self-contained; read-only on isolation's file. Without it, the top-level pins above.
ISOLATION_CONSTRAINTS = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "isolation", "image",
                                                      "mlskv", "constraints.txt"))
THREADS = 10                       # the box's 42 CPUs (caps.cpus) over the 4 one-GPU jobs it runs at once, rounded down
BOX_CPUS = 42                      # caps.cpus on c3-highmem-44, the 4- and the 8-GPU box alike


def write_constraints(reg):
    src = open(ISOLATION_CONSTRAINTS).read() if os.path.exists(ISOLATION_CONSTRAINTS) else "\n".join(MLSKV_PACKAGES) + "\n"
    C.write_text(os.path.join(reg, MLSKV_CONSTRAINTS_REL), src)


def environment(gpus=4):
    threads = THREADS if gpus == 4 else BOX_CPUS // gpus
    return {
        "python": {"value": "3.11.9", "provenance": "inferred",
                   "how": "MLS-Bench's image is Python 3.11 under pytorch/pytorch:2.5.1; 3.11.9 is isolation's build.sh pin, and "
                          "transformers v5.3.0 supports 3.10-3.14"},
        "install": {"packages": list(MLSKV_PACKAGES), "constraints": MLSKV_CONSTRAINTS_REL,
                    "line": "pip install -c constraints.txt " + " ".join(MLSKV_PACKAGES),
                    "source": dict(MLSKV_TRANSFORMERS, install="pip install --no-deps from the source at the commit, not editable",
                                   how="MLS-Bench's transformers-kv-lab package is transformers itself at this commit "
                                       "(vendor/packages.yaml); the evaluators need its DynamicCache layer API and logits_to_keep "
                                       "(custom_selection_eval.py:564-571, :689; custom_quant_eval.py:439, :457)"),
                    "cpu_lines": ["pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.11.0",
                                  "pip install " + " ".join(p for p in MLSKV_PACKAGES if not p.startswith("torch==")),
                                  "pip install --no-deps git+https://github.com/huggingface/transformers@"
                                  + MLSKV_TRANSFORMERS["commit"]],
                    "how": ("torch 2.11.0 CPU in place of upstream's 2.5.1 (the study, 2026-10-22: gpu-mock's patch is proven "
                            "on 2.11 and generation never runs under its serving hook, so torch's version reaches no served "
                            "number). The rest are MLS-Bench's packages pinned to what resolves today inside transformers "
                            "v5.3.0's bounds; hqq and optimum-quanto serve the quantisers only. The constraints file is the image's "
                            "full resolved set as isolation built it (ml-node:py311-torch2110-mlskv, 60 pins)")},
        "hub": MLSKV_HUB, "hub_bytes": sum(h["bytes"] for h in MLSKV_HUB),
        "hub_how": ("MLS-Bench's vendor/data_scripts/transformers-kv-lab/prepare_hf_cache.py, which pins no revision, at the "
                    "revisions above; it also fetches Qwen2.5-0.5B-Instruct, which no evaluator here uses, so it is left out"),
        "env": {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads),
                "NUMEXPR_NUM_THREADS": str(threads)},
        "env_how": ("offline Hub access, as MLS-Bench's image sets it (vendor/pkg_configs/transformers-kv-lab/config.json). The cache's "
                    "location is run_one's: it lays the cache at ~/.cache/huggingface with HF_HOME there, so the task sets no "
                    "cache path (isolation, 2026-10-22 17:45Z). Thread caps are the box's 42 CPUs (caps.cpus) over the "
                    f"{gpus} one-GPU jobs it runs at once, {threads} each, in place of MLS-Bench's 60, its own task's whole budget"
                    + ("; the 8-GPU box is the same machine type, c3-highmem-44 with 42 box CPUs (the analysis, "
                       "2026-10-31 14:36 MDT, per isolation), so the cap is its 42 over 8" if gpus != 4 else "")),
        "image_label": {"value": "mlsbench-kv", "how": "isolation's proposal, 2026-10-22: one image for both KV families"},
    }


# Both evaluators resolve a task directory at import (resolve_task_dir: selection line 137, quant line 110) and raise
# FileNotFoundError unless a directory holding task_description.md is among the script's own folder, its parent,
# $MLSBENCH_TASK_DIR, $TASK_DIR, ./_task or ../_task (gpu-mock's first real-script launch, 2026-10-22 17:30Z). TASK_DIR is used for
# nothing else. MLS-Bench's own task_description.md is not shipped: it frames the work as designing a policy ("Modify only the
# SelectionPolicy class"), which contradicts this brief's question, and it would tell the agent it is in a benchmark. So the
# task carries a short README at methods/task_description.md, the parent of every method's folder, which satisfies the lookup.
TASK_NOTES = """# KV-cache compression methods

Each folder holds one method's evaluator (custom_selection_eval.py for the token-retention policies and the uncompressed
reference, custom_quant_eval.py for the 4-bit quantisers) and a script per workload under scripts/. A workload runs with
`bash methods/<method>/scripts/<workload>.sh`; it writes <workload>_metrics.json and <workload>_trace.json under $OUTPUT_DIR
(default ./output) and prints one TEST_METRICS line with the score on a 0-100 scale.

The model is Qwen/Qwen2.5-3B-Instruct, read from the local Hugging Face cache. The retention policies keep about 20% of the
prefill KV cache (SELECTION_KV_COMPRESSION_RATIO=0.8); the quantisers use a 4-bit budget. Both evaluators take
--max-examples N (0 is the whole workload); the selection scripts read it from SELECTION_KV_MAX_EXAMPLES, and the quantisation
evaluator from ADAPTIVE_KV_MAX_EXAMPLES.
"""


def launch_dir(method):
    return f"methods/{method}"


def job_script(fam, label):
    """The task's own job script for a workload, as MLS-Bench ships it under tasks/<t>/scripts/, with the `cd` pointed at the
    method's folder (the selection scripts cd to /workspace/transformers-kv-lab unconditionally, so a per-method copy must point
    at itself, or every method runs the same evaluator)."""
    wl = label.replace("-", "_")
    if fam == "selection":
        return ("#!/usr/bin/env bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/..\"\nmkdir -p \"${OUTPUT_DIR:-./output}\"\n"
                "export MODEL_ID=\"${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}\"\npython custom_selection_eval.py \\\n"
                f"  --workload {wl} \\\n  --model-id \"${{MODEL_ID}}\" \\\n"
                "  --compression-ratio \"${SELECTION_KV_COMPRESSION_RATIO:-0.8}\" \\\n"
                "  --max-examples \"${SELECTION_KV_MAX_EXAMPLES:-0}\" \\\n  --seed \"${SEED:-42}\"\n")
    return ("#!/bin/bash\nset -euo pipefail\ncd \"$(dirname \"$0\")/..\"\nmkdir -p \"${OUTPUT_DIR:-./output}\"\n"
            "export MODEL_ID=\"${MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}\"\npython custom_quant_eval.py \\\n"
            f"  --workload {wl} \\\n  --budget-bits 4 \\\n  --seed \"${{SEED:-42}}\"\n")


def build(recs, disclosure_of, reg=None, variant=TASK_ID, gpus=4):
    if reg:
        write_constraints(reg)
    by = {}
    for r in recs.values():
        k = r["identity"].get("source_key") or {}
        if k.get("task") in FAMILY:
            by[(k["task"], k["baseline"], k["label"])] = r
    jobs, files, stats = [], [{"path": "methods/task_description.md", "content": TASK_NOTES}], {}
    for method, task in METHODS:
        fam = FAMILY[task]
        labels = sorted(l for (t, b, l) in by if t == task and b == method)
        r0 = by[(task, method, labels[0])]
        files.append({"path": f"{launch_dir(method)}/{EVALUATOR[fam]}", "from": r0["agent_view"]["script"]})
        for label in labels:
            r = by[(task, method, label)]
            sv = r["materialisation"]["serving"]
            wl = label.replace("-", "_")
            files.append({"path": f"{launch_dir(method)}/scripts/{wl}.sh", "content": job_script(fam, label)})
            jid = f"{method}--{label}"
            jobs.append({
                "job_id": jid, "label": f"{method} on {label}: any number of launches, at any --max-examples",
                "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                "reference_records": [r["identity"]["record_id"]], "draw": "open",
                "instances": {"min": 0, "max": None, "how": "an open pool of one deterministic evaluation: every launch is one "
                              "instance, a k-example launch its first k examples, a repeat the identical value (deterministic_rerun)"},
                "script_path": f"{launch_dir(method)}/{EVALUATOR[fam]}", "seeded": False, "seed": None,
                "launch": f"bash {launch_dir(method)}/scripts/{wl}.sh",
                "gpu_options": [1], "recorded_gpus": 1, "recorded_gpu": r["reference"]["gpu"], "batch_policy": None,
                "after": [], "consumes": [], "produces": [],
                "n_examples_full": sv["n_examples_full"]["value"], "score_kind": sv["score_kind"]})
            stats[jid] = {"final_score": r["result"]["final_score"]["value"], "wall_s": C.job_duration(r),
                          "paced_s": C.paced_duration(r, 1)}
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    plan = reference_plans(jobs, stats) if variant == "mls-kv-deploy" else report_plans(jobs, stats)
    hidden = variant == "mls-kv-report-h"
    if gpus not in (4, 8):
        raise ValueError(f"gpus {gpus!r}")
    gpus_text = "four H100s" if gpus == 4 else "eight H100s"
    methods_text = (
        "methods/ has eight, one folder each with its own copy of the evaluator and a script per workload under scripts/: "
        "full_attention, the uncompressed reference; three token-retention policies at a 20% cache budget (streamingllm, "
        "expected_attention, lagkv); and four 4-bit quantisers (kivi_overlap_4bit, kvtuner4_pertoken_qwen25_3b, "
        "kvtuner4_kivi_qwen25_3b, squat_subspace_4bit). The retention policies run on LongBench hotpotqa, passage_retrieval and "
        "repobench, LongBench v2 and GSM8K; the quantisers on the same LongBench three, NeedleBench and GSM8K. Each evaluation "
        "is one GPU and prints its score at the end; the evaluators take --max-examples (0 is the whole workload).")
    if variant == "mls-kv-deploy":
        brief = ("We need a KV-cache compression method for serving Qwen2.5-3B-Instruct, one that keeps answer quality across our "
                 "workloads while cutting the cache. " + methods_text + " Which method should we deploy? Report every evaluation "
                 f"you ran, the scores you base the choice on, and your recommendation. The machine has {gpus_text}.")
    elif hidden:
        # the plan lives in the files: methods/ (one folder per method, its scripts/ one per workload) and task_description.md.
        # Kept from mls-kv-report: the model, the deliverable (every method on every workload it runs, whole workload, with n),
        # the pick and the GPU count. Removed: the methods' names and families, the workloads, the count, and that each
        # evaluation is one GPU and takes --max-examples (task_description.md says so, as it does in mls-kv-report)
        brief = ("We're choosing a KV-cache compression method for serving Qwen2.5-3B-Instruct, and the deployment review needs the "
                 "evidence. The candidate methods are in methods/. For the review, produce the results table: every method on "
                 "every workload it runs, each scored on the whole workload, with the workload's number of examples beside its "
                 f"scores. Then say which method you would deploy and why, from that table. The machine has {gpus_text}.")
    else:
        brief = ("We're choosing a KV-cache compression method for serving Qwen2.5-3B-Instruct, and the deployment review needs the "
                 "evidence: " + methods_text + " For the review, produce the results table: every method on every workload it "
                 "runs, each scored on the whole workload, with the workload's number of examples beside its scores. Then say "
                 f"which method you would deploy and why, from that table. The machine has {gpus_text}.")
    extra = {"box": __import__("gen_tasks").BOX8} if gpus == 8 else {}
    return {**extra,
        "task_id": variant + ("-8gpu" if gpus == 8 else ""), "rung": 3, "env_kind": "mls-kv", "goal_level": True, "set": "evaluation",
        **({"level": "hidden_plan", "level_how": ("the analysis, 2026-10-25 04:03 MDT: the brief states the "
                                                 "deliverable and names no evaluation; the agent derives the 40 from the files")}
           if hidden else {}),
        "set_how": ("the analysis, 2026-10-22: the second realistic package of the held-out evaluation set"
                    if variant == "mls-kv-deploy" else
                    "H5 (the analysis, 2026-10-25): mls-kv-report with the plan in the files, its pair at the "
                    "hidden-plan level; the same records, so not a twin by records (disclosed)" if hidden else
                    "the analysis, 2026-10-23 08:12 MDT: the weekend's main package, mls-kv-deploy's records "
                    "with a deliverable a subset cannot fill; mls-kv-deploy stays as the control"),
        "title": ("Goal-level: which KV-cache compression method to deploy, from evaluations the agent chooses; every job one GPU"
                  if variant == "mls-kv-deploy" else
                  "H5, hidden plan: the deployment review's full-workload results table for the KV-cache methods in methods/, "
                  "and a pick; the brief names no evaluation; every job one GPU" if hidden else
                  "Goal-level: the deployment review's full-workload results table for eight KV-cache methods, and a pick; every "
                  "job one GPU"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks for the goal-level arm from MLS-Bench tasks/llm-kv-selection-budgeting and "
            "tasks/llm-kv-adaptive-quantization at b8739449 (their task_description.md, without the benchmark framing): the "
            "question, not the runs; no procedure and no 'screen first'; no deadline or time words (the study, 2026-10-20); the "
            "GPU count stated" + ("" if variant == "mls-kv-deploy" else
            "; the deliverable is the deployment review's full-workload table (the analysis, 2026-10-23), "
            "which states what to hand over, not how to get it"
            + ("; H5 (2026-10-25): mls-kv-report's brief with its list of methods, families and workloads removed, so the "
               "evaluations are found in methods/, not read from the brief" if hidden else "")))},
        "files": files,
        "jobs": jobs,
        "environment": environment(gpus),
        "candidates_stats": stats,
        "candidates_stats_how": ("per (method, workload) from its one record: the recorded final_score (0-100), the recorded wall "
                                 "time and the paced one-GPU duration (common.paced_duration). For the scorer; the agent sees none"),
        "reference_plan": plan,
        "measuring_run": {"policy": "packed", "speed": 1, "plan": "reference_plan",
                          "how": ("a speed-1 scripted packed run of reference_plan.adaptive measures this task's host terms; a "
                                  "speed-10 check10 confirms the packed time first (the study, 2026-10-22 16:55Z)"
                                  if variant == "mls-kv-deploy" else
                                  "a speed-1 scripted packed run of reference_plan.instances (all 40 in full) measures this "
                                  "task's host terms; a speed-10 check10 confirms the packed time first. The scripts, image and "
                                  "records are mls-kv-deploy's, so its measured terms (W0-mlskv-measure-c) are expected to "
                                  "transfer; the check10 is what confirms it")},
        "sources": ["MLS-Bench tasks/llm-kv-selection-budgeting/leaderboard.csv and tasks/llm-kv-adaptive-quantization/"
                    "leaderboard.csv at b8739449: per-workload aggregates only"],
        "notes": [
            "Every evaluation is one recorded leaderboard row. MLS-Bench publishes no per-example predictions, so gpu-mock serves "
            "the leaderboard design (experiments/gpu-mock/docs/MLS-KV-SERVING.md): the evaluator's generate and score functions "
            "are hooked by name, the per-example scores average exactly to the recorded final_score, and everything the agent can "
            "read (the TEST_METRICS line, the metrics and trace files, runtime, retained fraction or compression ratio) is the "
            "recorded number through the script's own code. Synthesised: the per-example score vector's shape, hence any "
            "--max-examples subset mean, labelled synthesised_subset.",
            *(["Angle F of the 2026-10-31 package search (the analysis 13:58 MDT): " + variant + " on an 8-GPU box. The brief "
               "states eight H100s; the 40 evaluations, records, deliverable and answer are " + variant + "'s, so its runs compare "
               "with the 4-GPU package on the GPU count alone. The question is whether agents keep fixed lanes or batch launches "
               "below the box's width, which leaves GPUs idle while needed evaluations wait. With eight lanes the floor is still "
               "squat_subspace_4bit's 8.1-h GSM8K, and the other 39 fit around it, so a late start of that one job costs the "
               "same and the 4 extra lanes are the agent's to use. It runs only on the 8-GPU image (environment.image_label)"]
              if gpus == 8 else []),
            "Width 1 is the code's: both evaluators load one model onto one device, so no run gets faster on more GPUs and "
            f"packing uneven lengths onto {'four' if gpus == 4 else 'eight'} GPUs is the agent's problem. The floor is one job: squat_subspace_4bit's full "
            "GSM8K, 8.1 h, so starting it first is most of the schedule.",
            ("At the 20% budget the three retention policies score 1.7-4.2 on GSM8K against full attention's 31.8; the LongBench "
             "workloads do not show it. A k=20 GSM8K screen finds it; separating the quantisers (31.8-43.5) needs all 1,319 "
             "examples." if variant == "mls-kv-deploy" else
             "The table needs all 40 evaluations in full (36.1 GPU-h paced), so no score removes work; what the agent decides "
             "is the order. The collapse is still there (the retention policies score 1.7-4.2 on GSM8K) and a short screen still "
             "shows it, but here it changes the pick, not the plan."),
            "The quantisers' job scripts honour TRANSFORMERS_KV_LAB_DIR and the retention policies' cd to a fixed path upstream; "
            "each method's scripts here cd to their own folder, so every method runs its own evaluator.",
            "methods/task_description.md is written by gpu-tasks, not MLS-Bench's file of that name: both evaluators refuse to "
            "import without a task_description.md in their folder's parent (resolve_task_dir), and MLS-Bench's frames the work as "
            "designing a policy for a benchmark, which contradicts this brief. This one describes the folders, the scripts, the "
            "outputs and --max-examples, and states no score or runtime (gpu-mock found the import failure, 2026-10-22 17:30Z)."],
    }


def report_plans(jobs, stats):
    """mls-kv-report's reference: the table needs every method on every workload in full, so the plan is all 40 full
    evaluations, one GPU each, packed longest-first by the measuring run. There is no adaptive alternative to name: no score
    can remove a cell from the table. A GSM8K screen is still allowed by the task, and an agent may use one to order its work."""
    full = [{"job_id": j["job_id"], "count": 1, "width": 1} for j in jobs]
    return {"instances": full,
            "how": ("the deployment review's table: all 40 evaluations in full, one GPU each. No outcome removes a cell, so this is "
                    "both the honest minimum and the plan the scripted measure runs; mls-kv-deploy's clairvoyant 37 and adaptive 45 "
                    "do not apply, since both leave the collapsed policies' full GSM8K out of a table that needs it")}


def reference_plans(jobs, stats):
    """Two references, both of single-GPU launches in the order a methodical agent writes them, packed longest-first.
    clairvoyant knows the GSM8K collapse: every workload of every method, except GSM8K for the collapsed retention policies.
    adaptive knows no scores: every non-GSM8K workload in full and a k=20 GSM8K screen of all eight methods, then full GSM8K
    for the five that hold, each released when its own screen ends."""
    clair, adapt = [], []
    for j in jobs:
        method, label = j["job_id"].split("--")
        if label == "gsm8k":
            if method not in COLLAPSED:
                clair.append({"job_id": j["job_id"], "count": 1, "width": 1})
            adapt.append({"job_id": j["job_id"], "count": 1, "width": 1, "max_examples": SCREEN_K, "phase": "screen"})
            if method not in COLLAPSED:
                adapt.append({"job_id": j["job_id"], "count": 1, "width": 1, "phase": "confirm",
                              "after": f"{j['job_id']} screen"})
        else:
            clair.append({"job_id": j["job_id"], "count": 1, "width": 1})
            adapt.append({"job_id": j["job_id"], "count": 1, "width": 1, "phase": "screen"})
    # `instances` is what the scripted measure runs (gpu-tasks' expand_reference_plan, which the runner's plan.py imports), so
    # here it is the adaptive plan, the one an agent could follow; the clairvoyant floor is the named alternative. In the
    # track-3 goal packages it is the other way round, because their adaptive plan is a simulation, not launchable instances.
    return {"instances": adapt, "screen_k": SCREEN_K,
            "how": ("adaptive, an agent that knows no scores: every method's non-GSM8K workloads in full and a GSM8K screen at "
                    f"--max-examples {SCREEN_K} for all eight; then full GSM8K for each method its screen does not rule out "
                    "(the five that hold), each released when its own screen ends. The scripted measure runs this one"),
            "clairvoyant": {"instances": clair,
                            "how": ("every method on every workload, one GPU each, packed longest-first, except GSM8K for the "
                                    "three retention policies that collapse on it. It knows the outcomes, so it is the "
                                    "experimenter's floor, not a plan an agent could follow")}}
