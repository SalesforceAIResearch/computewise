"""Registry records from MLS-Bench's published baseline runs (Imbernoulli/MLS-Bench).

usage: python3 -I gen_mlsbench.py <MLS-Bench clone> <registry dir> [task ...]

One record per (task, baseline, seed, job label) with a recorded `elapsed_<label>`.
Durations are whole-job wall time and results are final metrics; no curve is recorded.
Default tasks: every llm-pretrain-* task and the two KV-cache evaluation tasks.
Contract: ../RECORD-FORMAT.md.
"""
import collections
import csv
import json
import os
import re
import shlex
import subprocess
import ast
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REPO = "https://github.com/Imbernoulli/MLS-Bench"
# GPT-2 medium as custom_pretrain.py builds it: 24 layers, width 1024, vocab 50304, block 1024, no biases, tied embedding.
# 12 x 24 x 1024^2 + 50304 x 1024 + 1024 x 1024 = 355.5M parameters, saved as fp32 model_state_dict only (no optimizer state).
CKPT_BYTES_345M = (12 * 24 * 1024 ** 2 + 50304 * 1024 + 1024 * 1024) * 4
CKPT_HOW = ("355.5M fp32 parameters of the task's fixed GPT-2 medium (12 x 24 x 1024^2 + 50304 x 1024 + 1024^2); "
            "custom_pretrain.py saves model_state_dict and model_args only; norm gains add under 0.1%")
FIXED_ARCH = ("llm-pretrain-optimizer", "llm-pretrain-lr-schedule", "llm-pretrain-loss")
DEFAULT_TASKS = ("llm-kv-selection-budgeting", "llm-kv-adaptive-quantization")


def hms(s):
    p = [int(x) for x in s.split(":")]
    while len(p) < 3:
        p = [0] + p
    return p[0] * 3600 + p[1] * 60 + p[2]


def sh_defaults(sh):
    """`${NAME:-default}` and NAME=value assignments in a job script."""
    env = {}
    for m in re.finditer(r"\b([A-Z_][A-Z0-9_]*)=\$\{\1:-([^}]*)\}", sh):
        env[m.group(1)] = m.group(2)
    for m in re.finditer(r"(?:^|\s)([A-Z_][A-Z0-9_]*)=([0-9][0-9.e\-]*)(?=\s|\\|$)", sh, re.M):
        env.setdefault(m.group(1), m.group(2))
    return env


def compile_flag(clone, task):
    """compile_model in the task's own scaffold copy of custom_pretrain.py."""
    p = os.path.join(clone, "harbor", "tasks-modal", f"mls-bench__{task}", "environment", "_scaffold", "nanoGPT", "custom_pretrain.py")
    if not os.path.exists(p):
        return None
    m = re.search(r"^\s*compile_model\s*=\s*(True|False)", open(p).read(), re.M)
    return (m.group(1) == "True") if m else None


CPU_METRIC_BLOCKER = ("gpu-mock serving: the final metric is a CPU-side number; gpu-mock has no generate()-level short-circuit and no "
                      "serving point for a CPU-side final metric yet (gpu-mock, 2026-10-19; expected to move, see RECORD-FORMAT.md §11)")


def workload_max_new_tokens(src, task):
    """{workload: max_new_tokens} from WORKLOAD_CONFIGS in the task's evaluator in the Harbor scaffold; empty if not found."""
    import glob
    out = {}
    for p in glob.glob(os.path.join(src, "harbor", "tasks-docker", f"mls-bench__{task}", "environment", "_scaffold", "*", "custom_*_eval.py")):
        text = open(p).read()
        m = re.search(r"WORKLOAD_CONFIGS\s*=\s*\{(.*?)\n\}", text, re.S)
        if not m:
            continue
        for name, body in re.findall(r'"([a-z0-9_]+)":\s*\{(.*?)\n\s*\}', m.group(1), re.S):
            n = re.search(r'"max_new_tokens":\s*(\d+)', body)
            if n:
                out[name] = int(n.group(1))
    return out


LINEAGE_BLOCKER = ("identity: the variants' evaluations share script bytes, argv and non-exempt env and differ only by the checkpoint they "
                   "load; gpu-mock has to resolve the loaded checkpoint to the job that wrote it (lineage) before it can pick this record "
                   "(gpu-tasks, 2026-10-19; not yet raised with gpu-mock, because the MLS-Bench tasks are deferred)")
TRACE_PRINT = re.compile(r'\n    print\(\n        "TRACE_METRICS: "\n(?:        .*\n)*?    \)\n')
SIBLING_NAMES = (("the same public text workloads as llm-kv-adaptive-quantization.", "the same public text workloads as the quantisation evaluator."),)
ENTRY_FILES = (("custom_selection_eval.py", "transformers-kv-lab/custom_selection_eval.py"),
               ("custom_quant_eval.py", "transformers-kv-lab/custom_quant_eval.py"),
               ("custom_pretrain.py", "nanoGPT/custom_pretrain.py"),
               ("nanogpt_lm_eval.py", "lm-evaluation-harness/nanogpt_lm_eval.py"))


def entry_for(sh):
    """The file the job's python process runs, as a path under the scaffold, or None."""
    for token, rel in ENTRY_FILES:
        if token in sh:
            return rel
    return None


def load_ops(path):
    """OPS from an MLS-Bench edit file without executing it: string constants, names bound to them, lists and dicts.
    Every edit file of the four tasks is of that shape; anything else raises and the record stays unmaterialised."""
    tree = ast.parse(open(path).read(), filename=path)
    names = {}

    def ev(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in names:
                return names[node.id]
            raise ValueError(f"{path}: unbound name {node.id}")
        if isinstance(node, ast.List):
            return [ev(e) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return tuple(ev(e) for e in node.elts)
        if isinstance(node, ast.Dict):
            return {ev(k): ev(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"{path}: unsupported expression {type(node).__name__} at line {node.lineno}")

    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            try:
                names[st.targets[0].id] = ev(st.value)
            except ValueError:
                pass
    if "OPS" not in names:
        raise ValueError(f"{path}: OPS could not be read without executing the file")
    return names["OPS"]


def apply_ops(text, ops, rel_file, path):
    """Apply MLS-Bench edit ops in sequence, as harbor_adapter._stage_task_scaffold and the oracle's solve.sh do:
    1-indexed lines, end_line -1 = end of file, content gets a trailing newline, later ops see earlier ones' result."""
    lines = text.splitlines(keepends=True)
    for op in ops:
        if op.get("file") != rel_file:
            raise ValueError(f"{path}: op targets {op.get('file')!r}, not the entry file {rel_file!r}")
        kind = op["op"]
        content = op.get("content", "")
        if content and not content.endswith("\n"):
            content += "\n"
        new = content.splitlines(keepends=True)
        if kind == "replace":
            s, e = int(op["start_line"]) - 1, int(op["end_line"])
            e = len(lines) if e == -1 else e
            lines = lines[:s] + new + lines[e:]
        elif kind == "insert":
            a = int(op.get("after_line", 0))
            lines = lines[:a] + new + lines[a:]
        elif kind == "delete":
            s, e = int(op["start_line"]) - 1, int(op.get("end_line", op["start_line"]))
            lines = lines[:s] + lines[e:]
        elif kind == "create":
            lines = op.get("content", "").splitlines(keepends=True)
        else:
            raise ValueError(f"{path}: unknown op {kind!r}")
    return "".join(lines)


def materialise(clone, tdir, task, entry_rel, edit_rel):
    """(bytes the job's python process runs, how) for a baseline: the Harbor scaffold's entry file with the baseline's edit ops
    applied, then the TRACE_METRICS print removed where present. None if the scaffold file is missing."""
    scaffold = os.path.join(clone, "harbor", "tasks-docker", f"mls-bench__{task}", "environment", "_scaffold", entry_rel)
    if not os.path.exists(scaffold):
        return None, f"no scaffold file {entry_rel} under harbor/tasks-docker/mls-bench__{task}"
    code = open(scaffold).read()
    how = [f"harbor/tasks-docker/mls-bench__{task}/environment/_scaffold/{entry_rel} at the MLS-Bench commit"]
    if edit_rel:
        ops = load_ops(os.path.join(tdir, edit_rel))
        mine = [op for op in ops if op.get("file") == entry_rel]
        if mine:
            code = apply_ops(code, mine, entry_rel, edit_rel)
            how.append(f"with the {len(mine)} edit op(s) of {edit_rel} applied by line, as harbor_adapter does")
        if len(mine) != len(ops):
            how.append(f"{len(ops) - len(mine)} op(s) of {edit_rel} target other files and were not applied")
    code, n = TRACE_PRINT.subn("\n", code)
    if n:
        how.append("then the TRACE_METRICS print, which named the benchmark task id, removed at the sandbox's request (2026-10-19)")
    for old_text, new_text in SIBLING_NAMES:
        if old_text in code:
            code = code.replace(old_text, new_text)
            how.append(f"and the docstring's {old_text.split()[-1].rstrip('.')!r}, a benchmark task id, renamed to {new_text.split(' as ')[-1].rstrip('.')!r}")
    return code, "; ".join(how)


def shell_values(sh):
    """Variable assignments in a job script with their defaults resolved: NAME="${NAME:-x}" gives x, NAME="${OTHER}/y" follows OTHER.
    Variables the script never assigns stay as ${NAME} placeholders."""
    vals = {}
    for m in re.finditer(r'^(?:export\s+)?([A-Z_][A-Z0-9_]*)=("?)([^\s"]*)\2\s*$', sh, re.M):
        name, val = m.group(1), m.group(3)
        val = re.sub(r"\$\{%s:-([^}]*)\}" % name, r"\1", val)
        vals[name] = substitute(val, vals)
    return vals


def substitute(s, vals):
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\$\{([A-Z_][A-Z0-9_]*):-([^}]*)\}", lambda m: vals.get(m.group(1), m.group(2)), s)
        s = re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: vals[m.group(1)] if m.group(1) in vals else m.group(0), s)
    return s


def python_launch(sh):
    """(argv of the python process the job script starts, with the script's own defaults substituted, basename first;
    env the script sets on that process's command line). Values the script leaves to the caller keep a ${NAME} placeholder."""
    vals = shell_values(sh)
    joined = re.sub(r"\\\n\s*", " ", sh)
    for m in re.finditer(r"^((?:[A-Z_][A-Z0-9_]*=\S+\s+)*)(python3?|torchrun)\s+(.*)$", joined, re.M):
        if ".py" not in m.group(3):
            continue
        env = dict(re.findall(r"([A-Z_][A-Z0-9_]*)=(\S+)", m.group(1)))
        env = {k: substitute(v.strip('"'), vals) for k, v in env.items()}
        args = shlex.split(substitute(m.group(3), vals))
        while args and not args[0].endswith(".py"):
            args.pop(0)
        if not args:
            continue
        return [os.path.basename(args[0])] + args[1:], env
    return None, {}



def main():
    clone, reg = sys.argv[1], sys.argv[2]
    tasks_root = os.path.join(clone, "tasks")
    names = sys.argv[3:] or sorted(t for t in os.listdir(tasks_root) if t.startswith("llm-pretrain-")) + list(DEFAULT_TASKS)
    commit = subprocess.run(["git", "-C", clone, "log", "-1", "--format=%H"], capture_output=True, text=True).stdout.strip()
    out_dir = os.path.join(reg, "records")
    recs, counts = [], collections.Counter()
    for t in names:
        tdir = os.path.join(tasks_root, t)
        cfg = json.load(open(os.path.join(tdir, "config.json")))
        rows = list(csv.DictReader(open(os.path.join(tdir, "leaderboard.csv"))))
        cmds = {x["label"]: x for x in cfg["test_cmds"]}
        for row in rows:
            if row.get("is_final") != "true":
                counts["skipped: not a final row"] += 1
                continue
            base = row["model"].replace("baseline:", "")
            seed = row["seed"]
            edit_rel = cfg.get("baselines", {}).get(base, {}).get("edit_ops")
            edit_bytes = open(os.path.join(tdir, edit_rel)).read() if edit_rel and os.path.exists(os.path.join(tdir, edit_rel)) else ""
            for label, x in cmds.items():
                ev = row.get(f"elapsed_{label}")
                if ev in (None, ""):
                    counts["skipped: no elapsed time for the job"] += 1
                    continue
                sh = open(os.path.join(tdir, x["cmd"])).read()
                env = sh_defaults(sh)
                metrics = {k[: -len(label) - 1]: float(v) for k, v in row.items()
                           if k and k.endswith("_" + label) and not k.startswith("elapsed_") and v not in ("", None)}
                is_train = "torchrun" in sh or "custom_pretrain.py" in sh
                gpus = x.get("compute")
                ws = int(gpus) if gpus and float(gpus).is_integer() else gpus
                steps = int(env["MAX_ITERS"]) if is_train and env.get("MAX_ITERS") else None
                bsz = int(env["BATCH_SIZE"]) if env.get("BATCH_SIZE") else None
                ga = int(env["GRAD_ACCUM"]) if env.get("GRAD_ACCUM") else None
                gbt = bsz * ga * 1024 if (is_train and bsz and ga) else None
                elapsed = float(ev)
                status = "completed" if metrics else "failed"
                basis = json.dumps({"task": t, "edit": C.sha256_text(edit_bytes), "script": C.sha256_text(sh), "cmd": x, "commit": commit}, sort_keys=True)
                cfg_hash = C.sha256_text(basis)
                ckpt = re.search(r'CKPT_PATH="\$\{OUTPUT_DIR\}/(ckpt_[^"]+)"', sh)
                lr_id = f"cosine-with-warmup:lr={env.get('LEARNING_RATE', '6e-4 (script default)')}" if is_train else None
                if is_train:
                    needs, blockers = ["train_loss", "val_loss"], []
                    serving = {"mode": "per_op", "how": "the script prints the train loss every 10 iterations and estimate_loss() every "
                               "EVAL_INTERVAL; no curve was recorded, so those reads need a modelled curve before replay"}
                    reads = ["loss.item() every 10 iterations on the master rank", "estimate_loss(): 200 x .item() per split every EVAL_INTERVAL",
                             "final validation, WikiText-2 and LAMBADA perplexity from per-chunk .item()"]
                elif "nanogpt_lm_eval.py" in sh:
                    needs = []
                    blockers = [CPU_METRIC_BLOCKER]
                    serving = {"mode": "results_at_end",
                               "how": "lm-eval scores each answer choice by log-likelihood: the forward must return full [B, T, V] logits "
                                      "and the accuracy is a CPU-side number no tensor hook sees. Served from result.* at the final print "
                                      "once gpu-mock has a serving point for a CPU-side final metric; duration paced from wall_total_s",
                               "final_print": "lm-eval's results table on stdout; the script passes no --output_path, so no file is written",
                               "results_file": {"path": None, "how": "none written"},
                               "scoring": {"tasks": ["hellaswag", "arc_easy", "piqa", "winogrande"], "num_fewshot": 0, "batch_size": 1,
                                           "how": "nanogpt_lm_eval.py arguments in the task's lm_eval_*.sh"}}
                    reads = ["per-choice log-likelihood from full-vocabulary logits, one .item() per request"]
                else:
                    needs = []
                    blockers = [CPU_METRIC_BLOCKER]
                    wl = re.search(r"--workload\s+(\S+)", sh)
                    wl = wl.group(1) if wl else None
                    mnt = workload_max_new_tokens(clone, t).get(wl)
                    serving = {"mode": "results_at_end",
                               "how": "the job decodes text with generate() and scores the strings on the CPU: its final metric is a Python "
                                      "number no tensor hook sees, and generate() expects [B, T, V] logits every step. Served from result.* "
                                      "at the job's final print once gpu-mock has a generate()-level short-circuit and a serving point for a "
                                      "CPU-side final metric; duration paced from wall_total_s",
                               "final_print": "one stdout line `TEST_METRICS: final_score=... runtime_seconds=...`, which MLS-Bench's parser.py reads",
                               "results_file": {"path": "$OUTPUT_DIR/<workload>_metrics.json", "format": "json", "keys": sorted(metrics),
                                                "how": "maybe_write_output_artifacts() in the evaluator writes <workload>_metrics.json and "
                                                       "<workload>_trace.json when OUTPUT_DIR is set; the task's scripts set it (default ./output). "
                                                       "`keys` are the recorded ones; the file holds a few more (latency, efficiency)"},
                               "decode": {"workload": wl,
                                          "max_new_tokens": C.tv(mnt, "recorded", "WORKLOAD_CONFIGS in the evaluator") if mnt else C.absent("workload not in the evaluator's table"),
                                          "examples": C.absent("not recorded: the whole split unless SELECTION_KV_MAX_EXAMPLES is set"),
                                          "generated_tokens": C.absent("per-example lengths were not recorded; a placeholder is bounded by max_new_tokens")}}
                    reads = ["per-token argmax .item() during decoding"]
                rid = f"mls-{t}-{C.slug(base)}-{C.slug(label)}-s{seed}"
                entry_rel = entry_for(sh)
                code, code_how = materialise(clone, tdir, t, entry_rel, edit_rel) if entry_rel else (None, "no known entry file in the job script")
                script_rel, sha = None, None
                if code is not None:
                    sha = C.sha256_text(code)
                    script_rel = f"artifacts/mls/scripts/{sha[:12]}/{os.path.basename(entry_rel)}"
                    out = os.path.join(reg, script_rel)
                    if not os.path.exists(out):
                        os.makedirs(os.path.dirname(out), exist_ok=True)
                        open(out, "w").write(code)
                    counts["materialised"] += 1
                else:
                    counts["not materialised: " + code_how] += 1
                argv, argv_env = python_launch(sh)
                if argv is None:
                    argv, argv_how = ["bash", x["cmd"]], "test_cmds[].cmd in the task's config.json; the python command line could not be read from the job script"
                else:
                    argv_how = ("the python command line in the job script (" + x["cmd"] + ") with the script's own defaults substituted; "
                                "values the caller supplies keep a ${NAME} placeholder")
                argv_exempt = sorted({argv[i - 1] for i in range(1, len(argv)) if "${" in argv[i] and argv[i - 1].startswith("--")}
                                     | ({"--model-id"} if "--model-id" in argv else set()))
                if "nanogpt_lm_eval.py" in sh:
                    blockers = blockers + [LINEAGE_BLOCKER]
                rec = {
                    "schema": C.SCHEMA_RECORD,
                    "identity": {
                        "record_id": rid, "kind": "train" if is_train else "eval",
                        "entry_hash": sha,
                        "entry_hash_how": code_how,
                        "argv": argv, "argv_provenance": "derived", "argv_how": argv_how,
                        "argv_exempt": argv_exempt,
                        "argv_exempt_how": "flags whose values are paths the caller chooses (a checkpoint under OUTPUT_DIR) or the model's "
                                           "location: a local path to the same weights is the staged form, and the fingerprint's model and "
                                           "parameter count carry that identity",
                        "env": {"SEED": seed, **argv_env}, "env_how": "SEED from the leaderboard row; the rest is what the job script sets on the python command line, defaults substituted",
                        "seed": C.tv(int(seed) if seed.isdigit() else seed, "recorded", "leaderboard row"),
                        "source_key": {"task": t, "baseline": base, "label": label},
                        "replicate_group": "mls-cfg-" + cfg_hash[:12],
                        "match_policy": "exact_only",
                        "match_policy_how": "no curve or LR trace was recorded; replay needs the exact scaffold-plus-edit bytes",
                        "env_exempt": ["OUTPUT_DIR", "ENV", "MODEL_ID", "TRANSFORMERS_KV_LAB_DIR", "DATA_ROOT", "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE"],
                        "env_exempt_how": "variables the job scripts read for output and cache paths, not for what they compute; a per-job "
                                          "OUTPUT_DIR is how the agent keeps jobs from writing one directory",
                        "trajectory": {"config_hash": cfg_hash, "config_hash_how": "sha256 of task, baseline edit bytes, job script, job declaration and repo commit",
                                       "environment": {"mls_bench_commit": commit, "package": x.get("package")},
                                       "lr_schedule": {"identity": lr_id, "source": "get_lr in custom_pretrain.py, fixed by the task" if is_train else None, "trace": None},
                                       "data": {"dataset": "ClimbMix train.bin/val.bin (nvidia/Nemotron-ClimbMix, GPT-2 tokens)" if is_train else
                                                (env.get("MODEL_ID") or "") + " on the job's workload", "how": "the job script and its prepare script"},
                                       "precision": C.tv("bf16", "recorded", "dtype in the scaffold"),
                                       "global_batch_tokens": gbt},
                        "fingerprint": None,
                        "fingerprint_inputs": ({"model": "gpt", "num_layers": int(env["N_LAYER"]) if env.get("N_LAYER") else None,
                                                "model_dim": int(env["N_EMBD"]) if env.get("N_EMBD") else None, "vocab_size": 50304,
                                                "optimizers": None, "micro_batch_shape": [bsz, 1024] if bsz else None, "world_size": ws,
                                                "global_batch_tokens": gbt, "precision": "bf16", "torch_compile": compile_flag(clone, t),
                                                "num_parameters": None,
                                                "num_parameters_how": "fp256/0.3 field 11, null: the scaffold's GPT varies with the variant's "
                                                                      "edit, and these records are exact_only, so the structural index "
                                                                      "never serves them"}
                                               if is_train else None),
                        "consumes": [{"artefact": ckpt.group(1), "producer_label": ckpt.group(1)[5:-3]}] if ckpt else [],
                    },
                    "mode": "replay",
                    "source": {"name": "MLS-Bench baseline leaderboard", "repo": REPO, "commit": commit, "task": t,
                               "path": f"tasks/{t}/leaderboard.csv", "row_timestamp": row["timestamp"], "licence": "none declared", "read_on": C.READ_ON},
                    "reference": {"gpu": "NVIDIA H100 (variant not stated)", "gpu_provenance": "inferred",
                                  "gpu_how": "the README says tasks without an h200 block 'keep their H100 baseline' and the default GPU type is H100",
                                  "gpu_count": ws, "gpu_count_provenance": "recorded", "interconnect": None, "precision": "bf16"},
                    "launch": {
                        "launcher": "torchrun --nproc_per_node=<visible GPUs>" if is_train else "python",
                        "world_size": C.tv(ws, "recorded", "`compute` in config.json"),
                        "per_device_microbatch": C.tv(bsz, "recorded", "BATCH_SIZE in the job script") if bsz else C.absent(),
                        "accumulation_steps": C.tv(ga // ws if (ga and isinstance(ws, int) and ws and ga % ws == 0) else None, "derived",
                                                   "GRAD_ACCUM divided by the world size inside custom_pretrain.py") if ga else C.absent(),
                        "global_batch_tokens": C.tv(gbt, "derived", "BATCH_SIZE x GRAD_ACCUM x 1024") if gbt else C.absent(),
                        "batch_policy": C.tv("fixed_global_grow_accumulation", "derived", "the scaffold divides GRAD_ACCUM by the world size")
                                        if is_train else C.absent("single-GPU evaluation"),
                        "alternatives": {"h200": x.get("h200")} if x.get("h200") else {},
                    },
                    "steps": steps, "planned_steps": steps,
                    "per_step_gpu_s": round(elapsed / steps, 6) if steps else None,
                    "per_step_how": "elapsed / MAX_ITERS: an upper bound, since elapsed includes setup, compile and evaluation" if steps else None,
                    "timing": {
                        "pacing": "total",
                        "wall_total_s": C.tv(elapsed, "recorded",
                                             "elapsed_<label> in the leaderboard, equal to the job's own runtime_seconds, which it measures "
                                             "from just before loading data and model"
                                             if metrics.get("runtime_seconds") is not None and abs(metrics["runtime_seconds"] - elapsed) < 1
                                             else "elapsed_<label> in the leaderboard: the harness's wall clock around the whole command"),
                        "train_time_s": C.absent("not recorded separately"), "setup_s": C.absent(), "eval_s": C.absent(), "checkpoint_s": C.absent(),
                    },
                    "memory": {"peak_allocated_bytes": None, "peak": C.absent("not recorded")},
                    "series": {},
                    "materialisation": {"needs_series": needs, "blockers": blockers, "serving": serving, "reads": reads},
                    "result": {k: C.tv(v, "recorded") for k, v in metrics.items()},
                    "outcome": {"status": status, "provenance": "derived",
                                "how": "the leaderboard row has metrics for this job" if metrics else "the row has an elapsed time but no metrics for this job"},
                    "failures": [] if metrics else [{"mode": "crash", "time_s": elapsed, "step": None, "at": "unknown", "provenance": "derived",
                                                     "how": "no metrics recorded for this job; the cause is not recorded"}],
                    "artefacts": {"produces": (["ckpt_<label>.pt", "model_source_<label>.py"] if is_train
                                               else [] if "nanogpt_lm_eval.py" in sh else ["$OUTPUT_DIR/<workload>_metrics.json", "$OUTPUT_DIR/<workload>_trace.json"]),
                                  "checkpoint_bytes": (C.tv(CKPT_BYTES_345M, "inferred", CKPT_HOW) if (is_train and t in FIXED_ARCH)
                                                       else C.absent("not recorded; the architecture varies with the edit" if is_train else None))},
                    "host": {"cpus_per_gpu": C.tv(12, "recorded", "MLS-Bench README: GPU tasks get 12 CPUs per GPU, 16 minimum, and 64 GB RAM")},
                    "agent_view": {"script": script_rel, "upstream": f"{REPO}/tree/{commit}/tasks/{t}",
                                   "launch_command": f"bash {x['cmd']}", "brief": f"tasks/{t}/task_description.md",
                                   "recognisability": "medium: a public benchmark task, though recently published"},
                    "disclosure": {"limit_s": hms(x["time"]), "limit_how": "the task's own wall-clock limit for this job (config.json `time`)"},
                }
                recs.append(rec)
                counts["records"] += 1
    by_label = collections.defaultdict(list)
    for r in recs:
        by_label[r["identity"]["source_key"]["label"]].append(r)
    for label, rs in by_label.items():
        med = C.median([r["timing"]["wall_total_s"]["value"] for r in rs])
        for r in rs:
            v = r["timing"]["wall_total_s"]["value"]
            if med and v < 0.1 * med:
                r["outcome"]["suspect"] = (f"recorded duration {v:.0f} s is under a tenth of the median {med:.0f} s for job `{label}` across "
                                           f"{len(rs)} recorded baselines; the run may have failed early or skipped work, though it has metrics")
                counts["flagged suspect duration"] += 1
    C.add_disclosure(recs)
    for r in recs:
        fi = r["identity"]["fingerprint_inputs"]
        if fi:
            r["identity"]["fingerprint"], r["identity"]["fingerprint_version"] = C.fingerprint(fi)
            r["identity"]["fingerprint_how"] = ("gpu-mock's reference implementation over fingerprint_inputs, from the job script's "
                                                "environment and the scaffold; optimizers are null because each baseline's edit sets them")
        else:
            r["identity"]["fingerprint_how"] = "not computed: an evaluation job has no training loop to fingerprint"
        r["readiness"] = C.readiness(r)
        C.write_json(os.path.join(out_dir, r["identity"]["record_id"] + ".json"), r)
    print(f"mls-bench: {len(recs)} records from {len(names)} tasks, commit {commit}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print("  outcomes:", dict(collections.Counter(r["outcome"]["status"] for r in recs)))
    print("  replay-ready:", sum(r["readiness"]["replay"] for r in recs))


if __name__ == "__main__":
    main()
