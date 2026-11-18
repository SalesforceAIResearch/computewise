"""All-required cache/decode report on eight H100s, owned by K2.

The source inventory order is preserved. The public reference ranks by examples
times configured generation budget, never by elapsed seconds. New-domain replay
records use r8.5.1 whole-job batch-return adapters. Analysis' 11:51/11:55 MST
ruling permits recorded command wall with GPU/host time explicitly unseparated.

--inspect writes only the lane readout/public configuration, not the registry.
"""
import argparse
import ast
import copy
import csv
import difflib
import gzip
import heapq
import json
import math
import pathlib
import shlex
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C

TASK_ID = "pkg-k2-latepole"
COMMIT = "b873944969c53b01e4ee6e843c77767b8c702485"
DLM = "dlm-dkv-policy"
QUANT = "llm-kv-adaptive-quantization"
SELECTION = "llm-kv-selection-budgeting"
SPARSE = "mlsys-sparse-attention-inference"
DOMAINS = (DLM, QUANT, SELECTION, SPARSE)
NEW_DOMAINS = {DLM, SPARSE}
WALL_DISCLOSURE = ("not separated: GPU is shown busy for the whole recorded wall time; "
                   "outer host setup runs additionally, for real, outside the seam")
DLM_ENV = {"DLM_MAX_EXAMPLES": "0", "DLM_MAX_PROMPTS": "0", "DLM_APPLY_CHAT_TEMPLATE": "1"}
SPARSE_ENV = {"SPARSE_ATTN_DTYPE": "fp16", "SPARSE_ATTN_IMPL": "sdpa"}
VENDORED_SOURCE = HERE.parent / "survey" / "k2-latepole-inputs"
LANE = HERE.parent.parent / "package-search" / "lanes" / "unneeded-k2"


def source_root(source=None):
    return pathlib.Path(source) if source else (VENDORED_SOURCE if VENDORED_SOURCE.exists()
                                              else LANE / "sources" / "mls-late-pole")


def split_size(root, filename, config, split):
    data = json.loads((root / "dataset-sizes" / filename).read_text())
    matches = [x["num_rows"] for x in data["size"]["splits"]
               if x["config"] == config and x["split"] == split]
    if len(matches) != 1:
        raise ValueError((filename, config, split, matches))
    return matches[0]


def analysis_workloads(source=None):
    """Private source-size readout; never staged as a consolidated job table."""
    root = source_root(source)
    gsm = split_size(root, "openai--gsm8k.json", "main", "test")
    math = split_size(root, "HuggingFaceH4--MATH-500.json", "default", "test")
    arc = split_size(root, "allenai--ai2_arc.json", "ARC-Challenge", "test")
    human = len([line for line in gzip.decompress((root / "dataset-sizes" / "HumanEval.jsonl.gz").read_bytes()).splitlines()
                 if line.strip()])
    essays = sum(bool(line.strip()) for line in (root / "dataset-sizes" / "PaulGrahamEssays.jsonl").open())
    acquisition = json.loads((root / "dataset-sizes" / "acquisition.json").read_text())
    v2 = next(x["rows"] for x in acquisition if x.get("file") == "LongBench-v2-data.json" and "rows" in x)
    items = {}

    def add(task, name, count, budget, count_how, budget_how):
        items[task + "/" + name] = {
            "examples": count, "generation_budget": budget,
            "count_source": count_how, "budget_source": budget_how,
        }

    for task in (QUANT, SELECTION):
        for label, count, budget, why in (
            ("gsm8k", gsm, 256, "openai/gsm8k main test dataset-size response"),
            ("longbench-hotpotqa", 300, 32, "LongBench-E card: 100 examples in each of three length bands"),
            ("longbench-passage-retrieval", 300, 32, "LongBench-E card: 100 examples in each of three length bands"),
            ("longbench-repobench", 300, 64, "LongBench-E card: 100 examples in each of three length bands"),
        ):
            add(task, label, count, budget, why, "original evaluator WORKLOAD_CONFIGS.max_new_tokens")
    add(QUANT, "needlebench-niah", essays * 3, 32,
        f"{essays} actual PaulGrahamEssays rows times the evaluator's three NEEDLE_DEPTHS (0.10, 0.50, 0.90)",
        "original evaluator WORKLOAD_CONFIGS.max_new_tokens")
    add(SELECTION, "longbench-v2", v2, 128,
        "503 rows counted in the actual source data.json; hash and acquisition retained",
        "original evaluator WORKLOAD_CONFIGS.max_new_tokens")
    for label, count, budget, why in (
        ("math", math, 256, "HuggingFaceH4/MATH-500 test dataset-size response"),
        ("humaneval", human, 512, "nonempty rows in actual HumanEval.jsonl.gz"),
        ("lm-eval", arc, 64, "allenai/ai2_arc ARC-Challenge test dataset-size response"),
    ):
        add(DLM, label, count, budget, why, "original evaluator WORKLOAD_CONFIGS.num_steps")
    for label, count, budget in (("niah_8k", 50, 24), ("longbench_qasper", 50, 64),
                                ("longbench_multifieldqa_en", 100, 64)):
        add(SPARSE, label, count, budget,
            "original launch script --max-cases (real input cap)",
            "original run_llm.py niah max_new_tokens=24 or script --max-new-tokens 64")
    return {"schema": "p006-latepole-source-size-analysis/1", "source_commit": COMMIT,
            "workloads": items,
            "note": "Private analysis only. These combined count/budget rows are not agent inputs; the real workspace keeps data and original scripts/configs in their original forms."}


def source_jobs(source=None):
    """Preserve inventory task/CSV/command order and reject aggregate seed rows."""
    root, rows = source_root(source), []
    for task in DOMAINS:
        cfg = json.loads((root / "tasks" / task / "config.json").read_text())
        commands = {x["label"]: x for x in cfg["test_cmds"]}
        for line, row in enumerate(csv.DictReader((root / "tasks" / task / "leaderboard.csv").open()), 2):
            if row.get("is_final") != "true" or row.get("seed", "").lower() in ("", "mean", "std", "avg", "average"):
                continue
            elapsed = [float(row["elapsed_" + label]) for label in commands if row.get("elapsed_" + label)]
            for label, command in commands.items():
                value = row.get("elapsed_" + label)
                if value is None or command["compute"] != 1:
                    continue
                wall = float(value)
                if wall < 1200 or elapsed.count(wall) > 1:
                    continue
                method = row["model"].removeprefix("baseline:")
                rows.append({"job_id": f"{task}--{method}--{label}--s{row['seed']}",
                             "task": task, "method": method, "workload": label, "seed": int(row["seed"]),
                             "source_csv": f"tasks/{task}/leaderboard.csv", "source_csv_line": line,
                             "source_wall_s": wall, "source_row": row, "source_command": command["cmd"]})
    if len(rows) != 38 or sum(r["task"] in NEW_DOMAINS for r in rows) != 20:
        raise ValueError("source cohort moved; recompute design, do not silently replace jobs")
    return rows


def reference_order(public_jobs, workloads):
    """Rank public workload sizes only; stable ties preserve file order."""
    def work(job):
        cfg = workloads["workloads"][job["task"] + "/" + job["workload"]]
        return cfg["examples"] * cfg["generation_budget"]
    return [j["job_id"] for j in sorted(public_jobs, key=lambda job: -work(job))]


def record_id(row):
    prefix = "k2-latepole-" if row["task"] in NEW_DOMAINS else "mls-"
    return f"{prefix}{row['task']}-{C.slug(row['method'])}-{C.slug(row['workload'])}-s{row['seed']}"


def source_metrics(row):
    raw, label = row["source_row"], row["workload"]
    if row["task"] == "mlsys-sparse-attention-inference":
        stem = {"niah_8k": "niah", "longbench_qasper": "qasper",
                "longbench_multifieldqa_en": "multifieldqa"}[label]
        return {key: float(raw[key]) for key in raw if key.startswith(stem + "_")}
    return {key[:-len(label)-1]: float(value) for key, value in raw.items()
            if key.endswith("_" + label) and not key.startswith("elapsed_") and value}


def scaffold_path(task, name, source=None):
    package = "dLLM-cache" if task == DLM else "sparse-attn-eval"
    return source_root(source) / "harbor/tasks-docker" / ("mls-bench__" + task) / "environment/_scaffold" / package / name


def literal_assignment(path, name):
    """Read retained baseline source as data, never execute its edit program."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"{path}: literal {name} absent")


def replace_class(code, name, replacement):
    node = next(n for n in ast.parse(code).body if isinstance(n, ast.ClassDef) and n.name == name)
    lines = code.splitlines(keepends=True)
    return "".join(lines[:node.lineno - 1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


def original_method(task, method, source=None):
    name = "custom_dlm_eval.py" if task == DLM else "run_llm.py"
    code = scaffold_path(task, name, source).read_text()
    edit = source_root(source) / "tasks" / task / "edits" / (method + ".edit.py")
    if task == DLM:
        code = replace_class(code, "DLMRefreshPolicy", literal_assignment(edit, "_POLICY"))
    return code


def extract_batch(code, function, first_line, helper, parameters, returns):
    """Move an unchanged contiguous function-body span to a module helper.

    All original statements remain in their original order, including timers,
    allocation reads, progress prints and error paths. Only local values cross
    the new function boundary. No result or duration is inserted into code.
    """
    tree = ast.parse(code)
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = code.splitlines(keepends=True)
    starts = [i for i in range(outer.lineno, outer.end_lineno)
              if lines[i].rstrip() == first_line]
    if len(starts) != 1:
        raise ValueError(f"{function}: ambiguous extraction start {first_line}")
    terminal = outer.body[-1]
    if not (isinstance(terminal, ast.Expr) and isinstance(terminal.value, ast.Call)
            and isinstance(terminal.value.func, ast.Name) and terminal.value.func.id == "print"):
        raise ValueError(function + ": last statement is not original terminal print")
    start, stop = starts[0], terminal.lineno - 1
    definition_start = min([outer.lineno] + [d.lineno for d in outer.decorator_list]) - 1
    body = "".join(lines[start:stop]).rstrip() + "\n"
    signature = f"def {helper}({', '.join(parameters)}):\n"
    result = "    return {" + ", ".join(f"{name!r}: {expression}" for name, expression in returns.items()) + "}\n\n\n"
    assign = f"    metrics = {helper}({', '.join(parameters)})\n"
    assign += "".join(f"    {expression} = metrics[{name!r}]\n" for name, expression in returns.items())
    return ("".join(lines[:definition_start]) + signature + body + result
            + "".join(lines[definition_start:start]) + assign + "".join(lines[stop:]))


def extracted_method(task, method, source=None):
    original = original_method(task, method, source)
    code = original
    if task == DLM:
        code = extract_batch(code, "run_evaluation", "    final_scores: list[float] = []", "evaluate_batch",
                             ["model", "tokenizer", "device", "examples", "policy", "workload_name", "regime_name"],
                             {name: name for name in ("final_score", "reuse_ratio", "refresh_ratio", "tokens_per_s", "peak_mem_mb")})
    else:
        for function, marker, helper, params, quality in (
            ("env_niah", "    correct = 0", "evaluate_niah_batch",
             ["args", "cases", "model", "tok", "filler_token_pool", "instr_text"], "acc"),
            ("env_qasper", "    f1_sum = 0.0", "evaluate_qasper_batch",
             ["args", "examples", "model", "tok"], "f1"),
            ("env_multifieldqa_en", "    f1_sum = 0.0", "evaluate_multifieldqa_batch",
             ["args", "examples", "model", "tok"], "f1"),
        ):
            code = extract_batch(code, function, marker, helper, params,
                                 {name: name for name in (quality, "stats", "elapsed")})
    ast.parse(code)
    return original, code


def extraction_artifacts(destination, source=None):
    """Preserve originals/refactors/diffs for the explicit equivalence check."""
    destination = pathlib.Path(destination)
    seen, rows = set(), []
    for row in source_jobs(source):
        key = row["task"], row["method"]
        if row["task"] not in NEW_DOMAINS or key in seen:
            continue
        seen.add(key)
        original, extracted = extracted_method(*key, source)
        directory = destination / row["task"] / row["method"]
        directory.mkdir(parents=True, exist_ok=True)
        for name, code in (("original.py", original), ("extracted.py", extracted)):
            (directory / name).write_text(code)
        diff = "".join(difflib.unified_diff(original.splitlines(keepends=True), extracted.splitlines(keepends=True),
                                           fromfile="original.py", tofile="extracted.py"))
        (directory / "extraction.diff").write_text(diff)
        rows.append({"domain": key[0], "method": key[1], "original_sha256": C.sha256_text(original),
                     "extracted_sha256": C.sha256_text(extracted), "diff": str(directory / "extraction.diff")})
    print(json.dumps({"extractions": rows, "executed": False,
                      "scope": "pure source extraction artifacts; equivalence execution not yet run"}, indent=2))


def entry_source(task, method):
    module = ("dlm_" if task == DLM else "sparse_") + method
    if task == DLM:
        original = scaffold_path(task, "custom_dlm_eval.py").read_text()
        main = original[original.index("if __name__ == '__main__':"):]
        return f"from {module} import *\n\n" + main
    return f"from {module} import main\n\nif __name__ == '__main__':\n    main()\n"


def project_module_sources(task, method, source=None):
    """Method-local files staged alongside the evaluator."""
    if task == DLM:
        return {}
    custom = scaffold_path(task, "custom_sparse_attn.py", source).read_text()
    custom = replace_class(custom, "SparseAttention", literal_assignment(
        source_root(source) / "tasks" / task / "edits" / (method + ".edit.py"), "_CONTENT"))
    return {"custom_sparse_attn": custom,
            "harness": scaffold_path(task, "harness.py", source).read_text()}


def project_dependency_files(task, method, source=None):
    """Bind project files by task-root path, including source-lazy imports."""
    root = source_root(source)
    if task == SPARSE:
        return {f"methods/{task}/{method}/{name}.py": C.sha256_text(code)
                for name, code in project_module_sources(task, method, source).items()}
    files = {}

    def add(src, dest):
        files[dest] = C.sha256_bytes((root / src).read_bytes())

    if method in {"vanilla_uncached", "dllm_cache"}:
        package = root / "dependency-probes/dLLM-cache"
        for path in sorted((package / "dllm_cache").rglob("*.py")):
            add(path.relative_to(root), "shared/dLLM-cache/" + path.relative_to(package).as_posix())
        if len(files) != 9:
            raise ValueError("pinned dllm_cache package dependency files missing")
        pin = next(item["revision"] for item in json.loads((root / "model-metadata/acquisition.json").read_text())
                   if item["repo"] == "GSAI-ML/LLaDA-8B-Instruct")
        for name in ("modeling_llada", "configuration_llada"):
            add("model-metadata/GSAI-ML--LLaDA-8B-Instruct-" + name + ".py",
                "data/huggingface_cache/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/" + pin + "/" + name + ".py")
    elif method == "d2cache":
        source_prefix = "dependency-probes/tasks/dlm-dkv-policy/third_party/official_dlm_cache_baselines/d2cache/"
        dest_prefix = "shared/dlm-task/third_party/official_dlm_cache_baselines/d2cache/"
        for path in ("src/__init__.py", "src/models/__init__.py", "src/models/llada/__init__.py",
                     "src/models/llada/modeling_llada.py", "src/models/llada/configuration_llada.py"):
            add(source_prefix + path, dest_prefix + path)
    elif method == "elastic_cache":
        source_prefix = "dependency-probes/tasks/dlm-dkv-policy/third_party/official_dlm_cache_baselines/elastic_cache/llada/model/"
        dest_prefix = "shared/dlm-task/third_party/official_dlm_cache_baselines/elastic_cache/llada/model/"
        for name in ("modeling_llada.py", "configuration_llada.py"):
            add(source_prefix + name, dest_prefix + name)
    else:
        raise ValueError("unknown DLM dependency branch: " + method)
    return files


def workspace_assets(row, reg, source=None):
    """Ordinary method source and source-equivalent launch, no metric constants."""
    task, method = row["task"], row["method"]
    root, reg = source_root(source), pathlib.Path(reg)
    folder = f"methods/{task}/{method}"
    files = []

    def asset(name, code, *, pin=False):
        rel = f"artifacts/k2-latepole/{task}/{method}/{name}"
        C.write_text(str(reg / rel), code)
        file = {"path": folder + "/" + name, "from": rel}
        if pin:
            file["sha256"] = C.sha256_text(code)
        files.append(file)
        return rel

    original, extracted = extracted_method(task, method, source)
    dependencies = project_module_sources(task, method, source)
    module = ("dlm_" if task == DLM else "sparse_") + method
    asset(module + ".py", extracted)
    entry = "custom_dlm_eval.py" if task == DLM else "run_llm.py"
    entry_code = entry_source(task, method)
    entry_artifact = asset(entry, entry_code)
    interpreter = "/opt/envs/b/bin/python"
    if task == DLM:
        argv = [entry, "--workload", row["workload"].replace("-", "_"), "--regime", "final", "--seed", str(row["seed"])]
        setup = ('export DLLM_CACHE_DIR="$ROOT/shared/dLLM-cache"\n'
                 'export MLSBENCH_TASK_DIR="$ROOT/shared/dlm-task"\n'
                 'export DLM_MAX_EXAMPLES="${DLM_MAX_EXAMPLES:-${DLM_MAX_PROMPTS:-0}}"\n'
                 'export DLM_MAX_PROMPTS="${DLM_MAX_PROMPTS:-0}"\n'
                 'export DLM_APPLY_CHAT_TEMPLATE="${DLM_APPLY_CHAT_TEMPLATE:-1}"\n'
                 'export LLADA_MODEL_DIR="$(' + interpreter + ' -c '
                 + shlex.quote("from huggingface_hub import snapshot_download; print(snapshot_download('GSAI-ML/LLaDA-8B-Instruct', local_files_only=True))") + ')"\n')
        seam = module + ":evaluate_batch"
        live = {"peak_mem_mb": "recorded_peak_mb"}
    else:
        for name, code in dependencies.items():
            asset(name + ".py", code)
        argv = [entry, "--env", row["workload"], "--model-path", "Qwen/Qwen2.5-1.5B-Instruct"]
        data_flags = {"niah_8k": ["--niah-cases", "data/niah/cases.jsonl", "--filler-text", "data/longctx/wikitext103_test.txt", "--context-len", "8192", "--max-cases", "50"],
                      "longbench_qasper": ["--qasper-jsonl", "data/longbench-qasper/qasper.jsonl", "--max-context-len", "8192", "--max-new-tokens", "64", "--max-cases", "50"],
                      "longbench_multifieldqa_en": ["--multifieldqa-jsonl", "data/longbench-qasper/multifieldqa_en.jsonl", "--max-context-len", "8192", "--max-new-tokens", "64", "--max-cases", "100"]}
        argv += data_flags[row["workload"]] + ["--seed", str(row["seed"]), "--density-budget", "0.25"]
        setup = ""
        seam = module + ":" + {"niah_8k": "evaluate_niah_batch", "longbench_qasper": "evaluate_qasper_batch", "longbench_multifieldqa_en": "evaluate_multifieldqa_batch"}[row["workload"]]
        live = {"elapsed": "seam_elapsed_s"}
    argv_workspace = [folder + "/" + argv[0]] + argv[1:]
    command = interpreter + (" -u " if task == SPARSE else " ") + shlex.join(argv_workspace)
    script_name = "scripts/" + row["workload"].replace("-", "_") + ".sh"
    script = (root / "tasks" / task / row["source_command"]).read_text()
    root_setup = 'ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"\ncd "$ROOT"\n'
    root_setup += ('export HF_HOME="$ROOT/data/huggingface_cache"\n'
                   'export HF_DATASETS_CACHE="$HF_HOME/datasets"\n'
                   'export HF_MODULES_CACHE="$HF_HOME/modules"\n')
    if task == DLM:
        script = script.replace('if [ -d /workspace ]; then\n  cd /workspace\nfi\n', '')
        script = script.replace('python3 "${DLLM_ROOT}/custom_dlm_eval.py"', interpreter + " " + folder + "/" + entry)
    else:
        script = script.replace("cd /workspace", 'cd "$ROOT"')
        script = script.replace("python -u sparse-attn-eval/run_llm.py", interpreter + " -u " + folder + "/" + entry)
        script = script.replace("/data/qwen2.5-1.5b-instruct", "Qwen/Qwen2.5-1.5B-Instruct")
        script = script.replace("/data/", "$ROOT/data/")
    script = script.replace("\n", "\n# Usage: bash " + folder + "/" + script_name + "; Python interpreter: " + interpreter + "\n" + root_setup + setup, 1)
    asset(script_name, script, pin=task == SPARSE)
    return {"files": files, "launch": "bash " + folder + "/" + script_name,
            "entry": entry_artifact, "code": entry_code, "argv": argv,
            "module_artifact": files[0]["from"], "module_sha256": C.sha256_text(extracted),
            "original_sha256": C.sha256_text(original), "seam": seam, "live_fields": live,
            "module_shas": project_dependency_files(task, method, source)}


def make_source_record(row, reg, source=None):
    assets = workspace_assets(row, reg, source)
    metrics = source_metrics(row)
    if row["task"] == DLM:
        aggregate = {k: metrics[k] for k in ("final_score", "reuse_ratio", "tokens_per_s")}
        aggregate["refresh_ratio"] = 1.0 - metrics["reuse_ratio"]
    else:
        stem, quality = {"niah_8k": ("niah", "acc"), "longbench_qasper": ("qasper", "f1"), "longbench_multifieldqa_en": ("multifieldqa", "f1")}[row["workload"]]
        aggregate = {quality: metrics[stem + "_" + quality], "stats": {"mean": metrics[stem + "_density"]}}
    memory_file = source_root(source) / "memory-model.json"
    profiles = json.loads(memory_file.read_text()) if memory_file.exists() else {}
    profile = profiles.get(row["task"])
    blockers = [] if profile else ["native model weights plus working-set memory model not yet supplied; zero peak is not a usable instrument reading"]
    n = math.ceil(row["source_wall_s"] / 60)
    shares = [1.0 / n] * n
    shares[-1] += 1.0 - sum(shares)
    config_hash = C.sha256_text(json.dumps([row["task"], row["method"], row["workload"], assets["module_sha256"]]))
    record = {"schema": C.SCHEMA_RECORD,
        "identity": {"record_id": record_id(row), "kind": "eval", "entry_hash": C.sha256_text(assets["code"]),
                     "entry_hash_how": "CLI-preserving entry imports the source evaluator after pure extraction into a library module",
                     "argv": assets["argv"], "argv_exempt": [], "env": dict(DLM_ENV if row["task"] == DLM else SPARSE_ENV), "env_exempt": [],
                     "seed": C.tv(row["seed"], "recorded", "source leaderboard seed column"),
                     "replicate_group": "k2-latepole-" + config_hash[:16], "match_policy": "exact_only",
                     "trajectory": {"config_hash": config_hash, "config_hash_how": "domain, method, workload and extracted evaluator bytes",
                                    "environment": {"source_commit": COMMIT, "transformers": "4.46.3"},
                                    "lr_schedule": {"identity": None, "source": None, "trace": None},
                                    "data": {"source_workload": row["workload"]}, "precision": C.tv("bfloat16" if row["task"] == DLM else "float16", "inferred", "original model loader dtype")},
                     "fingerprint": None, "fingerprint_inputs": None, "consumes": [],
                     "source_key": {"task": row["task"], "baseline": row["method"], "label": row["workload"]}},
        "mode": "replay", "source": {"repo": "https://github.com/Imbernoulli/MLS-Bench", "commit": COMMIT,
                    "path": row["source_csv"] + "#line=" + str(row["source_csv_line"]),
                    "source_row": row["source_row"], "licence": "none declared", "read_on": "2026-11-02"},
        "reference": {"gpu": "NVIDIA H100 80GB HBM3", "gpu_provenance": "inferred", "gpu_how": "MLS source environment globally specifies H100, not a per-run device log",
                      "gpu_count": 1, "gpu_count_provenance": "recorded", "gpu_count_how": "source test_cmds.compute=1 and single-device script",
                      "precision": "bfloat16" if row["task"] == DLM else "float16", "interconnect": None},
        "launch": {"launcher": "python", "world_size": C.tv(1, "derived", "source single-model single-GPU evaluator"),
                   "per_device_microbatch": C.absent(), "accumulation_steps": C.absent(), "global_batch_tokens": C.absent(),
                   "batch_policy": C.absent("no width conversion"), "alternatives": {}},
        "steps": None, "planned_steps": None, "per_step_gpu_s": None,
        "timing": {"pacing": "total", "wall_total_s": C.tv(row["source_wall_s"], "recorded", row["source_csv"] + " elapsed_" + row["workload"] + "; " + WALL_DISCLOSURE),
                   "gpu_host_split": WALL_DISCLOSURE, "setup_s": C.absent("outer setup runs for real; source does not separate it from recorded elapsed"),
                   "eval_s": C.absent(), "checkpoint_s": C.absent()},
        "memory": {"peak_allocated_bytes": profile["peak_bytes"] if profile else None,
                   "peak_bytes": profile["peak_bytes"] if profile else None,
                   "peak": C.tv(profile["peak_bytes"], "modelled", profile["how"]) if profile else C.absent("source records no memory; modeled profile pending")},
        "series": {}, "materialisation": {"reads": ["terminal quality aggregate", "live memory/timer instrument"], "needs_series": [], "blockers": blockers,
            "serving": {"mode": "whole_job", "seam": assets["seam"], "n_units": n, "unit": "equal subdivision of recorded command wall, at most 60 seconds",
                        "seam_sha256": assets["module_sha256"],
                        "unit_shares": shares, "result_key": "final_metrics", "result_kind": "metric dict", "result_shape": "dict",
                        "live_fields": assets["live_fields"], "how": "r8.5 required for trajectory-env and module-content binding, plus r8.3 live_fields; function extracted for serving; behaviour unchanged; no per-example values are invented"}},
        "result": {"source_metrics": C.tv(metrics, "recorded", "original source CSV terminal metric cells"),
                   "final_metrics": C.tv(aggregate, "derived" if row["task"] == DLM else "recorded", "source terminal metrics, refresh_ratio=1-reuse_ratio for DLM; peak memory comes from simulator memory model and elapsed is live, neither stored as a recorded result")},
        "outcome": {"status": "completed", "provenance": "inferred", "how": "final seeded source row with finite completed-command elapsed and terminal metrics"},
        "failures": [], "artefacts": {"produces": []}, "host": {"weights_bytes": C.absent(), "data_bytes": C.absent()},
        "agent_view": {"script": assets["entry"], "launch_command": "/opt/envs/b/bin/python " + shlex.join(assets["argv"]),
                       "inputs": [row["workload"]], "workspace_files": assets["files"], "workspace_launch": assets["launch"],
                       "module_artifact": assets["module_artifact"], "module_sha256": assets["module_sha256"],
                       "recognisability": "actual source model/method/workload names retained"}}
    if assets["module_shas"]:
        record["materialisation"]["serving"]["module_shas"] = assets["module_shas"]
        record["materialisation"]["serving"]["how"] += "; r8.5.1 required to bind task-root project file paths on disk, including lazy imports, without importing them"
    if profile and profile.get("weights_bytes"):
        record["memory"]["weights_bytes"] = C.tv(profile["weights_bytes"], "modelled", profile["how"])
    record["readiness"] = C.readiness(record)
    return record


def derive(reg, source=None):
    reg = pathlib.Path(reg)
    records = [make_source_record(row, reg, source) for row in source_jobs(source) if row["task"] in NEW_DOMAINS]
    C.add_disclosure(records)
    for record in records:
        C.write_json(str(reg / "records" / (record["identity"]["record_id"] + ".json")), record)
    print(json.dumps({"new_records": len(records), "replay_ready_fields": sum(r["readiness"]["replay"] for r in records),
                      "actual_replay": "not yet exercised", "output": str(reg)}, indent=2))
    return records


def model_hub_entries(source=None):
    root = source_root(source) / "model-metadata"
    entries = []
    for acquisition in json.loads((root / "acquisition.json").read_text()):
        meta = json.loads((root / (acquisition["repo"].replace("/", "--") + ".json")).read_text())
        files = {f["rfilename"]: f["size"] for f in meta["siblings"]
                 if f["rfilename"] not in (".gitattributes", "README.md", "LICENSE")}
        entries.append({"kind": "model", "repo": acquisition["repo"], "revision": acquisition["revision"],
                        "files": files, "bytes": sum(files.values()), "python_environment": "b",
                        "revision_how": acquisition["how"]})
    return entries


def dataset_hub_entries(source=None):
    root = source_root(source)
    entries = []
    for pin in json.loads((root / "input-layout-hub-pins.json").read_text()):
        if pin["repo"] not in {"HuggingFaceH4/MATH-500", "openai/openai_humaneval", "allenai/ai2_arc"}:
            continue
        local = root / "raw-inputs" / pin["repo"]
        files = {p.relative_to(local).as_posix(): p.stat().st_size for p in local.rglob("*") if p.is_file()}
        entries.append({"kind": "dataset", "repo": pin["repo"], "revision": pin["revision"], "files": files,
                        "bytes": sum(files.values()), "python_environment": "b", "split": "test",
                        **({"config": "ARC-Challenge"} if pin["repo"] == "allenai/ai2_arc" else {}),
                        "load_alias": "openai_humaneval" if pin["repo"] == "openai/openai_humaneval" else pin["repo"],
                        "revision_how": "pinned actual input files retained in the staged-reader acquisition; source run did not log a per-run dataset revision"})
    return entries


def build_scratch(reg, source=None):
    """Build a concrete task in owned scratch, reading canonical KV inputs only."""
    import gen_tasks as GT
    reg = pathlib.Path(reg)
    if reg.resolve() == (HERE.parent / "registry").resolve():
        raise ValueError("this lane does not write the canonical registry")
    native = derive(reg, source)
    recs = {r["identity"]["record_id"]: r for r in native}
    live = HERE.parent / "registry"
    for path in sorted((live / "records").glob("mls-llm-kv-*.json")):
        record = json.loads(path.read_text())
        recs[record["identity"]["record_id"]] = record
        C.write_json(str(reg / "records" / path.name), record)
    for path in (live / "records").glob("example-order-*.json"):
        shutil.copyfile(path, reg / "records" / path.name)
    task = build(recs, GT.disclosure_of, reg, source)
    # The old KV entries and their real artifact bytes remain unchanged.
    for record in recs.values():
        path = record["agent_view"].get("script")
        if path and not (reg / path).exists():
            (reg / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live / path, reg / path)
    for file in task["files"]:
        rel = file.get("from")
        if rel and rel.startswith("artifacts/") and not (reg / rel).exists():
            (reg / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live / rel, reg / rel)
    import siblings as SIB
    GT.REG_DIR = str(reg)
    task = GT.complete_task(task, recs, str(reg), SIB.sibling_pools(recs))
    C.write_json(str(reg / "tasks" / (TASK_ID + ".json")), task)
    print(json.dumps({"task_id": TASK_ID, "task_sha256": C.sha256_bytes((reg / "tasks" / (TASK_ID + ".json")).read_bytes()),
                      "jobs": len(task["jobs"]), "readiness": task.get("readiness"), "horizon_s": task.get("horizon_s"),
                      "actual_replay": "not yet exercised", "scratch": str(reg)}, indent=2))
    return task


def full_input_layout(reg, source=None):
    """Stage the actual method/source layout without an image or GPU launch."""
    import gen_tasks as GT
    reg = pathlib.Path(reg)
    if reg.resolve() == (HERE.parent / "registry").resolve():
        raise ValueError("input-only artifact cannot be installed in the canonical registry")
    records = derive(reg, source)
    recs = {r["identity"]["record_id"]: r for r in records}
    live = HERE.parent / "registry"
    for path in (live / "records").glob("mls-llm-kv-*.json"):
        record = json.loads(path.read_text())
        recs[record["identity"]["record_id"]] = record
    task = build(recs, GT.disclosure_of, reg, source)
    files = task["files"] + [{k: v for k, v in task["scripted_reference"].items() if k in ("path", "from")}]
    files.append({"path": "INPUT-LAYOUT-ONLY.txt", "content": "Input-layout check only; no GPU jobs or replay. Use --describe-only.\n"})
    for file in files:
        rel = file.get("from")
        if rel and rel.startswith("artifacts/") and not (reg / rel).exists():
            (reg / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(live / rel, reg / rel)
    artifact = {"task_id": "pkg-k2-latepole-inputs38", "brief": {"text": "Input-layout inspection only; no GPU jobs or replay."}, "jobs": [], "files": files,
                "notes": ["Full method/source/input layout inspection only; no GPU jobs, no replay, never an installed task.",
                          "Actual 38-row RUNS.tsv and both original/relocated evaluator/script paths, so wrapper basename collisions are exercised.",
                          "Prepare genuine NIAH/QA inputs with prepare/source_inputs.py, then reference/latepole.py --workspace <tree> --gpus 8 --describe-only.",
                          "No image mapping or model-weight loading required. Model/cache preparation and actual replay remain separate."]}
    path = reg / "tasks" / (artifact["task_id"] + ".json")
    C.write_json(str(path), artifact)
    print(json.dumps({"file": str(path), "files": len(files), "jobs": 0,
                      "sha256": C.sha256_bytes(path.read_bytes()), "executed": False}, indent=2))


def check_record(record, reg):
    """Source correspondence for new records; actual replay remains a box check."""
    if not record["identity"]["record_id"].startswith("k2-latepole-"):
        return []
    errors = []
    rows = {record_id(row): row for row in source_jobs() if row["task"] in NEW_DOMAINS}
    row = rows.get(record["identity"]["record_id"])
    if row is None:
        return ["new source record has no matching pinned source job"]
    if record["source"].get("commit") != COMMIT:
        errors.append("source commit differs from pinned MLS-Bench source")
    if record["source"].get("source_row") != row["source_row"]:
        errors.append("source row differs from the pinned CSV")
    if record["reference"].get("gpu_count") != 1 or "H100" not in str(record["reference"].get("gpu")):
        errors.append("source allocation is not native one-H100")
    if record["reference"].get("gpu_provenance") == "recorded":
        errors.append("source hardware is globally inferred, not logged per run")
    metrics = record.get("result", {}).get("source_metrics", {}).get("value")
    if metrics != source_metrics(row):
        errors.append("source_metrics differs from original terminal metrics")
    aggregate = record.get("result", {}).get("final_metrics", {}).get("value")
    if row["task"] == DLM:
        expected = {k: source_metrics(row)[k] for k in ("final_score", "reuse_ratio", "tokens_per_s")}
        expected["refresh_ratio"] = 1.0 - expected["reuse_ratio"]
        live = {"peak_mem_mb": "recorded_peak_mb"}
    else:
        stem, quality = {"niah_8k": ("niah", "acc"), "longbench_qasper": ("qasper", "f1"), "longbench_multifieldqa_en": ("multifieldqa", "f1")}[row["workload"]]
        expected = {quality: metrics[stem + "_" + quality], "stats": {"mean": metrics[stem + "_density"]}}
        live = {"elapsed": "seam_elapsed_s"}
    if aggregate != expected:
        errors.append("served aggregate differs from source terminal cells/declared arithmetic")
    if record.get("materialisation", {}).get("serving", {}).get("live_fields") != live:
        errors.append("live timer/model-memory getters differ from supported contract")
    if record.get("identity", {}).get("env") != (DLM_ENV if row["task"] == DLM else SPARSE_ENV):
        errors.append("full-workload trajectory environment defaults differ")
    duration = record.get("timing", {}).get("wall_total_s", {})
    paced = duration.get("value")
    if paced != row["source_wall_s"] or duration.get("provenance") != "recorded":
        errors.append("pacing must retain exact recorded command wall, with recorded provenance")
    if record.get("timing", {}).get("pacing") != "total":
        errors.append("whole-command pacing is not total")
    if record.get("timing", {}).get("gpu_host_split") != WALL_DISCLOSURE:
        errors.append("unseparated GPU/host wall-time disclosure absent or changed")
    script = pathlib.Path(reg) / record["agent_view"]["script"]
    if not script.exists() or C.sha256_bytes(script.read_bytes()) != record["identity"].get("entry_hash"):
        errors.append("entry artifact absent or hash differs")
    if record["identity"].get("entry_hash") != C.sha256_text(entry_source(row["task"], row["method"])):
        errors.append("entry is not the source-CLI-preserving library launcher")
    module = pathlib.Path(reg) / record["agent_view"].get("module_artifact", "")
    if not module.is_file() or C.sha256_bytes(module.read_bytes()) != record["agent_view"].get("module_sha256"):
        errors.append("extracted module artifact absent or differs from source derivation")
    expected_module = extracted_method(row["task"], row["method"])[1]
    if record["agent_view"].get("module_sha256") != C.sha256_text(expected_module):
        errors.append("module is not the pure extraction derived from the pinned source/baseline")
    if record.get("materialisation", {}).get("serving", {}).get("seam_sha256") != C.sha256_text(expected_module):
        errors.append("authoritative serving seam_sha256 does not bind the pure source extraction")
    expected_shas = project_dependency_files(row["task"], row["method"])
    if record.get("materialisation", {}).get("serving", {}).get("module_shas", {}) != expected_shas:
        errors.append("serving module_shas does not bind all pinned project dependency files")
    workspace = {f["path"]: f["from"] for f in record["agent_view"]["workspace_files"]}
    for path, wanted in expected_shas.items():
        if path not in workspace:
            continue  # DLM shared git checkouts and pinned model cache are checked in the box, not registry artifacts.
        dependency = pathlib.Path(reg) / workspace[path]
        if not dependency.is_file() or C.sha256_bytes(dependency.read_bytes()) != wanted:
            errors.append("project dependency artifact absent or differs from pinned source: " + path)
    return errors


def schedule(order, durations, gpus=8):
    free = [(0.0, i) for i in range(gpus)]
    heapq.heapify(free)
    trace = []
    for jid in order:
        start, gpu = heapq.heappop(free)
        end = start + durations[jid]
        heapq.heappush(free, (end, gpu))
        trace.append({"job_id": jid, "gpu": gpu, "start_s": start, "end_s": end})
    return max(t for t, _ in free), trace


def input_files(reg, source=None, include_reference=False):
    """Stage real source/data forms, without a count/budget summary table."""
    root, reg = source_root(source), pathlib.Path(reg)
    files = []

    def asset(src, dest):
        rel = "artifacts/k2-latepole-inputs/" + dest
        path = reg / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        path.write_bytes(data)
        files.append({"path": dest, "from": rel, "sha256": C.sha256_bytes(data)})

    for task in DOMAINS:
        for src in sorted((root / "tasks" / task / "scripts").glob("*.sh")):
            asset(src, f"source/tasks/{task}/scripts/{src.name}")
        asset(root / "tasks" / task / "config.json", f"source/tasks/{task}/config.json")
        scaffold = root / "harbor/tasks-docker" / ("mls-bench__" + task) / "environment/_scaffold"
        for src in sorted(scaffold.rglob("*.py")):
            asset(src, f"source/harbor/tasks-docker/mls-bench__{task}/environment/_scaffold/" + src.relative_to(scaffold).as_posix())
        package = "dLLM-cache" if task == DLM else "sparse-attn-eval" if task == SPARSE else None
        if package:
            asset(root / "vendor/pkg_configs" / package / "config.json", "source/vendor/pkg_configs/" + package + "/config.json")
    for src in sorted((root / "vendor/data_scripts").rglob("*.py")):
        asset(src, "source/" + src.relative_to(root).as_posix())
    for src in sorted((root / "raw-inputs").rglob("*")):
        if src.is_file():
            asset(src, "data/hf/" + src.relative_to(root / "raw-inputs").as_posix())
    for spec in json.loads((root / "input-layout-hub-files.json").read_text()):
        if spec.get("error"):
            if spec.get("repo") == "opencompass/NeedleBench":
                continue  # small pinned original file is a normal artifact above
            raise ValueError("source Hub pin missing: " + str(spec))
        files.append({"path": "data/hf/" + spec["repo"] + "/", "hub_files": spec,
                      "from": "original pinned dataset files, no derived count manifest"})
    asset(root / "prepare_inputs.py", "prepare/source_inputs.py")
    if include_reference:
        asset(root / "reference.py", "reference/latepole.py")
    return files


def input_layout(reg, source=None):
    """A zero-job input-layout check, never an installed replay package."""
    reg = pathlib.Path(reg)
    rows = source_jobs(source)
    files = input_files(reg, source, include_reference=True)
    inventory = "domain\tmethod\tworkload\tseed\tcommand\n" + "".join(
        f"{r['task']}\t{r['method']}\t{r['workload']}\t{r['seed']}\tbash source/tasks/{r['task']}/{r['source_command']}\n"
        for r in rows)
    files += [{"path": "RUNS.tsv", "content": inventory},
              {"path": "INPUT-LAYOUT-ONLY.txt", "content": "Input-layout check only. No jobs, no replay, never a task. The reader may run --describe-only; original source commands are not a replay launch plan.\n"}]
    task = {"task_id": "pkg-k2-latepole-inputs", "brief": {"text": "Input-layout inspection only; no replay jobs."},
            "files": files, "jobs": [],
            "notes": ["Input-layout check only, no jobs, no replay, never a task.",
                      "Before reading, run python <tree>/prepare/source_inputs.py --workspace <tree> on the staging host. This prepares genuine source input JSONLs, not outputs.",
                      "Then run python <tree>/reference/latepole.py --workspace <tree> --gpus 8 --describe-only. Requires pyarrow for original Parquet input metadata.",
                      "Large Hub inputs are pinned and SHA256-checked; they are fetched on the staging host, not the controller."]}
    path = reg / "tasks/pkg-k2-latepole-inputs.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(task, indent=2) + "\n")
    print(json.dumps({"file": str(path), "files": len(files), "jobs": 0,
                      "status": "input-layout artifact only; neither staged nor executed"}, indent=2))


def build(recs, disclosure_of, reg=None, source=None):
    """Task integration after authentic domain records/staging are available.

    New domain adapters provide their real launch_command and workspace_files in
    agent_view. All original dependencies stay in their adapter's files. Missing
    records are reported instead of creating fabricated timing/serving values.
    """
    rows, workloads = source_jobs(source), analysis_workloads(source)
    jobs, files, missing = [], [], []
    required = [record_id(row) for row in rows]
    absent = [rid for rid in required if rid not in recs]
    if absent:
        raise ValueError("late-pole source records absent:\n" + "\n".join(absent))
    # Reuse the existing authentic KV staging/launch definitions unchanged.
    # This call is for the integrator's scratch/canonical build, not --inspect.
    import gen_mlskv_task as KV
    import gen_tasks as GT
    kv_task = KV.build(recs, disclosure_of, reg, variant="mls-kv-report", gpus=8)
    kv_jobs = {rid: job for job in kv_task["jobs"] for rid in job["records"]}
    kv_scripts = {kv_jobs[rid]["launch"].removeprefix("bash ") for rid in required if rid in kv_jobs}
    for original_file in kv_task["files"]:
        if "/scripts/" in original_file["path"] and original_file["path"] not in kv_scripts:
            continue
        file = copy.deepcopy(original_file)
        if "/scripts/" in file["path"]:
            file["content"] = file["content"].replace("\npython ", "\n/opt/envs/a/bin/python ")
            file["content"] = file["content"].replace("\n", "\n# Usage: bash " + file["path"] + "; Python interpreter: /opt/envs/a/bin/python\n", 1)
        files.append(file)
    for row in rows:
        rid = record_id(row)
        record = recs.get(rid)
        if record is None:
            missing.append(rid)
            continue
        view, ident = record["agent_view"], record["identity"]
        is_kv = row["task"] not in NEW_DOMAINS
        workspace = [] if is_kv else view.get("workspace_files")
        launch = kv_jobs[rid]["launch"] if is_kv else view.get("workspace_launch")
        if (not is_kv and not workspace) or not launch:
            missing.append(rid + " (authentic adapter workspace_files/workspace_launch absent)")
            continue
        files.extend(workspace)
        job = {"job_id": row["job_id"], "label": f"{row['method']}, {row['workload']}",
               "records": [rid], "replicate_group": ident["replicate_group"],
               "draw": "exact_seed", "seed": row["seed"], "recorded_gpus": 1,
               "recorded_gpu": record["reference"]["gpu"], "batch_policy": None, "gpu_options": [1],
               "launch": launch, "after": [], "consumes": [], "produces": [],
               "source_task": row["task"], "workload": row["workload"], "method": row["method"]}
        if is_kv:
            for key in ("script_path", "n_examples_full", "score_kind"):
                if key in kv_jobs[rid]:
                    job[key] = kv_jobs[rid][key]
        else:
            job["script_path"] = f"methods/{row['task']}/{row['method']}/" + ("custom_dlm_eval.py" if row["task"] == DLM else "run_llm.py")
            job["n_examples_full"] = workloads["workloads"][row["task"] + "/" + row["workload"]]["examples"]
            job["score_kind"] = "pass@1" if row["task"] == DLM and row["workload"] == "humaneval" else "accuracy" if row["task"] == DLM or row["workload"] == "niah_8k" else "f1"
        job["disclosure"] = disclosure_of([rid], recs, "1x " + job["recorded_gpu"])
        jobs.append(job)
    if missing:
        raise ValueError("late-pole adapter integration is incomplete:\n" + "\n".join(missing))
    unique_files = {}
    for file in files:
        if file["path"] in unique_files and unique_files[file["path"]] != file:
            raise ValueError("adapter staging collision: " + file["path"])
        unique_files[file["path"]] = file
    inventory = "domain\tmethod\tworkload\tseed\tinterpreter\tcommand\n" + "".join(
        f"{r['task']}\t{r['method']}\t{r['workload']}\t{r['seed']}\t/opt/envs/{'b' if r['task'] in NEW_DOMAINS else 'a'}/bin/python\t{j['launch']}\n"
        for r, j in zip(rows, jobs))
    public_rows = [{key: row[key] for key in ("job_id", "task", "workload")} for row in rows]
    files = list(unique_files.values()) + input_files(reg, source) + [{"path": "RUNS.tsv", "content": inventory}]
    files += [{"path": "shared/dLLM-cache/dllm_cache/", "git": {"repo": "maomaocun/dLLM-Cache", "commit": "17235bffc8c5b587a2dc6f7dc76fcd01eab76e3a", "keep": "dllm_cache"}},
              {"path": "shared/dlm-task/third_party/", "git": {"repo": "Imbernoulli/MLS-Bench", "commit": COMMIT, "keep": "tasks/dlm-dkv-policy/third_party"}}]
    reference_artifact = "artifacts/k2-latepole/reference.py"
    C.write_text(str(pathlib.Path(reg) / reference_artifact), (source_root(source) / "reference.py").read_text())
    environment = KV.environment(8)
    environment["image_label"] = {"value": "k2-latepole", "how": "isolation-approved two Python environments in one image; actual image build/import/replay reported by owners"}
    environment["python_environments"] = {
        "a": {"path": "/opt/envs/a", "jobs": "eighteen reused KV evaluations", "requirements": "existing mlsbench-kv stack"},
        "b": {"path": "/opt/envs/b", "jobs": "eleven DLM and nine sparse evaluations", "requirements": "survey/k2-latepole-input-check-2026-11-02/environment-b.freeze.txt", "hf_home": "<workspace>/data/huggingface_cache", "hf_datasets_cache": "<workspace>/data/huggingface_cache/datasets", "hf_modules_cache": "<workspace>/data/huggingface_cache/modules", "how": "46-package freeze after actual E2 resolution/import/check, 2026-11-02 12:11 MST; source-unpinned extras resolved, CPU torch2.11 replaces source CUDA2.5.1 under instrument; processed dataset cache separate from environment a's datasets5.x"},
    }
    environment["hub"] += model_hub_entries(source) + dataset_hub_entries(source)
    environment["hub_bytes"] = sum(h["bytes"] for h in environment["hub"])
    environment["preparation"] = {
        "inputs": "python prepare/source_inputs.py --workspace <tree>; actual NIAH/QA source input preparation, already exercised on historical layout",
        "dlm_cache": "source/vendor/data_scripts/dLLM-cache/prepare_hf_datasets.py --data-root <tree>/data, under environment b; preserve original openai_humaneval alias and ARC-Challenge configuration; no change to evaluator loader",
        "sparse_filler": "source/vendor/data_scripts/sparse-attn-eval/prepare_longctx.py --data-root <tree>/data; original source prepares data/longctx/wikitext103_test.txt; source has wikitext2 fallback",
        "cache_status": "new model and DLM processed caches/filler not yet exercised by this lane; stage through gpu-tasks/isolation, not on controller",
        "how": "ordinary source inputs/caches only; no pre-made results/checkpoints/logs or prior attempt"}
    return {"task_id": TASK_ID, "rung": 3, "env_kind": "mls-latepole", "level": "goal",
            "goal_level": True, "set": "evaluation", "box": copy.deepcopy(GT.BOX8),
            "title": "Cache and decoding comparison, complete requested rows",
            "brief": {"text": "Compare the listed cache and decoding methods on the evaluations listed in RUNS.tsv. Each row’s requested evaluation is defined by its command and the script’s default case limits. Produce a results table for every row, with its number of evaluated examples, quality and accompanying efficiency metrics. Explain the tradeoffs the table supports. The machine has eight H100s.",
                      "provenance": "research reproduction report, 2026-11-02; source identities retained"},
            "jobs": jobs, "files": files,
            "output_isolation": {"flag": "OUTPUT_DIR", "exempt": "reused KV output directory only; DLM and sparse write stdout and no shared result file"},
            "environment": environment,
            "scripted_reference": {
                "path": "reference/latepole.py", "from": reference_artifact,
                "python": "/opt/envs/b/bin/python",
                "args": ["--workspace", ".", "--gpus", "8"],
                "how": "scripted-only controller reads actual original input/cache files and original budget flags/configs; no private size table, score table or recorded seconds",
            },
            "latepole": {"reference_order": reference_order(public_rows, workloads),
                         "reference_rule": "descending examples times generation budget, read from real data/cache files and original scripts/configs; stable file-order ties",
                         "reference_input_status": "reader 68ee37e0b720 executed on E2 staged input layout, exit 0, 2026-11-02 11:35 MST; its 44-row order restricted to this unchanged 38-row subset attains the source-wall bound; measured replay/bar 2 pending",
                         "ordering": "natural inventory task/CSV/command order unchanged",
                         "new_domain_jobs": 20, "old_kv_jobs": 18,
                         "timing_fidelity": WALL_DISCLOSURE,
                         "demask_drop": "six demask jobs omitted; no recorded generation-region durations; analysis accepted the 38-row cohort at 11:55 MST",
                         "source_wall_estimate_only": True},
            "sources": ["MLS-Bench commit " + COMMIT],
            "environment_requirements": {
                "base": "one r8.5.1+ box image, /opt/envs/a for eighteen KV jobs, /opt/envs/b for eleven DLM and nine sparse jobs, patch in both; source major-version incompatibility requires separate environments",
                "status": "isolation accepted layout; new-domain imports, live peak/timer return support and actual replay remain outstanding",
            },
            "horizon_override": {
                "maximum_s": 216000,
                "how": "the analysis 2026-11-02 11:07 MST: 60 h, disclosed; 38-row source-wall best 8.10 h, FCFS 10.18 h, serial work 53.26 h (approximately 79.89 h under serial-times-1.5 before setup). Retain uncapped rule_horizon_s. Actual replay and host overhead remain unmeasured.",
            },
            "notes": ["DLM/sparse: " + WALL_DISCLOSURE + "; authorized by the analysis 2026-11-02 11:51/11:55 MST. No per-example durations or outputs are invented.",
                      "No recorded seconds or outcomes are present in the public reference inputs.",
                      "Eighteen KV jobs reuse G1/H5/G4 source records; twenty jobs (52.63%) are from two other domains.",
                      "A function extracted for serving; behaviour unchanged. Original timer/memory/control-flow statements and terminal prints are retained; extraction diffs are held in the lane. Stub equivalence and actual replay are reported separately.",
                      "Eight GPUs are a capacity condition for late-pole waste; this is separate from the held K2 precision capacity dial."]}


def inspect(source, out):
    rows, config = source_jobs(source), analysis_workloads(source)
    public = [{key: row[key] for key in ("job_id", "task", "workload")} for row in rows]
    order = reference_order(public, config)
    times = {row["job_id"]: row["source_wall_s"] for row in rows}
    floor = max(sum(times.values()) / 8, max(times.values()))
    ref, trace = schedule(order, times)
    fcfs, _ = schedule([row["job_id"] for row in rows], times)
    result = {"basis": "recorded source wall times; DLM/sparse GPU/host not separated by analysis ruling; not agent measurements",
              "jobs": len(rows), "private_source_size_analysis": config, "reference_order": order,
              "reference_makespan_s": ref, "source_wall_lower_bound_s": floor,
              "reference_over_lower_bound": ref / floor, "fcfs_makespan_s": fcfs,
              "fcfs_over_reference": fcfs / ref, "reference_trace": trace,
              "visibility": "consolidated count/budget and jobs JSON proposals withdrawn; neither is staged; actual source data and scripts/configs remain separate",
              "remaining_input_check": "reference must read actual workspace data/cache cardinalities and original CLI/config budgets; no consolidated size table"}
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "late-pole-analysis-sizes.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "late-pole-reference-readout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("private_source_size_analysis", "reference_order", "reference_trace")}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=pathlib.Path)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--inspect", type=pathlib.Path)
    modes.add_argument("--input-layout", type=pathlib.Path, help="write only a zero-job input-layout check under this scratch registry")
    modes.add_argument("--extract", type=pathlib.Path, help="write baseline originals, pure extractions and diffs for checking")
    modes.add_argument("--records", type=pathlib.Path, help="generate the twenty source records and genuine workspace artifacts into a scratch registry")
    modes.add_argument("--build", type=pathlib.Path, help="build a concrete task under an owned scratch registry; never canonical")
    modes.add_argument("--input-layout-full", type=pathlib.Path, help="emit only a zero-job full method/source layout for exercising the actual reader")
    args = ap.parse_args()
    if args.inspect:
        inspect(args.source, args.inspect)
    elif args.input_layout:
        input_layout(args.input_layout, args.source)
    elif args.extract:
        extraction_artifacts(args.extract, args.source)
    elif args.records:
        derive(args.records, args.source)
    elif args.build:
        build_scratch(args.build, args.source)
    else:
        full_input_layout(args.input_layout_full, args.source)
