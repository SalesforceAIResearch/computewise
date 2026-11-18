"""Brief-only reference: read original input files and source budgets.

This runs on the staged box, never on the laptop's downloaded source. It reads
the ordinary RUNS.tsv plus the original evaluator/data layout. It never reads a
record, leaderboard, private size table or recorded duration. --describe-only
exercises the input reader without launching GPU work.
"""
import argparse
import ast
import csv
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import zipfile


def unique_source(root, basename):
    original = root / "source"
    candidates = sorted((original if original.is_dir() else root).rglob(basename))
    if not candidates:
        raise FileNotFoundError(f"original source {basename} is not staged")
    return candidates[0]


def original_constant(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"original constant {name} absent from {path}")


def launch_environment(row):
    env = dict(os.environ)
    tokens = shlex.split(row.get("command", ""))
    if tokens and tokens[0] == "env":
        tokens = tokens[1:]
    for token in tokens:
        if not re.fullmatch(r"[A-Za-z_]\w*=.*", token, re.DOTALL):
            break
        key, value = token.split("=", 1)
        env[key] = value
    return env


def original_env_float_tuple(path, name, env):
    """Read the source's env-backed float tuple without executing source code."""
    values = [n.value for n in ast.parse(path.read_text()).body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if len(values) != 1:
        raise ValueError(f"ambiguous original tuple {name} in {path}")
    value = values[0]
    try:
        literal = ast.literal_eval(value)
    except ValueError:
        literal = None
    if isinstance(literal, (tuple, list)):
        return tuple(float(x) for x in literal)
    calls = [n for n in ast.walk(value) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "get"]
    if len(calls) != 1 or len(calls[0].args) != 2:
        raise ValueError(f"unsupported original env tuple {name}")
    key, default = (ast.literal_eval(arg) for arg in calls[0].args)
    if not isinstance(key, str) or not isinstance(default, str):
        raise ValueError(f"non-string original env tuple {name}")
    expected = ast.parse(
        f"tuple(float(x) for x in os.environ.get({key!r}, {default!r}).split(',') if x.strip())",
        mode="eval").body
    if ast.dump(value, include_attributes=False) != ast.dump(expected, include_attributes=False):
        raise ValueError(f"unsupported original env tuple expression {name}")
    return tuple(float(x) for x in env.get(key, default).split(",") if x.strip())


def dataset_file(workspace, repo, filename):
    path = workspace / "data" / "hf" / repo / filename
    if not path.is_file():
        raise FileNotFoundError(f"actual source dataset file not staged: {path}")
    return path


def parquet_rows(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path).metadata.num_rows


def resolve_input(workspace, path):
    if str(path).startswith("/workspace/"):
        return workspace / path.relative_to("/workspace")
    if str(path).startswith("/data/"):
        return workspace / "data" / path.relative_to("/data")
    return path if path.is_absolute() else workspace / path


def original_script(root, domain, basename):
    original = root / "source" / "tasks" / domain / "scripts" / basename
    if original.is_file():
        return original, original.read_text()
    candidates = [p for p in root.rglob(basename) if domain in str(p)]
    if not candidates:
        raise FileNotFoundError(f"original {domain}/scripts/{basename} is not staged")
    texts = {p.read_text() for p in candidates}
    if len(texts) != 1:
        raise ValueError(f"different original script copies for {domain}/{basename}")
    return candidates[0], next(iter(texts))


def flag_number(text, flag):
    match = re.search(re.escape(flag) + r"\s+(\d+)", text)
    if not match:
        raise ValueError("not a literal original flag: " + flag)
    return int(match.group(1))


def input_path(text):
    match = re.search(r"--data-path\s+([^\s\\]+)", text)
    if not match:
        raise ValueError("original script has no data path")
    return Path(os.path.expandvars(match.group(1)))


def nonempty_jsonl(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        return sum(bool(line.strip()) for line in stream)


def sparse_token_budget(path, workload, script):
    if workload != "niah_8k":
        return flag_number(script, "--max-new-tokens")
    tree = ast.parse(path.read_text())
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "env_niah")
    values = [kw.value.value for n in ast.walk(func) if isinstance(n, ast.Call)
              for kw in n.keywords if kw.arg == "max_new_tokens" and isinstance(kw.value, ast.Constant)]
    if len(values) != 1:
        raise ValueError("ambiguous original NIAH token budget")
    return values[0]


def read_sizes(workspace, rows):
    """Read each distinct workload once. No precomputed size summary is used."""
    cache, evidence = {}, []
    for row in rows:
        domain, workload = row["domain"], row["workload"]
        key = domain, workload
        if key in cache:
            continue
        if domain.startswith("llm-kv-"):
            basename = "custom_quant_eval.py" if domain.endswith("adaptive-quantization") else "custom_selection_eval.py"
            path = unique_source(workspace, basename)
            budget = original_constant(path, "WORKLOAD_CONFIGS")[workload.replace("-", "_")]["max_new_tokens"]
            if workload == "gsm8k":
                count = parquet_rows(dataset_file(workspace, "openai/gsm8k", "main/test-00000-of-00001.parquet"))
            elif workload == "longbench-v2":
                count = len(json.loads(dataset_file(workspace, "THUDM/LongBench-v2", "data.json").read_text()))
            elif workload == "needlebench-niah":
                count = nonempty_jsonl(dataset_file(workspace, "opencompass/NeedleBench", "PaulGrahamEssays.jsonl"))
                count *= len(original_env_float_tuple(path, "NEEDLE_DEPTHS", launch_environment(row)))
            else:
                name = original_constant(path, "WORKLOAD_CONFIGS")[workload.replace("-", "_")]["dataset_name"]
                with zipfile.ZipFile(dataset_file(workspace, "THUDM/LongBench", "data.zip")) as archive:
                    with archive.open("data/" + name + ".jsonl") as stream:
                        count = sum(bool(line.strip()) for line in stream)
            where = f"actual pinned dataset file under data/hf; {path}::WORKLOAD_CONFIGS (and NEEDLE_DEPTHS for Needle)"
        elif domain == "dlm-dkv-policy":
            basename = "custom_dlm_eval.py"
            name = workload.replace("-", "_")
            path = unique_source(workspace, basename)
            budget = original_constant(path, "WORKLOAD_CONFIGS")[name]["num_steps"]
            if workload == "math":
                count = nonempty_jsonl(dataset_file(workspace, "HuggingFaceH4/MATH-500", "test.jsonl"))
            elif workload == "humaneval":
                count = parquet_rows(dataset_file(workspace, "openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"))
            else:
                count = parquet_rows(dataset_file(workspace, "allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"))
            where = f"actual pinned source dataset JSONL/Parquet rows; {path}::WORKLOAD_CONFIGS"
        elif domain == "llm-dllm-demask-strategy":
            script_name = workload.replace("-", "_") + ".sh"
            path, script = original_script(workspace, domain, script_name)
            count = flag_number(script, "--n-samples") if workload == "dream-text" else nonempty_jsonl(resolve_input(workspace, input_path(script)))
            budget = flag_number(script, "--steps")
            where = f"{path}: original data path/counted rows or original --n-samples; --steps"
        elif domain == "mlsys-sparse-attention-inference":
            path, script = original_script(workspace, domain, workload + ".sh")
            cap_match = re.search(r"--max-cases\s+\$\{MAX_CASES:-(\d+)\}", script)
            if not cap_match:
                raise ValueError("no original max-cases default")
            cap = int(os.environ.get("MAX_CASES", cap_match.group(1)))
            data_flag = {"niah_8k": "--niah-cases", "longbench_qasper": "--qasper-jsonl",
                         "longbench_multifieldqa_en": "--multifieldqa-jsonl"}[workload]
            match = re.search(re.escape(data_flag) + r"\s+([^\s\\]+)", script)
            if not match:
                raise ValueError("original sparse script has no real data path")
            count = min(cap, nonempty_jsonl(resolve_input(workspace, Path(os.path.expandvars(match.group(1))))))
            budget = sparse_token_budget(unique_source(workspace, "run_llm.py"), workload, script)
            where = f"{path}: actual input rows capped by original max-cases; original generation limit"
        else:
            raise ValueError("unrecognised original domain: " + domain)
        if count <= 0 or budget <= 0:
            raise ValueError(f"empty workload {key}: {count}, {budget}")
        cache[key] = count * budget
        evidence.append({"domain": domain, "workload": workload, "examples_read": count,
                         "configured_budget_read": budget, "from": where})
    return cache, evidence


def run_queue(workspace, ordered, gpus):
    waiting, active, completed = list(ordered), {}, []
    logs = workspace / "reference_logs"
    logs.mkdir(exist_ok=True)
    while waiting or active:
        for gpu in range(gpus):
            if gpu in active or not waiting:
                continue
            row = waiting.pop(0)
            log = (logs / (f"{len(completed):03d}-gpu{gpu}-" + row["method"] + "-" + row["workload"] + ".log")).open("w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            proc = subprocess.Popen(row["command"], shell=True, cwd=workspace, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = proc, row, log
        for gpu, (proc, row, log) in list(active.items()):
            code = proc.poll()
            if code is not None:
                log.close()
                completed.append({"domain": row["domain"], "method": row["method"],
                                  "workload": row["workload"], "returncode": code})
                del active[gpu]
                if code:
                    raise RuntimeError(f"source job failed: {row}, exit {code}")
        if active:
            time.sleep(0.2)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", "--root", type=Path, default=Path.cwd())
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--describe-only", action="store_true")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    os.chdir(workspace)
    if (workspace / "INPUT-LAYOUT-ONLY.txt").exists() and not args.describe_only:
        raise SystemExit("input-layout check only: use --describe-only; this tree has no replay jobs")
    rows = list(csv.DictReader((workspace / "RUNS.tsv").open(), delimiter="\t"))
    work, evidence = read_sizes(workspace, rows)
    ordered = sorted(rows, key=lambda row: -work[(row["domain"], row["workload"])])
    print(json.dumps({"input_reader": "original data files/cache and source scripts/configs",
                      "workloads_read": evidence,
                      "order": [{k: row[k] for k in ("domain", "method", "workload")} for row in ordered]}, indent=2), flush=True)
    if not args.describe_only:
        print(json.dumps({"completed": run_queue(workspace, ordered, args.gpus)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
