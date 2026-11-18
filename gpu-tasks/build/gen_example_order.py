"""The KV example-order tables gpu-mock's r4 shard serving reads.

usage: python3 -I build/gen_example_order.py <staged task tree> <out dir> [--check-only]

<staged task tree> is a KV task as the box sees it (methods/<method>/custom_*_eval.py, methods/task_description.md) with its
Hugging Face cache, run with HF_HOME pointing at that cache and HF_HUB_OFFLINE=1, as the box runs. For each workload it imports a
selection and a quant evaluator from the tree and calls the evaluator's own load_workload(name, 0), the full workload, then writes
<out dir>/example-order-<families>-<workload>.json:
  {"schema": "p006-example-order/0.1", "families": [...], "workload": ..., "n_examples_full": N,
   "key": "sha256(example_id\\0prompt)", "order": [key of example i], "ids": [example_id of example i], "how": ...}
order[i] is the sha256 of the example's example_id, a NUL, and its prompt, exactly as the loader builds them (after the template,
before tokenisation): position i in the record's per-example vector. The key includes example_id because NeedleBench's prompts
repeat (opencompass/NeedleBench's PaulGrahamEssays.jsonl holds 49 essays three times each, so 441 examples have 147 distinct
prompts); example_id is unique in every workload.

Checks, each fatal: every key is unique; N equals the records' n_examples_full for that workload; and where both families run a
workload, their two orders are identical (the evaluators' prompt builders are byte-identical; this confirms it on the data).

The evaluators import torch and transformers at module level and the quant one calls torch.finfo there. Neither is needed to
build the examples: in the box image the real ones load (CPU only, no model is loaded, no GPU or CUDA is touched); where they
are absent they are stubbed before import. The module is imported
under its own path, so its resolve_task_dir finds methods/task_description.md as on the box. Standard library besides what the
evaluator's loaders import (datasets, huggingface_hub, pyarrow): run it under the box image's Python.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types

# workload -> families that run it, and the record source_key.task and label each family's records use
WORKLOADS = {
    "gsm8k": ("selection", "quant"),
    "longbench_hotpotqa": ("selection", "quant"),
    "longbench_passage_retrieval": ("selection", "quant"),
    "longbench_repobench": ("selection", "quant"),
    "longbench_v2": ("selection",),
    "needlebench_niah": ("quant",),
}
TASK_OF = {"selection": "llm-kv-selection-budgeting", "quant": "llm-kv-adaptive-quantization"}
EVALUATOR = {"selection": "custom_selection_eval.py", "quant": "custom_quant_eval.py"}
METHOD = {"selection": "full_attention", "quant": "kivi_overlap_4bit"}      # one method folder per family: the loaders are shared


def stub_torch():
    """torch and transformers as empty modules with the few names the evaluators touch at import time."""
    t = types.ModuleType("torch")
    t.Tensor = type("Tensor", (), {})
    t.float16 = "float16"
    t.finfo = lambda dtype: types.SimpleNamespace(bits=16)
    t.no_grad = lambda *a, **k: (lambda f: f)
    for sub in ("nn", "nn.functional"):
        m = types.ModuleType("torch." + sub)
        sys.modules["torch." + sub] = m
    t.nn = sys.modules["torch.nn"]
    t.nn.functional = sys.modules["torch.nn.functional"]
    t.nn.Module = type("Module", (), {})
    sys.modules["torch"] = t
    tr = types.ModuleType("transformers")
    for n in ("AutoModelForCausalLM", "AutoTokenizer", "DynamicCache"):
        setattr(tr, n, type(n, (), {}))
    tr.set_seed = lambda *a, **k: None
    sys.modules["transformers"] = tr


def load_evaluator(tree, fam):
    path = os.path.join(tree, "methods", METHOD[fam], EVALUATOR[fam])
    spec = importlib.util.spec_from_file_location(f"eo_{fam}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod                        # dataclasses look the module up by name while the class is built
    spec.loader.exec_module(mod)
    return mod, path


def key_of(example):
    return hashlib.sha256((str(example["example_id"]) + "\0" + example["prompt"]).encode("utf-8")).hexdigest()


def n_full(reg, fam, workload):
    """n_examples_full from the records of this family's workload (source_key.label uses '-', the loaders '_')."""
    label = workload.replace("_", "-")
    vals = set()
    rdir = os.path.join(reg, "records")
    for f in os.listdir(rdir):
        if not f.startswith("mls-llm-kv-") or not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(rdir, f)))
        k = r["identity"].get("source_key") or {}
        if k.get("task") == TASK_OF[fam] and k.get("label") == label:
            vals.add(r["materialisation"]["serving"]["n_examples_full"]["value"])
    if len(vals) != 1:
        raise SystemExit(f"{fam} {workload}: records give n_examples_full {sorted(vals)}")
    return vals.pop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tree")
    ap.add_argument("out")
    ap.add_argument("--registry", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"))
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    stubbed = importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None
    if stubbed:
        stub_torch()                                    # outside the box image; inside it the real ones load, on CPU, no model
    mods = {fam: load_evaluator(a.tree, fam) for fam in ("selection", "quant")}
    os.makedirs(a.out, exist_ok=True)
    for wl, fams in WORKLOADS.items():
        orders = {}
        for fam in fams:
            mod, path = mods[fam]
            exs = mod.load_workload(wl, 0).examples
            keys, ids = [key_of(e) for e in exs], [str(e["example_id"]) for e in exs]
            if len(set(keys)) != len(keys):
                raise SystemExit(f"{fam} {wl}: {len(keys) - len(set(keys))} duplicate keys")
            want = n_full(a.registry, fam, wl)
            if len(keys) != want:
                raise SystemExit(f"{fam} {wl}: {len(keys)} examples, records say n_examples_full {want}")
            orders[fam] = (keys, ids, path)
        if len({tuple(k) for k, _, _ in orders.values()}) != 1:
            raise SystemExit(f"{wl}: the families build different orders: {sorted(orders)}")
        keys, ids, _ = next(iter(orders.values()))
        name = f"example-order-{'both' if len(fams) == 2 else fams[0]}-{wl}.json"
        out = {"schema": "p006-example-order/0.1", "families": list(fams), "workload": wl, "n_examples_full": len(keys),
               "key": "sha256(example_id\\0prompt)", "order": keys, "ids": ids,
               "how": ("build/gen_example_order.py: each family's own load_workload(" + repr(wl) + ", 0) on the staged task tree "
                       "with its Hugging Face cache, offline, " + ("torch and transformers stubbed" if stubbed else
                       "in the box image with its own torch and transformers, CPU only") + " (no model loaded); evaluators "
                       + ", ".join(f"{f}: {os.path.relpath(p, a.tree)} sha256 "
                                   + hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12] for f, (_, _, p) in sorted(orders.items()))
                       + "; the families' orders checked identical; keys unique; n equals the records' n_examples_full")}
        if not a.check_only:
            with open(os.path.join(a.out, name), "w") as fh:
                json.dump(out, fh)
                fh.write("\n")
        print(f"{name}: {len(keys)} examples, families {list(fams)}, first id {ids[0]}")


if __name__ == "__main__":
    main()
