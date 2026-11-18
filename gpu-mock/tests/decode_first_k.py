#!/usr/bin/env python3
"""r6: the real per-example scores of a KV record's first K examples, measured
by running the record's own evaluator for real (run inside the KV box with the
mock switched off: NVRT_DISABLE=1, so torch is plain torch and nothing is served).

One job is one contiguous slice [start, end) of one record's workload, in the
order the evaluator's own load_workload(name, 0) builds it, which is the order
of gpu-tasks' example-order tables. For the job the harness:
  - takes the record's evaluator by entry_hash from the scripts dir and checks
    its sha256 (the exact bytes the record was run with);
  - imports it as a module (its task directory is a marker file; the data comes
    from the Hugging Face cache, offline);
  - loads the model in bf16, the recorded precision on H100 (the evaluator picks
    float32 on a CPU device, so from_pretrained is given torch_dtype=bfloat16;
    nothing else in the evaluator changes);
  - slices load_workload to [start, end) and wraps score_prediction to record
    each example's id, content key and score, then runs the evaluator's own
    main() under the record's argv: its own decode, its own scorer;
  - checks every captured key against the order table at its position.
Writes <out>/<record_id>.<start>-<end>.json (atomically) and exits 0 only if
every example in the slice was scored and every key matched its position.

    python3 -I decode_first_k.py --records R --scripts S --order O --job <record_id>:<start>:<end> --out D [--threads T]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time


def key_of(example) -> str:
    return hashlib.sha256((str(example["example_id"]) + "\0" + example["prompt"]).encode("utf-8")).hexdigest()


def family_of(rec) -> str | None:
    task = ((rec.get("identity") or {}).get("source_key") or {}).get("task") or ""
    return "quant" if "quant" in task else ("selection" if "selection" in task else None)


def order_for(order_dir, fam, wl):
    for name in os.listdir(order_dir):
        if not (name.startswith("example-order-") and name.endswith(".json")):
            continue
        d = json.load(open(os.path.join(order_dir, name)))
        if d.get("workload") == wl and fam in (d.get("families") or []):
            return d
    return None


def _cpu_model() -> str:
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--scripts", required=True, help="dir of <entry_hash[:12]>/custom_*_eval.py")
    ap.add_argument("--order", required=True, help="dir of example-order-*.json")
    ap.add_argument("--job", required=True, help="<record_id>:<start>:<end>")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()

    rid, start, end = a.job.rsplit(":", 2)
    start, end = int(start), int(end)
    rec = json.load(open(os.path.join(a.records, rid + ".json")))
    ident = rec["identity"]
    eh = ident["entry_hash"]
    ser = rec["materialisation"]["serving"]
    wl = ser["decode"]["workload"]
    fam = family_of(rec)
    n_full = int(ser["n_examples_full"]["value"])
    if not (0 <= start < end <= n_full):
        print(f"bad slice [{start}, {end}) for n_full {n_full}", file=sys.stderr)
        return 2

    sdir = os.path.join(a.scripts, eh[:12])
    srcs = [f for f in os.listdir(sdir) if f.startswith("custom_") and f.endswith("_eval.py")]
    if len(srcs) != 1:
        print(f"{sdir}: expected one evaluator, found {srcs}", file=sys.stderr)
        return 2
    src_path = os.path.join(sdir, srcs[0])
    if hashlib.sha256(open(src_path, "rb").read()).hexdigest() != eh:
        print(f"{src_path}: sha256 is not the record's entry_hash {eh}", file=sys.stderr)
        return 2
    order = order_for(a.order, fam, wl)
    if order is None or len(order.get("order") or []) != n_full:
        print(f"no order table of {n_full} for {fam}/{wl}", file=sys.stderr)
        return 2

    if a.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(a.threads)
        os.environ["MKL_NUM_THREADS"] = str(a.threads)

    # the evaluator resolves its task directory by a task_description.md beside it
    work = tempfile.mkdtemp(prefix="dfk-")
    mdir = os.path.join(work, "methods", "m")
    os.makedirs(mdir)
    ev_path = os.path.join(mdir, srcs[0])
    shutil.copy(src_path, ev_path)
    open(os.path.join(work, "methods", "task_description.md"), "w").write("# task\n")
    os.environ["MLSBENCH_TASK_DIR"] = os.path.join(work, "methods")
    os.environ.setdefault("OUTPUT_DIR", os.path.join(work, "output"))

    import torch
    if a.threads > 0:
        torch.set_num_threads(a.threads)
    # a measured decode must be real inference: refuse to run with the replay layer loaded,
    # which would serve the record's values instead of computing them
    if any(m in sys.modules for m in ("replay", "cuda_hooks", "virtual_clock")) or torch.cuda.is_available():
        print("the replay layer is loaded (or a cuda device is visible): not a measured decode; set NVRT_DISABLE=1", file=sys.stderr)
        return 3

    spec = importlib.util.spec_from_file_location("dfk_evaluator", ev_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # bf16, the recorded precision: the evaluator would pick float32 on a CPU device
    real_auto = mod.AutoModelForCausalLM

    class _BF16Auto:
        @staticmethod
        def from_pretrained(*args, **kw):
            kw["torch_dtype"] = torch.bfloat16
            return real_auto.from_pretrained(*args, **kw)
    mod.AutoModelForCausalLM = _BF16Auto

    orig_load, orig_score = mod.load_workload, mod.score_prediction
    seen = []

    def load_workload(name, max_examples=0):
        w = orig_load(name, 0)
        w.examples = w.examples[start:end]
        return w

    def score_prediction(workload_name, example, prediction):
        s = orig_score(workload_name, example, prediction)
        seen.append({"example_id": str(example["example_id"]), "key": key_of(example), "score": float(s),
                     "prediction": prediction or "", "t": time.time()})
        # one line per example, so a running slice shows its pace
        print(f"R6PROGRESS {len(seen)}/{end - start} {example['example_id']} score={float(s):.1f} t={time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
        return s

    mod.load_workload, mod.score_prediction = load_workload, score_prediction
    t0 = time.time()
    argv = list(ident["argv"])
    sys.argv = [ev_path] + [str(x) for x in argv[1:]]
    t0 = time.time()
    t_start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    loaded = {}
    real_fp = _BF16Auto.from_pretrained

    def _capture(*args, **kw):
        m = real_fp(*args, **kw)
        loaded["model"] = m
        return m
    _BF16Auto.from_pretrained = staticmethod(_capture)
    mod.main()
    wall = time.time() - t0
    t_end = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    import transformers
    m = loaded.get("model")
    cfg = getattr(m, "config", None)
    rev = None
    try:
        from huggingface_hub import try_to_load_from_cache
        f = try_to_load_from_cache(cfg._name_or_path if cfg else "Qwen/Qwen2.5-3B-Instruct", "config.json")
        rev = os.path.basename(os.path.dirname(f)) if isinstance(f, str) else None
    except Exception:
        rev = None

    ok, bad = True, []
    exs = []
    for j, s in enumerate(seen):
        pos = start + j
        match = pos < len(order["order"]) and s["key"] == order["order"][pos]
        if not match:
            ok = False
            bad.append(pos)
        exs.append({"i": pos, "example_id": s["example_id"], "key": s["key"], "score": s["score"],
                    "prediction": s["prediction"], "key_matches_order": match})
    if len(seen) != end - start:
        ok = False
    out = {"schema": "p006-measured-scores-slice/0.1", "record_id": rid, "entry_hash": eh, "family": fam, "workload": wl,
           "n_examples_full": n_full, "start": start, "end": end, "examples": exs, "complete": ok,
           "keys_mismatched": bad, "wall_s": round(wall, 1),
           "evaluator": {"path": os.path.relpath(src_path, a.scripts), "sha256": eh},
           "argv": argv,
           "decode": {"device": "cpu", "dtype": str(getattr(m, "dtype", "bfloat16")).replace("torch.", ""),
                      "torch": torch.__version__, "transformers": transformers.__version__, "model_revision": rev,
                      "attn_implementation": getattr(cfg, "_attn_implementation", None),
                      "image": os.environ.get("R6_IMAGE"), "machine": os.environ.get("R6_MACHINE") or platform.node(),
                      "cpu": _cpu_model(), "threads": torch.get_num_threads(),
                      "start": t_start, "end": t_end, "no_replay": os.environ.get("NVRT_DISABLE") == "1",
                      "harness_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
                      "how": ("NVRT_DISABLE=1: the replay layer is not loaded, so torch is plain CPU torch and nothing is served; "
                              "the record's own evaluator and argv; the model loaded in bfloat16, the recorded precision "
                              "(the evaluator picks float32 on a CPU device); examples taken by slicing load_workload(name, 0) "
                              "to [start, end), the full workload's own order")}}
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"{rid}.{start}-{end}.json")
    tmp = fn + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(out, fh)
    os.replace(tmp, fn)
    print(f"{rid} [{start},{end}) scored {len(seen)} complete={ok} mean={sum(s['score'] for s in seen) / max(1, len(seen)):.3f} wall={wall:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
