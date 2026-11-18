#!/usr/bin/env python3
"""Known answer for analysis's case of 2026-10-23 04:25Z (G-mlskv-haiku-L0-r3 on the ba36 KV box):
the run's four evaluator launches `custom_selection_eval.py --workload gsm8k --max-examples K`
replayed through the registry with mls-kv-deploy's records. The launch omits the job script's
`--compression-ratio 0.8`, which is the script's own default, so with the parsed arguments sent
(as the patch now does) each must match the full_attention gsm8k record as `exact_effective`
with subset {k, 1319}; a different ratio must still miss.
    python3 -I tests/known_answer_haiku_l0r3.py <registry dir>
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "daemon"))
import gpu_daemon as gd  # noqa: E402

reg_dir = Path(sys.argv[1])
reg = gd.Registry(reg_dir / "records" if (reg_dir / "records").is_dir() else reg_dir)
rid = "mls-llm-kv-selection-budgeting-full-attention-gsm8k-s42"
rec = reg._by_id.get(rid)
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
chk("the full_attention gsm8k record is in the registry", rec is not None)
if rec is None:
    print(f"{n - fails} passed, {fails} failed"); sys.exit(1)
H = rec["identity"]["entry_hash"]
# what custom_selection_eval.py's parser yields for `--workload gsm8k --max-examples K` (defaults: model-id Qwen/Qwen2.5-3B-Instruct, seed 7, compression-ratio 0.8, attn sdpa, cpu False)
def eff(k, ratio="0.8"):
    return {"workload": "gsm8k", "model_id": "Qwen/Qwen2.5-3B-Instruct", "seed": "7", "compression_ratio": ratio, "attn_implementation": "sdpa", "cpu": False, "max_examples": str(k)}
for j, k in ((2, 5), (3, 20), (5, 2), (7, 10)):
    argv = ["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", str(k)]
    r, mode, info = reg.lookup(None, H, argv, job_key=f"haiku-{j}", stage="exact", script_path="/w/methods/full_attention/custom_selection_eval.py", argv_effective=eff(k))
    chk(f"haiku's launch {j} (k={k}): exact_effective on the full_attention gsm8k record", r is not None and reg._rid(r) == rid and info.get("match") == "exact_effective" and mode == "replay", f"{mode} {info.get('match')} {reg._rid(r) if r else None}")
    sub = gd._subset_of(gd._per_example(r) or {}, argv, {}) if r else None
    chk(f"   subset {{k: {k}, n_full: 1319}}", sub == {"k": k, "n_full": 1319}, str(sub))
r, mode, info = reg.lookup(None, H, ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.3", "--max-examples", "5"], job_key="haiku-x", stage="exact", script_path="/w/x.py", argv_effective=eff(5, "0.3"))
chk("a ratio no record has (0.3) still misses", r is None and info.get("match") == "none", f"{mode} {info.get('match')}")
nm = reg.near_miss(H, ["custom_selection_eval.py", "--workload", "gsm8k", "--compression-ratio", "0.3", "--max-examples", "5"])
chk("…and the near_miss names the ratio", nm is not None and nm.get("only_launch") == ["0.3"], str(nm))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
