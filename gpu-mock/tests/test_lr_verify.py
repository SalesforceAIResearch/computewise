#!/usr/bin/env python3
"""lr_verify against the registry's own traces (RECORD-FORMAT §5). Torch-free.
    python3 -I tests/test_lr_verify.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "torch_patch"))
import lr_verify  # noqa: E402
REG = ROOT.parent / "gpu-tasks" / "registry" / "records"
PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else "")); PASS += bool(cond); FAIL += (not cond)

def trace(rid):
    return json.loads((REG / f"{rid}.json").read_text())["identity"]["trajectory"]["lr_schedule"]["trace"]

sft = trace("trl-sft-h100-sxmx1-0022af66-s1-afosdmfe")       # after_step, steps 1..30, linear from 1.41e-5, no warmup
dpo = trace("trl-dpo-h100-sxmx4-e5295792-s42-ig7dh8vc")      # after_step, steps 1,10,20,..., warmup 150 to 1e-3
check("after_step: step 1 is unconstrained (trace starts at 1)", lr_verify.expected_lr(sft, 1) is None)
check("after_step: step 2 expects trace[step 1]", lr_verify.expected_lr(sft, 2) == [1.363e-05], str(lr_verify.expected_lr(sft, 2)))
check("a step absent from a sparse trace constrains nothing", lr_verify.expected_lr(dpo, 5) is None)
check("sparse trace: step 11 expects the value logged at 10", lr_verify.expected_lr(dpo, 11) == [6.666666666666667e-05])
check("match within relative 1e-6", lr_verify.mismatch(sft, 2, [1.363e-05 * (1 + 5e-7)]) is None)
check("every param group must match the single value", lr_verify.mismatch(sft, 2, [1.363e-05, 1.363e-05]) is None)
m = lr_verify.mismatch(sft, 2, [1.363e-05, 2e-05])
check("a second group off the value is a mismatch, group named", m is not None and m["group"] == 1, str(m))
m = lr_verify.mismatch(sft, 2, [1.41e-05])
check("SFT job at step 2 still on the initial rate (different schedule length) mismatches", m is not None, str(m))
# the collision gpu-tasks found: an SFT job verified against a reward-model record (same lr, other step count)
rm_ids = sorted(p.stem for p in REG.glob("trl-reward*-0022af66-*.json")) or sorted(p.stem for p in REG.glob("trl-rm*.json"))
if rm_ids:
    rm = trace(rm_ids[0])
    # the first optimizer.step() call both traces constrain: an SFT job carrying
    # its own rate there must fail verification against the reward-model record
    both = [k for k in range(1, 2000) if lr_verify.expected_lr(sft, k) and lr_verify.expected_lr(rm, k)]
    check(f"some step is constrained by both traces ({rm_ids[0]}: steps {rm['steps'][:4]}…)", bool(both), str(both[:5]))
    if both:
        k = both[0]; e_sft, e_rm = lr_verify.expected_lr(sft, k), lr_verify.expected_lr(rm, k)
        check(f"SFT and reward-model rates differ at step {k}", e_sft != e_rm, f"sft={e_sft} rm={e_rm}")
        check(f"an SFT job's rate at step {k} fails verification against the reward-model record",
              lr_verify.mismatch(rm, k, e_sft) is not None)
else:
    print("[skip] no reward-model record found under the 0022af66 commit")
check("at_step alignment compares at k", lr_verify.expected_lr({"alignment": "at_step", "steps": [3], "values": [0.5]}, 3) == [0.5])
check("per_group: one list per group", lr_verify.expected_lr({"steps": [1], "values": [1.0], "per_group": [[1.0], [0.1]]}, 2) == [1.0, 0.1])
check("no trace: no constraint", lr_verify.mismatch(None, 2, [1.0]) is None)
print(f"\n{PASS} passed, {FAIL} failed"); sys.exit(1 if FAIL else 0)
