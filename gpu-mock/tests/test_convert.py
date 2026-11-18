#!/usr/bin/env python3
"""Unit test for convert(record, gpu_count): the one converter both the daemon
and the reference-schedule solver call (GPU-COUNT-CONVERSION-NOTE §5, §7).

    python3 -I tests/test_convert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "convert"))
from convert import convert  # noqa: E402

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    PASS += cond
    FAIL += (not cond)


def serving(policy, rec_gpus=8, steady=0.150668, first=5.418, eval_pv=1.0045):
    return {"world_size_recorded": rec_gpus, "batch_policy": policy,
            "steady_step_s": steady, "first_step_s": first,
            "eval_s_per_validation": eval_pv, "startup_in_train_s": 5.236,
            "setup_s": 0.0}


# grow_accumulation: 8 -> 4 is 2x on steady, eval; first step = one-off + 2x steady
c = convert(serving("fixed_global_grow_accumulation"), 4)
check("accum 8->4 factor 2.0", abs(c["time_factor"] - 2.0) < 1e-9, str(c["time_factor"]))
check("accum steady doubles", abs(c["steady_step_s"] - 0.301336) < 1e-6, f"{c['steady_step_s']:.6f}")
check("accum first = one-off + 2x steady",
      abs(c["first_step_s"] - ((5.418 - 0.150668) + 0.301336)) < 1e-6, f"{c['first_step_s']:.4f}")
check("accum eval doubles", abs(c["eval_s_per_validation"] - 2.009) < 1e-6)
check("accum curve applies", c["curve_applies"] is True)
check("accum memory factor 1 (same micro-batch)", c["memory_factor"] == 1.0)
check("startup unscaled", c["startup_in_train_s"] == 5.236)

# grow_accumulation 8 -> 1 is 8x
c = convert(serving("fixed_global_grow_accumulation"), 1)
check("accum 8->1 factor 8.0", abs(c["time_factor"] - 8.0) < 1e-9)

# grow_microbatch: 8 -> 4 is one doubling -> 1.25x; 8 -> 2 two doublings -> 1.5625x
c = convert(serving("fixed_global_grow_microbatch"), 4)
check("microbatch 8->4 factor 1.25", abs(c["time_factor"] - 1.25) < 1e-9, str(c["time_factor"]))
check("microbatch memory factor = ratio (activations grow)", c["memory_factor"] == 2.0)
c = convert(serving("fixed_global_grow_microbatch"), 2)
check("microbatch 8->2 factor 1.5625", abs(c["time_factor"] - 1.5625) < 1e-9)

# fixed_per_device: curve does not apply
c = convert(serving("fixed_per_device"), 4)
check("per_device curve_applies False", c["curve_applies"] is False)
check("per_device time_factor None", c["time_factor"] is None)
check("per_device steady passed through untouched", c["steady_step_s"] == 0.150668)

# same count: identity
c = convert(serving("fixed_global_grow_accumulation"), 8)
check("same gpu count: factor 1, identity", c["time_factor"] == 1.0 and c["provenance"] == "identity")

# more GPUs than recorded: clamped, never faster than recorded
c = convert(serving("fixed_global_grow_accumulation", rec_gpus=4), 8)
check("more GPUs: clamped to 1.0", c["time_factor"] == 1.0 and c["provenance"].startswith("clamped"))

# records with missing fields degrade gracefully
c = convert({"world_size_recorded": 8, "batch_policy": None, "steady_step_s": None}, 4)
check("missing steady: no crash, steady None", c["steady_step_s"] is None)

# sharded state (memory.sharded_bytes): a ZeRO-3 / FSDP record recorded on 4 GPUs
# holding 30 GiB of sharded state and 10 GiB of the rest per card (40 GiB peak)
GiB = 2 ** 30
def sharded(rec_gpus=4, sh=30 * GiB, peak=40 * GiB, weights=None, policy="fixed_global_grow_accumulation"):
    s = serving(policy, rec_gpus=rec_gpus)
    s["memory"] = {"peak_bytes": peak, "sharded_bytes": sh, "weights_bytes": weights}
    return s
c = convert(sharded(), 4)
check("sharded, same width: memory factor 1.0", abs(c["memory_factor"] - 1.0) < 1e-12, str(c["memory_factor"]))
c = convert(sharded(), 2)
check("sharded 4->2: per-card peak 10 + 60 = 70 GiB (factor 1.75)", abs(c["memory_factor"] * 40 - 70) < 1e-9, f"{c['memory_factor'] * 40:.3f}")
c = convert(sharded(), 1)
check("sharded 4->1: per-card peak 10 + 120 = 130 GiB, past one 80 GB card", abs(c["memory_factor"] * 40 - 130) < 1e-9, f"{c['memory_factor'] * 40:.3f}")
check("sharded 4->1: time as the batch policy says (accumulation: 4x), not changed by the field", abs(c["time_factor"] - 4.0) < 1e-12)
c = convert(sharded(), 8)
check("sharded 4->8: per-card peak 10 + 15 = 25 GiB (memory is not clamped)", abs(c["memory_factor"] * 40 - 25) < 1e-9, f"{c['memory_factor'] * 40:.3f}")
check("sharded 4->8: time still clamped at 1.0", c["time_factor"] == 1.0)
c = convert(sharded(weights=20 * GiB), 1)
check("sharded weights 20 GiB (all sharded) on 1 card: 80 GiB held at load (factor 4)", abs(c["weights_factor"] - 4.0) < 1e-12, str(c["weights_factor"]))
c = convert(sharded(sh=10 * GiB, weights=20 * GiB), 2)
check("weights 20 GiB, 10 sharded, 4->2: 10 + 20 = 30 GiB (factor 1.5)", abs(c["weights_factor"] - 1.5) < 1e-12, str(c["weights_factor"]))
c = convert(sharded(policy="fixed_global_grow_microbatch"), 2)
check("sharded + grow_microbatch 4->2: unsharded part x2 (activations), sharded x2: 20 + 60 = 80 GiB",
      abs(c["memory_factor"] * 40 - 80) < 1e-9, f"{c['memory_factor'] * 40:.3f}")
c = convert(sharded(sh=50 * GiB), 2)
check("sharded bytes above the peak are capped at the peak (all of it scales)", abs(c["memory_factor"] - 2.0) < 1e-12, str(c["memory_factor"]))
s0 = serving("fixed_global_grow_accumulation", rec_gpus=4); s0["memory"] = {"peak_bytes": 40 * GiB}
c = convert(s0, 1)
check("no sharded field: memory factor 1.0 as before, no weights factor", c["memory_factor"] == 1.0 and "weights_factor" not in c)
check("provenance names the sharded scaling", "sharded state" in convert(sharded(), 1).get("memory_provenance", ""))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
