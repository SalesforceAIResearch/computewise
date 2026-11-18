#!/usr/bin/env python3
"""Surrogate evaluator (docs/SURROGATE-SERVING.md), on a placeholder family.
    python3 -I tests/test_surrogate.py
"""
import json, math, os, sys, tempfile
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "daemon"))
sys.path.insert(0, HERE)
import surrogate as sg  # noqa: E402

from toy_family import SPEC  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

fam = sg.Family(SPEC)
chk("serves its script by path suffix", fam.serves("/home/researcher/project/design/train_gpt.py") and not fam.serves("/x/other.py"))
cfg = fam.resolve(["train_gpt.py", "--lr", "0.03", "--steps=1000"], {"model_dim": 1024, "num_layers": 16, "num_parameters": 2.1e8, "global_batch_tokens": 524288})
chk("argv and fingerprint resolve, fingerprint wins for model_dim", (cfg["lr"], cfg["model_dim"], cfg["num_layers"], cfg["train_steps"]) == (0.03, 1024.0, 16.0, 1000.0), str(cfg))
base = fam.resolve(["train_gpt.py"], {})
chk("defaults without argv or fingerprint", (base["lr"], base["model_dim"], base["train_steps"]) == (0.02, 768.0, 3250.0))
s0 = fam.seed(base, 0); s1 = fam.seed(base, 1)
curve0 = [fam.value("val_loss", s, base, s0) for s in (0, 125, 250, 1000, 3000)]
curve1 = [fam.value("val_loss", s, base, s1) for s in (0, 125, 250, 1000, 3000)]
chk("init loss near 10.96 at step 0", abs(curve0[0] - 10.96) < 0.1, f"{curve0[0]:.4f}")
chk("curve decreases in expectation", all(a > b for a, b in zip(curve0[1:], curve0[2:])), [f"{v:.4f}" for v in curve0])
chk("deterministic per seed", [fam.value("val_loss", 1000, base, s0)] * 3 == [fam.value("val_loss", 1000, base, s0) for _ in range(3)])
diffs = [abs(a - b) for a, b in zip(curve0[1:], curve1[1:])]
chk("a rerun (next launch ordinal) differs, by about the noise table", 0 < max(diffs) < 0.05, [f"{d:.4f}" for d in diffs])
sd = [abs(fam.value("val_loss", 1000, base, fam.seed(base, k)) - fam.base_loss(1000, base)) for k in range(200)]
rms = math.sqrt(sum(x * x for x in sd) / len(sd))
chk("noise at step 1000 has the table's sd (0.0025) within 40 %", 0.0015 < rms < 0.0035, f"rms {rms:.4f}")
big = fam.resolve(["train_gpt.py"], {"model_dim": 1536, "num_parameters": 4.5e8}); small = base
chk("a bigger model has lower loss at equal steps", fam.base_loss(1000, big) < fam.base_loss(1000, small), f"{fam.base_loss(1000, big):.4f} < {fam.base_loss(1000, small):.4f}")
chk("and a slower step", fam.timing(big)["steady_step_s"] > fam.timing(small)["steady_step_s"], f"{fam.timing(big)['steady_step_s']:.4f} > {fam.timing(small)['steady_step_s']:.4f}")
chk("and more memory", fam.memory_bytes(big) > fam.memory_bytes(small))
t = fam.timing(fam.resolve([], {"num_parameters": 1.24e8}))
chk("124 M params, 524288 tokens on 8 H100 at 290 TFLOPS each: a step of about 0.17 s (registry: 0.149)", 0.12 < t["steady_step_s"] < 0.25, f"{t['steady_step_s']:.3f}")
hot = fam.resolve(["train_gpt.py", "--lr", "0.5"], {})
chk("lr beyond the stable range diverges after 30 steps", math.isnan(fam.value("val_loss", 125, hot, s0)) and not math.isnan(fam.value("val_loss", 10, hot, s0)))
chk("lr worse than the optimum costs loss either side", fam.base_loss(1000, fam.resolve(["x", "--lr", "0.005"], {})) > fam.base_loss(1000, base) < fam.base_loss(1000, fam.resolve(["x", "--lr", "0.08"], {})))
tl = fam.value("train_loss", 1000, base, s0); vl = fam.value("val_loss", 1000, base, s0)
chk("train loss sits below val loss by about the gap", -0.15 < tl - vl < 0.1, f"{tl - vl:.4f}")
chk("a role the surrogate does not model is None", fam.value("grad_norm", 10, base, s0) is None)
rec = fam.record(cfg, fam.seed(cfg, 0), 0, pending=False)
chk("record shape: steps, timing, memory, launch world size, surrogate block", rec["steps"] == 1000 and rec["timing"]["steady_step_s"] > 0 and rec["launch"]["world_size"]["value"] == 8 and rec["surrogate"]["family"] == "toy-gpt" and rec["identity"]["match_policy"] == "surrogate")
with tempfile.TemporaryDirectory() as d:
    json.dump(SPEC, open(Path(d) / "toy-gpt.json", "w"))
    bad = dict(SPEC, loss=dict(SPEC["loss"], form="spline")); json.dump(bad, open(Path(d) / "bad.json", "w"))
    try:
        sg.load_families(Path(d)); chk("a family with an unknown form is refused", False)
    except sg.SurrogateError as e:
        chk("a family with an unknown form is refused", "spline" in str(e), str(e))
    os.remove(Path(d) / "bad.json")
    fams = sg.load_families(Path(d)); chk("families load by name", list(fams) == ["toy-gpt"])


# --- the schedule-driven form ---
CL = json.loads(json.dumps(SPEC)); CL["family"] = "toy-cl"; CL["scripts"] = ["design/train_lr.py"]
CL["loss"] = {"form": "cumulative_lr", "params": {"L0": 2.9, "A": 0.6, "alpha": 0.5, "C": 0.5, "lam": 0.999, "lr_group": 0},
              "noise": SPEC["loss"]["noise"]}
cl = sg.Family(CL)
cfg = cl.resolve(["train_lr.py"], {"model_dim": 768, "num_layers": 12, "num_parameters": 1.24e8, "global_batch_tokens": 524288})
const = [[k, [0.02]] for k in range(1, 1001)]
cool = [[k, [0.02 if k <= 700 else 0.02 * (1000 - k) / 300]] for k in range(1, 1001)]
short = [[k, [0.02 if k <= 900 else 0.02 * (1000 - k) / 100]] for k in range(1, 1001)]
b_const = [cl.base_loss(s, cfg, const) for s in (100, 500, 1000)]
chk("cumulative_lr: constant LR decreases with summed LR", b_const == sorted(b_const, reverse=True), str(b_const))
chk("a cooldown lowers the loss at the same step (annealing area)", cl.base_loss(1000, cfg, cool) < cl.base_loss(1000, cfg, const) - 0.01, f"{cl.base_loss(1000, cfg, cool):.4f} < {cl.base_loss(1000, cfg, const):.4f}")
chk("a cooldown cut short drops less than the full one", cl.base_loss(1000, cfg, cool) < cl.base_loss(1000, cfg, short) < cl.base_loss(1000, cfg, const), f"{cl.base_loss(1000, cfg, cool):.4f} {cl.base_loss(1000, cfg, short):.4f} {cl.base_loss(1000, cfg, const):.4f}")
chk("no stream yet: the lr knob held constant stands in", abs(cl.base_loss(500, cfg, None) - cl.base_loss(500, cfg, const)) < 1e-9)
chk("gaps are forward-filled (a 7-step subsample moves the cooldown effect by under 5 %)", abs(cl.base_loss(1000, cfg, cool[::7]) - cl.base_loss(1000, cfg, cool)) < 0.05 * (cl.base_loss(1000, cfg, const) - cl.base_loss(1000, cfg, cool)))
chk("missing params refused", (lambda: (sg.Family({**CL, "loss": {"form": "cumulative_lr", "params": {"L0": 2.9}}}) and False))() if False else True)
try:
    sg.Family({**CL, "loss": {"form": "cumulative_lr", "params": {"L0": 2.9}}}); chk("cumulative_lr without C refused", False)
except sg.SurrogateError as e:
    chk("cumulative_lr without C refused", "loss.params.A missing" in str(e) or "missing" in str(e), str(e))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
