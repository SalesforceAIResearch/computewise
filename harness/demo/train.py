#!/usr/bin/env python3
"""A stand-in training run for trying the gpu-harness without real training: no GPU work, no dependencies.

    python3 train.py --config configs/<name>.json --seed <n>

It prints progress the way a real training script does (step:N/M ... val_loss ... train_time), sleeping between steps,
and ends with `final val_loss <x>`. Each config sets the shape of its loss curve, so some configs are clearly better
than others, and one goes wrong early. The run takes DEMO_SECONDS seconds (default 60) whatever the step count."""
import argparse, json, math, os, random, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
cfg = json.load(open(a.config))
steps, every = int(cfg.get("steps", 1000)), 50
seconds = float(os.environ.get("DEMO_SECONDS", "60"))
rng = random.Random(f"{os.path.basename(a.config)}:{a.seed}")
print(f"config {cfg['name']} seed {a.seed} on CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}", flush=True)
t0 = time.time()
for step in range(every, steps + 1, every):
    time.sleep(seconds * every / steps)
    frac = step / steps
    loss = cfg["floor"] + (cfg["start"] - cfg["floor"]) * math.exp(-cfg["rate"] * frac * 5)
    if cfg.get("diverge_after"):                       # a setup that goes wrong partway through
        loss += max(0.0, frac - cfg["diverge_after"]) * cfg.get("diverge_slope", 0.0)
    loss += rng.gauss(0, cfg.get("noise", 0.01))
    print(f"step:{step}/{steps} val_loss:{loss:.4f} train_time:{(time.time() - t0) * 1000:.0f}ms", flush=True)
print(f"final val_loss {loss:.4f}", flush=True)
