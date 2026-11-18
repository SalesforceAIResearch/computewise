#!/usr/bin/env bash
# Surrogate serving inside the box image (docs/SURROGATE-SERVING.md §3–§4):
# a script a family names, with no record covering it, is synthesised in two
# phases, every served value is labelled, a rerun is a different realisation,
# a divergent configuration goes NaN once with a failure event, the LR stream
# the script's own schedule produces reaches the daemon, and the printed
# losses are the family's curve and not the real CPU arithmetic.
# Three launches of one stand-in (the toy family of tests/toy_family.py, with
# `design/train_gpt.py` as its script): --lr 0.02 twice, --lr 0.2 once.
# Usage (on the VM):  sudo IMG=sha256:<box id> [PATCH_SRC=1] bash tests/test_surrogate_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
M=(-v "$G/src:/gm/src:ro" -v "$G/tests:/gm/tests:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg/records /tmp/reg/surrogates /tmp/work/design /tmp/out; cd /tmp/work
python - <<PY
import json, sys
sys.path.insert(0, "/gm/tests")
from toy_family import SPEC
spec = json.loads(json.dumps(SPEC))
spec["knobs"]["train_steps"] = {"flag": "--steps", "default": 30}
spec["loss"]["divergence"] = {"when": {"lr_above": 0.12}, "after_steps": 5, "then": "nan"}
spec["validation_steps"] = {"every": 10, "and_last": True}
json.dump(spec, open("/tmp/reg/surrogates/toy-gpt.json", "w"))
PY
cat > design/train_gpt.py <<PY
import sys, math, torch
lr = float(sys.argv[sys.argv.index("--lr") + 1]) if "--lr" in sys.argv else 0.02
steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 30
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
decay = [p for p in model.parameters() if p.ndim >= 2]; other = [p for p in model.parameters() if p.ndim < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}], lr=lr)
x = torch.randn(8, 16, device="cuda"); xv = torch.randn(8, 16, device="cuda")
def lr_at(k):   # warm-up over 3 steps, hold, linear cooldown over the last third
    if k <= 3: return lr * k / 3
    c0 = steps * 2 // 3
    if k > c0: return lr * (steps - k + 1) / (steps - c0)
    return lr
for step in range(1, steps + 1):
    for g in opt.param_groups: g["lr"] = lr_at(step)
    loss = model(x).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if not math.isfinite(loss.item()):
        print(f"nan at step {step}", flush=True); sys.exit(3)
    print(f"step {step} lr {lr_at(step):.5f} train_loss {loss.item():.4f}", flush=True)
    if step % 10 == 0 or step == steps:
        model.eval()
        with torch.no_grad():
            vl = model(xv).mean().item()
        model.train()
        print(f"val step {step} val_loss {vl:.4f}", flush=True)
print("done", flush=True)
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg/records --gpus 4 --speed 20 --occupancy /tmp/out/occupancy.jsonl > /tmp/out/daemon.log 2>&1 &
D=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
# the daemon logs job_bye when the job'"'"'s socket closes; wait for it before the next launch
wait_bye() { for i in $(seq 100); do c=$(grep -c "\"kind\":\"job_bye\"" /tmp/out/occupancy.jsonl 2>/dev/null); [ "${c:-0}" -ge "$1" ] && return; sleep 0.1; done; echo "job_bye $1 not seen in 10 s"; }
python design/train_gpt.py --lr 0.02 --steps 30 > a.out 2> a.err; ra=$?; wait_bye 1
python design/train_gpt.py --lr 0.02 --steps 30 > b.out 2> b.err; rb=$?; wait_bye 2
python design/train_gpt.py --lr 0.2 --steps 30 > c.out 2> c.err; rc=$?; wait_bye 3
kill $D 2>/dev/null; wait $D 2>/dev/null
echo "--- run A (exit $ra):"; head -4 a.out; grep "^val\|^done" a.out
echo "--- run B (exit $rb):"; grep "^val\|^done" b.out
echo "--- run C (exit $rc):"; tail -2 c.out
echo "--- daemon:"; grep -i "surrogate famil\|error\|traceback" /tmp/out/daemon.log | head -3
python /gm/tests/surrogate_probe_check.py /tmp/out "$ra" "$rb" "$rc" /tmp/work'
