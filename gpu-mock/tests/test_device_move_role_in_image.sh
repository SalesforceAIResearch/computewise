#!/usr/bin/env bash
# next/ r8.8 known answer: a served validation value survives a device move.
# Astra's TP4 NeMo fixture on r8.7 (2026-11-03): the daemon served val_loss 0.94 at
# step 2 (64 reads), yet MLPerf's callback logged eval_accuracy 0.0 and stopped with
# success. NeMo's inference_epoch_end reduces the per-batch losses and ends
# `avg_loss.type(torch.float32).cuda()`; FakeGPU's Tensor.to/.cuda()/.cpu() call
# torch's original on the inner CPU tensor and wrap the result, so the role tag
# that every other op's result keeps was dropped, and the broadcast, self.log,
# Lightning's reduction and its .item() read the stand-in's own zero.
#
# The stand-in runs 4 replayed training steps and a validation after steps 2 and 4
# (plus a warmup validation before step 1, as NeMo's run_training_warmup does).
# Each validation reduces exactly as NeMo's non-drop-last path (vstack of
# [loss_sum, count], sum, divide, .type(float32).cuda()), broadcasts in place, then
# goes through a Lightning-shaped _ResultMetric (clone, value*bs accumulate, divide
# by the cumulated batch size, detach) and convert_tensors_to_scalars' .item(), and
# applies MLPerf's stop rule (val_loss < 0.925 -> stop, success).
# Record: val_loss 0.94 at step 2, 0.92 at step 4.
#   DM    r8.8: step-2 reads 0.94 on every path (direct, after .cuda(), .cpu(),
#         .to("cpu"), the Lightning-shaped reduction), no stop; step-4 reads 0.92,
#         stop with success at step 4; 4 paced optimizer steps
#   DM0   replay off (NVRT_NO_REPLAY=1): the real computation, finite values, exit 0
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_device_move_role_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work /tmp/out; cd /tmp/work
cat > nemo_val.py <<PY
import os, torch
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(8, 16, device="cuda")

class ResultMetric:                         # the arithmetic of Lightning _ResultMetric (mean reduction)
    def __init__(self):
        self.value = torch.tensor(0.0, device="cuda"); self.n = torch.tensor(0, device="cuda")
    def update(self, v, bs):
        v = v.clone()
        self.value = self.value + v * bs; self.n = self.n + bs
    def compute(self):
        return (self.value.clone() / self.n).detach()

def validate(tag):
    model.eval()
    loss_vals = []
    with torch.no_grad():
        for _ in range(2):                  # two validation batches: [loss_sum, n_samples]
            loss = model(x).mean()
            loss_vals.append(torch.stack([loss * 4, torch.tensor(4.0, device="cuda")]))
    tot = torch.vstack(loss_vals).sum(axis=0)
    avg = tot[0] / tot[1]
    direct = float(avg.item())
    loss = avg.type(torch.float32).cuda()   # NeMo inference_epoch_end, non-drop-last path
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast(loss, 0)
    after_cuda = float(loss.item())
    host = float(loss.cpu().item()); host_to = float(loss.to("cpu").item())
    rm = ResultMetric(); rm.update(loss, 1)
    lightning = rm.compute().item()         # convert_tensors_to_scalars
    model.train()
    stop = lightning < 0.925                # MLPerf MetricsLogger.log_metrics
    print("VAL %s direct=%.4f after_cuda=%.4f cpu=%.4f to_cpu=%.4f lightning=%.4f stop=%s"
          % (tag, direct, after_cuda, host, host_to, lightning, stop), flush=True)
    return stop

validate("warmup")                          # run_training_warmup: before run_start, before step 1
for step in range(1, 5):
    loss = model(x).mean(); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if step in (2, 4):
        if validate("step%d" % step):
            print("RUN_STOP success at step %d" % step, flush=True)
            break
print("done", flush=True)
PY
python3 - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/nemo_val.py", "rb").read()).hexdigest()
json.dump({"identity": {"record_id": "standin-nemo-val", "entry_hash": h, "argv": ["nemo_val.py"],
                        "match_policy": "exact_only", "replicate_group": "standin-nemo-val"},
           "mode": "replay", "steps": 4, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.2, "eval_s": 0.2},
           "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
           "series": {"train_loss": {"steps": [1, 2, 3, 4], "values": [1.3, 1.2, 1.1, 1.0]},
                      "val_loss": {"steps": [2, 4], "values": [0.94, 0.92]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/nv.json", "w"))
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
CUDA_VISIBLE_DEVICES=0 timeout 120 python3 nemo_val.py > /tmp/out/DM.out 2>&1; echo "DM rc=$?"
CUDA_VISIBLE_DEVICES=0 NVRT_NO_REPLAY=1 timeout 120 python3 nemo_val.py > /tmp/out/DM0.out 2>&1; echo "DM0 rc=$?"
kill $D 2>/dev/null; wait $D 2>/dev/null
for t in DM DM0; do echo "--- $t:"; grep -E "^VAL|^RUN_STOP|^done|Error|Traceback" /tmp/out/$t.out | head -8; done
python3 - <<PY
import json, collections
c = collections.Counter()
for l in open("/tmp/out/occ.jsonl"):
    e = json.loads(l)
    if e.get("kind") == "enqueue": c[(e.get("job"), e.get("unit"))] += 1
print("enqueues (job, unit):", dict(c))
PY
'
