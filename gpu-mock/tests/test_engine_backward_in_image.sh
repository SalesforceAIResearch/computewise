#!/usr/bin/env bash
# next/ r8.6 known answer: a backward that calls the C++ autograd engine directly
# (Megatron's custom_backward, schedules.py, taken when
# config.deallocate_pipeline_outputs) runs, under replay, the job's own grad-ready
# hooks on every parameter, as on hardware. Before r8.6 the engine saw an empty
# graph (the fake cuda tensor's graph lives on its inner CPU tensor), so no hook
# fired and Megatron's finish_grad_sync asserted "0/N params have grad available"
# (Astra's TP4 NeMo fixture, step-probe-1..4, 2026-11-02).
#
# The stand-in reproduces Megatron DDP's mechanism exactly:
#   - for each param: grad_acc = p.expand_as(p).grad_fn.next_functions[0][0];
#     grad_acc.register_hook(param_hook), param_hook asserting p.grad is not None,
#     doing p.main_grad.add_(p.grad.data), p.grad = None, then
#     bucket.register_grad_ready(p);
#   - each step: forward, then Variable._execution_engine.run_backward(...) on the
#     loss, then bucket.finish_grad_sync() (asserts every param was marked), then
#     optimizer.step(), then the bucket resets.
# Records: a 4-step step-series record keyed to the script.
#   ENG  run_backward path: 4 steps, every bucket full (8/8), exit 0, 4 paced steps
#   LB   the same with loss.backward() and NO hooks: plain stub, exit 0, 4 steps
#        (r8.6 changes nothing for a loss.backward() job)
#   ENG0 run_backward path with replay off (NVRT_NO_REPLAY=1, the modelled path):
#        a REAL backward on the inner CPU tensors, real hooks, 8/8, exit 0 (before
#        r8.6 FakeGPU's engine call reached an empty graph here too: 0/8)
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_engine_backward_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work /tmp/out; cd /tmp/work
cat > meg.py <<PY
import os, sys, torch
from torch.autograd import Variable
MODE = os.environ.get("MODE", "engine")
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 32),
                            torch.nn.ReLU(), torch.nn.Linear(32, 32), torch.nn.Linear(32, 1)).cuda()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.SGD(params, lr=0.01)
class Bucket:
    def __init__(s, ps): s.params = set(ps); s.params_with_grad = set()
    def register_grad_ready(s, p):
        assert p not in s.params_with_grad, "Cannot set grad twice"
        s.params_with_grad.add(p)
    def finish_grad_sync(s):
        assert len(s.params_with_grad) == len(s.params), (
            f"Communication call has not been issued for this bucket "
            f"({len(s.params_with_grad)}/{len(s.params)} params have grad available)")
    def reset(s): s.params_with_grad = set()
bucket = Bucket(params)
grad_accs = []
if MODE != "plain":
    for p in params:
        p.main_grad = torch.zeros_like(p)
        def make_hook(p):
            def param_hook(*unused):
                assert p.grad is not None, "param.grad being None is not safe when overlap_grad_reduce is True"
                p.main_grad.add_(p.grad.data)
                p.grad = None
                bucket.register_grad_ready(p)
            return param_hook
        ga = p.expand_as(p).grad_fn.next_functions[0][0]
        ga.register_hook(make_hook(p))
        grad_accs.append(ga)
x = torch.randn(8, 32, device="cuda")
for step in range(4):
    loss = model(x).mean()
    if MODE == "plain":
        loss.backward()
    else:
        Variable._execution_engine.run_backward(tensors=(loss,), grad_tensors=(torch.ones_like(loss),),
            keep_graph=False, create_graph=False, inputs=tuple(), allow_unreachable=True, accumulate_grad=True)
        bucket.finish_grad_sync()
        print(f"step {step} bucket {len(bucket.params_with_grad)}/{len(bucket.params)}", flush=True)
        bucket.reset()
    opt.step(); opt.zero_grad(set_to_none=True)
print("done", flush=True)
PY
python3 - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/meg.py", "rb").read()).hexdigest()
json.dump({"identity": {"record_id": "standin-meg", "entry_hash": h, "argv": ["meg.py"],
                        "match_policy": "exact_only", "replicate_group": "standin-meg"},
           "mode": "replay", "steps": 4, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.2},
           "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
           "series": {"train_loss": {"steps": [1, 2, 3, 4], "values": [2.0, 1.5, 1.2, 1.0]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/meg.json", "w"))
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
CUDA_VISIBLE_DEVICES=0 MODE=engine timeout 120 python3 meg.py > /tmp/out/ENG.out 2>&1; echo "ENG rc=$?"
CUDA_VISIBLE_DEVICES=0 MODE=plain  timeout 120 python3 meg.py > /tmp/out/LB.out  2>&1; echo "LB rc=$?"
CUDA_VISIBLE_DEVICES=0 MODE=engine NVRT_NO_REPLAY=1 timeout 120 python3 meg.py > /tmp/out/ENG0.out 2>&1; echo "ENG0 rc=$?"
kill $D 2>/dev/null; wait $D 2>/dev/null
for t in ENG LB ENG0; do echo "--- $t:"; grep -E "^step|^done|Error|assert|available" /tmp/out/$t.out | head -6; done
python3 - <<PY
import json, collections
c = collections.Counter()
for l in open("/tmp/out/occ.jsonl"):
    e = json.loads(l)
    if e.get("kind") == "enqueue" and e.get("unit") == "step": c[e.get("job")] += 1
    if e.get("kind") == "lookup": print("LOOKUP job", e.get("job"), e.get("match"), e.get("record_id"))
print("paced steps per job:", dict(c))
PY
'
