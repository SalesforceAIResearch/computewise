#!/usr/bin/env bash
# next/ r8.7 known answer: NeMo's optimizer path under replay. Astra's TP4 NeMo
# fixture on r8.6 (2026-11-03, all four ranks) got past the backward and died in
# the optimizer: Lightning's automatic optimization calls
# McoreDistributedOptimizer.step(closure), a torch Optimizer subclass whose
# __init__ never calls Optimizer.__init__ (so it is never neutralised), which
# runs the closure and then Megatron's DistributedOptimizer.step(); that one's
# clip_grad_norm ran TransformerEngine's multi-tensor kernel on the fake tensors
# ('expected input to be on cuda'), and no optimizer step was ever paced.
#
# The stand-in reproduces that structure (NeMo's mcore_optim.py, read from
# Astra's rebuilt image; megatron.core.optimizer.optimizer's class layout):
#   megatron/core/optimizer/optimizer.py   MegatronOptimizer (abstract step) and
#      MixedPrecisionOptimizer, whose step() clip-norms the grads, steps an inner
#      torch SGD and counts REAL_STEPS (the arithmetic replay must not run);
#   megatron/core/optimizer/distrib_optimizer.py, imported AFTER replay engaged:
#      DistributedOptimizer, whose own step() counts DIST_STEPS and calls super();
#   McoreDistributedOptimizer(torch.optim.Optimizer), no Optimizer.__init__, whose
#      step(closure) runs the closure and then the Megatron optimizer's step;
#   a Lightning-style loop: the forward and backward run INSIDE the closure, an
#      LR scheduler wraps the wrapper's step.
# Records: a 4-step step-series record keyed to the script.
#   MEG   replay: exit 0, 4 closures run (4 forwards, 4 backwards), 0 real Megatron
#         steps and 0 DistributedOptimizer steps, 4 paced steps, each wrapper step
#         returns the closure's loss
#   MEG0  replay off (NVRT_NO_REPLAY=1): the real steps run, 4 and 4, exit 0
#   PL    a plain torch SGD stepped with a closure under replay: 4 closures run,
#         4 paced steps (before r8.7 a neutralised step skipped the closure, so a
#         Lightning job paced one step for the whole run)
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_megatron_optimizer_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work/megatron/core/optimizer /tmp/out; cd /tmp/work
touch megatron/__init__.py megatron/core/__init__.py megatron/core/optimizer/__init__.py
cat > megatron/core/optimizer/optimizer.py <<PY
from abc import ABC, abstractmethod
import torch
REAL_STEPS = [0]
class MegatronOptimizer(ABC):
    def __init__(self, optimizer):
        self.optimizer = optimizer
    @abstractmethod
    def step(self):
        pass
    @property
    def param_groups(self):
        return self.optimizer.param_groups
    @param_groups.setter
    def param_groups(self, v):
        self.optimizer.param_groups = v
    @property
    def state(self):
        return self.optimizer.state
class MixedPrecisionOptimizer(MegatronOptimizer):
    def clip_grad_norm(self, clip):
        grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
        norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)) if grads else torch.zeros(())
        return float(norm)
    @torch.no_grad()
    def step(self):
        grad_norm = self.clip_grad_norm(1.0)
        self.optimizer.step()
        REAL_STEPS[0] += 1
        return True, grad_norm, None
PY
cat > megatron/core/optimizer/distrib_optimizer.py <<PY
from .optimizer import MixedPrecisionOptimizer
DIST_STEPS = [0]
class DistributedOptimizer(MixedPrecisionOptimizer):
    def step(self):
        DIST_STEPS[0] += 1
        return super().step()
PY
cat > nemo_like.py <<PY
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from megatron.core.optimizer import optimizer as mo     # imported before the first CUDA use, as NeMo does
MODE = os.environ.get("MODE", "meg")
class McoreDistributedOptimizer(torch.optim.Optimizer):
    def __init__(self, optim):
        self.defaults = {}
        self.mcore_optimizer = optim
    def zero_grad(self, set_to_none=True):
        self.mcore_optimizer.optimizer.zero_grad(set_to_none)
    def step(self, closure):
        loss = None
        if closure is not None:
            loss = closure()
        self.mcore_optimizer.step()
        return loss
    def _get_state(self):
        return self.mcore_optimizer.state if hasattr(self, "mcore_optimizer") else []
    def _set_state(self, value):
        self.mcore_optimizer.state = value
    state = property(_get_state, _set_state)
    def _get_param_groups(self):
        return self.mcore_optimizer.param_groups if hasattr(self, "mcore_optimizer") else []
    def _set_param_groups(self, value):
        self.mcore_optimizer.param_groups = value
    param_groups = property(_get_param_groups, _set_param_groups)
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
# distrib_optimizer loads after the first CUDA use (replay engaged): its class is
# wrapped when it is defined, through MegatronOptimizer.__init_subclass__
from megatron.core.optimizer import distrib_optimizer as do
inner = torch.optim.SGD(model.parameters(), lr=0.01)
opt = McoreDistributedOptimizer(do.DistributedOptimizer(inner)) if MODE != "plain" else inner
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
x = torch.randn(8, 32, device="cuda")
n = {"fwd": 0, "bwd": 0}
rets = []
for step in range(4):
    def closure():
        loss = model(x).mean(); n["fwd"] += 1
        loss.backward(); n["bwd"] += 1
        return loss
    r = opt.step(closure=closure)
    rets.append(r is not None)
    sched.step()
    opt.zero_grad(set_to_none=True)
print("closures fwd=%d bwd=%d real_megatron_steps=%d dist_steps=%d step_returned_loss=%s" % (n["fwd"], n["bwd"], mo.REAL_STEPS[0], do.DIST_STEPS[0], rets), flush=True)
print("done", flush=True)
PY
python3 - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/nemo_like.py", "rb").read()).hexdigest()
json.dump({"identity": {"record_id": "standin-nemo", "entry_hash": h, "argv": ["nemo_like.py"],
                        "match_policy": "exact_only", "replicate_group": "standin-nemo"},
           "mode": "replay", "steps": 4, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.2},
           "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
           "series": {"train_loss": {"steps": [1, 2, 3, 4], "values": [2.0, 1.5, 1.2, 1.0]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/nemo.json", "w"))
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
CUDA_VISIBLE_DEVICES=0 MODE=meg NVRT_DEBUG=1 timeout 120 python3 nemo_like.py > /tmp/out/MEG.out 2>&1; echo "MEG rc=$?"
CUDA_VISIBLE_DEVICES=0 MODE=meg NVRT_NO_REPLAY=1 timeout 120 python3 nemo_like.py > /tmp/out/MEG0.out 2>&1; echo "MEG0 rc=$?"
CUDA_VISIBLE_DEVICES=0 MODE=plain timeout 120 python3 nemo_like.py > /tmp/out/PL.out 2>&1; echo "PL rc=$?"
kill $D 2>/dev/null; wait $D 2>/dev/null
for t in MEG MEG0 PL; do echo "--- $t:"; grep -E "^closures|^done|Error|Megatron optimizer|Traceback" /tmp/out/$t.out | head -6; done
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
