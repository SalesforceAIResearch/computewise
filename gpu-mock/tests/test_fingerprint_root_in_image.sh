#!/usr/bin/env bash
# Root selection for the fingerprint, inside an image with gpu-mock installed:
# wrappers are unwrapped (DDP, torch.compile, peft) and a no-grad forward that
# comes first does not claim the root. Uses the dry-run mode
# (NVRT_FINGERPRINT_ONLY=1 prints the inputs JSON at the first optimizer step).
# Usage (on the VM):  sudo IMG=gm-dpo:x86 [PATCH_SRC=1] bash tests/test_fingerprint_root_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" -v /srv/work/gpu-mock/hf-cache:/hf:ro -e HF_HOME=/hf "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_FINGERPRINT_ONLY=1 NVRT_NO_REPLAY=1
cd /tmp
cat > common.py <<PY
import json, os, sys, torch, torch.nn as nn
class Tiny(nn.Module):
    def __init__(self, d=16, v=50, L=2):
        super().__init__(); self.wte = nn.Embedding(v, d); self.h = nn.ModuleList([nn.Linear(d, d) for _ in range(L)]); self.head = nn.Linear(d, v)
    def forward(self, x):
        y = self.wte(x)
        for l in self.h: y = torch.relu(l(y))
        return self.head(y).float().mean()
def train_one(model, x):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(x); loss.backward(); opt.step()   # the dry run prints and exits at this step
PY
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "[ok  ] $1 -> $2"; else fail=$((fail+1)); echo "[FAIL] $1 -> $2 (want $3)"; fi; }
fp() { grep -o "NVRT_FINGERPRINT .*" | sed "s/NVRT_FINGERPRINT //" | python -c "import json,sys; i=json.loads(sys.stdin.readline())[\"inputs\"]; print(i[\"model\"], i[\"num_parameters\"], i[\"micro_batch\"])"; }
# 1. plain tiny model: model=tiny, params = 50*16 + 2*(16*16+16) + 16*50+50 = 800+544+850 = 2194
out=$(python -c "from common import *; m=Tiny().cuda(); train_one(m, torch.zeros(4,8,dtype=torch.long,device=\"cuda\"))" 2>/dev/null | fp)
chk "plain: model, params, micro_batch" "$out" "tiny 2194 4"
# 2. DDP-wrapped (1 rank over the fake nccl): unwrapped to tiny
out=$(RANK=0 WORLD_SIZE=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=29511 python -c "
from common import *; import torch.distributed as dist
dist.init_process_group(backend=\"nccl\"); m=nn.parallel.DistributedDataParallel(Tiny().cuda()); train_one(m, torch.zeros(4,8,dtype=torch.long,device=\"cuda\"))" 2>/dev/null | fp)
chk "DDP wrapper unwrapped" "$out" "tiny 2194 4"
# 3. a no-grad forward of a DIFFERENT, bigger module first (a reference model): the trained one is the root
out=$(python -c "
from common import *
ref=Tiny(d=32).cuda(); m=Tiny().cuda(); x=torch.zeros(4,8,dtype=torch.long,device=\"cuda\")
with torch.no_grad(): ref(x)
train_one(m, x)" 2>/dev/null | fp)
chk "no-grad reference forward first does not claim the root" "$out" "tiny 2194 4"
# 4. peft LoRA on gpt2: the base model count, 124,439,808 (gpu-tasks artefacts.model_parameters)
if python -c "import transformers, peft" 2>/dev/null; then
out=$(python -c "
from common import *
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
m = AutoModelForCausalLM.from_pretrained(\"gpt2\"); m = get_peft_model(m, LoraConfig(r=8, target_modules=[\"c_attn\"])).cuda()
x = torch.zeros(2,8,dtype=torch.long,device=\"cuda\")
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
o = m(input_ids=x, labels=x); (o.loss if hasattr(o, \"loss\") else o).backward(); opt.step()" 2>/dev/null | fp)
chk "peft LoRA on gpt2: base model count" "$out" "gpt2 124439808 2"
else echo "[skip] peft LoRA on gpt2: transformers or peft is not installed in this image"; fi
echo; echo "$pass passed, $fail failed"; [ "$fail" = 0 ]'
