#!/usr/bin/env bash
# next/ r8.2 known answer: group-aware get_rank / get_world_size and NCCL->gloo
# new_group, for a model-parallel SINGLETON subgroup (NeMo PP=1, TP=4). Reproduces
# Astra's subgroup-probe-4 (2026-11-02): before the fix the getters ignored the
# group and returned the GLOBAL rank/world size, so a singleton PP group reported
# rank 0..3 and size 4 instead of rank 0 and size 1, failing NeMo's
# init_model_parallel assertion. The dist hooks engage only after a first CUDA
# touch with a live daemon, so this test does that before creating groups, exactly
# as NeMo does.
#
# 4 ranks (torchrun), daemon --gpus 4. Each rank r:
#   - inits process group backend=nccl (-> hidden gloo), touches CUDA (engages replay hooks),
#   - makes a SINGLETON group g_r = new_group([r]) for EVERY rank r' (so each rank holds 4
#     singleton group handles), then for its OWN singleton queries get_rank(g_r)/get_world_size(g_r)
#     (member: expect 0 / 1) and for another rank's singleton queries membership (nonmember: expect -1),
#   - also an explicit new_group(ranks=[0,1], backend="nccl") must NOT raise (NCCL->gloo).
# PASS: every member query (0,1), every nonmember -1, no ProcessGroupNCCL no-GPU error.
# CONTROL (NVRT_NO_DISTGROUP_FIX unset vs a frozen r8 tree) shows the old global 0..3 / 4.
#
# Usage (on a box):  sudo IMG=<box> G=<r8.2 tree> [PATCH_SRC=1] bash tests/test_subgroup_rank_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
FAKEGPU=${FAKEGPU:-/srv/work/gpu-mock/FakeGPU}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
W=$(mktemp -d /srv/work/gpu-mock/subgrp.XXXX)
mkdir -p "$W/reg"
cat > "$W/prep.sh" <<'PREP'
set -e
SRC=/gm/FakeGPU/fakegpu
mkdir -p /gm/pylib; rm -rf /gm/pylib/nvcompat; cp -r "$SRC" /gm/pylib/nvcompat
find /gm/pylib/nvcompat -name __pycache__ -type d -prune -exec rm -rf {} +
sed -i -E 's/^(\s*)import fakegpu$/\1import nvcompat as fakegpu/; s/^(\s*)from fakegpu import/\1from nvcompat import/' /gm/pylib/nvcompat/*.py
PREP
cat > "$W/sg.py" <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group(backend="nccl")
r = int(os.environ["RANK"]); ws = int(os.environ["WORLD_SIZE"])
# first CUDA touch -> engages the replay/dist hooks (as NeMo does before groups)
_ = torch.zeros(4, device="cuda") + 1
groups = {i: dist.new_group([i]) for i in range(ws)}       # one singleton per rank
mine = groups[r]
mem_rank = dist.get_rank(group=mine)
mem_ws = dist.get_world_size(group=mine)
other = (r + 1) % ws
non_rank = dist.get_rank(group=groups[other])              # nonmember -> expect -1
# an explicit NCCL new_group must not raise on a no-GPU host (NCCL->gloo translate)
nccl_ok = True
try:
    _ = dist.new_group(ranks=list(range(ws)), backend="nccl")
except Exception as e:
    nccl_ok = False; print(f"RANK{r} NCCL_NEW_GROUP_RAISED {type(e).__name__}: {e}", flush=True)
ok = (mem_rank == 0 and mem_ws == 1 and non_rank == -1 and nccl_ok)
print(f"RANK{r} member_rank={mem_rank} member_ws={mem_ws} nonmember_rank={non_rank} nccl_ok={nccl_ok} PASS={ok}", flush=True)
dist.destroy_process_group()
PY
echo "=== r8.2 subgroup known answer $(date -u +%FT%TZ)  image $IMG"
echo "patch $(cd "$G/src/torch_patch" && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
cat > "$W/run.sh" <<RUN
set -uo pipefail
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/w/gm.sock
python3 -I /gm/src/daemon/gpu_daemon.py --socket /w/gm.sock --registry /w/reg --gpus 4 --speed 50 --occupancy /w/occ.jsonl > /w/daemon.log 2>&1 &
for i in \$(seq 200); do [ -S /w/gm.sock ] && break; sleep 0.1; done
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29555
torchrun --nproc_per_node=4 --nnodes=1 /w/sg.py
RUN
sudo docker run --entrypoint '' --rm --cpus 8 --memory 16g --network none \
  -v "$G/src:/gm/src:ro" -v "$FAKEGPU:/gm/FakeGPU:ro" \
  -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
  -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
  -v "$W:/w" -e NVRT_SOCKET=/w/gm.sock -e FAKEGPU_PROFILE_DIR=/gm/FakeGPU/profiles \
  -e NVRT_PROFILE=h100 -e PYTHONPATH=/gm/pylib "$IMG" \
  bash -c 'bash /w/prep.sh; bash /w/run.sh' 2>&1 | grep -E "RANK[0-9]|PASS|RAISED|Error|Traceback" | head -20
echo "=== expect: every rank member_rank=0 member_ws=1 nonmember_rank=-1 nccl_ok=True PASS=True ==="
rm -rf "$W"
echo "=== done $(date -u +%FT%TZ) ==="
