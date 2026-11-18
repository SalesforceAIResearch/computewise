#!/usr/bin/env bash
# Inside an image with gpu-mock installed (/opt/gpu-mock): the torch patch's
# device count and the daemon GPU that replay paces on, under the launch
# environments the tasks use. No daemon needed: replay._daemon_gpu() is pure.
# Usage (on the VM):  sudo IMG=gm-dpo:x86 bash tests/test_gpu_index_in_image.sh
set -uo pipefail
IMG="${IMG:-gm-dpo:x86}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
G=$(cd "$(dirname "$0")/.." && pwd)
PATCH=(); [ "${PATCH_SRC:-0}" = 1 ] && PATCH=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
pass=0; fail=0
# env | expected "device_count daemon_gpu"
while IFS='|' read -r envs want; do
  got=$(docker run --entrypoint '' --rm --cpus 2 --memory 4g "${PATCH[@]}" "$IMG" bash -c "source /usr/local/nvidia/env.sh; unset LD_PRELOAD; env $envs python -c 'import torch, replay; print(torch.cuda.device_count(), replay._daemon_gpu())'" 2>/dev/null | tail -1)
  if [ "$got" = "$want" ]; then pass=$((pass+1)); m="ok  "; else fail=$((fail+1)); m="FAIL"; fi
  echo "[$m] ${envs:-<no CVD, no LOCAL_RANK>} -> '$got' (want '$want')"
done <<'CASES'
|4 0
CUDA_VISIBLE_DEVICES=2|1 2
CUDA_VISIBLE_DEVICES=2,3 LOCAL_RANK=1|2 3
CUDA_VISIBLE_DEVICES=3 LOCAL_RANK=3|1 3
LOCAL_RANK=3|4 3
CUDA_VISIBLE_DEVICES=3,1|2 3
CASES
echo; echo "$pass passed, $fail failed"; [ "$fail" = 0 ]
