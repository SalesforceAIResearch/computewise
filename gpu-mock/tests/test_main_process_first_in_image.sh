#!/usr/bin/env bash
# Does accelerate's local_main_process_first() serialise ranks under the patch,
# as it does on a real node? In the DPO replay all four ranks ran the 32-process
# dataset map at the same time (four progress bars, then a cgroup OOM kill).
# On a real node rank 0 maps first and the others read its cache. The main rank busy-waits 3 s
# on the REAL clock (time.clock_gettime is not patched), so the check does not
# depend on time.sleep. Run patched and unpatched (control: accelerate on CPU, gloo).
# Usage (on the VM):  sudo IMG=gm-dpo:x86 [PATCH_SRC=1] bash tests/test_main_process_first_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"
M=(); [ "${PATCH_SRC:-0}" = 1 ] && M=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
cd /tmp; cat > mpf.py <<PY
import os, time
R = lambda: time.clock_gettime(time.CLOCK_REALTIME)
from accelerate import PartialState
s = PartialState(); t0 = float(os.environ["T0"])
with s.local_main_process_first():
    t_in = R() - t0
    if s.is_local_main_process:
        u = time.clock_gettime(time.CLOCK_MONOTONIC)
        while time.clock_gettime(time.CLOCK_MONOTONIC) - u < 3.0:
            pass
    t_out = R() - t0
import torch.distributed as dist
print(f"MPF rank={s.process_index} type={s.distributed_type} n={s.num_processes} backend={dist.get_backend() if dist.is_initialized() else None} enter={t_in:.2f} leave={t_out:.2f}", flush=True)
PY
# The check is about accelerate (TRL images); an image without it (the mng3
# packages) has nothing to check here, so say so instead of printing nothing.
python -c "import accelerate" 2>/dev/null || { echo "SKIP: accelerate is not in this image, so local_main_process_first has nothing to serialise here"; exit 0; }
for mode in patched unpatched; do
  [ "$mode" = unpatched ] && unset PYTHONPATH
  echo "--- $mode"
  T0=$(python -c "import time; print(time.clock_gettime(time.CLOCK_REALTIME))") accelerate launch --num_processes 4 mpf.py 2>/dev/null | grep "^MPF" | sort
done'
