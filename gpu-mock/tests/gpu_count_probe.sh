#!/usr/bin/env bash
# One line per box image: the GPU count its installed patch carries, nvidia-smi -L's count and
# torch.cuda.device_count(), e.g. "baked=4 smi=4 torch=4". Used by gate_sequence_r7.sh.
# Usage:  bash tests/gpu_count_probe.sh <image id> [<N>, default 4]
IMG=$1; N=${2:-4}
docker run --entrypoint '' --rm --network none -e N="$N" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
b=$(grep -c "_INSTALLED_DEVICE_COUNT = \"$N\"" /usr/local/nvidia/python/cuda_hooks.py)
s=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")
t=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
echo "baked=$b smi=$s torch=$t"' 2>&1 | tail -1
