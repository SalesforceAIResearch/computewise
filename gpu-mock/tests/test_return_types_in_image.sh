#!/usr/bin/env bash
# Inside the box: ops that return a torch.return_types structseq must return the
# same type on a fake-cuda tensor as on a CPU tensor, with .values/.indices
# working (FakeGPU's tree-map rebuilt them as plain tuples; G-mlskv-haiku-L3,
# 2026-10-23), and a traceback through a fake-cuda op must show no frame from
# the patch or the vendored library (an agent read the wrapper's file name as
# the cause), unless NVRT_DEBUG=1.
# Usage (on the VM):  sudo IMG=sha256:<box id> [PATCH_SRC=1] bash tests/test_return_types_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
M=(-v "$G/tests:/gm/tests:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 2 --memory 4g "${M[@]}" "$IMG" bash -c 'source /usr/local/nvidia/env.sh; cd /tmp; python /gm/tests/return_types_check.py 2>&1 | grep -v Warning'
