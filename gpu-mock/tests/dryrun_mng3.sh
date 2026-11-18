#!/usr/bin/env bash
# Live fingerprint dry run: the adamh track-3 script (registry entry_hash
# 19ccac6c…) under torchrun, N ranks, with the gpu-mock patch loaded via the
# sitecustomize shim and NVRT_FINGERPRINT_ONLY=1. Every rank prints one
# NVRT_FINGERPRINT line at its first optimizer.step() and exits.
#
# The hash must equal fingerprint(record.identity.fingerprint_inputs) under the
# same fp256 version (compute it with src/torch_patch/fingerprint.py <record>).
#
# NVRT_DEVICE_COUNT=8 is set for THIS validation only, because the record
# was made on 8 GPUs and world_size is a fingerprint field; the mock box is 4.
#
# Usage: NPROC=8 PORT=29700 bash tests/dryrun_mng3.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
D="$G/.build-scratch/dryrun-mng3"
PY=/tmp/gpu-mock-prior-art/venv-fakegpu/bin
cd "$D"
export PYTHONPATH="$G/.build-scratch/site${PYTHONPATH:+:$PYTHONPATH}"
export NVRT_FINGERPRINT_ONLY=1
export NVRT_DEVICE_COUNT="${NVRT_DEVICE_COUNT:-8}"
export NVRT_NO_TIMING=1          # the fingerprint needs no pacing
export NVRT_DEBUG=1
NPROC="${NPROC:-8}"; PORT="${PORT:-29700}"

timeout "${TIMEOUT:-300}" "$PY/torchrun" --nproc_per_node="$NPROC" --master_port="$PORT" \
  train_gpt_simple_adamh.py > dryrun.log 2>&1
rc=$?
echo "torchrun exit: $rc"
echo "--- errors (if any):"; grep -v triton dryrun.log | grep -iE "Traceback|Error|error:" | head -10
echo "--- fingerprints by rank:"
grep -o 'NVRT_FINGERPRINT {.*' dryrun.log | python3 -I -c '
import sys, json
hs = set(); n = 0
for l in sys.stdin:
    d = json.loads(l.split(" ", 1)[1]); hs.add(d["fingerprint"]); n += 1
    print("rank", d["rank"], d["fingerprint"])
    if n == 1: print("  inputs:", json.dumps(d["inputs"]))
print("ranks reported:", n, "distinct hashes:", len(hs))
import os
exp = os.environ.get("EXPECTED")          # read from env: this block is single-quoted
if exp:
    print("MATCH" if hs == {exp} else "MISMATCH vs expected " + exp)
'
