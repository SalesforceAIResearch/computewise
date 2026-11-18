#!/usr/bin/env bash
# A script that changes directory before its first CUDA touch must still be
# identified. H4's recipe scripts chdir to their own directory before their
# data check; the patch then read sys.argv[0] ("runs/c4/train_gpt.py") against
# the NEW directory, found nothing, and sent no entry hash and no script path,
# so all 21 lookups of W0-h4-measure (2026-10-24, filter box b5890d2b) were
# `match none` and every job OOM'd on the model path, although each record was
# path_bound to that path with the staged script's exact hash. The entry
# script is now resolved once against the launch directory (sitecustomize's
# LAUNCH_CWD, or __main__.__file__), and hello carries that directory.
# Three launches of one stand-in against a one-record registry shaped like
# H4's c4 record (path_bound, bound_path runs/c4/train_gpt.py, argv
# ["train_gpt.py", "--seed", "0"], seed 0, entry_hash = the script's bytes):
#   1. chdir to the script's directory BEFORE `import torch`;
#   2. chdir AFTER `import torch`, before the first CUDA touch;
#   3. no chdir, launched from the project directory (regression).
# Each must be served the record (match exact or path, the served losses) with
# entry_hash_live equal to the script's sha256 and the script path resolved.
# Usage (on the VM):  sudo IMG=<box image id> [PATCH_SRC=1] bash tests/test_chdir_script_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work/runs/c4; cd /tmp/work
write_script() {  # $1 = before | after | none
python - "$1" <<PY
import sys
mode = sys.argv[1]
body = """import os, sys
MODE = %r
if MODE == "before":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
import torch
if MODE == "after":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
torch.manual_seed(seed)
print("cwd at first cuda touch:", os.getcwd(), flush=True)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(8, 16, device="cuda")
for step in range(3):
    loss = model(x).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    print(f"step {step} loss {loss.item():.4f}", flush=True)
print("done", flush=True)
""" % mode
open("runs/c4/train_gpt.py", "w").write(body)
PY
}
write_record() {  # $1 = sha256 of the script
python - "$1" <<PY
import json, sys
json.dump({"identity": {"record_id": "standin-h4-c4", "kind": "train", "entry_hash": sys.argv[1],
                        "argv": ["train_gpt.py", "--seed", "0"], "seed": {"value": 0},
                        "match_policy": "path_bound", "bound_path": "runs/c4/train_gpt.py",
                        "replicate_group": "standin-h4-c4",
                        "derived_from": {"kind": "synthesised_filter_variant", "record": "parent", "steps": [1, 3]}},
           "mode": "replay", "steps": 3, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.15},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
           "series": {"train_loss": {"steps": [1, 2, 3], "values": [2.0, 1.9, 1.8]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/standin.json", "w"))
PY
}
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = "$3" ]; then echo "[ok] $1: $2"; else echo "[FAIL] $1: got $2 want $3"; fails=$((fails+1)); fi; }
for mode in before after none; do
  write_script $mode
  SHA=$(sha256sum runs/c4/train_gpt.py | cut -c1-64)
  write_record "$SHA"
  python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ-$mode.jsonl > /tmp/daemon-$mode.log 2>&1 &
  D=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
  python runs/c4/train_gpt.py --seed 0 > r-$mode.out 2> r-$mode.err; rc=$?
  for i in $(seq 100); do grep -q "\"kind\":\"job_bye\"" /tmp/occ-$mode.jsonl 2>/dev/null && break; sleep 0.1; done
  kill $D 2>/dev/null; wait $D 2>/dev/null; rm -f /tmp/gm.sock
  L=$(grep "\"kind\":\"lookup\"" /tmp/occ-$mode.jsonl | head -1)
  echo "--- chdir $mode (exit $rc): $(grep "cwd at first" r-$mode.out)"; echo "    lookup: ${L:0:420}"
  chk "chdir $mode: exits 0 and finishes" "$rc:$(grep -c "^done" r-$mode.out)" "0:1"
  chk "chdir $mode: the lookup matched (not none)" "$(echo "$L" | grep -c "\"match\":\"none\"")" 0
  chk "chdir $mode: served the record standin-h4-c4" "$(echo "$L" | grep -c "\"record_id\":\"standin-h4-c4\"")" 1
  chk "chdir $mode: served losses are the record (2.0, 1.9, 1.8)" "$(grep -c "loss 2.0000\|loss 1.9000\|loss 1.8000" r-$mode.out)" 3
done
echo "$((n-fails)) passed, $fails failed"
[ "$fails" = 0 ]'
