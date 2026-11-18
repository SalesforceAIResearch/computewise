#!/usr/bin/env bash
# next/ (after r6): a launch that sees no CUDA device runs on the CPU, as on
# hardware, even when its argv matches a record. The real lagkv selection
# evaluator on gsm8k, --max-examples 1, one daemon (speed 50), the patch under
# test mounted over the box's (PATCH_SRC):
#   EMPTY  CUDA_VISIBLE_DEVICES=  (G-kvrep-haiku-L2's empty pool token): rc 0,
#          TEST_METRICS printed from a real CPU decode, the daemon logs
#          replay_disengaged reason no_visible_cuda_device, no exception event;
#   BAD    CUDA_VISIBLE_DEVICES=7 (no such GPU on a 4-GPU box): the same;
#   GPU0   CUDA_VISIBLE_DEVICES=0: served as on r6 (rc 0, lookup exact, no disengage);
#   UNSET  CUDA_VISIBLE_DEVICES unset: served (all four visible).
# On the r6 box without the fix, EMPTY and BAD fail with
#   AttributeError: 'Tensor' object has no attribute 'past_key_values'.
# Usage (on the E2):  sudo IMG=sha256:<KV box> G=<tree> [PATCH_SRC=1] bash tests/test_no_cuda_in_image.sh
set -uo pipefail
IMG=${IMG:-sha256:5994258c9617cee5b4db7566f93afbd1c020d2e480b5ce4d120f0969b07f39ee}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
HF=/srv/cache/hf/mlsbench-kv/02c58c389f0f/hf; KIT=/srv/work/gpu-mock/mls-kit; TREE=/srv/cache/tasks/mls-kv-deploy/b85d30e78b4d/project
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro")
HFC=$(mktemp -d /srv/work/gpu-mock/hfcopy.XXXX); for x in datasets xet STAGED.json; do [ -e "$HF/$x" ] && cp -a "$HF/$x" "$HFC/"; done; mkdir -p "$HFC/hub"
docker run --entrypoint '' --rm --cpus 8 --memory 32g --network none "${M[@]}" -v "$HFC:/hf" -v "$HF/hub:/hf/hub:ro" -v "$KIT:/kit:ro" -v "$TREE:/tree:ro" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/out /tmp/work; cd /tmp/work; cp /kit/records/mls-llm-kv-*.json /tmp/reg/; cp -a /tree/methods methods
mkdir -p _task; echo "# t" > _task/task_description.md; export MLSBENCH_TASK_DIR=/tmp/work/_task
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 50 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
A="--workload gsm8k --model-id Qwen/Qwen2.5-3B-Instruct --compression-ratio 0.8 --max-examples 1 --seed 42"
run() { local t=$1; shift; local t0=$(date +%s); (cd methods/lagkv && env "$@" timeout 1500 python3 custom_selection_eval.py $A > /tmp/out/$t.out 2>&1); echo "$t rc=$? wall=$(( $(date +%s) - t0 ))s" >> /tmp/out/rc.txt; }
run EMPTY CUDA_VISIBLE_DEVICES=
run BAD CUDA_VISIBLE_DEVICES=7
run GPU0 CUDA_VISIBLE_DEVICES=0
run UNSET -u CUDA_VISIBLE_DEVICES
kill $D 2>/dev/null; wait $D 2>/dev/null
cat /tmp/out/rc.txt
for t in EMPTY BAD GPU0 UNSET; do echo "--- $t:"; grep -hE "TEST_METRICS|AttributeError|Traceback" /tmp/out/$t.out | head -3 | cut -c1-160; done
python3 - <<PY
import json
ev = [json.loads(l) for l in open("/tmp/out/occ.jsonl")]
names = {}
for e in ev:
    if e.get("kind") in ("lookup", "exception", "replay_disengaged"):
        print(e["kind"], e.get("job"), e.get("match"), (e.get("record_id") or "")[11:60], e.get("reason"), repr(e.get("cuda_visible_devices")), (e.get("message") or "")[:60])
PY
'
rc=$?; rm -rf "$HFC"; exit $rc
