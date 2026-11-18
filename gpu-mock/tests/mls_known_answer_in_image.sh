#!/usr/bin/env bash
# The MLS-Bench KV real-script known answer (docs/MLS-KV-SERVING.md §4): the
# package's own script and wrapper, the staged offline HF cache, this tree's
# patch over the box's and this tree's daemon, the 40 records with their
# serving blocks. One launch of <policy> on <workload> at
# SELECTION_KV_MAX_EXAMPLES=<k> (0 = full); checks the TEST_METRICS line, the
# metrics file and the occupancy against the record.
# Usage (on the VM):  sudo IMG=sha256:<mlskv box> HF=<cache tree ending in /hf> KIT=<mls-kit dir> \
#        [POLICY=full_attention] [WORKLOAD=gsm8k] [K=20] [SPEED=20] [OUT=<host dir>] [DIRECT=1] bash tests/mls_known_answer_in_image.sh
#        DIRECT=1 (selection family only) launches `python custom_selection_eval.py --workload W --max-examples K` without the
#        job script, so the record is matched on the script's parsed defaults (match exact_effective)
#        HF_MODE=rw mounts the cache writable (the LongBench loaders write lock and extraction files; give them a copy, never the shared cache)
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:?mlskv box image id}"; HF="${HF:?hf cache tree}"; KIT="${KIT:?mls kit dir}"
POLICY="${POLICY:-full_attention}"; WORKLOAD="${WORKLOAD:-gsm8k}"; K="${K:-20}"; SPEED="${SPEED:-20}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro" -v "$G/tests:/gm/tests:ro" -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" -v "$HF:/hf:${HF_MODE:-ro}" -v "$KIT:/kit:ro")
if [ -n "${OUT:-}" ]; then mkdir -p "$OUT"; chmod 777 "$OUT"; M+=(-v "$OUT:/tmp/out"); fi
docker run --entrypoint '' --rm --cpus 8 --memory 48g --network none "${M[@]}" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e POLICY="$POLICY" -e WORKLOAD="$WORKLOAD" -e K="$K" -e SPEED="$SPEED" -e DIRECT="${DIRECT:-0}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work /tmp/out; rm -f /tmp/out/*.jsonl /tmp/out/*.out /tmp/out/*.err /tmp/out/*.log; cd /tmp/work
cp /kit/records/*.json /tmp/reg/
python - <<PY
import json, os, shutil
t = json.load(open("/kit/tasks/mls-kv-deploy.json"))
for f in t["files"]:
    p = f["path"]; os.makedirs(os.path.dirname(p), exist_ok=True)
    if f.get("from"):
        shutil.copy(os.path.join("/kit", f["from"].replace("artifacts/mls/", "")), p)
    else:
        open(p, "w").write(f.get("content") or "")
    if p.endswith(".sh"): os.chmod(p, 0o755)
print("package tree materialised:", sum(len(fs) for _, _, fs in os.walk("methods")), "files")
PY
# the script resolves the MLS-Bench task directory by its task_description.md (script dir, its parent,
# MLSBENCH_TASK_DIR / TASK_DIR, or cwd/_task); the package has to ship it. Until it
# does, a marker here keeps the known answer running; gsm8k reads its data from the HF cache, not from it.
mkdir -p /tmp/work/_task; [ -d /kit/task ] && cp -R /kit/task/. /tmp/work/_task/; [ -f /tmp/work/_task/task_description.md ] || echo "# llm-kv-selection-budgeting (marker for the known answer)" > /tmp/work/_task/task_description.md
export MLSBENCH_TASK_DIR=/tmp/work/_task
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed "$SPEED" --occupancy /tmp/out/occupancy.jsonl > /tmp/out/daemon.log 2>&1 &
D=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
export SELECTION_KV_MAX_EXAMPLES="$K" ADAPTIVE_KV_MAX_EXAMPLES="$K" OUTPUT_DIR=/tmp/work/output   # selection and quant read different env knobs; the quant wrappers pass no --max-examples
T0=$(date +%s.%N)
if [ "${DIRECT:-0}" = 1 ]; then
  # the evaluator itself, without the job script and its explicit flags, as G-mlskv-haiku-L0-r3
  # launched it: the record must still be served on the script'"'"'s own parsed defaults (exact_effective)
  (cd methods/$POLICY && python custom_selection_eval.py --workload $WORKLOAD --max-examples $K) > /tmp/out/run.out 2> /tmp/out/run.err; rc=$?
else
  bash methods/$POLICY/scripts/$WORKLOAD.sh > /tmp/out/run.out 2> /tmp/out/run.err; rc=$?
fi
T1=$(date +%s.%N)
for i in $(seq 100); do grep -q "\"kind\":\"job_bye\"" /tmp/out/occupancy.jsonl 2>/dev/null && break; sleep 0.1; done
kill $D 2>/dev/null; wait $D 2>/dev/null
echo "--- run exit $rc, wall $(python -c "print(round($T1-$T0,1))") s real at speed $SPEED"
grep -E "TEST_METRICS|Traceback|Error|error" /tmp/out/run.out /tmp/out/run.err | cut -c1-240 | head -8
grep -o "nvrt replay.*" /tmp/out/run.err | grep -E "lookup|per_example|engaged|load took" | head -6 | cut -c1-200
ls output/ 2>/dev/null; cat output/${WORKLOAD}_metrics.json 2>/dev/null
python /gm/tests/mls_known_answer_check.py /tmp/out /tmp/work "$POLICY" "$WORKLOAD" "$K" "$rc"'
