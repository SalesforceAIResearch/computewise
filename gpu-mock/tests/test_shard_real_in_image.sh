#!/usr/bin/env bash
# next/ r4 on the real MLS-Bench evaluators: shard runners of the shapes agents
# wrote (astracc's run_gsm8k_shard.py: spec_from_file_location, load_workload
# sliced, score_prediction wrapped, sys.argv set, main()), against the staged
# package tree, the offline HF cache, the 40 records and gpu-tasks' six
# example-order tables. The real loader builds the keys, so a pass means the
# tables and the patch agree on the data, not only on a stand-in.
#   Q0, Q1  kivi_overlap_4bit (quant) on gsm8k, shards [100, 104) and [1315, 1319)
#           on GPUs 0 and 1 at once: exact_module (or exact_effective_module),
#           positions 100..103 and 1315..1318, each wrapper sees 4 served scores,
#           the printed final_score is their mean, no example_order_miss;
#   S0      full_attention (selection) on gsm8k, Fable's harness shape (argv[0] x,
#           a printer) with --max-examples 3: positions 0..2, a prefix;
#   NH      kivi on needlebench_niah, shard [3, 6): positions 3..5 (NeedleBench's
#           prompts repeat three times; example_id keeps them apart).
# Every served score is checked against the record's vector at its position
# (gpu_daemon._score_vector, the daemon's own function, on the record).
# Usage (on the VM):  sudo IMG=sha256:<mlskv box> HF=<cache tree ending in /hf> KIT=<mls-kit dir> TREE=<staged project/ dir> \
#        ORDER=<dir of example-order-*.json> [SPEED=20] bash tests/test_shard_real_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:?mlskv box image id}"; HF="${HF:?hf cache tree}"; KIT="${KIT:?mls kit dir}"; TREE="${TREE:?staged project dir}"; ORDER="${ORDER:?order tables dir}"
SPEED="${SPEED:-20}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
# the LongBench and NeedleBench loaders write lock files beside the dataset cache: a writable copy of datasets/,
# the model hub read-only
HFC=$(mktemp -d /srv/work/gpu-mock/hfcopy.XXXX); for x in datasets xet STAGED.json; do [ -e "$HF/$x" ] && cp -a "$HF/$x" "$HFC/"; done; mkdir -p "$HFC/hub"
M=(-v "$G/src:/gm/src:ro" -v "$G/tests:/gm/tests:ro" -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro"
   -v "$HFC:/hf" -v "$HF/hub:/hf/hub:ro" -v "$KIT:/kit:ro" -v "$TREE:/tree:ro" -v "$ORDER:/order:ro")
docker run --entrypoint '' --rm --cpus 12 --memory 64g --network none "${M[@]}" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e SPEED="$SPEED" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/out /tmp/work; cd /tmp/work
cp /kit/records/mls-llm-kv-*.json /tmp/reg/; cp /order/example-order-*.json /tmp/reg/
cp -a /tree/methods /tmp/work/methods
mkdir -p /tmp/work/_task; echo "# marker" > /tmp/work/_task/task_description.md; export MLSBENCH_TASK_DIR=/tmp/work/_task
cat > /tmp/work/shard.py <<PY
import importlib.util, json, sys
path, wl, start, end, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
spec = importlib.util.spec_from_file_location("kv_shard_eval", path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
orig_load, orig_score = module.load_workload, module.score_prediction
def load_workload(name, max_examples=0):
    w = orig_load(name, 0); w.examples = w.examples[start:end]; return w
scores = []
def score_prediction(workload_name, example, prediction):
    s = orig_score(workload_name, example, prediction); scores.append([example["example_id"], s]); return s
module.load_workload, module.score_prediction = load_workload, score_prediction
sys.argv = [path] + sys.argv[6:]
module.main()
json.dump(scores, open(out, "w"))
PY
cat > /tmp/work/fable.py <<PY
import sys, importlib.util, json
spec = importlib.util.spec_from_file_location("ev", sys.argv[1])
ev = importlib.util.module_from_spec(spec); sys.modules["ev"] = ev; spec.loader.exec_module(ev)
orig = ev.score_prediction; got = []
def patched(w, ex, pred):
    s = orig(w, ex, pred); got.append([ex["example_id"], s]); print("SCORE", s, flush=True); return s
ev.score_prediction = patched
out = sys.argv[2]
sys.argv = ["x", "--workload", "gsm8k", "--model-id", "Qwen/Qwen2.5-3B-Instruct", "--compression-ratio", "0.8", "--max-examples", "3", "--seed", "42"]
ev.main()
json.dump(got, open(out, "w"))
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed "$SPEED" --occupancy /tmp/out/occupancy.jsonl > /tmp/out/daemon.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
grep -E "example order|refused" /tmp/out/daemon.log | head
Q=/tmp/work/methods/kivi_overlap_4bit/custom_quant_eval.py; F=/tmp/work/methods/full_attention/custom_selection_eval.py
QA="--workload gsm8k --budget-bits 4 --seed 42"
(CUDA_VISIBLE_DEVICES=0 python shard.py $Q gsm8k 100 104 /tmp/out/Q0.json $QA > /tmp/out/Q0.out 2> /tmp/out/Q0.err; echo $? > /tmp/out/Q0.rc) & P0=$!
(CUDA_VISIBLE_DEVICES=1 python shard.py $Q gsm8k 1315 1319 /tmp/out/Q1.json $QA > /tmp/out/Q1.out 2> /tmp/out/Q1.err; echo $? > /tmp/out/Q1.rc) & P1=$!
wait $P0 $P1
(CUDA_VISIBLE_DEVICES=2 python fable.py $F /tmp/out/S0.json > /tmp/out/S0.out 2> /tmp/out/S0.err; echo $? > /tmp/out/S0.rc)
(CUDA_VISIBLE_DEVICES=3 python shard.py $Q needlebench_niah 3 6 /tmp/out/NH.json --workload needlebench_niah --budget-bits 4 --seed 42 > /tmp/out/NH.out 2> /tmp/out/NH.err; echo $? > /tmp/out/NH.rc)
sleep 1; kill $D 2>/dev/null; wait $D 2>/dev/null
for t in Q0 Q1 S0 NH; do echo "--- $t: rc $(cat /tmp/out/$t.rc); $(grep -h TEST_METRICS /tmp/out/$t.out | cut -c1-200)"; grep -hE "Traceback|Error" /tmp/out/$t.err | head -3; done
python /gm/tests/shard_real_check.py
'
rc=$?; rm -rf "$HFC"; exit $rc
