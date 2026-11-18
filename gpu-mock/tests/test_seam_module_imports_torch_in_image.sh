#!/usr/bin/env bash
# next/ r8.9.2 known answer: a whole_job seam module that imports torch AT ITS TOP.
# The late-pole entries are one line, `from dlm_vanilla_uncached import *` /
# `from sparse_streaming_llm import main`, and their seam modules start with
# `import torch`. That import runs the torch-time early probe while the seam module is
# still executing its body, before its seamed function is defined, so r8.1-r8.9.1
# withdrew with whole_job_seam_not_found (gpu-mock's late-pole proof on the installed
# r8.9.1 box, 2026-11-03 08:09Z). Every earlier stand-in imported torch lazily inside
# the seam module or defined the seam before importing torch.
#   T    entry `from wjtop import *`, wjtop imports torch first, defines evaluate later
#                                                         -> RECORDED 0.77, units paced
#   T2   the same with `from wjtop import main` and main() calling evaluate
#                                                         -> RECORDED 0.77
#   L    the earlier lazy shape (torch imported inside the seam module's function)
#                                                         -> RECORDED 0.77 (unchanged)
#   N    NVRT_NO_REPLAY=1                                 -> own code
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_seam_module_imports_torch_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/out /tmp/t; cd /tmp/t
cat > wjtop.py <<PY
import json
import torch                          # FIRST: the early probe runs here, before evaluate exists
import torch.nn as nn
def evaluate(model):
    x = torch.zeros(8, 32, device="cuda")
    for _ in range(4):
        x = torch.tanh(model(x))
    return {"score": round(float(x.mean().item()), 6)}
def main():
    model = nn.Linear(32, 32).cuda()
    print("FINAL " + json.dumps(evaluate(model), sort_keys=True), flush=True)
PY
cat > wjlazy2.py <<PY
import json
def evaluate(model):
    import torch
    x = torch.zeros(8, 32, device="cuda")
    for _ in range(4):
        x = torch.tanh(model(x))
    return {"score": round(float(x.mean().item()), 6)}
PY
printf "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nfrom wjtop import *\nif __name__ == \"__main__\":\n    main()\n" > entry_star.py
printf "import sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nfrom wjtop import main\nif __name__ == \"__main__\":\n    main()\n" > entry_main.py
printf "import json, sys, os\nsys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\nimport wjlazy2\nimport torch\nm = torch.nn.Linear(32, 32).cuda()\nprint(\"FINAL \" + json.dumps(wjlazy2.evaluate(m), sort_keys=True), flush=True)\n" > entry_lazy.py
python3 - <<PY
import json, hashlib
for name, entry, seam in (("t", "entry_star.py", "wjtop:evaluate"), ("t2", "entry_main.py", "wjtop:evaluate"), ("l", "entry_lazy.py", "wjlazy2:evaluate")):
    h = hashlib.sha256(open(f"/tmp/t/{entry}", "rb").read()).hexdigest()
    json.dump({"identity": {"record_id": "wj-" + name, "entry_hash": h, "argv": [entry], "match_policy": "exact_only",
                            "replicate_group": "wj-" + name, "env": {}, "env_exempt": [], "trajectory": {"environment": {}}},
               "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": 2.0, "provenance": "recorded"}},
               "memory": {"peak_bytes": 1 << 28}, "series": {},
               "result": {"final_metrics": {"value": {"score": 0.77}, "provenance": "recorded", "how": "test"}},
               "materialisation": {"serving": {"mode": "whole_job", "seam": seam, "unit": "phase", "n_units": 4,
                                               "unit_shares": [0.25] * 4, "result_key": "final_metrics", "result_kind": "metric dict",
                                               "result_shape": "dict", "module_shas": {}}},
               "readiness": {"replay": True}, "failures": []}, open(f"/tmp/reg/{name}.json", "w"))
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 20 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
CUDA_VISIBLE_DEVICES=0 timeout 120 python3 entry_star.py > /tmp/out/T.out 2>&1; echo "T rc=$?"
CUDA_VISIBLE_DEVICES=0 timeout 120 python3 entry_main.py > /tmp/out/T2.out 2>&1; echo "T2 rc=$?"
CUDA_VISIBLE_DEVICES=0 timeout 120 python3 entry_lazy.py > /tmp/out/L.out 2>&1; echo "L rc=$?"
CUDA_VISIBLE_DEVICES=0 NVRT_NO_REPLAY=1 timeout 120 python3 entry_star.py > /tmp/out/N.out 2>&1; echo "N rc=$?"
kill $D 2>/dev/null; wait $D 2>/dev/null
for t in T T2 L N; do echo "--- $t: $(grep -h "^FINAL" /tmp/out/$t.out | head -1)"; grep -hE "nvrt replay.*(whole_job|withdrawn|deferred)" /tmp/out/$t.out | head -3; done
python3 - <<PY
import json, collections
c = collections.Counter()
for l in open("/tmp/out/occ.jsonl"):
    e = json.loads(l)
    if e.get("kind") == "replay_disengaged": print("DISENGAGED job", e.get("job"), e.get("reason"))
    if e.get("kind") == "job_bye": print("BYE job", e.get("job"), "served", e.get("served"), "units", e.get("whole_job_units"))
PY
'
