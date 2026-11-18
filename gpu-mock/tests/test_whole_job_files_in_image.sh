#!/usr/bin/env bash
# next/ r8.5.1 known answer: whole_job FILE binds.
# serving.module_shas maps a path RELATIVE TO THE TASK ROOT to its sha256; each file
# is hashed on disk at seam install, imported yet or not, so a file the job imports
# lazily (DLM: dllm_cache hooks and the LLaDA modeling files, inside the generation
# path) is bound too. Mismatch or missing -> the job runs its own code, no recorded
# aggregate, draw returned.
#
# The k2-latepole layout, in miniature. Task root /tmp/task, launcher cd's there:
#   methods/m1/entry.py        the entry (CLI wrapper)
#   methods/m1/wjseam.py       the seam module: evaluate_batch
#   methods/m1/wjmeth.py       the method file, imported by the seam module at load
#   shared/lazy/wjlazy.py      a project file imported LAZILY inside evaluate_batch
#                              (never loaded under replay: the seam replaces it)
#   B    bare launch, all files as recorded            -> RECORDED 0.77, served=result
#   P    method file edited after the record           -> own code, whole_job_identity_module_dep
#   L    LAZY file edited after the record (not loaded) -> own code, whole_job_identity_module_dep
#   G    a declared file missing                        -> own code, whole_job_identity_module_dep
#   O    launched from another cwd (root found from the entry's ancestors) -> RECORDED
#   B2   bare relaunch after the refusals               -> RECORDED (draw returned, not a rerun)
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_whole_job_files_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 12g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
T=/tmp/task; mkdir -p /tmp/reg /tmp/out $T/methods/m1 $T/shared/lazy
cat > $T/methods/m1/wjmeth.py <<PY
SCALE = 1.0
PY
cat > $T/shared/lazy/wjlazy.py <<PY
def hook(x):
    return x
PY
cat > $T/methods/m1/wjseam.py <<PY
import wjmeth
def evaluate_batch(model):
    import sys, torch
    sys.path.insert(0, "/tmp/task/shared/lazy")
    import wjlazy                     # LAZY: only when the real evaluate_batch runs
    x = torch.zeros(8, 128, device="cuda")
    for _ in range(10):
        x = wjlazy.hook(torch.tanh(model(x)))
    return {"score": round(float(x.mean().item()) * wjmeth.SCALE, 6)}
PY
cat > $T/methods/m1/entry.py <<PY
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wjseam import evaluate_batch
model = torch.nn.Linear(128, 128).cuda()
print("FINAL " + json.dumps(evaluate_batch(model), sort_keys=True), flush=True)
PY
python3 - <<PY
import json, hashlib
T = "/tmp/task"
h = lambda p: hashlib.sha256(open(f"{T}/{p}", "rb").read()).hexdigest()
deps = {p: h(p) for p in ("methods/m1/wjmeth.py", "shared/lazy/wjlazy.py")}
r = {"identity": {"record_id": "wjf", "entry_hash": h("methods/m1/entry.py"),
                  "argv": ["entry.py"], "match_policy": "exact_only", "replicate_group": "wjf",
                  "env": {}, "env_exempt": [], "trajectory": {"environment": {}}},
     "mode": "replay", "steps": None,
     "timing": {"pacing": "total", "wall_total_s": {"value": 4.0, "provenance": "recorded"}},
     "memory": {"peak_bytes": 1 << 28}, "series": {},
     "result": {"final_metrics": {"value": {"score": 0.77}, "provenance": "recorded", "how": "test"}},
     "materialisation": {"serving": {"mode": "whole_job", "seam": "wjseam:evaluate_batch", "unit": "phase",
                                     "n_units": 4, "unit_shares": [0.25] * 4, "result_key": "final_metrics",
                                     "result_kind": "metric dict", "result_shape": "dict",
                                     "seam_sha256": h("methods/m1/wjseam.py"), "module_shas": deps}},
     "readiness": {"replay": True}, "failures": []}
json.dump(r, open("/tmp/reg/wjf.json", "w"))
print("module_shas", {k: v[:12] for k, v in deps.items()})
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
launch() {  # launch <tag>: from the task root, as the launcher does (cd "$ROOT")
  ( cd /tmp/task && CUDA_VISIBLE_DEVICES=0 timeout 120 python3 methods/m1/entry.py > /tmp/out/$1.out 2>&1 ); echo "$1 rc=$?" >> /tmp/out/rc.txt
}
launch B
cp $T/methods/m1/wjmeth.py /tmp/wjmeth.bak; echo "SCALE = 2.0" > $T/methods/m1/wjmeth.py
launch P; cp /tmp/wjmeth.bak $T/methods/m1/wjmeth.py
cp $T/shared/lazy/wjlazy.py /tmp/wjlazy.bak; printf "def hook(x):\n    return x * 1.0\n" > $T/shared/lazy/wjlazy.py
launch L; cp /tmp/wjlazy.bak $T/shared/lazy/wjlazy.py
mv $T/shared/lazy/wjlazy.py /tmp/wjlazy.moved
launch G; mv /tmp/wjlazy.moved $T/shared/lazy/wjlazy.py
mkdir -p /tmp/elsewhere; ( cd /tmp/elsewhere && CUDA_VISIBLE_DEVICES=0 timeout 120 python3 /tmp/task/methods/m1/entry.py > /tmp/out/O.out 2>&1 ); echo "O rc=$?" >> /tmp/out/rc.txt
launch B2
kill $D 2>/dev/null; wait $D 2>/dev/null
cat /tmp/out/rc.txt
for t in B P L G O B2; do echo "$t: $(grep -h "^FINAL" /tmp/out/$t.out | head -1)"; done
python3 - <<PY
import json
for l in open("/tmp/out/occ.jsonl"):
    e = json.loads(l)
    if e.get("kind") == "replay_disengaged": print("DISENGAGED job", e.get("job"), e.get("reason"), "|", e.get("detail"))
    if e.get("kind") == "job_bye": print("BYE job", e.get("job"), "served", e.get("served"), "units", e.get("whole_job_units"))
    if e.get("kind") == "whole_job_probe": print("PROBE job", e.get("job"), e.get("match"))
PY
'
