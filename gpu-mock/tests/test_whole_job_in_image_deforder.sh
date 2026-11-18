#!/usr/bin/env bash
# next/ r8 in the box: the whole_job serving mode, end to end, with the patch.
# A stand-in script puts its model on the GPU (the job's first CUDA use, where
# replay engages), then calls a `__main__` function (run_experiment) that does
# the GPU work and prints the dict it returns; a library-style stand-in does the
# same in an imported module's class method. One daemon (speed 10), records:
#   S   the __main__ seam: exit 0, the script prints the RECORDED dict (not what
#       its own code would compute), its wall is about the recorded 20 s / 10,
#       the daemon logs 4 whole_job units and job_bye served=result;
#   L   the library seam (wjlib.engine:Engine.evaluate, imported after torch):
#       the same, through the post-import hook;
#   K   S killed after 2 of its 4 units: no result line, job_bye served=none,
#       the draw back (the next launch's draw is a first draw, not a rerun);
#   N   a record whose seam names a function the script lacks: replay withdrawn
#       (whole_job_seam_not_found), the script runs its own code and prints what
#       that computes, exit 0;
#   R   a script whose first CUDA use is INSIDE the seamed function: the call is
#       already running at engagement, so replay is withdrawn
#       (whole_job_seam_already_running) and the script prints its own result;
#   P   a plain step record of another script on the same daemon still serves
#       as before (the mode touches nothing else).
# Usage (on the E2):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_whole_job_in_image.sh
set -uo pipefail
IMG=${IMG:-sha256:028ef5c0d049a4ef89f0a1ffe13cd929dbaf564ec667481f6b65f5b402db7e53}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work/wjlib /tmp/out; cd /tmp/work
cat > exp.py <<PY
import argparse, json, torch
p = argparse.ArgumentParser(); p.add_argument("--env", default="frostbite"); p.add_argument("--seed", type=int, default=1)
args = p.parse_args()
torch.manual_seed(args.seed)
def run_experiment(env, seed):
    net = torch.nn.Linear(64, 64).cuda()
    x = torch.randn(32, 64, device="cuda")
    for _ in range(50):
        x = torch.tanh(net(x))
    return {"mean_return": float(x.mean().item()), "env": env}
res = run_experiment(args.env, args.seed)
print("FINAL " + json.dumps(res, sort_keys=True), flush=True)
PY
cat > wjlib/__init__.py <<PY
PY
cat > wjlib/engine.py <<PY
import torch
class Engine:
    def __init__(self, task):
        self.task = task
    def evaluate(self, n):
        x = torch.ones(n, 8, device="cuda")
        return {"score": float((x @ x.T).sum().item()), "task": self.task}
PY
cat > lib.py <<PY
import json, sys, torch
w = torch.ones(8, device="cuda")
from wjlib.engine import Engine
print("FINAL " + json.dumps(Engine(sys.argv[1]).evaluate(16), sort_keys=True), flush=True)
PY
cat > inside.py <<PY
import json, torch
def run_inside():
    x = torch.ones(4, device="cuda")
    return {"own": float(x.sum().item())}
print("FINAL " + json.dumps(run_inside(), sort_keys=True), flush=True)
PY
cat > nos.py <<PY
import json, torch
w = torch.ones(2, device="cuda")
def compute():
    return {"own": float(torch.ones(4, device="cuda").sum().item())}
print("FINAL " + json.dumps(compute(), sort_keys=True), flush=True)
PY
python - <<PY
import hashlib, json
def h(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def rec(rid, script, argv, seam, result, n=4, wall=20.0):
    return {"identity": {"record_id": rid, "entry_hash": h(script), "argv": argv, "match_policy": "exact_only", "replicate_group": rid},
            "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": wall, "provenance": "recorded"}},
            "memory": {"peak_allocated_bytes": 2 << 30, "peak": 2 << 30}, "series": {},
            "result": {"final": {"value": result, "provenance": "recorded", "how": "test"}},
            "materialisation": {"serving": {"mode": "whole_job", "seam": seam, "n_units": n, "unit": "iteration",
                                            "result_key": "final", "result_kind": "metric dict", "result_shape": "dict"}},
            "readiness": {"replay": True}, "failures": []}
json.dump(rec("wj-main", "/tmp/work/exp.py", ["exp.py", "--env", "frostbite"], "__main__:run_experiment",
              {"mean_return": 1234.5, "env": "frostbite"}), open("/tmp/reg/main.json", "w"))
json.dump(rec("wj-lib", "/tmp/work/lib.py", ["lib.py", "mteb-task"], "wjlib.engine:Engine.evaluate",
              {"score": 0.6123, "task": "mteb-task"}), open("/tmp/reg/lib.json", "w"))
json.dump(rec("wj-noseam", "/tmp/work/nos.py", ["nos.py"], "__main__:not_there", {"own": -1.0}), open("/tmp/reg/nos.json", "w"))
json.dump(rec("wj-inside", "/tmp/work/inside.py", ["inside.py"], "__main__:run_inside", {"own": -2.0}), open("/tmp/reg/inside.json", "w"))
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
run() { local t=$1; shift; local t0=$(date +%s.%N); (CUDA_VISIBLE_DEVICES=0 timeout 120 "$@" > /tmp/out/$t.out 2>&1); local rc=$?; echo "$t rc=$rc wall=$(python3 -c "print(round($(date +%s.%N)-$t0,2))")" >> /tmp/out/rc.txt; }
run S python exp.py --env frostbite
run L python lib.py mteb-task
run N python nos.py
run R python inside.py
# K: the same record, killed after its second unit (2 of 4 units at 0.5 s each)
(CUDA_VISIBLE_DEVICES=0 python exp.py --env frostbite > /tmp/out/K.out 2>&1 & echo $! > /tmp/out/K.pid; wait) & sleep 0.1
for i in $(seq 200); do n=$(grep -c "\"unit\": \"whole_job\"" /tmp/out/occ.jsonl 2>/dev/null); [ "${n:-0}" -ge 6 ] && break; sleep 0.05; done
kill -9 $(cat /tmp/out/K.pid) 2>/dev/null; sleep 1
run S2 python exp.py --env frostbite
kill $D 2>/dev/null; wait $D 2>/dev/null
cat /tmp/out/rc.txt
for t in S L N R K S2; do echo "$t: $(grep -h "^FINAL" /tmp/out/$t.out | head -1)"; done
python3 - <<PY
import json
ev = [json.loads(l) for l in open("/tmp/out/occ.jsonl")]
jobs = {}
for e in ev:
    if e.get("kind") == "lookup": jobs[e["job"]] = e.get("record_id"), e.get("draw")
    if e.get("kind") == "job_bye":
        rid, draw = jobs.get(e["job"], (None, None))
        print("BYE", rid, draw, e.get("served"), e.get("whole_job_units"), e.get("draw_returned_incomplete"))
    if e.get("kind") == "replay_disengaged": print("DISENGAGED", e.get("record_id"), e.get("reason"))
print("UNITS", sum(1 for e in ev if e.get("kind") == "enqueue" and e.get("unit") == "whole_job"))
PY
'
