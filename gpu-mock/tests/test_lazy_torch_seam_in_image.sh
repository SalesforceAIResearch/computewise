#!/usr/bin/env bash
# next/ r8.1 in the box: the whole_job LIBRARY-seam fix for a script that imports
# torch LAZILY, inside the seamed function (the real TabArena single_cell.py shape,
# 2026-11-02). Reproduces the case the plain r8 in-image test could not: that test's
# wj-lib stand-in imports torch at TOP level before the library module, so the
# torch-import hook fires before the seamed call; here NOTHING imports torch until
# the seamed method runs, so without the pre-import arm the seam install withdraws
# (whole_job_seam_already_running) and the script runs its own code.
#
# Layout (mirrors TTC):
#   entry.py        imports NO torch at top; imports the library module (ttclib.runner)
#                   whose class method _run it then CALLS; _run is where torch is first
#                   imported and the model placed on the fake GPU (the first CUDA use).
#   ttclib/runner.py  class Runner: _run(self) -> dict  -- imports torch INSIDE _run,
#                   builds a model on cuda, computes a result, returns a dict. The
#                   record's seam is ttclib.runner:Runner._run.
#
# PASS (with the fix): entry.py prints the RECORDED dict (not what _run computes),
#   the daemon logs the seam installed before _run and 4 whole_job units paced,
#   job_bye served=result, exit 0, and NO whole_job_seam_already_running.
# Control (fix OFF via NVRT_NO_PREIMPORT=1): the OLD behaviour --
#   whole_job_seam_already_running, the script runs its own code.
#
# Usage (on a box):  sudo IMG=<box> G=<r8.1 tree> [PATCH_SRC=1] bash tests/test_lazy_torch_seam_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro")
[ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work/ttclib /tmp/out; cd /tmp/work
# entry.py: NO torch at top; import the library module, then call its _run.
cat > entry.py <<PY
import json
from ttclib.runner import Runner
res = Runner("bioresponse").run()
print("FINAL " + json.dumps(res, sort_keys=True), flush=True)
PY
cat > ttclib/__init__.py <<PY
PY
# the library module: _run imports torch INSIDE it and does the first CUDA use.
cat > ttclib/runner.py <<PY
class Runner:
    def __init__(self, task):
        self.task = task
    def run(self):
        return self._run()
    def _run(self):
        import torch                      # LAZY: the first import torch in the process
        net = torch.nn.Linear(32, 32).cuda()   # first CUDA use, INSIDE the seam
        x = torch.randn(16, 32, device="cuda")
        for _ in range(30):
            x = torch.tanh(net(x))
        return {"metric_error": float(x.mean().item()), "task": self.task}
PY
# a whole_job record for entry.py, seam = ttclib.runner:Runner._run, recorded result differs
# from what _run computes so we can tell served from own-code.
python3 - <<PY
import json, hashlib
h = hashlib.sha256(open("/tmp/work/entry.py","rb").read()).hexdigest()
rec = {"identity": {"record_id": "lazy-ttc", "entry_hash": h,
                    "argv": ["entry.py"], "match_policy": "exact_only", "replicate_group": "lazy-ttc"},
       "mode": "replay", "steps": None,
       "timing": {"pacing": "total", "wall_total_s": {"value": 20.0, "provenance": "recorded"}},
       "memory": {"peak_allocated_bytes": 1<<30, "peak": 1<<30}, "series": {},
       "result": {"final_metrics": {"value": {"metric_error": 0.4242, "task": "RECORDED"},
                                    "provenance": "recorded", "how": "test"}},
       "materialisation": {"serving": {"mode": "whole_job",
                           "seam": "ttclib.runner:Runner._run",
                           "unit": "iteration", "n_units": 4, "unit_shares": [0.25,0.25,0.25,0.25],
                           "result_key": "final_metrics", "result_kind": "metric dict", "result_shape": "value"}},
       "readiness": {"replay": True}, "failures": []}
json.dump(rec, open("/tmp/reg/lazy.json","w"))
print("REC lazy-ttc entry_hash", h[:16])
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 50 --occupancy /tmp/work/occ.jsonl > /tmp/work/daemon.log 2>&1 &
for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done

echo "=== WITH FIX (pre-import arm on) ==="
python3 entry.py > /tmp/out/fix.out 2>&1; echo "rc=$?"
echo "--- entry output:"; grep FINAL /tmp/out/fix.out
echo "--- seam / engage lines:"; grep -oE "pre-import whole_job arm[^\"]*|seam installed at [^ ;]*|whole_job early probe -> record=[^ ]*|whole_job_seam_already_running|served its recorded result" /tmp/out/fix.out /tmp/work/daemon.log 2>/dev/null | sort -u
echo "--- units + served (occupancy):"; python3 -c "
import json
u=0; served=None
for L in open(\"/tmp/work/occ.jsonl\"):
    try: e=json.loads(L)
    except: continue
    if e.get(\"kind\")==\"enqueue\" and e.get(\"unit\"): u+=1
    if e.get(\"kind\")==\"job_bye\": served=e.get(\"served\"); wj=e.get(\"whole_job_units\")
print(\"enqueue units\", u, \"job_bye served\", served, \"whole_job_units\", wj if \"wj\" in dir() else None)
"

echo "=== CONTROL (fix OFF: NVRT_NO_PREIMPORT=1) ==="
rm -f /tmp/work/occ.jsonl /tmp/work/daemon.log
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 50 --occupancy /tmp/work/occ.jsonl > /tmp/work/daemon.log 2>&1 &
for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
NVRT_NO_PREIMPORT=1 python3 entry.py > /tmp/out/ctl.out 2>&1; echo "rc=$?"
echo "--- entry output:"; grep FINAL /tmp/out/ctl.out
echo "--- seam lines:"; grep -oE "whole_job_seam_already_running|whole_job early probe -> record=[^ ]*|seam installed at [^ ;]*" /tmp/out/ctl.out /tmp/work/daemon.log 2>/dev/null | sort -u
' 2>&1 | grep -vE "^nvrt replay\[" | head -40
