#!/usr/bin/env bash
# next/ r8.5 known answer: a whole_job record is served ONLY to the configuration
# it was recorded under (the study / analysis 2026-11-02). Three binds, each
# checked in the job's own interpreter at seam install:
#   env      identity.env (minus env_exempt): unset == the declared value; a set
#            value must equal it exactly
#   runtime  identity.trajectory.environment package keys == importlib.metadata
#            .version() of the installed distribution (commit keys, package, os,
#            python are provenance and not checked)
#   module   serving.seam_sha256 == sha256 of the loaded seam module's file
# A mismatch withdraws replay: the job runs its own code (its own FINAL line, its
# own wall), no recorded aggregate, and the daemon gives the draw back.
#
# Library-seam stand-in (the k2-latepole shape): entry.py builds a model on cuda,
# reads an env knob the way the source does (unset -> '0'), and calls
# wjid.evaluate_batch(model, limit); the record's result is {score: 0.77}.
#   B   bare launch, everything matches             -> RECORDED score 0.77, served=result
#   E   KNOB=2 (a cut run)                          -> own code, reason whole_job_identity_env
#   D   KNOB=0 (set to the declared value)          -> RECORDED (set == declared is a match)
#   R   record pins a package version not installed -> own code, whole_job_identity_runtime
#   M   seam module edited after the record          -> own code, whole_job_identity_module
#   X   KNOB is in env_exempt, KNOB=5               -> RECORDED (exempt keys are not checked)
#   B2  bare launch again after E/R/M               -> RECORDED, draw is a first draw, not a rerun
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_whole_job_identity_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 12g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work/lib /tmp/out; cd /tmp/work
export PYTHONPATH=/tmp/work/lib:${PYTHONPATH:-}
cat > lib/wjid.py <<PY
def evaluate_batch(model, limit):
    import torch
    x = torch.zeros(8, 256, device="cuda")
    for _ in range(20):
        x = torch.tanh(model(x))
    return {"score": round(float(x.mean().item()), 6), "limit": limit}
PY
cat > entry.py <<PY
import json, os, torch
from wjid import evaluate_batch
limit = int(os.environ.get("KNOB") or "0")
model = torch.nn.Linear(256, 256).cuda()
res = evaluate_batch(model, limit)
print("FINAL " + json.dumps(res, sort_keys=True), flush=True)
PY
mk() {  # mk <variant>: write the ONE record the daemon loads for this launch
python3 - "$1" <<PY
import json, hashlib, importlib.metadata as md, sys
v = sys.argv[1]
eh = hashlib.sha256(open("/tmp/work/entry.py","rb").read()).hexdigest()
ms = hashlib.sha256(open("/tmp/work/lib/wjid.py","rb").read()).hexdigest()
tv = md.version("torch")
env, exempt, pins, seam = {"KNOB": "0"}, [], {"torch": tv, "source_commit": "deadbeef", "python": "3.x"}, ms
if v == "R": pins = dict(pins, torch=tv + ".not-this-one")
if v == "M": seam = "0" * 64
if v == "X": exempt = ["KNOB"]
r = {"identity": {"record_id": "wjid", "entry_hash": eh, "argv": ["entry.py"], "match_policy": "exact_only",
                  "replicate_group": "wjid", "env": env, "env_exempt": exempt, "trajectory": {"environment": pins}},
     "mode": "replay", "steps": None,
     "timing": {"pacing": "total", "wall_total_s": {"value": 4.0, "provenance": "recorded"}},
     "memory": {"peak_bytes": 1 << 28}, "series": {},
     "result": {"final_metrics": {"value": {"score": 0.77, "limit": 0}, "provenance": "recorded", "how": "test"}},
     "materialisation": {"serving": {"mode": "whole_job", "seam": "wjid:evaluate_batch", "unit": "phase",
                                     "n_units": 4, "unit_shares": [0.25]*4, "result_key": "final_metrics",
                                     "result_kind": "metric dict", "result_shape": "dict", "seam_sha256": seam}},
     "readiness": {"replay": True}, "failures": []}
json.dump(r, open("/tmp/reg/wjid.json", "w"))
PY
}
startd() { rm -f /tmp/gm.sock; python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ-$1.jsonl > /tmp/out/d-$1.log 2>&1 & echo $! > /tmp/out/d.pid; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done; }
stopd() { kill $(cat /tmp/out/d.pid) 2>/dev/null; sleep 0.3; }
launch() {  # launch <tag> [ENV=VAL]
  local t=$1; shift
  ( env "$@" CUDA_VISIBLE_DEVICES=0 timeout 120 python3 entry.py > /tmp/out/$t.out 2>&1 ); echo "$t rc=$?" >> /tmp/out/rc.txt
}
# each variant runs on its own daemon so its occupancy is separate; B/E/D/B2 share one
mk ok; startd main
launch B
launch E KNOB=2
launch D KNOB=0
launch B2
stopd
for v in R M X; do
  mk $v; startd $v
  if [ $v = X ]; then launch $v KNOB=5; else launch $v; fi
  stopd
done
cat /tmp/out/rc.txt
for t in B E D B2 R M X; do echo "$t: $(grep -h "^FINAL" /tmp/out/$t.out | head -1)"; done
python3 - <<PY
import json, glob
for f in sorted(glob.glob("/tmp/out/occ-*.jsonl")):
    ev = [json.loads(l) for l in open(f)]
    tag = f.split("occ-")[1].split(".")[0]
    for e in ev:
        if e.get("kind") == "replay_disengaged":
            print(tag, "DISENGAGED", e.get("reason"), "|", e.get("detail"))
        if e.get("kind") == "job_bye":
            print(tag, "BYE job", e.get("job"), "served", e.get("served"), "units", e.get("whole_job_units"))
        if e.get("kind") == "lookup":
            print(tag, "LOOKUP job", e.get("job"), "draw", e.get("draw"), "match", e.get("match"))
PY
'
