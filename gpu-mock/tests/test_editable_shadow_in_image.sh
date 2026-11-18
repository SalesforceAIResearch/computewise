#!/usr/bin/env bash
# next/ r8.9.1 known answer: an image-installed EDITABLE distribution is the installed
# package, not a project file shadowing it. r8.9's shadow check counted as library only
# files under sysconfig's stdlib/site-packages; a `pip install -e` package's files live
# in its source checkout, so the TTC image's editable TabArena
# (/opt/sol-ttc/tabarena/packages/tabarena/src/tabarena) was read as a project file
# shadowing the `tabarena` distribution and the real 805f cell was refused at step 0
# (Sol's installed-box probe on b51b86fb, 2026-11-03 05:37Z; a false refusal).
#
# The stand-in reproduces the image's shape: as root, a package `edpkg` at
# /opt/edsrc/edpkg (root-owned, read-only to uid 1000), installed `pip install -e
# --no-deps --no-build-isolation` (direct_url.json dir_info.editable); its module holds
# the whole_job seam. The job runs as uid 1000 with a record whose module_shas is {}.
#   ED   the entry imports edpkg (the editable install)          -> served RECORDED, no shadow
#   SH   a file the agent wrote beside the entry, named like the installed package the
#        entry imports (edpkg.py, found first on sys.path[0])  -> refused, identity_module_shadow
#        (a numpy.py there breaks torch's own import before any check, as on hardware)
#   SD   the agent's own stdlib-named sched.py in the task root -> refused, identity_module_shadow
#   UE   the agent's own `pip install --user -e` of a workspace package named like the
#        installed one (edpkg), so its user-site finder wins  -> user-site metadata, files owned
#                                                                by the agent: not trusted, refused,
#                                                                identity_module_shadow
# The trust rule: the module's file lies under the source root (direct_url.json) of an
# editable distribution whose metadata is in SYSTEM site-packages, it is not under the
# entry's directory, and neither it nor its directory is owned by the job's user.
# Usage (on a box):  sudo IMG=<box> G=<tree> [PATCH_SRC=1] bash tests/test_editable_shadow_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
set -e
# --- as root: the image-installed editable package
mkdir -p /opt/edsrc/edpkg
cat > /opt/edsrc/edpkg/__init__.py <<PY
def evaluate(model):
    import torch
    x = torch.zeros(8, 32, device="cuda")
    for _ in range(4):
        x = torch.tanh(model(x))
    return {"score": round(float(x.mean().item()), 6)}
PY
cat > /opt/edsrc/pyproject.toml <<PY
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "edpkg"
version = "0.1.0"
[tool.setuptools]
packages = ["edpkg"]
PY
pip install --no-index --no-deps --no-build-isolation -q -e /opt/edsrc > /tmp/pipe.log 2>&1 || { echo "EDITABLE_INSTALL_FAILED"; tail -5 /tmp/pipe.log; }
chmod -R a-w /opt/edsrc
python3 -c "import edpkg, importlib.metadata as md, json; d=md.distribution(\"edpkg\"); print(\"EDITABLE\", edpkg.__file__, json.loads(d.read_text(\"direct_url.json\"))[\"dir_info\"])"
set +e
mkdir -p /tmp/reg /tmp/out /tmp/task; chmod 777 /tmp/out /tmp/task /tmp/reg
cat > /tmp/task/entry.py <<PY
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
mode = os.environ.get("MODE", "ED")
if mode == "SD":
    import sched                      # the agent'"'"'s sched.py beside the entry
import torch, edpkg
model = torch.nn.Linear(32, 32).cuda()
print("FINAL " + json.dumps(edpkg.evaluate(model), sort_keys=True), flush=True)
PY
python3 - <<PY
import json, hashlib
h = hashlib.sha256(open("/tmp/task/entry.py", "rb").read()).hexdigest()
json.dump({"identity": {"record_id": "edrec", "entry_hash": h, "argv": ["entry.py"], "match_policy": "exact_only",
                        "replicate_group": "edrec", "env": {}, "env_exempt": ["MODE"], "trajectory": {"environment": {}}},
           "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": 2.0, "provenance": "recorded"}},
           "memory": {"peak_bytes": 1 << 28}, "series": {},
           "result": {"final_metrics": {"value": {"score": 0.77}, "provenance": "recorded", "how": "test"}},
           "materialisation": {"serving": {"mode": "whole_job", "seam": "edpkg:evaluate", "unit": "phase", "n_units": 2,
                                           "unit_shares": [0.5, 0.5], "result_key": "final_metrics", "result_kind": "metric dict",
                                           "result_shape": "dict", "module_shas": {}}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/ed.json", "w"))
PY
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 20 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done; chmod 777 /tmp/gm.sock
U="setpriv --reuid=1000 --regid=1000 --clear-groups"; command -v setpriv >/dev/null || U="su -s /bin/sh nobody -c"
run() { ( cd /tmp/task && $U env HOME=/tmp/task NVRT_SOCKET=/tmp/gm.sock CUDA_VISIBLE_DEVICES=0 MODE=$1 timeout 120 python3 entry.py > /tmp/out/$1.out 2>&1 ); echo "$1 rc=$?"; }
run ED
$U sh -c "printf \"def evaluate(model):\\n    return {\\\"own\\\": 1}\\n\" > /tmp/task/edpkg.py"; run SH; rm -f /tmp/task/edpkg.py
$U sh -c "printf \"QUEUE = []\\n\" > /tmp/task/sched.py"; run SD; rm -f /tmp/task/sched.py
# UE: the agent installs its own editable package (user site, writable dist-info) named like edpkg
mkdir -p /tmp/agentws && chown 1000:1000 /tmp/agentws
$U sh -c "mkdir -p /tmp/agentws/own/edpkg && cp /opt/edsrc/edpkg/__init__.py /tmp/agentws/own/edpkg/ && cp /opt/edsrc/pyproject.toml /tmp/agentws/own/"
( cd /tmp/task && $U env HOME=/tmp/task pip install --user --no-index --no-deps --no-build-isolation -q -e /tmp/agentws/own > /tmp/out/ue-pip.log 2>&1 ); echo "UE pip rc=$? $(tail -1 /tmp/out/ue-pip.log)"
( cd /tmp/task && $U env HOME=/tmp/task NVRT_SOCKET=/tmp/gm.sock CUDA_VISIBLE_DEVICES=0 MODE=ED timeout 120 python3 -c "import edpkg; print(\"UE_EDPKG\", edpkg.__file__)" 2>&1 | grep UE_EDPKG )
run UE_ED
kill $D 2>/dev/null; wait $D 2>/dev/null
for t in ED SH SD UE_ED; do echo "--- $t: $(grep -h "^FINAL" /tmp/out/$t.out 2>/dev/null | head -1)"; done
python3 - <<PY
import json
for l in open("/tmp/out/occ.jsonl"):
    e = json.loads(l)
    if e.get("kind") == "replay_disengaged": print("DISENGAGED job", e.get("job"), e.get("reason"), "|", (e.get("detail") or "")[:170])
    if e.get("kind") == "whole_job_probe": print("PROBE job", e.get("job"), e.get("match"))
PY
'
