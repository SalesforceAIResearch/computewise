#!/usr/bin/env bash
# next/ r8.3 known answer: whole_job live_fields. A whole_job record may name LIVE
# instrument fields the CSV never recorded (DLM peak_memory_mb, a paced-seam
# elapsed); the patch reads the instrument IN the seamed context after pacing and
# merges into the served dict, but NEVER overrides a recorded key. Analysis done-when:
# (1) a live field inside the seam == what the patched torch call returns outside it;
# (2) no recorded field can be overridden by a live one.
#
# A library-seam stand-in (the k2-latepole shape): an imported module's batch helper
# allocates a model on cuda (real fake-GPU memory), then returns {}; the record's
# result has recorded quality {score: 0.77} and live_fields
# {peak_memory_mb: max_memory_allocated, elapsed_s: seam_elapsed_s, score: max_memory_allocated}.
# PASS: served dict has score=0.77 (RECORDED, NOT overridden by the live spec on 'score'),
# peak_memory_mb ~= the real allocated bytes (live), elapsed_s > 0 (live).
#
# Usage:  sudo IMG=<box> G=<r8.3 tree> [PATCH_SRC=1] bash tests/test_live_fields_in_image.sh
set -uo pipefail
IMG=${IMG:?set IMG to a box image}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
FAKEGPU=${FAKEGPU:-/srv/work/gpu-mock/FakeGPU}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
W=$(mktemp -d /srv/work/gpu-mock/livef.XXXX); mkdir -p "$W/reg" "$W/lib"
cat > "$W/prep.sh" <<'PREP'
set -e
SRC=/gm/FakeGPU/fakegpu
if [ -d "$SRC" ]; then
  mkdir -p /gm/pylib; rm -rf /gm/pylib/nvcompat; cp -r "$SRC" /gm/pylib/nvcompat
  find /gm/pylib/nvcompat -name __pycache__ -type d -prune -exec rm -rf {} +
  sed -i -E 's/^(\s*)import fakegpu$/\1import nvcompat as fakegpu/; s/^(\s*)from fakegpu import/\1from nvcompat import/' /gm/pylib/nvcompat/*.py
fi
PREP
# library module with the seamed batch helper (torch imported INSIDE it -> r8.1 lazy path)
cat > "$W/lib/k2helper.py" <<'PY'
def run_batch(model):
    # the per-case loop would run here; replaced ONCE by the seam. The MODEL was
    # allocated OUTSIDE (entry.py) -- the real k2-latepole shape -- so the live
    # max_memory_allocated reflects the outer model's real fake-GPU allocation.
    import torch
    x = torch.zeros(16, 2048, device="cuda"); _ = model(x)
    return {}
PY
cat > "$W/entry.py" <<'PY'
import json, torch
from k2helper import run_batch
model = torch.nn.Sequential(*[torch.nn.Linear(2048, 2048) for _ in range(8)]).cuda()  # ~128 MB, OUTSIDE the seam
res = run_batch(model)
print("FINAL " + json.dumps(res, sort_keys=True), flush=True)
PY
python3 - "$W" <<'PY'
import json, hashlib, sys
W=sys.argv[1]
h=hashlib.sha256(open(f"{W}/entry.py","rb").read()).hexdigest()
rec={"identity":{"record_id":"livef","entry_hash":h,"argv":["entry.py"],"match_policy":"exact_only","replicate_group":"livef"},
     "mode":"replay","timing":{"pacing":"total","wall_total_s":{"value":8.0,"provenance":"recorded"}},
     "memory":{"peak":134283264,"peak_bytes":134283264},
     "result":{"agg":{"value":{"score":0.77},"provenance":"recorded"}},
     "materialisation":{"serving":{"mode":"whole_job","seam":"k2helper:run_batch","unit":"phase","n_units":4,
        "unit_shares":[0.25,0.25,0.25,0.25],"result_key":"agg","result_kind":"metric dict","result_shape":"dict",
        "live_fields":{"peak_memory_mb":"recorded_peak_mb","elapsed_s":"seam_elapsed_s","score":"max_memory_allocated"}}},
     "readiness":{"replay":True}}
json.dump(rec,open(f"{W}/reg/livef.json","w"))
print("REC livef entry_hash", h[:16])
PY
echo "=== r8.3 live_fields known answer $(date -u +%FT%TZ)  image $IMG"
cat > "$W/run.sh" <<'RUN'
set -uo pipefail
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/w/gm.sock PYTHONPATH=/gm/pylib:/w/lib
[ -d /gm/pylib/nvcompat ] || export PYTHONPATH=/usr/local/nvidia/pylib:/w/lib
python3 -I /gm/src/daemon/gpu_daemon.py --socket /w/gm.sock --registry /w/reg --gpus 4 --speed 50 --occupancy /w/occ.jsonl > /w/daemon.log 2>&1 &
for i in $(seq 200); do [ -S /w/gm.sock ] && break; sleep 0.1; done
cd /w && python3 entry.py
RUN
sudo docker run --entrypoint '' --rm --cpus 8 --memory 24g --network none \
  -v "$G/src:/gm/src:ro" -v "$FAKEGPU:/gm/FakeGPU:ro" \
  -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
  -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
  -v "$W:/w" -e NVRT_SOCKET=/w/gm.sock -e FAKEGPU_PROFILE_DIR=/gm/FakeGPU/profiles \
  -e NVRT_PROFILE=h100 "$IMG" bash -c 'bash /w/prep.sh; bash /w/run.sh' 2>&1 | grep -E "FINAL|REC|Error|Traceback|seam installed|served its recorded" | head
echo "--- expect FINAL: score=0.77 (recorded, NOT overridden), peak_memory_mb a live byte count (~1.3e8), elapsed_s>0 ---"
rm -rf "$W"
echo "=== done $(date -u +%FT%TZ) ==="
