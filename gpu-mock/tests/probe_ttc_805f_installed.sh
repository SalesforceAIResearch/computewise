#!/usr/bin/env bash
# PROBE (not a gate): Sol's canonical 805f cell on the INSTALLED tabular box, with a
# gpu-mock tree's patch mounted over the installed one. It reproduces Sol's
# installed-box probe of 2026-11-03 05:37Z (r8.9: identity_module_shadow on the image's
# editable TabArena, a false refusal) without run_one: the same image, the staged
# pkg-sol-tabular-science project tree that run produced (task c533c980), the canonical
# record sol-ttc-tabpfn3-aggregation-96-bioresponse-split0 from the deployed registry,
# uid 1000, --network none, the baked read-only /opt/cache/openml with a private
# writable copy as the adapter expects. No weights: a served cell never loads them.
# Expect on the fixed tree: the seam engages, no replay_disengaged, the cell's result
# file holds the record's metric, units paced.
# Usage (on the E2):  sudo G=<gpu-mock tree> [IMG=<box>] [SIDE_SRC=<daemon src dir>] bash tests/probe_ttc_805f_installed.sh
set -uo pipefail
G=${G:?gpu-mock tree whose src/ to mount}
IMG=${IMG:-sha256:b51b86fbc53b61dbb3fec3f51313ec50b05104763697a1e4ff33db64290dd1e6}
STAGED=${STAGED:-/srv/work/sol-independent/ttc-r891-entry-20261103T0626Z/cache/tasks/pkg-sol-tabular-science/c533c980d999-5ec759ba/project}
[ -d "$STAGED" ] || { echo "STAGED tree missing: $STAGED"; exit 2; }
REC=${REC:-/opt/computewise/gpu-tasks/registry/records/sol-ttc-tabpfn3-aggregation-96-bioresponse-split0.json}
PYSITE=$(docker run --rm --entrypoint python3 --network none "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
W=$(mktemp -d /srv/work/gpu-mock/ttc805f.XXXX); mkdir -p "$W/reg" "$W/out"; cp -a "$STAGED" "$W/project"; cp "$REC" "$W/reg/"
chown -R 1000:1000 "$W/project" "$W/out"; chmod 777 "$W/reg" "$W"
echo "=== 805f installed-box probe $(date -u +%FT%TZ) image ${IMG:7:12} tree patch $(cd $G/src/torch_patch && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
echo "adapter $(sha256sum $W/project/tools/single_cell.py | cut -c1-16)  record $(python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print(r['identity']['record_id'], r['identity']['entry_hash'][:12], r['materialisation']['serving'].get('module_shas'))" "$REC")"
docker run --entrypoint '' --rm --cpus 8 --memory 24g --network none \
  -v "$G/src:/gm/src:ro" -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
  -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
  -v "$W:/w" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
python3 -I /gm/src/daemon/gpu_daemon.py --socket /w/gm.sock --registry /w/reg --gpus 4 --speed 50 --occupancy /w/out/occ.jsonl > /w/out/daemon.log 2>&1 &
for i in $(seq 100); do [ -S /w/gm.sock ] && break; sleep 0.1; done; chmod 777 /w/gm.sock
cd /w/project
setpriv --reuid=1000 --regid=1000 --clear-groups env HOME=/w/project NVRT_SOCKET=/w/gm.sock NVRT_DEBUG=1 CUDA_VISIBLE_DEVICES=0 \
  timeout 900 python3 tools/single_cell.py --dataset Bioresponse --split 0 --arm tabpfn3_aggregation_96 --output results/installed-box-cell.json > /w/out/cell.log 2>&1
echo "CELL rc=$?"
echo "--- result:"; head -c 600 /w/project/results/installed-box-cell.json 2>/dev/null; echo
echo "--- nvrt:"; grep -E "nvrt replay.*(whole_job|withdrawn|not served|engaged|shadow|in-process)" /w/out/cell.log | head -8
echo "--- errors:"; grep -E "Traceback|Error" /w/out/cell.log | head -4
python3 - <<PY
import json, collections
c = collections.Counter()
for l in open("/w/out/occ.jsonl"):
    e = json.loads(l)
    k = e.get("kind")
    if k == "lookup": print("LOOKUP", e.get("match"), e.get("draw"), e.get("record_id"))
    if k == "replay_disengaged": print("DISENGAGED", e.get("reason"), "|", (e.get("detail") or "")[:200])
    if k == "enqueue": c[e.get("unit")] += 1
    if k == "materialise" and e.get("role") == "whole_job_result": print("RESULT served tier", e.get("tier"))
print("enqueues by unit:", dict(c))
PY
'
# keep the run's own files (occupancy, daemon and cell logs, the result) for the record
KEEP=${KEEP:-/srv/work/gpu-mock/ttc805f-runs/$(date -u +%Y%m%dT%H%M%SZ)}; mkdir -p "$KEEP"
cp -a "$W/out/." "$KEEP/" 2>/dev/null; cp -a "$W/project/results" "$KEEP/results" 2>/dev/null
rm -rf "$W"
echo "=== done $(date -u +%FT%TZ); kept in $KEEP"
