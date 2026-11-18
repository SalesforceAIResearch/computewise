#!/usr/bin/env bash
# PROBE (not a gate): the seeded initial-weights model_digest of each SF
# checkpoint-collision W0 pair member, for analysis's artifact-identity question
# (2026-11-03 20:44 MST): does each W0 checkpoint's model_digest equal its own seeded
# initialisation?
#
# Same environment as W0-sfck-reference: the r7.1 packages box it ran on (2f997a19,
# installed patch cd4c9ea3), the daemon source of its sidecar (9130def6 = 18dab164),
# the deployed registry, the staged pkg-sf-checkpoint-collision tree (the two entry
# scripts' bytes are W0's: 17ad493a / 77c3f975), uid 1000, network none, the same
# argv (--seed N --out-dir members/<method>/seed<N>). Each member is run UNMODIFIED
# through tests/capture_init_state.py, which stops the script at its first
# `for step in range(...)` line (before any forward, backward or optimizer step) and
# torch.saves {"model": model.state_dict()}. Two arms per member:
#   R   replay as in W0 (the record matched, replay engaged)
#   N   NVRT_NO_REPLAY=1: the script's own initialisation with no replay at all
# Digests: isolation's tools/weights_digest.py (935475e7), the W0 capture's own tool.
# Each capture is digested at once and its .pt deleted (about 0.57 GB each); the digests,
# logs and the comparison with W0's model_digest are kept under /srv/work/gpu-mock/sf-init-digests/.
# Usage (on the E2):  sudo bash tests/probe_sf_init_digests.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG=${IMG:-sha256:2f997a197e6ef4d5a7e8ac431d933e31ffe62ce472b39f43c86e0185ce6e22c6}
SIDE=${SIDE:-sha256:9130def60af36d337f5758f8e913ec278f7cd699f4399a3e9cb2edd447f2afb1}
STAGED=${STAGED:-/srv/cache/tasks/pkg-sf-checkpoint-collision/ddde9addd294-2878209f/project}
REG=${REG:-/opt/computewise/gpu-tasks/registry/records}
DIG=${DIG:-/opt/computewise/isolation/tools/weights_digest.py}
W0=${W0:-}          # W0's weights-digests.jsonl (copied from the controller)
SEEDS=${SEEDS:-"0 2 3 4 5 6 8 9 10"}
ARMS=${ARMS:-"R N"}
K=/srv/work/gpu-mock/sf-init-digests/$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $K/home/project/init $K/logs $K/daemon
echo "=== SF init-digest probe $(date -u +%FT%TZ) box ${IMG:7:12} sidecar ${SIDE:7:12} -> $K"
echo "entry hashes: $(sha256sum $STAGED/ensemble_train/contra-soft-muon/contra-muon-to-soft-muon.py | cut -c1-12) $(sha256sum $STAGED/ensemble_train/radial-damp/radial_scale_then_radius_correct.py | cut -c1-12); digest tool $(sha256sum $DIG | cut -c1-8)"
docker run --rm --entrypoint sh --network none -v $K/daemon:/out $SIDE -c 'cp -a /opt/gpu-daemon/src/. /out/'
echo "daemon src $(cd $K/daemon && find daemon bridge convert -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
chown -R 1000:1000 $K/home $K/logs
for arm in $ARMS; do for seed in $SEEDS; do for m in contra-soft-muon radial-damp; do
  case $m in contra-soft-muon) S=ensemble_train/contra-soft-muon/contra-muon-to-soft-muon.py;; radial-damp) S=ensemble_train/radial-damp/radial_scale_then_radius_correct.py;; esac
  tag=$arm-$m-s$seed; NR=""; [ $arm = N ] && NR="-e NVRT_NO_REPLAY=1"
  docker run --entrypoint '' --rm --cpus 4 --memory 16g --network none --user 1000:1000 $NR \
    -v $STAGED:/stage:ro -v $K/daemon:/gm/src:ro -v $REG:/reg:ro -v $G/tests/capture_init_state.py:/gm/capture_init_state.py:ro \
    -v $K/home/project/init:/init -e HOME=/tmp/h -e CUDA_VISIBLE_DEVICES=0 -e NVRT_SOCKET=/tmp/gm.sock "$IMG" bash -c "
source /usr/local/nvidia/env.sh 2>/dev/null || true
mkdir -p /tmp/h /tmp/w && cd /tmp/w && ln -s /stage/data data && ln -s /stage/ensemble_train ensemble_train
python3 -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /reg --gpus 4 --speed 1 --occupancy /tmp/occ.jsonl > /tmp/d.log 2>&1 &
for i in \$(seq 200); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
timeout 600 python3 /gm/capture_init_state.py /init/$tag.pt $S --seed $seed --out-dir members/$m/seed$seed; echo \"RC \$?\"
python3 -c \"
import json
for l in open('/tmp/occ.jsonl'):
    e = json.loads(l)
    if e.get('kind') == 'lookup': print('LOOKUP', e.get('match'), e.get('draw'), e.get('record_id'))
    if e.get('kind') == 'replay_disengaged': print('DISENGAGED', e.get('reason'))
\" 2>/dev/null" > $K/logs/$tag.log 2>&1
  echo "$tag: $(grep -E '^CAPTURE|^RC|^LOOKUP|^DISENGAGED' $K/logs/$tag.log | tr '\n' ' ' | cut -c1-260)"
  # digest this capture with the W0 tool at once, then drop the .pt (0.57 GB each)
  if [ -f $K/home/project/init/$tag.pt ]; then
    printf '{"members": [{"checkpoint": "init/%s.pt"}]}' "$tag" > $K/home/project/init-index.json
    mkdir -p $K/tmp $K/dg/$tag; (cd $K/dg/$tag && python3 -I $DIG $K/home $K/tmp init-index.json 1 4294967296 600) > $K/dg/$tag/summary.json 2>&1
    cat $K/dg/$tag/weights-digests.jsonl >> $K/weights-digests.jsonl 2>/dev/null
    rm -f $K/home/project/init/$tag.pt
  fi
done; done; done
echo "digested $(wc -l < $K/weights-digests.jsonl 2>/dev/null || echo 0) captures"
python3 - $K "$W0" <<'PY' | tee $K/compare.txt
import json, os, sys, collections
K, w0 = sys.argv[1], sys.argv[2]
cap = {}
for l in open(K + "/weights-digests.jsonl"):
    r = json.loads(l)
    tag = os.path.basename(r["listed"])[:-3]          # <arm>-<method>-s<seed>
    arm, rest = tag.split("-", 1); m, s = rest.rsplit("-s", 1)
    cap[(arm, m, int(s))] = r.get("model_digest") or ("ERROR " + str(r.get("model_error") or r.get("error")))
ref = {}
if w0:
    for l in open(w0):
        r = json.loads(l); p = r["listed"].split("/")
        ref[(p[1], int(p[2][4:]))] = r.get("model_digest")
print("method seed | W0 model_digest | init R (replay) | init N (no replay) | W0==R W0==N R==N")
for (m, s) in sorted({(m, s) for (_, m, s) in cap}, key=lambda x: (x[1], x[0])):
    w, r, n = ref.get((m, s)), cap.get(("R", m, s)), cap.get(("N", m, s))
    f = lambda a, b: "-" if not a or not b else ("EQ" if a == b else "DIFF")
    print(f"{m:16s} {s:2d} | {(w or '-')[:16]} | {(r or '-')[:16]} | {(n or '-')[:16]} | {f(w, r)} {f(w, n)} {f(r, n)}")
PY
echo "=== done $(date -u +%FT%TZ); kept in $K"
