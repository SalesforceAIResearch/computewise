#!/usr/bin/env bash
# The real segment chain (gpu-tasks' build/gen_segment.py): three consecutive
# single-GPU jobs of contra-soft-muon seed 0, each resuming the previous one's
# checkpoint, replayed against the candidate records outside the registry.
# Runs on the GCP host in the modded-nanogpt image, one job at a time, with the
# daemon and bridge in a sidecar on a named volume. Checks per segment: exact
# lookup on the segment's own script hash and argv (the --resume path included),
# the served validation losses against the record (first and last), the
# checkpoint written by --stop-at, and exit 0. Also: a segment 2 launched before
# segment 1's save fails on the missing file with no daemon contact.
#
# Usage (on the VM):  sudo TREE=<dir with contra-soft-muon/ and data/fineweb10B/> \
#                        [IMG=ml-node:py311-torch2110] [SPEED=100] bash tests/replay_segments.sh
#   SHARDS   a directory of fineweb10B shards to link into TREE/data/fineweb10B if TREE lacks them
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd); EXP=$(cd "$G/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"; SPEED="${SPEED:-100}"
CAND="$EXP/gpu-tasks/registry/candidates/segment-contra-soft-muon"
SCRIPT="$EXP/gpu-tasks/registry/artifacts/mng3/scripts/4464f5ce77f7/contra-muon-to-soft-muon.py"
D="${OUT:-/srv/runs/gm-segments}"; OCC="$D/occupancy.jsonl"
VOL=gm-seg-sock; SIDE=gm-seg-sidecar
[ -d "$CAND/records" ] || { echo "no candidate records at $CAND/records" >&2; exit 2; }
[ -f "$SCRIPT" ] || { echo "no segment script at $SCRIPT" >&2; exit 2; }
rm -rf "$D"; mkdir -p "$D/work/contra-soft-muon" "$D/work/checkpoints" "$D/work/data"; : > "$OCC"
cp "$SCRIPT" "$D/work/contra-soft-muon/"
if [ -n "${TREE:-}" ] && [ -d "$TREE/data/fineweb10B" ]; then ln -s "$TREE/data/fineweb10B" "$D/work/data/fineweb10B"
elif [ -n "${SHARDS:-}" ] && [ -d "$SHARDS" ]; then ln -s "$SHARDS" "$D/work/data/fineweb10B"
else echo "need TREE with data/fineweb10B, or SHARDS" >&2; exit 2; fi
echo "shards: $(ls "$D/work/data/fineweb10B/" | wc -l) files in $(readlink -f "$D/work/data/fineweb10B")"
docker rm -f "$SIDE" gm-seg-job >/dev/null 2>&1; docker volume rm "$VOL" >/dev/null 2>&1; docker volume create "$VOL" >/dev/null
cleanup() { docker rm -f "$SIDE" gm-seg-job >/dev/null 2>&1; docker volume rm "$VOL" >/dev/null 2>&1; }
trap cleanup EXIT
docker run -d --name "$SIDE" --cpus 1 --cpuset-cpus 0-1 --memory 1g --network none --label p006.role=gpu-daemon \
  -v "$G/src:/gm/src:ro" -v "$CAND/records:/registry:ro" -v "$VOL:/var/run/nvidia-persistenced" -v "$D:/run-out" \
  python:3.11-slim-bookworm sh -c "python -I /gm/src/daemon/gpu_daemon.py --socket /var/run/nvidia-persistenced/socket \
    --registry /registry --gpus 4 --speed $SPEED --occupancy /run-out/occupancy.jsonl & python -I /gm/src/bridge/overrides_bridge.py \
    --socket /var/run/nvidia-persistenced/socket --overrides /var/run/nvidia-persistenced/overrides.yaml & wait" >/dev/null
for _ in $(seq 1 100); do docker run --entrypoint '' --rm --cpus 1 --memory 512m -v "$VOL:/s" "$IMG" test -S /s/socket 2>/dev/null && break; sleep 0.2; done
PATCH_MOUNT=(); [ "${PATCH_SRC:-0}" = 1 ] && PATCH_MOUNT=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
run_seg() {  # name, launch line
  local name="$1"; shift
  timeout "${TIMEOUT:-3600}" docker run --rm --name gm-seg-job --cpus 16 --cpuset-cpus 16-31 --memory 64g --network none "${PATCH_MOUNT[@]}" \
    -v "$D/work:/work" -v "$VOL:/var/run/nvidia-persistenced" -e CUDA_VISIBLE_DEVICES=0 -e NVRT_DEBUG=1 -e PYTHONUNBUFFERED=1 \
    "$IMG" bash -c "source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; cd /work; $*; echo JOB_EXIT=\$?" > "$D/$name.log" 2>&1
  echo "--- $name: $(grep -o 'JOB_EXIT=[0-9]*' "$D/$name.log") | $(grep -oE 'lookup\[exact\] -> mode=[a-z]+ record=[^ ]*' "$D/$name.log" | head -1) | val lines: $(grep -cE '^step:[0-9]+/[0-9]+ val_loss' "$D/$name.log")"
  grep -E '^step:[0-9]+/[0-9]+ val_loss' "$D/$name.log" | sed -n '1p;$p' | cut -c1-120
  grep -E 'Traceback|Error' "$D/$name.log" | grep -v Warning | head -3
}
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "[ok  ] $1 -> $2"; else fail=$((fail+1)); echo "[FAIL] $1 -> $2 (want $3)"; fi; }
echo "##### segment 2 before segment 1 (must fail on the missing checkpoint, no daemon contact)"
run_seg early "torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py --seed 0 --resume checkpoints/seed0-step1000.pt --stop-at 2000"
chk "early segment 2 fails" "$(grep -o 'JOB_EXIT=[0-9]*' "$D/early.log")" "JOB_EXIT=1"
chk "early segment 2 names the missing file" "$(grep -c 'checkpoints/seed0-step1000.pt' "$D/early.log" | awk '{print ($1>0)}')" 1
chk "early segment 2 never said hello" "$(grep -c '"kind":"job_hello"' "$OCC")" 0
echo "##### segment 1"
run_seg seg1 "torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py --seed 0 --stop-at 1000"
chk "segment 1 exits 0" "$(grep -o 'JOB_EXIT=[0-9]*' "$D/seg1.log")" "JOB_EXIT=0"
chk "segment 1 wrote its checkpoint" "$(test -s "$D/work/checkpoints/seed0-step1000.pt" && echo yes || echo no)" yes
chk "checkpoint holds no fake class" "$(python3 -I -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); d=b''.join(z.read(n) for n in z.namelist() if n.endswith('data.pkl')); print(b'fakegpu' in d or b'FakeCuda' in d)" "$D/work/checkpoints/seed0-step1000.pt" 2>/dev/null)" False
echo "##### segment 2"
run_seg seg2 "torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py --seed 0 --resume checkpoints/seed0-step1000.pt --stop-at 2000"
chk "segment 2 exits 0" "$(grep -o 'JOB_EXIT=[0-9]*' "$D/seg2.log")" "JOB_EXIT=0"
chk "segment 2 resumed at 1000" "$(grep -c 'Resumed from checkpoints/seed0-step1000.pt at step 1000' "$D/seg2.log")" 1
chk "segment 2 wrote its checkpoint" "$(test -s "$D/work/checkpoints/seed0-step2000.pt" && echo yes || echo no)" yes
echo "##### segment 3"
run_seg seg3 "torchrun --standalone --nproc_per_node=1 contra-soft-muon/contra-muon-to-soft-muon.py --seed 0 --resume checkpoints/seed0-step2000.pt"
chk "segment 3 exits 0" "$(grep -o 'JOB_EXIT=[0-9]*' "$D/seg3.log")" "JOB_EXIT=0"
echo "##### lookups"; grep '"kind":"lookup"' "$OCC" | grep -oE '"mode":"[a-z]+","record_id":"[^"]*","match":"[a-z_]+"|"derived_from":\{[^}]*\}' | paste -sd' ' | fold -w 220
chk "three exact replay lookups" "$(grep '"kind":"lookup"' "$OCC" | grep -c '"mode":"replay".*"match":"exact"')" 3
echo "##### served validation losses vs the records"; python3 -I - "$D" "$CAND/records" <<'PY'
import json, re, sys, glob
D, R = sys.argv[1], sys.argv[2]
recs = {json.load(open(p))["identity"]["record_id"]: json.load(open(p)) for p in glob.glob(R + "/*.json")}
occ = [json.loads(l) for l in open(D + "/occupancy.jsonl") if l.strip()]
rids = [e["record_id"] for e in occ if e["kind"] == "lookup" and e.get("record_id")]
bad = 0
for name, rid in zip(("seg1", "seg2", "seg3"), rids):
    vals = [float(m.group(2)) for m in re.finditer(r"^step:(\d+)/\d+ val_loss:([0-9.]+)", open(D + f"/{name}.log", errors="replace").read(), re.M)]
    want = recs[rid]["series"]["val_loss"]["values"]
    ok = len(vals) == len(want) and all(abs(a - b) < 5e-5 for a, b in zip(vals, want))
    bad += not ok
    print(f"{name} {rid[-14:]}: printed {len(vals)} validations, record {len(want)}; first {vals[:1]} vs {want[:1]}; last {vals[-1:]} vs {want[-1:]}; {'MATCH' if ok else 'MISMATCH'}")
print("VAL_LOSS", "MATCH" if not bad else "MISMATCH")
PY
echo; echo "$pass passed, $fail failed; run dir $D"
