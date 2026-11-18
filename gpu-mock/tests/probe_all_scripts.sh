#!/usr/bin/env bash
# Every mng3 script in the registry to its first optimizer step under replay,
# on an image: the gate-1 probe the study asked for on 2026-10-21, after
# 33 of the 86 scripts raised at step 0 on the mng3 image (a NaN guard on a
# served loss). For each registry/artifacts/mng3/scripts/<hash12>/<name>.py the
# record whose identity.entry_hash is the file's sha256 supplies the argv (the
# shortest such argv, so a chain's first segment rather than a resumed one);
# the script runs from its own work dir (script, data/fineweb10B shards, logs/)
# with NVRT_PROBE_STEPS=1, LANES at a time, one per daemon GPU, in the
# sidecar topology. Pass = exit 0 and the probe marker. A script with no
# matching record is listed and not run.
# Usage (on the VM):
#   sudo [IMG=ml-node:py311-torch2110] [DIMG=gpu-daemon:py311-torch2110] [PATCH_SRC=1] \
#        [LANES=4] [SPEED=100] [SHARDS=<dir of fineweb10B shards>] [OUT=/srv/runs/gm-probe-scripts] \
#        [ONLY=<hash12 prefix or name substring>] bash tests/probe_all_scripts.sh
#   PATCH_SRC=1 mounts this tree's torch patch over the image's and runs the daemon
#   from this tree's src; otherwise both come from the images (a gate).
set -uo pipefail
# EXP overrides: a frozen release tree sits outside experiments/
G=$(cd "$(dirname "$0")/.." && pwd); EXP=${EXP:-$(cd "$G/.." && pwd)}; REG="$EXP/gpu-tasks/registry"
IMG="${IMG:-ml-node:py311-torch2110}"; DIMG="${DIMG:-gpu-daemon:py311-torch2110}"
LANES="${LANES:-4}"; SPEED="${SPEED:-100}"; D="${OUT:-/srv/runs/gm-probe-scripts}"
SCRIPTS="$REG/artifacts/mng3/scripts"
if [ -z "${SHARDS:-}" ]; then
  SHARDS=$(ls -d /srv/cache/tasks/*/*/project/data/fineweb10B 2>/dev/null | while read -r d; do echo "$(ls "$d" | wc -l) $d"; done | sort -rn | head -1 | cut -d' ' -f2)
fi
[ -n "$SHARDS" ] && [ -d "$SHARDS" ] || { echo "no SHARDS directory (fineweb10B shards)" >&2; exit 2; }
VOL=gm-probe-sock; SIDE=gm-probe-sidecar
rm -rf "$D"; mkdir -p "$D/logs" "$D/work"; OCC="$D/occupancy.jsonl"; : > "$OCC"
# index: one line per script: hash12|name|record_id|argv (tab-separated argv)
python3 -I - "$REG" "$SCRIPTS" "${ONLY:-}" > "$D/index.txt" <<'PY'
import glob, hashlib, json, os, sys
reg, scripts, only = sys.argv[1], sys.argv[2], sys.argv[3]
by_hash = {}; not_ready = {}
for p in glob.glob(os.path.join(reg, "records", "*.json")):
    try:
        d = json.load(open(p))
    except Exception:
        continue
    ident = d.get("identity") or {}
    if ident.get("match_policy") == "path_bound":
        continue
    h = ident.get("entry_hash"); argv = ident.get("argv") or []
    if not h or not argv:
        continue
    # only a replay-ready record can be probed: an undrawable one sends the job
    # down the modelled path, real compute on the CPU, minutes per step
    if not (d.get("readiness") or {}).get("replay"):
        not_ready.setdefault(h, ident.get("record_id") or os.path.basename(p)[:-5])
        continue
    # prefer a launch that needs no checkpoint (no --resume), then the shortest
    # argv, then the record id, so the choice is the same on every machine
    # (a tie on length picked a resumed segment on one host, 2026-10-21)
    rid = ident.get("record_id") or os.path.basename(p)[:-5]
    key = ("--resume" in argv, len(argv), rid)
    cur = by_hash.get(h)
    if cur is None or key < cur[2]:
        by_hash[h] = (rid, list(argv), key)
n_norec = 0
for f in sorted(glob.glob(os.path.join(scripts, "*", "*.py"))):
    h12 = os.path.basename(os.path.dirname(f)); name = os.path.basename(f)
    if only and only not in h12 and only not in name:
        continue
    h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    rec = by_hash.get(h)
    if rec is None:
        why = "NO_REPLAY_RECORD:" + not_ready[h] if h in not_ready else "NO_RECORD"
        print(f"{h12}|{name}|{why}|"); n_norec += 1; continue
    rid, argv, _ = rec
    print(f"{h12}|{name}|{rid}|" + "\t".join(argv))
print(f"indexed {sum(1 for _ in glob.glob(os.path.join(scripts, '*', '*.py')))} scripts, {n_norec} without a record", file=sys.stderr)
PY
N=$(grep -vc "|NO_" "$D/index.txt"); NR=$(grep -c "|NO_" "$D/index.txt")
echo "scripts to probe: $N (no record: $NR) | image $IMG | sidecar $DIMG | patch $([ "${PATCH_SRC:-0}" = 1 ] && echo tree || echo image) | lanes $LANES | shards $SHARDS"
docker rm -f "$SIDE" >/dev/null 2>&1; docker ps -a --format '{{.Names}}' | grep '^gm-probe-' | xargs -r docker rm -f >/dev/null 2>&1
docker volume rm "$VOL" >/dev/null 2>&1; docker volume create "$VOL" >/dev/null
cleanup() { docker rm -f "$SIDE" >/dev/null 2>&1; docker ps -a --format '{{.Names}}' | grep '^gm-probe-' | xargs -r docker rm -f >/dev/null 2>&1; docker volume rm "$VOL" >/dev/null 2>&1; }
trap cleanup EXIT
if [ "${PATCH_SRC:-0}" = 1 ]; then SRC=/gm/src; SMNT=(-v "$G/src:/gm/src:ro"); else SRC=/opt/gpu-daemon/src; SMNT=(); fi
docker run -d --name "$SIDE" --entrypoint '' --cpus 2 --cpuset-cpus 0-1 --memory 2g --network none --label p006.role=gpu-daemon \
  "${SMNT[@]}" -v "$REG/records:/registry:ro" -v "$VOL:/var/run/nvidia-persistenced" -v "$D:/run-out" \
  "$DIMG" sh -c "python3 -I $SRC/daemon/gpu_daemon.py --socket /var/run/nvidia-persistenced/socket --registry /registry --gpus 4 --speed $SPEED \
    --occupancy /run-out/occupancy.jsonl > /run-out/daemon.log 2>&1 & python3 -I $SRC/bridge/overrides_bridge.py \
    --socket /var/run/nvidia-persistenced/socket --overrides /var/run/nvidia-persistenced/overrides.yaml > /run-out/bridge.log 2>&1 & wait" >/dev/null
for _ in $(seq 1 150); do docker run --entrypoint '' --rm --cpus 1 --memory 512m -v "$VOL:/s" "$IMG" test -S /s/socket 2>/dev/null && break; sleep 0.2; done
docker run --entrypoint '' --rm --cpus 1 --memory 512m -v "$VOL:/s" "$IMG" test -S /s/socket || { echo "sidecar socket never appeared"; cat "$D/daemon.log"; exit 1; }
PATCH_MOUNT=(); [ "${PATCH_SRC:-0}" = 1 ] && PATCH_MOUNT=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
T0=$(date +%s); i=0
while IFS='|' read -r h12 name rid argv; do
  case "$rid" in NO_*) continue;; esac
  while [ "$(jobs -rp | wc -l)" -ge "$LANES" ]; do sleep 1; done
  # the daemon serves 4 GPUs (--gpus 4 below): a lane maps onto GPU i mod 4,
  # not i mod LANES (with LANES=8 half the jobs asked for devices 4-7 and died
  # before any daemon contact, image probe 01:49Z 2026-10-21)
  lane=$((i % 4)); i=$((i+1))
  W="$D/work/$h12"; mkdir -p "$W/logs" "$W/data"; cp "$SCRIPTS/$h12/$name" "$W/"
  CMD="torchrun --standalone --nproc_per_node=1 $(printf '%s' "$argv" | tr '\t' ' ')"
  (
    t=$(date +%s)
    # --hostname localhost: torchrun's c10d rendezvous resolves the container hostname, which an isolated container cannot (first run, 2026-10-21)
    timeout 900 docker run --rm --name "gm-probe-$h12" --entrypoint '' --cpus 2 --memory 16g --network none --hostname localhost "${PATCH_MOUNT[@]}" \
      -v "$W:/work" -v "$SHARDS:/work/data/fineweb10B:ro" -v "$VOL:/var/run/nvidia-persistenced" \
      -e CUDA_VISIBLE_DEVICES="$lane" -e NVRT_PROBE_STEPS=1 -e NVRT_DEBUG=1 -e PYTHONUNBUFFERED=1 \
      "$IMG" bash -c "source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; cd /work; $CMD; echo JOB_EXIT=\$?" > "$D/logs/$h12.log" 2>&1
    echo "$h12 $name $rid $(( $(date +%s) - t ))s $CMD" >> "$D/launched.txt"
  ) &
done < "$D/index.txt"
wait
T1=$(date +%s)
{
echo "probe: $N scripts, $((T1-T0)) s wall, lanes $LANES, speed $SPEED | image $IMG | sidecar $DIMG | patch $([ "${PATCH_SRC:-0}" = 1 ] && echo tree || echo image)"
pass=0; fail=0; env=0
while IFS='|' read -r h12 name rid argv; do
  case "$rid" in NO_RECORD) echo "SKIP  $h12 $name: no record with this entry hash"; continue;; NO_REPLAY_RECORD:*) echo "SKIP  $h12 $name: no replay-ready record (${rid#NO_REPLAY_RECORD:} is not replay-ready; the modelled path would run real compute)"; continue;; esac
  L="$D/logs/$h12.log"
  ex=$(grep -oE '^JOB_EXIT=[0-9]+' "$L" | tail -1 | cut -d= -f2)
  if grep -q "probe: reached optimizer step 1" "$L" && [ "$ex" = 0 ]; then pass=$((pass+1)); echo "PASS  $h12 $name ($rid)"
  elif ! grep -q "nvrt replay\[r0\]: lookup" "$L" && grep -qE '^(ModuleNotFoundError|ImportError|SyntaxError)' "$L"; then
    # died before its first CUDA touch on a missing module or unparseable
    # source: the image or the artifact, not the mock (listed, not counted as
    # a mock failure; gpu-tasks has the list, 2026-10-21)
    env=$((env+1)); echo "ENV   $h12 $name ($rid): $(grep -E '^(ModuleNotFoundError|ImportError|SyntaxError)' "$L" | head -1 | cut -c1-160)"
  else fail=$((fail+1)); echo "FAIL  $h12 $name ($rid): exit ${ex:-none}; $(grep -E 'Error|error:|Traceback|Killed|non-finite' "$L" | grep -v Warning | tail -1 | cut -c1-200)"
  fi
done < "$D/index.txt"
echo "lookups: $(grep -c '"kind":"lookup"' "$OCC") | modes: $(grep -o '"kind":"lookup".*"mode":"[a-z]*"' "$OCC" | grep -o '"mode":"[a-z]*"' | sort | uniq -c | tr '\n' ' ')"
echo "materialise tiers: $(grep -o '"kind":"materialise".*"tier":"[a-z_]*"' "$OCC" | grep -o '"tier":"[a-z_]*"' | sort | uniq -c | tr '\n' ' ')"
echo "PROBE $([ "$fail" = 0 ] && echo PASS || echo FAIL): $pass passed, $fail failed, $env cannot run in this image (missing module or unparseable artifact), $NR skipped (no replay-ready record)"
echo "run dir: $D"
} 2>&1 | tee "$D/verdict.txt"
