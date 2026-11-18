#!/usr/bin/env bash
# REPLAY of one trl-regression-battery job (one process, one GPU) on the GCP host,
# in the sidecar topology of tests/replay_dpo.sh: daemon + bridge in their own
# container, the socket on a named volume, an nvidia-smi sampler, the job capped
# like the agent box. The job runs the task's own launch line from a copy of
# gpu-tasks' staged task tree, with a writable copy of its staged HF cache, so
# replay engages on the exact key and the script's Trainer runs with no model math.
#
# Usage (on the VM):  sudo JOB=dpo-seed1 SPEED=100 bash tests/replay_battery.sh
#   JOB          a job_id of tasks/trl-regression-battery.json (dpo-seed1, reward-modeling-seed1, sft-seed1, ...)
#   IMG=gm-battery:x86   an image with the battery's stack and gpu-mock installed
#   PATCH_SRC=1          mount this tree's torch patch over the installed copy
#   GPU=0                the GPU the job is pinned to (CUDA_VISIBLE_DEVICES)
#   OUT=/srv/runs/gm-battery-<JOB>   run directory
#   JOB_CPUS=16 JOB_CPUSET=16-31 JOB_MEM=64g   caps on the job container
#   CPROFILE=1           profile the replayed job (rank 0) into <run dir>/work/replay.prof
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
EXP=$(cd "$G/.." && pwd); REG="$EXP/gpu-tasks/registry"
JOB="${JOB:?set JOB to a battery job_id}"
IMG="${IMG:-gm-battery:x86}"; SPEED="${SPEED:-100}"; GPU="${GPU:-0}"
STAGE="${STAGE:-/opt/computewise-deploy/stage/gpu-tasks/trl-regression-battery}"
HF_COPY="${HF_COPY:-/srv/work/gpu-mock/battery-hf}"
D="${OUT:-/srv/runs/gm-battery-$JOB}"; OCC="$D/occupancy.jsonl"
VOL=gm-bat-sock; NET=gm-bat-net; SIDE=gm-bat-sidecar
LAUNCH=$(python3 -I -c "import json,sys; t=json.load(open(sys.argv[1])); print(next(j['launch'] for j in t['jobs'] if j['job_id']==sys.argv[2]))" "$REG/tasks/trl-regression-battery.json" "$JOB") \
  || { echo "no job $JOB in the battery task" >&2; exit 2; }
rm -rf "$D"; mkdir -p "$D"; : > "$OCC"
cp -a "$STAGE/tree" "$D/work"
# the staged cache belongs to gpu-tasks: copy it once to a writable cache of our own
# (hub/ is only read; datasets/ gets the jobs' map outputs)
[ -d "$HF_COPY/hub" ] || { mkdir -p "$HF_COPY"; cp -a "$STAGE/hf/." "$HF_COPY/"; }
docker rm -f "$SIDE" gm-bat-job gm-bat-smi >/dev/null 2>&1
docker volume rm "$VOL" >/dev/null 2>&1; docker volume create "$VOL" >/dev/null
docker network rm "$NET" >/dev/null 2>&1
cleanup() { docker rm -f "$SIDE" gm-bat-job gm-bat-smi >/dev/null 2>&1; docker volume rm "$VOL" >/dev/null 2>&1; }
trap cleanup EXIT

docker run -d --name "$SIDE" --cpus 1 --cpuset-cpus 0-1 --memory 1g --network none --label p006.role=gpu-daemon \
  -v "$G/src:/gm/src:ro" -v "$REG/records:/registry:ro" \
  -v "$VOL:/var/run/nvidia-persistenced" -v "$D:/run-out" \
  python:3.11-slim-bookworm sh -c "python -I /gm/src/daemon/gpu_daemon.py \
    --socket /var/run/nvidia-persistenced/socket --registry /registry --gpus 4 --speed $SPEED \
    --occupancy /run-out/occupancy.jsonl & python -I /gm/src/bridge/overrides_bridge.py \
    --socket /var/run/nvidia-persistenced/socket --overrides /var/run/nvidia-persistenced/overrides.yaml & wait" >/dev/null
for _ in $(seq 1 100); do
  docker run --entrypoint '' --rm --cpus 1 --memory 512m -v "$VOL:/s" "$IMG" test -S /s/socket 2>/dev/null && break
  sleep 0.2
done
docker run -d --rm --name gm-bat-smi --cpus 0.5 --cpuset-cpus 0-1 --memory 256m --network none \
  -v "$VOL:/var/run/nvidia-persistenced" "$IMG" bash -c 'source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
  while :; do echo "$(date +%s.%N | cut -c1-14) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | paste -sd";")"; sleep 1; done' \
  > /dev/null 2>&1

PATCH_MOUNT=(); [ "${PATCH_SRC:-0}" = 1 ] && PATCH_MOUNT=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
( sleep 3; while docker ps --format '{{.Names}}' | grep -qx gm-bat-job; do docker stats --no-stream --format '{{.MemUsage}}' gm-bat-job; sleep 5; done ) > "$D/job-mem.log" 2>&1 &
T0=$(date +%s)
timeout "${TIMEOUT:-5400}" docker run --rm --name gm-bat-job --cpus "${JOB_CPUS:-16}" --cpuset-cpus "${JOB_CPUSET:-16-31}" --memory "${JOB_MEM:-64g}" \
  --network none "${PATCH_MOUNT[@]}" \
  -v "$D/work:/work" -v "$VOL:/var/run/nvidia-persistenced" -v "$HF_COPY:/hf" \
  -e HF_HOME=/hf -e PYTHONPATH=/work -e CUDA_VISIBLE_DEVICES="$GPU" -e WANDB_MODE=offline \
  -e NVRT_DEBUG=1 -e PYTHONUNBUFFERED=1 ${CPROFILE:+-e NVRT_CPROFILE=/work/replay.prof} \
  "$IMG" bash -c "source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; export PYTHONPATH=/work:\$PYTHONPATH; cd /work; $LAUNCH; echo JOB_EXIT=\$?" \
  > "$D/job.log" 2>&1
rc=$?; T1=$(date +%s)
{
docker logs gm-bat-smi > "$D/nvidia-smi.log" 2>&1; docker rm -f gm-bat-smi >/dev/null 2>&1
echo "job $JOB   docker exit: $rc   wall: $((T1-T0)) s   speed $SPEED"
echo "--- replay:"; grep -oE "nvrt replay\[r[0-9]\]: (lookup|replay engaged|pacing on|holding|Trainer seam|replay withdrawn|lr trace).*" "$D/job.log" | head -8
echo "--- job exit:"; grep -E "^JOB_EXIT=" "$D/job.log"
echo "--- errors:"; tr '\r' '\n' < "$D/job.log" | grep -E "Traceback|Error:|SIGSEGV|Signal|Killed" | grep -vE "FutureWarning|UserWarning" | head -6
echo "--- nvidia-smi:"; python3 -I "$G/tests/smi_summary.py" "$D/nvidia-smi.log"
echo "--- job container memory:"; python3 -I -c "
import re, sys
v = [float(m.group(1)) * {'MiB': 1/1024, 'GiB': 1, 'KiB': 1/1024**2}[m.group(2)] for l in open(sys.argv[1]) for m in [re.match(r'([0-9.]+)(MiB|GiB|KiB)', l.strip())] if m]
print(f'samples {len(v)}, peak {max(v):.1f} GiB' if v else 'no samples')" "$D/job-mem.log"
RID=$(grep '"kind":"lookup"' "$OCC" | grep -o '"record_id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -n "$RID" ]; then echo "--- record check ($RID):"; python3 -I "$G/tests/check_trainer_replay.py" "$D/job.log" "$REG/records/$RID.json"
else echo "--- record check: no lookup in the occupancy log, so replay never engaged (RECORD CHECK FAIL)"; fi
echo "--- host time:"; python3 -I "$G/tests/host_times.py" "$OCC"
echo "run dir: $D"
} 2>&1 | tee "$D/verdict.log"
