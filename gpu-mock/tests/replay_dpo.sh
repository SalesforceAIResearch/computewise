#!/usr/bin/env bash
# End-to-end REPLAY of the TRL DPO smoke-task job (trl-dpo-loss-variants-4gpu) in
# the SIDECAR topology, on the GCP host. Not for the controller: the full job writes
# ~10.5 GB of real checkpoints and runs 4 ranks plus a daemon.
#   - a named Docker volume holds the daemon socket (AF_UNIX sockets do not cross
#     a virtiofs bind mount, and a named volume is the topology isolation uses);
#   - the daemon runs in its OWN container, registry read-only, no network;
#   - the job container sources the INSTALLED bundle (/usr/local/nvidia/env.sh, as the
#     agent's box does) and runs accelerate launch --num_processes 4 dpo.py.
# The job builds GPT-2 via from_pretrained, loads the real dataset, runs
# DPOTrainer.train() with no model math, reads loss/grad_norm from the record,
# and paces to the recorded train time divided by SPEED.
#
# Usage (on the VM):  sudo SPEED=100 bash tests/replay_dpo.sh
#   IMG=gm-dpo:x86            test image (tests/images/dpo-smoke/Dockerfile)
#   PATCH_SRC=1               mount this tree's src/torch_patch over the installed copy
#   HF_CACHE=/srv/work/gpu-mock/hf-cache   gpt2 + the hh-rlhf dataset, populated online once
#   OUT=/srv/runs/gm-replay-dpo   run directory (log, occupancy, checkpoints)
#   JOB_CPUS=16 JOB_CPUSET=16-31 JOB_MEM=64g   the job container, capped like the agent box
#                               (isolation's run_one.py default: 16 cores, 64 GB)
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
REG=$(cd "$G/../gpu-tasks/registry" && pwd)
IMG="${IMG:-gm-dpo:x86}"
SCRIPT_SHA=85be5703e343
SPEED="${SPEED:-100}"
HF_CACHE="${HF_CACHE:-/srv/work/gpu-mock/hf-cache}"
D="${OUT:-/srv/runs/gm-replay-dpo}"; OCC="$D/occupancy.jsonl"
VOL=gm-dpo-sock; NET=gm-dpo-net; SIDE=gm-dpo-sidecar
rm -rf "$D"; mkdir -p "$D/work" "$HF_CACHE"; : > "$OCC"
cp "$REG/artifacts/trl/scripts/$SCRIPT_SHA/dpo.py" "$D/work/dpo.py"
# A previous run of this test owns these names; remove all three before starting, or
# `docker run --name gm-dpo-job` fails and the cleanup trap then kills the old job.
docker rm -f "$SIDE" gm-dpo-job gm-dpo-smi >/dev/null 2>&1
docker volume rm "$VOL" >/dev/null 2>&1; docker volume create "$VOL" >/dev/null
docker network rm "$NET" >/dev/null 2>&1; docker network create --internal "$NET" >/dev/null
cleanup() { docker rm -f "$SIDE" gm-dpo-job gm-dpo-smi >/dev/null 2>&1; docker volume rm "$VOL" >/dev/null 2>&1; docker network rm "$NET" >/dev/null 2>&1; }
trap cleanup EXIT

# Populate the HF cache once (online), so the replay itself runs offline as the box does.
if [ ! -e "$HF_CACHE/.populated" ]; then
  docker run --rm --cpus 4 --memory 8g -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=0 -e HF_DATASETS_OFFLINE=0 \
    -e TRANSFORMERS_OFFLINE=0 "$IMG" python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer; from datasets import load_dataset
AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2')
load_dataset('trl-internal-testing/hh-rlhf-helpful-base-trl-style'); print('hf cache populated')" \
    && touch "$HF_CACHE/.populated" || { echo "HF cache population failed" >&2; exit 1; }
fi

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

# The record's exact argv, so the exact key matches and replay engages at
# interpreter start. --output_dir is argv_exempt; --dataset_num_proc is not.
ARGV='--dataset_name trl-internal-testing/hh-rlhf-helpful-base-trl-style --model_name_or_path gpt2 --per_device_train_batch_size 16 --learning_rate 1e-3 --gradient_accumulation_steps 4 --logging_steps 10 --eval_steps 500 --output_dir=/work/out --warmup_steps 150 --bf16 --logging_first_step --no_remove_unused_columns --dataset_num_proc 32'
PATCH_MOUNT=(); [ "${PATCH_SRC:-0}" = 1 ] && PATCH_MOUNT=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run -d --rm --name gm-dpo-smi --cpus 0.5 --cpuset-cpus 0-1 --memory 256m --network none \
  -v "$VOL:/var/run/nvidia-persistenced" "$IMG" bash -c 'source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
  while :; do echo "$(date +%s.%N | cut -c1-14) $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | paste -sd";")"; sleep 1; done' \
  > /dev/null 2>&1
( sleep 3; while docker ps --format '{{.Names}}' | grep -qx gm-dpo-job; do docker stats --no-stream --format '{{.MemUsage}}' gm-dpo-job; sleep 5; done ) > "$D/job-mem.log" 2>&1 &
T0=$(date +%s)
timeout "${TIMEOUT:-2400}" docker run --rm --name gm-dpo-job --cpus "${JOB_CPUS:-16}" --cpuset-cpus "${JOB_CPUSET:-16-31}" --memory "${JOB_MEM:-64g}" --network "$NET" "${PATCH_MOUNT[@]}" \
  -v "$D/work:/work" -v "$VOL:/var/run/nvidia-persistenced" -v "$HF_CACHE:/hf" -e HF_HOME=/hf \
  -e NVRT_DEBUG=1 -e PYTHONUNBUFFERED=1 \
  "$IMG" bash -c "source /usr/local/nvidia/env.sh; cd /work; accelerate launch --num_processes 4 --mixed_precision bf16 dpo.py $ARGV; echo JOB_EXIT=\$?" \
  > "$D/replay.log" 2>&1
rc=$?; T1=$(date +%s)
echo "docker exit: $rc   wall: $((T1-T0)) s   speed $SPEED"
echo "--- replay engagement:"; grep -E "replay\[r[0-9]\]: (lookup|replay engaged)" "$D/replay.log" | head -8
echo "--- job exit:"; grep -E "^JOB_EXIT=" "$D/replay.log"
echo "--- errors:"; grep -iE "Traceback|Error:|SIGSEGV|Signal 11|exitcode" "$D/replay.log" | grep -vE "FutureWarning|UserWarning" | head -8
echo "--- trainer log lines:"; grep -oE "\{'loss': [^}]*\}|\{'train_runtime': [^}]*\}" "$D/replay.log" | head -6; echo "  ..."; grep -oE "\{'loss': [^}]*\}|\{'train_runtime': [^}]*\}" "$D/replay.log" | tail -3
echo "--- checkpoints:"; du -sh "$D/work/out" 2>/dev/null; ls "$D/work/out" 2>/dev/null | head
docker logs gm-dpo-smi > "$D/nvidia-smi.log" 2>&1; docker rm -f gm-dpo-smi >/dev/null 2>&1
echo "--- nvidia-smi:"; python3 -I "$G/tests/smi_summary.py" "$D/nvidia-smi.log"
echo "--- job container memory:"; python3 -I -c "
import re, sys
v = [float(m.group(1)) * {'MiB': 1/1024, 'GiB': 1, 'KiB': 1/1024**2}[m.group(2)] for l in open(sys.argv[1]) for m in [re.match(r'([0-9.]+)(MiB|GiB|KiB)', l.strip())] if m]
print(f'samples {len(v)}, peak {max(v):.1f} GiB' if v else 'no samples')" "$D/job-mem.log"
RID=$(grep '"kind":"lookup"' "$OCC" | grep -o '"record_id":"[^"]*"' | head -1 | cut -d'"' -f4)
if [ -n "$RID" ]; then echo "--- record check ($RID):"; python3 -I "$G/tests/check_trainer_replay.py" "$D/replay.log" "$REG/records/$RID.json"
else echo "--- record check: no lookup in the occupancy log, so replay never engaged (RECORD CHECK FAIL)"; fi
echo "--- host time per job:"; python3 -I "$G/tests/host_times.py" "$OCC"
echo "run dir: $D"
