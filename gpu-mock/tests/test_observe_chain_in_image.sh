#!/usr/bin/env bash
# The observe chain: daemon -> bridge -> overrides.yaml -> mock NVML -> nvidia-smi.
# A driver job keeps GPU 1 busy for 8 s and holds 20 GiB on it. nvidia-smi in the
# image must show GPU 1 busy and about 20480 MiB used, and GPUs 0, 2 and 3 idle.
# Usage (on the VM):  sudo IMG=gm-dpo:x86 bash tests/test_observe_chain_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"; VOL=gm-obs-sock
docker rm -f gm-obs-side gm-obs-drv >/dev/null 2>&1; docker volume rm $VOL >/dev/null 2>&1; docker volume create $VOL >/dev/null
trap 'docker rm -f gm-obs-side gm-obs-drv >/dev/null 2>&1; docker volume rm $VOL >/dev/null 2>&1' EXIT
docker run -d --name gm-obs-side --cpus 1 --cpuset-cpus 0-1 --memory 1g --network none -v "$G/src:/gm/src:ro" -v "$VOL:/var/run/nvidia-persistenced" \
  python:3.11-slim-bookworm sh -c "python -I /gm/src/daemon/gpu_daemon.py --socket /var/run/nvidia-persistenced/socket --gpus 4 --speed 1 --occupancy /tmp/occ.jsonl & python -I /gm/src/bridge/overrides_bridge.py --socket /var/run/nvidia-persistenced/socket --overrides /var/run/nvidia-persistenced/overrides.yaml & wait" >/dev/null
sleep 2
docker run -d --name gm-obs-drv --cpus 1 --cpuset-cpus 0-1 --memory 512m --network none -v "$G/src:/gm/src:ro" -v "$VOL:/var/run/nvidia-persistenced" python:3.11-slim-bookworm python -I -c "
import sys, time; sys.path.insert(0, '/gm/src/torch_patch')
from daemon_client import DaemonClient
c = DaemonClient('/var/run/nvidia-persistenced/socket'); c.hello(argv=['drv.py'])
c.enqueue(gpu=1, op='alloc', modelled_s=0.0, mem_delta=20 * 1024 ** 3)
t = time.monotonic()
while time.monotonic() - t < 8:
    c.enqueue(gpu=1, op='k', modelled_s=0.2); c.sync(gpu=1)
time.sleep(30)" >/dev/null
OUT=$(docker run --entrypoint '' --rm --cpus 1 --cpuset-cpus 0-1 --memory 512m --network none -v "$VOL:/var/run/nvidia-persistenced" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
S=""; for i in $(seq 10); do L="t+$i s  $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | paste -sd";")"; echo "$L"; S="$S
$L"; sleep 1; done
# verdict: some sample shows GPU 1 busy with 20480 MiB, and no sample shows another GPU busy
if echo "$S" | grep -qE "1, 100, 20480" && ! echo "$S" | grep -qE "(^|;)[023], 100,"; then echo "OBSERVE CHAIN PASS: GPU 1 busy with 20480 MiB in nvidia-smi, GPUs 0/2/3 idle"; else echo "OBSERVE CHAIN FAIL: expected GPU 1 at 100 % with 20480 MiB and the others idle"; fi
echo "--- memory total/used/free MiB: $(nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits | paste -sd";")"
echo "--- overrides.yaml ($(stat -c %s /var/run/nvidia-persistenced/overrides.yaml 2>/dev/null) bytes):"; head -c 1200 /var/run/nvidia-persistenced/overrides.yaml 2>/dev/null; echo
echo "--- nvidia-smi:"; nvidia-smi | tail -7')
echo "$OUT"
echo "--- sidecar log:"; docker logs gm-obs-side 2>&1 | tail -5
echo "$OUT" | grep "OBSERVE CHAIN" || echo "OBSERVE CHAIN FAIL: no verdict line"
