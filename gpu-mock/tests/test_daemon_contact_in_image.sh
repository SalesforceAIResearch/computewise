#!/usr/bin/env bash
# Who contacts the daemon, inside an image with gpu-mock installed (/opt/gpu-mock).
# Starts a daemon in the container, runs the kinds of interpreter a box sees, and
# reads the occupancy log:
#   - python -c / a CPU-only script / a script that imports torch but never uses
#     CUDA / torch.cuda.device_count()   -> no hello at all;
#   - a script that puts one tensor on cuda  -> exactly one job_hello, with t_start;
#   - the overrides bridge                   -> observer_hello, never job_hello.
# Also times interpreter start with and without the site dir on PYTHONPATH.
# Usage (on the VM):  sudo IMG=gm-dpo:x86 [PATCH_SRC=1] bash tests/test_daemon_contact_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --gpus 4 --occupancy /tmp/occ.jsonl >/tmp/daemon.log 2>&1 &
for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
cd /tmp; printf "print(sum(range(10)))\n" > cpu_only.py
printf "import torch\nx = torch.ones(3)\nprint(torch.cuda.device_count(), x.sum().item())\n" > torch_no_cuda.py
printf "import torch\ny = torch.zeros(2, device=\"cuda\")\nprint(\"cuda\", y.device)\n" > touches_cuda.py
n() { grep -c "\"kind\":\"$1\"" /tmp/occ.jsonl 2>/dev/null || true; }
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "[ok  ] $1 -> $2"; else fail=$((fail+1)); echo "[FAIL] $1 -> $2 (want $3)"; fi; }
python -c "print(1)" >/dev/null;                 chk "python -c: job_hello count" "$(n job_hello)" 0
python cpu_only.py >/dev/null;                   chk "cpu-only script: job_hello count" "$(n job_hello)" 0
python torch_no_cuda.py >/dev/null;              chk "imports torch, no cuda: job_hello count" "$(n job_hello)" 0
python -c "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))" > /tmp/probe.txt
chk "device query output" "$(cat /tmp/probe.txt)" "4 NVIDIA H100 80GB HBM3"
chk "device query: job_hello count" "$(n job_hello)" 0
chk "find_spec(torch) before import still patches" "$(python -c "import importlib.util; importlib.util.find_spec(\"torch\"); import torch; print(torch.cuda.device_count())" 2>/dev/null)" 4
if python -c "import transformers" 2>/dev/null; then
  chk "transformers before torch still patches" "$(python -c "import transformers, torch; print(torch.cuda.device_count(), type(torch.nn.parallel.DistributedDataParallel).__name__ != \"\", torch.nn.parallel.DistributedDataParallel.__name__)" 2>/dev/null)" "4 True _FakeDistributedDataParallel"
else echo "[skip] transformers before torch still patches: transformers is not installed in this image"; fi
if python -c "import trl" 2>/dev/null; then
  chk "trl first still patches" "$(python -c "from trl import DPOTrainer; import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)" "NVIDIA H100 80GB HBM3"
else echo "[skip] trl first still patches: trl is not installed in this image"; fi
chk "virtual clock and FLOP charger installed" "$(python -c "import torch, virtual_clock as vc, flop_charger as fc; print(vc._time_patched, fc._installed)" 2>/dev/null)" "True True"
chk "import torch prints no warning" "$(python -c "import torch" 2>&1 | grep -ci warn)" 0
python touches_cuda.py > /tmp/t.txt 2>/tmp/t.err; chk "touches cuda: output" "$(cat /tmp/t.txt)" "cuda cuda:0"
chk "touches cuda: job_hello count" "$(n job_hello)" 1
chk "touches cuda: lookup logged" "$(n lookup)" 1
chk "job_hello carries t_start" "$(grep "\"kind\":\"job_hello\"" /tmp/occ.jsonl | grep -c "\"t_start\":1")" 1
grep -q "lookup\[exact\]" /tmp/t.err && chk "exact lookup ran at first cuda touch" 1 1 || chk "exact lookup ran at first cuda touch" 0 1
accelerate launch --num_processes 1 cpu_only.py >/dev/null 2>&1; chk "accelerate launcher + cpu script: job_hello count" "$(n job_hello)" 1
timeout 3 python -I /gm/src/bridge/overrides_bridge.py --socket /tmp/gm.sock --overrides /tmp/ov.yaml >/dev/null 2>&1
chk "bridge: observer_hello count" "$(n observer_hello)" 1
chk "bridge: observer_bye count" "$(n observer_bye)" 1
chk "bridge: job_hello still 1" "$(n job_hello)" 1
t() { local s=$(date +%s%N); for i in 1 2 3 4 5; do "$@" >/dev/null 2>&1; done; echo $(( ($(date +%s%N)-s)/5000000 )); }
echo "interpreter start, ms per run (mean of 5): with site dir $(t python -c pass); without $( (unset PYTHONPATH; t python -c pass) ); import torch with site dir $(t python -c "import torch")"
echo; echo "$pass passed, $fail failed"; [ "$fail" = 0 ]'
