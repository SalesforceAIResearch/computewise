#!/usr/bin/env bash
# Runs INSIDE the container for probe_layer_b.sh. Separated from the launcher so
# the shell quoting is sane. Expects libmockfs preloaded, MOCK_PCI_ROOT set, the
# bundle at /opt/mock, and stage_dev_nodes.sh at /stage_dev_nodes.sh.
set -e
apt-get update >/dev/null 2>&1 && apt-get install -y kmod pciutils >/dev/null 2>&1
GPUS=4 bash /stage_dev_nodes.sh >/dev/null 2>&1
export LD_LIBRARY_PATH=/opt/mock/lib
pass=0; fail=0
chk(){ if eval "$2"; then echo "[ok  ] $1"; pass=$((pass+1)); else echo "[FAIL] $1"; fail=$((fail+1)); fi; }

chk "4 /dev/nvidia GPU nodes"      '[ $(ls /dev/nvidia[0-9]* 2>/dev/null | wc -l) -eq 4 ]'
chk "/dev/nvidiactl present"       '[ -c /dev/nvidiactl ]'
chk "/dev/nvidia-uvm present"      '[ -c /dev/nvidia-uvm ]'
chk "driver version file matches"  'grep -q 580.65.06 /proc/driver/nvidia/version'
chk "lsmod shows nvidia clean"     'lsmod 2>/dev/null | grep -q "^nvidia "'
chk "lsmod no errors on nvidia"    '! lsmod 2>&1 1>/dev/null | grep -qi nvidia'
chk "/sys/module/nvidia/version"   '[ "$(cat /sys/module/nvidia/version)" = 580.65.06 ]'
chk "per-GPU info has H100"        'grep -q "H100 80GB HBM3" /proc/driver/nvidia/gpus/0000:1a:00.0/information'

smi=$(/opt/mock/bin/nvidia-smi.real --query-gpu=uuid --format=csv,noheader -i 0 2>/dev/null | tr -d ' ')
prc=$(grep "GPU UUID" /proc/driver/nvidia/gpus/0000:1a:00.0/information | awk '{print $NF}')
chk "nvidia-smi UUID == /proc UUID" '[ "$smi" = "$prc" ]'

echo; echo "$pass passed, $fail failed"
[ $fail -eq 0 ]
