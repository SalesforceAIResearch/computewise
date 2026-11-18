#!/usr/bin/env bash
# Create the /dev/nvidia* character device nodes a real GPU node has. Runs INSIDE
# the container at start; needs mknod (CAP_MKNOD, which docker --cap-add grants,
# or run as root which has it by default in the default container). The nodes are
# never opened against a real driver — they exist so `ls /dev/nvidia*` and code
# that stats them (not opens them) sees what a real node shows. Any process that
# actually open()s them for ioctl would fail, which is also what happens on a
# node whose driver is not loaded; real CUDA goes through libcuda, which we mock.
#
# Majors/minors match a real NVIDIA driver:
#   /dev/nvidia0..N-1   major 195, minor 0..N-1   (the GPUs)
#   /dev/nvidiactl      major 195, minor 255       (control)
#   /dev/nvidia-uvm     major 510, minor 0         (unified memory; major varies)
#   /dev/nvidia-uvm-tools  major 510, minor 1
#   /dev/nvidia-modeset major 195, minor 254
set -euo pipefail
GPUS="${GPUS:-4}"
UVM_MAJOR="${UVM_MAJOR:-510}"

for i in $(seq 0 $((GPUS-1))); do
  [ -e "/dev/nvidia$i" ] || mknod -m 666 "/dev/nvidia$i" c 195 "$i"
done
[ -e /dev/nvidiactl ]        || mknod -m 666 /dev/nvidiactl        c 195 255
[ -e /dev/nvidia-modeset ]   || mknod -m 666 /dev/nvidia-modeset   c 195 254
[ -e /dev/nvidia-uvm ]       || mknod -m 666 /dev/nvidia-uvm       c "$UVM_MAJOR" 0
[ -e /dev/nvidia-uvm-tools ] || mknod -m 666 /dev/nvidia-uvm-tools c "$UVM_MAJOR" 1

echo "staged /dev/nvidia* nodes:"
ls -l /dev/nvidia* 2>/dev/null
