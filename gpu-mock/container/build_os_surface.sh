#!/usr/bin/env bash
# Build the layer-B OS surface: the fake /proc and /sys tree that libmockfs
# redirects to, matching a 4×H100 node with driver 580.65.06. Writes the tree to
# $OUT/mock-pci. The device nodes (/dev/nvidia*) are created at container start
# (stage_dev_nodes.sh) because mknod needs privilege the build step lacks.
#
# A real H100 node shows, and an agent may cross-check against nvidia-smi:
#   /proc/driver/nvidia/version          driver + GCC build string
#   /proc/driver/nvidia/gpus/<pci>/information   per-GPU: model, UUID, bus
#   /proc/modules (lsmod)                nvidia, nvidia_uvm, nvidia_drm modules
#   /sys/module/nvidia/version           the driver version
#   /sys/bus/pci/devices/<pci>/          vendor 0x10de, class 0x030200, etc.
#
# Usage:  OUT=$HOME/gpu-mock-bundle DRIVER=580.65.06 GPUS=4 bash build_os_surface.sh
set -euo pipefail

OUT="${OUT:?set OUT to the bundle dir}"
DRIVER="${DRIVER:-580.65.06}"
GPUS="${GPUS:-4}"
# Kernel-module arch in the driver banner: the presented node's arch, default the build host's.
KARCH="${KARCH:-$(uname -m | sed s/arm64/aarch64/)}"
ROOT="$OUT/mock-pci"
rm -rf "$ROOT"
mkdir -p "$ROOT/proc/driver/nvidia/gpus" "$ROOT/sys/module/nvidia" \
         "$ROOT/sys/bus/pci/devices" "$ROOT/sys/module/nvidia_uvm" \
         "$ROOT/proc/driver/nvidia"

# H100 SXM PCI bus ids, matching the Mokka h100 config device_defaults.
BUSES=(0000:1a:00.0 0000:1b:00.0 0000:4a:00.0 0000:4b:00.0 \
       0000:8a:00.0 0000:8b:00.0 0000:ca:00.0 0000:cb:00.0)
H100_DEVID="2330"   # H100 SXM 80GB PCI device id (0x2330), vendor 0x10de

# /proc/driver/nvidia/version — the exact format a real driver writes.
cat > "$ROOT/proc/driver/nvidia/version" <<EOF
NVRM version: NVIDIA UNIX Open Kernel Module for ${KARCH}  ${DRIVER}  Release Build  (dvs-builder@U16-I3-C04-2-2)  $(date -u '+%a %b %d %H:%M:%S UTC %Y')
GCC version:  gcc version 12.2.0 (Debian 12.2.0-14)
EOF

# /sys/module/<name>/ — the real lsmod (kmod) reads coresize, refcnt, initstate,
# holders/ and version here, not just /proc/modules. Build the full substructure
# so lsmod runs clean. Each tuple: name, coresize, refcnt, "holder1 holder2".
#   nvidia      is held by nvidia_uvm and nvidia_modeset
#   nvidia_modeset is held by nvidia_drm
_mod() {  # name coresize refcnt holders...
  local name="$1" size="$2" ref="$3"; shift 3
  local md="$ROOT/sys/module/$name"
  mkdir -p "$md/holders" "$md/sections" "$md/parameters" "$md/drivers"
  echo "$size" > "$md/coresize"
  echo "$ref"  > "$md/refcnt"
  echo "live"  > "$md/initstate"
  echo "O"     > "$md/taint"
  for h in "$@"; do ln -sf "../$h" "$md/holders/$h"; done
}
_mod nvidia         62849024 "$GPUS" nvidia_uvm nvidia_modeset
_mod nvidia_uvm     1703936  2
_mod nvidia_modeset 1507328  1 nvidia_drm
_mod nvidia_drm     122880   0
echo "${DRIVER}" > "$ROOT/sys/module/nvidia/version"
echo "${DRIVER}" > "$ROOT/sys/module/nvidia/srcversion" 2>/dev/null || true

# /proc/modules — lsmod's fallback and what `cat /proc/modules` shows.
cat > "$ROOT/proc/modules" <<EOF
nvidia_uvm 1703936 2 - Live 0x0000000000000000 (OE)
nvidia_drm 122880 0 - Live 0x0000000000000000 (OE)
nvidia_modeset 1507328 1 nvidia_drm, Live 0x0000000000000000 (OE)
nvidia 62849024 ${GPUS} nvidia_uvm,nvidia_modeset, Live 0x0000000000000000 (OE)
EOF

# per-GPU /proc/driver/nvidia/gpus/<pci>/information
for i in $(seq 0 $((GPUS-1))); do
  bus="${BUSES[$i]}"
  uuid=$(printf 'GPU-01000100-0000-0000-0000-%012d' "$i")
  d="$ROOT/proc/driver/nvidia/gpus/$bus"
  mkdir -p "$d"
  cat > "$d/information" <<EOF
Model: 		 NVIDIA H100 80GB HBM3
IRQ:   		 $((33 + i))
GPU UUID: 	 ${uuid}
Video BIOS: 	 96.00.74.00.09
Bus Type: 	 PCIe
DMA Size: 	 52 bits
DMA Mask: 	 0xfffffffffffff
Bus Location: 	 ${bus}
Device Minor: 	 ${i}
GPU Excluded:	 No
EOF
  # /sys/bus/pci/devices/<pci>/ attributes an agent may read
  s="$ROOT/sys/bus/pci/devices/$bus"
  mkdir -p "$s"
  echo "0x10de" > "$s/vendor"
  echo "0x${H100_DEVID}" > "$s/device"
  echo "0x030200" > "$s/class"      # 3D controller
  echo "0x10de" > "$s/subsystem_vendor"
  echo "0x16c1" > "$s/subsystem_device"
  echo "$((i))" > "$s/numa_node"
  echo "enabled" > "$s/enable"
done

echo "layer-B tree: $ROOT"
echo "device nodes: run container/stage_dev_nodes.sh inside the container (needs mknod)"
