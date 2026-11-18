#!/usr/bin/env bash
# Build the layer-A runtime bundle for the Docker host's OWN architecture (x86_64 on a Linux build host; arm64 for
# development only; there is no cross build):
#   runtime/lib/libnvidia-ml.so*  NVIDIA/k8s-test-infra's mock NVML (Apache-2.0)
#   runtime/lib/libcuda.so*       NVIDIA/k8s-test-infra's stub CUDA driver (Apache-2.0), with libcudart.so links
#   runtime/bin/nvidia-smi.real   the genuine nvidia-smi from NVIDIA's nvidia-utils-580 package, RPATH ../lib
#   runtime/lib/libmockfs.so*     k8s-test-infra's /proc and /sys shim (built, then left out of the box)
#   runtime/pins.env              every pin below
#   h100-<GPUS>gpu.yaml           k8s-test-infra's H100 profile trimmed to GPUS devices
# The first three come from runtime.Dockerfile beside this script, built at the pins below. container/stage_bundle.sh
# calls this script and verifies the ELF architecture of every binary it produces.
#
# Usage:  MOKKA=<k8s-test-infra checkout> OUT=<bundle dir> bash build_bundle.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

OUT="${OUT:?set OUT to a bundle dir}"
MOKKA="${MOKKA:?set MOKKA to a NVIDIA/k8s-test-infra checkout (for libmockfs and the H100 config)}"
GPUS="${GPUS:-4}"
ARCH=$(docker version --format '{{.Server.Arch}}')   # amd64 | arm64
PLATFORM="linux/$ARCH"

# The mock NVML is built from the k8s-test-infra revision the paper's bundles used; the stub libcuda from the revision
# MOKKA is checked out at, which carries shims/libcuda. nvidia-utils-580 pins come from NVIDIA's repository index.
NVML_REVISION=f7bbbf025110c63c04567cf42e357af32fa2f62d
LIBCUDA_REVISION=a82e8b9cafd62e6f242bb4df41f71ffa11aa4c83
NVIDIA_KEY_FINGERPRINT=EB693B3035CD5710E231E123A4B469963BF863CC
NVIDIA_UTILS_VERSION=580.65.06-0ubuntu1
NVIDIA_UTILS_SHA256_AMD64=3e9c39d206196e4bb32d06efaf55cbd2f13e54b163928f26bd8d38b7cbba8fcc
NVIDIA_UTILS_SHA256_ARM64=25b291a8d147e545ac3152d50e2818b523782ef07c97f381e2d0478f6d8ec63e

echo "building runtime bundle for $PLATFORM: mock NVML @${NVML_REVISION:0:8}, libcuda @${LIBCUDA_REVISION:0:8}, nvidia-utils-580=$NVIDIA_UTILS_VERSION"
rm -rf "$OUT/runtime"; mkdir -p "$OUT/runtime"
docker buildx build --platform "$PLATFORM" --file "$HERE/runtime.Dockerfile" --target export \
  --build-arg "NVML_REVISION=$NVML_REVISION" --build-arg "LIBCUDA_REVISION=$LIBCUDA_REVISION" \
  --build-arg "NVIDIA_KEY_FINGERPRINT=$NVIDIA_KEY_FINGERPRINT" --build-arg "NVIDIA_UTILS_VERSION=$NVIDIA_UTILS_VERSION" \
  --build-arg "NVIDIA_UTILS_SHA256_AMD64=$NVIDIA_UTILS_SHA256_AMD64" --build-arg "NVIDIA_UTILS_SHA256_ARM64=$NVIDIA_UTILS_SHA256_ARM64" \
  --output "type=local,dest=$OUT/runtime" "$HERE"
for f in bin/nvidia-smi.real lib/libnvidia-ml.so.1 lib/libcuda.so.1; do test -e "$OUT/runtime/$f" || { echo "bundle is missing $f" >&2; exit 1; }; done
for f in lib/libcuda.so lib/libcudart.so lib/libcudart.so.12 lib/libcudart.so.13; do test -L "$OUT/runtime/$f" || { echo "bundle is missing the link $f" >&2; exit 1; }; done
printf '%s\n' "NVML_REVISION=$NVML_REVISION" "LIBCUDA_REVISION=$LIBCUDA_REVISION" "NVIDIA_KEY_FINGERPRINT=$NVIDIA_KEY_FINGERPRINT" \
  "NVIDIA_UTILS_VERSION=$NVIDIA_UTILS_VERSION" "NVIDIA_UTILS_SHA256_AMD64=$NVIDIA_UTILS_SHA256_AMD64" \
  "NVIDIA_UTILS_SHA256_ARM64=$NVIDIA_UTILS_SHA256_ARM64" > "$OUT/runtime/pins.env"

# libmockfs (the LD_PRELOAD shim for /proc/driver/nvidia, /sys/bus/pci,
# /proc/modules, /sys/module): build it from the k8s-test-infra checkout in a gcc
# container and add it to the bundle lib.
echo "building libmockfs for $PLATFORM"
LM="$OUT/libmockfs-src"; rm -rf "$LM"; mkdir -p "$LM"
cp -a "$MOKKA/shims/libmockfs/src" "$MOKKA/shims/libmockfs/Makefile" "$LM/"
docker run --rm --cpus 2 --memory 2g --platform "$PLATFORM" -v "$LM:/src" -w /src gcc:13-bookworm \
  bash -c 'make libmockfs.so.1.0.0 >/dev/null 2>&1 || make 2>&1 | grep -v "No rule to make target .testbin" ; ls libmockfs.so.1.0.0'
cp "$LM/libmockfs.so.1.0.0" "$OUT/runtime/lib/"
ln -sf libmockfs.so.1.0.0 "$OUT/runtime/lib/libmockfs.so.1"
ln -sf libmockfs.so.1 "$OUT/runtime/lib/libmockfs.so"
rm -rf "$LM"

echo "rendering ${GPUS}-GPU H100 config"
python3 -I - "$MOKKA/pkg/gpu/mocknvml/configs/mock-nvml-config-h100.yaml" \
             "$OUT/h100-${GPUS}gpu.yaml" "$GPUS" <<'PY'
import re, sys
src_path, out_path, gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(src_path).read().splitlines(keepends=True)
out, in_devices, keep = [], False, True
for ln in lines:
    if re.match(r'^devices:\s*$', ln):
        in_devices = True; out.append(ln); continue
    if in_devices:
        m = re.match(r'^  - index:\s*(\d+)', ln)
        if m:
            keep = int(m.group(1)) < gpus
        if keep:
            out.append(ln)
    else:
        out.append(ln)
open(out_path, "w").write("".join(out))
print("wrote", out_path, "with", gpus, "GPUs")
PY

echo
echo "bundle:  $OUT/runtime ($PLATFORM)"
echo "config:  $OUT/h100-${GPUS}gpu.yaml"
