#!/usr/bin/env bash
# Layer-B fidelity probe: the OS surface a suspicious agent cross-checks against
# nvidia-smi. See _layer_b_inside.sh for the checks.
# Usage: BUNDLE=.../runtime MOCKPCI=.../mock-pci CONFIG=.../h100-4gpu.yaml \
#        STAGE_DEV=container/stage_dev_nodes.sh INSIDE=container/_layer_b_inside.sh \
#        bash probe_layer_b.sh
set -euo pipefail
BUNDLE="${BUNDLE:?runtime bundle dir}"; MOCKPCI="${MOCKPCI:?fake tree}"
CONFIG="${CONFIG:?h100 config}"; STAGE_DEV="${STAGE_DEV:?stage_dev_nodes.sh}"
INSIDE="${INSIDE:?_layer_b_inside.sh}"; IMAGE="${IMAGE:-debian:bookworm-slim}"
docker run --rm --platform linux/arm64 --cap-add MKNOD \
  -v "$BUNDLE/lib:/opt/mock/lib:ro" -v "$BUNDLE/bin:/opt/mock/bin:ro" \
  -v "$CONFIG:/etc/mock/config.yaml:ro" -v "$MOCKPCI:/mock-pci:ro" \
  -v "$STAGE_DEV:/stage_dev_nodes.sh:ro" -v "$INSIDE:/inside.sh:ro" \
  -e LD_LIBRARY_PATH=/opt/mock/lib -e MOCK_NVML_CONFIG=/etc/mock/config.yaml \
  -e MOCK_PCI_ROOT=/mock-pci -e LD_PRELOAD=/opt/mock/lib/libmockfs.so.1 \
  "$IMAGE" bash /inside.sh
