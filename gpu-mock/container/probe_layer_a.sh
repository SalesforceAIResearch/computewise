#!/usr/bin/env bash
# Layer-A fidelity probe: the real NVIDIA nvidia-smi running against Mokka's
# mock libnvidia-ml, in a clean arm64 Debian container with NO GPU. Proves
# enumeration, live overrides (the bridge's channel) and failure injection.
#
# Needs: Colima arm64 running; the runtime bundle built (container/build_bundle.sh)
# into a directory under $HOME (Colima shares $HOME, not /tmp).
#
# Usage:  BUNDLE=/path/to/runtime CONFIG=/path/to/h100-4gpu.yaml \
#         bash probe_layer_a.sh
set -euo pipefail

BUNDLE="${BUNDLE:?set BUNDLE to the runtime bundle dir (bin/ lib/)}"
CONFIG="${CONFIG:?set CONFIG to the 4-GPU H100 yaml}"
IMAGE="${IMAGE:-debian:bookworm-slim}"
SCRATCH="$(mktemp -d "${BUNDLE%/*}/probe-XXXX")"
trap 'rm -rf "$SCRATCH"' EXIT

smi() {  # run the real nvidia-smi against the mock, with the given overrides file
  local overrides="$1"; shift
  local mounts=(-v "$BUNDLE/lib:/opt/mock/lib:ro" -v "$BUNDLE/bin:/opt/mock/bin:ro"
                -v "$CONFIG:/etc/mock/config.yaml:ro")
  local env=(-e LD_LIBRARY_PATH=/opt/mock/lib -e MOCK_NVML_CONFIG=/etc/mock/config.yaml)
  if [[ -n "$overrides" ]]; then
    mounts+=(-v "$overrides:/etc/mock/overrides.yaml:ro")
    env+=(-e MOCK_NVML_OVERRIDES=/etc/mock/overrides.yaml)
  fi
  docker run --rm --platform linux/arm64 "${mounts[@]}" "${env[@]}" "$IMAGE" \
    /opt/mock/bin/nvidia-smi.real "$@"
}

pass=0; fail=0
check() { if eval "$2"; then echo "[ok  ] $1"; pass=$((pass+1));
          else echo "[FAIL] $1"; fail=$((fail+1)); fi; }

echo "### enumeration"
LIST="$(smi '' -L)"; echo "$LIST"
check "four H100 enumerated" '[[ $(grep -c "NVIDIA H100 80GB HBM3" <<<"$LIST") -eq 4 ]]'

echo "### live overrides (the bridge channel)"
cat > "$SCRATCH/ov.yaml" <<'JSON'
{"devices":{"0":{"utilization":{"gpu":87,"memory":61},"memory":{"used_bytes":42949672960,"free_bytes":41949672960},"processes":[{"pid":4321,"type":"C","name":"python","used_memory_mib":40960,"sm_util":87,"mem_util":61}]}}}
JSON
CSV="$(smi "$SCRATCH/ov.yaml" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"
echo "$CSV"
check "GPU0 shows 40960 MiB used"  '[[ "$(sed -n 1p <<<"$CSV")" == *"40960"* ]]'
check "GPU0 shows 87% util"        '[[ "$(sed -n 1p <<<"$CSV")" == *" 87"* ]]'
PROC="$(smi "$SCRATCH/ov.yaml")"
check "process row shows python"   '[[ "$PROC" == *"python"* && "$PROC" == *"4321"* ]]'

echo "### failure injection (Xid 79 / fallen off bus)"
cat > "$SCRATCH/fail.yaml" <<'JSON'
{"devices":{"2":{"failure":{"mode":"fallen_off_bus","xid":{"code":79}}}}}
JSON
TEMP="$(smi "$SCRATCH/fail.yaml" --query-gpu=index,temperature.gpu --format=csv,noheader)"
echo "$TEMP"
check "GPU2 reports lost"           '[[ "$(sed -n 3p <<<"$TEMP")" == *"lost"* ]]'
check "GPU0 still healthy"          '[[ "$(sed -n 1p <<<"$TEMP")" == *"34"* ]]'

echo
echo "$pass passed, $fail failed"
[[ $fail -eq 0 ]]
