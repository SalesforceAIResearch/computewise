#!/usr/bin/env bash
# Build the DPO test images on the GCP host from a staged bundle, then probe each:
#   gm-dpo:x86            pandas pinned to 2.2.2
#   gm-dpo:x86-unpinned   isolation's pip lines as written (pandas unpinned)
# Usage (on the VM):  sudo bash tests/images/dpo-smoke/build.sh      CTX=/opt/computewise-deploy/gm-ctx by default
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
CTX="${CTX:-/opt/computewise-deploy/gm-ctx}"
for spec in "pandas==2.2.2 gm-dpo:x86" "pandas gm-dpo:x86-unpinned"; do
  set -- $spec
  echo "##### build $2 (PANDAS_SPEC=$1)"
  docker buildx build ${BUILDX_BUILDER:+--builder "$BUILDX_BUILDER"} --load -f "$HERE/Dockerfile" --build-arg PANDAS_SPEC="$1" -t "$2" "$CTX" > "/tmp/build-${2//[:\/]/_}.log" 2>&1
  echo "build rc=$?"; grep -E "stack:|gpu-mock installed|nvidia-smi ->|ERROR" "/tmp/build-${2//[:\/]/_}.log" | tail -4
done
for img in gm-dpo:x86 gm-dpo:x86-unpinned; do
  echo "##### probe $img"; docker run --entrypoint '' --rm --cpus 2 --memory 4g -v "$HERE:/probe:ro" "$img" bash /probe/probe.sh
done
echo "##### done"
