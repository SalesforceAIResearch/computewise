#!/usr/bin/env bash
# Build gm-battery:x86 on the GCP host from tests/images/battery/Dockerfile.
# The context is a staged bundle (default /opt/computewise-deploy/gm-ctx/gpu-mock, from
# container/stage_bundle.sh) with the torch patch and the install scripts
# refreshed from THIS tree, plus gpu-tasks' constraints for the battery.
# Usage (on the VM):  sudo [TAG=gm-battery:x86] bash tests/images/battery/build.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); GM=$(cd "$HERE/../../.." && pwd); EXP=$(cd "$GM/.." && pwd)
BUNDLE="${BUNDLE:-/opt/computewise-deploy/gm-ctx/gpu-mock}"
CTX="${CTX:-/srv/work/gpu-mock/battery-ctx}"
TAG="${TAG:-gm-battery:x86}"
CONS="$EXP/gpu-tasks/registry/artifacts/trl/env/0022af66577e/constraints.txt"
rm -rf "$CTX"; mkdir -p "$CTX"
cp -a "$BUNDLE" "$CTX/gpu-mock"
rm -rf "$CTX/gpu-mock/torch_patch"; mkdir -p "$CTX/gpu-mock/torch_patch"
cp "$GM"/src/torch_patch/*.py "$CTX/gpu-mock/torch_patch/"
install -m 0755 "$GM/container/install_into_image.sh" "$GM/container/stage_dev_nodes.sh" "$CTX/gpu-mock/"
cp "$CONS" "$CTX/constraints.txt"
echo "torch_patch: $(cd "$CTX/gpu-mock/torch_patch" && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
docker build -f "$HERE/Dockerfile" -t "$TAG" "$CTX"
echo "built $TAG"
