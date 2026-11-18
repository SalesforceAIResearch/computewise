#!/usr/bin/env bash
# The sol-tabular-ttc-r8 box and its gpu-daemon sidecar, on a machine holding Sol's source image and OpenML snapshot
# (c3-15, or the E2 after a transfer: SNAP and SC then name the copies), as root, under systemd-run. Usage, with the
# deployed tree's isolation/image/ttc/ beside it:
#   [SNAP=<openml-cache dir> SC=<single_cell.py>] build_ttc.sh <gpu-mock tree> <tag suffix>   e.g. /srv/work/isolation/ttc/gm-r89 r89
# Pins are build.sh's (Claude Code, Codex, Debian snapshot, sidecar base); the base is Sol's image by id, the OpenML
# snapshot and single_cell.py by sha256, and the build stops if any of them differs. Writes progress, logs and a manifest
# under /srv/work/isolation/ttc/build-<stamp>/; the images are ml-node:sol-tabular-ttc-<suffix> and
# gpu-daemon:sol-tabular-ttc-<suffix>, new tags only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); IMG=$(cd "$HERE/.." && pwd); EXP=$(cd "$IMG/.." && cd .. && pwd)
GM=${1:?gpu-mock tree}; SUF=${2:?tag suffix}
BASE=sol-ttc:probe-torch291-20261102
BASE_ID=sha256:493603b6fe4c88afa9c0bec682ad34a4a139f1378584049e464cf69638d762bd
SNAP=${SNAP:-/srv/work/sol-independent/ttc-probe/openml-cache}
SC=${SC:-/srv/work/sol-independent/ttc-probe/ttc-reg-cache805f/artifacts/sol-tabular/single_cell.py}
SC_SHA=805f54de3901f299316c314fc97244bd0b1511969798dfd57311809be4424a07
eval "$(grep -E '^(SIDECAR_BASE_IMAGE|DEBIAN_SNAPSHOT|NVIDIA_DRIVER|CLAUDE_VERSION|CODEX_VERSION|CODEX_X64_SHA256|CLAUDE_X64_SHA512)=' "$IMG/build.sh")"
LABEL=sol-tabular-ttc-r8; IMAGE=ml-node:sol-tabular-ttc-$SUF; SIDECAR=gpu-daemon:sol-tabular-ttc-$SUF
S=$(date -u +%Y%m%dT%H%M%SZ); W=/srv/work/isolation/ttc/build-$S; C=$W/ctx; mkdir -p $C $W/src
P=$W/progress; log() { echo "$* $(date -u +%FT%TZ)" >> $P; }
log "started base=$BASE image=$IMAGE"
[ "$(docker image inspect -f '{{.Id}}' $BASE)" = "$BASE_ID" ] || { log "base is not 493603b6; stopping"; exit 1; }
[ "$(sha256sum $SC | cut -c1-64)" = "$SC_SHA" ] || { log "single_cell.py is not 805f54de; stopping"; exit 1; }
[ -d "$GM/container" ] && [ -d "$GM/src/torch_patch" ] || { log "$GM is not a gpu-mock tree"; exit 1; }
PATCHHASH=$(cd "$GM/src/torch_patch" && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -c1-64)
INSTALLHASH=$(sha256sum "$GM/container/install_into_image.sh" | cut -c1-64)
log "tree patch=${PATCHHASH:0:16} install=${INSTALLHASH:0:16}"
NCPU=$(nproc); BUILDER=isolation-ttc-$SUF
docker buildx rm $BUILDER >/dev/null 2>&1 || true
docker buildx create --name $BUILDER --driver docker-container --driver-opt network=host \
  --driver-opt "cpuset-cpus=$((NCPU - 16))-$((NCPU - 1))" --driver-opt memory=32g --driver-opt memory-swap=32g >/dev/null
trap 'docker buildx rm $BUILDER >/dev/null 2>&1 || true' EXIT
verify_run() { local n=$1; shift; docker run --rm --name "isolation-ttc-verify-$n-$$" --cpus 2 --memory 4g --memory-swap 4g "$@"; }

# the sidecar: gpu-mock's daemon, bridge and convert from this tree, on build.sh's sidecar base
mkdir -p $W/sidecar/src
for d in daemon bridge convert; do mkdir -p $W/sidecar/src/$d; cp "$GM/src/$d/"*.py $W/sidecar/src/$d/; done
SIDECAR_SRC_HASH=$(cd $W/sidecar/src && find . -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
# gpu-mock's INDEX hashes the same files without the ./ prefix (daemon/... not ./daemon/...); recorded beside it
SIDECAR_INDEX_HASH=$(cd $W/sidecar/src && find daemon bridge convert -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
DOCKER_BUILDKIT=1 docker build -f "$IMG/Dockerfile.sidecar" -t "$SIDECAR" --build-arg "BASE_IMAGE=$SIDECAR_BASE_IMAGE" \
  --build-arg "SRC_SHA256=$SIDECAR_SRC_HASH" --label p006.role=gpu-daemon-image $W/sidecar > $W/sidecar.log 2>&1
SIDECAR_ID=$(docker image inspect -f '{{.Id}}' "$SIDECAR")
verify_run sidecar --network none --entrypoint python3 "$SIDECAR" -I /opt/gpu-daemon/src/daemon/gpu_daemon.py --help >/dev/null
log "sidecar exit=0 id=${SIDECAR_ID:7:12} src=${SIDECAR_SRC_HASH:0:16} index=${SIDECAR_INDEX_HASH:0:16}"

# the context: gpu-mock's bundle, the harnesses, the scheduler, box-init, the snapshot and the adapter
UPSTREAM=$W/src/gm OUT=$C/gpu-mock GPUS=4 DRIVER=$NVIDIA_DRIVER BUILDX_BUILDER=$BUILDER bash "$GM/container/stage_bundle.sh" > $W/stage.log 2>&1
log "stage exit=0"
GMHASH=$(sha256sum "$C/gpu-mock/BUNDLE-MANIFEST.txt" | cut -d' ' -f1)
cp "$IMG/box-init" "$IMG/bash-oom-reset" $C/
mkdir -p $C/scheduler && cp -R "$EXP/scheduler/bin" "$EXP/scheduler/slurm" "$EXP/scheduler/install.sh" $C/scheduler/ \
  && find $C/scheduler -name __pycache__ -prune -exec rm -rf {} +
SCHED_SHA256=$(cd $C/scheduler && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
curl -fsSL "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-$CLAUDE_VERSION.tgz" -o $W/src/claude.tgz
python3 -I -c 'import base64, hashlib, sys
d = base64.b64encode(hashlib.sha512(open(sys.argv[1], "rb").read()).digest()).decode()
sys.exit(0 if d == sys.argv[2] else "Claude Code tarball: sha512 does not match the pin")' $W/src/claude.tgz "$CLAUDE_X64_SHA512"
mkdir $W/src/claude && tar xzf $W/src/claude.tgz -C $W/src/claude && cp $W/src/claude/package/claude $C/claude
curl -fsSL "https://registry.npmjs.org/@openai/codex/-/codex-$CODEX_VERSION-linux-x64.tgz" -o $W/src/codex.tgz
echo "$CODEX_X64_SHA256  $W/src/codex.tgz" | sha256sum -c --quiet
mkdir -p $C/codex && tar xzf $W/src/codex.tgz -C $C/codex --strip-components=3 package/vendor/x86_64-unknown-linux-musl
[ -x $C/codex/bin/codex ]
mkdir -p $C/openml && cp -a $SNAP/org $C/openml/
OPENML_SHA=$(cd $C/openml && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
cp $SC $C/single_cell.py
log "context: scheduler=${SCHED_SHA256:0:16} openml=${OPENML_SHA:0:16} ($(find $C/openml -type f | wc -l) files) single_cell=${SC_SHA:0:16}"
BUILD_JSON=$(python3 -I -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:]), sort_keys=True))' \
  "base_image=$BASE" "base_image_id=$BASE_ID" "debian_snapshot=$DEBIAN_SNAPSHOT" "task_env=$LABEL" "claude_code=$CLAUDE_VERSION" \
  "codex=$CODEX_VERSION" "codex_tarball_sha256=$CODEX_X64_SHA256" "scheduler_sha256=$SCHED_SHA256" \
  "gpu_mock_bundle_manifest_sha256=$GMHASH" "torch_patch_sha256=$PATCHHASH" "gpu_mock_install_sha256=$INSTALLHASH" \
  "sidecar_image=$SIDECAR" "sidecar_image_id=$SIDECAR_ID" "sidecar_src_sha256=$SIDECAR_SRC_HASH" "sidecar_src_index_sha256=$SIDECAR_INDEX_HASH" \
  "openml_cache_sha256=$OPENML_SHA" "single_cell_sha256=$SC_SHA")
DOCKER_BUILDKIT=1 docker build --network host -f "$HERE/Dockerfile" -t "$IMAGE" \
  --build-arg "BASE_IMAGE=$BASE" --build-arg "DEBIAN_SNAPSHOT=$DEBIAN_SNAPSHOT" --build-arg "BUILD_JSON=$BUILD_JSON" \
  --label p006.role=box-image --label "p006.task-env=$LABEL" --label p006.gpus=4 \
  --label "p006.sidecar-image=$SIDECAR" --label "p006.sidecar-image-id=$SIDECAR_ID" --label "p006.codex=$CODEX_VERSION" \
  --label "p006.scheduler=${SCHED_SHA256:0:16}" $C > $W/build.log 2>&1
ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
log "box exit=0 id=${ID:7:12}"

# verify: GPUs, harnesses, torch kept, the snapshot read-only to the agent, nothing of the mock named in pip
verify_run smi --entrypoint sh "$IMAGE" -c 'stage-dev-nodes >/dev/null && nvidia-smi -L' > $W/verify-smi.txt
[ "$(grep -c '^GPU ' $W/verify-smi.txt)" = 4 ] || { log "nvidia-smi -L did not list 4"; exit 1; }
verify_run codex --network none --entrypoint codex "$IMAGE" --version | grep -qx "codex-cli $CODEX_VERSION"
verify_run py --network none --user 1000:1000 --entrypoint python3 "$IMAGE" -c '
import os, torch
print("TORCH", torch.__version__)
print("CUDA", torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))
c = os.environ["OPENML_CACHE_DIR"]; w = []
for r, ds, fs in os.walk(c):
    w += [p for p in [r] + [os.path.join(r, f) for f in fs] if os.access(p, os.W_OK)]
print("CACHE", c, "files", sum(len(f) for _, _, f in os.walk(c)), "writable_by_agent", len(w))
' > $W/verify-py.txt 2>&1
grep -q "^TORCH 2.9.1" $W/verify-py.txt && grep -q "writable_by_agent 0$" $W/verify-py.txt || { log "verify-py failed: $(tr '\n' ' ' < $W/verify-py.txt)"; exit 1; }
verify_run pip --network none --entrypoint sh "$IMAGE" -c 'pip list 2>/dev/null | grep -ic "fakegpu\|gpu-mock\|gpu_mock"' > $W/verify-pip.txt || true
log "verify: $(tr '\n' ' ' < $W/verify-py.txt) pip-mock-names=$(cat $W/verify-pip.txt)"
docker image inspect "$IMAGE" > $W/image-inspect.json
log "done image=${ID:7:12} sidecar=${SIDECAR_ID:7:12}"
