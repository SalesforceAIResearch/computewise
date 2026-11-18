#!/usr/bin/env bash
# The k2-latepole-8gpu box and its gpu-daemon sidecar, on the E2, as root, under systemd-run. Usage, with the deployed
# tree's isolation/image/latepole/ beside it:
#   build_latepole.sh <gpu-mock tree> <tag suffix>        e.g. /srv/work/isolation/latepole/gm-r89 r89
# Pins are build.sh's (base image, Debian snapshot, torch index, Claude Code, Codex, sidecar base) and
# build_mlskv_layer.sh's transformers commit. Environment a's list is ml-node:py311-torch2110-mlskv's pip freeze, and the
# build stops if env-a.txt is not that image's freeze (less torch, pip and transformers). Writes progress, logs and a
# manifest under /srv/work/isolation/latepole/build-<stamp>/; the images are ml-node:k2-latepole-<suffix> and
# gpu-daemon:k2-latepole-<suffix>, new tags only.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); IMG=$(cd "$HERE/.." && pwd); EXP=$(cd "$IMG/.." && cd .. && pwd)
GM=${1:?gpu-mock tree}; SUF=${2:?tag suffix}
eval "$(grep -E '^(SIDECAR_BASE_IMAGE|DEBIAN_SNAPSHOT|TORCH_CPU_INDEX|NVIDIA_DRIVER|CLAUDE_VERSION|CODEX_VERSION|CODEX_X64_SHA256|CLAUDE_X64_SHA512)=' "$IMG/build.sh")"
BASE_IMAGE=python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317
TORCH_VERSION=2.11.0+cpu
TRANSFORMERS_COMMIT=$(sed -n 's/^TRANSFORMERS_COMMIT=\([0-9a-f]*\).*/\1/p' "$IMG/mlskv/build_mlskv_layer.sh")
[ ${#TRANSFORMERS_COMMIT} = 40 ] || { echo "no transformers commit in build_mlskv_layer.sh" >&2; exit 2; }
GPUS=8; LABEL=k2-latepole-8gpu; IMAGE=ml-node:k2-latepole-$SUF; SIDECAR=gpu-daemon:k2-latepole-$SUF
S=$(date -u +%Y%m%dT%H%M%SZ); W=/srv/work/isolation/latepole/build-$S; C=$W/ctx; mkdir -p $C $W/src
P=$W/progress; log() { echo "$* $(date -u +%FT%TZ)" >> $P; }
log "started image=$IMAGE"
[ -d "$GM/container" ] && [ -d "$GM/src/torch_patch" ] || { log "$GM is not a gpu-mock tree"; exit 1; }
# environment a must be the KV image's own set
KVF=$W/mlskv-freeze.txt; CID=$(docker create --entrypoint true ml-node:py311-torch2110-mlskv)
docker cp "$CID:/usr/local/share/image-manifest/pip-freeze.txt" $KVF >/dev/null; docker rm "$CID" >/dev/null
diff <(grep -v -E '^(torch==|transformers @|pip==)' $KVF) "$HERE/env-a.txt" > $W/env-a.diff \
  || { log "env-a.txt is not ml-node:py311-torch2110-mlskv's freeze: $(tr '\n' ' ' < $W/env-a.diff)"; exit 1; }
grep -qx "transformers @ git+https://github.com/huggingface/transformers@$TRANSFORMERS_COMMIT" $KVF \
  || { log "the KV image's transformers is not $TRANSFORMERS_COMMIT"; exit 1; }
PATCHHASH=$(cd "$GM/src/torch_patch" && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -c1-64)
INSTALLHASH=$(sha256sum "$GM/container/install_into_image.sh" | cut -c1-64)
ENVA=$(sha256sum "$HERE/env-a.txt" | cut -c1-64); ENVB=$(sha256sum "$HERE/env-b.txt" | cut -c1-64)
log "tree patch=${PATCHHASH:0:16} install=${INSTALLHASH:0:16} env-a=${ENVA:0:16} env-b=${ENVB:0:16} transformers=${TRANSFORMERS_COMMIT:0:12}"
NCPU=$(nproc); BUILDER=isolation-latepole-$SUF
docker buildx rm $BUILDER >/dev/null 2>&1 || true
docker buildx create --name $BUILDER --driver docker-container --driver-opt network=host \
  --driver-opt "cpuset-cpus=$((NCPU - 16))-$((NCPU - 1))" --driver-opt memory=32g --driver-opt memory-swap=32g >/dev/null
trap 'docker buildx rm $BUILDER >/dev/null 2>&1 || true' EXIT
verify_run() { local n=$1; shift; docker run --rm --name "isolation-latepole-verify-$n-$$" --cpus 2 --memory 8g --memory-swap 8g "$@"; }

# the sidecar: gpu-mock's daemon, bridge and convert from this tree, on build.sh's sidecar base
mkdir -p $W/sidecar/src
for d in daemon bridge convert; do mkdir -p $W/sidecar/src/$d; cp "$GM/src/$d/"*.py $W/sidecar/src/$d/; done
SIDECAR_SRC_HASH=$(cd $W/sidecar/src && find . -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
# gpu-mock's INDEX hashes the same files without the ./ prefix (daemon/... not ./daemon/...); recorded beside build.sh's
SIDECAR_INDEX_HASH=$(cd $W/sidecar/src && find daemon bridge convert -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
DOCKER_BUILDKIT=1 docker build -f "$IMG/Dockerfile.sidecar" -t "$SIDECAR" --build-arg "BASE_IMAGE=$SIDECAR_BASE_IMAGE" \
  --build-arg "SRC_SHA256=$SIDECAR_SRC_HASH" --label p006.role=gpu-daemon-image $W/sidecar > $W/sidecar.log 2>&1
SIDECAR_ID=$(docker image inspect -f '{{.Id}}' "$SIDECAR")
verify_run sidecar --network none --entrypoint python3 "$SIDECAR" -I /opt/gpu-daemon/src/daemon/gpu_daemon.py --help >/dev/null
log "sidecar exit=0 id=${SIDECAR_ID:7:12} src=${SIDECAR_SRC_HASH:0:16} index=${SIDECAR_INDEX_HASH:0:16}"

# the context: gpu-mock's 8-GPU bundle, the harnesses, the scheduler, box-init and the two environment lists
UPSTREAM=$W/src/gm OUT=$C/gpu-mock GPUS=$GPUS DRIVER=$NVIDIA_DRIVER BUILDX_BUILDER=$BUILDER bash "$GM/container/stage_bundle.sh" > $W/stage.log 2>&1
[ -f $C/gpu-mock/h100-${GPUS}gpu.yaml ] || { log "the bundle has no h100-${GPUS}gpu.yaml"; exit 1; }
log "stage exit=0"
GMHASH=$(sha256sum "$C/gpu-mock/BUNDLE-MANIFEST.txt" | cut -d' ' -f1)
cp "$IMG/box-init" "$IMG/bash-oom-reset" "$HERE/env-a.txt" "$HERE/env-b.txt" $C/
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
log "context: scheduler=${SCHED_SHA256:0:16}"
BUILD_JSON=$(python3 -I -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:]), sort_keys=True))' \
  "base_image=$BASE_IMAGE" "debian_snapshot=$DEBIAN_SNAPSHOT" "torch=$TORCH_VERSION" "torch_index=$TORCH_CPU_INDEX" \
  "task_env=$LABEL" "gpus=$GPUS" "env_a=/opt/envs/a" "env_a_list_sha256=$ENVA" "env_a_from=ml-node:py311-torch2110-mlskv pip freeze" \
  "env_b=/opt/envs/b" "env_b_list_sha256=$ENVB" "env_b_from=gpu-tasks/survey/k2-latepole-input-check-2026-11-02/environment-b.freeze.txt" \
  "transformers_commit_env_a=$TRANSFORMERS_COMMIT" "claude_code=$CLAUDE_VERSION" "codex=$CODEX_VERSION" \
  "codex_tarball_sha256=$CODEX_X64_SHA256" "scheduler_sha256=$SCHED_SHA256" "gpu_mock_bundle_manifest_sha256=$GMHASH" \
  "torch_patch_sha256=$PATCHHASH" "gpu_mock_install_sha256=$INSTALLHASH" \
  "sidecar_image=$SIDECAR" "sidecar_image_id=$SIDECAR_ID" "sidecar_src_sha256=$SIDECAR_SRC_HASH" "sidecar_src_index_sha256=$SIDECAR_INDEX_HASH")
DOCKER_BUILDKIT=1 docker build --network host -f "$HERE/Dockerfile" -t "$IMAGE" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "DEBIAN_SNAPSHOT=$DEBIAN_SNAPSHOT" \
  --build-arg "TORCH_CPU_INDEX=$TORCH_CPU_INDEX" --build-arg "TORCH_VERSION=$TORCH_VERSION" \
  --build-arg "TRANSFORMERS_COMMIT=$TRANSFORMERS_COMMIT" --build-arg "BUILD_JSON=$BUILD_JSON" --build-arg "BOX_GPUS=$GPUS" \
  --label p006.role=box-image --label "p006.task-env=$LABEL" --label "p006.gpus=$GPUS" \
  --label "p006.sidecar-image=$SIDECAR" --label "p006.sidecar-image-id=$SIDECAR_ID" --label "p006.codex=$CODEX_VERSION" \
  --label "p006.scheduler=${SCHED_SHA256:0:16}" $C > $W/build.log 2>&1
ID=$(docker image inspect -f '{{.Id}}' "$IMAGE")
log "box exit=0 id=${ID:7:12}"

# verify (the installed torch is checked by its distribution version: the patch presents torch.__version__ as a CUDA
# build, 2.11.0+cu130, found on the first build 2026-11-03 05:53Z): 8 GPUs, harnesses, each environment's own sitecustomize, stack and device count, the base interpreter clean
verify_run smi --entrypoint sh "$IMAGE" -c 'env $(cat /etc/box-gpus) stage-dev-nodes >/dev/null && nvidia-smi -L' > $W/verify-smi.txt
[ "$(grep -c '^GPU ' $W/verify-smi.txt)" = $GPUS ] || { log "nvidia-smi -L did not list $GPUS"; exit 1; }
verify_run codex --network none --entrypoint codex "$IMAGE" --version | grep -qx "codex-cli $CODEX_VERSION"
cat > $W/verify_env.py <<'VPY'
import importlib, importlib.metadata as md, sys
import sitecustomize
print("SITE", sitecustomize.__file__)
for m in sys.argv[1].split():
    importlib.import_module(m)
import torch, cuda_hooks
print("TRANSFORMERS", md.version("transformers"))
print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "COUNT", torch.cuda.device_count())
print("TORCHDIST", md.version("torch"))
VPY
for e in a b; do
  case $e in
    a) mods="transformers torch datasets accelerate huggingface_hub tokenizers safetensors sentencepiece google.protobuf hqq optimum.quanto nvcompat"; tv=5.3.0;;
    b) mods="transformers torch datasets accelerate huggingface_hub tokenizers safetensors sentencepiece google.protobuf scipy pandas nvcompat"; tv=4.46.3;;
  esac
  verify_run py$e --network none --user 1000:1000 -v $W/verify_env.py:/tmp/verify_env.py:ro --entrypoint /opt/envs/$e/bin/python \
    "$IMAGE" /tmp/verify_env.py "$mods" > $W/verify-py$e.txt 2>&1 || true
  grep -qx "SITE /opt/envs/$e/lib/python3.11/site-packages/sitecustomize.py" $W/verify-py$e.txt \
    && grep -qx "TRANSFORMERS $tv" $W/verify-py$e.txt && grep -qx "TORCHDIST 2.11.0+cpu" $W/verify-py$e.txt && grep -q "^TORCH .* COUNT $GPUS$" $W/verify-py$e.txt \
    || { log "verify env $e failed: $(tr '\n' ' ' < $W/verify-py$e.txt | cut -c1-600)"; exit 1; }
  log "verify env $e: $(grep -E '^(TRANSFORMERS|TORCH)' $W/verify-py$e.txt | tr '\n' ' ')"
done
verify_run base --network none --entrypoint sh "$IMAGE" -c 'test -x /usr/local/bin/python3 && /usr/local/bin/python3 -c "
import importlib.util as u, site
print(\"BASE\", [m for m in (\"torch\", \"transformers\", \"sitecustomize\") if u.find_spec(m)])"' > $W/verify-base.txt 2>&1
grep -qx "BASE \[\]" $W/verify-base.txt || { log "the base interpreter is not clean: $(cat $W/verify-base.txt)"; exit 1; }
verify_run pip --network none --entrypoint sh "$IMAGE" -c 'for p in /opt/envs/a/bin/pip /opt/envs/b/bin/pip pip; do $p list 2>/dev/null; done | grep -ic "fakegpu\|gpu-mock\|gpu_mock"' > $W/verify-pip.txt || true
log "verify: base interpreter clean, pip-mock-names=$(cat $W/verify-pip.txt)"
CID=$(docker create --entrypoint true "$IMAGE"); mkdir -p $W/manifest
docker cp "$CID:/usr/local/share/image-manifest/." $W/manifest/ >/dev/null; docker rm "$CID" >/dev/null
docker image inspect "$IMAGE" > $W/image-inspect.json
log "done image=${ID:7:12} sidecar=${SIDECAR_ID:7:12}"
