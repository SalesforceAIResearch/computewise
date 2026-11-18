#!/usr/bin/env bash
# Builds the agent box image, and the gpu-daemon image that runs gpu-mock's daemon beside each box, from
# pinned inputs, on the Docker host that runs the boxes: the GCP VM
# build-host, never the controller. Run as root from the deployed tree:
#
#   sudo bash /opt/computewise/isolation/image/build.sh [--task trl-dpo-loss-variants-4gpu] \
#        [--image ml-node:py311-torch240] [--sidecar-image gpu-daemon:py311-torch240] \
#        [--compare <an earlier manifest>] [--bundle <dir>] [--no-cache] [--gpu-mock <dir>] [--work <dir>] [--gpus <n>]
#   e.g. the battery: --task trl-regression-battery --image ml-node:py310-torch212 (Python 3.10.13, torch 2.1.2)
#
# --gpu-mock takes gpu-mock's tree (its container/ and src/) from <dir> instead of the deployed experiments/gpu-mock,
# so two revisions can build at once, each from its own unpacked archive (analysis, 2026-10-23: the tell revision and
# the width revision both want the E2 the same evening). Two builds at once also need their own --work and their own
# --image and --sidecar-image tags; the builder container is named after the work tree, so they never share one.
#
# --bundle takes a gpu-mock bundle that stage_bundle.sh already produced on this host, instead of
# staging one. The bundle carries gpu-mock's torch patch as of its staging, while the gpu-daemon image
# is always built from the tree, so build.sh warns when the two differ.
# --gpus <n> (default 4) is the GPU count the box presents: gpu-mock's NVML profile, its OS surface and /dev/nvidia0..n-1.
# A count other than 4 makes its own image, labelled p006.gpus=<n> and with p006.task-env=<label>-<n>gpu, so a task asks
# for it by label (box.gpus: n and environment.image_label "<label>-<n>gpu") and it never answers a 4-GPU task's lookup
#. Tag it apart from the 4-GPU set too.
# --no-cache builds both images without Docker's layer cache: the from-scratch build that a
# reproducibility check compares.
#
# Every input is pinned below, or comes from the task's own `environment` in the registry (Python
# version, packages, constraints). Changing a pin makes a new image. Each build writes a manifest to
# /srv/build/manifests/: image id, pins, the Python, Debian and Claude Code versions in the image,
# and a hash of the gpu-mock files that went in. --compare exits 1 when this build's versions differ
# from an earlier manifest's, which is how a rebuild is checked to reproduce.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
EXP=$(cd "$HERE/../.." && pwd)
TASK=trl-dpo-loss-variants-4gpu
IMAGE=ml-node:py311-torch240
COMPARE=""
BUNDLE=""
NOCACHE=""
SIDECAR_IMAGE=""
WORK=/srv/build/box
MANIFESTS=/srv/build/manifests
while [ $# -gt 0 ]; do
  case "$1" in
    --task) TASK=$2; shift 2;;
    --image) IMAGE=$2; shift 2;;
    --compare) COMPARE=$2; shift 2;;
    --bundle) BUNDLE=$2; shift 2;;
    --sidecar-image) SIDECAR_IMAGE=$2; shift 2;;
    --no-cache) NOCACHE=--no-cache; shift;;
    --work) WORK=$2; shift 2;;
    --gpu-mock) GM_TREE=$2; shift 2;;
    --gpus) GPUS_ARG=$2; shift 2;;
    *) echo "unknown argument $1" >&2; exit 2;;
  esac
done
GM_TREE=$(cd "${GM_TREE:-$EXP/gpu-mock}" && pwd)
[ -d "$GM_TREE/container" ] && [ -d "$GM_TREE/src/torch_patch" ] || { echo "$GM_TREE is not a gpu-mock tree (container/, src/)" >&2; exit 2; }
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 2; }
[ "$(uname -m)" = x86_64 ] || { echo "the Claude Code and torch pins below are for x86_64 hosts" >&2; exit 2; }

# ---- pins ---------------------------------------------------------------------------------------
# The gpu-daemon image runs gpu-mock's daemon, bridge and convert on this base, whatever the task's Python.
SIDECAR_BASE_IMAGE=python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317
DEBIAN_SNAPSHOT=20261015T000000Z
TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu
NVIDIA_DRIVER=580.65.06
GPUS=${GPUS_ARG:-4}
case "$GPUS" in [1-9]|1[0-6]) ;; *) echo "--gpus takes 1 to 16, got '$GPUS'" >&2; exit 2;; esac
# After changing CLAUDE_VERSION, rerun tools/cc-first-request/run.sh on the VM: the box's Claude Code environment
# (run_one.py) works around a first-request behaviour of this version (notes/thinking-display-400-2026-10-20.md).
CLAUDE_VERSION=2.1.303
# Codex CLI for --harness codex (the Codex driver's spec, 2026-10-21): the self-contained musl build from the npm
# platform package, without the node launcher (the box has no node; Claude Code is its native binary too).
CODEX_VERSION=0.163.0
CODEX_X64_SHA256=30efc97738b975940d78b59a9bea115fa34040b732807d54ec35ae6b7528de94
CLAUDE_X64_SHA512=yUbFvO0As5DKZfeu102GLBtrGz0ghkFteHyW4rbkKVXZCwSWisXqtbm4Nolt9AlKSCAggQoyqCEG7EEF0wXciQ==
# --------------------------------------------------------------------------------------------------

TASKJSON=$EXP/gpu-tasks/registry/tasks/$TASK.json
[ -f "$TASKJSON" ] || { echo "no task $TASK" >&2; exit 2; }
eval "$(python3 -I - "$TASKJSON" "$TASK" <<'PY'
import json, shlex, sys
e = json.load(open(sys.argv[1]))["environment"]
pk = e["install"]["packages"]
print("PYVER=" + shlex.quote(e["python"]["value"]))
print("CONS=" + shlex.quote(e["install"]["constraints"]))
# The image's p006.task-env label is the task's environment.image_label.value when it names one, else the task id:
# the same rule as run_one.py's image_label_of(), so the image the build labels is the image a run looks for. The
# modded-nanogpt package tasks all name `mng3-packages`, which is no task's id, so labelling by task id would
# build an image no task resolves to (found 2026-10-20, before any such image was shipped).
il = e.get("image_label") or {}
print("LABEL=" + shlex.quote(str(il.get("value") or sys.argv[2])))
print("PKGS=" + shlex.quote(" ".join(p for p in pk if not p.startswith("torch=="))))
print("TORCH=" + shlex.quote(next(p.split("==", 1)[1] for p in pk if p.startswith("torch=="))))
# The modules the verify step imports: every task package by its import name, plus gpu-mock's vendored FakeGPU
# (`nvcompat` since the 2026-10-21 scrub of mock names; it was `fakegpu`), importable once torch is imported. A task's
# packages are its own (the TRL tasks carry transformers, accelerate, datasets and peft; the modded-nanogpt
# packages carry torch, numpy and setuptools), so the check follows the task instead of naming one stack.
RENAME = {"typing-extensions": "typing_extensions", "pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
          "beautifulsoup4": "bs4", "msgpack-python": "msgpack", "protobuf": "google.protobuf"}
mods = []
for spec in pk:
    name = spec.split("==", 1)[0].split("[", 1)[0].strip().lower()
    mod = RENAME.get(name, name.replace("-", "_"))
    if mod not in mods:
        mods.append(mod)
mods.append("nvcompat")
print("VERIFY_MODS=" + shlex.quote(" ".join(mods)))
PY
)"
# a count other than 4 is its own image label, so it never answers a 4-GPU task's lookup (see --gpus above)
[ "$GPUS" = 4 ] || case "$LABEL" in *-${GPUS}gpu) ;; *) LABEL="$LABEL-${GPUS}gpu";; esac
# The box's base image follows the Python the task recorded. Each digest is the multi-arch index's, so a rebuild
# pulls the same image. To support another version, add it here with its digest.
case "$PYVER" in
  3.11.9)  BASE_IMAGE=python:3.11.9-slim-bookworm@sha256:8fb099199b9f2d70342674bd9dbccd3ed03a258f26bbd1d556822c6dfc60c317 ;;
  3.10.13) BASE_IMAGE=python:3.10.13-slim-bookworm@sha256:1326d0fd281d283b077fd249e618339a44c9ca5aae6e05cb4f069a087e827922 ;;
  *) echo "task wants Python $PYVER; build.sh has no pinned base image for it" >&2; exit 2 ;;
esac
TORCH_VERSION="$TORCH+cpu"

echo "== fresh work tree $WORK"
rm -rf "$WORK"
mkdir -p "$WORK/src" "$WORK/ctx" "$MANIFESTS"
C=$WORK/ctx

echo "== gpu-mock bundle, staged by gpu-mock's stage_bundle.sh at the revisions it pins"
# Every build step that fetches runs on the host network, because on 2026-10-19 containers on this host's
# default bridge could not resolve DNS (the only resolver was the metadata server, which DOCKER-USER blocks).
# The buildx builder is a container, so it is capped like every container on this host (gcp/AGENTS.md):
# pinned to the host's top four cores, with a memory cap and no swap. Boxes take core blocks from core 2
# upward, so a build shares cores with a box only when the boxes already fill the host; don't build then. It is created fresh for each build and removed when build.sh exits, even on failure, so no
# builder is left running between builds. A fresh builder has no layer cache for gpu-mock's runtime build step.
# One builder per work tree, so two builds at once never remove each other's (the default work tree keeps the old name).
BUILDER=isolation-hostnet
[ "$WORK" = /srv/build/box ] || BUILDER="isolation-hostnet-$(echo "$WORK" | sha256sum | cut -c1-8)"
NCPU=$(nproc)
BUILDER_CPUS=$((NCPU - 4))-$((NCPU - 1))
BUILDER_MEM=8g
docker buildx rm "$BUILDER" >/dev/null 2>&1 || true
docker buildx create --name "$BUILDER" --driver docker-container --driver-opt network=host \
  --driver-opt "cpuset-cpus=$BUILDER_CPUS" --driver-opt "memory=$BUILDER_MEM" \
  --driver-opt "memory-swap=$BUILDER_MEM" >/dev/null
trap 'docker buildx rm "$BUILDER" >/dev/null 2>&1 || true' EXIT
# The image checks below are one-off containers, capped and named the same way: verify_run <name> <args>.
verify_run() { local n=$1; shift; docker run --rm --name "isolation-build-verify-$n-$$" --cpus 2 --memory 4g --memory-swap 4g "$@"; }
if [ -n "$BUNDLE" ]; then
  [ -f "$BUNDLE/BUNDLE-MANIFEST.txt" ] || { echo "$BUNDLE has no BUNDLE-MANIFEST.txt" >&2; exit 2; }
  grep -q '^arch: amd64' "$BUNDLE/BUNDLE-MANIFEST.txt" || { echo "$BUNDLE is not an amd64 bundle" >&2; exit 2; }
  mkdir -p "$C/gpu-mock" && cp -a "$BUNDLE/." "$C/gpu-mock/"
else
  UPSTREAM=$WORK/src OUT=$C/gpu-mock GPUS=$GPUS DRIVER=$NVIDIA_DRIVER BUILDX_BUILDER=$BUILDER \
    bash "$GM_TREE/container/stage_bundle.sh"
fi
BUNDLE_REVS=$(sed -n '1,4p' "$C/gpu-mock/BUNDLE-MANIFEST.txt" | tr '\n' ';')
PATCHHASH=$(cd "$C/gpu-mock/torch_patch" && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -d' ' -f1)
TREE_PATCHHASH=$(cd "$GM_TREE/src/torch_patch" && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -d' ' -f1)
if [ "$PATCHHASH" != "$TREE_PATCHHASH" ]; then
  echo "   WARNING: the bundle's torch patch is not the tree's; the box gets the bundle's patch and the"
  echo "   gpu-daemon image gets the tree's daemon, so the two may not match. Restage without --bundle."
fi

echo "== gpu-daemon image, from the gpu-mock tree"
: "${SIDECAR_IMAGE:=gpu-daemon:${IMAGE##*:}}"
SC=$WORK/sidecar
for d in daemon bridge convert; do
  mkdir -p "$SC/src/$d"
  cp "$GM_TREE/src/$d/"*.py "$SC/src/$d/"
done
SIDECAR_SRC_HASH=$(cd "$SC/src" && find . -name '*.py' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
DOCKER_BUILDKIT=1 docker build $NOCACHE -f "$HERE/Dockerfile.sidecar" -t "$SIDECAR_IMAGE" \
  --build-arg "BASE_IMAGE=$SIDECAR_BASE_IMAGE" --build-arg "SRC_SHA256=$SIDECAR_SRC_HASH" \
  --label p006.role=gpu-daemon-image "$SC" 2>&1 | tee "$WORK/docker-build-sidecar.log"
SIDECAR_ID=$(docker image inspect -f '{{.Id}}' "$SIDECAR_IMAGE")
verify_run sidecar --network none --entrypoint python3 "$SIDECAR_IMAGE" -I /opt/gpu-daemon/src/daemon/gpu_daemon.py --help >/dev/null

echo "== build context"
[ -f "$C/gpu-mock/h100-${GPUS}gpu.yaml" ] && [ "$GPUS" != 4 ] && cp "$C/gpu-mock/h100-${GPUS}gpu.yaml" "$C/gpu-mock/h100-4gpu.yaml"
cp "$EXP/gpu-tasks/registry/$CONS" "$C/constraints.txt"
cp "$HERE/box-init" "$HERE/bash-oom-reset" "$C/"
# The scheduler's installable parts only: its AGENTS.md, probe/ and tests/ name us and stay out of the image.
mkdir -p "$C/scheduler" && cp -R "$EXP/scheduler/bin" "$EXP/scheduler/slurm" "$EXP/scheduler/install.sh" "$C/scheduler/" \
  && find "$C/scheduler" -name __pycache__ -prune -exec rm -rf {} +   # a selftest's bytecode is no part of the install
SCHED_SHA256=$(cd "$C/scheduler" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d' ' -f1)
curl -fsSL "https://registry.npmjs.org/@anthropic-ai/claude-code-linux-x64/-/claude-code-linux-x64-$CLAUDE_VERSION.tgz" \
  -o "$WORK/src/claude.tgz"
python3 -I -c 'import base64, hashlib, sys
d = base64.b64encode(hashlib.sha512(open(sys.argv[1], "rb").read()).digest()).decode()
sys.exit(0 if d == sys.argv[2] else "Claude Code tarball: sha512 does not match the pin")' "$WORK/src/claude.tgz" "$CLAUDE_X64_SHA512"
mkdir "$WORK/src/claude"
tar xzf "$WORK/src/claude.tgz" -C "$WORK/src/claude"
cp "$WORK/src/claude/package/claude" "$C/claude"
curl -fsSL "https://registry.npmjs.org/@openai/codex/-/codex-$CODEX_VERSION-linux-x64.tgz" -o "$WORK/src/codex.tgz"
echo "$CODEX_X64_SHA256  $WORK/src/codex.tgz" | sha256sum -c --quiet || { echo "Codex tarball: sha256 does not match the pin" >&2; exit 1; }
mkdir -p "$C/codex"
tar xzf "$WORK/src/codex.tgz" -C "$C/codex" --strip-components=3 package/vendor/x86_64-unknown-linux-musl
[ -x "$C/codex/bin/codex" ] || { echo "Codex tarball: no vendor/x86_64-unknown-linux-musl/bin/codex" >&2; exit 1; }
GMHASH=$(sha256sum "$C/gpu-mock/BUNDLE-MANIFEST.txt" | cut -d' ' -f1)
# gpu-mock's install script shapes the image as much as the patch does (the nvidia-smi wrapper, the in-place rename
# of the NVML library's variable names, the vendoring of FakeGPU), and neither of the two exchanged hashes covers it
#. The copy in the build context is the one the Dockerfile runs.
INSTALLHASH=$(sha256sum "$C/gpu-mock/install_into_image.sh" | cut -d' ' -f1)
BUILD_JSON=$(python3 -I -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:]), sort_keys=True))' \
  "base_image=$BASE_IMAGE" "debian_snapshot=$DEBIAN_SNAPSHOT" "torch=$TORCH_VERSION" "torch_index=$TORCH_CPU_INDEX" \
  "task_env=$LABEL" "built_from_task=$TASK" "constraints=$CONS" "packages=$PKGS" "gpu_mock_bundle=$BUNDLE_REVS" "claude_code=$CLAUDE_VERSION" "codex=$CODEX_VERSION" "codex_tarball_sha256=$CODEX_X64_SHA256" "scheduler_sha256=$SCHED_SHA256" "gpu_mock_bundle_manifest_sha256=$GMHASH" "torch_patch_sha256=$PATCHHASH" "gpu_mock_install_sha256=$INSTALLHASH" \
  "sidecar_image=$SIDECAR_IMAGE" "sidecar_image_id=$SIDECAR_ID" "sidecar_src_sha256=$SIDECAR_SRC_HASH")

echo "== docker build $IMAGE"
DOCKER_BUILDKIT=1 docker build $NOCACHE --network host -f "$HERE/Dockerfile" -t "$IMAGE" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" --build-arg "DEBIAN_SNAPSHOT=$DEBIAN_SNAPSHOT" \
  --build-arg "TORCH_CPU_INDEX=$TORCH_CPU_INDEX" --build-arg "TORCH_VERSION=$TORCH_VERSION" \
  --build-arg "TASK_PACKAGES=$PKGS" --build-arg "BUILD_JSON=$BUILD_JSON" \
  $( [ "$GPUS" = 4 ] || echo --build-arg "BOX_GPUS=$GPUS" ) \
  --label p006.role=box-image --label "p006.task-env=$LABEL" --label "p006.gpus=$GPUS" \
  --label "p006.sidecar-image=$SIDECAR_IMAGE" --label "p006.sidecar-image-id=$SIDECAR_ID" --label "p006.codex=$CODEX_VERSION" --label "p006.scheduler=${SCHED_SHA256:0:16}" \
  "$C" 2>&1 | tee "$WORK/docker-build.log"

echo "== verify"
verify_run smi --entrypoint sh "$IMAGE" -c 'env $(cat /etc/box-gpus 2>/dev/null) stage-dev-nodes >/dev/null && nvidia-smi -L' | tee "$WORK/verify-nvidia-smi.txt"
verify_run codex --network none --entrypoint codex "$IMAGE" --version | tee "$WORK/verify-codex.txt"
grep -qx "codex-cli $CODEX_VERSION" "$WORK/verify-codex.txt" || { echo "codex --version did not read codex-cli $CODEX_VERSION" >&2; exit 1; }
[ "$(grep -c '^GPU ' "$WORK/verify-nvidia-smi.txt")" = "$GPUS" ] || { echo "nvidia-smi -L did not list $GPUS GPUs" >&2; exit 1; }
# -i, or docker run does not forward the heredoc and python3 runs an empty script and exits 0: the mng3 build of
# 2026-10-20 18:04 UTC "verified" nothing that way (an empty verify-python.txt), caught by reading the file. The
# non-empty check below makes that class fail the build instead of passing it.
verify_run python -i --entrypoint python3 "$IMAGE" -I - "$VERIFY_MODS" <<'PY' | tee "$WORK/verify-python.txt"
import importlib, sys
out = []
for name in sys.argv[1].split():
    m = importlib.import_module(name)  # a missing module fails the build here
    out.append(f"{name} {getattr(m, '__version__', 'imported')}")
print(" ".join(out))
PY
[ -s "$WORK/verify-python.txt" ] || { echo "the verify step printed nothing, so it imported nothing" >&2; exit 1; }
for m in $VERIFY_MODS; do grep -q "\b$m " "$WORK/verify-python.txt" || { echo "verify did not report $m" >&2; exit 1; }; done
# tiktoken's GPT-2 files must load with the network off, from the cache the Dockerfile filled (the data-prep package).
if [[ " $VERIFY_MODS " == *" tiktoken "* ]]; then
  verify_run tiktoken --network none -e TIKTOKEN_CACHE_DIR=/usr/local/share/tiktoken --entrypoint python3 "$IMAGE" -I -c \
    "import tiktoken; e = tiktoken.get_encoding('gpt2'); print('tiktoken gpt2 offline', e.n_vocab, e.encode('hello world'))" \
    | tee "$WORK/verify-tiktoken.txt"
  grep -q "tiktoken gpt2 offline 50257" "$WORK/verify-tiktoken.txt" || { echo "tiktoken's gpt2 encoding did not load offline" >&2; exit 1; }
fi

echo "== manifest"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
MF=$MANIFESTS/$(echo "$IMAGE" | tr ':/' '__')-$STAMP.json
D=$WORK/manifest
mkdir -p "$D"
CID=$(docker create --entrypoint true "$IMAGE")
docker cp "$CID:/usr/local/share/image-manifest/." "$D/" >/dev/null
docker rm "$CID" >/dev/null
python3 -I - "$D" "$IMAGE" "$(docker image inspect -f '{{.Id}}' "$IMAGE")" "$C/gpu-mock/BUNDLE-MANIFEST.txt" "$STAMP" > "$MF" <<'PY'
import json, os, sys
d, image, image_id, bundle_manifest, stamp = sys.argv[1:6]
r = lambda n: open(os.path.join(d, n)).read()
print(json.dumps({"image": image, "image_id": image_id, "built_utc": stamp, "build_args": json.loads(r("build-args.json")),
                  "gpu_mock_bundle": open(bundle_manifest).read().split("\n\n")[0].splitlines(), "python": r("python-version.txt").strip(),
                  "claude_version": r("claude-version.txt").strip(),
                  "pip": sorted(r("pip-freeze.txt").splitlines()), "dpkg": sorted(r("dpkg.txt").splitlines())}, indent=1))
PY
ln -sf "$MF" "$MANIFESTS/latest-$(echo "$IMAGE" | tr ':/' '__').json"
echo "manifest: $MF"

if [ -n "$COMPARE" ]; then
  echo "== compare with $COMPARE"
  python3 -I - "$COMPARE" "$MF" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = []
for k in ("python", "claude_version"):
    if a[k] != b[k]:
        diff.append(f"{k}: {a[k]} -> {b[k]}")
for k in ("pip", "dpkg"):
    for line in sorted(set(a[k]) ^ set(b[k])):
        diff.append(f"{k}: {'-' if line in a[k] else '+'} {line}")
if a.get("gpu_mock_bundle") != b.get("gpu_mock_bundle"):
    diff.append(f"gpu-mock bundle revisions: {a.get('gpu_mock_bundle')} -> {b.get('gpu_mock_bundle')}")
for k in ("torch_patch_sha256", "sidecar_src_sha256", "gpu_mock_install_sha256"):
    if k not in a["build_args"] or k not in b["build_args"]:
        print(f"note: {k} is not in both manifests, so it was not compared")
    elif a["build_args"][k] != b["build_args"][k]:
        diff.append(f"{k}: {a['build_args'][k]} -> {b['build_args'][k]}")
info = a["build_args"].get("gpu_mock_bundle_manifest_sha256") != b["build_args"].get("gpu_mock_bundle_manifest_sha256")
print("\n".join(diff) if diff else "same versions")
if info:
    print("note: the bundle's file hashes differ (compiled binaries are not byte-reproducible); revisions are compared above")
sys.exit(1 if diff else 0)
PY
fi
