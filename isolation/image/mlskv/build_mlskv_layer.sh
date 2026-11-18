#!/usr/bin/env bash
# Derive the MLS-Bench KV box (task env `mlsbench-kv`) from a gated box image, by id, on the Docker host (the E2), as root:
#   build_mlskv_layer.sh <base image id> <new tag> [--compare <manifest>]
# e.g. build_mlskv_layer.sh sha256:fe5688282432… ml-node:py311-torch2110-mlskv
# The base is a gated mng3 box (gpu-mock's patch, the sidecar label, Claude Code, Codex, box-init), kept as it is; this adds
# MLS-Bench's transformers-kv-lab package (huggingface/transformers at TRANSFORMERS_COMMIT, v5.3.0) and its extras under
# constraints.txt beside this file. The new image takes label p006.task-env=mlsbench-kv and keeps the base's sidecar label,
# so run_one pairs it with the base's gpu-daemon image. When gpu-mock's next pair is gated, rebuild on that base: minutes.
# Writes /srv/build/manifests/<tag>-mlskv-<stamp>.json; --compare checks a second build resolved the same versions.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$1; TAG=$2; shift 2
COMPARE=""; [ "${1:-}" = "--compare" ] && COMPARE=$2
TRANSFORMERS_COMMIT=aad13b87ed59f2afcfaebc985f403301887a35fc     # MLS-Bench b873944's vendor/packages.yaml; tag v5.3.0
MLSBENCH_COMMIT=b873944969c53b01e4ee6e843c77767b8c702485
if ! BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE" 2>/dev/null); then
  f=$(ls /srv/archive/images/$(echo "${BASE#sha256:}" | cut -c1-12).tar.gz 2>/dev/null) || { echo "no image $BASE and no archive" >&2; exit 1; }
  gunzip -c "$f" | docker load >/dev/null
  BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE")
fi
SHORT=${BASE_ID#sha256:}; SHORT=${SHORT:0:12}
BASE_LABEL=$(docker image inspect -f '{{index .Config.Labels "p006.task-env"}}' "$BASE_ID")
# the base's GPU count carries over (build.sh --gpus): a base built for n GPUs other than 4 makes mlsbench-kv-<n>gpu
NG=$(docker image inspect -f '{{index .Config.Labels "p006.gpus"}}' "$BASE_ID"); case "$NG" in ''|'<no value>') NG=4;; esac
KVLABEL=mlsbench-kv; [ "$NG" = 4 ] || KVLABEL=mlsbench-kv-${NG}gpu
SIDECAR=$(docker image inspect -f '{{index .Config.Labels "p006.sidecar-image-id"}}' "$BASE_ID")
[ -n "$SIDECAR" ] || { echo "base $SHORT names no sidecar image" >&2; exit 1; }
W=$(mktemp -d /srv/build/mlskv-layer.XXXX); trap 'rm -rf "$W"' EXIT
cp "$HERE/Dockerfile" "$HERE/constraints.txt" "$W/"
CONS_SHA=$(sha256sum "$W/constraints.txt" | cut -d' ' -f1)
BUILD_JSON=$(python3 -I -c 'import json, sys; print(json.dumps(dict(a.split("=", 1) for a in sys.argv[1:]), sort_keys=True))' \
  "base_image_id=$BASE_ID" "base_task_env=$BASE_LABEL" "transformers_commit=$TRANSFORMERS_COMMIT" "transformers_version=5.3.0" \
  "mlsbench_commit=$MLSBENCH_COMMIT" "constraints_sha256=$CONS_SHA" \
  "torch_upstream=2.5.1 (cu121, MLS-Bench vendor/pkg_configs/transformers-kv-lab)" \
  "torch_here=2.11.0+cpu, the base's (generation never runs under the serving hook; v5.3.0 takes torch>=2.4; the study 2026-10-22)")
docker tag "$BASE_ID" "$TAG-base-$SHORT"
echo "== docker build $TAG FROM $SHORT"
DOCKER_BUILDKIT=1 docker build --network host ${NOCACHE:+--no-cache} --build-arg "BASE=$TAG-base-$SHORT" \
  --build-arg "TRANSFORMERS_COMMIT=$TRANSFORMERS_COMMIT" --build-arg "BUILD_JSON=$BUILD_JSON" \
  --label "p006.task-env=$KVLABEL" --label "p006.gpus=$NG" --label "p006.mlskv-base-image-id=$BASE_ID" \
  --label "p006.transformers-commit=$TRANSFORMERS_COMMIT" -t "$TAG" "$W" 2>&1 | tail -20
# The temporary base tag goes: two tagged images with one label refuse the by-label lookup. The base stays, untagged
# or under its own tags; never rmi a base an image was derived from (2026-10-21).
docker rmi "$TAG-base-$SHORT" >/dev/null
ID=$(docker image inspect -f '{{.Id}}' "$TAG")
echo "== verify $ID"
v() { docker run --rm --name "isolation-mlskv-verify-$1-$$" --cpus 2 --memory 8g --memory-swap 8g --network none "${@:2}"; }
v py --entrypoint python3 "$ID" -I -c "import transformers, torch, datasets, accelerate, huggingface_hub, tokenizers, safetensors, sentencepiece, google.protobuf, hqq, optimum.quanto, nvcompat
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
assert transformers.__version__ == '5.3.0', transformers.__version__
print('transformers', transformers.__version__, 'torch', torch.__version__, 'datasets', datasets.__version__, 'hub', huggingface_hub.__version__, 'tokenizers', tokenizers.__version__)" | tee "$W/verify-python.txt"
v smi --entrypoint sh "$ID" -c 'env $(cat /etc/box-gpus 2>/dev/null) stage-dev-nodes >/dev/null && nvidia-smi -L' | tee "$W/verify-smi.txt"
[ "$(grep -c '^GPU ' "$W/verify-smi.txt")" = "$NG" ] || { echo "nvidia-smi did not list $NG GPUs" >&2; exit 1; }
v codex --entrypoint codex "$ID" --version | grep -qx 'codex-cli 0.163.0' || { echo "codex missing" >&2; exit 1; }
for k in p006.task-env p006.box-stdin p006.codex p006.sidecar-image-id; do
  echo "label $k=$(docker image inspect -f "{{index .Config.Labels \"$k\"}}" "$ID")"
done | tee "$W/verify-labels.txt"
grep -qx "label p006.task-env=$KVLABEL" "$W/verify-labels.txt" && grep -q "label p006.sidecar-image-id=$SIDECAR" "$W/verify-labels.txt"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); MF=/srv/build/manifests/$(echo "$TAG" | tr ':/' '__')-mlskv-$STAMP.json
mkdir -p "$W/m"; CID=$(docker create --entrypoint true "$ID"); docker cp "$CID:/usr/local/share/image-manifest/." "$W/m/" >/dev/null; docker rm "$CID" >/dev/null
python3 -I - "$W" "$TAG" "$ID" "$STAMP" > "$MF" <<'PY'
import json, os, sys
w, tag, iid, stamp = sys.argv[1:5]
m = os.path.join(w, "m")
print(json.dumps({"image": tag, "image_id": iid, "built_utc": stamp,
                  "build_args": json.load(open(os.path.join(m, "mlskv-layer.json"))),
                  "pip": sorted(open(os.path.join(m, "pip-freeze.txt")).read().split()),
                  "verify": open(os.path.join(w, "verify-python.txt")).read().strip()}, indent=1))
PY
echo "manifest: $MF"
if [ -n "$COMPARE" ]; then
  python3 -I -c 'import json, sys; a, b = (json.load(open(p)) for p in sys.argv[1:3]); d = sorted(set(a["pip"]) ^ set(b["pip"])); print("same versions" if not d else "\n".join(d)); sys.exit(1 if d else 0)' "$COMPARE" "$MF"
fi
echo "built $TAG = $ID (from $SHORT)"
