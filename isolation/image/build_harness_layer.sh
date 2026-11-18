#!/usr/bin/env bash
# Add Grok Build and OpenCode to an image in service, by id, on the Docker host (the E2), as root:
#   build_harness_layer.sh <base image id or tag> <new tag> [--compare <manifest>]
# e.g. build_harness_layer.sh sha256:04827d41a7e0… ml-node:py311-torch2110-harn
# The pinned binaries are fetched into a fresh directory and checked against their sha256 before anything else; the
# new image inherits the base's labels (task env, box-stdin, sidecar, codex, scheduler) and adds p006.grok,
# p006.opencode and p006.base-image-id. Its p006.role is box-image-by-id, not box-image: run_one.py's default
# lookup takes the one tagged image with p006.role=box-image and the task's label, and refuses two, so a harness image
# beside the frozen one would stop every line on that label. Lines on it name it with --image, which checks the task
# label and not the role (analysis' ruling 2026-10-26 22:24Z: only Grok and OpenCode lines select it). Unlike
# build_codex_layer.sh it moves no tag of the base. Writes a
# manifest to /srv/build/manifests/<tag>-harnesses-<stamp>.json; --compare checks a second build has the same binaries
# and versions (the Install checkpoint for a layer, built twice).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$1; TAG=$2; shift 2
COMPARE=""; [ "${1:-}" = "--compare" ] && COMPARE=$2
# Grok Build 1.0.70: xAI's own artefact, the same bytes on its GCS fallback (checked 2026-10-26)
GROK_VERSION=1.0.70
GROK_URL="https://x.ai/cli/grok-$GROK_VERSION-linux-x86_64.gz"
GROK_URL_FALLBACK="https://storage.googleapis.com/grok-build-public-artifacts/cli/grok-$GROK_VERSION-linux-x86_64.gz"
GROK_GZ_SHA256=11d27a30adfb75c4b9b180072ab27412a1916f054ff5497ab70fb9a77d304bcc
GROK_SHA256=22d281c5964c5444147328096ba2a221f3f0874c6858e27c825656a80f6e58e9
# OpenCode 1.19.4: the release asset its installer fetches; the sha256 equals GitHub's asset digest (2026-10-26)
OPENCODE_VERSION=1.19.4
OPENCODE_URL="https://github.com/anomalyco/opencode/releases/download/v$OPENCODE_VERSION/opencode-linux-x64.tar.gz"
OPENCODE_TGZ_SHA256=849bd6b3d959a1d2ef200ed1634a51c79a45ca236aa26665d75debd694805cce
OPENCODE_SHA256=538fcb1b6408c8d4608822acee19f7ff5b3a3eb3383c0633c4f06170857312a7
BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE")
mkdir -p /srv/build/manifests
W=$(mktemp -d /srv/build/harness-layer.XXXX); trap 'rm -rf "$W"' EXIT
mkdir "$W/dl" "$W/ctx"
curl -fsSL "$GROK_URL" -o "$W/dl/grok.gz" || curl -fsSL "$GROK_URL_FALLBACK" -o "$W/dl/grok.gz"
echo "$GROK_GZ_SHA256  $W/dl/grok.gz" | sha256sum -c --quiet || { echo "Grok Build: sha256 does not match the pin" >&2; exit 1; }
curl -fsSL "$OPENCODE_URL" -o "$W/dl/opencode.tgz"
echo "$OPENCODE_TGZ_SHA256  $W/dl/opencode.tgz" | sha256sum -c --quiet || { echo "OpenCode: sha256 does not match the pin" >&2; exit 1; }
gunzip -c "$W/dl/grok.gz" > "$W/ctx/grok"
tar xzf "$W/dl/opencode.tgz" -C "$W/ctx" opencode
chmod 0755 "$W/ctx/grok" "$W/ctx/opencode"
echo "$GROK_SHA256  $W/ctx/grok" | sha256sum -c --quiet || { echo "Grok Build binary: sha256 does not match the pin" >&2; exit 1; }
echo "$OPENCODE_SHA256  $W/ctx/opencode" | sha256sum -c --quiet || { echo "OpenCode binary: sha256 does not match the pin" >&2; exit 1; }
cp "$HERE/Dockerfile.harnesses" "$W/ctx/Dockerfile"
printf '{"base_image_id": "%s", "grok": "%s", "grok_gz_sha256": "%s", "grok_sha256": "%s", "opencode": "%s", "opencode_tgz_sha256": "%s", "opencode_sha256": "%s"}\n' \
  "$BASE_ID" "$GROK_VERSION" "$GROK_GZ_SHA256" "$GROK_SHA256" "$OPENCODE_VERSION" "$OPENCODE_TGZ_SHA256" "$OPENCODE_SHA256" > "$W/ctx/harness-layer.json"
# NOCACHE=1 for the second, from-scratch build of the check. FROM takes a local tag: BuildKit reads a bare sha256 as a
# registry name, so the base gets a temporary tag that is removed again below.
SHORT=${BASE_ID#sha256:}; SHORT=${SHORT:0:12}
docker tag "$BASE_ID" "harness-base-$SHORT:build"
DOCKER_BUILDKIT=1 docker build --network none ${NOCACHE:+--no-cache} --build-arg "BASE=harness-base-$SHORT:build" -t "$TAG" \
  --label "p006.grok=$GROK_VERSION" --label "p006.opencode=$OPENCODE_VERSION" --label "p006.base-image-id=$BASE_ID" \
  --label "p006.role=box-image-by-id" "$W/ctx"
docker rmi "harness-base-$SHORT:build" >/dev/null
NEW_ID=$(docker image inspect -f '{{.Id}}' "$TAG")
run() { docker run --rm --network none --cpus 1 --memory 1g --user 1000:1000 -e HOME=/tmp --entrypoint "$1" "$TAG" "${@:2}"; }
GV=$(run grok --version); OV=$(run opencode --version)
[ "$GV" = "grok $GROK_VERSION (f50904eaee6c)" ] || { echo "grok --version reads '$GV'" >&2; exit 1; }
[ "$OV" = "$OPENCODE_VERSION" ] || { echo "opencode --version reads '$OV'" >&2; exit 1; }
IN=$(docker run --rm --network none --cpus 1 --memory 1g --entrypoint sha256sum "$TAG" /usr/local/lib/grok/grok /usr/local/lib/opencode/opencode | cut -c1-64 | tr '\n' ' ')
[ "$IN" = "$GROK_SHA256 $OPENCODE_SHA256 " ] || { echo "the binaries in the image ($IN) are not the pinned ones" >&2; exit 1; }
[ "$(docker image inspect -f '{{ index .Config.Labels "p006.role" }}' "$TAG")" = box-image-by-id ] || { echo "p006.role is not box-image-by-id" >&2; exit 1; }
for l in p006.task-env p006.box-stdin p006.sidecar-image p006.sidecar-image-id p006.codex p006.scheduler; do
  a=$(docker image inspect -f "{{ index .Config.Labels \"$l\" }}" "$BASE_ID"); b=$(docker image inspect -f "{{ index .Config.Labels \"$l\" }}" "$TAG")
  [ "$a" = "$b" ] || { echo "label $l: base '$a', new '$b'" >&2; exit 1; }
done
STAMP=$(date -u +%Y%m%dT%H%M%SZ); MF=/srv/build/manifests/$(echo "$TAG" | tr ':/' '__')-harnesses-$STAMP.json
printf '{"image": "%s", "image_id": "%s", "base_image_id": "%s", "grok": "%s", "grok_sha256": "%s", "grok_version_output": "%s", "opencode": "%s", "opencode_sha256": "%s", "opencode_version_output": "%s", "built_utc": "%s"}\n' \
  "$TAG" "$NEW_ID" "$BASE_ID" "$GROK_VERSION" "$GROK_SHA256" "$GV" "$OPENCODE_VERSION" "$OPENCODE_SHA256" "$OV" "$STAMP" > "$MF"
echo "== $TAG = $NEW_ID (base $BASE_ID), grok $GROK_VERSION, opencode $OPENCODE_VERSION"
echo "manifest: $MF"
if [ -n "$COMPARE" ]; then
  python3 -I - "$COMPARE" "$MF" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = [k for k in ("base_image_id", "grok", "grok_sha256", "grok_version_output", "opencode", "opencode_sha256",
                    "opencode_version_output") if a[k] != b[k]]
print("same versions" if not diff else "DIFF: " + ", ".join(diff))
sys.exit(1 if diff else 0)
PY
fi
