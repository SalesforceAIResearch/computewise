#!/usr/bin/env bash
# Derive the Codex image from an image in service, by id, on the Docker host (the E2), as root:
#   build_codex_layer.sh <base image id or tag> <new tag> [--compare <manifest>]
# e.g. build_codex_layer.sh sha256:b41619603945… ml-node:py311-torch2110
# Before retagging it keeps the base: `docker save` to /srv/archive/images/<id>.tar.gz and a
# tag <new tag>-base-<12 hex>, so an untagged base never vanishes on the containerd store. The new image inherits the
# base's labels (task env, box-stdin, sidecar) and adds p006.codex and p006.base-image-id. Writes a manifest to
# /srv/build/manifests/<tag>-codex-<stamp>.json with the codex tree's file hashes; --compare checks a second build has
# the same tree and versions (the Install checkpoint for a layer, built twice).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BASE=$1; TAG=$2; shift 2
COMPARE=""; [ "${1:-}" = "--compare" ] && COMPARE=$2
CODEX_VERSION=0.163.0
CODEX_X64_SHA256=30efc97738b975940d78b59a9bea115fa34040b732807d54ec35ae6b7528de94
# The base by id; restored from its archive if the store has dropped it (untagged images can vanish on containerd).
if ! BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE" 2>/dev/null); then
  f=$(ls /srv/archive/images/$(echo "${BASE#sha256:}" | cut -c1-12).tar.gz 2>/dev/null) || { echo "no image $BASE and no archive" >&2; exit 1; }
  gunzip -c "$f" | docker load >/dev/null
  BASE_ID=$(docker image inspect -f '{{.Id}}' "$BASE")
fi
SHORT=${BASE_ID#sha256:}; SHORT=${SHORT:0:12}
mkdir -p /srv/archive/images /srv/build/manifests
if [ ! -s "/srv/archive/images/$SHORT.tar.gz" ]; then
  echo "== keeping the base: docker save $BASE_ID -> /srv/archive/images/$SHORT.tar.gz"
  docker save "$BASE_ID" | gzip -1 > "/srv/archive/images/$SHORT.tar.gz.part" && mv "/srv/archive/images/$SHORT.tar.gz.part" "/srv/archive/images/$SHORT.tar.gz"
fi
docker tag "$BASE_ID" "$TAG-base-$SHORT"
W=$(mktemp -d /srv/build/codex-layer.XXXX); trap 'rm -rf "$W"' EXIT
curl -fsSL "https://registry.npmjs.org/@openai/codex/-/codex-$CODEX_VERSION-linux-x64.tgz" -o "$W/codex.tgz"
echo "$CODEX_X64_SHA256  $W/codex.tgz" | sha256sum -c --quiet || { echo "Codex tarball: sha256 does not match the pin" >&2; exit 1; }
mkdir "$W/ctx" "$W/ctx/codex"
tar xzf "$W/codex.tgz" -C "$W/ctx/codex" --strip-components=3 package/vendor/x86_64-unknown-linux-musl
cp "$HERE/Dockerfile.codex" "$W/ctx/Dockerfile"
TREE=$(cd "$W/ctx/codex" && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64)
printf '{"base_image_id": "%s", "codex": "%s", "codex_tarball_sha256": "%s", "codex_tree_sha256": "%s"}\n' \
  "$BASE_ID" "$CODEX_VERSION" "$CODEX_X64_SHA256" "$TREE" > "$W/ctx/codex-layer.json"
# NOCACHE=1 for the second, from-scratch build of the check.
# FROM takes the base's local tag: BuildKit reads a bare sha256 as a registry name (docker.io/library/sha256:…).
DOCKER_BUILDKIT=1 docker build --network none ${NOCACHE:+--no-cache} --build-arg "BASE=$TAG-base-$SHORT" -t "$TAG" \
  --label "p006.codex=$CODEX_VERSION" --label "p006.base-image-id=$BASE_ID" "$W/ctx"
NEW_ID=$(docker image inspect -f '{{.Id}}' "$TAG")
V=$(docker run --rm --network none --cpus 1 --memory 1g --entrypoint codex "$TAG" --version)
[ "$V" = "codex-cli $CODEX_VERSION" ] || { echo "codex --version reads '$V'" >&2; exit 1; }
for l in p006.task-env p006.box-stdin p006.sidecar-image p006.sidecar-image-id p006.role; do
  a=$(docker image inspect -f "{{ index .Config.Labels \"$l\" }}" "$BASE_ID"); b=$(docker image inspect -f "{{ index .Config.Labels \"$l\" }}" "$TAG")
  [ "$a" = "$b" ] || { echo "label $l: base '$a', new '$b'" >&2; exit 1; }
done
INTREE=$(docker run --rm --network none --cpus 1 --memory 1g --entrypoint sh "$TAG" -c 'cd /usr/local/lib/codex && find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-64')
[ "$INTREE" = "$TREE" ] || { echo "the codex tree in the image ($INTREE) is not the tarball's ($TREE)" >&2; exit 1; }
STAMP=$(date -u +%Y%m%dT%H%M%SZ); MF=/srv/build/manifests/$(echo "$TAG" | tr ':/' '__')-codex-$STAMP.json
printf '{"image": "%s", "image_id": "%s", "base_image_id": "%s", "codex": "%s", "codex_tarball_sha256": "%s", "codex_tree_sha256": "%s", "codex_version_output": "%s", "built_utc": "%s"}\n' \
  "$TAG" "$NEW_ID" "$BASE_ID" "$CODEX_VERSION" "$CODEX_X64_SHA256" "$TREE" "$V" "$STAMP" > "$MF"
# The base's tag goes again: a second image under the task label makes run_one.py's lookup refuse (the Codex driver,
# 2026-10-21). The base stays in /srv/archive/images/ and its layers stay referenced by the new image.
docker rmi "$TAG-base-$SHORT" >/dev/null
echo "== $TAG = $NEW_ID (base $BASE_ID), codex $CODEX_VERSION, tree $TREE"
echo "manifest: $MF"
if [ -n "$COMPARE" ]; then
  python3 -I - "$COMPARE" "$MF" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
diff = [k for k in ("base_image_id", "codex", "codex_tarball_sha256", "codex_tree_sha256", "codex_version_output") if a[k] != b[k]]
print("same versions" if not diff else "DIFF: " + ", ".join(diff))
sys.exit(1 if diff else 0)
PY
fi
