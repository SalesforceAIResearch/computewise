#!/bin/sh
# What does Claude Code send on its first model request? Runs the box image's Claude Code in a capped container
# with no network, against a local stub that applies the gateway's rule on thinking.display (see stub.py). No
# model is called and nothing goes through the router or the forwarder. Variants (VARIANTS="a b" picks a subset): no
# setting, CLAUDE_CODE_THINKING_DISPLAY_UPDATES=0, =false, --thinking-display omitted, and for gap G7
# --thinking-display summarized with and without the variable (the stub answers a summarised request with a
# thinking block, so the stream shows whether Claude Code surfaces its text).
# Rerun it whenever the image's Claude Code pin changes. On the VM, as root, while no run is in flight:
#
#   sudo sh /opt/computewise/isolation/tools/cc-first-request/run.sh [image]
#
# Output: /srv/work/isolation/cc-first-request-<stamp>/ with each variant's stub log (<variant>.jsonl), Claude
# Code's stream and stderr, meta.txt and summary.txt. Background: isolation/notes/thinking-display-400-2026-10-20.md.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=${1:-ml-node:py311-torch240}
OUT=/srv/work/isolation/cc-first-request-$(date -u +%Y%m%dT%H%M%SZ)
N=$(nproc)
mkdir -p "$OUT" && chown 1000:1000 "$OUT"
run() {
  docker run --rm --name isolation-cc-first-request-$$ --network none "$@" \
    --cpuset-cpus "$((N - 2))-$((N - 1))" --cpus 2 --memory 2g --memory-swap 2g --pids-limit 512 \
    --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges \
    -e "VARIANTS=${VARIANTS:-}" \
    -v "$HERE/stub.py:/t/stub.py:ro" -v "$HERE/inside.sh:/t/inside.sh:ro" -v "$OUT:/out" \
    --entrypoint sh "$IMG" /t/inside.sh
}
run --add-host api-gw.internal:127.0.0.1; rc=$?
if [ "$rc" = 125 ]; then
  echo "docker refused --add-host with --network none; the base URL is 127.0.0.1" >> "$OUT/meta.txt"
  run; rc=$?
fi
echo "image: $(docker image inspect -f '{{.Id}}' "$IMG")" >> "$OUT/meta.txt"
echo "docker exit: $rc" >> "$OUT/meta.txt"
python3 -I "$HERE/analyze.py" "$OUT" | tee "$OUT/summary.txt"
echo "output: $OUT"
