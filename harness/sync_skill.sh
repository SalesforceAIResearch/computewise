#!/usr/bin/env bash
# Copy a gpu-harness skill tree into this plugin, replacing the one there, and check its build stamp.
#
#   ./sync_skill.sh <source skill dir> [claude|codex]
#
# claude (the default) replaces skills/gpu-harness/, the Claude Code edition; codex replaces codex-skills/gpu-harness/,
# the Codex edition. The two editions run the same harness: only SKILL.md differs, because each tells the agent how to
# start the harness and its teammates with its own tool's mechanics. The source is a directory holding SKILL.md,
# bin/gpuh, gpuharness/ and stamp.py (for example a frozen copy of the version the paper measured). Every file is
# copied with its mode, so bin/gpuh stays executable; __pycache__ is left out, and files that are not in the source
# are removed. Then stamp.py --check confirms that SKILL.md carries the build stamp of the code beside it, and the
# script reports whether the two editions still carry identical code.
set -euo pipefail

src="${1:?usage: sync_skill.sh <source skill dir> [claude|codex]}"
edition="${2:-claude}"
src="$(cd "$src" && pwd)"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$edition" in
  claude) dst="$here/skills/gpu-harness" ;;
  codex)  dst="$here/codex-skills/gpu-harness" ;;
  *) echo "refused: edition must be claude or codex, not $edition" >&2; exit 2 ;;
esac

for f in SKILL.md bin/gpuh stamp.py gpuharness/cli.py; do
  [ -e "$src/$f" ] || { echo "refused: $src has no $f; is it a gpu-harness skill directory?" >&2; exit 2; }
done
[ "$src" != "$dst" ] || { echo "refused: the source is this plugin's own copy" >&2; exit 2; }

mkdir -p "$dst"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete --exclude '__pycache__' "$src/" "$dst/"
else
  rm -rf "$dst" && mkdir -p "$dst"
  (cd "$src" && tar -cf - --exclude '__pycache__' .) | (cd "$dst" && tar -xpf -)
fi

echo "copied $src -> $dst ($edition edition)"
if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$dst/SKILL.md"; else sha256sum "$dst/SKILL.md"; fi
(cd "$dst" && python3 -I stamp.py --check)

# both editions should run the same harness code
a="$here/skills/gpu-harness"; b="$here/codex-skills/gpu-harness"
if [ -d "$a" ] && [ -d "$b" ]; then
  if diff -rq --exclude '__pycache__' --exclude SKILL.md "$a" "$b" >/dev/null; then
    echo "the Claude Code and Codex editions carry identical code"
  else
    echo "warning: the two editions' code differs (diff -r --exclude SKILL.md $a $b); sync the other edition too" >&2
  fi
fi
