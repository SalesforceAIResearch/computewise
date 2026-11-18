#!/usr/bin/env bash
# Stage the complete gpu-mock bundle that isolation's image build consumes
# (experiments/isolation/image/Dockerfile copies it as ctx/gpu-mock/), built for
# the Docker host's own architecture. Run it on the GCP host for the real box.
#
#   $OUT/runtime/              bin/ lib/ from build_bundle.sh, plus pins.env
#   $OUT/h100-<GPUS>gpu.yaml   Mokka H100 profile trimmed to GPUS devices
#   $OUT/mock-pci/             fake /proc and /sys tree (build_os_surface.sh)
#   $OUT/torch_patch/          src/torch_patch/*.py
#   $OUT/FakeGPU/              FakeGPU source at the pinned commit, no .git
#   $OUT/install_into_image.sh, $OUT/stage_dev_nodes.sh, $OUT/strip_comments.py
#   $OUT/BUNDLE-MANIFEST.txt   arch, pinned revisions, sha256 of every file
#
# It refuses to finish if any ELF file in runtime/ is not the host's arch.
# Usage:  UPSTREAM=/srv/work/gpu-mock/upstream OUT=/opt/computewise-deploy/gm-ctx/gpu-mock bash stage_bundle.sh
# BUILDX_BUILDER names a builder for the runtime build; unset, the default one is
# used (on build-host the default bridge resolves DNS since 2026-10-20).
# (on the VM, run with sudo: docker needs it there; files are chowned back.)
set -euo pipefail
# Under sudo the clones belong to the login user; let git operate on them.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="*"
HERE=$(cd "$(dirname "$0")" && pwd); GM=$(cd "$HERE/.." && pwd)
UPSTREAM="${UPSTREAM:?dir for the k8s-test-infra and FakeGPU clones}"
OUT="${OUT:?bundle output dir}"
GPUS="${GPUS:-4}"; DRIVER="${DRIVER:-580.65.06}"
# k8s-test-infra at this revision supplies libmockfs and the H100 profile; build_bundle.sh pins the mock NVML and
# libcuda revisions it builds in runtime.Dockerfile.
MOKKA_REV=a82e8b9cafd62e6f242bb4df41f71ffa11aa4c83
FAKEGPU_REV=3b5b6b47d1c6fc259b41a7ee07deb0c18ef1c81a

_clone() {  # dir url rev
  local d="$UPSTREAM/$1"
  [ -d "$d/.git" ] || git clone -q "$2" "$d"
  git -C "$d" fetch -q origin 2>/dev/null || true
  git -C "$d" checkout -q --detach "$3"
  test "$(git -C "$d" rev-parse HEAD)" = "$3" || { echo "$1 is not at $3" >&2; exit 1; }
}
mkdir -p "$UPSTREAM" "$OUT"
# Under sudo, give the clones and the output back to the login user even on failure.
[ -n "${SUDO_UID:-}" ] && trap 'chown -R "$SUDO_UID:$SUDO_GID" "$OUT" "$UPSTREAM"' EXIT
_clone k8s-test-infra https://github.com/NVIDIA/k8s-test-infra "$MOKKA_REV"
_clone FakeGPU        https://github.com/FanBB2333/FakeGPU     "$FAKEGPU_REV"

MOKKA="$UPSTREAM/k8s-test-infra" OUT="$OUT" GPUS="$GPUS" \
  bash "$HERE/build_bundle.sh"
OUT="$OUT" DRIVER="$DRIVER" GPUS="$GPUS" bash "$HERE/build_os_surface.sh"

rm -rf "$OUT/torch_patch" "$OUT/FakeGPU"
mkdir -p "$OUT/torch_patch" "$OUT/FakeGPU"
cp "$GM"/src/torch_patch/*.py "$OUT/torch_patch/"
git -C "$UPSTREAM/FakeGPU" archive --format=tar "$FAKEGPU_REV" | tar -x -C "$OUT/FakeGPU"
install -m 0755 "$HERE/install_into_image.sh" "$HERE/stage_dev_nodes.sh" "$HERE/strip_comments.py" "$OUT/"

ARCH=$(docker version --format '{{.Server.Arch}}')
python3 -I - "$OUT" "$ARCH" "$MOKKA_REV" "$FAKEGPU_REV" <<'PY'
import hashlib, os, sys
out, arch, mk, fg = sys.argv[1:5]
want = {"amd64": 62, "arm64": 183}[arch]          # ELF e_machine: x86-64, AArch64
names = {62: "x86-64", 183: "aarch64"}
bad, elves, rows = [], 0, []
for root, _, files in os.walk(out):
    for f in sorted(files):
        p = os.path.join(root, f)
        if os.path.islink(p) or f == "BUNDLE-MANIFEST.txt":
            continue
        data = open(p, "rb").read()
        rows.append(f"{hashlib.sha256(data).hexdigest()}  {os.path.relpath(p, out)}")
        if data[:4] == b"\x7fELF":
            elves += 1
            m = int.from_bytes(data[18:20], "little")
            if m != want:
                bad.append(f"{os.path.relpath(p, out)}: {names.get(m, m)}")
if bad:
    sys.exit("wrong-arch ELF files:\n  " + "\n  ".join(bad))
with open(os.path.join(out, "BUNDLE-MANIFEST.txt"), "w") as fh:
    fh.write(f"arch: {arch} ({names[want]})\nk8s-test-infra: {mk} (runtime pins: runtime/pins.env)\nFakeGPU: {fg}\n")
    fh.write(f"elf files: {elves}, all {names[want]}\n\n" + "\n".join(sorted(rows, key=lambda r: r[66:])) + "\n")
print(f"bundle ok: {elves} ELF files, all {names[want]}; {len(rows)} files hashed")
PY
echo "staged: $OUT"
