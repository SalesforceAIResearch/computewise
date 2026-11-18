#!/usr/bin/env bash
# Install gpu-mock into an image, as the agent will meet it. Everything lands
# under PREFIX (default /usr/local/nvidia, where NVIDIA's own CUDA images keep
# driver libraries), with no name an agent can read as a mock:
#   PREFIX/lib/        Mokka libnvidia-ml.so*, libcuda.so*, libcudart.so* (from the bundle);
#                      the NVML library's environment-variable names are rewritten in
#                      place, MOCK_* -> HOST_* (same length), so neither `env` nor
#                      `strings` shows them
#   PREFIX/sbin/nvidia-smi   the genuine NVIDIA binary (RPATH ../lib), not on PATH
#   /usr/local/bin/nvidia-smi  a wrapper that sets HOST_NVML_CONFIG/HOST_NVML_OVERRIDES for
#                      that one process and execs PREFIX/sbin/nvidia-smi: the agent's shell
#                      carries no NVML variable at all
#   PREFIX/etc/nvml.yaml       the H100 profile
#   PREFIX/python/     the torch layer (cuda_hooks, replay, daemon client …)
#   PREFIX/pylib/nvcompat/     FakeGPU, vendored under a neutral name, beside python/ so a
#                      dev copy of the patch mounted over python/ does not hide it; its
#                      dist-info and console script are removed, so `pip list` shows nothing
#   <site-packages>/sitecustomize.py  applies the patch when a process imports torch; it
#                      carries PREFIX/python baked in, so no PYTHONPATH is needed
#   PREFIX/env.sh      the runtime environment: LD_LIBRARY_PATH only
# Dropped on 2026-10-21 (the env-name scrub): libmockfs (layer B, never enabled in the
# box), nvml-mock-ctl, the mock-pci tree (libmockfs's), and every GPU_MOCK_*/MOCK_*
# variable from the agent's environment. Patch variables are NVRT_* and the box sets
# none: the socket default is /var/run/nvidia-persistenced/socket, the device count
# default 4, and the speed knob comes from the daemon's hello reply.
#
# Usage: install_into_image.sh --bundle <dir with lib/ and bin/nvidia-smi.real> \
#          --config <profile.yaml> --src <torch_patch dir> [--prefix /usr/local/nvidia] \
#          [--fakegpu <FakeGPU checkout to pip install first>] [--python python3] [--mock-pci <ignored>]
# Runtime needs: the shared mount holding the daemon socket at
# /var/run/nvidia-persistenced/socket and the bridge's overrides.yaml beside it.
set -euo pipefail
PREFIX=/usr/local/nvidia; BUNDLE=""; CONFIG=""; SRC=""; FAKEGPU=""; PYTHON=python3
while [ $# -gt 0 ]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2;;
    --bundle) BUNDLE="$2"; shift 2;;
    --config) CONFIG="$2"; shift 2;;
    --mock-pci) shift 2;;            # accepted for older Dockerfiles; nothing uses it now
    --src) SRC="$2"; shift 2;;
    --fakegpu) FAKEGPU="$2"; shift 2;;
    --python) PYTHON="$2"; shift 2;;
    *) echo "unknown arg $1" >&2; exit 2;;
  esac
done
for v in BUNDLE CONFIG SRC; do [ -n "${!v}" ] || { echo "--${v,,} is required" >&2; exit 2; }; done
PYSITE=$("$PYTHON" -c 'import site; print(site.getsitepackages()[0])')

mkdir -p "$PREFIX"/{lib,lib64,sbin,etc,python,pylib}
cp -a "$BUNDLE"/lib/. "$PREFIX/lib/"
rm -f "$PREFIX"/lib/libmockfs* "$PREFIX"/lib/*.h
rm -rf "$PREFIX"/lib/proc "$PREFIX"/lib/sys
# the NVML library's variable names, in place and at the same length
"$PYTHON" - "$PREFIX/lib" <<'PY'
import os, sys
d = sys.argv[1]
for n in os.listdir(d):
    p = os.path.join(d, n)
    if os.path.islink(p) or not os.path.isfile(p):
        continue
    with open(p, "rb") as f:
        b = f.read()
    if b[:4] != b"\x7fELF" or b"MOCK_" not in b:
        continue
    nb = b.replace(b"MOCK_", b"HOST_")
    assert len(nb) == len(b)
    with open(p, "wb") as f:
        f.write(nb)
    print(f"  renamed {b.count(b'MOCK_')} variable names in {n}")
PY
if [ -f "$BUNDLE/bin/nvidia-smi.real" ]; then install -m 0755 "$BUNDLE/bin/nvidia-smi.real" "$PREFIX/sbin/nvidia-smi"
elif [ -f "$BUNDLE/sbin/nvidia-smi" ]; then install -m 0755 "$BUNDLE/sbin/nvidia-smi" "$PREFIX/sbin/nvidia-smi"
else echo "no nvidia-smi binary in the bundle" >&2; exit 2; fi
install -m 0644 "$CONFIG" "$PREFIX/etc/nvml.yaml"
# the box's GPU count is the profile's device count, written into the torch layer
# below, so torch.cuda.device_count() and nvidia-smi -L always agree
NGPU=$(grep -cE '^  - index:[[:space:]]*[0-9]+' "$CONFIG")
[ "$NGPU" -ge 1 ] || { echo "install_into_image: no devices in $CONFIG" >&2; exit 2; }
rm -f /usr/local/bin/nvidia-smi
cat > /usr/local/bin/nvidia-smi <<W
#!/bin/sh
# NVIDIA System Management Interface
export HOST_NVML_CONFIG="$PREFIX/etc/nvml.yaml"
export HOST_NVML_OVERRIDES="/var/run/nvidia-persistenced/overrides.yaml"
export LD_LIBRARY_PATH="$PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
exec "$PREFIX/sbin/nvidia-smi" "\$@"
W
chmod 0755 /usr/local/bin/nvidia-smi

# the torch layer, installed without its comments and docstrings (strip_comments.py
# beside this script checks each copy is the same program); the repository and the
# release archive keep the commented source. NVRT_KEEP_COMMENTS=1 installs it as is.
HERE=$(cd "$(dirname "$0")" && pwd)
STRIP="$HERE/strip_comments.py"
if [ "${NVRT_KEEP_COMMENTS:-0}" != 1 ] && [ ! -f "$STRIP" ]; then
  echo "install_into_image: $STRIP missing (stage_bundle.sh ships it); refusing to install the commented source" >&2; exit 3
fi
_install_py() {   # <src.py> <dst.py>
  if [ "${NVRT_KEEP_COMMENTS:-0}" != 1 ]; then
    "$PYTHON" -I "$STRIP" "$1" "$2" || { echo "install_into_image: strip failed for $1" >&2; exit 3; }
  else
    cp "$1" "$2"
  fi
}
rm -rf "$PREFIX/python"; mkdir -p "$PREFIX/python"
for f in "$SRC"/*.py; do
  [ "$(basename "$f")" = sitecustomize.py ] && continue
  if [ "$(basename "$f")" = cuda_hooks.py ]; then
    _TMP_CH=$(mktemp); sed "s|__NVRT_DEVICE_COUNT__|$NGPU|" "$f" > "$_TMP_CH.py"
    grep -q "_INSTALLED_DEVICE_COUNT = \"$NGPU\"" "$_TMP_CH.py" || { echo "install_into_image: device count not written into cuda_hooks.py" >&2; exit 3; }
    _install_py "$_TMP_CH.py" "$PREFIX/python/cuda_hooks.py"; rm -f "$_TMP_CH" "$_TMP_CH.py"
    continue
  fi
  _install_py "$f" "$PREFIX/python/$(basename "$f")"
done
chmod 0644 "$PREFIX/python/"*.py
_TMP_SC=$(mktemp)
sed "s|__NVRT_PYTHON_DIR__|$PREFIX/python|" "$SRC/sitecustomize.py" > "$_TMP_SC.py"
_install_py "$_TMP_SC.py" "$PYSITE/sitecustomize.py"
rm -f "$_TMP_SC" "$_TMP_SC.py"
chmod 0644 "$PYSITE/sitecustomize.py"

# FakeGPU: install if a checkout is given, then vendor whatever is installed
if [ -n "$FAKEGPU" ]; then
  "$PYTHON" -m pip install --no-cache-dir "$FAKEGPU" >/dev/null
fi
FG=$("$PYTHON" -c 'import fakegpu, os; print(os.path.dirname(fakegpu.__file__))' 2>/dev/null || true)
if [ -n "$FG" ]; then
  rm -rf "$PREFIX/pylib"; mkdir -p "$PREFIX/pylib"
  mv "$FG" "$PREFIX/pylib/nvcompat"
  find "$PREFIX/pylib/nvcompat" -name __pycache__ -type d -prune -exec rm -rf {} +
  # its three absolute self-imports
  sed -i -E 's/^(\s*)import fakegpu$/\1import nvcompat as fakegpu/; s/^(\s*)from fakegpu import/\1from nvcompat import/' "$PREFIX"/pylib/nvcompat/*.py
  # its native helper binary kept FakeGPU's name (isolation's sweep, 2026-10-21):
  # rename it and the one CLI module that names it (a path the patch never uses)
  if [ -f "$PREFIX/pylib/nvcompat/_native/fakegpu-the study" ]; then
    mv "$PREFIX/pylib/nvcompat/_native/fakegpu-the study" "$PREFIX/pylib/nvcompat/_native/nvidia-the study"
    sed -i 's/fakegpu-the study/nvidia-the study/g' "$PREFIX"/pylib/nvcompat/distributed_cli.py
  fi
  rm -rf "$PYSITE"/fakegpu-*.dist-info "$PYSITE"/fakegpu
  rm -f /usr/local/bin/fakegpu "$(dirname "$(command -v "$PYTHON")")/fakegpu" 2>/dev/null || true
  # its own comments and docstrings too: they describe a fake GPU (98 of its 119
  # modules, 2026-10-28). Only serving_benchmark.py reads its __doc__ (an argparse
  # description), a CLI module the patch never runs. The MIT notice travels with the
  # release archive (docs/RELEASE-NOTE.md), not inside the box.
  n_fg=0
  while IFS= read -r f; do
    _install_py "$f" "$f.stripped" && mv "$f.stripped" "$f" && n_fg=$((n_fg + 1))
  done < <(find "$PREFIX/pylib/nvcompat" -name "*.py" -type f)
  echo "  FakeGPU vendored as nvcompat; dist-info and console script removed; $n_fg modules stripped"
fi

# runtime environment: what NVIDIA's CUDA images export, nothing else
cat > "$PREFIX/env.sh" <<ENV
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV

echo "installed under $PREFIX (python site: $PYSITE), $NGPU GPUs"
echo "  nvidia-smi -> /usr/local/bin/nvidia-smi (wrapper) -> $PREFIX/sbin/nvidia-smi"
echo "  source $PREFIX/env.sh at runtime; run stage_dev_nodes.sh once at entrypoint (CAP_MKNOD)"
