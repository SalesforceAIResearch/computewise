#!/usr/bin/env bash
# PROBE (not a gate): does the r8 torch patch work on the TTC source stack (torch
# 2.9.1 + TabArena), and does the whole_job import-time seam replace
# ExperimentRunner._run before it runs? The TTC image carries NO gpu-mock; the r8
# patch is mounted. Three questions, reported separately, each pass / FAIL / NOT
# REACHED (a failure stops the later ones):
#   1  the patch applies on torch 2.9.1: `import torch` then the sitecustomize
#      hook loads cuda_hooks with no exception, and torch.cuda.device_count()==4;
#   2  the seam installs before _run: with a whole_job record for single_cell.py's
#      hash and seam ExperimentRunner._run, the daemon logs the seam engaged
#      (whole_job early probe) before single_cell.py's own code runs _run;
#   3  _run returns the recorded dict: single_cell.py's result.json holds the
#      record's recorded metric_error/time fields, not what a real _run computed.
# Stand-in note: until Sol's record overlay lands, step 3 uses a stand-in record
# keyed to the real single_cell.py sha256 (ce2a162f...) with a chosen metric; it
# proves the seam path, NOT TabArena's real numbers. Labelled STANDIN in output.
# Usage (on c3-15):  sudo bash tests/probe_ttc_torch291.sh
set -uo pipefail
IMG=sha256:493603b6fe4c88afa9c0bec682ad34a4a139f1378584049e464cf69638d762bd
G=/srv/work/gpu-mock/next-r8-probe
FAKEGPU=/srv/work/gpu-mock/FakeGPU
REALREG=/srv/work/sol-independent/ttc-probe/ttc-reg/records
CACHE=/srv/work/sol-independent/ttc-probe/openml-cache
SC_SHA=ce2a162fe033e46bf488a4f00ef71c970ca8595eadd0309c6ad528fef3977353
PYSITE=$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
W=$(mktemp -d /srv/work/gpu-mock/ttcprobe.XXXX)
mkdir -p "$W/cache"; cp -a "$CACHE/." "$W/cache/" 2>/dev/null || true
# cp -a preserved the source read-only perms; single_cell.py rewrites the
# OpenML .pkl cache, so make the copy writable (the earlier probe hit
# [Errno 30] Read-only file system on dataset_46912.pkl).
chmod -R u+w "$W/cache" 2>/dev/null || true
echo "=== TTC torch-2.9.1 probe $(date -u +%FT%TZ)  image $IMG  patch $(cd $G/src/torch_patch && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)"
# The whole_job record for single_cell.py, seam = ExperimentRunner._run. Prefer the
# REAL overlay record (Sol's generate_records.py output); its blockers make it
# readiness 0, so FOR THE PROBE ONLY we force replay=True (labelled, NOT a replay-ready
# claim). If the overlay isn't present, a stand-in keyed to single_cell.py's hash proves
# the seam path with a chosen value (labelled STANDIN), not TabArena's numbers.
mkdir -p "$W/reg"
REALREC="$REALREG/sol-ttc-tabpfn3-aggregation-96-bioresponse-split0.json"
python3 /srv/work/gpu-mock/next-r8-probe/probe_mkrec.py "$W/reg/ttc.json" "$SC_SHA" "$REALREC"
# PREP: inside the image, vendor FakeGPU as `nvcompat` exactly as install_into_image.sh's
# core steps do (pip install, move to nvcompat, rewrite its three self-imports), so the
# patch's `from nvcompat import ...` resolves. The native-binary rename and docstring
# stripping are install cosmetics that do not affect whether the patch APPLIES on torch 2.9.1.
PREP='set -e
# Vendor nvcompat OFFLINE by copying the pure-Python package tree, exactly as
# install_into_image.sh produces as its RESULT (it pip-builds then copies the
# fakegpu/ dir to pylib/nvcompat and rewrites self-imports). The native cmake
# .so build is NOT needed: the patch imports only nvcompat._upstream /
# torch_patch / _allocator (pure .py, no native load at import), and replay
# intercepts CUDA before the driver libs are ever called. Copying the tree is
# faithful to the shipped layout and works under --network none.
SRC=/gm/FakeGPU/fakegpu
if [ -d "$SRC" ]; then
  mkdir -p /gm/pylib; rm -rf /gm/pylib/nvcompat; cp -r "$SRC" /gm/pylib/nvcompat
  find /gm/pylib/nvcompat -name __pycache__ -type d -prune -exec rm -rf {} +
  sed -i -E "s/^(\\s*)import fakegpu\\$/\\1import nvcompat as fakegpu/; s/^(\\s*)from fakegpu import/\\1from nvcompat import/" /gm/pylib/nvcompat/*.py
  python3 -c "import nvcompat, nvcompat._upstream, nvcompat.torch_patch, nvcompat._allocator; print(\"NVCOMPAT_VENDORED\", nvcompat.__file__)" 2>&1 | tail -1
else echo FAKEGPU_SRC_MISSING "$SRC"; fi'

run() {  # run <tag> <extra docker args...> -- <cmd...>
  docker run --entrypoint '' --rm --cpus 8 --memory 32g --network none \
    -v "$G/src:/gm/src:ro" -v "$FAKEGPU:/gm/FakeGPU:ro" \
    -v "$G/src/torch_patch:/usr/local/nvidia/python:ro" \
    -v "$G/src/torch_patch/sitecustomize.py:$PYSITE/sitecustomize.py:ro" \
    -v "$W:/w" -v "$W/cache:/opt/cache/openml" \
    -e NVRT_SOCKET=/w/gm.sock -e OPENML_CACHE_DIR=/opt/cache/openml -e NVRT_DEBUG=1 \
    -e FAKEGPU_PROFILE_DIR=/gm/FakeGPU/profiles -e NVRT_PROFILE=h100 \
    -e PYTHONPATH=/gm/pylib "$IMG" bash -c "$PREP; $1"
}
# 1: the patch applies on torch 2.9.1
echo "--- 1. patch applies on torch 2.9.1"
run 'source /usr/local/nvidia/env.sh 2>/dev/null || true; python3 -c "
import torch; print(\"TORCH_VERSION\", torch.__version__)
import cuda_hooks  # applies on import
print(\"DEVICE_COUNT\", torch.cuda.device_count(), \"AVAIL\", torch.cuda.is_available())
print(\"NAME\", torch.cuda.get_device_name(0) if torch.cuda.device_count() else \"none\")
"' 2>&1 | grep -E "TORCH_VERSION|DEVICE_COUNT|NAME|Error|Traceback|nvrt" | head -12
# 2 and 3: the daemon, then single_cell.py; the early probe should engage the seam
echo "--- 2,3. seam-before-_run and recorded dict (STANDIN record)"
run 'source /usr/local/nvidia/env.sh 2>/dev/null || true
python3 -I /gm/src/daemon/gpu_daemon.py --socket /w/gm.sock --registry /w/reg --gpus 4 --speed 50 --occupancy /w/occ.jsonl > /w/daemon.log 2>&1 &
for i in $(seq 100); do [ -S /w/gm.sock ] && break; sleep 0.1; done
cd /opt/sol-ttc 2>/dev/null || cd /
timeout 1200 python3 /opt/sol-ttc/single_cell.py --dataset Bioresponse --split 0 --arm tabpfn3_aggregation_96 --output /w/result.json > /w/sc.log 2>&1
echo "SINGLE_CELL rc=$?"
echo "--- result.json:"; cat /w/result.json 2>/dev/null | head -c 400; echo
echo "--- daemon whole_job events:"; grep -oE "whole_job[a-z_]*|early probe|engaging before the seam|replay engaged" /w/daemon.log | sort | uniq -c
echo "--- single_cell tail:"; tail -5 /w/sc.log | cut -c1-160' 2>&1 | grep -vE "^nvrt replay|^$" | head -40
rm -rf "$W"
echo "=== probe done $(date -u +%FT%TZ)"
