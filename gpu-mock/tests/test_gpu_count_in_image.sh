#!/usr/bin/env bash
# next/ (after r6): a box's GPU count is its NVML profile's, everywhere. The tree's
# install_into_image.sh is run inside a box image against a profile rendered for
# N GPUs (build_bundle.sh's own renderer), then for N = 8 and N = 4:
#   1  the installed cuda_hooks.py carries N, and no placeholder is left;
#   2  torch.cuda.device_count() is N, and nvidia-smi -L lists N GPUs;
#   3  CUDA_VISIBLE_DEVICES=N-1 sees one device (the last card is a real one);
#      =N sees none, as a driver would;
#   4  a job on the last card paces on daemon GPU N-1, in the launch shapes
#      tests/test_gpu_index_in_image.sh covers (CVD alone, a CVD pair with
#      LOCAL_RANK, LOCAL_RANK alone).
# N = 4 is the control: every 4-GPU box serves as before.
# Usage (on the E2):  sudo IMG=sha256:<box> G=<tree> UP=<upstream dir> bash tests/test_gpu_count_in_image.sh
set -uo pipefail
IMG=${IMG:-sha256:5994258c9617cee5b4db7566f93afbd1c020d2e480b5ce4d120f0969b07f39ee}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
UP=${UP:-/srv/work/gpu-mock/upstream}
SRC_YAML="$UP/k8s-test-infra/pkg/gpu/mocknvml/configs/mock-nvml-config-h100.yaml"
W=$(mktemp -d /srv/work/gpu-mock/gpucount.XXXX)
pass=0; fail=0
ok()  { echo "[ok  ] $1"; pass=$((pass+1)); }
bad() { echo "[FAIL] $1"; fail=$((fail+1)); }
for N in 8 4; do
  # the renderer build_bundle.sh uses, verbatim
  python3 -I - "$SRC_YAML" "$W/h100-${N}gpu.yaml" "$N" <<'PY'
import re, sys
src_path, out_path, gpus = sys.argv[1], sys.argv[2], int(sys.argv[3])
lines = open(src_path).read().splitlines(keepends=True)
out, in_devices, keep = [], False, True
for ln in lines:
    if re.match(r'^devices:\s*$', ln):
        in_devices = True; out.append(ln); continue
    if in_devices:
        m = re.match(r'^  - index:\s*(\d+)', ln)
        if m:
            keep = int(m.group(1)) < gpus
        if keep:
            out.append(ln)
    else:
        out.append(ln)
open(out_path, "w").write("".join(out))
PY
  o=$(docker run --entrypoint '' --rm --cpus 4 --memory 8g --network none -e N="$N" \
        -v "$G:/gm:ro" -v "$W:/w:ro" "$IMG" bash -c '
set -u
source /usr/local/nvidia/env.sh 2>/dev/null || true
# reinstall the torch layer and the profile as a build would, from the tree under test
PYSITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
# the installed driver libraries and nvidia-smi of the image stand in for the bundle
mkdir -p /tmp/b/bin /tmp/b/lib && cp -a /usr/local/nvidia/lib/. /tmp/b/lib/ && cp /usr/local/nvidia/sbin/nvidia-smi /tmp/b/bin/nvidia-smi.real
bash /gm/container/install_into_image.sh --prefix /usr/local/nvidia --bundle /tmp/b --config /w/h100-${N}gpu.yaml \
  --src /gm/src/torch_patch --python python3 > /tmp/install.log 2>&1 || { echo "INSTALL_FAILED"; tail -5 /tmp/install.log; exit 0; }
echo "INSTALL $(grep -m1 "^installed under" /tmp/install.log)"
echo "BAKED $(grep -c "_INSTALLED_DEVICE_COUNT = \"$N\"" /usr/local/nvidia/python/cuda_hooks.py) PLACEHOLDER $(grep -c __NVRT_DEVICE_COUNT__ /usr/local/nvidia/python/cuda_hooks.py)"
echo "SMI $(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")"
python3 -c "import torch; print(\"TORCH\", torch.cuda.device_count(), torch.cuda.is_available())"
CUDA_VISIBLE_DEVICES=$((N-1)) python3 -c "import torch; print(\"LAST\", torch.cuda.device_count())"
CUDA_VISIBLE_DEVICES=$N python3 -c "import torch; print(\"PAST\", torch.cuda.device_count(), torch.cuda.is_available())"
# the daemon GPU a replayed job paces on (replay._daemon_gpu, as tests/test_gpu_index_in_image.sh reads it)
P1=$(CUDA_VISIBLE_DEVICES=$((N-1)) python3 -c "import torch, replay; print(replay._daemon_gpu())" 2>/dev/null | tail -1)
P2=$(CUDA_VISIBLE_DEVICES=$((N-2)),$((N-1)) LOCAL_RANK=1 python3 -c "import torch, replay; print(replay._daemon_gpu())" 2>/dev/null | tail -1)
P3=$(LOCAL_RANK=$((N-1)) python3 -c "import torch, replay; print(replay._daemon_gpu())" 2>/dev/null | tail -1)
echo "PACED $P1 $P2 $P3"
' 2>&1)
  echo "--- N=$N"; echo "$o" | grep -E "^(INSTALL|BAKED|SMI|TORCH|LAST|PAST|PACED)" | cut -c1-200
  grep -q "^INSTALL installed under .*, $N GPUs" <<<"$o" && ok "N=$N: the install reads $N GPUs from the profile" || bad "N=$N: install: $(grep -m1 -E "INSTALL" <<<"$o")"
  grep -q "^BAKED 1 PLACEHOLDER 0" <<<"$o" && ok "N=$N: cuda_hooks.py carries $N, no placeholder left" || bad "N=$N: $(grep -m1 BAKED <<<"$o")"
  grep -q "^SMI $N\$" <<<"$o" && ok "N=$N: nvidia-smi -L lists $N" || bad "N=$N: $(grep -m1 SMI <<<"$o")"
  grep -q "^TORCH $N True" <<<"$o" && ok "N=$N: torch.cuda.device_count() is $N" || bad "N=$N: $(grep -m1 TORCH <<<"$o")"
  grep -q "^LAST 1\$" <<<"$o" && ok "N=$N: CUDA_VISIBLE_DEVICES=$((N-1)) sees 1 device" || bad "N=$N: $(grep -m1 LAST <<<"$o")"
  grep -q "^PAST 0 False" <<<"$o" && ok "N=$N: CUDA_VISIBLE_DEVICES=$N sees none" || bad "N=$N: $(grep -m1 PAST <<<"$o")"
  grep -q "^PACED $((N-1)) $((N-1)) $((N-1))$" <<<"$o" && ok "N=$N: the last card paces on daemon GPU $((N-1)) (CVD alone, CVD pair with LOCAL_RANK=1, LOCAL_RANK alone)" || bad "N=$N: $(grep -m1 PACED <<<"$o")"
done
rm -rf "$W"
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]
