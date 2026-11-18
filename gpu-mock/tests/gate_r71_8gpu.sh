#!/usr/bin/env bash
# Gate for r7.1's 8-GPU box images (isolation's 11-02 --gpus 8 build of /srv/work/gpu-mock/next-r7.1), run on the E2:
#   BOX=<ml-node ...-8gpu id> KV=<ml-node ...-mlskv-8gpu id> SIDE=<its gpu-daemon id> [SCHED=<p006.scheduler prefix>] bash tests/gate_r71_8gpu.sh
# The 4-GPU r7.1 images were gated by gate_sequence_r71.sh (gm-gate-18da-2, 11-02 16:57Z). r7.1 is sidecar-only (patch = r7's).
# Checks, each a pass/fail line:
#   0  the packages box's manifest names the r7.1 tree's patch, sidecar and install; the KV box is built
#      on it with the same installed patch; both name the sidecar
#   1  in the packages box: the installed cuda_hooks.py carries 8, nvidia-smi -L lists 8, torch.cuda.device_count() is 8
#   2  the same in the KV box
#   3  a box started by run_one's entrypoint (box-init, which stages /dev/nvidia*): 8 device nodes
#   4  a job on CUDA_VISIBLE_DEVICES=7 paces on daemon GPU 7, and =8 sees no device
#   5  r7's own tests on the KV box with the INSTALLED patch: no-CUDA (4/4) and sharded memory
#   6  the sidecar carries the r7.1 daemon source; r7.1's lookup-reply test (5/5) against it
#   7  both boxes' p006.scheduler label matches SCHED (fake-slurm's ec7510c9 for this set)
set -u
G=${G:-/srv/work/gpu-mock/next-r7.1}; W=/srv/work/gpu-mock; L=$W/gate-r71-8gpu.log
BOX=${BOX:?}; KV=${KV:?}; SIDE=${SIDE:?}
pass=0; fail=0
ok()  { echo "[ok  ] $1" | tee -a $L; pass=$((pass+1)); }
bad() { echo "[FAIL] $1" | tee -a $L; fail=$((fail+1)); }
echo "start $(date -u +%H:%M:%SZ) BOX=$BOX KV=$KV SIDE=$SIDE tree=$G" >> $L
TREE_PATCH=$(cd $G/src/torch_patch && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -d" " -f1)
TREE_SIDE=$(cd $G/src && find ./daemon ./bridge ./convert -name "*.py" | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d" " -f1)
TREE_INST=$(sha256sum $G/container/install_into_image.sh | cut -d" " -f1)
# the build manifest that names this image id (build.sh writes one per image under /srv/build/manifests/)
man_for() { grep -l "${1#sha256:}" /srv/build/manifests/*.json 2>/dev/null | head -1; }
# the packages box's own manifest names the tree it was built from; the KV box is a
# layer on it (build_mlskv_layer.sh), whose manifest names only its base_image_id,
# so it is checked by that base plus the installed patch bytes being the base's
inst_patch() { docker run --rm --entrypoint sh --network none "$1" -c 'cd /usr/local/nvidia/python && ls *.py | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d" " -f1'; }
lg=$(docker inspect "$BOX" --format '{{index .Config.Labels "p006.gpus"}}' 2>/dev/null)
m=$(man_for "$BOX")
got=$( [ -n "$m" ] && python3 -c "import json,sys; b=json.load(open(sys.argv[1]))['build_args']; print(b.get('torch_patch_sha256'), b.get('sidecar_src_sha256'), b.get('gpu_mock_install_sha256'))" "$m")
[ "$got" = "$TREE_PATCH $TREE_SIDE $TREE_INST" ] && [ "$lg" = 8 ] && ok "0 $BOX: manifest's patch, sidecar and install are the r7.1 tree's; p006.gpus=8 ($m)" \
  || bad "0 $BOX: manifest '${m:-none}' has '$got' (tree $TREE_PATCH $TREE_SIDE $TREE_INST), p006.gpus '$lg'"
mk=$(man_for "$KV")
kbase=$( [ -n "$mk" ] && python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['build_args'].get('base_image_id'))" "$mk")
kg=$(docker inspect "$KV" --format '{{index .Config.Labels "p006.gpus"}}' 2>/dev/null)
pb=$(inst_patch "$BOX"); pk=$(inst_patch "$KV")
[ "$kbase" = "$BOX" ] && [ "$kg" = 8 ] && [ -n "$pb" ] && [ "$pb" = "$pk" ] \
  && ok "0 $KV: built on $BOX (its manifest's base_image_id), p006.gpus=8, installed patch identical to the base's (${pb:0:16})" \
  || bad "0 $KV: base_image_id '$kbase' (want $BOX), p006.gpus '$kg', installed patch ${pk:0:16} vs base ${pb:0:16}"
side=$(docker inspect "$BOX" --format '{{index .Config.Labels "p006.sidecar-image-id"}}' 2>/dev/null)
kside=$(docker inspect "$KV" --format '{{index .Config.Labels "p006.sidecar-image-id"}}' 2>/dev/null)
[ "$side" = "$SIDE" ] && [ "$kside" = "$SIDE" ] && ok "0 both boxes name sidecar $SIDE" || bad "0 sidecar labels: box '$side', KV '$kside', want $SIDE"
probe() {  # probe <img>: count lines
  docker run --entrypoint '' --rm --network none --cpus 2 --memory 4g "$1" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
echo "BAKED $(grep -c "_INSTALLED_DEVICE_COUNT = \"8\"" /usr/local/nvidia/python/cuda_hooks.py)"
echo "SMI $(nvidia-smi -L 2>/dev/null | grep -c "^GPU ")"
python3 -c "import torch; print(\"TORCH\", torch.cuda.device_count())"
CUDA_VISIBLE_DEVICES=7 python3 -c "import torch, replay; print(\"LAST\", torch.cuda.device_count(), replay._daemon_gpu())"
CUDA_VISIBLE_DEVICES=8 python3 -c "import torch; print(\"PAST\", torch.cuda.device_count())"' 2>&1
}
for pair in "1:$BOX" "2:$KV"; do
  n=${pair%%:*}; img=${pair#*:}; o=$(probe "$img")
  grep -q "^BAKED 1" <<<"$o" && grep -q "^SMI 8$" <<<"$o" && grep -q "^TORCH 8$" <<<"$o" \
    && ok "$n $img: patch carries 8, nvidia-smi -L 8, torch 8" || bad "$n $img: $(tr '\n' ' ' <<<"$o" | cut -c1-200)"
  [ "$n" = 1 ] && { grep -q "^LAST 1 7$" <<<"$o" && grep -q "^PAST 0$" <<<"$o" \
    && ok "4 $img: CUDA_VISIBLE_DEVICES=7 sees 1 and paces on daemon GPU 7; =8 sees none" || bad "4 $img: $(grep -E "^(LAST|PAST)" <<<"$o" | tr '\n' ' ')"; }
done
# 3: the entrypoint path: box-init stages the nodes from /etc/box-gpus
# box-init runs its arguments as the harness (output to /run/box) and then stays up, as a box does,
# so the box is started detached with a harness that ends, read with exec, and stopped
RB=$(mktemp -d /srv/work/gpu-mock/runbox.XXXX)
C=$(docker run -d --rm --network none --cap-drop ALL --cap-add MKNOD --cap-add SETUID --cap-add SETGID -v "$RB:/run/box" "$BOX" true)
o=""; for i in $(seq 60); do o=$(docker exec "$C" sh -c 'ls /dev/nvidia[0-9]* 2>/dev/null | wc -l' 2>/dev/null); [ "$o" = 8 ] && break; sleep 1; done
docker stop -t 2 "$C" >/dev/null 2>&1; rm -rf "$RB"
[ "$o" = 8 ] && ok "3 $BOX: box-init (the image entrypoint, box caps) staged 8 device nodes" || bad "3 $BOX: box-init staged '$o' device nodes"
# 5: r7's own tests on the KV box, installed patch
cd $G && IMG=$KV G=$G bash tests/test_no_cuda_in_image.sh > $W/gate-r71-8gpu-nocuda.log 2>&1
grep -q "^EMPTY rc=0" $W/gate-r71-8gpu-nocuda.log && grep -q "^GPU0 rc=0" $W/gate-r71-8gpu-nocuda.log \
  && grep -q "no_visible_cuda_device ''" $W/gate-r71-8gpu-nocuda.log \
  && ok "5 no-CUDA on the 8-GPU KV box (installed patch)" || bad "5 no-CUDA: see $W/gate-r71-8gpu-nocuda.log"
cd $G && IMG=$KV G=$G PATCH_SRC=0 bash tests/test_sharded_memory_in_image.sh > $W/gate-r71-8gpu-sharded.log 2>&1
grep -q "^W1 rc=1" $W/gate-r71-8gpu-sharded.log && grep -q "^DONE_W4 4 DONE_W2 2 DONE_P 1" $W/gate-r71-8gpu-sharded.log \
  && ok "5 sharded memory on the 8-GPU KV box" || bad "5 sharded: see $W/gate-r71-8gpu-sharded.log"
# r7.1's additions: the sidecar image carries the r7.1 daemon source, r7.1's own lookup-reply test against it,
# and the scheduler label isolation named for this set (SCHED, a prefix of p006.scheduler)
SD=$(mktemp -d $W/r71side8.XXXX); docker run --rm --entrypoint sh --network none -v $SD:/out $SIDE -c 'cp -a /opt/gpu-daemon/src/. /out/' 2>/dev/null
ss=$(cd $SD && find ./daemon ./bridge ./convert -name "*.py" | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -d" " -f1)
[ "$ss" = "$TREE_SIDE" ] && ok "6 $SIDE: the daemon source it carries is the r7.1 tree's (${ss:0:16})" || bad "6 $SIDE: carries daemon source ${ss:0:16}, tree ${TREE_SIDE:0:16}"
cp -a $G/src/torch_patch $SD/torch_patch 2>/dev/null
cd $G && REG=/opt/computewise/gpu-tasks/registry/records python3 -I tests/test_lookup_no_results_daemon.py $SD > $W/gate-r71-8gpu-lookup.log 2>&1
grep -q "5 passed, 0 failed" $W/gate-r71-8gpu-lookup.log && ok "6 lookup reply carries no result, on the 8-GPU sidecar's daemon ($(tail -1 $W/gate-r71-8gpu-lookup.log))" || bad "6 lookup: $(tail -1 $W/gate-r71-8gpu-lookup.log)"
rm -rf $SD
SCHED=${SCHED:-}
for img in $BOX $KV; do
  l=$(docker image inspect -f '{{ index .Config.Labels "p006.scheduler" }}' "$img" 2>/dev/null)
  [ -n "$SCHED" ] && case "$l" in "$SCHED"*) ok "7 $img: p006.scheduler=$l matches SCHED=$SCHED";; *) bad "7 $img: p006.scheduler '$l' does not match SCHED=$SCHED";; esac
done
echo "$pass passed, $fail failed  done $(date -u +%H:%M:%SZ)" | tee -a $L
[ "$fail" = 0 ]
