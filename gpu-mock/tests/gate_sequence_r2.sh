#!/usr/bin/env bash
# Gate sequence for next/ r2 (host-CPU spin, libcuda mapped, no score oracle, clock follows wall), run as a transient unit on the E2:
#   systemd-run --unit=gm-gate-<TAG> -p WorkingDirectory=/srv/work/gpu-mock/next-r2 bash -c 'TAG=<TAG> bash /srv/work/gpu-mock/next-r2/tests/gate_sequence_r2.sh'
# The gate tree is the frozen r2 tree (/srv/work/gpu-mock/next-r2), not the 3ec0 gate tree, and must hash to the built manifest.
# Waits for isolation's build unit p006-iso-build<TAG> to end with `mlskv exit=0` and `filter exit=0` in its
# progress file (the filter box can finish after mlskv) and for no p6-box-/gm- container; checks the main tags
# moved off the bfbd ids and the box labels name the sidecars; then, in this order:
#   0. r2's own in-image tests on the built packages box (installed patch): host CPU, oracle and clock;
#   1. the model-id known answer on the KV box (r4's direct launch, --max-examples 20);
#   2. gate_new_pair on the packages box against /srv/runs/gm-gate2-muonh-3ec0 (no recorded time moves: SHIFT=0);
#   3. the two KV known answers; 4. the in-image suite on the dataprep box; 5. the in-image suite on the filter box.
set -u
TAG=${TAG:?}; W=/srv/work/gpu-mock; L=$W/gate-$TAG-wait.log; P=/opt/computewise/gpu-mock-dev/tests/plans
G=/srv/work/gpu-mock/next-r2; ISO=/opt/computewise/isolation
built() { [ "$(systemctl is-active p006-iso-build$TAG 2>/dev/null)" != active ] && grep -qs "mlskv exit=0" /srv/build/build$TAG-*/progress 2>/dev/null && grep -qs "filter exit=0" /srv/build/build$TAG-*/progress 2>/dev/null; }
n=0; until built && ! docker ps --format "{{.Names}}" | grep -qE "^(p6-box-|gm-)"; do n=$((n+1)); [ $n -gt 2160 ] && { echo "gave up at $(date -u +%H:%M:%SZ)" >> $L; exit 3; }; sleep 20; done
id() { docker images --no-trunc --format "{{.Repository}}:{{.Tag}} {{.ID}}" | awk -v t="$1" '$1==t{print $2}' | head -1; }
lbl() { docker inspect "$1" --format '{{index .Config.Labels "p006.sidecar-image-id"}}'; }
BOX=$(id ml-node:py311-torch2110); SIDE=$(id gpu-daemon:py311-torch2110); DP=$(id ml-node:py311-torch2110-dataprep); DPS=$(id gpu-daemon:py311-torch2110-dataprep)
KV=$(id ml-node:py311-torch2110-mlskv); FILT=$(id ml-node:py311-torch2110-filter); FSIDE=$(id gpu-daemon:py311-torch2110-filter)
echo "start $(date -u +%H:%M:%SZ) BOX=$BOX SIDE=$SIDE DP=$DP KV=$KV FILT=$FILT FSIDE=$FSIDE box-label=$(lbl $BOX) filter-label=$(lbl $FILT)" >> $L
[ "$BOX" != sha256:04827d41a7e001c0cff154c9e46e80ab4871cddfb09b7c6c14e9cb8366a6ce18 ] || { echo "main tag still 3ec0, abort" >> $L; exit 4; }
[ "$FILT" != sha256:39aed8ead3cd2269172d47e3de1a19d57bcb15d8f3ce09c223cd17485dd5f920 ] || { echo "filter tag still 3ec0's, abort" >> $L; exit 4; }
[ "$(lbl $BOX)" = "$SIDE" ] || { echo "packages box sidecar label mismatch, abort" >> $L; exit 5; }
[ "$(lbl $FILT)" = "$FSIDE" ] || { echo "filter box sidecar label mismatch, abort" >> $L; exit 5; }
# the gate tree must be the built revision: its tests and daemon source are what the in-image suite runs
TREE_PATCH=$(cd $G/src/torch_patch && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -c1-16)
TREE_SIDE=$(cd $G/src && find ./daemon ./bridge ./convert -name "*.py" | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16)
MAN=$(ls -t /srv/build/manifests/ml-node_py311-torch2110-2026*.json | head -1)
MAN_PATCH=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"build_args\"][\"torch_patch_sha256\"][:16])" "$MAN" 2>/dev/null)
MAN_SIDE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))[\"build_args\"][\"sidecar_src_sha256\"][:16])" "$MAN" 2>/dev/null)
echo "gate tree patch=$TREE_PATCH sidecar=$TREE_SIDE; built manifest patch=$MAN_PATCH sidecar=$MAN_SIDE ($MAN)" >> $L
[ "$TREE_PATCH" = "$MAN_PATCH" ] && [ "$TREE_SIDE" = "$MAN_SIDE" ] || { echo "gate tree is not the built revision, abort" >> $L; exit 6; }
# 0. r2's own tests on the built packages box, installed patch (PATCH_SRC=0)
cd $G && IMG=$BOX PATCH_SRC=0 bash tests/test_host_cpu_in_image.sh > $W/gate-$TAG-hostcpu.log 2>&1; echo "hostcpu exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$BOX PATCH_SRC=0 bash tests/test_oracle_and_clock_in_image.sh > $W/gate-$TAG-oracle.log 2>&1; echo "oracle exit $? $(date -u +%H:%M:%SZ)" >> $L
# 1. model-id known answer on the KV box
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20-modelid.json --out /srv/runs/gm-mls-kv-modelid-ka-$TAG --cost-tag gm-mls-kv-modelid-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-modelid-ka-$TAG-run_one.log 2>&1; echo "modelid-ka exit $? $(date -u +%H:%M:%SZ)" >> $L
# 2. the pair gate
cd $G && BOX=$BOX SIDE=$SIDE TAG=$TAG REF=/srv/runs/gm-gate2-muonh-3ec0 SHIFT=0 SCHED=11c325bf bash tests/gate_new_pair.sh > $W/gate-$TAG.log 2>&1; echo "gate exit $? $(date -u +%H:%M:%SZ)" >> $L
# 3. KV known answers
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG --cost-tag gm-mls-kv-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-run_one.log 2>&1; echo "kv1 exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-fa-gsm8k-direct-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG-direct --cost-tag gm-mls-kv-ka-$TAG-direct --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-direct-run_one.log 2>&1; echo "kv2 exit $? $(date -u +%H:%M:%SZ)" >> $L
# 4./5. suites on the dataprep and filter boxes (installed patch)
cd $G && IMG=$DP INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-dataprep-suite.log 2>&1; echo "dataprep suite $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$FILT INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-filter-suite.log 2>&1; echo "filter suite $(date -u +%H:%M:%SZ)" >> $L
echo done >> $W/gate-$TAG.log; echo "done $(date -u +%H:%M:%SZ)" >> $L
