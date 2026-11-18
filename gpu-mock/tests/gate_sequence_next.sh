#!/usr/bin/env bash
# Gate sequence for the explicit-default-flag revision (patch + daemon; fix:gpumock-explicit-default-flag), lean for H5's critical path, run as a transient unit on the E2:
#   systemd-run --unit=gm-gate-<TAG> -p WorkingDirectory=/opt/computewise/gpu-mock bash -c 'TAG=<TAG> bash /srv/work/gpu-mock/gate_sequence_next.sh'
# Waits for isolation's build unit p006-iso-build<TAG> to end with `mlskv exit=0` and `filter exit=0` in its
# progress file (the filter box can finish after mlskv) and for no p6-box-/gm- container; checks the main tags
# moved off the bfbd ids and the box labels name the sidecars; then, in this order:
#   1. the model-id known answer on the KV box (r4's direct launch, --max-examples 20);
#   2. gate_new_pair on the packages box against /srv/runs/gm-gate2-muonh-37c9;
#   3. the two KV known answers; 4. the in-image suite on the dataprep box; 5. the in-image suite on the filter box.
set -u
TAG=${TAG:?}; W=/srv/work/gpu-mock; L=$W/gate-$TAG-wait.log; P=/opt/computewise/gpu-mock-dev/tests/plans
G=/opt/computewise/gpu-mock; ISO=/opt/computewise/isolation
built() { [ "$(systemctl is-active p006-iso-build$TAG 2>/dev/null)" != active ] && grep -qs "mlskv exit=0" /srv/build/build$TAG-*/progress 2>/dev/null && grep -qs "filter exit=0" /srv/build/build$TAG-*/progress 2>/dev/null; }
n=0; until built && ! docker ps --format "{{.Names}}" | grep -qE "^(p6-box-|gm-)"; do n=$((n+1)); [ $n -gt 2160 ] && { echo "gave up at $(date -u +%H:%M:%SZ)" >> $L; exit 3; }; sleep 20; done
id() { docker images --no-trunc --format "{{.Repository}}:{{.Tag}} {{.ID}}" | awk -v t="$1" '$1==t{print $2}' | head -1; }
lbl() { docker inspect "$1" --format '{{index .Config.Labels "p006.sidecar-image-id"}}'; }
BOX=$(id ml-node:py311-torch2110); SIDE=$(id gpu-daemon:py311-torch2110); DP=$(id ml-node:py311-torch2110-dataprep); DPS=$(id gpu-daemon:py311-torch2110-dataprep)
KV=$(id ml-node:py311-torch2110-mlskv); FILT=$(id ml-node:py311-torch2110-filter); FSIDE=$(id gpu-daemon:py311-torch2110-filter)
echo "start $(date -u +%H:%M:%SZ) BOX=$BOX SIDE=$SIDE DP=$DP KV=$KV FILT=$FILT FSIDE=$FSIDE box-label=$(lbl $BOX) filter-label=$(lbl $FILT)" >> $L
[ "$BOX" != sha256:93fc30fae3aadfc5d7eb2ee5a44e8fa7a4c581eefa7a51402079f72496411d3f ] || { echo "main tag still bfbd, abort" >> $L; exit 4; }
[ "$FILT" != sha256:552984a90a3dd4f66f9b24f2891b146a91126acd2097fb0973852381a391e4f2 ] || { echo "filter tag still bfbd's, abort" >> $L; exit 4; }
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
# 1. model-id known answer on the KV box (the rule this revision changes)
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20-modelid.json --out /srv/runs/gm-mls-kv-modelid-ka-$TAG --cost-tag gm-mls-kv-modelid-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-modelid-ka-$TAG-run_one.log 2>&1; echo "modelid-ka exit $? $(date -u +%H:%M:%SZ)" >> $L
# 2. the pair gate
cd $G && BOX=$BOX SIDE=$SIDE TAG=$TAG REF=/srv/runs/gm-gate2-muonh-bfbd SHIFT=0 SCHED=11c325bf bash tests/gate_new_pair.sh > $W/gate-$TAG.log 2>&1; echo "gate exit $? $(date -u +%H:%M:%SZ)" >> $L
# 3. KV known answers
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG --cost-tag gm-mls-kv-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-run_one.log 2>&1; echo "kv1 exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-fa-gsm8k-direct-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG-direct --cost-tag gm-mls-kv-ka-$TAG-direct --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-direct-run_one.log 2>&1; echo "kv2 exit $? $(date -u +%H:%M:%SZ)" >> $L
# 4./5. suites on the dataprep and filter boxes (installed patch)
cd $G && IMG=$DP INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-dataprep-suite.log 2>&1; echo "dataprep suite $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$FILT INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-filter-suite.log 2>&1; echo "filter suite $(date -u +%H:%M:%SZ)" >> $L
echo done >> $W/gate-$TAG.log; echo "done $(date -u +%H:%M:%SZ)" >> $L
