#!/usr/bin/env bash
# Gate sequence for revision 81f76686 (H4 chdir fix + uid fix), run as a transient unit on the E2:
#   systemd-run --unit=gm-gate-<TAG> -p WorkingDirectory=/opt/computewise/gpu-mock bash -c 'TAG=<TAG> bash /srv/work/gpu-mock/gate_sequence_81f76686.sh'
# Waits for isolation's build unit p006-iso-build<TAG> to end with `mlskv exit=0` and `filter exit=0` in its
# progress file (the filter box can finish after mlskv) and for no p6-box-/gm- container; checks the main tags
# moved off the 37c9 ids and the box labels name the sidecars; then, in this order:
#   1. the H4 known answer through run_one on the NEW filter box (the item H4 is held on);
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
[ "$BOX" != sha256:c139c5c91598acee9efdb77c5a528b50c87fd293e142205e90121fe84e87bcb9 ] || { echo "main tag still 37c9, abort" >> $L; exit 4; }
[ "$FILT" != sha256:b5890d2b96782b5220d8d7bae5ed009f853a4106c7c61809ddc986629c87c405 ] || { echo "filter tag still 37c9's, abort" >> $L; exit 4; }
[ "$(lbl $BOX)" = "$SIDE" ] || { echo "packages box sidecar label mismatch, abort" >> $L; exit 5; }
[ "$(lbl $FILT)" = "$FSIDE" ] || { echo "filter box sidecar label mismatch, abort" >> $L; exit 5; }
# 1. H4 known answer on the new filter box (speed 1, as W0-h4-measure ran)
cd $ISO && python3 -I run_one.py --task pkg-h4-filter-ablation --harness scripted --plan $P/h4-none-c4-seed0.json --out /srv/runs/gm-h4-ka-$TAG --cost-tag gm-h4-ka-$TAG --speed 1 --after-deadline continue --image $FILT --sidecar-image $FSIDE > $W/h4-ka-$TAG-run_one.log 2>&1; echo "h4-ka exit $? $(date -u +%H:%M:%SZ)" >> $L
# 2. the pair gate
cd $G && BOX=$BOX SIDE=$SIDE TAG=$TAG REF=/srv/runs/gm-gate2-muonh-37c9 SHIFT=0 SCHED=11c325bf bash tests/gate_new_pair.sh > $W/gate-$TAG.log 2>&1; echo "gate exit $? $(date -u +%H:%M:%SZ)" >> $L
# 3. KV known answers
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG --cost-tag gm-mls-kv-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-run_one.log 2>&1; echo "kv1 exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-fa-gsm8k-direct-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG-direct --cost-tag gm-mls-kv-ka-$TAG-direct --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-direct-run_one.log 2>&1; echo "kv2 exit $? $(date -u +%H:%M:%SZ)" >> $L
# 4./5. suites on the dataprep and filter boxes (installed patch)
cd $G && IMG=$DP INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-dataprep-suite.log 2>&1; echo "dataprep suite $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$FILT INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-filter-suite.log 2>&1; echo "filter suite $(date -u +%H:%M:%SZ)" >> $L
echo done >> $W/gate-$TAG.log; echo "done $(date -u +%H:%M:%SZ)" >> $L
