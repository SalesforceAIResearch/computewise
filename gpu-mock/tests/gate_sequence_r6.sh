#!/usr/bin/env bash
# Gate sequence for next/ r6 (measured per-example scores for each KV record's first k; r4 underneath), run as a transient unit on the E2:
#   systemd-run --unit=gm-gate-<TAG> -p WorkingDirectory=/srv/work/gpu-mock/next-r6 bash -c 'TAG=<TAG> bash /srv/work/gpu-mock/next-r6/tests/gate_sequence_r6.sh'
# The gate tree is the frozen r6 tree (/srv/work/gpu-mock/next-r6), not the 3ec0 gate tree, and must hash to the built manifest.
# r6's own test comes first (0r6): the measured-scores known answer on the built KV box with the INSTALLED patch.
# r4's own tests come first (0a): the module-entry shard tests on the built KV box with the INSTALLED patch, the stand-in
# and the real evaluators with gpu-tasks' order tables ($ORDER, the survey copy until the tables are staged in records/).
# Waits for isolation's build unit p006-iso-build<TAG> to end with `mlskv exit=0` and `filter exit=0` in its
# progress file (the filter box can finish after mlskv) and for no p6-box-/gm- container; checks the main tags
# moved off the bfbd ids and the box labels name the sidecars; then, in this order:
#   0r6. r6's: measured scores served by every launch shape, on the built KV box (installed patch);
#   0a. r4's: shard serving on the stand-in and on the real evaluators, on the built KV box (installed patch);
#   0. r3's in-image tests on the built packages box (installed patch): installed text, exec'd evaluator, host CPU, oracle and clock;
#   1. the model-id known answer on the KV box (r4's direct launch, --max-examples 20);
#   2. gate_new_pair on the packages box against r4's /srv/runs/gm-gate2-muonh-<r4 tag> (no recorded time moves: SHIFT=0);
#   3. the two KV known answers; 4. the in-image suite on the dataprep box; 5. the in-image suite on the filter box.
set -u
TAG=${TAG:?}; W=/srv/work/gpu-mock; L=$W/gate-$TAG-wait.log; P=/opt/computewise/gpu-mock-dev/tests/plans
G=/srv/work/gpu-mock/next-r6; ORDER=${ORDER:-/srv/work/gpu-mock/r4wip/order}; HF=/srv/cache/hf/mlsbench-kv/02c58c389f0f/hf; KIT=/srv/work/gpu-mock/mls-kit; KVTREE=/srv/cache/tasks/mls-kv-deploy/b85d30e78b4d/project; ISO=/opt/computewise/isolation
built() { [ "$(systemctl is-active p006-iso-build$TAG 2>/dev/null)" != active ] && grep -qs "mlskv exit=0" /srv/build/build$TAG-*/progress 2>/dev/null && grep -qs "filter exit=0" /srv/build/build$TAG-*/progress 2>/dev/null; }
n=0; until built && ! docker ps --format "{{.Names}}" | grep -qE "^(p6-box-|gm-)"; do n=$((n+1)); [ $n -gt 2160 ] && { echo "gave up at $(date -u +%H:%M:%SZ)" >> $L; exit 3; }; sleep 20; done
id() { docker images --no-trunc --format "{{.Repository}}:{{.Tag}} {{.ID}}" | awk -v t="$1" '$1==t{print $2}' | head -1; }
lbl() { docker inspect "$1" --format '{{index .Config.Labels "p006.sidecar-image-id"}}'; }
BOX=$(id ml-node:py311-torch2110); SIDE=$(id gpu-daemon:py311-torch2110); DP=$(id ml-node:py311-torch2110-dataprep); DPS=$(id gpu-daemon:py311-torch2110-dataprep)
KV=$(id ml-node:py311-torch2110-mlskv); FILT=$(id ml-node:py311-torch2110-filter); FSIDE=$(id gpu-daemon:py311-torch2110-filter)
echo "start $(date -u +%H:%M:%SZ) BOX=$BOX SIDE=$SIDE DP=$DP KV=$KV FILT=$FILT FSIDE=$FSIDE box-label=$(lbl $BOX) filter-label=$(lbl $FILT)" >> $L
[ "$BOX" != sha256:2e51bccf13213637fbcf2ae5090bb0207a41cb9be358e4f3992d7e889f665537 ] || { echo "main tag still r4, abort" >> $L; exit 4; }
[ "$FILT" != sha256:9b163ddd8dfe04079c85d50609e33200badfd0e69d51b906819e307c38f47b01 ] || { echo "filter tag still r4's, abort" >> $L; exit 4; }
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
# 0r6. r6's own test on the built KV box, installed patch (PATCH_SRC=0)
cd $G && IMG=$KV PATCH_SRC=0 bash tests/test_measured_in_image.sh > $W/gate-$TAG-measured.log 2>&1; echo "measured exit $? $(date -u +%H:%M:%SZ)" >> $L
# 0a. r4's tests on the built KV box, installed patch (PATCH_SRC=0)
cd $G && IMG=$KV PATCH_SRC=0 bash tests/test_shard_serving_in_image.sh > $W/gate-$TAG-shard.log 2>&1; echo "shard exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$KV HF=$HF KIT=$KIT TREE=$KVTREE ORDER=$ORDER SPEED=20 bash tests/test_shard_real_in_image.sh > $W/gate-$TAG-shard-real.log 2>&1; echo "shard-real exit $? $(date -u +%H:%M:%SZ)" >> $L
# 0. r3's tests on the built packages box, installed patch (PATCH_SRC=0)
cd $G && IMG=$BOX bash tests/test_installed_text_in_image.sh > $W/gate-$TAG-text.log 2>&1; echo "text exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$KV bash tests/test_installed_text_in_image.sh > $W/gate-$TAG-text-kv.log 2>&1; echo "text-kv exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$BOX PATCH_SRC=0 bash tests/test_exec_evaluator_in_image.sh > $W/gate-$TAG-exec.log 2>&1; echo "exec exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$BOX PATCH_SRC=0 bash tests/test_host_cpu_in_image.sh > $W/gate-$TAG-hostcpu.log 2>&1; echo "hostcpu exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$BOX PATCH_SRC=0 bash tests/test_oracle_and_clock_in_image.sh > $W/gate-$TAG-oracle.log 2>&1; echo "oracle exit $? $(date -u +%H:%M:%SZ)" >> $L
# 1. model-id known answer on the KV box
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20-modelid.json --out /srv/runs/gm-mls-kv-modelid-ka-$TAG --cost-tag gm-mls-kv-modelid-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-modelid-ka-$TAG-run_one.log 2>&1; echo "modelid-ka exit $? $(date -u +%H:%M:%SZ)" >> $L
# 2. the pair gate
cd $G && EXP=/opt/computewise BOX=$BOX SIDE=$SIDE TAG=$TAG REF=/srv/runs/gm-gate2-muonh-${REFTAG:-91a3} SHIFT=0 SCHED=11c325bf bash tests/gate_new_pair.sh > $W/gate-$TAG.log 2>&1; echo "gate exit $? $(date -u +%H:%M:%SZ)" >> $L
# 3. KV known answers
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-kivi-gsm8k-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG --cost-tag gm-mls-kv-ka-$TAG --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-run_one.log 2>&1; echo "kv1 exit $? $(date -u +%H:%M:%SZ)" >> $L
cd $ISO && python3 -I run_one.py --task mls-kv-deploy --harness scripted --plan $P/mls-kv-deploy-fa-gsm8k-direct-k20.json --out /srv/runs/gm-mls-kv-ka-$TAG-direct --cost-tag gm-mls-kv-ka-$TAG-direct --speed 20 --after-deadline continue --image $KV --sidecar-image $SIDE > $W/mls-kv-ka-$TAG-direct-run_one.log 2>&1; echo "kv2 exit $? $(date -u +%H:%M:%SZ)" >> $L
# 4./5. suites on the dataprep and filter boxes (installed patch)
cd $G && IMG=$DP INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-dataprep-suite.log 2>&1; echo "dataprep suite $(date -u +%H:%M:%SZ)" >> $L
cd $G && IMG=$FILT INSTALLED=1 ADV_RUNS=1 bash tests/run_vm_suite.sh > $W/gate-$TAG-filter-suite.log 2>&1; echo "filter suite $(date -u +%H:%M:%SZ)" >> $L
echo done >> $W/gate-$TAG.log; echo "done $(date -u +%H:%M:%SZ)" >> $L
