#!/usr/bin/env bash
# Every in-image check against THIS tree's torch patch, on the GCP host, then
# optionally the full DPO replay. Each container is capped and pinned.
#   sudo [IMG=gm-dpo:x86] [INSTALLED=1] [ADV_RUNS=3] [REPLAY=1 SPEED=100 OUT=/srv/runs/...] bash tests/run_vm_suite.sh
#   INSTALLED=1 tests the patch baked into IMG (a gate); otherwise this tree's patch is mounted over it.
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"; TP="$G/src/torch_patch"
if [ "${INSTALLED:-0}" = 1 ]; then
  # Test the patch INSTALLED in the image (a gate on a built image): mount the
  # tests and the daemon's source only, never this tree's torch_patch, so
  # adversarial_probes.py's sys.path insert finds no tree copy and the image's
  # patch loads through sitecustomize.
  # the daemon-level tests import gpu_daemon and convert from the tree, so mount those too (until 2026-10-23 they
  # crashed silently with ModuleNotFoundError in tree mode; the installed-patch mode mounts all of src)
  MNT=(-v "$G/tests:/gm/tests:ro" -v "$G/src/daemon:/gm/src/daemon:ro" -v "$G/src/bridge:/gm/src/bridge:ro" -v "$G/src/convert:/gm/src/convert:ro")
  PS=0
else
  MNT=(-v "$G/tests:/gm/tests:ro" -v "$TP:/gm/src/torch_patch:ro" -v "$G/src/daemon:/gm/src/daemon:ro" -v "$G/src/convert:/gm/src/convert:ro" -v "$G/src/bridge:/gm/src/bridge:ro" -v "$TP:/usr/local/nvidia/python:ro" -v "$TP/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
  PS=1
fi
echo "patch under test: $([ "$PS" = 1 ] && echo "this tree ($(cd "$TP" && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -c1-16))" || echo "the one installed in $IMG")"
docker run --entrypoint '' --rm --cpus 1 --memory 1g "${MNT[@]}" "$IMG" bash -c 'cd /usr/local/nvidia/python && LC_ALL=C ls *.py | xargs sha256sum | sha256sum | cut -c1-16 | sed "s/^/installed torch_patch in the image: /"'
PRE='source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT; cd /tmp'
NOISE='datetime.datetime size|FutureWarning|UserWarning|warnings.warn'
echo "##### $(date -u +%T) adversarial probes x${ADV_RUNS:-3}"
for i in $(seq "${ADV_RUNS:-3}"); do
  docker run --entrypoint '' --rm --cpus 4 --cpuset-cpus 24-27 --memory 8g "${MNT[@]}" -e RUN="$i" -e NOISE="$NOISE" "$IMG" bash -c "$PRE"'; timeout 600 python /gm/tests/adversarial_probes.py 2>&1 | grep -vE "$NOISE" > a.txt; echo "run $RUN: ok $(grep -c "^\[ok" a.txt), TELL $(grep -c "^\[TELL" a.txt)"; grep "^\[TELL" a.txt'
done
echo "##### $(date -u +%T) layer D"
docker run --entrypoint '' --rm --cpus 2 --cpuset-cpus 24-25 --memory 4g "${MNT[@]}" "$IMG" bash -c "$PRE; python /gm/tests/test_layer_d.py 2>&1 | tail -1"
echo "##### $(date -u +%T) step time (startup inside step 1)"
docker run --entrypoint '' --rm --cpus 1 --cpuset-cpus 24-24 --memory 2g "${MNT[@]}" "$IMG" bash -c "$PRE; python /gm/tests/test_step_time.py 2>&1 | tail -1"
echo "##### $(date -u +%T) exact key order-insensitive + near_miss (daemon, in the image's python)"
docker run --entrypoint '' --rm --cpus 1 --cpuset-cpus 24-24 --memory 2g "${MNT[@]}" "$IMG" bash -c "$PRE; for t in test_argv_order test_argv_order_daemon test_host_gap_daemon test_effective_args_daemon test_single_process_daemon test_failure_serving_daemon test_path_exact_daemon test_short_run_return_daemon test_abs_path_args_daemon test_prefix_full_schedule_daemon; do python /gm/tests/\$t.py 2>&1 | tail -1 | grep -E 'passed, [0-9]+ failed' || echo \"FAIL \$t: no result line\"; done"
for t in test_clock_in_image test_gpu_index_in_image test_daemon_contact_in_image test_fingerprint_root_in_image test_optimizer_neutralise_in_image test_main_process_first_in_image test_observe_chain_in_image test_resume_in_image test_nan_guard_in_image test_path_bound_in_image test_surrogate_in_image test_return_types_in_image test_identity_surface_in_image test_failure_serving_in_image test_faulthandler_in_image test_dynamo_preimport_in_image test_chdir_script_in_image test_explicit_default_in_image; do
  echo "##### $(date -u +%T) $t"; IMG="$IMG" PATCH_SRC="$PS" bash "$G/tests/$t.sh" 2>&1 | grep -vE "$NOISE" | tail -14
done
# Gate 1 also includes tests/probe_all_scripts.sh (every registry script to its
# first optimizer step, sidecar topology, ~10 min): run it separately, with the
# same IMG and PATCH_SRC, and DIMG for the sidecar image under test.
if [ "${REPLAY:-0}" = 1 ]; then
  echo "##### $(date -u +%T) DPO replay"
  IMG="$IMG" PATCH_SRC="$PS" SPEED="${SPEED:-100}" OUT="${OUT:-/srv/runs/gm-replay-dpo}" bash "$G/tests/replay_dpo.sh" 2>&1 | tail -40
fi
echo "##### $(date -u +%T) done"
