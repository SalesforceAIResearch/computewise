#!/usr/bin/env bash
# One r6 decode job in the r4 KV box with the mock off. Usage: r6_run_one.sh <job> <threads> <cpuset>
set -uo pipefail
JOB=$1; TH=${2:-10}; CPUS=${3:-}
IMG=sha256:c7579a3943ec390fa3e12ba56d7faecd866998f9b90dd7f7f9ff932339c6d3b6
R=/opt/computewise/gpu-tasks/registry
HF=$(ls -d /srv/cache/hf/mlsbench-kv/*/hf | while read d; do [ -d "$d/hub/models--Qwen--Qwen2.5-3B-Instruct" ] && echo "$d" && break; done)
W=/srv/work/gpu-mock/r6; OUT=${R6_OUT:-out}; mkdir -p $W/$OUT $W/log   # R6_OUT: a separate output dir (the full-n fidelity decode)
# Checked just before each decode, so a running batch honours them for every job it has not started:
#   the slice is already done here (its file exists, perhaps copied from another machine);
#   the job is listed in $W/skip.txt (reassigned to another machine);
#   $W/STOP exists (stop between jobs).
RID=${JOB%:*:*}; SE=${JOB#"$RID":}; SLICE="$W/$OUT/$RID.${SE/:/-}.json"
[ -s "$SLICE" ] && { echo "skip $JOB: slice already done"; exit 0; }
[ -f "$W/skip.txt" ] && grep -qxF "$JOB" "$W/skip.txt" && { echo "skip $JOB: reassigned (skip.txt)"; exit 0; }
[ -f "$W/STOP" ] && { echo "skip $JOB: STOP"; exit 0; }
HFC=$(mktemp -d $W/hfc.XXXX); for x in datasets xet STAGED.json; do [ -e "$HF/$x" ] && cp -a "$HF/$x" "$HFC/"; done; mkdir -p "$HFC/hub"
PIN=(); [ -n "$CPUS" ] && PIN=(--cpuset-cpus "$CPUS")
docker run --rm --network none --label p006.r6=decode "${PIN[@]}" --cpus "$TH" --memory 24g --entrypoint '' \
  -v "$HFC:/hf" -v "$HF/hub:/hf/hub:ro" -v "$R:/reg:ro" -v "$W:/w" \
  -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 -e HF_DATASETS_OFFLINE=1 -e NVRT_DISABLE=1 \
  -e R6_IMAGE="$IMG" -e R6_MACHINE="${R6_MACHINE:-$(hostname)}" "$IMG" \
  bash -c "unset LD_PRELOAD; python3 -I /w/decode_first_k.py --records /reg/records --scripts /reg/artifacts/mls/scripts --order /reg/records --job '$JOB' --out /w/$OUT --threads $TH"
rc=$?; rm -rf "$HFC"; echo "rc=$rc"; exit $rc
