#!/usr/bin/env bash
# C1 tolerance repeats: N scripted packed_overlap runs of one task, one box at a time,
# each preceded by gcp's 60 s host-memory label. Runs ON the VM, as root, detached with systemd-run.
#   repeat_check.sh <n> <task> <host-terms.json> [extra run_one.py flags...]
# Interruptible between runs: touch $OUT/STOP and the driver exits after the run in flight ends.
# Each run directory gets mem_baseline_pre.json (gcp's MEMJSON); $OUT/index.tsv lists the runs in order.
set -u
N=$1; TASK=$2; TERMS=$3; shift 3
EXP=/opt/computewise
OUT=/srv/work/runner/repeats
mkdir -p "$OUT"
for i in $(seq 1 "$N"); do
  [ -e "$OUT/STOP" ] && { echo "stop file found before repeat $i; exiting"; exit 0; }
  pre="$OUT/mem-pre-$i.json"
  python3 -I "$EXP/gcp/host/mem-baseline.py" 60 > "$pre"
  log="$OUT/repeat-$i.log"
  (cd "$EXP/runner" && python3 -I run_smoke.py --task "$TASK" --harness scripted --policy packed_overlap \
      --host-terms "$TERMS" --max-hours 1.5 -- "$@") > "$log" 2>&1
  run=$(grep -m1 -o '/srv/runs/[^ ]*' "$log" | head -1)
  [ -n "$run" ] && [ -d "$run" ] && cp "$pre" "$run/mem_baseline_pre.json"
  printf '%s\t%s\t%s\n' "$i" "$(date -u +%FT%TZ)" "$run" >> "$OUT/index.tsv"
done
echo "all $N repeats done"
