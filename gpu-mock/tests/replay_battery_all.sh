#!/usr/bin/env bash
# One replay of each battery script, one after another (never two heavy jobs at
# once on the VM): dpo-seed1, reward-modeling-seed1, sft-seed1. Each writes its
# own run directory /srv/runs/gm-battery-<job>; this prints each job's verdict.
# Usage (on the VM):  sudo [IMG=gm-battery:x86] [PATCH_SRC=1] [SPEED=100] bash tests/replay_battery_all.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
JOBS="${JOBS:-dpo-seed1 reward-modeling-seed1 sft-seed1}"
for j in $JOBS; do
  echo "##### $(date -u +%T) $j"
  JOB="$j" bash "$G/tests/replay_battery.sh" 2>&1 | tail -40
done
echo "##### $(date -u +%T) verdicts"
for j in $JOBS; do printf "%-24s %s\n" "$j" "$(grep -h "RECORD CHECK" /srv/runs/gm-battery-$j/verdict.log 2>/dev/null | tail -1)"; done
