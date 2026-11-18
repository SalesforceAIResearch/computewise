#!/usr/bin/env bash
# Apply gpu-mock's checks to a run_one.py run directory (isolation's single-run
# contract): for every job log under agent-logs/, compare what the Trainer
# printed with the record the daemon served it (from occupancy.jsonl), then
# the per-job host times, and the run's images and sidecar source hash.
# Usage (on the VM, as root):  bash tests/check_run_one.sh /srv/runs/<run dir>
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd); REG="${REG:-$G/../gpu-tasks/registry}"   # a frozen tree has no registry beside it: pass REG (the deployed one on the E2)
R="${1:?run dir}"; O="$R/occupancy.jsonl"
echo "=== $R"
python3 -I -c "
import json, sys
s = json.load(open(sys.argv[1] + '/status.json')); r = json.load(open(sys.argv[1] + '/run.json'))
print('status:', {k: s.get(k) for k in ('status', 'harness_exit_code', 'teardown_reason', 'start_utc', 'end_utc')})
img = r.get('image') or {}; sc = r.get('sidecar') or {}
print('image:', img if isinstance(img, str) else {k: img.get(k) for k in ('name', 'id', 'tag') if k in img})
print('sidecar:', {k: sc.get(k) for k in ('image', 'id', 'src_sha256') if k in sc})
print('caps:', r.get('caps'))" "$R" 2>&1 | cut -c1-240
echo "--- occupancy: $(grep -oE '"kind":"[a-z_]+"' "$O" | sort | uniq -c | tr '\n' ' ')"
grep '"kind":"lookup"' "$O" | grep -oE '"job":[0-9]+|"mode":"[a-z]+"|"record_id":"[^"]*"|"match":"[a-z_]+"' | paste -sd' ' | fold -w 200
NLOGS=$(ls "$R"/agent-logs/*.log 2>/dev/null | wc -l)
for L in "$R"/agent-logs/*.log; do
  [ -f "$L" ] || continue
  name=$(basename "$L" .log)
  # the record this job replayed: with one job log, the one lookup; else the lookup
  # whose job's hello argv names the script in the label (battery names)
  if [ "$NLOGS" = 1 ]; then
    RID=$(grep '"kind":"lookup"' "$O" | grep -o '"record_id":"[^"]*"' | head -1 | cut -d'"' -f4)
  else
    script=$(echo "$name" | sed -E 's/^(dpo|reward-modeling|sft).*/\1/' | sed 's/reward-modeling/reward_modeling/')
    RID=$(python3 -I -c "
import json, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
jobs = {e['job'] for e in ev if e['kind'] == 'job_hello' and any(sys.argv[2] + '.py' in str(a) for a in e.get('argv', []))}
for e in ev:
    if e['kind'] == 'lookup' and e.get('job') in jobs and e.get('record_id'):
        print(e['record_id']); break" "$O" "$script")
  fi
  echo "--- $name -> record ${RID:-NONE}"
  if [ -z "$RID" ]; then echo "    no lookup for this job: replay never engaged (RECORD CHECK FAIL)"
  elif grep -qE '^step:[0-9]+/[0-9]+ val_loss:' "$L"; then
    python3 -I "$G/tests/check_mng3_replay.py" "$L" "$REG/records/$RID.json" | sed 's/^/    /'
  else python3 -I "$G/tests/check_trainer_replay.py" "$L" "$REG/records/$RID.json" | sed 's/^/    /'
  fi
  echo "    tracebacks in the job log: $(grep -c Traceback "$L") | exit line: $(grep -oE 'JOB_EXIT=[0-9]+|exit(code)?[ =:]+-?[0-9]+' "$L" | tail -1)"
done
echo "--- host time per job:"; python3 -I "$G/tests/host_times.py" "$O"
