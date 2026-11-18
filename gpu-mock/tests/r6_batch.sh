#!/usr/bin/env bash
# r6: run a list of decode jobs on one machine in P lanes. The jobs are split
# into P bins, longest first into the least-loaded bin, and each lane runs its
# bin in turn, so no two decodes ever share a lane. Each decode gets T threads
# (a --cpus cap, no core pinning: the scheduler spreads the threads, which
# avoids pairing two decodes on one core's hyperthreads). A job whose slice
# file already exists is skipped, so a batch can be re-run after an
# interruption without redoing finished work.
# Usage (as a systemd unit):  r6_batch.sh <joblist> <P> <T>
#   joblist: lines `<record_id>:<start>:<end>[<TAB>estimated seconds]`
set -uo pipefail
LIST=$1; P=${2:-4}; T=${3:-11}
W=/srv/work/gpu-mock/r6; OUT=${R6_OUT:-out}; mkdir -p "$W/$OUT" "$W/log"; export R6_OUT=$OUT
cd "$W"
python3 - "$LIST" "$P" "$W/$OUT" <<'PY' > "$W/log/bins.$$"
import os, sys
lst, p, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
done = set(os.listdir(out)) if os.path.isdir(out) else set()
jobs = []
for ln in open(lst):
    f = ln.rstrip("\n").split("\t")
    if not f[0]:
        continue
    rid, s, e = f[0].rsplit(":", 2)
    if f"{rid}.{s}-{e}.json" in done:
        continue
    jobs.append((f[0], float(f[1]) if len(f) > 1 and f[1] else 1.0))
jobs.sort(key=lambda j: -j[1])
load = [0.0] * p
for job, est in jobs:
    b = min(range(p), key=lambda i: load[i])
    load[b] += est
    print(b, job)
PY
echo "$(date -u +%H:%M:%SZ) batch start: $(wc -l < "$W/log/bins.$$") jobs in $P lanes x $T threads" >> "$W/log/batch.log"
lane() {  # $1 lane number
  awk -v b="$1" '$1 == b {print $2}' "$W/log/bins.$$" | while read -r job; do
    local tag; tag=$(echo "$job" | tr ':' '_')
    ./r6_run_one.sh "$job" "$T" "" > "$W/log/$tag.log" 2>&1
    local rc=$?
    echo "$(date -u +%H:%M:%SZ) lane $1 $job rc=$rc $(grep -h 'scored' "$W/log/$tag.log" | tail -1)" >> "$W/log/batch.log"
  done
}
for b in $(seq 0 $((P - 1))); do lane "$b" & done
wait
echo "$(date -u +%H:%M:%SZ) batch done" >> "$W/log/batch.log"
rm -f "$W/log/bins.$$"
