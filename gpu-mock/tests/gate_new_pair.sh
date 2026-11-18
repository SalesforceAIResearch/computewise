#!/usr/bin/env bash
# Everything the study asked for before an mng3 image pair ships
# (2026-10-21), in one unattended run on the E2, in this order:
#   1. gate 2: the muonh known-answer plan through isolation's run_one.py on
#      the pair (the job is host-bound at speed 100, so --after-deadline
#      continue: a slow host must not void the comparison);
#   2. the record check of that run, and the old-vs-new comparison against a
#      reference run dir (default: gate 2 on the first mng3 image);
#   3. gate 1: the in-image suite against the pair's installed patch;
#   4. the all-scripts probe on the pair's images;
#   5. the scheduler check (fake-slurm's scheduler rides in the box image from the
#      8d8 pair on; inert without run_one --scheduler on): the image label, the
#      commands present, no controller and no socket at container start, and the
#      gate-2 run (scheduler off) with no scheduler log;
#   6. the trial and per_example stand-ins on the installed patch;
#   7. task-draw known answers against the registry (pkg-p20-tail-q's seeded reruns, the exact key).
# Usage (on the VM, as root):
#   BOX=<box image id> SIDE=<sidecar image id> [TAG=v2] [REF=/srv/runs/gm-gate2-muonh] [SHIFT=0] [STEPS=1234567] [SCHED=<expected p006.scheduler label prefix>] bash tests/gate_new_pair.sh
#   SHIFT  the train_time difference new-old the comparison must show after step 0: 0 for a
#          change that moves no recorded time (the scrub, a rename), 184.258 for the one that
#          folded the start-up into step 1 against a pre-fix reference
#   STEPS  which of the seven to run, by digit (default all): e.g. STEPS=14 for gate 2 and the probe
#   SCHED  the scheduler hash isolation named for the pair (a prefix of the p006.scheduler label); checked when given
# Outputs under /srv/work/gpu-mock/gate-<TAG>-*.log and /srv/runs/gm-gate2-muonh-<TAG>, /srv/runs/gm-probe-scripts-<TAG>;
# a one-screen summary at /srv/work/gpu-mock/gate-<TAG>-summary.txt.
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd); EXP=${EXP:-$(cd "$G/.." && pwd)}   # EXP overrides: a frozen release tree sits outside experiments/
BOX="${BOX:?box image id}"; SIDE="${SIDE:?sidecar image id}"; TAG="${TAG:-v2}"
REF="${REF:-/srv/runs/gm-gate2-muonh}"; SHIFT="${SHIFT:-0}"; STEPS="${STEPS:-1234567}"; SCHED="${SCHED:-}"
W=/srv/work/gpu-mock; RUN=/srv/runs/gm-gate2-muonh-$TAG; S="$W/gate-$TAG-summary.txt"
mkdir -p "$W"; : > "$S"
say() { echo "[$(date -u +%T)] $*" | tee -a "$S"; }
say "gate on box $BOX sidecar $SIDE (tag $TAG)"
case "$STEPS" in *1*)
say "1. gate 2 / known-answer: muonh-tuned-run1 of pkg-p6-par-long through run_one.py"
rm -rf "$RUN"
python3 -I "$EXP/isolation/run_one.py" --task pkg-p6-par-long --harness scripted \
  --plan "$G/tests/plans/pkg-p6-par-long-muonh-tuned-run1.json" --out "$RUN" --cost-tag "gm-gate2-muonh-$TAG" \
  --speed 100 --after-deadline continue --image "$BOX" --sidecar-image "$SIDE" > "$W/gate-$TAG-gate2.log" 2>&1
say "   run_one exit $? | status: $(python3 -I -c "import json,sys; s=json.load(open(sys.argv[1]+'/status.json')); print(s.get('status'), 'harness exit', s.get('harness_exit_code'))" "$RUN" 2>&1)"
REG="${REG:-$EXP/gpu-tasks/registry}" bash "$G/tests/check_run_one.sh" "$RUN" > "$W/gate-$TAG-check.log" 2>&1
say "   record check: $(grep -E 'RECORD CHECK (PASS|FAIL)' "$W/gate-$TAG-check.log" | head -1)"
# the task flag (decision slots, siblings, the task-run seed rule all live behind it) and the LR stream
say "   task flag: run.json gpu_daemon=$(python3 -I -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]+'/run.json')).get('gpu_daemon')))" "$RUN" 2>&1) | daemon log: $(grep -rhoE 'task [^:]+: rerun siblings for [0-9]+ primary records' "$RUN" 2>/dev/null | head -1 || echo none) | lookups with task_job: $(grep -c '"task_job":"' "$RUN/occupancy.jsonl" 2>/dev/null) | lr_stream.jsonl lines: $(wc -l < "$RUN/lr_stream.jsonl" 2>/dev/null || echo absent)"
say "   images used: $(python3 -I -c "import json,sys; r=json.load(open(sys.argv[1]+'/run.json')); print(r['image']['id'], r['sidecar']['id'], 'src', r['sidecar'].get('src_sha256','')[:16])" "$RUN" 2>&1)"
;; esac
case "$STEPS" in *2*)
say "2. comparison with $REF (expected train_time shift $SHIFT s after step 0)"
python3 -I "$G/tests/compare_runs.py" "$REF" "$RUN" "$SHIFT" > "$W/gate-$TAG-compare.log" 2>&1
grep -E '^\s*\[(ok|DIFF)\]|^COMPARE' "$W/gate-$TAG-compare.log" | tee -a "$S"
;; esac
case "$STEPS" in *3*)
say "3. gate 1: in-image suite on the installed patch"
IMG="$BOX" INSTALLED=1 ADV_RUNS=3 bash "$G/tests/run_vm_suite.sh" > "$W/gate-$TAG-suite.log" 2>&1
grep -E 'installed torch_patch|^run [0-9]+: ok|passed, [0-9]+ failed|TELL|VAL_LOSS|PASS|FAIL' "$W/gate-$TAG-suite.log" | grep -vE '^\[ok' | tee -a "$S"
;; esac
case "$STEPS" in *4*)
say "4. all-scripts probe on the images"
IMG="$BOX" DIMG="$SIDE" LANES=4 OUT="/srv/runs/gm-probe-scripts-$TAG" bash "$G/tests/probe_all_scripts.sh" > "$W/gate-$TAG-probe.log" 2>&1
grep -E '^(probe:|ENV|FAIL|SKIP|PROBE)' "$W/gate-$TAG-probe.log" | tee -a "$S"
;; esac
case "$STEPS" in *5*)
say "5. scheduler check (inert without --scheduler on)"
LBL=$(docker image inspect -f '{{ index .Config.Labels "p006.scheduler" }}' "$BOX" 2>/dev/null)
SCHEDNOTE=""; if [ -n "$SCHED" ]; then case "$LBL" in "$SCHED"*) SCHEDNOTE=" (matches SCHED)";; *) SCHEDNOTE=" (DOES NOT MATCH SCHED=$SCHED)";; esac; fi
say "   box label p006.scheduler=${LBL:-<none>}$SCHEDNOTE"
docker run --entrypoint '' --rm --cpus 2 --memory 2g "$BOX" bash -c '
sleep 3
for f in /usr/bin/sbatch /usr/bin/squeue /usr/sbin/slurmctld /usr/sbin/slurmstepd; do [ -e "$f" ] && echo "present $f" || echo "MISSING $f"; done
ps -eo pid,comm,args | grep -E "slurm|sched" | grep -v grep && echo "FAIL: a scheduler process at container start" || echo "ok: no scheduler process at container start"
ls -A /run/slurm 2>/dev/null | grep . && echo "FAIL: something under /run/slurm at container start" || echo "ok: nothing under /run/slurm"
ls -A /var/spool/slurmd 2>/dev/null | grep . && echo "note: /var/spool/slurmd not empty" || echo "ok: /var/spool/slurmd empty"
timeout 10 squeue >/tmp/sq.out 2>&1; echo "squeue without a controller: exit $? :: $(head -c 160 /tmp/sq.out)"' 2>&1 | sed 's/^/   /' | tee -a "$S"
[ -f "$RUN/run.json" ] && say "   gate-2 run: scheduler=$(python3 -I -c "import json,sys; r=json.load(open(sys.argv[1]+'/run.json')); print(r.get('inputs',{}).get('scheduler'), 'BOX_SCHEDULER', r.get('env',{}).get('BOX_SCHEDULER'))" "$RUN" 2>&1), scheduler.jsonl $( [ -e "$RUN/scheduler.jsonl" ] && echo PRESENT || echo absent )"
;; esac
case "$STEPS" in *6*)
say "6. trial and per_example stand-ins on the installed patch"
IMG="$BOX" bash "$G/tests/test_trials_in_image.sh" > "$W/gate-$TAG-trials.log" 2>&1; say "   trials: $(grep -E 'passed, [0-9]+ failed' "$W/gate-$TAG-trials.log" | tail -1) (exit $?)"
IMG="$BOX" bash "$G/tests/test_per_example_in_image.sh" > "$W/gate-$TAG-per-example.log" 2>&1; say "   per_example: $(grep -E 'passed, [0-9]+ failed' "$W/gate-$TAG-per-example.log" | tail -1)"
grep -E '^\[FAIL' "$W/gate-$TAG-trials.log" "$W/gate-$TAG-per-example.log" | tee -a "$S"
;; esac
case "$STEPS" in *7*)
say "7. task-draw known answers against the registry (the tree's daemon source = the built sidecar's, hash-checked by isolation)"
python3 -I "$G/tests/known_answer_p20tail.py" "$EXP/gpu-tasks/registry" "$EXP/gpu-tasks/registry/tasks/pkg-p20-tail-q.json" > "$W/gate-$TAG-p20tail.log" 2>&1; say "   pkg-p20-tail-q replay of Q-p20tail-sonnet-L0-bashdefault-r2b: $(grep -E 'passed, [0-9]+ failed' "$W/gate-$TAG-p20tail.log" | tail -1)"
grep -E '^\[FAIL' "$W/gate-$TAG-p20tail.log" | tee -a "$S"
python3 -I "$G/tests/test_argv_order.py" > "$W/gate-$TAG-argv-order.log" 2>&1; say "   exact key over the registry: $(grep -E 'passed, [0-9]+ failed' "$W/gate-$TAG-argv-order.log" | tail -1)"
python3 -I "$G/tests/known_answer_haiku_l0r3.py" "$EXP/gpu-tasks/registry" > "$W/gate-$TAG-haiku-l0r3.log" 2>&1; say "   G-mlskv-haiku-L0-r3 replay (effective arguments): $(grep -E 'passed, [0-9]+ failed' "$W/gate-$TAG-haiku-l0r3.log" | tail -1)"
grep -E '^\[FAIL' "$W/gate-$TAG-haiku-l0r3.log" | tee -a "$S"
;; esac
say "done. summary: $S"
