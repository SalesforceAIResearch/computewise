#!/bin/bash
# Staging check for registry tasks that carry an `environment`. Heavy: run it on the ComputeWise VM
# (experiments/gcp/host/vm.sh), never on the controller.
#
# usage: bash build/check_staging.sh <gpu-tasks root> <stage dir> <task_id>...      (tasks run in parallel)
#
# Per task, into <stage dir>/<task_id>/:
#   tree/, tree.staging/   build/stage_task.py, fresh (needs GitHub for the library checkout)
#   venv/                  the manifest's Python (uv), installed with the task's environment.install.cpu_lines: the same pins,
#                          torch's CPU wheel first because the host has no CUDA
#   hf/                    build/stage_hub.py, fresh; then hub/ and modules/ made read-only, datasets/ left writable
#   check/run/             a copy of tree/ that every command below runs from, with the manifest's env
#   check/summary.txt      one line per result
# Checks:
#   start   one step of each distinct script, one process: --use_cpu --max_steps 1, batch 2, accumulation 1
#   (no multi-process real-compute check: `accelerate launch --cpu --num_processes 4` starts one process, not four, without
#   MPI, which the first run of this check on 2026-10-20 showed; the steps check below starts the ranks under torchrun instead)
#   steps   each launch line once per distinct command (seed and output_dir ignored), in full: one process, or torchrun with
#           the line's process count, with build/stepcount on PYTHONPATH (the real Trainer loop on the real data with no
#           compute), counting optimizer steps per rank
set -u
D=${1:?gpu-tasks root}; S=${2:?stage dir}; shift 2
UV=${UV:-$HOME/.local/bin/uv}
H="python3 -I $D/build/check_staging_helpers.py"
CPU_OVR="--use_cpu --max_steps 1 --per_device_train_batch_size 2 --gradient_accumulation_steps 1"

one_task() {
  local t=$1 W=$S/$1 M=$S/$1/tree.staging C=$S/$1/check
  log() { echo "[$(date -u +%H:%M:%SZ)] $t: $*" >> "$C/summary.txt"; }
  [ -d "$W" ] && chmod -R u+w "$W"; rm -rf "$W"; mkdir -p "$C"
  python3 -I "$D/build/stage_task.py" "$t" "$W/tree" > "$C/stage_task.log" 2>&1
  log "stage_task exit $? | tree files $(find "$W/tree" -type f | wc -l), .git dirs $(find "$W/tree" -name .git | wc -l)"
  local mf="$M/MANIFEST.json" PY PKGS
  PY=$($H field "$mf" python); PKGS=$($H field "$mf" packages)
  "$UV" python install "$PY" >> "$C/install.log" 2>&1
  "$UV" venv -q -p "$PY" "$W/venv" >> "$C/install.log" 2>&1
  local args rc_i=0
  while read -r args; do                       # the task's environment.install.cpu_lines, run from the manifest directory
    (cd "$M" && eval "UV_HTTP_TIMEOUT=300 \"$UV\" pip install --python \"$W/venv/bin/python\" $args") >> "$C/install.log" 2>&1 || rc_i=$?
  done < <($H field "$mf" cpu_lines)
  log "install exit $rc_i | $("$W/venv/bin/python" -c 'import sys,torch,importlib.util as u;print("python",sys.version.split()[0],"| torch",torch.__version__,"| trl in venv:",u.find_spec("trl") is not None)' 2>&1 | tail -1)"
  "$W/venv/bin/python" -I "$D/build/stage_hub.py" "$mf" "$W/hf" > "$C/stage_hub.log" 2>&1
  log "stage_hub exit $? | cache $(du -sh "$W/hf" | cut -f1)"
  grep -E "^OFFLINE" "$C/stage_hub.log" | while read -r l; do log "  $l"; done
  chmod -R a-w "$W/hf/hub" "$W/hf/modules" 2>/dev/null
  local R=$C/run; cp -R "$W/tree" "$R"
  local -a ENVV; mapfile -t ENVV < <($H env "$mf" "$R" "$W/hf"); ENVV+=("PATH=$W/venv/bin:$PATH")
  log "env ${ENVV[*]} | $(cd "$R" && env "${ENVV[@]}" python -c 'import trl;print("trl from",trl.__file__,trl.__version__)' 2>&1 | tail -1)"
  local -a IDS LINES; local id line
  while IFS=$'\t' read -r id line; do IDS+=("$id"); LINES+=("$line"); done < <($H launches "$mf")
  local n=0 seen="" script cmd t0 rc k
  for line in "${LINES[@]}"; do
    script=$(grep -o -E "[^ ]+\.py" <<< "$line" | head -1)
    case " $seen " in *" $script "*) continue ;; esac
    seen="$seen $script"; n=$((n+1))
    cmd="$(sed -E 's/^accelerate launch --num_processes [0-9]+ /python /' <<< "$line") $CPU_OVR --output_dir=$C/out-start-$n"
    t0=$(date +%s); (cd "$R" && env "${ENVV[@]}" timeout 3600 bash -c "$cmd") > "$C/start-$n.log" 2>&1; rc=$?
    log "start $n ($script, one process): exit $rc in $(( $(date +%s)-t0 )) s | $(grep -E 'Error|train_runtime' "$C/start-$n.log" | tail -1 | cut -c1-200)"
  done
  seen=""
  for k in "${!LINES[@]}"; do
    line=${LINES[$k]}; id=${IDS[$k]}
    local key; key=$(sed -E 's/--seed[ =][0-9]+//; s/--output_dir[= ][^ ]+//' <<< "$line")
    case "$seen" in *"|$key|"*) log "steps $id: same command as an earlier line apart from seed or output_dir; not rerun"; continue ;; esac
    seen="$seen|$key|"
    cmd=$(sed -E 's/^accelerate launch --num_processes ([0-9]+) /torchrun --standalone --nproc_per_node \1 /' <<< "$line")
    cmd="$cmd --use_cpu --output_dir=$C/out-steps-$id"
    t0=$(date +%s)
    (cd "$R" && env "${ENVV[@]}" "PYTHONPATH=$R:$D/build/stepcount" "STEPCOUNT=$C/steps-$id" timeout 7200 bash -c "$cmd") > "$C/steps-$id.log" 2>&1; rc=$?
    log "steps $id: exit $rc in $(( $(date +%s)-t0 )) s | $($H steps "$C/steps-$id")"
    rm -rf "$C/out-steps-$id"
  done
  log "done | outputs $(du -sh "$C"/out-* 2>/dev/null | tr '\n' ' ')"
}

for t in "$@"; do one_task "$t" & done
wait
echo "[$(date -u +%H:%M:%SZ)] all done | free $(df -h "$S" | awk 'NR==2{print $4}')" >> "$S/check-done.txt"
