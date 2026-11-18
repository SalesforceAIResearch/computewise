#!/bin/bash
# Staging check for the modded-nanogpt package tasks. Heavy: run it on the ComputeWise VM, never on the controller.
#
# usage: bash build/check_staging_mng3.sh <gpu-tasks root> <stage dir> <task_id>
#
# What it checks, into <stage dir>/<task_id>/:
#   tree/          build/stage_task.py, fresh: the scripts (each checked against its records' entry hash) and the FineWeb
#                  shards (each checked against the sha256 the Hub's LFS pointer gives), with GPU_TASKS_HUB_CACHE=<stage dir>/
#                  hub-cache so one download serves every package task
#   venv/          the task's Python (uv), installed with environment.install.cpu_lines: torch's CPU wheel first, then the
#                  rest under the constraints
#   check/         summary.txt, one line per result; pip check; an import of torch and the scripts' standard-library imports;
#                  a syntax compile of every staged script; the shard headers (magic 20240520, version 1, token count)
# What it does not do: start a script. The scripts call torch.cuda, init_process_group("nccl") and torch.compile, so on a CPU
# host they start only under gpu-mock's FakeGPU and gloo shim. That is gpu-mock's replay check on the built image, not this.
set -u
D=${1:?gpu-tasks root}; S=${2:?stage dir}; t=${3:?task id}
UV=${UV:-$HOME/.local/bin/uv}
H="python3 -I $D/build/check_staging_helpers.py"
W=$S/$t M=$S/$t/tree.staging C=$S/$t/check
log() { echo "[$(date -u +%H:%M:%SZ)] $t: $*" | tee -a "$C/summary.txt"; }
[ -d "$W" ] && chmod -R u+w "$W"; rm -rf "$W"; mkdir -p "$C"
GPU_TASKS_HUB_CACHE=$S/hub-cache python3 -I "$D/build/stage_task.py" "$t" "$W/tree" > "$C/stage_task.log" 2>&1
log "stage_task exit $? | tree files $(find "$W/tree" -type f | wc -l), bytes $(du -sb "$W/tree" | cut -f1)"
mf="$M/MANIFEST.json"
PY=$($H field "$mf" python)
"$UV" python install "$PY" >> "$C/install.log" 2>&1
"$UV" venv -q -p "$PY" "$W/venv" >> "$C/install.log" 2>&1
rc_i=0
while read -r args; do
  (cd "$M" && eval "UV_HTTP_TIMEOUT=300 \"$UV\" pip install --python \"$W/venv/bin/python\" $args") >> "$C/install.log" 2>&1 || rc_i=$?
done < <($H field "$mf" cpu_lines)
log "install exit $rc_i | python $("$W/venv/bin/python" -c 'import sys; print(sys.version.split()[0])' 2>&1)"
"$UV" pip check --python "$W/venv/bin/python" > "$C/pip-check.log" 2>&1
log "pip check exit $? | $(tail -1 "$C/pip-check.log")"
"$W/venv/bin/python" -I -c 'import torch, numpy; print("torch", torch.__version__, "numpy", numpy.__version__, "threads", torch.get_num_threads())' > "$C/import.log" 2>&1
log "import exit $? | $(tail -1 "$C/import.log")"
rc_c=0
for f in $(cd "$W/tree" && find . -name "*.py" | sort); do
  "$W/venv/bin/python" -I -m py_compile "$W/tree/$f" >> "$C/compile.log" 2>&1 || { rc_c=1; echo "FAIL $f" >> "$C/compile.log"; }
done
log "compile of staged scripts exit $rc_c | $(cd "$W/tree" && find . -name "*.py" | wc -l) scripts"
"$W/venv/bin/python" -I - "$W/tree/data/fineweb10B" > "$C/shards.log" 2>&1 <<'PY'
import sys, os, numpy as np
d = sys.argv[1]
bad = 0
for f in sorted(os.listdir(d)):
    h = np.fromfile(os.path.join(d, f), dtype=np.int32, count=256)
    n = int(h[2]); size = os.path.getsize(os.path.join(d, f))
    ok = h[0] == 20240520 and h[1] == 1 and size == 1024 + 2 * n
    bad += not ok
    print(("ok  " if ok else "BAD ") + f"{f}: magic {h[0]}, version {h[1]}, tokens {n}, bytes {size}")
print(f"shards {len(os.listdir(d))}, bad {bad}")
sys.exit(1 if bad else 0)
PY
log "shard headers exit $? | $(tail -1 "$C/shards.log")"
log "done"
