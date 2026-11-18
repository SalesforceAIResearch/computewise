#!/usr/bin/env bash
# End-to-end REPLAY of the adamh track-3 job against the real registry:
#   daemon (registry = gpu-tasks records, 8 virtual GPUs, speed knob SPEED)
#   + torchrun --nproc_per_node=8 of the unmodified script (bytes = entry_hash)
#   + the gpu-mock patch via the sitecustomize shim.
# The job must run all 4875 steps with NO real compute, print the record's
# val_loss series exactly, print train_time close to the record's train_time_s
# series (the job clock reads recorded time whatever the knob), and finish in
# about (train_time + eval) / SPEED wall seconds.
#
# Usage: SPEED=100 NPROC=8 PORT=29800 bash tests/replay_mng3.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
REG=${CW_REGISTRY:-$G/../gpu-tasks/registry}/records
# TAG separates concurrent runs (scratch dir, socket); unset = the default run.
D="$G/.build-scratch/replay-mng3${TAG:+-$TAG}"
PY=/tmp/gpu-mock-prior-art/venv-fakegpu/bin
SPEED="${SPEED:-100}"; NPROC="${NPROC:-8}"; PORT="${PORT:-29800}"
# AF_UNIX paths are capped at 104 bytes on macOS (108 on Linux); the experiment
# directory alone is longer, so the socket lives at a short path.
SOCK="/tmp/gm-replay-mng3${TAG:+-$TAG}.sock"; OCC="$D/occupancy.jsonl"

rm -rf "$D"; mkdir -p "$D/data/fineweb10B" "$D/logs"
cp "$G/.build-scratch/dryrun-mng3/train_gpt_simple_adamh.py" "$D/"
cp "$G/.build-scratch/dryrun-mng3/data/fineweb10B/fineweb_val_000000.bin" "$D/data/fineweb10B/"
rm -f "$SOCK" "$OCC"
# Training data: 4875 steps x 524288 tokens = 2.56 B tokens (~5 GB) is not worth
# writing for a timing test, and under replay the token CONTENT is never used
# (forward is never computed). So: one 10-batch shard in the loader's exact
# format, and 500 symlinked shard names (the loader globs, sorts and reads each
# in turn). Loading is still real I/O. This is a TEST stand-in for FineWeb; a
# task workspace supplies the real dataset.
/tmp/gpu-mock-prior-art/venv-fakegpu/bin/python -I - "$D/data/fineweb10B" <<'PY'
import sys, os, numpy as np
out = sys.argv[1]; n = 10 * 524288 + 64
hdr = np.zeros(256, dtype=np.int32); hdr[0] = 20240520; hdr[1] = 1; hdr[2] = n
base = os.path.join(out, ".shard_base.bin")
with open(base, "wb") as f:
    f.write(hdr.tobytes())
    f.write(np.random.default_rng(1).integers(0, 50304, n, dtype=np.uint16).tobytes())
for i in range(500):
    os.symlink(".shard_base.bin", os.path.join(out, f"fineweb_train_{i:06d}.bin"))
print(f"train data: 500 shard names -> one {n}-token shard (10 batches each)")
PY

# daemon
python3 -I "$G/src/daemon/gpu_daemon.py" --socket "$SOCK" --registry "$REG" \
  --gpus "$NPROC" --speed "$SPEED" --occupancy "$OCC" > "$D/daemon.log" 2>&1 &
DPID=$!
for _ in $(seq 1 100); do [ -S "$SOCK" ] && break; sleep 0.05; done
[ -S "$SOCK" ] || { echo "daemon did not start"; cat "$D/daemon.log"; exit 1; }

cd "$D"
export PYTHONPATH="$G/.build-scratch/site${PYTHONPATH:+:$PYTHONPATH}"
export NVRT_SOCKET="$SOCK" NVRT_DEVICE_COUNT="$NPROC" NVRT_DEBUG=1
unset NVRT_FINGERPRINT_ONLY NVRT_NO_TIMING
T0=$(date +%s)
timeout "${TIMEOUT:-900}" "$PY/torchrun" --nproc_per_node="$NPROC" --master_port="$PORT" \
  train_gpt_simple_adamh.py > replay.log 2>&1
rc=$?; T1=$(date +%s)
kill "$DPID" 2>/dev/null
echo "torchrun exit: $rc   wall: $((T1-T0)) s   (recorded paced time ≈ 780 s / speed $SPEED ≈ $((780/SPEED)) s; wall also includes process start + imports)"
echo "--- replay engagement (rank 0 debug lines):"; grep "replay\[r0\]" replay.log | head -4
echo "--- errors:"; grep -v triton replay.log | grep -iE "Traceback|Error" | head -8
echo "--- printed val_loss vs the record the daemon actually drew:"
DRAWN=$(python3 -I -c "
import json,sys
for l in open('$OCC'):
    e=json.loads(l)
    if e['kind']=='lookup' and e.get('mode')=='replay' and e.get('record_id'):
        print(e['record_id']); break
")
echo "drawn record: ${DRAWN:-<none: replay did not engage>}"
[ -n "$DRAWN" ] || exit 1
# GPU-count conversion: at a launch GPU count other than the recorded one, the
# daemon paces the converted step time, so the job's own train_time scales by
# the logged time_factor (val_loss does not: the curve is unchanged).
FACTOR=$(python3 -I -c "
import json
for l in open('$OCC'):
    e=json.loads(l)
    if e['kind']=='convert' and e.get('time_factor') is not None:
        print(e['time_factor']); break
else: print(1.0)")
echo "time_factor applied by convert(): $FACTOR (launched $NPROC GPUs)"
python3 -I - "$REG/$DRAWN.json" replay.log "$FACTOR" <<'PY'
import json, re, sys
rec = json.load(open(sys.argv[1])); vl = rec["series"]["val_loss"]; tt = rec["series"]["train_time_s"]
factor = float(sys.argv[3])
exp_vl = dict(zip(vl["steps"], vl["values"]))
# converted expectation: steady part scales, the one-off first-step cost does not
steady = rec["timing"]["steady_step_s"]["value"]; first = rec["timing"]["first_step_s"]["value"]
one_off = max(0.0, first - steady)
exp_tt = {s: (0.0 if s == 0 else one_off + steady * factor * s) for s in tt["steps"]}
got_vl, got_tt = {}, {}
for line in open(sys.argv[2], errors="replace"):
    m = re.search(r"step:(\d+)/4875 val_loss:([\d.]+) train_time:([\d.]+)s", line)
    if m and int(m.group(1)) not in got_vl:
        got_vl[int(m.group(1))] = float(m.group(2)); got_tt[int(m.group(1))] = float(m.group(3))
n_ok = sum(1 for s, v in exp_vl.items() if s in got_vl and abs(got_vl[s] - v) < 1e-4)
print(f"val_loss: {n_ok}/{len(exp_vl)} validation steps match the record exactly")
for s in list(exp_vl)[:3] + [4875]:
    print(f"  step {s}: printed {got_vl.get(s)} recorded {exp_vl[s]} | train_time printed {got_tt.get(s)} recorded {exp_tt.get(s)}")
tt_ok = sum(1 for s, v in exp_tt.items() if s in got_tt and abs(got_tt[s] - v) <= max(1.0, 0.02 * v))
print(f"train_time: {tt_ok}/{len(exp_tt)} within 2% (+1 s) of the record")
print("VAL_LOSS_MATCH" if n_ok == len(exp_vl) else "VAL_LOSS_MISMATCH")
PY
echo "--- occupancy: enqueue events per GPU (want ~4875 steps + 40 validations + startup each):"
python3 -I - "$OCC" <<'PY'
import json, sys, collections
c = collections.Counter(); lk = []
for l in open(sys.argv[1]):
    e = json.loads(l)
    if e["kind"] == "enqueue": c[e["gpu"]] += 1
    if e["kind"] == "lookup": lk.append((e.get("rank"), e.get("match"), e.get("draw"), e.get("record_id")))
print("enqueues:", dict(sorted(c.items())))
print("lookups:", lk[:3], "... total", len(lk))
PY
