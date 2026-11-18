#!/usr/bin/env bash
# A job matched by the brief's path (gpu-tasks' prepared records: the agent
# writes the script, the record binds to its path) must have its live
# fingerprint checked against the record's at the first optimizer step and the
# daemon must log a `fingerprint_check` event either way. Until 2026-10-21 it
# never did: the structural stage lived in an optimizer step hook, and under
# replay the step is neutralised before any hook runs (Q-prepP6-haiku-L0-nosub:
# twelve path matches, zero checks). Two runs of one stand-in against a
# one-record registry whose entry hash does not match (so only the path can):
#   1. the record claims a different structure (99 layers, SGD): the event must
#      say the fingerprints disagree and name num_layers and optimizers;
#   2. the record carries the live fingerprint and inputs read from run 1: the
#      event must say they agree.
# Usage (on the VM):  sudo IMG=ml-node:py311-torch2110 [PATCH_SRC=1] bash tests/test_path_bound_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work/runs/standin; cd /tmp/work
cat > runs/standin/train.py <<PY
import torch
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
x = torch.randn(8, 16, device="cuda")
for step in range(3):
    loss = model(x).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    print(f"step {step} loss {loss.item():.4f}", flush=True)
print("done", flush=True)
PY
write_record() {  # $1 fingerprint  $2 inputs-json
python - "$1" "$2" <<PY
import json, sys
json.dump({"identity": {"record_id": "standin-pathbound", "entry_hash": "0" * 64, "argv": ["train.py"],
                        "match_policy": "path_bound", "bound_path": "runs/standin/train.py",
                        "replicate_group": "standin-pathbound", "fingerprint": sys.argv[1],
                        "fingerprint_inputs": json.loads(sys.argv[2])},
           "mode": "replay", "steps": 3, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.15},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
           "series": {"train_loss": {"steps": [1, 2, 3], "values": [2.0, 1.9, 1.8]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/standin.json", "w"))
PY
}
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = "$3" ]; then echo "[ok] $1: $2"; else echo "[FAIL] $1: got $2 want $3"; fails=$((fails+1)); fi; }
# run 1: the record claims a different structure
write_record "fp256:0000000000000000" "{\"model\": \"gpt\", \"num_layers\": 99, \"optimizers\": [\"SGD\"]}"
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ1.jsonl > /tmp/daemon1.log 2>&1 &
D1=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python runs/standin/train.py > r1.out 2> r1.err; r1=$?
for i in $(seq 100); do grep -q "\"kind\":\"job_bye\"" /tmp/occ1.jsonl 2>/dev/null && break; sleep 0.1; done   # logged when the socket closes
kill $D1 2>/dev/null; wait $D1 2>/dev/null; rm -f /tmp/gm.sock
echo "--- run 1 (exit $r1):"; cat r1.out; grep -o "nvrt replay.*" r1.err | grep -E "lookup|fingerprint" | head -3
E1=$(grep "\"kind\":\"fingerprint_check\"" /tmp/occ1.jsonl | head -1); echo "    event: ${E1:0:400}"
chk "run 1 exits 0 and finishes" "$r1:$(grep -c "^done" r1.out)" "0:1"
chk "run 1 matched by path" "$(grep -c "\"kind\":\"lookup\".*\"match\":\"path\"" /tmp/occ1.jsonl)" 1
chk "run 1 logged a fingerprint_check" "$(grep -c "\"kind\":\"fingerprint_check\"" /tmp/occ1.jsonl)" 1
chk "run 1: fingerprints disagree" "$(echo "$E1" | grep -c "\"agree\":false")" 1
chk "run 1: the event carries the live fingerprint" "$(echo "$E1" | grep -c "\"fingerprint_live\":\"fp256:")" 1
chk "run 1: the check names num_layers and optimizers" "$(echo "$E1" | grep -c "num_layers.*optimizers\|optimizers.*num_layers")" 1
# run 2: the record carries the live fingerprint and inputs
# the live fingerprint and inputs, from a dry run of the same script (the daemon event
# logs the live fingerprint but not its inputs)
LIVE=$(NVRT_FINGERPRINT_ONLY=1 python runs/standin/train.py 2>/dev/null | grep "^NVRT_FINGERPRINT " | head -1 | cut -c18-)
FP=$(python -c "import json,sys; print(json.loads(sys.argv[1])[\"fingerprint\"])" "$LIVE"); IN=$(python -c "import json,sys; print(json.dumps(json.loads(sys.argv[1])[\"inputs\"]))" "$LIVE")
echo "    live fingerprint: $FP"
write_record "$FP" "$IN"
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ2.jsonl > /tmp/daemon2.log 2>&1 &
D2=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python runs/standin/train.py > r2.out 2> r2.err; r2=$?
for i in $(seq 100); do grep -q "\"kind\":\"job_bye\"" /tmp/occ2.jsonl 2>/dev/null && break; sleep 0.1; done
kill $D2 2>/dev/null; wait $D2 2>/dev/null
E2=$(grep "\"kind\":\"fingerprint_check\"" /tmp/occ2.jsonl | head -1); echo "--- run 2 (exit $r2): event: ${E2:0:300}"
chk "run 2 exits 0" "$r2" 0
chk "run 2 logged a fingerprint_check" "$(grep -c "\"kind\":\"fingerprint_check\"" /tmp/occ2.jsonl)" 1
chk "run 2: fingerprints agree" "$(echo "$E2" | grep -c "\"agree\":true")" 1
chk "run 1 job_bye says fingerprint_differs true" "$(grep -c "\"kind\":\"job_bye\".*\"fingerprint_differs\":true" /tmp/occ1.jsonl)" 1
chk "run 2 job_bye says fingerprint_differs false" "$(grep -c "\"kind\":\"job_bye\".*\"fingerprint_differs\":false" /tmp/occ2.jsonl)" 1
chk "served losses are the record (2.0, 1.9, 1.8) in both runs" "$(cat r1.out r2.out | grep -c "loss 2.0000\|loss 1.9000\|loss 1.8000")" 6
echo "$((n-fails)) passed, $fails failed"
[ "$fails" = 0 ]'
