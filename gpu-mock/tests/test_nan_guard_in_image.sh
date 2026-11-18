#!/usr/bin/env bash
# NaN guards and comparisons on a served loss, under replay, inside an image.
# 33 of the 86 mng3 scripts run `if not torch.isfinite(loss).all(): raise`
# every micro-step; on 2026-10-21 that raised at step 0 on the mng3 image
# (the study's speed-10 checks): the record had no train loss and its first
# validation was at step 125, so the daemon served the default 0.0, and the
# predicate's result had inherited the loss role, so bool() answered 0.0.
# A stand-in script runs 4 steps against a one-record registry whose val_loss
# starts at step 2 and which has no train loss. Checks:
#   - isfinite/isnan/>/!= on the loss answer as on a GPU, every step;
#   - loss.item() is the nearest val loss (3.5) from step 0, never 0.0;
#   - float(torch.isfinite(loss)) is 1.0 (a bool result is its own data);
#   - the job exits 0 and reaches the end;
#   - NVRT_PROBE_STEPS=2 ends the job after optimizer step 2 with the marker.
# Usage (on the VM):  sudo IMG=ml-node:py311-torch2110 [PATCH_SRC=1] bash tests/test_nan_guard_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work; cd /tmp/work
cat > guard.py <<PY
import os, torch
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(8, 16, device="cuda")
for step in range(4):
    loss = model(x).mean()
    print(f"step {step} isfinite={bool(torch.isfinite(loss).all())} nan={bool(torch.isnan(loss))} "
          f"gt100={bool(loss > 100)} lt100={bool(loss < 100)} selfne={bool(loss != loss)} "
          f"item={loss.item():.4f} finite_float={float(torch.isfinite(loss)):.1f} "
          f"half={(loss / 2).item():.4f}", flush=True)
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"non-finite train loss at step {step}: {loss.item()}")
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    if os.environ.get("RAISE_AFTER") == str(step):
        raise RuntimeError("boom from the script itself")
print("done", flush=True)
PY
python - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/guard.py", "rb").read()).hexdigest()
json.dump({"identity": {"record_id": "standin-guard", "entry_hash": h, "argv": ["guard.py"],
                        "match_policy": "exact_only", "replicate_group": "standin-guard"},
           "mode": "replay", "steps": 4, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.2},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
           "series": {"val_loss": {"steps": [2, 4], "values": [3.5, 3.0]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/standin.json", "w"))
print("record for", h[:16], "val_loss at steps 2, 4; no train loss")
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ.jsonl > /tmp/daemon.log 2>&1 &
for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python guard.py > g.out 2> g.err; rg=$?
NVRT_PROBE_STEPS=2 python guard.py > p.out 2> p.err; rp=$?
RAISE_AFTER=1 python guard.py > x.out 2> x.err; rx=$?
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = "$3" ]; then echo "[ok] $1: $2"; else echo "[FAIL] $1: got $2 want $3"; fails=$((fails+1)); fi; }
echo "--- guard run (exit $rg):"; cat g.out; grep -o "nvrt replay.*" g.err | grep -E "lookup|engaged|predicate" | head -3
chk "guard run exits 0" "$rg" 0
chk "guard run reaches the end" "$(grep -c "^done" g.out)" 1
chk "no raise" "$(grep -c "non-finite" g.err)" 0
chk "isfinite True every step" "$(grep -c "isfinite=True" g.out)" 4
chk "isnan False every step" "$(grep -c " nan=False" g.out)" 4
chk "loss > 100 False every step" "$(grep -c "gt100=False" g.out)" 4
chk "loss < 100 True every step" "$(grep -c "lt100=True" g.out)" 4
chk "loss != loss False every step" "$(grep -c "selfne=False" g.out)" 4
chk "item() is the nearest val loss 3.5 from step 0" "$(grep -c "item=3.5000" g.out)" 4
chk "float(isfinite) is 1.0" "$(grep -c "finite_float=1.0" g.out)" 4
chk "(loss/2).item() still served (role kept on float ops)" "$(grep -c "half=3.5000" g.out)" 4
chk "daemon served step 0 from the val loss (val_derived), not default" "$(grep -c "\"step\":0.*\"tier\":\"val_derived\"" /tmp/occ.jsonl | awk "{print (\$1>0)?1:0}")" 1
chk "no default-tier materialisation" "$(grep -c "\"tier\":\"default\"" /tmp/occ.jsonl)" 0
echo "--- probe run (exit $rp):"; cat p.out; grep -o "nvrt replay.*probe.*" p.err | head -2
chk "probe run exits 0" "$rp" 0
chk "probe marker at optimizer step 2" "$(grep -c "probe: reached optimizer step 2" p.err)" 1
chk "probe run stops before the end" "$(grep -c "^done" p.out)" 0
chk "probe run printed steps 0 and 1 only" "$(grep -c "^step" p.out)" 2
echo "--- raising run (exit $rx):"; grep -o "RuntimeError.*" x.err | head -1
chk "raising run exits non-zero" "$([ "$rx" != 0 ] && echo 1 || echo 0)" 1
chk "daemon logged the exception with its type" "$(grep -c "\"kind\":\"exception\".*\"etype\":\"RuntimeError\"" /tmp/occ.jsonl)" 1
chk "exception event names the step" "$(grep -c "\"kind\":\"exception\".*\"step\":2" /tmp/occ.jsonl)" 1
RL=$(grep -n "boom from the script" guard.py | cut -d: -f1 | head -1)
echo "--- exception event: $(grep "\"kind\":\"exception\"" /tmp/occ.jsonl | grep -oE "\"etype\".*" | cut -c1-200)"
chk "exception event carries the innermost frame: guard.py at the raise line" "$(grep -c "\"kind\":\"exception\".*\"frame_file\":\"[^\"]*guard.py\".*\"frame_line\":$RL" /tmp/occ.jsonl)" 1
chk "enqueue events carry step and unit" "$(grep "\"kind\":\"enqueue\"" /tmp/occ.jsonl | grep -c "\"unit\":\"step\"" | awk "{print (\$1>0)?1:0}")" 1
echo "$((n-fails)) passed, $fails failed"
[ "$fails" = 0 ]'
