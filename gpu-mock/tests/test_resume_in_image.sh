#!/usr/bin/env bash
# Save-then-resume round trip under replay (gpu-tasks' segment chains,
# PACKAGES-2026-10-20 §8), inside an image with gpu-mock installed. A stand-in
# for the real segment script, until gpu-tasks sends one:
#   A: builds a small model on the fake GPU, replays k steps, saves model and
#      optimizer state dicts and its step;
#   B: builds the same model, loads A's file, replays k more steps.
# The daemon runs in the container with a one-record registry written here,
# keyed to the stand-in script's bytes, so both processes match exactly. Checks:
#   - both processes engage replay (two exact lookups) and exit 0;
#   - A's served losses are the record's steps 1..k, and B's are steps 1..k
#     AGAIN: the per-op path counts a process's own optimizer steps from zero,
#     so a real chain gives each segment its own record, re-based to 1..k;
#   - B loads with weights_only=True (torch >= 2.6's default) and map_location
#     cuda, so the file must hold plain tensors, not FakeGPU's class;
#   - the weights B loads equal the weights A saved (a checksum), and they are
#     A's initial weights, since a replayed step changes nothing;
#   - the saved optimizer state is empty (the real step never ran) and B's
#     load_state_dict accepts it;
#   - A's and B's fingerprints (dry-run mode) are identical.
# Usage (on the VM):  sudo IMG=gm-dpo:x86 [PATCH_SRC=1] bash tests/test_resume_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-gm-dpo:x86}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work; cd /tmp/work
cat > seg.py <<PY
import hashlib, os, sys, torch
K = 5
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
def checksum():
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]
seg = os.environ["SEG"]; start = 0
if seg == "b":
    ck = torch.load("/tmp/work/seg-a.pt", map_location="cuda", weights_only=True)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); start = ck["step"]
    print("B loaded: step", start, "| weights", checksum(), "| optimizer state entries", len(ck["optimizer"]["state"]), flush=True)
else:
    print("A initial weights", checksum(), flush=True)
x = torch.randn(8, 16, device="cuda")
for i in range(K):
    loss = model(x).mean()
    loss.backward()
    opt.step(); opt.zero_grad(set_to_none=True)
    print(f"{seg} step {start + i + 1} loss {loss.item():.4f}", flush=True)
if seg == "a":
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": K}, "/tmp/work/seg-a.pt")
    print("A saved: weights", checksum(), "| optimizer state entries", len(opt.state_dict()["state"]), flush=True)
PY
python - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/seg.py", "rb").read()).hexdigest()
vals = [round(2.0 - 0.1 * s, 4) for s in range(1, 11)]
json.dump({"identity": {"record_id": "standin-segment", "entry_hash": h, "argv": ["seg.py"],
                        "match_policy": "exact_only", "replicate_group": "standin-segment"},
           "mode": "replay", "steps": 10, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.5},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 30, "peak": 2 * 1024 ** 3},
           "series": {"train_loss": {"steps": list(range(1, 11)), "values": vals}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/standin.json", "w"))
print("record for", h[:16], "losses", vals)
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ.jsonl > /tmp/daemon.log 2>&1 &
for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
SEG=a python seg.py > a.out 2> a.err; ra=$?
SEG=b python seg.py > b.out 2> b.err; rb=$?
SEG=a NVRT_FINGERPRINT_ONLY=1 python seg.py 2>/dev/null | grep -o "\"fingerprint\":\"[^\"]*\"" > fa.txt
SEG=b NVRT_FINGERPRINT_ONLY=1 python seg.py 2>/dev/null | grep -o "\"fingerprint\":\"[^\"]*\"" > fb.txt
pass=0; fail=0
chk() { if [ "$2" = "$3" ]; then pass=$((pass+1)); echo "[ok  ] $1 -> $2"; else fail=$((fail+1)); echo "[FAIL] $1 -> $2 (want $3)"; fi; }
echo "--- A:"; cat a.out; grep -o "nvrt replay.*" a.err | grep -E "lookup|engaged" | head -2
echo "--- B:"; cat b.out; grep -o "nvrt replay.*" b.err | grep -E "lookup|engaged" | head -2
chk "A exit" "$ra" 0
chk "B exit" "$rb" 0
chk "A engaged replay on the exact key" "$(grep -c "lookup\[exact\] -> mode=replay" a.err)" 1
chk "B engaged replay on the exact key" "$(grep -c "lookup\[exact\] -> mode=replay" b.err)" 1
chk "A served losses = record steps 1..5" "$(grep -oE "a step [0-9]+ loss [0-9.]+" a.out | awk "{print \$5}" | paste -sd,)" "1.9000,1.8000,1.7000,1.6000,1.5000"
chk "B served losses = record steps 1..5 again (own count from zero)" "$(grep -oE "b step [0-9]+ loss [0-9.]+" b.out | awk "{print \$5}" | paste -sd,)" "1.9000,1.8000,1.7000,1.6000,1.5000"
chk "B counts its steps from the loaded step" "$(grep -oE "b step [0-9]+" b.out | head -1)" "b step 6"
ini=$(grep -o "A initial weights [0-9a-f]*" a.out | awk "{print \$4}"); sav=$(grep -o "A saved: weights [0-9a-f]*" a.out | awk "{print \$4}"); ld=$(grep -o "B loaded: step [0-9]* | weights [0-9a-f]*" b.out | awk "{print \$7}")
chk "A saved its initial weights (replayed steps changed nothing)" "$sav" "$ini"
chk "B loaded exactly what A saved" "$ld" "$sav"
chk "the checkpoint names no fake class" "$(python -c "import zipfile,sys; z=zipfile.ZipFile(\"/tmp/work/seg-a.pt\"); d=b\"\".join(z.read(n) for n in z.namelist() if n.endswith(\"data.pkl\")); print(b\"fakegpu\" in d or b\"FakeCuda\" in d)")" False
chk "saved optimizer state is empty" "$(grep -o "optimizer state entries [0-9]*" a.out | awk "{print \$4}")" 0
chk "B accepted the empty optimizer state" "$(grep -c "B loaded" b.out)" 1
chk "fingerprints of A and B are identical" "$( [ -s fa.txt ] && cmp -s fa.txt fb.txt && echo same || echo differ )" same
chk "occupancy: two exact replay lookups" "$(grep "\"kind\":\"lookup\"" /tmp/occ.jsonl | grep -c "\"match\":\"exact\"")" 2
chk "occupancy: ten paced steps" "$(grep -c "\"kind\":\"enqueue\"" /tmp/occ.jsonl)" 10
echo; echo "$pass passed, $fail failed"; [ "$fail" = 0 ]'
