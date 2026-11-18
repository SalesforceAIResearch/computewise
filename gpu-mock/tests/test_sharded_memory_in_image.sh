#!/usr/bin/env bash
# next/ (after r6): memory.sharded_bytes. A record recorded on 4 GPUs whose per-card
# peak is 40 GiB, 30 of it sharded state (ZeRO-3 / FSDP), weights 20 GiB all
# sharded. One daemon (4 GPUs, 80 GiB each, speed 10); the stand-in script is
# launched at three widths, each rank a process with the environment torchrun
# gives it (WORLD_SIZE, RANK, LOCAL_RANK; the script does not init a process
# group, so no rendezvous is needed in a box without a network):
#   W4  --nproc 4: every rank holds 40 GiB, all complete, exit 0;
#   W2  --nproc 2: every rank holds 10 + 60 = 70 GiB, fits, exit 0;
#   W1  one process: 10 + 120 = 130 GiB on one card: torch.OutOfMemoryError at
#       its first step in PyTorch's wording, exit 1, and the convert event logs
#       memory_factor 3.25 with the sharded provenance;
#   P   a record without the field at 1 GPU completes (its 40 GiB is unchanged).
# Usage (on the E2):  sudo IMG=<box> G=<tree> [PATCH_SRC=0: the image's installed patch; default 1, the tree's mounted] bash tests/test_sharded_memory_in_image.sh
set -uo pipefail
IMG=${IMG:-sha256:5994258c9617cee5b4db7566f93afbd1c020d2e480b5ce4d120f0969b07f39ee}
G=${G:-$(cd "$(dirname "$0")/.." && pwd)}
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-1}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 6 --memory 12g --network none "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh 2>/dev/null || true
export NVRT_SOCKET=/tmp/gm.sock
mkdir -p /tmp/reg /tmp/work /tmp/out; cd /tmp/work
cat > fsdp.py <<PY
import argparse, os, torch
p = argparse.ArgumentParser(); p.add_argument("--plain", action="store_true"); args = p.parse_args()
lr = int(os.environ.get("LOCAL_RANK", "0")); torch.cuda.set_device(lr)
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
x = torch.randn(8, 16, device="cuda")
for step in range(6):
    loss = model(x).mean()
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
print(f"rank {lr} done", flush=True)
PY
python - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/fsdp.py", "rb").read()).hexdigest()
G = 2 ** 30
def rec(rid, argv, sharded):
    m = {"peak_allocated_bytes": 40 * G, "peak": 40 * G, "weights_bytes": 20 * G}
    if sharded:
        m["sharded_bytes"] = {"value": 30 * G, "provenance": "derived", "how": "test: parameters, gradients and Adam state split 4 ways"}
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "match_policy": "exact_only", "replicate_group": rid},
            "mode": "replay", "steps": 6, "per_step_gpu_s": 0.5,
            "timing": {"steady_step_s": 0.5, "train_time_s": 3.0},
            "launch": {"world_size": {"value": 4, "provenance": "recorded"},
                       "batch_policy": {"value": "fixed_global_grow_accumulation", "provenance": "derived"}},
            "memory": m, "series": {"train_loss": {"steps": list(range(1, 7)), "values": [1.0 - 0.05 * i for i in range(6)]}},
            "readiness": {"replay": True}, "failures": []}
json.dump(rec("standin-sharded", ["fsdp.py"], True), open("/tmp/reg/sharded.json", "w"))
json.dump(rec("standin-plain", ["fsdp.py", "--plain"], False), open("/tmp/reg/plain.json", "w"))
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occ.jsonl > /tmp/out/d.log 2>&1 &
D=$!; for i in $(seq 100); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
run() { local t=$1; shift; (timeout 300 "$@" > /tmp/out/$t.out 2>&1); echo "$t rc=$?" >> /tmp/out/rc.txt; }
ranks() {  # ranks <tag> <width> [args]: every rank of that width at once, rc = the worst
  local t=$1 w=$2; shift 2; local pids=() rc=0
  for r in $(seq 0 $((w-1))); do
    (WORLD_SIZE=$w RANK=$r LOCAL_RANK=$r MASTER_PORT=29500 TORCHELASTIC_RUN_ID=$t timeout 300 python fsdp.py "$@" >> /tmp/out/$t.out 2>&1) & pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p || rc=$?; done
  echo "$t rc=$rc" >> /tmp/out/rc.txt
}
ranks W4 4
ranks W2 2
run W1 env WORLD_SIZE=1 python fsdp.py
run P env WORLD_SIZE=1 python fsdp.py --plain
kill $D 2>/dev/null; wait $D 2>/dev/null
cat /tmp/out/rc.txt
echo "W1_OOM $(grep -c "torch.OutOfMemoryError: CUDA out of memory" /tmp/out/W1.out)"
grep -m1 -o "Tried to allocate [0-9.]* MiB. GPU 0 has a total capacity of [0-9.]* GiB" /tmp/out/W1.out | sed "s/^/W1_MSG /"
echo "DONE_W4 $(grep -c "done" /tmp/out/W4.out) DONE_W2 $(grep -c "done" /tmp/out/W2.out) DONE_P $(grep -c "done" /tmp/out/P.out) OOM_W4 $(grep -c OutOfMemoryError /tmp/out/W4.out) OOM_W2 $(grep -c OutOfMemoryError /tmp/out/W2.out)"
python3 - <<PY
import json
ev = [json.loads(l) for l in open("/tmp/out/occ.jsonl")]
for e in ev:
    if e.get("kind") == "convert":
        print("CONVERT", e.get("record_id"), e.get("from_gpus"), e.get("to_gpus"), round(e.get("memory_factor") or 0, 4),
              round(e.get("weights_factor") or 0, 4) if e.get("weights_factor") else None, (e.get("memory_provenance") or "")[:40])
fl = [e for e in ev if e.get("kind") == "failure"]
print("FAILURES", [(e.get("label"), e.get("mode"), round((e.get("mine_bytes") or 0) / 2 ** 30, 1)) for e in fl])
held = {}
for e in ev:
    if e.get("kind") == "enqueue" and e.get("used_bytes") is not None:
        held[e["gpu"]] = max(held.get(e["gpu"], 0), e["used_bytes"])
print("PEAK_HELD_GIB", {g: round(b / 2 ** 30, 1) for g, b in sorted(held.items())})
PY
'
