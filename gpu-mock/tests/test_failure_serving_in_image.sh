#!/usr/bin/env bash
# Served failures end to end inside an image (docs/FAILURE-SERVING.md; the H2
# package). A stand-in training script with `--batch-size` runs 8 steps against
# a one-record registry whose failure is an OOM at step 5 triggered by
# batch size >= 96, with the run's recorded stderr. Checks:
#   1. --batch-size 128: exits 1, prints steps 0..4 and never "done", and its
#      stderr is the recorded text byte for byte (no live traceback);
#   2. --batch-size 64 (same record, trigger disarmed): exits 0, "done";
#   3. no flag (the script's default 32, resolved from its parsed arguments): exits 0;
#   4. CATCH=1 --batch-size 128: the script's own `except torch.OutOfMemoryError`
#      catches the served exception, prints "caught OOM", exits 0, stderr empty;
#   5. occupancy: failures_armed for every launch (armed for 1 and 4, disarmed
#      for 2 and 3), one `failure` event labelled served_failure per fired
#      launch, and job_bye for the failed job (the GPU is freed);
#   6. --fwd --batch-size 128 (a second record whose failure is placed at the
#      forward, with a recorded exception and NO stderr, NVRT_DEBUG unset):
#      exits 1 after steps 0..3, and the LIVE traceback ends at the script's
#      `loss = model(x)` line with no patch frame, last line the recorded message;
#   7. oversubscription: two `--big` launches (a third record holding a 50 GiB
#      peak, no failures) on the same GPU: the first completes, the second
#      OOMs at its first step with PyTorch's wording naming the first's pid.
# Usage (on the VM):  sudo IMG=<box image id> [PATCH_SRC=1] bash tests/test_failure_serving_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
PYSITE="${PYSITE:-$(docker run --rm --entrypoint python3 "$IMG" -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh; unset LD_PRELOAD MOCK_PCI_ROOT
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work; cd /tmp/work
cat > oomer.py <<PY
import argparse, os, torch
p = argparse.ArgumentParser(); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--fwd", action="store_true"); p.add_argument("--big", action="store_true")   # select the second / third record
args = p.parse_args()
torch.manual_seed(0)
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
opt = torch.optim.SGD(model.parameters(), lr=args.lr)
x = torch.randn(args.batch_size, 16, device="cuda")
def run():
    for step in range(8):
        loss = model(x).mean()
        print(f"step {step} loss={loss.item():.4f}", flush=True)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    print("done", flush=True)
if os.environ.get("CATCH") == "1":
    try:
        run()
    except torch.OutOfMemoryError as e:
        print(f"caught OOM: {str(e)[:40]} same_class={torch.OutOfMemoryError is torch.cuda.OutOfMemoryError}", flush=True)
else:
    run()
PY
cat > /tmp/work/recorded.err <<ERR
Traceback (most recent call last):
  File "/tmp/work/oomer.py", line 20, in <module>
    run()
  File "/tmp/work/oomer.py", line 13, in run
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
  File "/usr/local/lib/python3.11/site-packages/torch/_tensor.py", line 648, in backward
    torch.autograd.backward(
  File "/usr/local/lib/python3.11/site-packages/torch/autograd/__init__.py", line 353, in backward
    _engine_run_backward(
  File "/usr/local/lib/python3.11/site-packages/torch/autograd/graph.py", line 824, in _engine_run_backward
    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 1.12 GiB is free. Of the allocated memory 76.51 GiB is allocated by PyTorch, and 1.03 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
ERR
python - <<PY
import hashlib, json
h = hashlib.sha256(open("/tmp/work/oomer.py", "rb").read()).hexdigest()
err = open("/tmp/work/recorded.err").read()
json.dump({"identity": {"record_id": "standin-oom", "entry_hash": h, "argv": ["oomer.py"], "argv_exempt": ["--batch-size"],
                        "match_policy": "exact_only", "replicate_group": "standin-oom"},
           "mode": "replay", "steps": 8, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.4},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
           "series": {"train_loss": {"steps": list(range(1, 9)), "values": [1.0 - 0.05 * i for i in range(8)]}},
           "readiness": {"replay": True},
           "failures": [{"mode": "oom", "step": 5, "time_s": 0.25, "at": "during_training", "provenance": "recorded",
                         "how": "stand-in: the run OOMed at step 5 at batch 128",
                         "cause": {"value": err.strip().splitlines()[-1], "provenance": "recorded", "how": "last stderr line"},
                         "trigger": {"arg": "--batch-size", "op": "ge", "value": 96},
                         "stderr": {"value": err, "provenance": "recorded", "how": "the run'"'"'s stderr"},
                         "replay": {"value": "fire", "how": "the job'"'"'s own"}}]}, open("/tmp/reg/standin.json", "w"))
print("record for", h[:16], "oom at step 5 when --batch-size >= 96")
msg = err.strip().splitlines()[-1].split(": ", 1)[1]
json.dump({"identity": {"record_id": "standin-oom-fwd", "entry_hash": h, "argv": ["oomer.py", "--fwd"], "argv_exempt": ["--batch-size"],
                        "match_policy": "exact_only", "replicate_group": "standin-oom-fwd"},
           "mode": "replay", "steps": 8, "per_step_gpu_s": 0.05,
           "timing": {"steady_step_s": 0.05, "train_time_s": 0.4},
           "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
           "series": {"train_loss": {"steps": list(range(1, 9)), "values": [1.0 - 0.05 * i for i in range(8)]}},
           "readiness": {"replay": True},
           "failures": [{"mode": "oom", "step": 5, "time_s": 0.25, "at": "during_training", "provenance": "derived", "where": "forward",
                         "how": "stand-in: the OOM placed at the forward, message from the record'"'"'s figures, no recorded stderr",
                         "exception": {"type": "torch.OutOfMemoryError", "message": msg},
                         "trigger": {"arg": "--batch-size", "op": "ge", "value": 96},
                         "replay": {"value": "fire", "how": "the job'"'"'s own"}}]}, open("/tmp/reg/standin-fwd.json", "w"))
json.dump({"identity": {"record_id": "standin-big", "entry_hash": h, "argv": ["oomer.py", "--big"], "argv_exempt": ["--batch-size"],
                        "match_policy": "exact_only", "replicate_group": "standin-big"},
           "mode": "replay", "steps": 8, "per_step_gpu_s": 5.0,
           "timing": {"steady_step_s": 5.0, "train_time_s": 40.0},
           "memory": {"peak_allocated_bytes": 50 * 1024 ** 3, "peak": 50 * 1024 ** 3, "weights_bytes": 20 * 1024 ** 3},
           "series": {"train_loss": {"steps": list(range(1, 9)), "values": [1.0 - 0.05 * i for i in range(8)]}},
           "readiness": {"replay": True}, "failures": []}, open("/tmp/reg/standin-big.json", "w"))
print("plus: --fwd (oom at the forward of step 5, no stderr) and --big (50 GiB peak, 8 x 5 s)")
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/occ.jsonl > /tmp/daemon.log 2>&1 &
for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python oomer.py --batch-size 128 > a.out 2> a.err; ra=$?
python oomer.py --batch-size 64 > b.out 2> b.err; rb=$?
python oomer.py > c.out 2> c.err; rc=$?
CATCH=1 python oomer.py --batch-size 128 > d.out 2> d.err; rd=$?
env -u NVRT_DEBUG python oomer.py --fwd --batch-size 128 > e.out 2> e.err; re=$?
# oversubscription: the first --big holds 50 GiB for ~4 s of paced steps (speed 10); the second arrives 1.5 s in
python oomer.py --big > f1.out 2> f1.err & p1=$!
sleep 1.5
python oomer.py --big > f2.out 2> f2.err; rf2=$?
wait $p1; rf1=$?
sleep 1
fails=0; n=0
chk() { n=$((n+1)); if [ "$2" = 1 ]; then echo "[ok] $1"; else echo "[FAIL] $1: $3"; fails=$((fails+1)); fi; }
# NVRT_DEBUG notes (nvrt replay[r0]: ...) go to stderr; compare with only those removed, so any
# other line the box adds (a torch warning, a patch frame) fails the byte-for-byte check
grep -v "^nvrt " a.err > a.err.clean
chk "1: --batch-size 128 exits 1" "$([ $ra = 1 ] && echo 1 || echo 0)" "rc=$ra"
chk "1: steps 0..4 printed, never done" "$( (grep -q "^step 4 " a.out && ! grep -q "^step 5 " a.out && ! grep -q done a.out) && echo 1 || echo 0)" "$(tr "\n" "|" < a.out | cut -c1-200)"
chk "1: stderr is the recorded text byte for byte (no live traceback)" "$(cmp -s a.err.clean /tmp/work/recorded.err && echo 1 || echo 0)" "$(diff a.err.clean /tmp/work/recorded.err | head -5 | tr "\n" "|")"
chk "2: --batch-size 64 (same record, disarmed) exits 0 and finishes" "$([ $rb = 0 ] && grep -q done b.out && echo 1 || echo 0)" "rc=$rb $(tail -2 b.err | tr "\n" "|")"
chk "3: default batch size 32 (parsed arguments) exits 0 and finishes" "$([ $rc = 0 ] && grep -q done c.out && echo 1 || echo 0)" "rc=$rc $(tail -2 c.err | tr "\n" "|")"
chk "4: the script'"'"'s own except torch.OutOfMemoryError catches it, exits 0, stderr clean" "$([ $rd = 0 ] && grep -q "caught OOM" d.out && ! grep -v "^nvrt " d.err | grep -q . && echo 1 || echo 0)" "rc=$rd $(grep caught d.out) $(grep -v "^nvrt " d.err | head -3 | tr "\n" "|")"
chk "4: torch.OutOfMemoryError is torch.cuda.OutOfMemoryError in the box" "$(grep -q "same_class=True" d.out && echo 1 || echo 0)" "$(grep caught d.out)"
chk "6: --fwd --batch-size 128 exits 1 after steps 0..3 (the forward of step 5 raises)" "$([ $re = 1 ] && grep -q "^step 3 " e.out && ! grep -q "^step 4 " e.out && echo 1 || echo 0)" "rc=$re $(tr "\n" "|" < e.out | cut -c1-120)"
chk "6: the live traceback ends at the script forward line, no patch frame, the recorded message last" "$( (head -1 e.err | grep -q "^Traceback (most recent call last):" && grep -q "oomer.py" e.err && grep -q "loss = model(x)" e.err && ! grep -q "/usr/local/nvidia" e.err && tail -1 e.err | grep -q "^torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB") && echo 1 || echo 0)" "$(head -12 e.err | tr "\n" "|" | cut -c1-600)"
chk "7: the first --big completes" "$([ $rf1 = 0 ] && grep -q done f1.out && echo 1 || echo 0)" "rc=$rf1 $(tail -2 f1.err | tr "\n" "|")"
chk "7: the second --big OOMs at its first step with PyTorch wording naming the pid of the first" "$([ $rf2 = 1 ] && tail -1 f2.err | grep -q "^torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20480.00 MiB. GPU 0 has a total capacity of 80.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 30.00 GiB memory in use. Process $p1 has 50.00 GiB memory in use." && ! grep -q "^step 1 " f2.out && echo 1 || echo 0)" "rc=$rf2 pid1=$p1 $(tail -1 f2.err | cut -c1-300)"
python - <<PY
import json, sys
# NOTE: this heredoc sits inside the single-quoted bash -c body: no single quotes anywhere in it
ev = [json.loads(l) for l in open("/tmp/occ.jsonl")]
def tag(ok): return "[ok]" if ok else "[FAIL]"
fa = [(e.get("armed"), e.get("disarmed"), e.get("unresolved")) for e in ev if e["kind"] == "failures_armed"]
fe = [e for e in ev if e["kind"] == "failure" and e.get("label") == "served_failure"]
ov = [e for e in ev if e["kind"] == "failure" and e.get("label") == "oversubscribed"]
byes = {e["job"] for e in ev if e["kind"] == "job_bye"}
ok1 = fa == [([0], [], []), ([], [0], []), ([], [0], []), ([0], [], []), ([0], [], [])]
ok2 = len(fe) == 3 and all(e.get("step") == 5 for e in fe) and [e.get("record_id") for e in fe] == ["standin-oom", "standin-oom", "standin-oom-fwd"]
ok3 = all(e["job"] in byes for e in fe + ov)
ok4 = len(ov) == 1 and ov[0].get("mine_bytes") == 50 * 1024 ** 3 and [o.get("bytes") for o in ov[0].get("others") or []] == [50 * 1024 ** 3] and ov[0].get("step") == 1
holds = [e.get("bytes") for e in ev if e["kind"] == "hold"]
ok5 = holds == [20 * 1024 ** 3, 20 * 1024 ** 3]
fe_s = [(e.get("mode"), e.get("record_id"), e.get("step"), e.get("exception")) for e in fe]
ov_s = [(e.get("job"), e.get("mine_bytes"), e.get("others"), e.get("used_bytes")) for e in ov]
print(tag(ok1), "5: failures_armed per launch armed/disarmed/disarmed/armed/armed-fwd:", fa)
print(tag(ok2), "5: three served_failure events (launch 1, 4, fwd) at step 5:", fe_s)
print(tag(ok3), "5: every failed job said job_bye (GPU freed): failed=", [e["job"] for e in fe + ov], "byes=", sorted(byes))
print(tag(ok4), "7: one oversubscribed failure event at step 1, 50 GiB arriving beside 50 GiB:", ov_s)
print(tag(ok5), "7: both --big jobs held their 20 GiB of weights from engagement (hold events), the rest at step 1:", holds)
sys.exit(0 if ok1 and ok2 and ok3 and ok4 and ok5 else 1)
PY
[ $? = 0 ] || fails=$((fails+1)); n=$((n+5))
echo "=== a.err (first 3 / last 2 lines)"; head -3 a.err; tail -2 a.err
echo "$((n-fails)) passed, $fails failed"
exit $fails
'
