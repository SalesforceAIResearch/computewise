#!/usr/bin/env bash
# In-process trials in the box (2026-10-22): a modded-nanogpt-style stand-in
# runs `num_trials = int(sys.argv[-1])` trials in one process, each a new
# model and optimizer with steps from 0. Each trial must be served its own
# record (the rerun rule), paced from that record's step 1, with every event
# tagged `trial`. Before the fix trial 2 read the last value of trial 1's
# record for every step (newton-muon on c3-12).
# Usage (on the VM):  sudo IMG=sha256:<box id> [PATCH_SRC=1] bash tests/test_trials_in_image.sh
set -uo pipefail
G=$(cd "$(dirname "$0")/.." && pwd)
IMG="${IMG:-ml-node:py311-torch2110}"
M=(-v "$G/src:/gm/src:ro"); [ "${PATCH_SRC:-0}" = 1 ] && M+=(-v "$G/src/torch_patch:/usr/local/nvidia/python:ro" -v "$G/src/torch_patch/sitecustomize.py:${PYSITE:-/usr/local/lib/python3.11/site-packages}/sitecustomize.py:ro")
# OUT=<host dir>: keep the run's occupancy.jsonl, lr_stream.jsonl, daemon log and
# stdout on the host (the runner's per-trial known answer, 2026-10-22)
if [ -n "${OUT:-}" ]; then mkdir -p "$OUT"; chmod 777 "$OUT"; M+=(-v "$OUT:/tmp/out"); fi
docker run --entrypoint '' --rm --cpus 4 --memory 8g "${M[@]}" "$IMG" bash -c '
source /usr/local/nvidia/env.sh
export NVRT_SOCKET=/tmp/gm.sock NVRT_DEBUG=1
mkdir -p /tmp/reg /tmp/work /tmp/out; cd /tmp/work
rm -f /tmp/out/*.jsonl /tmp/out/*.out /tmp/out/*.err /tmp/out/*.log /tmp/out/*.json; rm -rf /tmp/out/artifacts   # a reused OUT dir starts clean
cat > train.py <<PY
import sys, torch
num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1
for trial in range(num_trials):
    torch.manual_seed(trial)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(8, 16, device="cuda")
    for step in range(3):
        loss = model(x).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        print(f"trial {trial + 1} step {step} loss {loss.item():.4f}", flush=True)
print("done", flush=True)
PY
cat > inplace.py <<PY
import sys, torch
num_trials = int(sys.argv[-1]) if len(sys.argv) > 1 else 1
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
x = torch.randn(8, 16, device="cuda")
for trial in range(num_trials):
    torch.manual_seed(trial)
    for name, p in model.named_parameters():          # re-initialise in place, as newton-muon does
        w = p.data
        if name.endswith("bias"): w.zero_()
        else: w.normal_(std=0.02)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(3):
        loss = model(x).mean()
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        print(f"trial {trial + 1} step {step} loss {loss.item():.4f}", flush=True)
print("done", flush=True)
PY
cat > phase.py <<PY
import torch
model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 1)).cuda()
x = torch.randn(8, 16, device="cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(3):
    loss = model(x).mean(); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    print(f"phase 1 step {step} loss {loss.item():.4f}", flush=True)
opt = torch.optim.SGD(model.parameters(), lr=1e-2)      # a phase change: same parameters, no re-init
for step in range(3, 6):
    loss = model(x).mean(); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    print(f"phase 2 step {step} loss {loss.item():.4f}", flush=True)
print("done", flush=True)
PY
EH=$(sha256sum train.py | cut -c1-64); EH2=$(sha256sum inplace.py | cut -c1-64); EH3=$(sha256sum phase.py | cut -c1-64)
python - "$EH" "$EH2" "$EH3" <<PY
import json, sys
eh, eh2, eh3 = sys.argv[1:4]
def rec(rid, h, argv, group, vals, steps=3):
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "match_policy": "exact_only", "replicate_group": group},
            "mode": "replay", "steps": steps, "per_step_gpu_s": 0.05, "timing": {"steady_step_s": 0.05, "first_step_s": 0.5},
            "memory": {"peak_allocated_bytes": 2 * 1024 ** 3, "peak": 2 * 1024 ** 3},
            "series": {"train_loss": {"steps": list(range(1, steps + 1)), "values": vals}},
            "readiness": {"replay": True}, "failures": []}
for i, vals in enumerate(([2.0, 1.9, 1.8], [3.0, 2.9, 2.8])):
    json.dump(rec(f"standin-run{i}", eh, ["train.py", "2"], "standin-trials", vals), open(f"/tmp/reg/run{i}.json", "w"))
for i, vals in enumerate(([4.0, 3.9, 3.8], [5.0, 4.9, 4.8])):
    json.dump(rec(f"inplace-run{i}", eh2, ["inplace.py", "2"], "inplace-trials", vals), open(f"/tmp/reg/inplace{i}.json", "w"))
json.dump(rec("phase-run0", eh3, ["phase.py"], "phase", [6.0, 5.9, 5.8, 5.7, 5.6, 5.5], steps=6), open("/tmp/reg/phase0.json", "w"))
json.dump(rec("phase-run1", eh3, ["phase.py"], "phase", [7.0, 6.9, 6.8, 6.7, 6.6, 6.5], steps=6), open("/tmp/reg/phase1.json", "w"))
PY
python -I /gm/src/daemon/gpu_daemon.py --socket /tmp/gm.sock --registry /tmp/reg --gpus 4 --speed 10 --occupancy /tmp/out/occupancy.jsonl > /tmp/out/daemon.log 2>&1 &
D=$!; for i in $(seq 50); do [ -S /tmp/gm.sock ] && break; sleep 0.1; done
python train.py 2 > r.out 2> r.err; rc=$?; cp r.out /tmp/out/stdout.txt; cp r.err /tmp/out/stderr.txt; cp train.py /tmp/out/train.py; cp /tmp/reg/*.json /tmp/out/ 2>/dev/null
for i in $(seq 100); do grep -q "\"kind\":\"job_bye\"" /tmp/out/occupancy.jsonl 2>/dev/null && break; sleep 0.1; done
python inplace.py 2 > i.out 2> i.err; ri=$?
for i in $(seq 100); do [ "$(grep -c "\"kind\":\"job_bye\"" /tmp/out/occupancy.jsonl 2>/dev/null)" -ge 2 ] && break; sleep 0.1; done
python phase.py > p.out 2> p.err; rp=$?
for i in $(seq 100); do [ "$(grep -c "\"kind\":\"job_bye\"" /tmp/out/occupancy.jsonl 2>/dev/null)" -ge 3 ] && break; sleep 0.1; done
kill $D 2>/dev/null; wait $D 2>/dev/null
echo "--- run (exit $rc):"; cat r.out; grep -o "nvrt replay.*trial.*" r.err | head -3
echo "--- in-place (exit $ri):"; cat i.out; grep -o "nvrt replay.*trial.*" i.err | head -3
echo "--- phase change (exit $rp):"; cat p.out; grep -o "nvrt replay.*trial.*" p.err | head -3
python - "$rc" "$ri" "$rp" <<PY
import json, sys
rc, ri, rp = (int(x) for x in sys.argv[1:4])
ev = [json.loads(l) for l in open("/tmp/out/occupancy.jsonl") if l.strip()]
out = open("r.out").read()
fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    mark = "ok" if cond else "FAIL"
    print(f"[{mark}] {name}" + (f": {detail}" if detail else ""))
chk("exits 0, both trials complete", rc == 0 and "done" in out and "trial 2 step 2" in out)
chk("trial 1 served record A", all(f"trial 1 step {k} loss {v:.4f}" in out for k, v in ((0, 2.0), (1, 1.9), (2, 1.8))), out[:200])
chk("trial 2 served record B from its own step 1 (not the frozen 1.8)", all(f"trial 2 step {k} loss {v:.4f}" in out for k, v in ((0, 3.0), (1, 2.9), (2, 2.8))), out[200:400])
byes_all = [e for e in ev if e["kind"] == "job_bye"]
JA = byes_all[0]["job"] if byes_all else None
look = [e for e in ev if e["kind"] == "lookup" and e["job"] == JA]
chk("two lookups: the second under #t2 with trial 2 and the other record", len(look) == 2 and str(look[1].get("job_key", "")).endswith("#t2") and look[1].get("trial") == 2 and look[1].get("record_id") != look[0].get("record_id"), str([(e.get("job_key"), e.get("trial"), e.get("record_id")) for e in look]))
chk("the job key is stable within the process (the #t2 key is the first key plus the suffix)", len(look) == 2 and str(look[1].get("job_key", "")) == str(look[0].get("job_key", "")) + "#t2", str([e.get("job_key") for e in look]))
enq = [e for e in ev if e["kind"] == "enqueue" and e.get("unit") == "step" and e["job"] == JA]
chk("enqueue steps restart at the boundary and carry the trial", [(e.get("step"), e.get("trial")) for e in enq] == [(1, None), (2, None), (3, None), (1, 2), (2, 2), (3, 2)], str([(e.get("step"), e.get("trial")) for e in enq]))
bye = [e for e in ev if e["kind"] == "job_bye"][0]
chk("job_bye: trials 2, last_step 3", bye.get("trials") == 2 and bye.get("last_step") == 3, str({k: bye.get(k) for k in ("trials", "last_step", "reason")}))
stream = [json.loads(l) for l in open("/tmp/out/lr_stream.jsonl")]
chk("lr_stream.jsonl: one line per trial", [(s["trial"], s["steps"]) for s in stream][:2] == [(1, 3), (2, 3)], str([(s["trial"], s["steps"]) for s in stream]))
# the newton / modded-nanogpt shape: model built once, parameters re-initialised in place per trial
iout = open("i.out").read()
chk("in-place re-init: trial 1 record A", ri == 0 and all(f"trial 1 step {k} loss {v:.4f}" in iout for k, v in ((0, 4.0), (1, 3.9), (2, 3.8))), iout[:200])
chk("in-place re-init: trial 2 record B from its own step 1 (the boundary seen without new parameters)", all(f"trial 2 step {k} loss {v:.4f}" in iout for k, v in ((0, 5.0), (1, 4.9), (2, 4.8))), iout[200:400])
byes = [e for e in ev if e["kind"] == "job_bye"]
jobs = [e["job"] for e in byes]
J2 = jobs[1] if len(jobs) > 1 else None
l2 = [e for e in ev if e["kind"] == "lookup" and e["job"] == J2]
chk("in-place: the trial-2 lookup says rebuilt false", len(l2) == 2 and l2[1].get("trial") == 2 and l2[1].get("rebuilt") is False, str([(e.get("trial"), e.get("rebuilt")) for e in l2]))
e2 = [e for e in ev if e["kind"] == "enqueue" and e["job"] == J2 and e.get("unit") == "step"]
first_steps = [(e.get("trial"), round(float(e.get("modelled_s") or 0), 3)) for e in e2 if e.get("step") == 1]
chk("in-place: trial 1 step 1 pays the compile-laden first step (0.5 s), trial 2 step 1 a steady step (0.05 s)", first_steps == [(None, 0.5), (2, 0.05)], str(first_steps))
e1 = [e for e in ev if e["kind"] == "enqueue" and e["job"] == jobs[0] and e.get("unit") == "step" and e.get("step") == 1]
chk("rebuilt model: the first step of each trial pays the compile (0.5 s)", [round(float(e.get("modelled_s") or 0), 3) for e in e1] == [0.5, 0.5], str([e.get("modelled_s") for e in e1]))
# a phase change: a second optimizer over the same, unchanged parameters is no boundary
pout = open("p.out").read()
J3 = jobs[2] if len(jobs) > 2 else None
l3 = [e for e in ev if e["kind"] == "lookup" and e["job"] == J3]
e3 = [e.get("step") for e in ev if e["kind"] == "enqueue" and e["job"] == J3 and e.get("unit") == "step"]
tr3 = byes[2].get("trials") if len(byes) > 2 else "no job"
chk("phase change: one lookup, steps 1..6 continuous, no trial", rp == 0 and len(l3) == 1 and e3 == [1, 2, 3, 4, 5, 6] and tr3 is None, f"lookups {len(l3)} steps {e3} trials {tr3}")
chk("phase change: served one record throughout (6.0 … 5.5)", all(f"loss {v:.4f}" in pout for v in (6.0, 5.9, 5.8, 5.7, 5.6, 5.5)), pout[:300])
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
PY'
