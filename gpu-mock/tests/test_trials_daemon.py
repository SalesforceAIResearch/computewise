#!/usr/bin/env python3
"""In-process trials at the daemon (2026-10-22): a lookup under `job_key#t2`
with `trial: 2` begins a new trial on the same connection: a fresh record
under the rerun rule, the job's step restarted, the finished trial's LR
stream written, events tagged `trial`, job_bye counting the trials.
    python3 -I tests/test_trials_daemon.py
"""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src" / "daemon")); sys.path.insert(0, str(HERE.parent / "src" / "torch_patch"))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

H = "b" * 64
def rec(rid, vals):
    return {"identity": {"record_id": rid, "entry_hash": H, "argv": ["train.py", "3"], "replicate_group": "tri", "match_policy": "exact_only"},
            "mode": "replay", "steps": 3, "per_step_gpu_s": 0.1, "timing": {"steady_step_s": 0.1},
            "series": {"train_loss": {"steps": [1, 2, 3], "values": vals}}, "readiness": {"replay": True}, "failures": []}

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td); R = tmp / "records"; R.mkdir()
    for i, vals in enumerate(([2.0, 1.9, 1.8], [3.0, 2.9, 2.8], [4.0, 3.9, 3.8])):
        (R / f"run{i}.json").write_text(json.dumps(rec(f"run{i}", vals)))
    d, sock, occ = start_daemon(tmp / "d", speed=1000.0, registry=R)
    c = DaemonClient(str(sock)); c.hello(argv=["train.py", "3"])
    r1 = c.lookup_v2(job_key="J", rank=0, world_size=1, entry_hash=H, argv=["train.py", "3"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("trial 1 served a record", r1.get("mode") == "replay" and r1.get("record_id") == "run0", str(r1.get("record_id")))
    for k in (1, 2, 3):
        c.enqueue(gpu=0, op="replay", modelled_s=0.001, step=k, unit="step", lr=[0.01])
    v3 = c.materialise("train_loss", step=3)
    v4 = c.materialise("train_loss", step=4)
    chk("past the record's end the value is a labelled continuation, not the last value repeated", v3 == 1.8 and v4 != 1.8 and abs(v4 - 1.8) < 0.1, f"{v3} {v4}")
    # the trial boundary: a new lookup under #t2 on the same connection
    r2 = c.lookup_v2(job_key="J#t2", rank=0, world_size=1, entry_hash=H, argv=["train.py", "3"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact", trial=2)
    chk("trial 2: a fresh record under the rerun rule", r2.get("record_id") == "run1" and r2.get("draw") == "without_replacement", str({k: r2.get(k) for k in ("record_id", "draw")}))
    stream = [json.loads(l) for l in (occ.parent / "lr_stream.jsonl").read_text().splitlines()]
    chk("trial 1's LR stream written at the boundary", len(stream) == 1 and stream[0]["trial"] == 1 and stream[0]["steps"] == 3, str(stream)[:120])
    for k in (1, 2):
        c.enqueue(gpu=0, op="replay", modelled_s=0.001, step=k, unit="step", lr=[0.02])
    chk("trial 2's reads come from the new record from its own step 1", c.materialise("train_loss", step=1) == 3.0 and c.materialise("train_loss", step=2) == 2.9)
    r2b = c.lookup_v2(job_key="J#t2", rank=1, world_size=1, entry_hash=H, argv=["train.py", "3"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact", trial=2)
    chk("another rank's trial-2 lookup is the cached record", r2b.get("record_id") == "run1")
    r3 = c.lookup_v2(job_key="J#t3", rank=0, world_size=1, entry_hash=H, argv=["train.py", "3"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact", trial=3)
    r4 = c.lookup_v2(job_key="J#t4", rank=0, world_size=1, entry_hash=H, argv=["train.py", "3"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact", trial=4)
    chk("trial 3 the last unused record, trial 4 recycled and labelled", r3.get("record_id") == "run2" and r4.get("draw") == "recycled", f"{r3.get('record_id')} {r4.get('draw')}")
    c.bye(); c.close()
    ev = read_occ(occ)
    look = [e for e in ev if e["kind"] == "lookup"]
    chk("lookup events carry the trial from 2 on", [e.get("trial") for e in look] == [None, 2, 2, 3, 4], str([e.get("trial") for e in look]))
    enq = [e for e in ev if e["kind"] == "enqueue"]
    chk("enqueue events carry the trial from 2 on, steps restart", [(e.get("step"), e.get("trial")) for e in enq] == [(1, None), (2, None), (3, None), (1, 2), (2, 2)], str([(e.get("step"), e.get("trial")) for e in enq]))
    mats = [e for e in ev if e["kind"] == "materialise"]
    chk("materialise events carry the trial", [e.get("trial") for e in mats][-2:] == [2, 2], str([e.get("trial") for e in mats]))
    bye = [e for e in ev if e["kind"] == "job_bye"][0]
    chk("job_bye: trials 4, last_step None (the last trial never stepped), reason bye", bye.get("trials") == 4 and bye.get("last_step") is None and bye.get("reason") == "bye", str({k: bye.get(k) for k in ("trials", "last_step", "reason")}))
    stream = [json.loads(l) for l in (occ.parent / "lr_stream.jsonl").read_text().splitlines()]
    chk("lr_stream.jsonl: one line per trial that stepped (trials 1 and 2)", [(s_["trial"], s_["steps"]) for s_ in stream] == [(1, 3), (2, 2)], str([(s_["trial"], s_["steps"]) for s_ in stream]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
