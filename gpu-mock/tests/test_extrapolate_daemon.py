#!/usr/bin/env python3
"""Reads past a record's last step are a labelled continuation, not the last
value repeated (2026-10-22): `tier: extrapolated`, one `record_exhausted`
event per (job, trial), deterministic per step, never rising above the tail.
    python3 -I tests/test_extrapolate_daemon.py
"""
import json, math, sys, tempfile
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

H = "c" * 64
steps = list(range(125, 3001, 125))
vals = [round(2.9 + 11.5 * (s + 40) ** -0.45, 5) for s in steps]
rec = {"identity": {"record_id": "long-run", "entry_hash": H, "argv": ["train.py"], "replicate_group": "lr", "match_policy": "exact_only"},
       "mode": "replay", "steps": 3000, "per_step_gpu_s": 0.1, "timing": {"steady_step_s": 0.1},
       "series": {"val_loss": {"steps": steps, "values": vals}}, "readiness": {"replay": True}, "failures": []}
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td); R = tmp / "records"; R.mkdir(); (R / "r.json").write_text(json.dumps(rec))
    d, sock, occ = start_daemon(tmp / "d", speed=1000.0, registry=R)
    c = DaemonClient(str(sock)); c.hello(argv=["train.py"])
    r = c.lookup_v2(job_key="L", rank=0, world_size=1, entry_hash=H, argv=["train.py"], env={}, script_path="/w/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("record served", r.get("record_id") == "long-run")
    within = [c.materialise("val_loss", step=s) for s in (3000, 2875)]
    chk("reads within the record are the record", within == [vals[-1], vals[-2]], str(within))
    beyond = {s: c.materialise("val_loss", step=s) for s in (3125, 3250, 3500, 4000, 6000)}
    chk("reads past the end are finite", all(isinstance(v, float) and math.isfinite(v) for v in beyond.values()), str(beyond))
    chk("the continuation keeps falling, slowly (no frozen value, no jump): under 0.2 nats by twice the length", beyond[6000] < beyond[3125] and vals[-1] - 0.2 < beyond[6000] < vals[-1] + 0.005 and len(set(beyond.values())) == 5, f"last recorded {vals[-1]} -> {beyond}")
    chk("deterministic per step", c.materialise("val_loss", step=3500) == beyond[3500])
    tr = c.materialise("train_loss", step=3250)
    chk("a derived stand-in past the end is extrapolated too, finite", isinstance(tr, float) and math.isfinite(tr) and tr != vals[-1], str(tr))
    many = c.materialise_many(["val_loss"], step=3375)
    chk("materialise_many past the end: extrapolated", isinstance(many.get("val_loss"), float) and many["val_loss"] != vals[-1], str(many))
    c.bye(); c.close()
    ev = read_occ(occ)
    mats = [e for e in ev if e["kind"] == "materialise"]
    tiers = [e.get("tier") for e in mats]
    chk("tiers: series within, extrapolated beyond", tiers[:2] == ["series", "series"] and set(tiers[2:]) == {"extrapolated"}, str(tiers))
    ex = [e for e in ev if e["kind"] == "record_exhausted"]
    chk("one record_exhausted event, at the first read past the end", len(ex) == 1 and ex[0].get("record_steps") == 3000 and ex[0].get("step") == 3125, str(ex))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
