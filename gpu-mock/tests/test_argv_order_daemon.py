#!/usr/bin/env python3
"""Through the daemon: a launch whose flags come in another order than the
record's is an exact match; a launch of a known script with a different value
is `match: none` and logs a `near_miss` event naming the differing tokens."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(__file__))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0; n = 0


def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


H = "e" * 64
REC_ARGV = ["train_gpt.py", "--seed", "7", "--resume", "checkpoints/seed7-step250.pt", "--stop-at", "500"]
rec = {"identity": {"record_id": "scq-seed7-cont", "entry_hash": H, "argv": REC_ARGV, "replicate_group": "scq-cont", "match_policy": "exact_only"},
       "mode": "replay", "steps": 500, "timing": {"pacing": "per_step", "startup_in_train_s": 0.0, "per_step_gpu_s": {"value": 0.1, "provenance": "recorded"}},
       "memory": {"peak_allocated_bytes": 8 * 1024 ** 3, "peak": 8 * 1024 ** 3},
       "series": {"train_loss": {"steps": [251, 500], "values": [3.3, 3.2]}}, "readiness": {"replay": True}, "failures": []}
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir(); (R / "r.json").write_text(json.dumps(rec))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    if True:
        c = DaemonClient(str(sock)); c.hello(argv=["train_gpt.py"])
        r1 = c.lookup_v2(job_key="A", rank=0, world_size=1, entry_hash=H, argv=["train_gpt.py", "--seed", "7", "--stop-at", "500", "--resume", "checkpoints/seed7-step250.pt"], env={}, subset_env={}, script_path="/w/train_gpt.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        chk("the brief's order (--stop-at before --resume) is an exact match of the record", r1.get("mode") == "replay" and r1.get("match") == "exact" and r1.get("record_id") == "scq-seed7-cont", str({k: r1.get(k) for k in ("mode", "match", "record_id")}))
        c.bye(); c.close()
        c2 = DaemonClient(str(sock)); c2.hello(argv=["train_gpt.py"])
        r2 = c2.lookup_v2(job_key="B", rank=0, world_size=1, entry_hash=H, argv=["train_gpt.py", "--seed", "7", "--stop-at=500", "--resume", "checkpoints/seed7-step250.pt"], env={}, subset_env={}, script_path="/w/train_gpt.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        chk("--stop-at=500 keys like --stop-at 500", r2.get("match") == "exact", str(r2.get("match")))
        c2.bye(); c2.close()
        c3 = DaemonClient(str(sock)); c3.hello(argv=["train_gpt.py"])
        r3 = c3.lookup_v2(job_key="C", rank=0, world_size=1, entry_hash=H, argv=["train_gpt.py", "--seed", "7", "--stop-at", "600", "--resume", "checkpoints/seed7-step250.pt"], env={}, subset_env={}, script_path="/w/train_gpt.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        chk("a different --stop-at value misses (mode model, match none)", r3.get("mode") != "replay" and r3.get("match") == "none", str({k: r3.get(k) for k in ("mode", "match")}))
        c3.bye(); c3.close()
    ev = read_occ(occ)
    nm = [e for e in ev if e["kind"] == "near_miss"]
    chk("one near_miss event, for the missing launch only, naming the closest record and the differing tokens", len(nm) == 1 and nm[0]["job"] == 3 and nm[0].get("record_id") == "scq-seed7-cont" and nm[0].get("only_launch") == ["600"] and nm[0].get("only_record") == ["500"] and nm[0].get("records_of_script") == 1, str([(e.get("job"), e.get("record_id"), e.get("only_launch"), e.get("only_record")) for e in nm]))
    looks = [e for e in ev if e["kind"] == "lookup"]
    chk("lookups: exact, exact, none", [e.get("match") for e in looks] == ["exact", "exact", "none"], str([e.get("match") for e in looks]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
