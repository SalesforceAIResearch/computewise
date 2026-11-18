#!/usr/bin/env python3
"""Path-valued arguments compare as files, not spellings (the study 2026-10-23
21:18Z, G-h3-haiku-L3: `--resume /home/researcher/project/checkpoints/x.pt`
against the record's `checkpoints/x.pt` missed and four jobs ran the modelled
path). An absolute path under the job's cwd, `./x`, `--k=<abs>`, and a
`../../` path from a subdirectory (through the effective stage) all serve the
record; a different file still misses with a near_miss naming the normalised
token."""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "daemon"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "torch_patch"))
sys.path.insert(0, os.path.dirname(__file__))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402
import gpu_daemon as gd  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

H = "c" * 64
ROOT = "/home/researcher/project"
def rec(rid, argv):
    return {"identity": {"record_id": rid, "entry_hash": H, "argv": argv, "match_policy": "exact_only", "replicate_group": rid},
            "mode": "replay", "steps": 1000, "timing": {"pacing": "per_step", "per_step_gpu_s": {"value": 0.01, "provenance": "recorded"}, "steady_step_s": 0.01},
            "launch": {"world_size": 1}, "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
            "series": {"train_loss": {"steps": list(range(1, 1001)), "values": [1.0] * 1000}}, "readiness": {"replay": True}, "failures": []}
chk("unit: abs under cwd -> relative; ./a -> a; a flag untouched; a plain value untouched; a HF id untouched",
    gd._norm_path_token(ROOT + "/checkpoints/a.pt", ROOT) == "checkpoints/a.pt" and gd._norm_path_token("./checkpoints/a.pt", ROOT) == "checkpoints/a.pt"
    and gd._norm_path_token("--resume", ROOT) == "--resume" and gd._norm_path_token("0.8", ROOT) == "0.8" and gd._norm_path_token("Qwen/Qwen2.5-3B-Instruct", ROOT) == "Qwen/Qwen2.5-3B-Instruct"
    and gd._norm_path_token("/elsewhere/a.pt", ROOT) == "/elsewhere/a.pt",
    str([gd._norm_path_token(x, ROOT) for x in (ROOT + "/checkpoints/a.pt", "./checkpoints/a.pt", "--resume", "0.8", "Qwen/Qwen2.5-3B-Instruct", "/elsewhere/a.pt")]))
chk("unit: --k=<abs> value normalised", gd._normalise_argv_paths(["t.py", "--resume=" + ROOT + "/checkpoints/a.pt"], ROOT) == ["t.py", "--resume=checkpoints/a.pt"], str(gd._normalise_argv_paths(["t.py", "--resume=" + ROOT + "/checkpoints/a.pt"], ROOT)))
chk("unit: _same_value treats path values by suffix", gd._same_value(ROOT + "/checkpoints/a.pt", "checkpoints/a.pt") and gd._same_value("../../checkpoints/a.pt", "checkpoints/a.pt") and not gd._same_value("checkpoints/b.pt", "checkpoints/a.pt") and gd._same_value("0.8", "0.8") and not gd._same_value("gsm8k", "longbench"))
with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "r26.json").write_text(json.dumps(rec("seed26-cont", ["train_gpt.py", "--resume", "./checkpoints/seed26-step1000.pt", "--seed", "26"])))
    (R / "r14.json").write_text(json.dumps(rec("seed14-cont", ["train_gpt.py", "--resume", "checkpoints/seed14-step1000.pt", "--seed", "14"])))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    jobs = 0
    def look(argv, cwd, eff=None):
        global jobs; jobs += 1
        c = DaemonClient(str(sock)); c.hello(argv=argv, cwd=cwd)
        r = c.lookup_v2(job_key=f"J{jobs}", rank=0, world_size=1, entry_hash=H, argv=argv, env={}, subset_env={}, argv_effective=eff,
                        script_path=cwd + "/train_gpt.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
        c.bye(); c.close(); return r
    r = look(["train_gpt.py", "--resume", "checkpoints/seed26-step1000.pt", "--seed", "26"], ROOT)
    chk("1: the relative spelling (record stored ./…, normalised at load): exact", r.get("match") == "exact" and r.get("record_id") == "seed26-cont", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(["train_gpt.py", "--resume", ROOT + "/checkpoints/seed26-step1000.pt", "--seed", "26"], ROOT)
    chk("2: the ABSOLUTE path under the cwd (Haiku's launch): exact, same record", r.get("match") == "exact" and r.get("record_id") == "seed26-cont", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(["train_gpt.py", "--resume=" + ROOT + "/checkpoints/seed14-step1000.pt", "--seed", "14"], ROOT)
    chk("3: --resume=<abs>: exact", r.get("match") == "exact" and r.get("record_id") == "seed14-cont", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(["train_gpt.py", "--resume", "./checkpoints/seed14-step1000.pt", "--seed", "14"], ROOT)
    chk("4: ./ spelling: exact", r.get("match") == "exact" and r.get("record_id") == "seed14-cont", str(r.get("match")))
    r = look(["train_gpt.py", "--resume", "../../checkpoints/seed26-step1000.pt", "--seed", "26"], ROOT + "/setups/x",
             eff={"resume": "../../checkpoints/seed26-step1000.pt", "seed": "26"})
    chk("5: from a subdirectory with ../../ (argparse values sent): exact_effective on the record", r.get("match") in ("exact", "exact_effective") and r.get("record_id") == "seed26-cont", str({k: r.get(k) for k in ("match", "record_id")}))
    r = look(["train_gpt.py", "--resume", ROOT + "/checkpoints/seed27-step1000.pt", "--seed", "27"], ROOT)
    chk("6: a different file still misses", r.get("match") == "none", str(r.get("match")))
    ev = read_occ(occ)
    nm = [e for e in ev if e["kind"] == "near_miss"]
    chk("7: the near_miss names the NORMALISED launch token (the exempt --seed stripped) against the record's", nm and nm[-1].get("only_launch") == ["checkpoints/seed27-step1000.pt"], str([(e.get("only_launch"), e.get("only_record")) for e in nm][-1:]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
