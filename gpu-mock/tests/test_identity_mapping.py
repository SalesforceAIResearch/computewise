#!/usr/bin/env python3
"""Identity and placement: the exact-key argv strip, and the mapping from a
process's local CUDA device to the daemon's physical GPU. Pure Python, no torch.
Reads gpu-tasks' registry records read-only.

    python3 -I tests/test_identity_mapping.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src" / "daemon"))
sys.path.insert(0, str(ROOT / "src" / "torch_patch"))
sys.path.insert(0, str(HERE))

import gpu_daemon  # noqa: E402
from gpu_daemon import _exact_key, _strip_argv  # noqa: E402
from daemon_client import DaemonClient, physical_gpu_index, visible_physical  # noqa: E402
from test_daemon_e2e import start_daemon  # noqa: E402

REG = ROOT.parent / "gpu-tasks" / "registry" / "records"
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    print(f"[{'ok  ' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    PASS += bool(cond)
    FAIL += (not cond)


def record(rid):
    return json.loads((REG / f"{rid}.json").read_text())


def rid_of(rec):
    return ((rec or {}).get("identity") or {}).get("record_id")


# ---------------------------------------------------------------- argv strip
SFT1 = "trl-sft-h100-sxmx1-0022af66-s1-afosdmfe"
SFT3 = "trl-sft-h100-sxmx1-0022af66-s3-s2gqemn2"
DPO = "trl-dpo-h100-sxmx4-e5295792-s42-ig7dh8vc"
sft = record(SFT1)["identity"]
EX = sft["argv_exempt"]


def launch_from(rec_argv, seed=None):
    """The task's launch line: the record's argv without --push_to_hub (tasks never
    push), a different output dir and no wandb, optionally another seed."""
    out, i = [], 0
    while i < len(rec_argv):
        a = rec_argv[i]
        if a == "--push_to_hub":
            i += 1; continue
        if a.startswith("--output_dir"):
            out.append("--output_dir=/work/out"); i += 1; continue
        if a.startswith("--report_to"):
            out.append("--report_to=none"); i += 1; continue
        if a == "--seed" and seed is not None:
            out += ["--seed", str(seed)]; i += 2; continue
        out.append(a); i += 1
    return out


rec_argv = sft["argv"]
launch = launch_from(rec_argv)
check("SFT: record (--push_to_hub --gradient_checkpointing) and launch (no push) strip equal",
      _strip_argv(rec_argv, EX) == _strip_argv(launch, EX),
      f"{_strip_argv(rec_argv, EX)[-3:]} vs {_strip_argv(launch, EX)[-3:]}")
check("SFT: --gradient_checkpointing survives the strip",
      "--gradient_checkpointing" in _strip_argv(rec_argv, EX))
base = ["dpo.py", "--model_name_or_path", "gpt2", "--logging_first_step"]
check("bare exempt flag no longer collides --bf16 with --fp16",
      _strip_argv(base + ["--bf16"], EX) != _strip_argv(base + ["--fp16"], EX))
check("DPO: --no_remove_unused_columns after --logging_first_step is kept",
      "--no_remove_unused_columns" in _strip_argv(base + ["--no_remove_unused_columns"], EX))
check("exempt flag with a negative value consumes it",
      _strip_argv(["x.py", "--logging_steps", "-1", "--bf16"], EX) == ["x.py", "--bf16"])
check("exempt flag with a word value consumes it",
      _strip_argv(["x.py", "--report_to", "wandb", "--bf16"], EX) == ["x.py", "--bf16"])
check("--k=v exempt is one token",
      _strip_argv(["x.py", "--output_dir=o", "--bf16"], EX) == ["x.py", "--bf16"])
check("trailing bare exempt flag",
      _strip_argv(["x.py", "--bf16", "--push_to_hub"], EX) == ["x.py", "--bf16"])
check("--seed N and --seed=N are stripped",
      _strip_argv(["x.py", "--seed", "3", "--bf16"], None) == ["x.py", "--bf16"]
      and _strip_argv(["x.py", "--seed=3", "--bf16"], None) == ["x.py", "--bf16"])
check("argv[0] reduced to its basename",
      _strip_argv(["/work/sft.py", "--bf16"], None) == ["sft.py", "--bf16"])

# ------------------------------------------------- real registry, exact lookup
for rid_want, seed in ((SFT1, 1), (SFT3, 3)):
    reg = gpu_daemon.Registry(REG)
    rec, mode, info = reg.lookup(None, sft["entry_hash"], launch_from(rec_argv, seed),
                                 job_key=f"t:{seed}")
    got = rid_of(rec)
    check(f"SFT seed {seed}: exact match on the task's launch line",
          info.get("match") == "exact" and got == rid_want,
          f"match={info.get('match')} draw={info.get('draw')} mode={mode} record={got}")

dpo = record(DPO)["identity"]
reg = gpu_daemon.Registry(REG)
rec, mode, info = reg.lookup(None, dpo["entry_hash"], launch_from(dpo["argv"]), job_key="t:dpo")
check("DPO: exact match still holds under the new strip",
      info.get("match") == "exact" and rid_of(rec) == DPO,
      f"match={info.get('match')} draw={info.get('draw')} record={rid_of(rec)}")

# ------------------------------------------ a miss must not pin a job's draw
dpo_rec = record(DPO)
fp = dpo_rec["identity"]["fingerprint"]
fp = fp.get("value") if isinstance(fp, dict) else fp
reg = gpu_daemon.Registry(REG)
rec0, mode0, info0 = reg.lookup(None, "0" * 64, ["dpo.py", "--other"], job_key="t:miss", stage="exact")
check("exact miss returns no record", rec0 is None and mode0 == "model", f"{mode0} {info0}")
rec1, mode1, info1 = reg.lookup(fp, "0" * 64, ["dpo.py", "--other"], job_key="t:miss", stage="structural")
group = {rid_of(r) for r in reg._by_fp.get(fp, [])}   # the DPO replicate group, all seeds
check("the same job's structural lookup is not pinned to the earlier miss",
      rec1 is not None and info1.get("match") == "structural" and rid_of(rec1) in group,
      f"match={info1.get('match')} draw={info1.get('draw')} record={rid_of(rec1)} group={sorted(group)}")
rec2, _, info2 = reg.lookup(fp, "0" * 64, ["dpo.py", "--other"], job_key="t:miss", stage="structural")
check("a hit is cached: a second rank gets the same record", rec2 is rec1 and info2 == info1)

# ------------------------------------------------ derived (prefix) records
PFX = "mng3-20260509-contra-soft-muon-h100-sxmx8-81540d3b-8dc-stop1000"
if (REG / f"{PFX}.json").exists():
    prec = record(PFX); pi = prec["identity"]
    reg = gpu_daemon.Registry(REG)
    rec, mode, info = reg.lookup(None, pi["entry_hash"], pi["argv"], job_key="t:pfx")
    check("prefix record: exact lookup on its own script hash", info.get("match") == "exact" and rid_of(rec) == PFX,
          f"match={info.get('match')} record={rid_of(rec)}")
    sv = gpu_daemon._serving_info(rec)
    check("prefix record: serving steps = k", sv["steps"] == 1000, str(sv["steps"]))
    check("prefix record: eval time scaled to the kept validations",
          sv["n_validations"] == 8 and abs(sv["eval_s_per_validation"] - 9.995 / 8) < 1e-9,
          f"{sv['n_validations']} validations, {sv['eval_s_per_validation']:.4f} s each")
    df = gpu_daemon._derived_from(rec)
    check("lookup event will carry derived_from (kind, parent, steps)",
          df == {"kind": "prefix", "record": "mng3-20260509-contra-soft-muon-h100-sxmx8-81540d3b-8dc",
                 "steps": [1, 1000], "stop_after_step": 1000}, str(df))
    check("a plain record has no derived_from", gpu_daemon._derived_from(record(DPO)) is None)
else:
    print(f"[skip] prefix record {PFX} not in the registry")

# ------------------------------------------ a binary file among the records
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    (d / "good.json").write_text(json.dumps(record(DPO)))
    (d / "._good.json").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X        ")   # an AppleDouble sidecar
    (d / "junk.json").write_bytes(b"\xff\xfe not json")
    try:
        r = gpu_daemon.Registry(d)
        check("the loader survives AppleDouble and binary files and keeps the good record",
              len(r._by_id) == 1 and DPO in r._by_id, str(sorted(r._by_id)))
    except Exception as e:
        check("the loader survives AppleDouble and binary files and keeps the good record", False, repr(e))

# --------------------------- job identity: launcher-less processes are separate jobs
from daemon_client import job_key as jk  # noqa: E402
orphan = {}                                   # no launcher, shell gone: WORLD_SIZE unset, run id unset
k1 = jk(orphan, pid=101, ppid=1, t_start=1000.0)
k2 = jk(orphan, pid=102, ppid=1, t_start=1000.5)
k3 = jk(orphan, pid=103, ppid=1, t_start=1001.0)
check("three orphaned plain-python jobs get three different keys", len({k1, k2, k3}) == 3, f"{k1} {k2} {k3}")
check("a reused pid with a later start time is a new job", jk(orphan, pid=101, ppid=1, t_start=5000.0) != k1)
check("WORLD_SIZE=1 without a launcher is still per process",
      jk({"WORLD_SIZE": "1"}, pid=7, ppid=1, t_start=1.0) != jk({"WORLD_SIZE": "1"}, pid=8, ppid=1, t_start=1.0))
launched = {"WORLD_SIZE": "4", "MASTER_PORT": "29500", "TORCHELASTIC_RUN_ID": "abc"}
check("ranks of one torchrun share one key",
      jk(launched, pid=201, ppid=200, t_start=1.0) == jk(launched, pid=202, ppid=200, t_start=1.1))
check("two torchruns differ by run id",
      jk(launched, pid=201, ppid=200) != jk(dict(launched, TORCHELASTIC_RUN_ID="def"), pid=301, ppid=300))
# the daemon end of the same defect: three jobs with distinct keys draw distinct seeds
reg = gpu_daemon.Registry(REG)
got = {}
for seed, key in ((2, k1), (3, k2), (1, k3)):
    rec, mode, info = reg.lookup(None, sft["entry_hash"], launch_from(rec_argv, seed), job_key=key)
    got[seed] = (rid_of(rec), info.get("draw"))
check("seed 1 draws its own record although a seedless launch took it first",
      got[1][0] == SFT1 and got[1][1] in ("exact_seed", "exact_seed_reused"), str(got[1]))
check("seed 3 draws its own record exactly", got[3] == (SFT3, "exact_seed"), str(got[3]))
check("seed 2 (no record) is substituted, not the same draw as another seed",
      got[2][1] in ("seed_substituted", "recycled") and got[2][0] in (SFT1, SFT3), str(got[2]))

# ----------------------------------- path_bound records (the agent wrote the script)
import hashlib as _hl
with tempfile.TemporaryDirectory() as td:
    d = Path(td)
    ref_hash = _hl.sha256(b"reference prefix script").hexdigest()
    base = {"mode": "replay", "steps": 10, "per_step_gpu_s": 0.05, "timing": {"steady_step_s": 0.05},
            "series": {"val_loss": {"steps": [5, 10], "values": [3.0, 2.5]}}, "readiness": {"replay": True}, "failures": []}
    for seed in (1, 2):
        rec = dict(base, identity={"record_id": f"prep-contra-s{seed}", "match_policy": "path_bound",
                                   "bound_path": "./runs/contra-soft-muon/train_gpt.py", "entry_hash": ref_hash,
                                   "argv": ["train_gpt.py", "--seed", str(seed)], "seed": seed, "replicate_group": "prep-contra",
                                   "fingerprint": "fp256:aaaa", "fingerprint_inputs": {"model": "gpt", "num_layers": 12, "optimizers": ["AdamW", "Muon"], "per_device_batch": 64}})
        (d / f"prep-contra-s{seed}.json").write_text(json.dumps(rec))
    r = gpu_daemon.Registry(d)
    check("bound paths are normalised (./ stripped)", list(r._by_path) == ["runs/contra-soft-muon/train_gpt.py"], str(list(r._by_path)))
    agent_hash = _hl.sha256(b"what the agent wrote").hexdigest()
    rec1, mode, info = r.lookup(None, agent_hash, ["train_gpt.py", "--seed", "2", "--lr", "0.1"], job_key="p1",
                                script_path="/home/researcher/task/runs/contra-soft-muon/train_gpt.py")
    check("an agent-written script at the bound path matches by path, seed from argv",
          info.get("match") == "path" and rid_of(rec1) == "prep-contra-s2" and mode == "replay", f"{info} {rid_of(rec1)}")
    rec2, _, info2 = r.lookup(None, agent_hash, ["train_gpt.py"], job_key="p2", script_path="/w/runs/contra-soft-muon/train_gpt.py")
    check("the same path from another cwd matches too, drawing without replacement", info2.get("match") == "path" and rid_of(rec2) == "prep-contra-s1", str(info2))
    rec3, _, info3 = r.lookup(None, ref_hash, ["train_gpt.py", "--seed", "1"], job_key="p3", script_path="/w/runs/contra-soft-muon/train_gpt.py")
    check("the reference script's bytes still match exactly, before the path", info3.get("match") == "exact", str(info3))
    rec4, mode4, info4 = r.lookup(None, agent_hash, ["train_gpt.py"], job_key="p4", script_path="/w/runs/other/train_gpt.py")
    check("an unbound path matches nothing and goes to the modelled path", rec4 is None and mode4 == "model" and info4.get("match") == "none", str(info4))
    rec5, _, info5 = r.lookup(None, agent_hash, ["train_gpt.py"], job_key="p5", script_path="/w/runs/contra-soft-muon/train_gpt.pyc")
    check("a longer file name is not a suffix match", rec5 is None, str(info5))
    # fingerprint_check: what differs is logged, nothing is withdrawn
    class _Occ:
        def __init__(self): self.events = []
        def event(self, kind, gpu, **f): self.events.append((kind, f))
    dmn = gpu_daemon.GPUDaemon.__new__(gpu_daemon.GPUDaemon); dmn.occ = _Occ()
    job = gpu_daemon.Job(job_id=9, pid=9, argv=[], cwd="", record=rec1, record_id=rid_of(rec1))
    resp = dmn._fingerprint_check(job, {"fingerprint": "fp256:bbbb", "match": "path",
                                        "inputs": {"model": "gpt", "num_layers": 12, "optimizers": ["muon", "adamw"], "micro_batch": 32}})
    check("fingerprint_check reports the differing fields only",
          resp["agree"] is False and set(resp["differs"]) == {"micro_batch"} and resp["differs"]["micro_batch"] == {"record": 64, "live": 32}, str(resp))
    check("fingerprint_check is logged", dmn.occ.events and dmn.occ.events[-1][0] == "fingerprint_check" and dmn.occ.events[-1][1]["agree"] is False)
    check("the job keeps its record after a disagreeing check", job.record is rec1)

# ------------------------------------------- CUDA_VISIBLE_DEVICES -> daemon GPU
cases_count = [(None, 4), ("", 0), ("0,1", 2), ("1,abc,2", 1), ("-1", 0), ("5", 0), ("3", 1)]
for cvd, want in cases_count:
    check(f"visible count CUDA_VISIBLE_DEVICES={cvd!r} -> {want}",
          len(visible_physical(cvd, 4)) == want, str(visible_physical(cvd, 4)))
cases_idx = [  # (local, cvd, want)
    (0, None, 0), (3, None, 3),          # accelerate / torchrun, no CVD
    (0, "2", 2),                           # pinned 1-GPU job
    (0, "2,3", 2), (1, "2,3", 3),          # 2-rank job on GPUs 2,3
    (3, "3", 3),                           # launcher isolating each rank
    (1, "3,1", 1), (0, "3,1", 3),          # order is CVD order, not sorted
]
for local, cvd, want in cases_idx:
    got = physical_gpu_index(local, cvd, 4)
    check(f"local {local} with CUDA_VISIBLE_DEVICES={cvd!r} -> GPU {want}", got == want, str(got))

# ------------------------------- the defect itself, at the daemon: 4 pinned jobs
def run_jobs(sock, gpus, unit=0.1, n=8):
    times = {}

    def job(k, g):
        c = DaemonClient(str(sock))
        c.hello(argv=[f"job{k}.py"])
        t0 = time.monotonic()
        for _ in range(n):
            c.enqueue(gpu=g, op="kernel", modelled_s=unit)
            c.sync(gpu=g)
        times[k] = time.monotonic() - t0
        c.bye(); c.close()

    ts = [threading.Thread(target=job, args=(k, g)) for k, g in enumerate(gpus)]
    for t in ts: t.start()
    for t in ts: t.join()
    return max(times.values())


with tempfile.TemporaryDirectory() as td:
    d, sock, occ = start_daemon(Path(td), 1.0)
    solo = run_jobs(sock, [0])
    pinned = [physical_gpu_index(0, str(k), 4) for k in range(4)]   # CUDA_VISIBLE_DEVICES=k, no LOCAL_RANK
    fixed = run_jobs(sock, pinned)
    old = run_jobs(sock, [0, 0, 0, 0])                                  # the old LOCAL_RANK-only mapping
    check("4 jobs pinned to GPUs 0-3 map to 4 distinct daemon GPUs", sorted(pinned) == [0, 1, 2, 3], str(pinned))
    check("4 pinned jobs run side by side (~1x a solo job)", fixed < 1.4 * solo,
          f"solo={solo:.2f}s four={fixed:.2f}s")
    check("control: all on GPU 0 serialises (~4x)", old > 3.0 * solo, f"solo={solo:.2f}s four-on-0={old:.2f}s")

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
