#!/usr/bin/env python3
"""Known answer for the study's case of 2026-10-23 00:15Z (run
Q-p20tail-sonnet-L0-bashdefault-r2b on the 6ef6 pair): the run's 22 launches
of pkg-p20-tail-q replayed through the daemon's registry with the task loaded.
Every served record must be listed for the task job the daemon attributes, and
a seeded rerun after a job's pool is spent must be served ITS job's sibling:
launch 21 (`--seed 3`, a run4 rerun) → run4's first sibling under task_job
run4, launch 23 (soda-fade `--seed 2`, a run3 rerun) → run3's first sibling
under run3. On 6ef6 they got run5's and run4's siblings (the recycle ignored
the seed).
    python3 -I tests/known_answer_p20tail.py <registry dir> <pkg-p20-tail-q.json>
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src" / "daemon"))
import gpu_daemon as gd  # noqa: E402

reg_dir, task_path = Path(sys.argv[1]), Path(sys.argv[2])
task = json.loads(task_path.read_text())
reg = gd.Registry(reg_dir / "records" if (reg_dir / "records").is_dir() else reg_dir)
reg.load_task(task_path)
jobs = {j["job_id"]: j for j in task["jobs"]}
# the run's launches in order: (box job id, argv)
LAUNCHES = [
    (2, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "1"]), (3, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "0"]),
    (4, ["contra-soft-muon-long/contra-muon-to-soft-muon.py", "--seed", "5"]), (5, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "2"]),
    (6, ["soda-fade/contra-muon-to-soft-muon.py", "--seed", "0"]), (7, ["soda-fade/contra-muon-to-soft-muon.py", "--seed", "1"]),
    (8, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "4"]), (9, ["radial-damp/radial_scale_then_radius_correct.py", "--seed", "2"]),
    (10, ["radial-damp/radial_scale_then_radius_correct.py", "--seed", "0"]), (11, ["soda-fade/contra-muon-to-soft-muon.py", "--seed", "3"]),
    (12, ["radial-damp/radial_scale_then_radius_correct.py", "--seed", "4"]), (13, ["ema-nesterov-aurora/radial_scale_then_radius_correct.py", "--seed", "9"]),
    (14, ["ema-nesterov-aurora/radial_scale_then_radius_correct.py", "--seed", "8"]), (15, ["muonh-tuned/train_gpt_simple_muonh.py"]),
    (16, ["ortho/train_gpt_simple.py"]), (17, ["newton-muon/train_gpt_simple_newton_muon.py"]),
    (18, ["soda-fade/contra-muon-to-soft-muon.py", "--seed", "2"]), (19, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "3"]),
    (20, ["radial-damp/radial_scale_then_radius_correct.py", "--seed", "3"]), (21, ["contra-soft-muon/contra-muon-to-soft-muon.py", "--seed", "3"]),
    (22, ["circuit-muon/train_gpt_circuit_muon.py", "--seed", "6"]), (23, ["soda-fade/contra-muon-to-soft-muon.py", "--seed", "2"]),
]
EXPECT = {21: ("contra-soft-muon-run4", "mng3-20260509-contra-soft-muon-h100-sxmx8-1b5bba11-405-stop250"),
          23: ("soda-fade-run3", "mng3-20260518-soda-fade-3000-n9-h200x4-ae3f35c8-b83-stop250")}


def entry_for(script):
    """The entry hash of the script's listed records (the unedited launch)."""
    for j in task["jobs"]:
        if script in str(j.get("launch") or "") or (j.get("script_path") or "") == script:
            for rid in j.get("records") or []:
                r = reg._by_id.get(rid)
                if r:
                    return r["identity"].get("entry_hash")
    return None


fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


served = {}
for k, (bid, argv) in enumerate(LAUNCHES):
    entry = entry_for(argv[0])
    rec, mode, info = reg.lookup(None, entry, argv, job_key=f"L{bid}", stage="exact", script_path="/w/" + argv[0])
    rid = reg._rid(rec) if rec else None
    tj = info.get("task_job")
    served[bid] = (rid, tj, info.get("draw"), info.get("sibling_of"))
    listed = set((jobs.get(tj) or {}).get("records") or []) | set((jobs.get(tj) or {}).get("rerun_siblings") or [])
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else None
    job_seed = (jobs.get(tj) or {}).get("seed")
    chk(f"launch {bid} {argv[0].split('/')[0]} seed {seed}: served a record listed for its task_job ({tj}, draw {info.get('draw')})", rec is not None and tj in jobs and rid in listed and (seed is None or job_seed is None or job_seed == seed), f"record {rid} task_job {tj} job seed {job_seed}")
for bid, (tj, rid) in EXPECT.items():
    got = served.get(bid)
    chk(f"launch {bid}: the seeded rerun gets its own job's first sibling ({tj} → …{rid[-22:]})", got and got[0] == rid and got[1] == tj and got[2] == "sibling", str(got))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
