#!/usr/bin/env python3
"""Option (b), the study 2026-10-21: a rerun of a job is served a sibling run.

Registry-level checks against a synthetic registry and task file:
  - the first launch of a seeded job gets its own record (exact_seed);
  - every rank of that job (same job key) gets the same record, not a sibling;
  - a second launch of the same seed (another job key) gets the first sibling
    from the task's rerun_siblings, draw=sibling, sibling_of the primary, rank 0;
    a third the second sibling; a fourth the primary again, draw=reused;
  - a path-bound job behaves the same, and its sibling is served by id even
    though the sibling is bound to another path;
  - a sibling that is not in the registry, or not replay-ready, is skipped;
  - with no task file the old behaviour holds (exact_seed_reused);
  - in a task run a repeated launch of a seed that has a record gets that
    record again (exact_seed_reused); substitution is only for seeds with no
    record (2026-10-24, G-g4na-sonnet-L0);
  - an unseeded pool keeps drawing without replacement (siblings not needed).
    python3 -I tests/test_rerun_siblings.py
"""
import json, os, sys, tempfile
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "daemon"))
import gpu_daemon as gd  # noqa: E402

H = "a" * 64
def rec(rid, seed=None, group="grp", bound=None, ready=True, argv=None):
    ident = {"record_id": rid, "entry_hash": H, "argv": argv if argv is not None else (["train.py", "--seed", str(seed)] if seed is not None else ["train.py"]),
             "replicate_group": group, "match_policy": "path_bound" if bound else "exact_only"}
    if bound:
        ident["bound_path"] = bound
    if seed is not None:
        ident["seed"] = {"value": seed}
    return {"identity": ident, "mode": "replay", "steps": 3, "per_step_gpu_s": 0.1,
            "timing": {"steady_step_s": 0.1}, "series": {"train_loss": {"steps": [1, 2, 3], "values": [1.0, 0.9, 0.8]}},
            "readiness": {"replay": ready}, "failures": []}

fails = 0; n = 0
def chk(name, got, want):
    global fails, n
    n += 1
    ok = got == want
    fails += 0 if ok else 1
    print(f"[{'ok' if ok else 'FAIL'}] {name}: {got!r}" + ("" if ok else f" (want {want!r})"))

with tempfile.TemporaryDirectory() as d:
    R = Path(d) / "records"; R.mkdir()
    # seeded exact records of one configuration: seeds 0..3; seed 3 not replay-ready
    for s_ in range(4):
        json.dump(rec(f"cfg-seed{s_}", seed=s_, ready=(s_ != 3)), open(R / f"cfg-seed{s_}.json", "w"))
    # path-bound prepared records, one per seed path
    for s_ in range(3):
        json.dump(rec(f"prep-seed{s_}", bound=f"runs/a/seed{s_}/train_gpt.py", group="prep", argv=["train_gpt.py"]),
                  open(R / f"prep-seed{s_}.json", "w"))
    # an unseeded pool of two
    for i in range(2):
        json.dump(rec(f"pool-{i}", group="pool", argv=["pool.py"]), open(R / f"pool-{i}.json", "w"))
    task = {"task_id": "t", "jobs": [
        {"job_id": "seed0", "records": ["cfg-seed0"], "rerun_siblings": ["cfg-seed1", "missing-id", "cfg-seed3", "cfg-seed2"]},
        {"job_id": "prep0", "records": ["prep-seed0"], "rerun_siblings": ["prep-seed1"]},
    ]}
    T = Path(d) / "task.json"; json.dump(task, open(T, "w"))

    reg = gd.Registry(R)
    chk("task loads two sibling lists", reg.load_task(T), 2)
    L = lambda jk, argv=None, sp=None: reg.lookup(None, H, argv if argv is not None else ["train.py", "--seed", "0"], job_key=jk, script_path=sp)
    r, m, i = L("J1");            chk("first launch of seed 0: its record", (i["match"], i["draw"], reg._rid(r)), ("exact", "exact_seed", "cfg-seed0"))
    r, m, i = L("J1");            chk("same job key again (another rank): same record, cached", (i["draw"], reg._rid(r)), ("exact_seed", "cfg-seed0"))
    r, m, i = L("J2");            chk("rerun: first sibling", (i["draw"], reg._rid(r), i.get("sibling_of"), i.get("sibling_rank")), ("sibling", "cfg-seed1", "cfg-seed0", 0))
    r, m, i = L("J3");            chk("rerun: skips a missing id and a not-ready record, takes the next", (i["draw"], reg._rid(r), i.get("sibling_rank")), ("sibling", "cfg-seed2", 3))
    r, m, i = L("J4");            chk("rerun with the list exhausted: the primary, reused", (i["draw"], reg._rid(r), i.get("siblings_exhausted")), ("reused", "cfg-seed0", 4))
    r, m, i = L("J2");            chk("the rerun's own second rank: its sibling, cached", (i["draw"], reg._rid(r)), ("sibling", "cfg-seed1"))
    # path-bound
    P = lambda jk: reg.lookup(None, "b" * 64, ["train_gpt.py"], job_key=jk, script_path="/home/researcher/project/runs/a/seed0/train_gpt.py")
    r, m, i = P("P1");            chk("path job, first launch: its own record", (i["match"], reg._rid(r)), ("path", "prep-seed0"))
    r, m, i = P("P2");            chk("path job rerun: the sibling bound to another path, by id", (i["match"], i["draw"], reg._rid(r), i.get("sibling_of")), ("path", "sibling", "prep-seed1", "prep-seed0"))
    r, m, i = P("P3");            chk("path job, third launch: primary reused", (i["draw"], reg._rid(r)), ("reused", "prep-seed0"))
    # a sibling already taken as someone's primary is skipped: launch seed1's own path first, then rerun seed0's path
    reg2 = gd.Registry(R); reg2.load_task(T)
    r, m, i = reg2.lookup(None, "b" * 64, ["train_gpt.py"], job_key="Q0", script_path="/x/runs/a/seed1/train_gpt.py")
    chk("seed1's own path takes prep-seed1 first", reg2._rid(r), "prep-seed1")
    r, m, i = reg2.lookup(None, "b" * 64, ["train_gpt.py"], job_key="Q1", script_path="/x/runs/a/seed0/train_gpt.py")
    r, m, i = reg2.lookup(None, "b" * 64, ["train_gpt.py"], job_key="Q2", script_path="/x/runs/a/seed0/train_gpt.py")
    chk("seed0's rerun does not take a sibling already served as another job's record", (i["draw"], reg2._rid(r)), ("reused", "prep-seed0"))
    # many registry records bound to ONE path (the Q prep packages): with the
    # task loaded, a first launch gets the job's listed record, not the pool's
    # rotation, and the rerun gets the listed sibling
    for s_ in range(3):
        json.dump(rec(f"q-run{s_}", bound="quick/a/train_gpt.py", group="q", argv=["train_gpt.py"]), open(R / f"q-run{s_}.json", "w"))
    task_q = {"task_id": "q", "jobs": [{"job_id": "a", "records": ["q-run1"], "rerun_siblings": ["q-run2"]}]}
    TQ = Path(d) / "task_q.json"; json.dump(task_q, open(TQ, "w"))
    regq = gd.Registry(R); regq.load_task(TQ)
    Q = lambda jk: regq.lookup(None, "c" * 64, ["train_gpt.py"], job_key=jk, script_path="/p/quick/a/train_gpt.py")
    r, m, i = Q("A1"); chk("one path, three records, task lists q-run1: first launch gets q-run1", (i["match"], regq._rid(r)), ("path", "q-run1"))
    r, m, i = Q("A2"); chk("its rerun gets the listed sibling q-run2, not the pool's q-run0", (i["draw"], regq._rid(r)), ("sibling", "q-run2"))
    r, m, i = Q("A3"); chk("third launch: primary reused (q-run0 is never served)", (i["draw"], regq._rid(r)), ("reused", "q-run1"))
    # a path the task does not list keeps the registry pool
    r, m, i = regq.lookup(None, "b" * 64, ["train_gpt.py"], job_key="A4", script_path="/p/runs/a/seed2/train_gpt.py")
    chk("an unlisted path still draws from the registry pool", (i["match"], regq._rid(r)), ("path", "prep-seed2"))
    # a decision slot: records null, candidate_pool lists the admissible records;
    # a launch whose exact key is in the pool is served from it, within the group
    task_d = {"task_id": "d", "jobs": [{"job_id": "confirm-1", "records": None, "bind": {"decision": "pick", "slot": 1},
                                         "candidate_pool": ["pool-0", "pool-1"]}],
              "decisions": [{"candidates": [{"confirm_records": ["pool-1"]}]}]}
    TD = Path(d) / "task_d.json"; json.dump(task_d, open(TD, "w"))
    regd = gd.Registry(R); regd.load_task(TD)
    r, m, i = regd.lookup(None, H, ["pool.py"], job_key="D1"); chk("decision slot: launch served from the candidate pool", (i["match"], regd._rid(r) in {"pool-0", "pool-1"}), ("exact", True))
    r2, m, i = regd.lookup(None, H, ["pool.py"], job_key="D2"); chk("its rerun: the other pool record, without replacement", (i["draw"], regd._rid(r2) != regd._rid(r)), ("without_replacement", True))
    chk("records listed via candidate_pool and confirm_records", sorted(regd._task_primaries), ["pool-0", "pool-1"])
    # slot binding: two slots share one pool of four; launches bind in order, every
    # rank of a launch the same way, a third launch rebinds to slot 1
    for i in range(4):
        json.dump(rec(f"cf-{i}", group=f"cfg{i}", argv=["cf.py", "--v", str(i)]), open(R / f"cf-{i}.json", "w"))
    task_s = {"task_id": "s", "jobs": [
        {"job_id": "screen-a", "records": ["cfg-seed0"], "rerun_siblings": ["cfg-seed1"]},
        {"job_id": "confirm-pick1", "records": None, "bind": {"decision": "pick", "slot": 1}, "candidate_pool": [f"cf-{i}" for i in range(4)]},
        {"job_id": "confirm-pick2", "records": None, "bind": {"decision": "pick", "slot": 2}, "candidate_pool": [f"cf-{i}" for i in range(4)]}],
        "decisions": [{"decision_id": "pick", "slots": 2, "candidates": [{"variant": f"v{i}", "confirm_records": [f"cf-{i}"]} for i in range(4)]}]}
    TS = Path(d) / "task_s.json"; json.dump(task_s, open(TS, "w"))
    regs = gd.Registry(R); regs.load_task(TS)
    r, m, i = regs.lookup(None, H, ["cf.py", "--v", "2"], job_key="C1")
    chk("first confirm launch binds to slot 1 with its candidate", (i.get("task_job"), i.get("bind"), i.get("candidate")), ("confirm-pick1", {"decision": "pick", "slot": 1}, "v2"))
    r, m, i = regs.lookup(None, H, ["cf.py", "--v", "0"], job_key="C2")
    chk("second confirm launch binds to slot 2", (i.get("task_job"), i.get("bind")["slot"], i.get("candidate")), ("confirm-pick2", 2, "v0"))
    r, m, i = regs.lookup(None, H, ["cf.py", "--v", "0"], job_key="C2")
    chk("the second launch's other rank: same binding", (i.get("task_job"), i.get("bind")["slot"]), ("confirm-pick2", 2))
    r, m, i = regs.lookup(None, H, ["cf.py", "--v", "3"], job_key="C3")
    chk("a third confirm launch rebinds to slot 1, counted", (i.get("bind")["slot"], i.get("slot_launches")), (1, 2))
    r, m, i = regs.lookup(None, H, ["train.py", "--seed", "0"], job_key="S1")
    chk("a screen job's record ties to its job, no bind", (i.get("task_job"), i.get("bind")), ("screen-a", None))
    r, m, i = regs.lookup(None, H, ["train.py", "--seed", "0"], job_key="S2")
    chk("its rerun's sibling ties to the same job", (i.get("draw"), i.get("task_job")), ("sibling", "screen-a"))
    # unseeded pool: without replacement, no siblings involved
    r1 = reg.lookup(None, H, ["pool.py"], job_key="U1")[0]; r2 = reg.lookup(None, H, ["pool.py"], job_key="U2")[0]
    chk("unseeded pool: two launches, two records", sorted([reg._rid(r1), reg._rid(r2)]), ["pool-0", "pool-1"])
    # no task file: old behaviour
    reg3 = gd.Registry(R)
    reg3.lookup(None, H, ["train.py", "--seed", "0"], job_key="N1")
    r, m, i = reg3.lookup(None, H, ["train.py", "--seed", "0"], job_key="N2")
    chk("without a task file a rerun is the same record, exact_seed_reused", (i["draw"], reg3._rid(r)), ("exact_seed_reused", "cfg-seed0"))


# ---------------------------------------------------------------------------
# The task-run seed rule (gpu-tasks, 15:07Z 2026-10-21, goal-level package G1
# where the agent picks its own seeds): no record served twice while an
# unused one remains, substitutes for a seed WITHOUT a record in descending
# seed order, recycled round-robin when the pool is spent. Amended 2026-10-24
# (G-g4na-sonnet-L0 ranks 19-20): a repeated launch of a seed that HAS a record
# gets that record again, `exact_seed_reused`, as hardware repeats a seed's
# curve; until then it fell to the substitution, a duplicate `--seed 1` took
# seed 0's record and the genuine `--seed 0` then recycled it, so two
# different seeds showed one curve.
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as d:
    R = Path(d) / "records"; R.mkdir()
    for s_ in range(3):
        json.dump(rec(f"g1-seed{s_}", seed=s_, group="g1"), open(R / f"g1-seed{s_}.json", "w"))
    T = Path(d) / "task.json"
    json.dump({"task_id": "g1", "jobs": [{"job_id": "adamw", "records": ["g1-seed0", "g1-seed1", "g1-seed2"]}]}, open(T, "w"))
    reg = gd.Registry(R); reg.load_task(T)
    G = lambda jk, seed: reg.lookup(None, H, ["train.py", "--seed", str(seed)], job_key=jk)
    r, m, i = G("S1", 0); chk("task run: seed 0 first launch, its record", (i["draw"], reg._rid(r)), ("exact_seed", "g1-seed0"))
    r, m, i = G("S2", 0); chk("seed 0 again (another job key): its OWN record again, exact_seed_reused, never another seed's", (i["draw"], reg._rid(r)), ("exact_seed_reused", "g1-seed0"))
    r, m, i = G("S3", 42); chk("unrecorded seed 42: substituted from the unused records, descending (seed 2 is still unused)", (i["draw"], reg._rid(r)), ("seed_substituted", "g1-seed2"))
    r, m, i = G("S4", 1); chk("seed 1, first launch: its record, unused until now", (i["draw"], reg._rid(r)), ("exact_seed", "g1-seed1"))
    r, m, i = G("S4b", 1); chk("the same seed again: the same record again, labelled (its siblings, if any, come from the rerun rule)", (i["draw"], reg._rid(r)), ("exact_seed_reused", "g1-seed1"))
    # pool spent: a seedless or unrecorded-seed launch recycles round-robin from the top, without forgetting what
    # was served (Q-p20tail-sonnet-L0-bashdefault-r2b, 2026-10-23, for the recorded-seed half, now exact_seed_reused)
    r, m, i = G("S5", 42); chk("pool spent, unrecorded seed: recycled round-robin from the top", (i["draw"], reg._rid(r)), ("recycled", "g1-seed2"))
    r, m, i = G("S5b", 42); chk("next recycled draw moves on (what was served is not forgotten)", (i["draw"], reg._rid(r)), ("recycled", "g1-seed1"))
    served = [reg._rid(reg.lookup(None, H, ["train.py", "--seed", "0"], job_key=k)[0]) for k in ("S1", "S2", "S3")]
    chk("the first three launches got two distinct records: seed 0 twice (a repeated seed repeats), seed 42's substitute once", (len(set(served)), served[0] == served[1]), (2, True))
    r, m, i = G("S6", 2); chk("seed 2, whose record went to seed 42's launch: its own record again, reused, not another seed's", (i["draw"], reg._rid(r)), ("exact_seed_reused", "g1-seed2"))

# a prefix record and the path-bound full record of ONE underlying run share
# `common.served_unit`: a task run never serves that run twice
with tempfile.TemporaryDirectory() as d:
    R = Path(d) / "records"; R.mkdir()
    for s_ in range(2):
        pre = rec(f"run{s_}-stop1250", seed=s_, group="opt", argv=["cand.py", "--seed", str(s_)]); pre["common"] = {"served_unit": f"run{s_}"}
        full = rec(f"run{s_}-full", bound="candidates/opt/cand.py", group="opt", argv=["cand.py"]); full["common"] = {"served_unit": f"run{s_}"}; full["identity"]["entry_hash"] = "f" * 64
        json.dump(pre, open(R / f"run{s_}-stop1250.json", "w")); json.dump(full, open(R / f"run{s_}-full.json", "w"))
    T = Path(d) / "task.json"
    json.dump({"task_id": "g1", "jobs": [{"job_id": "opt", "records": ["run0-stop1250", "run1-stop1250", "run0-full", "run1-full"]}]}, open(T, "w"))
    reg = gd.Registry(R); reg.load_task(T)
    r, m, i = reg.lookup(None, H, ["cand.py", "--seed", "1"], job_key="U1", script_path="/w/candidates/opt/cand.py")
    chk("unedited launch of seed 1: its prefix record", (i["match"], i["draw"], reg._rid(r)), ("exact", "exact_seed", "run1-stop1250"))
    r, m, i = reg.lookup(None, "e" * 64, ["cand.py", "--seed", "1"], job_key="U2", script_path="/w/candidates/opt/cand.py")
    chk("edited script (other entry hash), seed 1 again: path-bound full record of the OTHER run, the unit of run1 is used", (i["match"], i["draw"], reg._rid(r)), ("path", "seed_substituted", "run0-full"))
    r, m, i = reg.lookup(None, "e" * 64, ["cand.py", "--seed", "0"], job_key="U3", script_path="/w/candidates/opt/cand.py")
    chk("both units used: recycled, labelled", i["draw"], "recycled")
    # the same through lineage alone (gpu-tasks' goalpath records carry derived_from, no common block)
    for s_ in range(2):
        for name in (f"run{s_}-stop1250", f"run{s_}-full"):
            rr = json.load(open(R / f"{name}.json")); rr.pop("common", None)
            rr["identity"]["derived_from"] = {"kind": "prefix" if "stop" in name else "prepared", "record": f"run{s_}", "steps": [1, 3]}
            json.dump(rr, open(R / f"{name}.json", "w"))
    reg = gd.Registry(R); reg.load_task(T)
    chk("unit from derived_from.record when common.served_unit is absent", (reg._unit(reg._by_id["run1-stop1250"]), reg._unit(reg._by_id["run1-full"])), ("run1", "run1"))
    reg.lookup(None, H, ["cand.py", "--seed", "1"], job_key="V1", script_path="/w/candidates/opt/cand.py")
    r, m, i = reg.lookup(None, "e" * 64, ["cand.py", "--seed", "1"], job_key="V2", script_path="/w/candidates/opt/cand.py")
    chk("lineage alone keeps one run from being served twice", (i["draw"], reg._rid(r)), ("seed_substituted", "run0-full"))

print(f"\n{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
