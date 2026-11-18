#!/usr/bin/env python3
"""Shard serving at the daemon (next/ r4): a per_example launch that runs a
slice of its workload is served the record's own examples by content.
  1. example-order tables load from records/ beside the records, are never
     read as records, and a bad table (duplicate keys, wrong length, wrong key
     recipe) is refused whole;
  2. the module stage: a launch whose entry script is not recorded, but which
     loaded the recorded evaluator as a module (with its arguments behind an
     argv[0] of the runner's choosing), matches `exact_module`, and only when
     the record has an order table; the argv still has to match;
  3. a keyed shard [a, b) opens positions a..b-1: each score is the record
     vector's entry at that position, the materialise event says so and
     labels it a subset, and job_bye summarises the positions (contiguous,
     not a prefix);
  4. four shards that cover the workload serve the whole vector once, so
     their pooled mean is the recorded final_score;
  5. a key the table does not hold is served at its local index, logged
     `example_order_miss` and counted on job_bye;
  6. a whole launch with no keys (the evaluator run as a script) is unchanged:
     positions 0..n-1, a prefix, provenance `result`;
  7. the gate still holds: a score asked for a position the job did not pace
     is refused, whatever key the request carries; and an out-of-order unit
     opens nothing.
    python3 -I tests/test_shard_serving_daemon.py
"""
import hashlib, json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src" / "daemon")); sys.path.insert(0, str(HERE.parent / "src" / "torch_patch"))
from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402
import gpu_daemon as gd  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

N = 40
WALL = 40.0
HS, HR = "5" * 64, "e" * 64          # the evaluator's hash; the agent's runner's (unrecorded)
EX = [{"example_id": f"gsm8k-{i}", "prompt": f"Problem {i}: what is {i} + {i}?"} for i in range(N)]
KEY = lambda e: hashlib.sha256((e["example_id"] + "\0" + e["prompt"]).encode()).hexdigest()

def rec(rid, h, argv, score, task="llm-kv-selection-budgeting", workload="gsm8k"):
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "replicate_group": rid, "match_policy": "exact_only",
                         "source_key": {"task": task, "label": workload.replace("_", "-")}},
            "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": WALL, "provenance": "recorded"}},
            "memory": {"peak_allocated_bytes": 8 * 1024 ** 3, "peak": 8 * 1024 ** 3}, "series": {},
            "result": {"final_score": {"value": score, "provenance": "recorded"}},
            "materialisation": {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                            "n_examples_full": {"value": N, "provenance": "derived"}, "score_kind": "binary",
                                            "subset_source": {"flag": "--max-examples", "env": "SELECTION_KV_MAX_EXAMPLES"},
                                            "load_s": {"value": None, "provenance": "absent"}, "scale": [0, 100],
                                            "decode": {"workload": workload}}},
            "readiness": {"replay": True}, "failures": []}

def table(fams, workload, keys, ids, **over):
    d = {"schema": gd.EXAMPLE_ORDER_SCHEMA, "families": fams, "workload": workload, "n_examples_full": len(keys),
         "key": "sha256(example_id\\0prompt)", "order": keys, "ids": ids, "how": "test"}
    d.update(over)
    return d

def run_slice(c, idxs, keyed=True, startup=(0.0, 0.0)):
    """Pace and score examples EX[idxs] the way the patch's seams do: unit i
    priced at (wall - start-up) / n_full, carrying the example's key."""
    per = (WALL - startup[0]) / N
    out = []
    for j, i in enumerate(idxs):
        kw = {"startup_s": startup[0], "startup_real_s": startup[1]} if j == 0 else {}
        k = KEY(EX[i]) if keyed else None
        c.enqueue(gpu=0, op="replay", modelled_s=per, step=j + 1, unit="example", key=k, **kw)
        c.sync(gpu=0)
        out.append(c.materialise("example_score", j, key=k))
    return out

ARGV_S = ["custom_selection_eval.py", "--workload", "gsm8k"]
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td); R = tmp / "records"; R.mkdir()
    full = rec("mls-full-attention-gsm8k", HS, ARGV_S, 32.5)
    (R / "full.json").write_text(json.dumps(full))
    # a quant record of the same workload with NO table of its family: the module stage must not serve it
    (R / "q.json").write_text(json.dumps(rec("mls-kivi-hotpot", "7" * 64, ["custom_quant_eval.py", "--workload", "longbench_hotpotqa"], 40.0,
                                             task="llm-kv-adaptive-quantization", workload="longbench_hotpotqa")))
    keys, ids = [KEY(e) for e in EX], [e["example_id"] for e in EX]
    (R / "example-order-both-gsm8k.json").write_text(json.dumps(table(["selection", "quant"], "gsm8k", keys, ids)))
    (R / "example-order-bad-dup.json").write_text(json.dumps(table(["selection"], "longbench_v2", keys[:3] + keys[:1], ids[:4])))
    (R / "example-order-bad-n.json").write_text(json.dumps(table(["quant"], "longbench_hotpotqa", keys[:5], ids[:5], n_examples_full=6)))
    (R / "example-order-bad-key.json").write_text(json.dumps(table(["quant"], "needlebench_niah", keys[:5], ids[:5], key="sha256(prompt)")))

    # 1. tables
    reg = gd.Registry(tmp)
    chk("1a: the good table loads for both families; the three bad ones are refused",
        sorted(reg.example_order) == [("quant", "gsm8k"), ("selection", "gsm8k")], str(sorted(reg.example_order)))
    chk("1b: tables are not records", all(not r.startswith("example-order") for r in reg._by_id), str(sorted(reg._by_id)))
    chk("1c: order_for finds the record's table (family from source_key.task, workload from decode)",
        (reg.order_for(full) or {}).get("n") == N and reg.order_for(json.loads((R / "q.json").read_text())) is None)
    vec = gd._score_vector(full)

    d, sock, occ = start_daemon(tmp / "d", 1000.0, registry=R)
    MOD = [{"name": "kv_shard_eval", "path": "/home/researcher/project/methods/full_attention/custom_selection_eval.py", "sha256": HS}]

    # 2. the module stage
    c = DaemonClient(str(sock)); c.hello(argv=["run_gsm8k_shard.py", "--start", "10"])
    lk = c.lookup_v2(job_key="j-mod", entry_hash=HR, argv=["x", "--workload", "gsm8k"], modules=MOD, stage="exact")
    chk("2a: the evaluator loaded as a module matches exact_module, argv[0] taken from the module",
        lk.get("match") == "exact_module" and lk.get("mode") == "replay" and lk.get("record_id") == "mls-full-attention-gsm8k"
        and str(lk.get("entry_module", "")).endswith("custom_selection_eval.py"), str({k: lk.get(k) for k in ("match", "mode", "record_id", "entry_module")}))
    # 3. a keyed shard [10, 20)
    got = run_slice(c, list(range(10, 20)))
    chk("3a: shard [10, 20) is served the record's entries 10..19", got == vec[10:20], f"{got} vs {vec[10:20]}")
    c.bye()
    c2 = DaemonClient(str(sock)); c2.hello(argv=["x"])
    lk2 = c2.lookup_v2(job_key="j-mod-wrongargv", entry_hash=HR, argv=["x", "--workload", "longbench_v2"], modules=MOD, stage="exact")
    chk("2b: the module stage still needs the argv to match (longbench_v2 is not the record's)", lk2.get("match") == "none", str(lk2.get("match")))
    c2.bye()
    c3 = DaemonClient(str(sock)); c3.hello(argv=["x"])
    lk3 = c3.lookup_v2(job_key="j-mod-quant", entry_hash=HR, argv=["x", "--workload", "longbench_hotpotqa"],
                       modules=[{"name": "q", "path": "/p/custom_quant_eval.py", "sha256": "7" * 64}], stage="exact")
    chk("2c: a module whose records have no order table does not stand in", lk3.get("match") == "none", str(lk3.get("match")))
    c3.bye()
    c4 = DaemonClient(str(sock)); c4.hello(argv=["x"])
    lk4 = c4.lookup_v2(job_key="j-nomods", entry_hash=HR, argv=["x", "--workload", "gsm8k"], stage="exact")
    chk("2d: without the module list the runner misses, as before r4", lk4.get("match") == "none", str(lk4.get("match")))
    c4.bye()

    # 4. four shards covering the workload, one per GPU-less client in turn
    pooled = []
    for si, (a, b) in enumerate([(0, 10), (10, 20), (20, 30), (30, 40)]):
        cs = DaemonClient(str(sock)); cs.hello(argv=["x"])
        cs.lookup_v2(job_key=f"j-shard{si}", entry_hash=HR, argv=["x", "--workload", "gsm8k"], modules=MOD, stage="exact")
        pooled += run_slice(cs, list(range(a, b)))
        cs.bye()
    chk("4: four shards serve the whole vector once; pooled mean = the recorded 32.5",
        pooled == vec and abs(sum(pooled) / N - 32.5) < 1e-9, f"mean {sum(pooled) / N}")

    # 5. a key the table does not hold
    c5 = DaemonClient(str(sock)); c5.hello(argv=["x"])
    c5.lookup_v2(job_key="j-miss", entry_hash=HR, argv=["x", "--workload", "gsm8k"], modules=MOD, stage="exact")
    per = WALL / N
    c5.enqueue(gpu=0, op="replay", modelled_s=per, step=1, unit="example", key="f" * 64); c5.sync(gpu=0)
    v5 = c5.materialise("example_score", 0, key="f" * 64)
    c5.bye()

    # 6. the evaluator run as a script, whole, no keys
    c6 = DaemonClient(str(sock)); c6.hello(argv=ARGV_S)
    lk6 = c6.lookup_v2(job_key="j-script", entry_hash=HS, argv=ARGV_S, stage="exact")
    got6 = run_slice(c6, list(range(N)), keyed=False)
    c6.bye()
    chk("6a: the whole launch run as a script matches exact and is served the vector in order",
        lk6.get("match") == "exact" and got6 == vec, str(lk6.get("match")))

    # 7. the gate
    c7 = DaemonClient(str(sock)); c7.hello(argv=["x"])
    c7.lookup_v2(job_key="j-gate", entry_hash=HR, argv=["x", "--workload", "gsm8k"], modules=MOD, stage="exact")
    c7.enqueue(gpu=0, op="replay", modelled_s=per, step=1, unit="example", key=KEY(EX[5])); c7.sync(gpu=0)
    v7a = c7.materialise("example_score", 0, key=KEY(EX[6]))          # paced 5, asks 6's score by key
    v7b = c7.materialise("example_score", 0, key=KEY(EX[5]))          # its own
    c7.enqueue(gpu=0, op="replay", modelled_s=per, step=5, unit="example", key=KEY(EX[7])); c7.sync(gpu=0)   # out of order
    v7c = c7.materialise("example_score", 1, key=KEY(EX[7]))
    c7.bye()
    chk("7a: a score asked by the key of an example the job did not pace is refused", v7a is None, str(v7a))
    chk("7b: the job's own paced example is served", v7b == vec[5], f"{v7b} vs {vec[5]}")
    chk("7c: an out-of-order unit opens nothing, whatever its key", v7c is None, str(v7c))

    import time; time.sleep(0.2)
    ev = read_occ(occ)
    jobs = {e["job"]: e for e in ev if e.get("kind") == "job_hello"}
    look = {e["job"]: e for e in ev if e.get("kind") == "lookup"}
    byes = {e["job"]: e for e in ev if e.get("kind") == "job_bye"}
    jid = lambda key: next(j for j, e in look.items() if e.get("job_key") == key)
    j3 = jid("j-mod")
    m3 = [e for e in ev if e.get("kind") == "materialise" and e.get("job") == j3 and e.get("role") == "example_score"]
    chk("3b: each shard score's event names its record position and labels it a subset",
        [e.get("position") for e in m3] == list(range(10, 20)) and all(e.get("provenance") == "synthesised_subset" for e in m3),
        str([(e.get("example"), e.get("position"), e.get("provenance")) for e in m3][:3]))
    b3 = byes.get(j3) or {}
    chk("3c: job_bye summarises the shard: 10 positions 10..19, contiguous, not a prefix; entry_module named",
        b3.get("positions") == {"n": 10, "first": 10, "last": 19, "contiguous": True, "prefix": False}
        and str(b3.get("entry_module", "")).endswith("custom_selection_eval.py"), str({k: b3.get(k) for k in ("positions", "entry_module")}))
    chk("3d: the lookup event names the module and the table",
        look[j3].get("entry_module", "").endswith("custom_selection_eval.py") and look[j3].get("entry_module_hash") == HS
        and look[j3].get("example_order") == "example-order-both-gsm8k.json", str({k: look[j3].get(k) for k in ("entry_module", "example_order")}))
    j5 = jid("j-miss")
    miss = [e for e in ev if e.get("kind") == "example_order_miss" and e.get("job") == j5]
    chk("5: an unknown key is served at its local index, logged and counted",
        v5 == vec[0] and len(miss) == 1 and (byes.get(j5) or {}).get("example_order_misses") == 1, f"v={v5} miss={len(miss)}")
    j6 = jid("j-script")
    m6 = [e for e in ev if e.get("kind") == "materialise" and e.get("job") == j6 and e.get("role") == "example_score"]
    chk("6b: the script launch's events carry no position and provenance result; job_bye: a prefix of 40",
        all(e.get("position") is None and e.get("provenance") == "result" for e in m6)
        and (byes.get(j6) or {}).get("positions") == {"n": N, "first": 0, "last": N - 1, "contiguous": True, "prefix": True},
        str((byes.get(j6) or {}).get("positions")))
    ra = [e for e in ev if e.get("kind") == "read_ahead" and e.get("job") == jid("j-gate")]
    chk("7d: the refusals are logged as read_ahead with the position asked", len(ra) == 2 and ra[0].get("position") == 6,
        str([(e.get("step"), e.get("position")) for e in ra]))

print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
