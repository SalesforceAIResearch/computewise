#!/usr/bin/env python3
"""Measured per-example scores at the daemon (next/ r6): a record that carries
the real scores of its first k examples serves exactly those scores at
positions 0..k-1, the synthesised tail beyond, and the whole vector still
averages to the recorded final_score.
  1. without a measured block every vector is the one before r6 (all records
     of this test, and the shape of the change);
  2. a k-subset launch reads back the measured scores, provenance `measured`,
     no subset_model, measured_k on each event;
  3. a full launch's mean is the recorded final_score; the head is measured,
     the tail synthesised and labelled;
  4. a keyed shard inside the head gets measured scores at its own positions,
     one straddling k gets measured then synthesised;
  5. a measured head that the tail cannot balance is clamped and noted, never
     served out of the scale;
  6. a malformed block (k not the length of scores) is ignored: the vector is
     the synthesised one;
  7. daemon start logs the measured-record count and k.
    python3 -I tests/test_measured_scores_daemon.py
"""
import hashlib, json, sys, tempfile, time
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

N, WALL, K = 40, 40.0, 10
EX = [{"example_id": f"gsm8k-{i}", "prompt": f"Problem {i}"} for i in range(N)]
KEY = lambda e: hashlib.sha256((e["example_id"] + "\0" + e["prompt"]).encode()).hexdigest()

def rec(rid, h, score, kind="binary", measured=None):
    r = {"identity": {"record_id": rid, "entry_hash": h, "argv": ["custom_selection_eval.py", "--workload", "gsm8k"],
                      "replicate_group": rid, "match_policy": "exact_only", "source_key": {"task": "llm-kv-selection-budgeting"}},
         "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": WALL, "provenance": "recorded"}},
         "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30}, "series": {},
         "result": {"final_score": {"value": score, "provenance": "recorded"}},
         "materialisation": {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                         "n_examples_full": {"value": N, "provenance": "derived"}, "score_kind": kind,
                                         "subset_source": {"flag": "--max-examples", "env": "SELECTION_KV_MAX_EXAMPLES"},
                                         "load_s": {"value": None, "provenance": "absent"}, "scale": [0, 100], "decode": {"workload": "gsm8k"}}},
         "readiness": {"replay": True}, "failures": []}
    if measured is not None:
        r["materialisation"]["serving"]["per_example_scores"] = {
            "k": len(measured), "key": "sha256(example_id\\0prompt)", "scores": measured,
            "keys": [KEY(e) for e in EX[:len(measured)]], "example_ids": [e["example_id"] for e in EX[:len(measured)]], "how": "test"}
    return r

def paced(c, idxs, keyed=False):
    per = WALL / N; out = []
    for j, i in enumerate(idxs):
        k = KEY(EX[i]) if keyed else None
        c.enqueue(gpu=0, op="replay", modelled_s=per, step=j + 1, unit="example", key=k); c.sync(gpu=0)
        out.append(c.materialise("example_score", j, key=k))
    return out

HEAD = [100.0, 0.0, 0.0, 100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 100.0]          # 4 of 10 measured hits
HM = "a" * 64; HP = "b" * 64; HC = "c" * 64; HX = "d" * 64
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td); R = tmp / "records"; R.mkdir()
    plain = rec("plain", HP, 30.0)
    meas = rec("meas", HM, 30.0, measured=HEAD)
    clamp = rec("clamp", HC, 5.0, measured=[100.0] * K)                    # 10 hits measured, the record total is 2
    bad = rec("bad", HX, 30.0, measured=HEAD); bad["materialisation"]["serving"]["per_example_scores"]["k"] = 7
    for r in (plain, meas, clamp, bad):
        (R / f"{r['identity']['record_id']}.json").write_text(json.dumps(r))
    keys = [KEY(e) for e in EX]
    (R / "example-order-both-gsm8k.json").write_text(json.dumps({"schema": gd.EXAMPLE_ORDER_SCHEMA, "families": ["selection", "quant"],
        "workload": "gsm8k", "n_examples_full": N, "key": "sha256(example_id\\0prompt)", "order": keys, "ids": [e["example_id"] for e in EX]}))

    # 1, 3, 5, 6 at the vector level
    v_plain_r6 = gd._score_vector(plain)
    chk("1: a record without measured scores keeps its vector (mean = final_score, 0/100 values)",
        len(v_plain_r6) == N and abs(sum(v_plain_r6) / N - 30.0) < 1e-9 and set(v_plain_r6) <= {0.0, 100.0})
    vm = gd._score_vector(meas)
    chk("3a: the measured head is the vector's first k entries", vm[:K] == HEAD, str(vm[:K]))
    chk("3b: the whole vector still averages to the recorded final_score", abs(sum(vm) / N - 30.0) < 1e-9, f"{sum(vm) / N}")
    vc = gd._score_vector(clamp)
    chk("5: a head the tail cannot balance is clamped to the scale and noted",
        vc[:K] == [100.0] * K and min(vc) >= 0.0 and max(vc) <= 100.0 and "clamp" in gd._score_vector_note,
        str(gd._score_vector_note.get("clamp")))
    vb = gd._score_vector(bad)
    chk("6: a malformed block (k != len(scores)) is ignored", gd._measured_scores(bad) is None and abs(sum(vb) / N - 30.0) < 1e-9)

    d, sock, occ = start_daemon(tmp / "d", 1000.0, registry=R)
    # 2. a k-subset launch
    c = DaemonClient(str(sock)); c.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "5"])
    c.lookup_v2(job_key="j-sub", entry_hash=HM, argv=["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "5"], stage="exact")
    got = paced(c, list(range(5))); c.bye()
    chk("2a: a 5-example launch reads back the measured scores", got == HEAD[:5], str(got))
    # 3. a full launch
    c = DaemonClient(str(sock)); c.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k"])
    c.lookup_v2(job_key="j-full", entry_hash=HM, argv=["custom_selection_eval.py", "--workload", "gsm8k"], stage="exact")
    full = paced(c, list(range(N))); c.bye()
    chk("3c: a full launch prints the recorded final_score", abs(sum(full) / N - 30.0) < 1e-9 and full[:K] == HEAD, f"{sum(full) / N}")
    # 4. keyed shards: inside the head, and straddling k
    c = DaemonClient(str(sock)); c.hello(argv=["x"])
    c.lookup_v2(job_key="j-shard-in", entry_hash="e" * 64, argv=["x", "--workload", "gsm8k"],
                modules=[{"name": "m", "path": "/p/custom_selection_eval.py", "sha256": HM}], stage="exact")
    s_in = paced(c, [3, 4, 5], keyed=True); c.bye()
    c = DaemonClient(str(sock)); c.hello(argv=["x"])
    c.lookup_v2(job_key="j-shard-x", entry_hash="e" * 64, argv=["x", "--workload", "gsm8k"],
                modules=[{"name": "m", "path": "/p/custom_selection_eval.py", "sha256": HM}], stage="exact")
    s_x = paced(c, [8, 9, 10, 11], keyed=True); c.bye()
    chk("4a: a keyed shard inside the head gets the measured scores at its positions", s_in == HEAD[3:6], str(s_in))
    chk("4b: a shard straddling k gets measured then synthesised (the vector's own tail)", s_x == HEAD[8:10] + vm[10:12], f"{s_x} vs {HEAD[8:10] + vm[10:12]}")

    time.sleep(0.2)
    ev = read_occ(occ)
    look = {e["job"]: e for e in ev if e.get("kind") == "lookup"}
    jid = lambda key: next(j for j, e in look.items() if e.get("job_key") == key)
    def mats(key):
        return [e for e in ev if e.get("kind") == "materialise" and e.get("role") == "example_score" and e.get("job") == jid(key)]
    m_sub = mats("j-sub")
    chk("2b: each subset score is labelled measured, no subset_model, measured_k on the event",
        all(e.get("provenance") == "measured" and e.get("subset_model") is None and e.get("measured_k") == K for e in m_sub),
        str([(e.get("provenance"), e.get("measured_k")) for e in m_sub][:2]))
    m_full = mats("j-full")
    chk("3d: the full launch labels the head measured and the tail as before (result)",
        all(e.get("provenance") == "measured" for e in m_full[:K]) and all(e.get("provenance") == "result" for e in m_full[K:]),
        str(sorted({e.get("provenance") for e in m_full})))
    m_x = mats("j-shard-x")
    chk("4c: the straddling shard labels positions 8,9 measured and 10,11 synthesised_subset",
        [e.get("provenance") for e in m_x] == ["measured", "measured", "synthesised_subset", "synthesised_subset"],
        str([(e.get("position"), e.get("provenance")) for e in m_x]))
    rl = [e for e in ev if e.get("kind") == "registry_loaded"]
    chk("7: daemon start logs the measured records (2 usable: meas and clamp; bad is malformed) and k, and the order table's sha256",
        rl and rl[0].get("measured_records") == 2 and rl[0].get("measured_k") == [K]
        and (rl[0].get("example_order") or {}).get("example-order-both-gsm8k.json"), str(rl[:1]))

print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
