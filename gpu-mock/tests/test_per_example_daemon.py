#!/usr/bin/env python3
"""per_example serving at the daemon (docs/MLS-KV-SERVING.md): the score
vector's mean is the recorded final_score exactly, binary vectors carry the
pinned hit count, subsets are detected from the flag or the env and labelled
on lookup, materialise and job_bye, a repeated launch is deterministic_rerun.
    python3 -I tests/test_per_example_daemon.py
"""
import json, sys, tempfile
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

def rec(rid, h, argv, score, n_full, kind, wall=347.3):
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "replicate_group": rid, "match_policy": "exact_only"},
            "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": wall, "provenance": "recorded"}},
            "memory": {"peak_allocated_bytes": 8 * 1024 ** 3, "peak": 8 * 1024 ** 3}, "series": {},
            "result": {"final_score": {"value": score, "provenance": "recorded"}},
            "materialisation": {"serving": {"mode": "per_example", "generate_seam": "generate_with_selection", "score_seam": "score_prediction",
                                            "n_examples_full": {"value": n_full, "provenance": "derived"}, "score_kind": kind,
                                            "subset_source": {"flag": "--max-examples", "env": "SELECTION_KV_MAX_EXAMPLES"},
                                            "load_s": {"value": None, "provenance": "absent"}, "scale": [0, 100]}},
            "readiness": {"replay": True}, "failures": []}

def paced_scores(c, n, wall=347.3, n_full=None, first_startup=None):
    """Read scores 0..n-1 the way the patch's seams do (replay._pace_example,
    then the score seam): pace example i at the record's share (wall − start-up)
    / n_full, wait for its unit, read its score. Since next/ r2 the daemon serves
    a score only for an example the job paced, in order, at that share; at the
    test daemon's speed 1000 each unit is about a millisecond of wall."""
    out = []
    nf = n_full or n
    load = first_startup[0] if first_startup else 0.0
    per = max(0.0, wall - load) / nf
    for i in range(n):
        kw = {"startup_s": first_startup[0], "startup_real_s": first_startup[1]} if (first_startup and i == 0) else {}
        c.enqueue(gpu=0, op="replay", modelled_s=per, step=i + 1, unit="example", **kw)
        c.sync(gpu=0)
        out.append(c.materialise("example_score", i))
    return out

HB, HC, HD = "b" * 64, "c" * 64, "d" * 64
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td); R = tmp / "records"; R.mkdir()
    (R / "gsm.json").write_text(json.dumps(rec("mls-full-attention-gsm8k", HB, ["custom_selection_eval.py", "--workload", "gsm8k"], 31.842305, 1319, "binary")))
    hot = rec("mls-full-attention-hotpotqa", HC, ["custom_selection_eval.py", "--workload", "longbench_hotpotqa"], 61.083, 300, "continuous")
    psg = rec("mls-full-attention-passage", HD, ["custom_selection_eval.py", "--workload", "longbench_passage_retrieval"], 62.416667, 300, "retrieval")
    (R / "psg.json").write_text(json.dumps(psg))
    hot["readiness"] = {"replay": False, "missing": ["gpu-mock per_example serving, in progress 2026-10-22"]}   # as gpu-tasks shipped the 40
    (R / "hot.json").write_text(json.dumps(hot))
    # the vector itself
    vb = gd._score_vector(json.loads((R / "gsm.json").read_text()))
    chk("binary vector: n_full entries, 420 hits of 100, mean = 100·hits/n", len(vb) == 1319 and vb.count(100.0) == 420 and set(vb) <= {0.0, 100.0} and abs(sum(vb) / 1319 - 31.842305) < 1e-4, f"{len(vb)} hits {vb.count(100.0)} mean {sum(vb)/1319:.6f}")
    vc_ = gd._score_vector(json.loads((R / "hot.json").read_text()))
    chk("continuous vector: 300 entries in [0,100], mean exactly 61.083, not degenerate", len(vc_) == 300 and all(0 <= v <= 100 for v in vc_) and abs(sum(vc_) / 300 - 61.083) < 1e-9 and len(set(round(v, 3) for v in vc_)) > 50, f"mean {sum(vc_)/300:.9f} distinct {len(set(vc_))}")
    chk("deterministic per record id", gd._score_vector(json.loads((R / "hot.json").read_text())) == vc_)
    d, sock, occ = start_daemon(tmp / "d", speed=1000.0, registry=R)
    # a full launch
    c = DaemonClient(str(sock)); c.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k"])
    r = c.lookup_v2(job_key="F", rank=0, world_size=1, entry_hash=HB, argv=["custom_selection_eval.py", "--workload", "gsm8k"], env={}, subset_env={}, script_path="/w/methods/full_attention/custom_selection_eval.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    sv = r.get("serving") or {}
    chk("full launch: replay, per_example serving with the block and the wall; no final_score or outcome in the reply (next/ r2)", r.get("mode") == "replay" and (sv.get("per_example") or {}).get("score_seam") == "score_prediction" and sv.get("wall_total_s") == 347.3 and "final_score" not in sv and "outcome" not in sv and r.get("subset") is None, str({k: sv.get(k) for k in ("serving_mode", "wall_total_s", "final_score", "outcome")}) + str(r.get("subset")))
    scores = paced_scores(c, 1319, n_full=1319)
    chk("the script's mean of the served scores is the recorded final_score", abs(sum(scores) / 1319 - 31.842305) < 1e-4 and scores == vb, f"{sum(scores)/1319:.6f}")
    # a script with more examples than the record says paces them too (the
    # NeedleBench 147/441 case): the read is past n_full, not ahead of pacing
    c.enqueue(gpu=0, op="replay", modelled_s=347.3 / 1319, step=1320, unit="example"); c.sync(gpu=0)
    chk("a read past n_full is None", c.materialise("example_score", 1319) is None)
    c.enqueue(gpu=0, op="replay", modelled_s=347.3 / 1319, step=1321, unit="example"); c.sync(gpu=0)
    chk("a second read past n_full is None too (one overrun event per job, checked below)", c.materialise("example_score", 1320) is None)
    c.bye(); c.close()
    # a subset launch by flag, and one by env
    c2 = DaemonClient(str(sock)); c2.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "20"])
    r2 = c2.lookup_v2(job_key="S", rank=0, world_size=1, entry_hash=HB, argv=["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "20"], env={}, subset_env={}, script_path="/w/x.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("subset by flag: subset {k:20, n_full:1319} on the lookup; same record; draw deterministic_rerun", r2.get("subset") == {"k": 20, "n_full": 1319} and r2.get("record_id") == "mls-full-attention-gsm8k" and r2.get("draw") == "deterministic_rerun", str({k: r2.get(k) for k in ("subset", "record_id", "draw")}))
    # the patch states the launch's real start-up with its first example; the
    # daemon repeats it on every served score and on job_bye
    sub = paced_scores(c2, 20, n_full=1319, first_startup=(12.5, 13.25))
    chk("the subset's scores are the first 20 entries of the same vector", sub == vb[:20])
    c2.bye(); c2.close()
    c3 = DaemonClient(str(sock)); c3.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k"])
    r3 = c3.lookup_v2(job_key="E", rank=0, world_size=1, entry_hash=HB, argv=["custom_selection_eval.py", "--workload", "gsm8k"], env={}, subset_env={"SELECTION_KV_MAX_EXAMPLES": "50"}, script_path="/w/x.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("subset by env: k 50", r3.get("subset") == {"k": 50, "n_full": 1319}, str(r3.get("subset")))
    r4 = c3.lookup_v2(job_key="E2", rank=0, world_size=1, entry_hash=HB, argv=["custom_selection_eval.py", "--workload", "gsm8k", "--max-examples", "0"], env={}, subset_env={"SELECTION_KV_MAX_EXAMPLES": "50"}, script_path="/w/x.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("the flag wins over the env: --max-examples 0 is a full launch", r4.get("subset") is None, str(r4.get("subset")))
    c3.bye(); c3.close()
    c4 = DaemonClient(str(sock)); c4.hello(argv=["custom_selection_eval.py", "--workload", "longbench_hotpotqa"])
    r5 = c4.lookup_v2(job_key="H", rank=0, world_size=1, entry_hash=HC, argv=["custom_selection_eval.py", "--workload", "longbench_hotpotqa"], env={}, subset_env={}, script_path="/w/x.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("a record blocked only by the per_example serving is served by a daemon that has it, labelled", r5.get("mode") == "replay" and r5.get("record_id") == "mls-full-attention-hotpotqa", str({k: r5.get(k) for k in ("mode", "record_id", "match")}))
    c4.bye(); c4.close()
    # retrieval: the recorded sum is not a whole number of hits (62.416667 × 3 = 187.25), served exactly
    c5 = DaemonClient(str(sock)); c5.hello(argv=["custom_selection_eval.py", "--workload", "longbench_passage_retrieval"])
    r6 = c5.lookup_v2(job_key="P", rank=0, world_size=1, entry_hash=HD, argv=["custom_selection_eval.py", "--workload", "longbench_passage_retrieval"], env={}, subset_env={}, script_path="/w/x.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    pv = paced_scores(c5, 300, n_full=300)
    c5.bye(); c5.close()
    whole = sorted(v for v in pv if v in (0.0, 100.0)); frac = [v for v in pv if v not in (0.0, 100.0)]
    chk("retrieval: the served mean is the recorded final_score exactly (not a rounded hit count)", r6.get("record_id") == "mls-full-attention-passage" and abs(sum(pv) / 300 - 62.416667) < 1e-9, f"{sum(pv) / 300:.9f}")
    chk("retrieval: 187 hits, 112 misses and one example at the fractional remainder (25.0)", whole.count(100.0) == 187 and whole.count(0.0) == 112 and len(frac) == 1 and abs(frac[0] - 25.0001) < 0.01, f"hits {whole.count(100.0)} misses {whole.count(0.0)} frac {frac}")
    # the oracle route (next/ r2): a client that reads scores without pacing gets none
    co = DaemonClient(str(sock)); co.hello(argv=["custom_selection_eval.py", "--workload", "gsm8k"])
    ro = co.lookup_v2(job_key="O", rank=0, world_size=1, entry_hash=HB, argv=["custom_selection_eval.py", "--workload", "gsm8k"], env={}, subset_env={}, script_path="/w/methods/full_attention/custom_selection_eval.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    unpaced = [co.materialise("example_score", i) for i in range(50)]
    chk("read-ahead: 50 unpaced reads after an exact lookup are all refused (None)", ro.get("mode") == "replay" and unpaced == [None] * 50, str(unpaced[:5]))
    co.enqueue(gpu=0, op="replay", modelled_s=500.0, step=1, unit="example")   # 0.5 s of wall at the test daemon's speed 1000, above the floor
    early = co.materialise("example_score", 0)
    chk("read-ahead: example 0 paced but its unit (0.5 s of wall) not yet drained is refused", early is None, str(early))
    co.sync(gpu=0)
    chk("read-ahead: after the unit drains, example 0 is served its entry, and example 1 (unpaced) still is not", co.materialise("example_score", 0) == vb[0] and co.materialise("example_score", 1) is None)
    # forgeries (r2b): a unit labelled far ahead, a unit priced at zero, a huge claimed back-date
    co.enqueue(gpu=0, op="replay", modelled_s=347.3 / 1319, step=1000, unit="example"); co.sync(gpu=0)
    chk("forged label: a unit labelled step 1000 opens nothing (example 999 refused)", co.materialise("example_score", 999) is None)
    co.enqueue(gpu=0, op="replay", modelled_s=0.0, step=2, unit="example"); co.sync(gpu=0)
    chk("forged price: the job's next example (1) at zero cost opens nothing", co.materialise("example_score", 1) is None)
    co.enqueue(gpu=0, op="replay", modelled_s=347.3 / 1319, step=2, unit="example", host_gap_s=1e6)
    co.sync(gpu=0)
    bd = [e for e in read_occ(occ) if e["kind"] == "enqueue" and e.get("host_gap_s") is not None and e["job"] == max(x["job"] for x in read_occ(occ) if x["kind"] == "job_hello")]
    chk("forged back-date: a claimed host_gap_s of 1e6 s is logged and used clamped to the real gap since the job's previous unit ended (< 5 s)", bd and bd[-1]["host_gap_s"] < 5.0, str([e.get("host_gap_s") for e in bd][-3:]))
    chk("the honest next unit (example 1 at the record's share) opens example 1", co.materialise("example_score", 1) == vb[1])
    co.bye(); co.close()
    ev = read_occ(occ)
    chk("the lookup event says readiness_overridden per_example", any(e.get("readiness_overridden") == "per_example" for e in ev if e["kind"] == "lookup"))
    mats = [e for e in ev if e["kind"] == "materialise" and e["job"] == 2]
    chk("subset reads: tier result, provenance synthesised_subset, subset_model bernoulli, example index", mats and all(e.get("tier") == "result" and e.get("provenance") == "synthesised_subset" and e.get("subset_model") == {"kind": "bernoulli"} and e.get("subset") == {"k": 20, "n_full": 1319} for e in mats) and [e.get("example") for e in mats] == list(range(20)), str(mats[0] if mats else None)[:300])
    full = [e for e in ev if e["kind"] == "materialise" and e["job"] == 1]
    chk("full-launch reads: provenance result, no subset", all(e.get("provenance") == "result" and e.get("subset") is None and e.get("subset_model") is None for e in full) and len(full) == 1321, f"{len(full)} reads")
    ra = [e for e in ev if e["kind"] == "read_ahead"]
    oracle_job = max(e["job"] for e in ev if e["kind"] == "job_hello")
    chk("read_ahead events: 3 (capped), all the oracle client's, none for a job that paced", len(ra) == 3 and all(e["job"] == oracle_job for e in ra) and ra[0].get("paced") == 0, str([(e.get("job"), e.get("step"), e.get("paced"), e.get("wait_s")) for e in ra]))
    byes = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    over = [e for e in ev if e["kind"] == "failure" and e.get("mode") == "per_example_overrun"]
    chk("per_example_overrun: one failure event for the launch that read past n_full (step 1319, n_full 1319), none for the others", len(over) == 1 and over[0]["job"] == 1 and over[0].get("step") == 1319 and over[0].get("n_full") == 1319, str([(e.get("job"), e.get("step"), e.get("n_full")) for e in over]))
    enq2 = [e for e in ev if e["kind"] == "enqueue" and e["job"] == 2]
    chk("startup_s: logged on the first example's enqueue, on every served score of that launch, on its job_bye; absent where never sent", enq2 and enq2[0].get("startup_s") == 12.5 and all(e.get("startup_s") == 12.5 for e in mats) and byes[2].get("startup_s") == 12.5 and all(e.get("startup_s") is None for e in full) and byes[1].get("startup_s") is None and all(e.get("startup_real_s") == 13.25 for e in mats) and byes[2].get("startup_real_s") == 13.25, f"enq {[e.get('startup_s') for e in enq2]} scores {sorted({e.get('startup_s') for e in mats})} bye {byes[2].get('startup_s')}")
    chk("job_bye: served result; the subset launch carries its subset", byes[1].get("served") == "result" and byes[1].get("subset") is None and byes[2].get("subset") == {"k": 20, "n_full": 1319}, str({k: (v.get("served"), v.get("subset")) for k, v in byes.items()}))
    look = [e for e in ev if e["kind"] == "lookup"]
    chk("lookup events carry subset and the deterministic_rerun draw", [e.get("subset") for e in look][:2] == [None, {"k": 20, "n_full": 1319}] and look[1].get("draw") == "deterministic_rerun" and look[0].get("draw") != "deterministic_rerun", str([(e.get("draw"), e.get("subset")) for e in look]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
