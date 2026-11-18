#!/usr/bin/env python3
"""The whole_job serving mode at the daemon (next/ r8): a record whose serving
block is `mode: whole_job` is served its recorded result only to a job that has
paced every one of its units, in order, at the record's per-unit cost, and whose
last unit's GPU time has passed.
  1. the lookup reply carries the seam and unit plan, and no result: no
     result_key value, no `result.*`;
  2. a job that paces all n units and waits for the last gets the recorded
     result, tier result, and job_bye says served=result with units (n, n);
  3. a result asked for before the last unit is paced: None, a read_ahead
     event, and the job's draw goes back at bye (served=none);
  4. a result asked for after all units were enqueued but before the last one's
     GPU time passed: None; the same job asking again after it passed: served;
  5. units priced under the record's share, or out of order, do not count;
  6. unit_shares: unequal shares, each priced at its share, all count;
  7. a client that never paces (a bare lookup and a materialise) gets nothing;
  8. a record whose only blocker is the whole_job serving is drawable.
    python3 -I tests/test_whole_job_daemon.py
"""
import json, sys, tempfile, time
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

WALL, NU = 20.0, 4
RESULT = {"ndcg_at_10": 0.6123, "map_at_10": 0.5501, "main_score": 0.6123}

def rec(rid, h, n_units=NU, shares=None, ready=True, blockers=None):
    ser = {"mode": "whole_job", "seam": "mteb.abstasks.AbsTaskRetrieval:AbsTaskRetrieval._evaluate_subset",
           "unit": "corpus chunk", "n_units": n_units, "result_key": "scores", "result_kind": "score_dict",
           "result_shape": "dict"}
    if shares is not None:
        ser["unit_shares"] = shares
    r = {"identity": {"record_id": rid, "entry_hash": h, "argv": ["mteb", "run", "-m", "m", "-t", rid],
                      "replicate_group": rid, "match_policy": "exact_only"},
         "mode": "replay", "steps": None, "timing": {"pacing": "total", "wall_total_s": {"value": WALL, "provenance": "recorded"}},
         "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30}, "series": {},
         "result": {"scores": {"value": RESULT, "provenance": "recorded", "how": "test"}},
         "materialisation": {"serving": ser},
         "readiness": {"replay": ready, **({"missing": blockers} if blockers else {})}, "failures": []}
    return r

H = "a" * 64
def look(c, rid, key):
    c.hello(argv=["mteb", "run", "-m", "m", "-t", rid])
    return c.lookup_v2(job_key=key, rank=0, world_size=1, entry_hash=H, argv=["mteb", "run", "-m", "m", "-t", rid], env={},
                       subset_env={}, script_path=None, fingerprint=None, fingerprint_inputs=None, stage="exact")

def units(c, costs, start=1, sync=True):
    for j, cost in enumerate(costs):
        c.enqueue(gpu=0, op="replay", modelled_s=cost, step=start + j, unit="whole_job")
        if sync:
            c.sync(gpu=0)

def main():
    tmp = Path(tempfile.mkdtemp()); reg = tmp / "reg"; reg.mkdir()
    for rid in ("t1", "t2", "t3", "t5", "t7"):
        (reg / f"{rid}.json").write_text(json.dumps(rec(rid, H)))
    (reg / "t6.json").write_text(json.dumps(rec("t6", H, shares=[0.1, 0.2, 0.3, 0.4])))
    (reg / "t8.json").write_text(json.dumps(rec("t8", H, ready=False, blockers=["gpu-mock whole_job serving (next/ r8)"])))
    (reg / "t9.json").write_text(json.dumps(rec("t9", H, ready=False, blockers=["gpu-mock whole_job serving (next/ r8)", "the mteb box image"])))
    per = WALL / NU
    d, sock, occ = start_daemon(tmp / "d", speed=100.0, registry=reg)
    try:
        # 1
        c = DaemonClient(str(sock)); r = look(c, "t1", "j1")
        sv = r.get("serving") or {}; wj = sv.get("whole_job") or {}
        flat = json.dumps(r)
        chk("1: the reply carries the seam and unit plan", r.get("mode") == "replay" and wj.get("seam", "").endswith("_evaluate_subset")
            and wj.get("n_units") == NU and wj.get("result_shape") == "dict", json.dumps(wj))
        chk("1: the reply carries no result (no result_key, no recorded value)", "result_key" not in wj and "0.6123" not in flat
            and "ndcg_at_10" not in flat, flat[:200])
        # 2
        units(c, [per] * NU)
        v = c.materialise("whole_job_result", 0)
        chk("2: all units paced and drained: the recorded result, exactly", v == RESULT, str(v))
        c.bye(); c.close()
        # 3
        c = DaemonClient(str(sock)); look(c, "t2", "j2")
        units(c, [per] * (NU - 1))
        chk("3: asked after 3 of 4 units: nothing", c.materialise("whole_job_result", 0) is None)
        c.bye(); c.close()
        # 4
        c = DaemonClient(str(sock)); look(c, "t3", "j3")
        units(c, [per] * NU, sync=False)
        early = c.materialise("whole_job_result", 0)
        c.sync(gpu=0); time.sleep(0.1)
        late = c.materialise("whole_job_result", 0)
        chk("4: all 4 enqueued, before the last one's GPU time: nothing; after it: served", early is None and late == RESULT, f"{early} / {late}")
        c.bye(); c.close()
        # 5
        c = DaemonClient(str(sock)); look(c, "t5", "j5")
        units(c, [per * 0.5] * NU)
        cheap = c.materialise("whole_job_result", 0)
        for j in (3, 1, 2, 4):
            c.enqueue(gpu=0, op="replay", modelled_s=per, step=j, unit="whole_job")
        c.sync(gpu=0)
        shuffled = c.materialise("whole_job_result", 0)
        chk("5: units under the record's share, or out of order, do not count", cheap is None and shuffled is None, f"{cheap} / {shuffled}")
        c.bye(); c.close()
        # 6
        c = DaemonClient(str(sock)); look(c, "t6", "j6")
        units(c, [WALL * x for x in (0.1, 0.2, 0.3, 0.4)])
        chk("6: unequal unit_shares, each priced at its share: served", c.materialise("whole_job_result", 0) == RESULT)
        c.bye(); c.close()
        # 7
        c = DaemonClient(str(sock)); look(c, "t7", "j7")
        chk("7: a client that never paces gets nothing", c.materialise("whole_job_result", 0) is None)
        c.bye(); c.close()
        # 8
        c = DaemonClient(str(sock)); r8 = look(c, "t8", "j8"); c.bye(); c.close()
        c = DaemonClient(str(sock)); r9 = look(c, "t9", "j9"); c.bye(); c.close()
        chk("8: only the whole_job blocker: drawable; another blocker too: not", r8.get("mode") == "replay" and r9.get("mode") != "replay",
            f"{r8.get('mode')} / {r9.get('mode')}")
    finally:
        time.sleep(0.3)        # the in-process daemon (a daemon thread) logs job_bye as each connection closes
    ev = read_occ(occ)
    byes = {e.get("job"): e for e in ev if e.get("kind") == "job_bye"}
    lk = {e.get("record_id"): e.get("job") for e in ev if e.get("kind") == "lookup"}
    b1, b2 = byes.get(lk.get("t1")) or {}, byes.get(lk.get("t2")) or {}
    chk("2: job_bye says served=result, units (4, 4)", b1.get("served") == "result" and list(b1.get("whole_job_units") or []) == [4, 4], str({k: b1.get(k) for k in ("served", "whole_job_units")}))
    chk("3: job_bye says served=none, units (3, 4), and the draw went back", b2.get("served") == "none" and list(b2.get("whole_job_units") or []) == [3, 4]
        and b2.get("draw_returned_incomplete") is True, str({k: b2.get(k) for k in ("served", "whole_job_units", "draw_returned_incomplete")}))
    ra = [e for e in ev if e.get("kind") == "read_ahead" and e.get("role") == "whole_job_result"]
    chk("3, 4, 5, 7: each refused read logs read_ahead (role whole_job_result)", len(ra) >= 5, f"{len(ra)} events")
    mats = [e for e in ev if e.get("kind") == "materialise" and e.get("role") == "whole_job_result"]
    chk("2, 4, 6: each served result logs materialise tier result with provenance recorded",
        len(mats) == 3 and all(e.get("tier") == "result" and e.get("provenance") == "recorded" for e in mats), f"{len(mats)} events")
    # eval-metric gating (next/ r8): a val_loss/eval_accuracy read before the
    # series' first recorded step returns None, never a future or final value
    gd2 = gd        # the daemon module imported at the top of this file
    class Occ:
        def event(self, *a, **k): pass
    class J:
        record = {"identity": {"record_id": "r"},
                  "series": {"val_loss": {"steps": [192, 384], "values": [0.94, 0.92]},
                             "train_loss": {"steps": [192], "values": [0.94]}},
                  "result": {"val_loss": {"value": 0.92, "provenance": "recorded"}}, "steps": 400}
        record_id = "r"; trial = 1
        def __getattr__(self, name): return None
    dd = gd2.GPUDaemon.__new__(gd2.GPUDaemon); dd.occ = Occ(); dd._beyond_record = lambda job, st: st > 400
    def mat(**req): return dd._materialise(J(), req)
    chk("eval: val_loss before the first validation is None, not the final 0.92", mat(role="val_loss", step=10)["value"] is None, str(mat(role="val_loss", step=10)))
    chk("eval: val_loss at a validated step is the recorded value", mat(role="val_loss", step=200)["value"] == 0.94)
    chk("eval: a later step reads the latest validation at or before it, never a future one", mat(role="val_loss", step=390)["value"] == 0.92)
    chk("eval: train_loss before step 192 still gets the finite stand-in (NaN-guard path kept)", mat(role="train_loss", step=10)["value"] == 0.94)
    chk("eval: batched covering before the first validation is None", mat(roles=["val_loss"], step=10, pick="covering")["values"]["val_loss"] is None)
    chk("eval: batched covering never serves a future validation", mat(roles=["val_loss"], step=200, pick="covering")["values"]["val_loss"] == 0.94)

    print(f"{n - fails} passed, {fails} failed")
    sys.exit(1 if fails else 0)

main()
