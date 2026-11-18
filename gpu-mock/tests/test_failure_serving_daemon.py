#!/usr/bin/env python3
"""Served failures (docs/FAILURE-SERVING.md; analysis 2026-10-23, the H2
package): a record's failure with a `trigger` is armed only for a launch whose
argument satisfies it, so an over-limit launch OOMs at its step with the
recorded exception and stderr while a relaunch under the limit draws the same
record and runs clean; a failure naming a `rank` fires on that rank only; a
plain failure (no trigger) fires as before; suppressed ones never fire."""
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

H1, H2, H3 = "1" * 64, "2" * 64, "3" * 64
STDERR = ("Traceback (most recent call last):\n  File \"/workspace/train.py\", line 88, in <module>\n"
          "    loss.backward()\n  File \"/usr/local/lib/python3.11/site-packages/torch/_tensor.py\", line 648, in backward\n"
          "    torch.autograd.backward(\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. "
          "GPU 0 has a total capacity of 79.19 GiB of which 1.12 GiB is free.\n")
CAUSE = "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 1.12 GiB is free."
F_OOM = {"mode": "oom", "step": 5, "time_s": 12.0, "at": "during_training", "provenance": "recorded", "how": "test",
         "cause": {"value": CAUSE, "provenance": "recorded", "how": "last exception line"},
         "trigger": {"arg": "--batch-size", "op": "ge", "value": 96},
         "stderr": {"value": STDERR, "provenance": "recorded", "how": "the run's stderr"},
         "replay": {"value": "fire", "how": "the job's own"}}
F_SUPP = {"mode": "crash", "step": 1, "at": "during_training", "provenance": "recorded", "replay": {"value": "suppress", "how": "collision"}}
F_KILLED = {"mode": "killed", "step": 3, "at": "during_training", "provenance": "recorded", "cause": {"value": "KeyboardInterrupt"}, "replay": {"value": "fire"}}
F_RANK1 = {"mode": "xid", "step": 2, "at": "during_training", "provenance": "recorded", "rank": 1, "xid": 79, "replay": {"value": "fire"}}

def rec(rid, h, argv, failures, exempt=None):
    return {"identity": {"record_id": rid, "entry_hash": h, "argv": argv, "argv_exempt": exempt or [], "match_policy": "exact_only", "replicate_group": rid},
            "mode": "replay", "steps": 10, "timing": {"pacing": "per_step", "per_step_gpu_s": {"value": 0.01, "provenance": "recorded"}, "steady_step_s": 0.01},
            "launch": {"world_size": 1}, "memory": {"peak_allocated_bytes": 1 << 30, "peak": 1 << 30},
            "series": {"train_loss": {"steps": list(range(1, 11)), "values": [1.0] * 10}}, "readiness": {"replay": True}, "failures": failures}

with tempfile.TemporaryDirectory() as td:
    R = Path(td) / "records"; R.mkdir()
    (R / "oom.json").write_text(json.dumps(rec("oom-trig", H1, ["train.py", "--batch-size", "128"], [F_OOM, F_SUPP], exempt=["--batch-size"])))
    (R / "killed.json").write_text(json.dumps(rec("killed-plain", H2, ["ppo.py"], [F_KILLED])))
    (R / "rank.json").write_text(json.dumps(rec("rank1", H3, ["dist.py"], [F_RANK1])))
    (Path(td) / "d").mkdir()
    d, sock, occ = start_daemon(Path(td) / "d", speed=1000.0, registry=R)
    jobs = 0
    def job(h, argv, eff, rank=0):
        global jobs; jobs += 1
        c = DaemonClient(str(sock)); c.hello(argv=argv)
        r = c.lookup_v2(job_key=f"J{jobs}", rank=rank, world_size=1, entry_hash=h, argv=argv, env={}, subset_env={}, argv_effective=eff,
                        script_path="/w/" + argv[0], fingerprint=None, fingerprint_inputs=None, stage="exact")
        return c, r
    def armed(r): return [f.get("armed") for f in ((r.get("serving") or {}).get("failures") or [])]

    # A. over the limit: armed, fires once at step 5 with the recorded exception and stderr
    c, r = job(H1, ["train.py", "--batch-size", "128"], {"batch_size": "128", "lr": "0.001"})
    chk("A: --batch-size 128 draws the record (batch size exempt from identity)", r.get("match") == "exact" and r.get("record_id") == "oom-trig", str({k: r.get(k) for k in ("match", "record_id")}))
    chk("A: serving marks the triggered failure armed (suppressed one untouched by arming)", armed(r) == [True, True], str(armed(r)))
    f4 = c.pending_failure(step=4); f5 = c.pending_failure(step=5); f6 = c.pending_failure(step=6)
    chk("A: nothing at step 4", f4 is None, str(f4))
    chk("A: fires at step 5 as oom, labelled served_failure", (f5 or {}).get("mode") == "oom" and (f5 or {}).get("label") == "served_failure", str(f5)[:200])
    chk("A: the exception is the recorded cause (type torch.OutOfMemoryError, the real message)", (f5 or {}).get("exception") == {"type": "torch.OutOfMemoryError", "message": CAUSE.split(": ", 1)[1]}, str((f5 or {}).get("exception")))
    chk("A: the recorded stderr travels verbatim", (f5 or {}).get("stderr") == STDERR, repr((f5 or {}).get("stderr"))[:120])
    chk("A: the trigger is echoed", (f5 or {}).get("trigger") == F_OOM["trigger"], str((f5 or {}).get("trigger")))
    chk("A: fires once (step 6 clean)", f6 is None, str(f6))
    c.bye(); c.close()
    # B. under the limit: same record, disarmed, clean to the end
    c, r = job(H1, ["train.py", "--batch-size", "64"], {"batch_size": "64", "lr": "0.001"})
    chk("B: --batch-size 64 draws the same record", r.get("record_id") == "oom-trig", str(r.get("record_id")))
    chk("B: the triggered failure is disarmed for this launch", armed(r)[0] is False, str(armed(r)))
    chk("B: nothing fires at step 5 or 10", c.pending_failure(step=5) is None and c.pending_failure(step=10) is None)
    c.bye(); c.close()
    # C. the launch names no batch size and sent no parsed arguments: unresolved, not armed
    c, r = job(H1, ["train.py"], None)
    chk("C: unresolved trigger (no flag, no Namespace) is not armed", armed(r)[0] is False and c.pending_failure(step=5) is None, str(armed(r)))
    c.bye(); c.close()
    # D. the literal flag token resolves the trigger when no Namespace was sent
    c, r = job(H1, ["train.py", "--batch-size", "256"], None)
    f5 = c.pending_failure(step=5)
    chk("D: --batch-size 256 by literal token: armed and fires", armed(r)[0] is True and (f5 or {}).get("mode") == "oom", str(armed(r)) + " " + str((f5 or {}).get("mode")))
    c.bye(); c.close()
    # E. a plain failure (no trigger) behaves as before: fires, no exception payload for killed
    c, r = job(H2, ["ppo.py"], None)
    f3 = c.pending_failure(step=3)
    chk("E: killed at step 3 fires with no exception/stderr/exit_code (unchanged behaviour)", (f3 or {}).get("mode") == "killed" and f3.get("exception") is None and f3.get("stderr") is None and f3.get("exit_code") is None, str(f3))
    chk("E: no triggers → serving carries no armed flags", armed(r) == [None], str(armed(r)))
    c.bye(); c.close()
    # F. a failure that names a rank fires on that rank only
    c0, r0 = job(H3, ["dist.py"], None, rank=0); c1, r1 = job(H3, ["dist.py"], None, rank=1)
    chk("F: rank 0 does not see the rank-1 xid", c0.pending_failure(step=2) is None)
    g = c1.pending_failure(step=2)
    chk("F: rank 1 fires the xid", (g or {}).get("mode") == "xid" and g.get("xid") == 79, str(g))
    c0.bye(); c0.close(); c1.bye(); c1.close()
    # G. events: kind stays `failure` (the runner's scorer keys on it) with the served_failure label; failures_armed only for triggered records
    ev = read_occ(occ)
    fe = [e for e in ev if e["kind"] == "failure"]
    chk("G: four failure events (A, D, E, F), every one labelled served_failure with its record_id", len(fe) == 4 and all(e.get("label") == "served_failure" and e.get("record_id") for e in fe), str([(e.get("mode"), e.get("label"), e.get("record_id")) for e in fe]))
    chk("G: the oom events carry trigger, provenance, exception type and stderr size", all(e.get("trigger") and e.get("provenance") == "recorded" and e.get("exception") == "torch.OutOfMemoryError" and e.get("stderr_bytes") == len(STDERR) for e in fe if e.get("mode") == "oom"), str([(e.get("exception"), e.get("stderr_bytes")) for e in fe if e.get("mode") == "oom"]))
    fa = [e for e in ev if e["kind"] == "failures_armed"]
    chk("G: failures_armed logged for the four launches of the triggered record only: armed/disarmed/unresolved", [(e.get("armed"), e.get("disarmed"), e.get("unresolved")) for e in fa] == [([0, 1], [], []), ([1], [0], []), ([1], [], [0]), ([0, 1], [], [])], str([(e.get("armed"), e.get("disarmed"), e.get("unresolved")) for e in fa]))
    # H. `where`: a failure placed at the forward fires only when the forward seam asks, in its step
    F_FWD = {"mode": "oom", "step": 5, "at": "during_training", "provenance": "recorded", "where": "forward",
             "exception": {"type": "torch.OutOfMemoryError", "message": "CUDA out of memory. Tried to allocate 2.00 GiB."}, "replay": {"value": "fire"}}
    H4 = "4" * 64
    (R / "fwd.json").write_text(json.dumps(rec("fwd-oom", H4, ["fwd.py"], [F_FWD])))
    d.registry.load() if hasattr(d.registry, "load") else None
    try:
        d.registry.__init__(R)       # re-read the records dir (a fresh daemon would have it)
    except Exception:
        pass
    c, r = job(H4, ["fwd.py"], None)
    if r.get("record_id") != "fwd-oom":
        chk("H: (setup) the forward record is drawn", False, str(r.get("record_id")))
    else:
        chk("H: the optimizer-step seam does not fire a forward-placed failure", c.pending_failure(step=5) is None and c.pending_failure(step=5, where="backward") is None)
        g = c.pending_failure(step=5, where="forward")
        chk("H: the forward seam fires it with the recorded exception", (g or {}).get("mode") == "oom" and (g.get("exception") or {}).get("type") == "torch.OutOfMemoryError", str(g)[:160])
        chk("H: once", c.pending_failure(step=6, where="forward") is None)
    c.bye(); c.close()
    # I. oversubscription: the arrival that pushes a GPU past 80 GiB is the job that OOMs
    GiB = 1 << 30
    c1 = DaemonClient(str(sock)); c1.hello(argv=["a.py"])
    r1 = c1.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=60 * GiB, step=1, unit="step", oom_aware=True)
    chk("I: the first job's 60 GiB fits: oom False, no detail", r1.get("oom") is False and r1.get("oom_detail") is None, str({k: r1.get(k) for k in ("oom", "oom_detail")}))
    c2 = DaemonClient(str(sock)); c2.hello(argv=["b.py"])
    r2 = c2.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=30 * GiB, step=1, unit="step", oom_aware=True)
    det = r2.get("oom_detail") or {}
    chk("I: the second job's 30 GiB crosses the card: oom True with the detail (card, mine, the other holder by pid)",
        r2.get("oom") is True and det.get("capacity_bytes") == 80 * GiB and det.get("mine_bytes") == 30 * GiB
        and det.get("others") == [{"pid": os.getpid(), "bytes": 60 * GiB}], str(det))
    r1b = c1.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=0, step=2, unit="step")
    chk("I: the first job's next unit sees the flag but is not the arrival (no raise, no event)", r1b.get("oom") is True and r1b.get("oom_detail") is None, str({k: r1b.get(k) for k in ("oom", "oom_detail")}))
    c3 = DaemonClient(str(sock)); c3.hello(argv=["c.py"])
    r3 = c3.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=10 * GiB, step=1, unit="step")      # an old patch: not oom-aware
    chk("I: a client that is not oom-aware gets the detail but the daemon logs no failure for it", r3.get("oom") is True and (r3.get("oom_detail") or {}).get("mine_bytes") == 10 * GiB, str(r3.get("oom")))
    c2.bye(); c2.close(); c3.bye(); c3.close()
    import time; time.sleep(0.2)
    r1c = c1.enqueue(gpu=0, op="replay", modelled_s=0.001, mem_delta=0, step=3, unit="step")
    chk("I: after the crossing jobs end, the first job's GPU is under the line again", r1c.get("oom") is False, str(r1c.get("oom")))
    c1.bye(); c1.close()
    ev = read_occ(occ)
    ov = [e for e in ev if e["kind"] == "failure" and e.get("label") == "oversubscribed"]
    chk("I: exactly one oversubscribed failure event, for the oom-aware arrival, naming the other holder", len(ov) == 1 and ov[0].get("mode") == "oom" and ov[0].get("mine_bytes") == 30 * GiB and ov[0].get("others") == [{"pid": os.getpid(), "bytes": 60 * GiB}] and ov[0].get("used_bytes") == 90 * GiB, str([(e.get("job"), e.get("mine_bytes"), e.get("others")) for e in ov]))
    # J. the weights hold: memory from engagement, stepped up at the first unit; the same crossing rule
    h1 = DaemonClient(str(sock)); h1.hello(argv=["kv.py"])
    w = h1.hold(gpu=1, bytes=6 * GiB, oom_aware=True)
    chk("J: holding 6 GiB of weights on GPU 1 at engagement: used 6 GiB, no oom", w.get("used_bytes") == 6 * GiB and w.get("oom") is False and w.get("oom_detail") is None, str(w))
    e1 = h1.enqueue(gpu=1, op="replay", modelled_s=0.001, mem_delta=4 * GiB, step=1, unit="example", oom_aware=True)
    chk("J: the first example adds the rest of the peak (4 GiB): used 10 GiB", e1.get("used_bytes") == 10 * GiB and e1.get("oom") is False, str({k: e1.get(k) for k in ("used_bytes", "oom")}))
    h2 = DaemonClient(str(sock)); h2.hello(argv=["kv2.py"])
    w2 = h2.hold(gpu=1, bytes=75 * GiB, oom_aware=True)
    chk("J: a 75 GiB hold beside 10 GiB crosses the card at load: oom with the detail naming the holder", w2.get("oom") is True and (w2.get("oom_detail") or {}).get("mine_bytes") == 75 * GiB and (w2.get("oom_detail") or {}).get("others") == [{"pid": os.getpid(), "bytes": 10 * GiB}], str(w2.get("oom_detail")))
    h2.bye(); h2.close(); h1.bye(); h1.close()
    ev = read_occ(occ)
    holds = [(e.get("gpu"), e.get("bytes"), e.get("used_bytes")) for e in ev if e["kind"] == "hold"]
    ov2 = [e for e in ev if e["kind"] == "failure" and e.get("label") == "oversubscribed" and e.get("gpu") == 1]
    chk("J: two hold events on GPU 1 (6 GiB, then 75 GiB) and one oversubscribed failure for the second, step None", holds == [(1, 6 * GiB, 6 * GiB), (1, 75 * GiB, 85 * GiB)] and len(ov2) == 1 and ov2[0].get("step") is None and ov2[0].get("mine_bytes") == 75 * GiB, str(holds) + " " + str([(e.get("mine_bytes"), e.get("step")) for e in ov2]))
print(f"{n - fails} passed, {fails} failed")
sys.exit(1 if fails else 0)
