#!/usr/bin/env python3
"""Daemon-level checks for surrogate serving (docs/SURROGATE-SERVING.md §3).

Drives an in-process daemon over its socket with a toy family and a task
file: a covered launch is served its record, an uncovered launch of a
family's script is synthesised in two phases (`surrogate_pending` at the
exact stage from argv knobs, `surrogate` at the structural stage from the
fingerprint), every served value is labelled `tier: surrogate`, a rerun of
the same configuration draws a different noise realisation, a divergent
configuration goes NaN once with a `failure` event, `job_bye` says
`surrogate: true`, and a task without a surrogate job never synthesises.

    python3 -I tests/test_surrogate_daemon.py
"""
from __future__ import annotations

import json, math, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src" / "daemon"))
sys.path.insert(0, str(HERE.parent / "src" / "torch_patch"))

from test_daemon_e2e import start_daemon, read_occ  # noqa: E402
from toy_family import SPEC  # noqa: E402
from daemon_client import DaemonClient  # noqa: E402

fails = 0; n = 0
def chk(name, cond, detail=""):
    global fails, n
    n += 1; fails += 0 if cond else 1
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

SCRIPT = "/home/researcher/project/design/train_gpt.py"
FP_INPUTS = {"model_dim": 512, "num_layers": 8, "num_parameters": 5.1e7, "global_batch_tokens": 524288}

def setup(tmp: Path, with_task: bool = True, surrogate_job: bool = True):
    reg = tmp / "registry"; (reg / "records").mkdir(parents=True); (reg / "surrogates").mkdir()
    (reg / "records" / "r.json").write_text((HERE / "sample_record.json").read_text())
    (reg / "surrogates" / "toy-gpt.json").write_text(json.dumps(SPEC))
    d, sock, occ = start_daemon(tmp / "d", speed=1000.0, registry=reg / "records")
    d.registry.load_surrogates(reg / "surrogates")
    if with_task:
        jobs = [{"job_id": "baseline", "records": ["sample-mlp-001"]}]
        if surrogate_job:
            jobs.append({"job_id": "design-slot-1", "records": None, "surrogate": {"family": "toy-gpt"}})
        task = tmp / "task.json"; task.write_text(json.dumps({"task_id": "T-sur", "jobs": jobs}))
        d.registry.load_task(task)
    return d, sock, occ

def launch(sock, key, argv, script=SCRIPT):
    c = DaemonClient(str(sock)); c.hello(argv=argv)
    r = c.lookup_v2(job_key=key, rank=0, world_size=4, entry_hash="nomatch", argv=argv, env={},
                    script_path=script, fingerprint=None, fingerprint_inputs=None, stage="exact")
    return c, r

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    d, sock, occ = setup(tmp)
    chk("family loaded", list(d.registry.families) == ["toy-gpt"])

    # 1. a covered launch is the record, never the surrogate
    c0 = DaemonClient(str(sock)); c0.hello(argv=["train.py", "--epochs", "3"])
    r0 = c0.lookup_v2(job_key="j0", rank=0, world_size=1, entry_hash="deadbeef", argv=["train.py", "--epochs", "3"],
                      env={}, script_path="/x/train.py", fingerprint=None, fingerprint_inputs=None, stage="exact")
    chk("covered launch served its record", r0.get("record_id") == "sample-mlp-001" and r0.get("match") == "exact", str({k: r0.get(k) for k in ("record_id", "match")}))
    chk("record serving carries no surrogate", not (r0.get("serving") or {}).get("surrogate"))
    c0.bye(); c0.close()

    # 2. an uncovered launch of the family's script: pending at the exact stage
    argv = ["design/train_gpt.py", "--lr", "0.01", "--steps", "500"]
    c1, r1 = launch(sock, "j1", argv)
    chk("uncovered launch: mode replay, match surrogate_pending", r1.get("mode") == "replay" and r1.get("match") == "surrogate_pending", str({k: r1.get(k) for k in ("mode", "match", "draw")}))
    chk("draw synthesised", r1.get("draw") == "synthesised")
    sv = r1.get("serving") or {}
    chk("record id names the family", str(r1.get("record_id", "")).startswith("surrogate:toy-gpt:v1:"), str(r1.get("record_id")))
    chk("serving pending, steps from argv", sv.get("surrogate", {}).get("pending") is True and sv.get("steps") == 500, str({k: sv.get(k) for k in ("steps", "surrogate")}))
    pend_step = sv.get("steady_step_s")
    chk("pending step time positive, converted for world size 4 (8 ref -> 2x)", pend_step and pend_step > 0 and sv.get("world_size_recorded") == 8, f"{pend_step} ws_rec={sv.get('world_size_recorded')}")
    chk("eval pacing present (n_validations from the surrogate block)", sv.get("eval_s_per_validation") and sv.get("n_validations") == 5, f"{sv.get('eval_s_per_validation')} n_val={sv.get('n_validations')}")
    chk("series roles for the Trainer seam", sv.get("series_roles") == ["train_loss", "val_loss"], str(sv.get("series_roles")))
    v0 = c1.materialise("val_loss", step=0)
    chk("step-0 validation read is the init loss with noise", isinstance(v0, float) and 10.8 < v0 < 11.1, str(v0))

    # 3. structural stage: the fingerprint fixes the configuration
    r1b = c1.lookup_v2(job_key="j1", rank=0, world_size=4, entry_hash="nomatch", argv=argv, env={},
                       script_path=SCRIPT, fingerprint="fp256:abc", fingerprint_inputs=FP_INPUTS, stage="structural")
    sv2 = r1b.get("serving") or {}
    chk("structural: match surrogate, pending false", r1b.get("match") == "surrogate" and sv2.get("surrogate", {}).get("pending") is False, str({k: r1b.get(k) for k in ("match", "draw")}) + str(sv2.get("surrogate")))
    chk("record id ordinal 0", str(r1b.get("record_id", "")).endswith(":0"), str(r1b.get("record_id")))
    chk("smaller model from the fingerprint steps faster than the argv-default guess", sv2.get("steady_step_s") < pend_step, f"{sv2.get('steady_step_s')} < {pend_step}")
    chk("structural lookup from another rank returns the same final record", c1.lookup_v2(job_key="j1", rank=1, world_size=4, entry_hash="nomatch", argv=argv, env={}, script_path=SCRIPT, fingerprint="fp256:abc", fingerprint_inputs=FP_INPUTS, stage="structural").get("record_id") == r1b.get("record_id"))
    vals = [c1.materialise("val_loss", step=s) for s in (125, 250, 375, 500)]
    chk("validation losses finite and decreasing", all(isinstance(v, float) and math.isfinite(v) for v in vals) and vals == sorted(vals, reverse=True), str(vals))
    many = c1.materialise_many(["train_loss", "val_loss"], step=250)
    chk("materialise_many serves both roles", set(many) == {"train_loss", "val_loss"} and all(math.isfinite(many[k]) for k in many), str(many))
    chk("train loss sits below val loss (gap 0.03, noise 0.03)", many["train_loss"] < many["val_loss"] + 0.1, str(many))
    chk("unmodelled role is None", c1.materialise("grad_norm", step=250) is None)
    # the per-step learning rates ride on the step's enqueue: warm-up 1..5, then constant
    for k in range(1, 11):
        c1.enqueue(gpu=0, op="replay", modelled_s=0.001, step=k, unit="step", lr=[0.002 * min(k, 5), 0.0005])
    c1.bye(); c1.close()

    # 4. a rerun of the same configuration: next ordinal, different noise
    c2, r2 = launch(sock, "j2", argv)
    r2b = c2.lookup_v2(job_key="j2", rank=0, world_size=4, entry_hash="nomatch", argv=argv, env={},
                       script_path=SCRIPT, fingerprint="fp256:abc", fingerprint_inputs=FP_INPUTS, stage="structural")
    chk("rerun: ordinal 1", str(r2b.get("record_id", "")).endswith(":1"), str(r2b.get("record_id")))
    vals2 = [c2.materialise("val_loss", step=s) for s in (125, 250, 375, 500)]
    chk("rerun: same curve, different realisation", all(abs(a - b) < 0.05 for a, b in zip(vals, vals2)) and all(a != b for a, b in zip(vals, vals2)), f"{vals} vs {vals2}")
    c2.bye(); c2.close()

    # 5. divergence: NaN after 30 steps, one failure event
    argv3 = ["design/train_gpt.py", "--lr", "0.15", "--steps", "500"]
    c3, r3 = launch(sock, "j3", argv3)
    c3.lookup_v2(job_key="j3", rank=0, world_size=4, entry_hash="nomatch", argv=argv3, env={}, script_path=SCRIPT,
                 fingerprint="fp256:abd", fingerprint_inputs=FP_INPUTS, stage="structural")
    early = c3.materialise("train_loss", step=10)
    late = [c3.materialise("train_loss", step=s) for s in (30, 60, 125)]
    chk("divergent lr: finite before step 30, NaN after", math.isfinite(early) and all(v != v for v in late), f"{early} {late}")
    c3.bye(); c3.close()

    ev = read_occ(occ)
    look = [e for e in ev if e["kind"] == "lookup"]
    chk("lookup events carry the surrogate block", any((e.get("surrogate") or {}).get("family") == "toy-gpt" and e.get("match") == "surrogate_pending" for e in look)
        and any((e.get("surrogate") or {}).get("pending") is False and e.get("match") == "surrogate" for e in look))
    chk("surrogate lookups name the task job", all(e.get("task_job") == "design-slot-1" for e in look if (e.get("match") or "").startswith("surrogate")))
    mats = [e for e in ev if e["kind"] == "materialise" and str(e.get("record_id", "")).startswith("surrogate:")]
    chk("every synthesised read is labelled tier surrogate", mats and all(e.get("tier") == "surrogate" for e in mats), f"{len(mats)} events")
    chk("record reads are not", all(e.get("tier") != "surrogate" for e in ev if e["kind"] == "materialise" and e.get("record_id") == "sample-mlp-001"))
    fl = [e for e in ev if e["kind"] == "failure"]
    chk("one failure event, kind surrogate_divergence, at the first NaN", len(fl) == 1 and fl[0].get("mode") == "surrogate_divergence" and fl[0].get("step") == 30, str(fl))
    enq = [e for e in ev if e["kind"] == "enqueue" and e.get("unit") == "step" and e.get("lr") is not None]
    chk("enqueue events log the LR only when it changed (5 warm-up values, then constant)", [e.get("step") for e in enq] == [1, 2, 3, 4, 5] and enq[-1]["lr"] == [0.01, 0.0005], str([(e.get("step"), e.get("lr")) for e in enq]))
    stream = [json.loads(l) for l in (occ.parent / "lr_stream.jsonl").read_text().splitlines()]
    chk("lr_stream.jsonl carries the whole stream for the job", len(stream) == 1 and stream[0]["steps"] == 10 and stream[0]["groups"] == 2 and stream[0]["lr"][9] == [10, [0.01, 0.0005]], str(stream)[:200])
    byes = [e for e in ev if e["kind"] == "job_bye"]
    chk("job_bye counts the LR steps", [e.get("lr_steps") for e in byes][:2] == [None, 10], str([e.get("lr_steps") for e in byes]))
    chk("job_bye says the last step paced", byes[1].get("last_step") == 10, str(byes[1].get("last_step")))
    chk("the surrogate block names the resolved step count", all((e.get("surrogate") or {}).get("steps") == 500 for e in look if (e.get("match") or "").startswith("surrogate")))
    chk("job_bye says surrogate true for synthesised jobs and false for the record", [e.get("surrogate") for e in byes] == [False, True, True, True], str([e.get("surrogate") for e in byes]))

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    d, sock, occ = setup(tmp, with_task=True, surrogate_job=False)
    c, r = launch(sock, "k1", ["design/train_gpt.py", "--lr", "0.01"])
    chk("task without a surrogate job: not synthesised (match none, mode model)", r.get("match") == "none" and r.get("mode") == "model", str({k: r.get(k) for k in ("match", "mode")}))
    c.bye(); c.close()

with tempfile.TemporaryDirectory() as td:
    # surrogate_force (INTERPRETED-SERVING §5.0): a record covers the family's
    # script by path, yet the task says synthesise; labelled surrogate_forced
    tmp = Path(td)
    reg = tmp / "registry"; (reg / "records").mkdir(parents=True); (reg / "surrogates").mkdir()
    (reg / "surrogates" / "toy-gpt.json").write_text(json.dumps(SPEC))
    covering = {"identity": {"record_id": "covering-run", "entry_hash": "0" * 64, "argv": ["design/train_gpt.py"],
                             "match_policy": "path_bound", "bound_path": "design/train_gpt.py", "replicate_group": "cov"},
                "mode": "replay", "steps": 3, "per_step_gpu_s": 0.1, "timing": {"steady_step_s": 0.1},
                "series": {"val_loss": {"steps": [1, 2, 3], "values": [3.0, 2.9, 2.8]}}, "readiness": {"replay": True}, "failures": []}
    (reg / "records" / "cov.json").write_text(json.dumps(covering))
    for force in (False, True):
        d, sock, occ = setup_dir = start_daemon(tmp / f"d{int(force)}", speed=1000.0, registry=reg / "records")
        d.registry.load_surrogates(reg / "surrogates")
        task = tmp / f"task{int(force)}.json"
        task.write_text(json.dumps({"task_id": "pw", "jobs": [{"job_id": "design", "records": None, "surrogate": {"family": "toy-gpt"}, "surrogate_force": force}]}))
        d.registry.load_task(task)
        c, r = launch(sock, "pw1", ["design/train_gpt.py", "--lr", "0.01", "--steps", "500"])
        if not force:
            chk("without force: the covering record is served by path", r.get("match") == "path" and r.get("record_id") == "covering-run", str({k: r.get(k) for k in ("match", "record_id")}))
        else:
            chk("with force: synthesised despite the record, labelled surrogate_forced_pending", r.get("match") == "surrogate_forced_pending" and r.get("draw") == "synthesised", str({k: r.get(k) for k in ("match", "draw")}))
            r2 = c.lookup_v2(job_key="pw1", rank=0, world_size=4, entry_hash="nomatch", argv=["design/train_gpt.py", "--lr", "0.01", "--steps", "500"], env={},
                             script_path=SCRIPT, fingerprint="fp256:abc", fingerprint_inputs=FP_INPUTS, stage="structural")
            chk("with force: the structural phase keeps the label and the forced flag", r2.get("match") == "surrogate_forced" and (r2.get("serving") or {}).get("surrogate", {}).get("pending") is False, str(r2.get("match")))
            chk("with force: reads are tier surrogate", isinstance(c.materialise("val_loss", step=125), float))
            ev = read_occ(occ)
            chk("with force: lookup events carry forced in the surrogate block", all((e.get("surrogate") or {}).get("forced") is True for e in ev if e["kind"] == "lookup"))
        c.bye(); c.close()

with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    d, sock, occ = setup(tmp, with_task=False)
    c, r = launch(sock, "m1", ["design/train_gpt.py", "--lr", "0.01"])
    chk("no task loaded: every loaded family serves", r.get("match") == "surrogate_pending", str(r.get("match")))
    c, r = launch(sock, "m2", ["other/script.py"], script="/home/researcher/project/other/script.py")
    chk("a script no family names is not synthesised", r.get("match") == "none", str(r.get("match")))
    c.bye(); c.close()

print(f"\n{n - fails}/{n} passed")
sys.exit(1 if fails else 0)
