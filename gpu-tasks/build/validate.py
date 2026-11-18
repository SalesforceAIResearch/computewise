"""Check the registry against RECORD-FORMAT.md. Reports; never edits.

usage: python3 -I validate.py <registry dir>

Exit status 1 when a record or task breaks the contract. Readiness (whether gpu-mock can
replay a record today) is reported, not enforced: an unready record is valid.
"""
import collections
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

REQ_TOP = ("schema", "identity", "mode", "source", "reference", "launch", "steps", "per_step_gpu_s", "timing", "memory",
           "series", "materialisation", "result", "outcome", "failures", "agent_view", "readiness", "disclosure")
REQ_ID = ("record_id", "kind", "entry_hash", "argv", "seed", "replicate_group", "match_policy", "trajectory")
REQ_TIMING = ("pacing", "wall_total_s")
STATUSES = ("completed", "crashed", "failed", "killed", "truncated", "timeout")
POLICIES = ("exact_only", "structural_verified", "path_bound")
PACING = ("per_step", "per_logged_interval", "total")


def walk_tv(obj, path, errs):
    if isinstance(obj, dict):
        if "provenance" in obj and ("value" in obj or "steps" in obj):
            if obj["provenance"] not in C.PROVENANCE:
                errs.append(f"{path}: provenance {obj['provenance']!r}")
            if obj["provenance"] == "absent" and obj.get("value") is not None:
                errs.append(f"{path}: absent but has a value")
            if obj["provenance"] != "absent" and "value" in obj and obj["value"] is None and not obj.get("null_means"):
                errs.append(f"{path}: {obj['provenance']} but the value is null, with no null_means")
        for k, v in obj.items():
            walk_tv(v, f"{path}.{k}", errs)
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:3]):
            walk_tv(v, f"{path}[{i}]", errs)
    elif isinstance(obj, float) and not math.isfinite(obj):
        errs.append(f"{path}: non-finite number")


def check_series(name, s, errs):
    st, vs = s.get("steps"), s.get("values")
    if not isinstance(st, list) or not isinstance(vs, list) or len(st) != len(vs) or not st:
        errs.append(f"series.{name}: steps and values must be equal-length non-empty lists")
        return
    if any(b <= a for a, b in zip(st, st[1:])):
        errs.append(f"series.{name}: steps not strictly increasing")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vs):
        errs.append(f"series.{name}: non-finite value")


def check_rerun_siblings(t, recs):
    """gpu-mock's rerun design (b): each job's rerun_siblings are records of the job's own kind and steps that overlap no record
    any job of the task lists and no other sibling of the task (build/siblings.py)."""
    errs, seen = [], []
    units = [C.served_unit(recs[x]) for j in t["jobs"] for x in (j.get("records") or []) if x in recs]
    for j in t["jobs"]:
        sib = j.get("rerun_siblings")
        if sib is None:
            continue
        prim = recs.get((j.get("records") or [None])[0])
        if prim is None or j.get("draw") != "exact_seed":
            errs.append(f"{j['job_id']}: rerun_siblings on a job that is not an exact-seed draw of a known record")
            continue
        pd = prim["identity"].get("derived_from") or {}
        if pd.get("kind") == "segment":
            errs.append(f"{j['job_id']}: rerun_siblings on a segment (a resumed segment follows its checkpoint's run)")
        for x in sib:
            r = recs.get(x)
            if r is None:
                errs.append(f"{j['job_id']}: rerun sibling {x} not in the registry")
                continue
            rd = r["identity"].get("derived_from") or {}
            if rd.get("kind") != pd.get("kind") or r.get("steps") != prim.get("steps"):
                errs.append(f"{j['job_id']}: rerun sibling {x} is not of its primary's kind and steps")
            u = C.served_unit(r)
            if any(C.overlaps(u, v) for v in units):
                errs.append(f"{j['job_id']}: rerun sibling {x} overlaps a record a job of the task lists")
            if any(C.overlaps(u, v) for v in seen):
                errs.append(f"{j['job_id']}: rerun sibling {x} overlaps another rerun sibling of the task")
            seen.append(u)
    return errs


def check_derived(r):
    """A derived prefix record (build/gen_prefix.py): its script is its parent's with exactly the documented stop change, it ends
    at its stop step, and it is exact_only in a replicate group apart from its parent's."""
    d = r["identity"].get("derived_from")
    if not d:
        return []
    errs = []
    if d.get("kind") == "g4n_probe":
        # G4n's SOAP probe: the script's own STOP_STEP default set to 125, which
        # gen_prefix's stop_after refuses (its loop breaks on `step >= STOP_STEP`); build/gen_g4n.py checks it
        import gen_g4n  # noqa: E402  (same directory)
        return gen_g4n.check_record(r, REG)
    if d.get("kind") == "olp_stop_step":
        # pkg-obs-latepole's soap runs (lane OBS, 2026-11-02): stopped by the script's own STOP_STEP default, which
        # gen_prefix refuses; build/gen_obs_latepole.py checks them
        import gen_obs_latepole  # noqa: E402  (same directory)
        return gen_obs_latepole.check_record(r, REG)
    if d.get("kind") == "k_probe":
        # angle K's reference probes: a rejected candidate's first run stopped at the line that
        # rejects it, single-process; build/gen_unneeded_k.py checks them
        import gen_unneeded_k  # noqa: E402  (same directory)
        return gen_unneeded_k.check_record(r, REG)
    if d.get("kind") == "synthesised_filter_variant":
        # H4's records: a prefix record served as one recipe's run; build/gen_h4.py checks them
        import gen_h4  # noqa: E402  (same directory)
        return gen_h4.check_record(r, REG)
    if d.get("kind") == "oom_relaunch":
        # angle C's records: a prefix record served single-process with a --micro-batch
        # flag and a synthesised OOM; build/gen_failures.py checks them
        import gen_failures  # noqa: E402  (same directory)
        return gen_failures.check_record(r, REG)
    if d.get("kind") == "synthesised_yield_variant":
        # pkg-filter-yield's records (independent Sol, 2026-11-02): H4's recipe scripts under yieldlab/, a contra-soft-muon
        # prefix assigned to each recipe and seed; build/gen_filter_yield.py checks them
        import gen_filter_yield  # noqa: E402  (same directory)
        return gen_filter_yield.check_record(r, REG)
    if d.get("kind") == "silent_zero_lr":
        # pkg-sf-zero-lr's records (SF lane, 2026-11-02): a prefix made single-process with a --lr-scale flag, clean at 1 and
        # modelled flat at 0; build/gen_sf_zero_lr.py checks them
        import gen_sf_zero_lr  # noqa: E402  (same directory)
        return gen_sf_zero_lr.check_record(r, REG)
    if d.get("kind") == "checkpoint_output":
        # pkg-sf-checkpoint-collision's records (SF lane, 2026-11-02): a clean prefix whose script saves out/ckpt.pt by default,
        # with an --out-dir flag; build/gen_sf_checkpoint_collision.py checks them
        import gen_sf_checkpoint_collision  # noqa: E402  (same directory)
        return gen_sf_checkpoint_collision.check_record(r, REG)
    if d.get("kind") == "segment" and d.get("variant") == "save_dir_not_created":
        # D8's chain: a segment on the single-process segment script with the save's makedirs
        # removed, so its first save fails for real; build/gen_chain_save.py checks them
        import gen_chain_save  # noqa: E402  (same directory)
        return gen_chain_save.check_record(r, REG)
    if d.get("kind") not in ("prefix", "segment", "prepared", "data_checked"):
        return [f"derived_from.kind {d.get('kind')!r}"]
    if d["kind"] == "data_checked":
        import gen_dataprep  # noqa: E402  (same directory)
        via = os.path.join(REG, "records", d.get("via", "") + ".json")
        if not os.path.exists(via):
            return [f"derived_from.via {d.get('via')} not in the registry"]
        v = json.load(open(via))
        try:
            want = gen_dataprep.data_check(open(os.path.join(REG, v["agent_view"]["script"])).read(), gen_dataprep.data_files(v))
        except ValueError as e:
            return [f"the prefix script does not take the data check: {e}"]
        if C.sha256_text(want) != r["identity"]["entry_hash"]:
            errs.append("the data-checked script is not its prefix's with only the documented data check")
        if d.get("files") != list(gen_dataprep.data_files(v)):
            errs.append("derived_from.files is not the files its prefix's run reads")
        if r["series"] != v["series"] or r["steps"] != v["steps"] or r["timing"] != v["timing"]:
            errs.append("a data-checked record must serve exactly its prefix record's run")
        if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == v["identity"]["replicate_group"]:
            errs.append("a data-checked record must be exact_only, in a group apart from its prefix's")
        return errs
    if d["kind"] == "prepared":
        via = os.path.join(REG, "records", d.get("via", "") + ".json")
        if not os.path.exists(via):
            return [f"derived_from.via {d.get('via')} not in the registry"]
        v = json.load(open(via))
        want_hash = v["identity"]["entry_hash"]
        if (r["identity"].get("process_model") or {}).get("value") == "single":
            import single_process  # noqa: E402  (build/gen_h1.py: the reference is the prefix script made single-process)
            want_hash = C.sha256_text(single_process.single_process(open(os.path.join(REG, v["agent_view"]["script"])).read()))
        if r["identity"]["entry_hash"] != want_hash:
            errs.append("a prepared record's entry_hash must be its reference's (the prefix script"
                        + (", single-process)" if want_hash != v["identity"]["entry_hash"] else ")"))
        if r["series"] != v["series"] or r["steps"] != v["steps"] or r["timing"] != v["timing"]:
            errs.append("a prepared record must serve exactly its prefix record's run")
        if r["identity"]["match_policy"] != "path_bound" or not r["identity"].get("bound_path"):
            errs.append("a prepared record must be path_bound with a bound_path")
        return errs
    parent = os.path.join(REG, "records", d["record"] + ".json")
    if not os.path.exists(parent):
        return [f"derived_from.record {d['record']} not in the registry"]
    p = json.load(open(parent))
    pcode = open(os.path.join(REG, p["agent_view"]["script"])).read()
    if d["kind"] == "segment":
        import gen_segment  # noqa: E402  (same directory)
        a, b = d["steps"][0] - 1, d["steps"][1]
        try:
            want = gen_segment.segment_script(pcode)
            if (r["identity"].get("process_model") or {}).get("value") == "single":
                import single_process  # noqa: E402  (same directory; build/gen_h1.py's second documented change)
                want = single_process.single_process(want)
        except ValueError as e:
            return [f"the parent script does not take the segment change: {e}"]
        if C.sha256_text(want) != r["identity"]["entry_hash"]:
            errs.append("the segment script is not its parent's with only the documented save-and-resume change"
                        + (" and single_process" if (r["identity"].get("process_model") or {}).get("value") == "single" else ""))
        if r["steps"] != b - a or any(s["steps"][-1] > b - a for s in r["series"].values()):
            errs.append(f"steps or a series runs past the segment's {b - a} steps")
        if r["identity"]["argv"] != gen_segment.segment_argv(p["identity"]["argv"][0], p["identity"]["seed"]["value"], a, b, p["steps"]):
            errs.append("argv is not the segment's documented launch")
        if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
            errs.append("a segment record must be exact_only, in a group apart from its parent's")
        return errs
    import gen_prefix  # noqa: E402  (same directory)
    k = d.get("stop_after_step")
    try:
        want = gen_prefix.stop_after(pcode, k)
    except ValueError as e:
        return [f"the parent script does not take the stop change: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the prefix script is not its parent's with only the documented stop change")
    if r["steps"] != k or any(s["steps"][-1] > k for s in r["series"].values()):
        errs.append(f"steps or a series runs past the stop step {k}")
    if k not in p["series"]["val_loss"]["steps"]:
        errs.append(f"stop step {k} is not one of the parent's validation steps")
    if r["identity"]["match_policy"] != "exact_only":
        errs.append("a prefix record must be exact_only")
    if r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
        errs.append("a prefix record shares its parent's replicate group")
    return errs


def check_record(r):
    errs = []
    for k in REQ_TOP:
        if k not in r:
            errs.append(f"missing {k}")
    if errs:
        return errs
    if r["schema"] != C.SCHEMA_RECORD:
        errs.append(f"schema {r['schema']}")
    for k in REQ_ID:
        if k not in r["identity"]:
            errs.append(f"identity.{k} missing")
    if r["identity"].get("match_policy") not in POLICIES:
        errs.append("identity.match_policy")
    if r["identity"].get("match_policy") == "structural_verified":
        tr = (r["identity"]["trajectory"].get("lr_schedule") or {}).get("trace")
        if not tr or tr.get("alignment") not in ("after_step", "at_step"):
            errs.append("structural_verified needs trajectory.lr_schedule.trace with an alignment")
        elif len(tr["steps"]) != len(tr["values"]):
            errs.append("lr trace steps and values differ in length")
    fp, fi = r["identity"].get("fingerprint"), r["identity"].get("fingerprint_inputs")
    if fp:
        want, ver = C.fingerprint(fi or {})
        stored = r["identity"].get("fingerprint_version")
        if stored != ver:
            STALE_FP[(stored, ver)] += 1
        elif fp != want:
            errs.append("identity.fingerprint does not match gpu-mock's implementation over fingerprint_inputs")
    if r["mode"] not in C.MODES:
        errs.append("mode")
    for k in REQ_TIMING:
        if k not in r["timing"]:
            errs.append(f"timing.{k} missing")
    if r["timing"].get("pacing") not in PACING:
        errs.append("timing.pacing")
    if r["outcome"].get("status") not in STATUSES:
        errs.append(f"outcome.status {r['outcome'].get('status')}")
    if r["outcome"]["status"] in ("crashed", "failed", "killed") and not r["failures"]:
        errs.append("a failed outcome needs a failures[] entry")
    for f in r["failures"]:
        if f.get("mode") not in C.FAILURE_MODES:
            errs.append(f"failures[].mode {f.get('mode')}")
        rp = f.get("replay") or {}
        if rp.get("value") not in ("fire", "suppress"):
            errs.append("failures[].replay must be fire or suppress")
        if rp.get("value") == "suppress" and rp.get("continuation") not in ("complete", "sibling_required"):
            errs.append("a suppressed failure needs continuation complete or sibling_required")
        if rp.get("continuation") == "sibling_required" and not any("suppressed collision" in b for b in r["materialisation"].get("blockers", [])):
            errs.append("a sibling_required failure needs a matching blocker so readiness is false")
    for name, s in r["series"].items():
        check_series(name, s, errs)
    pol = r["launch"].get("batch_policy", {})
    if pol.get("value") is not None and pol["value"] not in C.BATCH_POLICIES:
        errs.append("launch.batch_policy")
    walk_tv(r, "", errs)
    if r["readiness"] != C.readiness(r):
        errs.append("readiness is stale: regenerate")
    errs.extend(check_derived(r))
    av = r["agent_view"].get("script")
    if av and not os.path.exists(os.path.join(REG, av)):
        errs.append(f"agent_view.script {av} not on disk")
    if av and r["identity"]["entry_hash"]:
        with open(os.path.join(REG, av), errors="ignore") as fh:
            if C.sha256_text(fh.read()) != r["identity"]["entry_hash"]:
                errs.append("entry_hash does not match the script file")
    return errs


def _get(d, dotted):
    for k in dotted.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


# A brief may not state a deadline or a time limit. The pattern catches the retired sentence
# ("I need the results within 40 minutes.") and its plain variants; a word like "quick" in a workload's name is not one.
DEADLINE_WORDS = re.compile(r"\bwithin\s+(\d+|an?|one|two|three|four|five|six|ten|twelve|half)\b[^.]*\b(hour|minute|second|day)s?\b"
                            r"|\bdeadline\b|\btime limit\b|\bby (the )?end of\b|\bas (fast|quickly|soon) as\b|\bno more than \d+\s*(h|min|hour|minute)",
                            re.I)


def check_task(t, recs, errs):
    jobs = {j["job_id"]: j for j in t["jobs"]}
    errs.extend(check_rerun_siblings(t, recs))
    lb = (t.get("stats") or {}).get("lower_bound_makespan_s")
    if "deadline_s" in t or "deadline_floor" in t or "deadline_how" in t:
        errs.append("deadline_s is retired: the field is horizon_s, hidden from the agent, and no brief states a deadline "
                    "(the study, 2026-10-20, design/IDEA.md, \"No deadline in the brief\")")
    for vn, vv in (t.get("brief_variants") or {}).items():
        if DEADLINE_WORDS.search(vv.get("text") or ""):
            errs.append(f"brief variant {vn} states a deadline or time limit, which no brief may (the study, 2026-10-20)")
        if not vv.get("how"):
            errs.append(f"brief variant {vn} without how")
    if DEADLINE_WORDS.search(t["brief"]["text"]):
        errs.append("the brief states a deadline or time limit, which no brief may (the study, 2026-10-20)")
    wi = (t.get("capture") or {}).get("weights_index") if isinstance(t.get("capture"), dict) else None
    if "capture" in t:
        # run_one's weights-index capture: the task names a JSON index of
        # weight files in the agent's home; run_one digests the files it lists. Types and ranges as isolation specified
        if not isinstance(t["capture"], dict) or set(t["capture"]) - {"weights_index"}:
            errs.append("capture: a dict whose only key is weights_index")
        elif not isinstance(wi, dict) or not isinstance(wi.get("path"), str) or not wi["path"]:
            errs.append("capture.weights_index needs a path")
        else:
            pth = wi["path"]
            if pth.startswith("/") or ".." in pth.split("/"):
                errs.append("capture.weights_index.path must be relative to the agent's home, with no '..'")
            for k, lo, hi in (("max_files", 1, 64), ("max_bytes_per_file", 1, None), ("max_seconds", 1, 1800)):
                v = wi.get(k)
                if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < lo or (hi is not None and v > hi)):
                    errs.append(f"capture.weights_index.{k} must be an int in [{lo}, {hi if hi else 'inf'}]")
            if set(wi) - {"path", "max_files", "max_bytes_per_file", "max_seconds"}:
                errs.append("capture.weights_index has keys isolation's run_one does not read")
    if "horizon_s" in t:
        serial = (t.get("stats") or {}).get("serial_makespan_s")
        if not lb or t["horizon_s"] < lb:
            errs.append("horizon_s is below the lower-bound makespan: no schedule can finish inside it")
        wide = (t.get("horizon_terms") or {}).get("serial_makespan_s_widest")
        capped = t.get("goal_level") and "override" in t.get("horizon_how", "") and wide and t["horizon_s"] >= wide
        if serial and t["horizon_s"] < serial and not capped:
            errs.append("horizon_s is below the serial makespan: an agent running jobs one at a time cannot finish inside it "
                        "(a goal-level task may cap it at its widest-width serial, with override in horizon_how)")
        if not t.get("horizon_how"):
            errs.append("horizon_s without horizon_how")
        wr = (t.get("work_remaining") or {}).get("rule")
        if wr != "jobs_not_completed" and not (wr == "open" and t.get("goal_level")):
            errs.append("work_remaining.rule missing or unknown (open only on a goal-level task)")
        ht = t.get("horizon_terms")
        if not isinstance(ht, dict) or "startup_total_s" not in ht or "serial_makespan_s" not in ht:
            errs.append("horizon_terms missing (serial_makespan_s and startup_total_s)")
        else:
            if serial is not None and abs(ht["serial_makespan_s"] - serial) > 1e-6:
                errs.append("horizon_terms.serial_makespan_s is not stats.serial_makespan_s")
            by = ht.get("startup_by_machine_type") or {}
            want_su = max(by.values()) if by else 0.0
            if abs((ht["startup_total_s"] or 0.0) - want_su) > 1e-6:
                errs.append("horizon_terms.startup_total_s is not the largest startup_by_machine_type")
            prep = (sum(float((j.get("prep") or {}).get("horizon_s") or 0) for j in t["jobs"])
                    + float((t.get("data_prep") or {}).get("horizon_s") or 0))
            if abs(float(ht.get("prep_total_s") or 0.0) - prep) > 1e-6:
                errs.append("horizon_terms.prep_total_s is not the sum of the jobs' prep.horizon_s and data_prep.horizon_s")
            want = C.colleague_round(1.5 * (ht["serial_makespan_s"] + (ht["startup_total_s"] or 0.0) + prep))
            if t["horizon_s"] != want and "override" not in t.get("horizon_how", ""):
                errs.append(f"horizon_s {t['horizon_s']} is not colleague_round(1.5 x (serial makespan + start-ups + preparation)) = {want}")
            if by and not (ht.get("measured") or {}).get("machine_type"):
                errs.append("horizon_terms with a start-up total needs measured.machine_type, caps and image")
            if bool(ht.get("provisional")) != (ht.get("fleet_machine_type") not in by):
                errs.append("horizon_terms.provisional must say whether the fleet's machine type is measured")
        fl = t.get("environment_floor")
        if not isinstance(fl, dict) or "by_machine_type" not in fl:
            errs.append("environment_floor missing (by_machine_type)")
        else:
            by = fl["by_machine_type"]
            for mt, e in by.items():
                ms = e.get("measured") or {}
                if e.get("value") is None or e["value"] < 0:
                    errs.append(f"environment_floor[{mt}] needs a non-negative value")
                if not (ms.get("caps") and (ms.get("image") or {}).get("id") and ms.get("machine_type") == mt):
                    errs.append(f"environment_floor[{mt}].measured needs the machine type, the caps and the image digest F was measured on")
            if fl.get("provenance") != ("derived" if by else "absent"):
                errs.append("environment_floor.provenance must be derived when measured and absent otherwise")
    env = t.get("environment")
    if env:
        cons = (env.get("install") or {}).get("constraints")
        if not cons or not os.path.exists(os.path.join(REG, cons)):
            errs.append(f"environment.install.constraints {cons} not on disk")
        for h in env.get("hub") or []:
            if not re.fullmatch(r"[0-9a-f]{40}", h.get("revision") or ""):
                errs.append(f"environment.hub {h.get('repo')}: revision is not a commit id")
            if not h.get("files") or sum(h["files"].values()) != h.get("bytes"):
                errs.append(f"environment.hub {h.get('repo')}: files missing or bytes not their sum")
        if env.get("hub") and sum(h["bytes"] for h in env["hub"]) != env.get("hub_bytes"):
            errs.append("environment.hub_bytes is not the sum of the entries")
    for f in t.get("files", []):
        g = f.get("git")
        if g and not (g.get("repo") and g.get("keep") and re.fullmatch(r"[0-9a-f]{40}", g.get("commit") or "")):
            errs.append(f"files[] {f['path']}: git entry needs repo, keep and a full commit id")
    if len(jobs) != len(t["jobs"]):
        errs.append("duplicate job_id")
    box = t["box"]["gpus"]
    bound = {jid for d in t.get("decisions", []) for jid in d["launches"]}
    for d in t.get("decisions", []):
        for c in d["candidates"]:
            for rid in c["records"]:
                if rid not in recs:
                    errs.append(f"decision {d['decision_id']}: record {rid} not in the registry")
        for jid in d["launches"]:
            j = jobs.get(jid) or {}
            sel = (j.get("bind") or {}).get("select_by")
            if sel:
                for c in d["candidates"]:
                    hits = [rid for rid in c["records"] if rid in recs and all(
                        _get(recs[rid]["identity"], k) == v for k, v in sel.items())]
                    if len(hits) != 1:
                        errs.append(f"job {jid}: candidate {c['variant']} has {len(hits)} records matching {sel}")
        for jid in d["launches"] + d["after"]:
            if jid not in jobs:
                errs.append(f"decision {d['decision_id']}: unknown job {jid}")
    for j in t["jobs"]:
        if j.get("records") is None and j["job_id"] not in bound:
            errs.append(f"job {j['job_id']}: no records and not bound by a decision")
        for rid in j.get("records") or []:
            if rid not in recs:
                errs.append(f"job {j['job_id']}: record {rid} not in the registry")
        for d in j.get("after", []):
            if d not in jobs:
                errs.append(f"job {j['job_id']}: depends on unknown job {d}")
        if not j.get("gpu_options") or any(g > box for g in j["gpu_options"]):
            errs.append(f"job {j['job_id']}: gpu_options {j.get('gpu_options')} outside the {box}-GPU box")
    ss = t.get("shared_state")
    if ss is None:
        errs.append("shared_state missing")
    elif ss["default_shared"] and not (ss.get("isolation") or {}).get("flag"):
        errs.append("jobs share a path by default but no isolation flag is declared")
    for j in t["jobs"]:
        for rid in j.get("records") or []:
            if rid in recs and any(f.get("replay", {}).get("continuation") == "sibling_required" for f in recs[rid]["failures"]):
                errs.append(f"job {j['job_id']} draws {rid}, which a suppressed collision truncated")
    seen, stack = set(), set()

    def visit(n):
        if n in stack:
            errs.append(f"dependency cycle through {n}")
            return
        if n in seen:
            return
        stack.add(n)
        for d in jobs[n].get("after", []):
            if d in jobs:
                visit(d)
        stack.discard(n)
        seen.add(n)
    for n in jobs:
        visit(n)


STALE_FP = collections.Counter()


def main():
    global REG
    REG = sys.argv[1]
    recs, bad, tables = {}, 0, {}
    rdir = os.path.join(REG, "records")
    for f in sorted(os.listdir(rdir)):
        if not f.endswith(".json"):
            continue
        r = json.load(open(os.path.join(rdir, f)))
        if C.is_non_record(r):
            tables[f] = r                                # an example-order table (gpu-mock r4), checked after the records
            continue
        rid = r.get("identity", {}).get("record_id")
        errs = check_record(r)
        if str(rid).startswith("sol-ttc-"):
            # independent Sol's TabArena cells (2026-11-02): a source record, not a derived one, so its own correspondence
            # check runs here; build/gen_sol_tabular.py reads its pinned inputs from survey/sol-tabular-ttc/
            import gen_sol_tabular  # noqa: E402  (same directory)
            errs += gen_sol_tabular.check_record(r, REG)
        if str(rid).startswith("k2-latepole-"):
            # the study K2's late-pole source records (2026-11-02): DLM and sparse-attention cells, not derived; their own
            # correspondence check, build/gen_k2_latepole.py, reads its inputs from survey/k2-latepole-inputs/
            import gen_k2_latepole  # noqa: E402  (same directory)
            errs += gen_k2_latepole.check_record(r, REG)
        ms = ((r.get("materialisation") or {}).get("serving") or {}).get("module_shas")
        if ms is not None:
            # gpu-mock's helper-dependency bind (2026-11-02): {task-root-relative .py path: sha256}; {} binds no staged file.
            # The shape only: build/module_shas.py --check compares every map with the staged bytes
            if not isinstance(ms, dict):
                errs.append("materialisation.serving.module_shas is not a map")
            else:
                for k, v in ms.items():
                    if (not isinstance(k, str) or k.startswith("/") or ".." in k.split("/") or not k.endswith(".py")
                            or not (isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v))):
                        errs.append(f"module_shas entry {k!r}: a task-root-relative .py path and a sha256 hex digest")
                        break
        if rid in recs:
            errs.append("duplicate record_id")
        if f != f"{rid}.json":
            errs.append(f"file name {f} != record_id")
        recs[rid] = r
        if errs:
            bad += 1
            print(f"RECORD {rid}: " + "; ".join(errs[:6]))
    # example-order tables (build/gen_example_order.py): one per family and workload, keys unique, as long as the records' vectors
    fam_of = {"llm-kv-selection-budgeting": "selection", "llm-kv-adaptive-quantization": "quant"}
    covered = {}
    for f, t in tables.items():
        errs = []
        order, ids, n = t.get("order") or [], t.get("ids") or [], t.get("n_examples_full")
        if not (len(order) == len(ids) == n):
            errs.append(f"order {len(order)}, ids {len(ids)}, n_examples_full {n}")
        if len(set(order)) != len(order):
            errs.append(f"{len(order) - len(set(order))} duplicate keys")
        for fam in t.get("families") or []:
            if (fam, t.get("workload")) in covered:
                errs.append(f"{fam} {t.get('workload')} also in {covered[fam, t.get('workload')]}")
            covered[fam, t.get("workload")] = f
        if errs:
            bad += 1
            print(f"TABLE {f}: " + "; ".join(errs))
    if tables:
        for r in recs.values():
            k = (r["identity"].get("source_key") or {}).get("task")
            serving = (r.get("materialisation") or {}).get("serving") or {}
            if k not in fam_of or "n_examples_full" not in serving:
                continue
            f = covered.get((fam_of[k], (serving.get("decode") or {}).get("workload")))
            if f is None or tables[f]["n_examples_full"] != serving["n_examples_full"]["value"]:
                bad += 1
                print(f"RECORD {r['identity']['record_id']}: no example-order table of its n_examples_full ({f})")
    # measured first-k scores (gpu-mock r6, build/gen_mlskv_first50.py): on the order table's first k positions, tail feasible
    for r in recs.values():
        s = (r.get("materialisation") or {}).get("serving") or {}
        pe = s.get("per_example_scores")
        if pe is None:
            continue
        errs = []
        k, n = pe.get("k"), s["n_examples_full"]["value"]
        f = covered.get((fam_of.get((r["identity"].get("source_key") or {}).get("task")), (s.get("decode") or {}).get("workload")))
        if f is None:
            errs.append("per_example_scores but no example-order table")
        elif not (isinstance(k, int) and 0 < k <= n and len(pe.get("keys") or []) == len(pe.get("example_ids") or [])
                  == len(pe.get("scores") or []) == k):
            errs.append(f"per_example_scores k {k} and its lists disagree")
        else:
            if pe["keys"] != tables[f]["order"][:k] or pe["example_ids"] != tables[f]["ids"][:k]:
                errs.append(f"per_example_scores keys or ids are not {f}'s first {k}")
            lo, hi = s["scale"]
            sc = pe["scores"]
            if not all(isinstance(x, (int, float)) and lo <= x <= hi for x in sc) or (
                    s.get("score_kind") == "binary" and not all(x in (lo, hi) for x in sc)):
                errs.append("per_example_scores: a score off the record's scale or kind")
            else:
                rest = r["result"]["final_score"]["value"] * n - sum(sc)
                if not -1e-6 * n <= rest <= (n - k) * hi + 1e-6 * n or (
                        s.get("score_kind") == "binary" and abs(rest / hi - round(rest / hi)) > 1e-3):   # 6-decimal final_score
                    errs.append(f"per_example_scores: the tail cannot reach final_score (needs {rest:.3f} over {n - k})")
        if pe.get("provenance") != "measured" or pe.get("key") != "sha256(example_id\\0prompt)":
            errs.append("per_example_scores provenance or key")
        if errs:
            bad += 1
            print(f"RECORD {r['identity']['record_id']}: " + "; ".join(errs))
    groups = collections.defaultdict(set)
    for r in recs.values():
        groups[r["identity"]["replicate_group"]].add(r["identity"]["trajectory"]["config_hash"])
    for g, hs in groups.items():
        if len(hs) != 1:
            bad += 1
            print(f"GROUP {g}: {len(hs)} config hashes")
    tdir = os.path.join(REG, "tasks")
    tasks = 0
    if os.path.isdir(tdir):
        for f in sorted(os.listdir(tdir)):
            if not f.endswith(".json"):
                continue
            t = json.load(open(os.path.join(tdir, f)))
            errs = []
            if t.get("schema") != C.SCHEMA_TASK:
                errs.append("schema")
            check_task(t, recs, errs)
            mr = t.get("model_runs") or {}
            got = C.repeated_records(t, recs)
            if mr.get("allowed") and got:
                errs.append("model_runs.allowed, but jobs can be served one recorded run twice: " + "; ".join(got[:2]))
            if t["task_id"].startswith("pkg-") and got:
                errs.append("a package must serve every job a distinct recorded run")
            if "allowed" not in mr:
                errs.append("model_runs missing: regenerate")
            tasks += 1
            want = C.task_class(t["jobs"], t["box"]["gpus"])[0] if t.get("jobs") else None
            if t.get("task_class") not in C.TASK_CLASSES:
                errs.append(f"task_class {t.get('task_class')!r}")
            elif t["task_class"] != want:
                errs.append(f"task_class {t['task_class']} but the jobs' GPU options give {want}")
            if errs:
                bad += 1
                print(f"TASK {t.get('task_id')}: " + "; ".join(errs[:6]))
    by_src = collections.Counter(r["identity"]["record_id"].split("-")[0] for r in recs.values())
    ready = collections.Counter(r["identity"]["record_id"].split("-")[0] for r in recs.values() if r["readiness"]["replay"])
    for (stored, ver), n in STALE_FP.items():
        bad += n
        print(f"STALE FINGERPRINTS: {n} records were hashed under {stored}, and gpu-mock's fingerprint.py is now {ver}. "
              "Nothing is wrong with the records; rebuild the registry (AGENTS.md, Rebuilding the registry).")
    print(f"records {len(recs)} | example-order tables {len(tables)} | tasks {tasks} | replicate groups {len(groups)} | contract violations {bad}")
    for s in sorted(by_src):
        print(f"  {s}: {by_src[s]} records, {ready[s]} replay-ready")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
