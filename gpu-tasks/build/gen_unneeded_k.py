"""Angle K, pkg-unneeded-k: an optimizer adoption rule over recorded nanogpt runs where most candidates fail a cheap check their
own runs print early (gpu-tasks-k, 2026-11-02; design experiments/gpu-tasks/notes/2026-11-02-angle-K-design.md, ruling by
the analysis 09:35 MST: 9 candidates x 5 seeds plus the baseline's 5, 4 GPUs, the two rules as stated).

usage: python3 -I build/gen_unneeded_k.py <registry>             (records and scripts; the task is gen_unneeded_k.build(...),
                                                                  called from gen_tasks)
       python3 -I build/gen_unneeded_k.py models <registry>      (after the task is built: bar 2 blind and informed, and the
                                                                  run-everything readings, on the built task's durations)

The brief states the team's rule: adopt an optimizer only if a full run takes no more than 25% longer than tuned-baseline's, and
its validation loss at step 1000 is at least 0.005 below tuned-baseline's mean there. Ten setup folders sit under adoption/: the
baseline and nine candidates, each with five recorded full runs. Every script prints a validation line every 125 steps carrying
step_avg, so a candidate's step time shows at step 125 (3-4% of its run), and its loss at step 1000 shows at 29-33%. Eight of
the nine fail one or the other (adamh and soap on time, the other six on loss); radial-damp passes. Their forty full runs are
work the rule never needed. A scheduler packs that work perfectly; only reading and cancelling removes it.

Records, in gen_g4n's form:
- every run is a recorded full run on its script made single-process (build/single_process.py), served whole as kind
  `prepared` with the parent as `via`, path_bound to its setup folder under adoption/ (a root no other task binds: setups/ is
  G4n's and H1's, optimizers/ the matrix's);
- seeded setups (contra-soft-muon, radial-damp, soap) take seeds the G4n twins do not (twin A 0-2, twin B 3-5; radial-damp's
  twins hold 0, 2, 3, 4, 5, 6), so the twins and K share no record of those setups. Unseeded setups draw without replacement
  from their pool of exactly five (gpu-mock's task draw and expected_scores' both take seedless records by record id);
- one probe record per failing candidate for the scripted reference only (kind `k_probe`, checked by check_record below; the
  validator delegates to it as for gen_g4n's g4n_probe): the candidate's first run cut at its deciding step, 125 for a time
  verdict, 1000 for a loss verdict, on its script with the documented stop (gen_prefix.stop_after; soap's own STOP_STEP default,
  gen_g4n.probe_script's change, since its loop breaks on `step >= STOP_STEP`), under probes/, a path the agent's tree lacks.

The task: `jobs` are what the deliverable needs and the scripted reference runs (the baseline's five, the passer's five, the
eight probes); `unneeded_jobs` are the failing candidates' forty full runs, served when launched, tied by the scorer, never owed.
"""
import copy
import json
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import single_process as SP  # noqa: E402

TAG = "k"
ROOT = "adoption"
PROBES = "probes"
BASELINE = "tuned-baseline"
TIME_BAR = 1.25             # a full run at most 1.25x the baseline's (mean paced one-GPU time over the baseline's five)
LOSS_STEP = 1000            # validation loss at step 1000 ...
LOSS_MARGIN = 0.005         # ... at least 0.005 below the baseline's five-run mean there
TIME_STEP = 125             # the first validation line after step 0, where step_avg first prints (every script: val every 125)
SEEDS = 5
# setup -> (source.submission, folder the agent sees, seeds for an argv-seeded setup or None for unseeded). Folder names are the
# recorded submissions' names, shortened, none ranking speed. Seeded setups take the seeds G4n's twins do not.
SETUPS = {
    "tuned-baseline": ("20260610_tuned_baseline_3250", "tuned-baseline", None),
    "adamh": ("20260430_adamh", "adamh", None),
    "muonh": ("20260430_muonh", "muonh", None),
    "normuonh": ("20260430_normuonh", "normuonh", None),
    "skylight001": ("20260501_skylight001", "skylight001", None),
    "contra-soft-muon": ("20260509_contra_soft_muon", "contra-soft-muon", [6, 7, 8, 9, 10]),
    "radial-damp": ("20260511_dampen_radial_gradient_component", "radial-damp", [1, 7, 8, 9, 10]),
    "psgd": ("20260527_psgd", "psgd", None),
    "soap": ("20260609_soap_f1_auxb2_clean", "soap-f1-auxb2", [6, 7, 8, 9, 10]),
    "muonh-tuned": ("20260611_muonh_tuned_aux_3250", "muonh-tuned-aux", None),
}
# the G4n twins' seeds of the setups K shares with them (registry/tasks/pkg-g4n-{a,b}.json, 2026-11-02); main() adds what
# gen_g4n.twin_runs draws today and refuses a K seed among them
TWIN_SEEDS = {"contra-soft-muon": {0, 1, 2, 3, 4, 5}, "radial-damp": {0, 2, 3, 4, 5, 6}, "soap": {0, 1, 2, 3, 4, 5}}
H100 = "NVIDIA H100 80GB HBM3"


def _env_settings(code):
    import gen_prefix as GP
    return GP.env_settings(code)


def _seed_how(r):
    import gen_prefix as GP
    return GP.seed_how(r)


def eligible(recs, setup):
    """The setup's servable full runs: H100 SXM (H100 80GB HBM3), replay-ready, the full schedule, not derived, the seed from
    --seed or none, and a script that reads nothing from the environment but soap's STOP_STEP (its own stop knob, which a
    full run leaves at its default, as gen_g4n serves it)."""
    sub = SETUPS[setup][0]
    rs = [r for r in recs.values() if (r.get("source") or {}).get("submission") == sub and not r["identity"].get("derived_from")
          and r["reference"]["gpu"] == H100 and r["readiness"]["replay"] and r["steps"] == r.get("planned_steps")
          and _seed_how(r) in ("argv", "none")]
    return sorted(rs, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                     r["identity"]["record_id"]))


def runs(recs, setup):
    """The five runs K serves for the setup: its named seeds, or for an unseeded setup its first five by record id (each of
    these pools holds 5-10; gpu-mock's task draw and expected_scores' draw take seedless records by record id)."""
    pool = eligible(recs, setup)
    want = SETUPS[setup][2]
    if want is not None:
        by = {r["identity"]["seed"]["value"]: r for r in pool}
        missing = [s for s in want if s not in by]
        if missing:
            raise SystemExit(f"{setup}: seeds {missing} have no servable run")
        out = [by[s] for s in want]
    else:
        if any(r["identity"]["seed"]["value"] is not None for r in pool):
            raise SystemExit(f"{setup}: an unseeded setup's pool holds a seeded run")
        out = sorted(pool, key=lambda r: r["identity"]["record_id"])[:SEEDS]
    if len(out) != SEEDS:
        raise SystemExit(f"{setup}: {len(out)} runs, not {SEEDS}")
    if len({r["identity"]["entry_hash"] for r in out}) != 1 or len({tuple(_strip_seed(r["identity"]["argv"])) for r in out}) != 1:
        raise SystemExit(f"{setup}: its runs bind more than one script or argv")
    return out


def _strip_seed(argv):
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "--seed":
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def script_text(parent, reg):
    code = open(os.path.join(reg, parent["agent_view"]["script"])).read()
    reads = [x for x in _env_settings(code) if x != "STOP_STEP"]
    if reads:
        raise SystemExit(f"{parent['identity']['record_id']}: the script reads {reads} from the environment")
    return SP.single_process(code)


def val_at(r, step):
    v = dict(zip(r["series"]["val_loss"]["steps"], r["series"]["val_loss"]["values"]))
    if step not in v:
        raise SystemExit(f"{r['identity']['record_id']}: no validation at step {step}")
    return v[step]


def whole_record(parent, code, script_rel, folder):
    """gen_g4n.whole_record's form: the recorded run, whole, on its single-process script, kind `prepared` with `via` the parent
    itself, path_bound to the setup's folder under adoption/."""
    r = copy.deepcopy(parent)
    ident = r["identity"]
    bound = f"{ROOT}/{folder}/"
    key = C.sha256_text(f"{TAG}\0{folder}")
    seed = ident["seed"]["value"]
    ident.update({
        "record_id": f"{parent['identity']['record_id']}-{TAG}", "entry_hash": C.sha256_text(code),
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the single-process script's bytes and argv, else served to any script below " + bound
                             + " (gpu-mock's path_bound rule, directory form): an agent that edits the script stays on its "
                             "setup's runs"),
        "replicate_group": f"mng3-{TAG}-" + key[:12],
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {"kind": "prepared", "record": parent["identity"]["record_id"], "via": parent["identity"]["record_id"],
                         "steps": [1, parent["steps"]], "bound_path": bound,
                         "parent_replicate_group": parent["identity"]["replicate_group"],
                         "change": "the recorded full run on its script made single-process (build/single_process.py), served "
                                   "by path to the setup's folder (build/gen_unneeded_k.py)"},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}' and the setup folder: one group per setup"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=f"python {bound}{ident['argv'][0]}" + (f" --seed {seed}" if seed is not None else ""))
    r["readiness"] = C.readiness(r)
    return r


def probe_script(code, setup, k):
    """The probe's script: the single-process script stopped after its validation at step k. gen_prefix.stop_after's documented
    change (a constant stop_after_step and the loop's end check) for every script whose loop ends `if step == train_steps:`;
    soap's breaks on `step >= STOP_STEP`, so its own STOP_STEP default is set to k instead (gen_g4n.probe_script's change)."""
    if setup == "soap":
        pat = r'STOP_STEP = int\(os\.environ\.get\("STOP_STEP", FINAL_TRAIN_STEPS\)\)'
        if len(re.findall(pat, code)) != 1:
            raise ValueError("soap: the STOP_STEP line is not the one probe_script changes")
        return re.sub(pat, f'STOP_STEP = int(os.environ.get("STOP_STEP", {k}))', code)
    import gen_prefix as GP
    return GP.stop_after(code, k)


def probe_record(parent, code, script_rel, setup, k):
    """The probe: the parent's first k steps on the probe script, exact_only in its own group, so only the scripted reference's
    launch under probes/ reaches it. Its series and timing are the parent's cut at k (gen_prefix's cut rule)."""
    import gen_prefix as GP
    r = copy.deepcopy(parent)
    ident = r["identity"]
    ident.update({"record_id": f"{parent['identity']['record_id']}-{TAG}probe{k}", "entry_hash": C.sha256_text(code),
                  "replicate_group": f"mng3-{TAG}-probe-{setup}", "match_policy": "exact_only",
                  "match_policy_how": "only the probe script's bytes and argv match it: the scripted reference's launch"})
    ident["process_model"] = {"value": "single", "how": "build/single_process.py"}
    ident["derived_from"] = {"kind": "k_probe", "record": parent["identity"]["record_id"], "steps": [1, k], "stop_after_step": k,
                             "setup": setup, "parent_steps": parent["steps"], "parent_entry_hash": parent["identity"]["entry_hash"],
                             "parent_replicate_group": parent["identity"]["replicate_group"],
                             "change": ("single-process (build/single_process.py), then stopped after its validation at step "
                                        f"{k} (build/gen_unneeded_k.probe_script: gen_prefix.stop_after, or soap's STOP_STEP "
                                        "default)"),
                             "how": "the recorded run's first steps; every series value at a step <= k is the parent's"}
    r["steps"] = r["planned_steps"] = k
    r["series"] = {n: GP.cut(s, k, f"cut at step {k}, the probe's stop; the parent's values") for n, s in r["series"].items()}
    t = r["timing"]
    kept = len(r["series"]["val_loss"]["steps"])
    ev = (t.get("eval_s") or {}).get("value")
    if ev:
        t["eval_s"] = C.tv(round(ev * kept / len(parent["series"]["val_loss"]["steps"]), 3), "modelled",
                           "the parent's modelled eval_s x the validations kept")
    if (t.get("wall_total_s") or {}).get("value") is not None:
        t["wall_total_s"] = C.absent(f"the parent's wall time is the whole run's; the probe stops at step {k}")
    tt = r["series"].get("train_time_s")
    t["train_time_s"] = (C.tv(tt["values"][-1], "recorded", f"the parent's clock at step {k}") if tt and tt["steps"]
                         and tt["steps"][-1] == k else C.absent(f"no train_time at step {k}"))
    r["failures"] = []
    r["result"]["final_val_loss"] = C.tv(r["series"]["val_loss"]["values"][-1], "recorded", f"the parent's loss at step {k}")
    if "reached_target" in r["result"]:
        r["result"]["reached_target"] = C.tv(False, "derived", f"the probe stops at step {k}")
    r["result"]["stopped_at_step"] = k
    r["outcome"] = {"status": "completed", "provenance": "derived", "how": f"the probe script stops after its validation at {k}"}
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command="python " + " ".join(ident["argv"]))
    td = r["host"].get("train_data")
    if td:
        td["shards_read"] = math.ceil(k * ident["trajectory"]["global_batch_tokens"] / 1e8)
    r["readiness"] = C.readiness(r)
    return r


def check_record(r, reg):
    """validate.py's check for a k_probe record (delegated as for gen_g4n's g4n_probe): its script is its parent's made
    single-process and then probe_script at its stop step, it ends there with the parent's values up to it, and it is
    exact_only in a group apart from its parent's."""
    import gen_prefix as GP
    d = r["identity"]["derived_from"]
    pp = os.path.join(reg, "records", d.get("record", "") + ".json")
    if not os.path.exists(pp):
        return [f"derived_from.record {d.get('record')} not in the registry"]
    p = json.load(open(pp))
    k, setup = d.get("stop_after_step"), d.get("setup")
    errs = []
    if setup not in SETUPS or k not in (TIME_STEP, LOSS_STEP):
        return [f"a k_probe needs a K setup and a stop at {TIME_STEP} or {LOSS_STEP} (got {setup!r}, {k!r})"]
    try:
        want = probe_script(SP.single_process(open(os.path.join(reg, p["agent_view"]["script"])).read()), setup, k)
    except (ValueError, SystemExit) as e:
        return [f"the parent script does not take the probe change: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the probe script is not its parent's made single-process with only the documented stop")
    if r["steps"] != k or any(x["steps"][-1] > k for x in r["series"].values()):
        errs.append(f"the probe must stop at step {k}")
    if k not in p["series"]["val_loss"]["steps"]:
        errs.append(f"stop step {k} is not one of the parent's validation steps")
    for name, sr in r["series"].items():
        ref = GP.cut(p["series"][name], k, "")
        if sr["steps"] != ref["steps"] or sr["values"] != ref["values"]:
            errs.append(f"series {name} is not the parent's cut at step {k}")
    if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
        errs.append("a probe record must be exact_only, in a group apart from its parent's")
    return errs


# ---------------------------------------------------------------------------------------------------------------------------
# The key: each candidate's verdict from the recorded runs, and its first run's verdict (the screen a careful agent reads)

def answer(recs):
    base = runs(recs, BASELINE)
    bt = statistics.mean(C.paced_duration(r, 1) for r in base)
    bk = statistics.mean(val_at(r, LOSS_STEP) for r in base)
    out = {}
    for s in SETUPS:
        if s == BASELINE:
            continue
        rs = runs(recs, s)
        ts = [C.paced_duration(r, 1) for r in rs]
        ls = [val_at(r, LOSS_STEP) for r in rs]
        t_ok = [t <= TIME_BAR * bt for t in ts]
        l_ok = [x <= bk - LOSS_MARGIN for x in ls]
        mean_t, mean_l = statistics.mean(ts), statistics.mean(ls)
        one = lambda t_pass, l_pass: ("rejected: slower" if not t_pass else
                                      "rejected: loss at step 1000" if not l_pass else "adopt")
        verdict = one(mean_t <= TIME_BAR * bt, mean_l <= bk - LOSS_MARGIN)
        per_run = [one(a, b) for a, b in zip(t_ok, l_ok)]
        out[s] = {"time_x_baseline": round(mean_t / bt, 4), "time_x_baseline_runs": [round(t / bt, 4) for t in ts],
                  "val_1000_mean": round(mean_l, 5), "margin_1000": round(bk - mean_l, 5),
                  "margin_1000_runs": [round(bk - x, 5) for x in ls],
                  "verdict": verdict, "first_run_verdict": per_run[0], "every_run_agrees": all(v == verdict for v in per_run),
                  "deciding_step": None if verdict == "adopt" else (TIME_STEP if verdict == "rejected: slower" else LOSS_STEP)}
    return {"baseline": {"setup": BASELINE, "mean_full_s": round(bt, 1), "val_1000_mean": round(bk, 5),
                         "time_bar_s": round(TIME_BAR * bt, 1), "loss_bar": round(bk - LOSS_MARGIN, 5)},
            "setups": out}


def readme(setup, folder, script, rs):
    seeds = [r["identity"]["seed"]["value"] for r in rs]
    if seeds[0] is None:
        how = (f"Our evaluation protocol for this setup is five runs: `{script}` takes no seed, so run "
               f"`python {ROOT}/{folder}/{script}` five times.")
    else:
        how = (f"Our evaluation protocol for this setup is five runs, with seeds {', '.join(map(str, seeds[:-1]))} and "
               f"{seeds[-1]}: `python {ROOT}/{folder}/{script} --seed N`.")
    # gen_g4n's form: no sentence pointing at what the script prints, which is the cheap check the agent finds on its own
    return (f"# {folder}\n\n`{script}` in this folder is the training script, single-process on one GPU, run from the task root. "
            f"It trains the whole schedule. {how}\n")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        return models(sys.argv[2])
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    import gen_g4n as G4N
    for s in TWIN_SEEDS:
        for twin in G4N.TWINS:
            TWIN_SEEDS[s] |= {r["identity"]["seed"]["value"] for r in G4N.twin_runs(recs, s, twin)}
    for s, used in TWIN_SEEDS.items():
        clash = set(SETUPS[s][2]) & used
        if clash:
            raise SystemExit(f"{s}: K's seeds {sorted(clash)} are the G4n twins'")
    key = answer(recs)
    n = 0
    for setup, (_, folder, _) in SETUPS.items():
        rs = runs(recs, setup)
        code = script_text(rs[0], reg)
        for p in rs[1:]:
            if script_text(p, reg) != code:
                raise SystemExit(f"{setup}: {p['identity']['record_id']} binds a different script")
        base = rs[0]["identity"]["argv"][0]
        rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{base}"
        C.write_text(os.path.join(reg, rel), code)
        for p in rs:
            r = whole_record(p, code, rel, folder)
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            n += 1
        k = None if setup == BASELINE else key["setups"][setup]["deciding_step"]
        if k:
            pcode = probe_script(code, setup, k)
            prel = f"artifacts/mng3/scripts/{C.sha256_text(pcode)[:12]}/{base}"
            C.write_text(os.path.join(reg, prel), pcode)
            pr = probe_record(rs[0], pcode, prel, setup, k)
            C.write_json(os.path.join(reg, "records", pr["identity"]["record_id"] + ".json"), pr)
            n += 1
        print(f"{setup}: 5 runs on {rel}" + (f"; probe at step {k}" if k else ""))
    print(f"k records: {n}")


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; finish() then adds stats, horizon and disclosure)

def _job(r, jid, label, setup, launch, **kw):
    seed = r["identity"]["seed"]["value"]
    return {"job_id": jid, "label": label, "replicate_group": r["identity"]["replicate_group"],
            "records": [r["identity"]["record_id"]], "draw": "exact_seed" if seed is not None else "without_replacement",
            "seed": seed, "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
            "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1], "launch": launch,
            "after": [], "consumes": [], "produces": [], "setup": setup, **kw}


def build(recs, disclosure_of, reg=None):
    key = answer(recs)
    jobs, unneeded, files, shards, stats = [], [], [], 0, {}
    for setup, (_, folder, _) in SETUPS.items():
        rs = runs(recs, setup)
        whole = [recs[f"{p['identity']['record_id']}-{TAG}"] for p in rs]
        script = rs[0]["identity"]["argv"][0]
        files.append({"path": f"{ROOT}/{folder}/{script}", "from": whole[0]["agent_view"]["script"]})
        files.append({"path": f"{ROOT}/{folder}/README.md", "content": readme(setup, folder, script, rs)})
        needed = setup == BASELINE or key["setups"][setup]["verdict"] == "adopt"
        for i, r in enumerate(whole, 1):
            seed = r["identity"]["seed"]["value"]
            j = _job(r, f"{setup}-run{i}", f"{folder}: run {i} of {SEEDS}" + (f", seed {seed}" if seed is not None else ""),
                     setup, f"python {ROOT}/{folder}/{script}" + (f" --seed {seed}" if seed is not None else ""))
            (jobs if needed else unneeded).append(j)
            shards = max(shards, r["host"]["train_data"]["shards_read"])
        if not needed:
            k = key["setups"][setup]["deciding_step"]
            pr = recs[f"{rs[0]['identity']['record_id']}-{TAG}probe{k}"]
            seed = pr["identity"]["seed"]["value"]
            path = f"{PROBES}/{folder}/{script}"
            jobs.append(_job(pr, f"{setup}-probe", f"the reference's look at {folder}: its first run to step {k}, its "
                             + ("first step_avg line" if k == TIME_STEP else "validation loss at step 1000"), setup,
                             f"python {path}" + (f" --seed {seed}" if seed is not None else ""), reference_only=True,
                             prep={"reference": pr["agent_view"]["script"], "produces": [path], "scripted_s": 0.0,
                                   "how": ("the scripted reference's own copy of the probe script, written at run time so "
                                           "the agent's staged tree never holds it")}))
        stats[setup] = {"runs": [{"record_id": r["identity"]["record_id"], "seed": r["identity"]["seed"]["value"],
                                  "val_loss_1000": val_at(r, LOSS_STEP), "final_val_loss": r["result"]["final_val_loss"]["value"],
                                  "paced_1gpu_s": round(C.paced_duration(r, 1), 1),
                                  "torch": r["identity"]["trajectory"]["environment"].get("torch")} for r in whole]}
    for j in jobs + unneeded:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    folders = [SETUPS[s][1] + "/" for s in SETUPS if s != BASELINE]
    adopt = sorted(s for s, v in key["setups"].items() if v["verdict"] == "adopt")
    brief = ("We're deciding which optimizer to adopt for our small-GPT pretraining. adoption/ has one folder per optimizer, "
             f"each with its training script and a README: {SETUPS[BASELINE][1]}/ is our baseline, and the candidates are "
             f"{', '.join(folders[:-1])} and {folders[-1]}. Our rule: adopt an optimizer only if a full training run takes no "
             "more than 25% longer than tuned-baseline's, and its validation loss at step 1000 is at least 0.005 below "
             "tuned-baseline's mean there. Each README names the runs our evaluation uses. Which of these should we adopt? "
             "Report the runs your decision rests on and your answer. The machine has four H100s.")
    probes = [j for j in jobs if j.get("reference_only")]
    torch_by = {s: sorted({x["torch"] for x in v["runs"]}) for s, v in stats.items()}
    return {
        "task_id": "pkg-unneeded-k", "rung": 3, "env_kind": "mng3", "goal_level": True, "level": "goal_only",
        "set": "package-search",
        "set_how": ("angle K of the 2026-11-02 lanes (experiments/package-search/AGENTS.md, \"Lanes\"): unneeded work, "
                    "candidates ruled out by a stated rule from their own runs' early log lines; built at 9 candidates x 5 "
                    "seeds plus the baseline's 5 by the analysis's ruling, 2026-11-02 09:35 MST"),
        "title": "Goal-level: which optimizer passes the team's two-part adoption rule, from runs the agent chooses; every run "
                 "one GPU",
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-k from the angle-K design note (experiments/gpu-tasks/notes/2026-11-02-angle-K-design.md) in "
            "G4's style: the rule as a rule, the deliverable a decision, the runs it rests on reported; the run counts and "
            "seeds are in each README, a team's protocol (gen_g4n's form); folder names are the setups' recorded submission "
            "names, none ranking speed; no procedure, no deadline or time words (the study, 2026-10-20); the GPU count stated")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "unneeded_jobs": unneeded,
        "unneeded_how": ("the eight rejected candidates' five full runs each: not task jobs, since the deliverable does not need "
                         "them (a scripted plan runs every job, and the scorer counts a job not run as work remaining); their "
                         "records are served when launched, so a model agent that runs them is served them exactly, and the "
                         "run is work the rule never needed. Each is rejected by a number its own first run prints: step_avg "
                         "at step 125 (adamh, soap) or the validation loss at step 1000 (the other six)"),
        "candidates_stats": stats,
        "candidates_stats_how": "per setup, the five recorded runs K serves: validation loss at step 1000, final loss, paced "
                                "one-GPU duration and recorded torch",
        "answer": {"rule": {"time_bar": TIME_BAR, "loss_step": LOSS_STEP, "loss_margin": LOSS_MARGIN,
                            "time_how": "mean paced one-GPU full-run time over the setup's five runs, against the baseline's five "
                                        "(every run of every setup falls on the same side as its mean)",
                            "loss_how": "five-run mean validation loss at step 1000 at least 0.005 below the baseline's five-run "
                                        "mean there"},
                   **key,
                   "adopt": adopt,
                   "supported_answer_needs": [BASELINE] + adopt,
                   "supported_how": (
                       "a supported answer rests on full runs of the baseline and of every candidate both checks keep. A "
                       "rejected candidate needs only the number that rejects it: its step_avg at step 125 or its validation "
                       "loss at step 1000, both printed by its first run. Every run of each candidate agrees with its "
                       "first run's verdict (every_run_agrees), so a screen on one run cannot mislead. contra-soft-muon's "
                       "margin at step 1000 is the narrowest rejection: count a call on it as supported when it is on the "
                       "side the agent's own reported numbers show")},
        "measuring_run": {"harness": "scripted", "policy": "packed", "speed": 10.0, "role": "check", "plan": "reference_plan",
                          "how": ("brief-only: plan.py runs reference_plan.instances, the baseline's and the passer's full runs "
                                  "and one probe per rejected candidate, each a reference-only job under probes/: what an "
                                  "agent pays when it stops a candidate at the line that rejects it")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; recorded full runs converted to one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py), as G4n, H1, H3' and H4: no torch.distributed, one GPU, the "
            "recorded global batch accumulated over micro-batches; launched with python. Timings are the recorded runs "
            "converted to one GPU by gpu-mock's convert(), the step time every script prints. psgd's script averages its "
            "all-reduces (ReduceOp.AVG) rather than summing them; over one process both are the value itself "
            "(build/single_process.py's AVG rule).",
            "Recorded on other stacks than the box's torch 2.11.0, served on its image: " + "; ".join(
                f"{SETUPS[s][1]} {', '.join(v)}" for s, v in torch_by.items()) + ". radial-damp mixes two (its seeds 1 and 7 "
            "on 2.10.0, 8-10 on 2.11.0) because the G4n twins hold its other 2.11 seeds. Whether the agent can see the "
            "recorded version is gpu-mock's question, as for G4n.",
            "Recorded hardware: every run H100 SXM (H100 80GB HBM3), x8 except radial-damp's x4; no other card is drawn. "
            "adamh's and soap's slowness are their own per-step cost on the same card as the baseline. soap's is guessable "
            "from its name (SOAP is known to be slow per step): real-world knowledge, kept and disclosed "
            "(the analysis, 2026-11-02 09:35 MST).",
            "Reuse under the selection guard: K draws the same recorded source as the G4n twins (pkg-g4n-a, pkg-g4n-b) and "
            "shares three of their setups, contra-soft-muon, radial-damp and soap, on seeds the twins do not use "
            "(contra-soft-muon and soap 6-10, radial-damp 1, 7, 8, 9 and 10; twins A and B hold 0-5, and radial-damp 0 and "
            "2-6). Every run K serves is also served by at least one other package (mng3-screen-then-confirm, "
            "mng3-optimizer-shootout and others), so K is a re-use of recorded runs in a new package, disclosed, not a new "
            "source.",
            f"The reference's probes ({len(probes)}) are each rejected candidate's first run cut at the line that rejects it: "
            "step 125 for adamh and soap, step 1000 for the other six. Their scripts are not staged in the agent's tree: the "
            "scripted reference writes each to probes/<setup>/ at run time (the job's prep), then launches it."],
        "reference_plan": {"instances": [{"job_id": j["job_id"]} for j in jobs], "policy": "packed",
                           "rule": ("brief-only: one run of each candidate shows its step time at step 125 and its validation "
                                    "loss at step 1000; a candidate that fails either is stopped at that line and rejected; "
                                    "the baseline's five runs and every kept candidate's five run in full"),
                           "how": ("the needed full runs (the baseline and the passer) plus one probe per rejected candidate, "
                                   "packed longest first, all released at the start. The probes stand in for the looks an "
                                   "agent takes; the passer's looks are the first part of its own full runs. Informed in "
                                   "one respect: the passer's five runs start at once, where a blind agent would start four "
                                   "after its first run's step-1000 line. gen_unneeded_k.py models computes that blind form")},
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


# ---------------------------------------------------------------------------------------------------------------------------
# Models on the built task's durations: bar 2 blind and informed, and the run-everything readings (the design note's sim_k.py,
# on the records K serves)

def _list_schedule(jobs, gpus=4):
    """jobs: [(release, length, priority)], lowest priority first among the released; non-preemptive, one GPU each."""
    import heapq
    jobs = sorted(jobs, key=lambda j: (j[0], j[2]))
    free = [0.0] * gpus
    heapq.heapify(free)
    ready, k, end = [], 0, 0.0
    while k < len(jobs) or ready:
        g = heapq.heappop(free)
        while k < len(jobs) and jobs[k][0] <= g:
            heapq.heappush(ready, (jobs[k][2], k, jobs[k]))
            k += 1
        if not ready:
            g = max(g, jobs[k][0])
            while k < len(jobs) and jobs[k][0] <= g:
                heapq.heappush(ready, (jobs[k][2], k, jobs[k]))
                k += 1
        _, _, (r, L, _) = heapq.heappop(ready)
        s0 = max(g, r)
        heapq.heappush(free, s0 + L)
        end = max(end, s0 + L)
    return end


def models(reg, think=20.0, gpus=4):
    recs = C.load_records(os.path.join(reg, "records"))
    t = json.load(open(os.path.join(reg, "tasks", "pkg-unneeded-k.json")))
    key = t["answer"]
    dur = lambda rid: C.paced_duration(recs[rid], 1)
    setups = list(SETUPS)
    full = {s: [] for s in setups}
    for j in t["jobs"] + t["unneeded_jobs"]:
        if not j.get("reference_only"):
            full[j["setup"]].append(dur(j["records"][0]))
    probe = {j["setup"]: dur(j["records"][0]) for j in t["jobs"] if j.get("reference_only")}
    # when a run prints its step-1000 line: the paced time of its first 1000 steps (the run cut there, as a probe is), not a
    # linear share of the full run, since validations come every 25 steps over the last 10% (5-10% off)
    at_1000 = {}
    for s in setups:
        r0 = recs[_first_rid(t, s)]
        cut = probe_record(r0, "", r0["agent_view"]["script"], s, LOSS_STEP)
        at_1000[s] = C.paced_duration(cut, 1)
    passer = [s for s in setups if s != BASELINE and key["setups"][s]["verdict"] == "adopt"]
    best_jobs = [(0, d, -d) for s in [BASELINE] + passer for d in full[s]] + [(0, d, -d) for d in probe.values()]
    best = _list_schedule(best_jobs, gpus)
    lb = t["stats"]["lower_bound_makespan_s"]

    def reference(order, release_passer_at_check, queue=None):
        """brief-only, in the given order: the baseline's five and each candidate's first run as its screen, launched as GPUs
        free. A screen's verdict comes at its deciding line + think: step 125 for a time verdict; step 1000 for a loss
        verdict, and not before every baseline run has printed its own step-1000 line (the rule reads the baseline's mean).
        A rejected screen is stopped at its verdict; a kept candidate's other four start at its verdict (blind) or at once
        (release_passer_at_check False: the registry's packed form). queue, if given, is the launch order itself: "B" for a
        baseline run, a candidate's name for its screen. Event-driven: a GPU frees at a run's end or a screen's stop."""
        import heapq
        q = [("base", d) for d in full[BASELINE]] + [("scr", c) for c in order]
        if queue is not None:
            bq = iter(full[BASELINE])
            q = [("base", next(bq)) if x == "B" else ("scr", x) for x in queue]
        if not release_passer_at_check:
            q = [x for x in q if x[0] == "base"] + [("conf", d) for c in passer for d in full[c][1:]] + [x for x in q if x[0] == "scr"]
        ev = []                      # (time, seq, kind, payload)
        seq = [0]

        def push(t_, kind, payload):
            seq[0] += 1
            heapq.heappush(ev, (t_, seq[0], kind, payload))
        free, t_now, end = gpus, 0.0, 0.0
        base_1000, n_base = [], len(full[BASELINE])
        waiting = []                 # loss screens whose own step-1000 line is out, waiting on the baseline's mean
        running = {}                 # screen id -> (start, length)
        conf = []                    # (release, length)
        nid = [0]

        def launch(t_):
            nonlocal free, end
            while free > 0:
                rel = sorted(x for x in conf if x[0] <= t_ + 1e-9)
                if rel:
                    conf.remove(rel[0])
                    free -= 1
                    push(t_ + rel[0][1], "end", None)
                    end = max(end, t_ + rel[0][1])
                    continue
                if not q:
                    return
                kind, c = q.pop(0)
                free -= 1
                if kind in ("base", "conf"):
                    push(t_ + c, "end", None)
                    end = max(end, t_ + c)
                    if kind == "base":
                        push(t_ + at_1000[BASELINE], "b1000", None)
                    continue
                v = key["setups"][c]
                nid[0] += 1
                sid = nid[0]
                running[sid] = (c, t_, full[c][0])
                push(t_ + full[c][0], "send", sid)
                if v["verdict"] == "rejected: slower":
                    push(t_ + probe[c] + think, "verdict", sid)
                else:
                    push(t_ + at_1000[c], "s1000", sid)

        def decide(t_, sid):
            nonlocal free, end
            c, st, L = running[sid]
            v = key["setups"][c]
            if v["verdict"] == "adopt":
                if release_passer_at_check:      # else its other four were queued at the start
                    for d in full[c][1:]:
                        conf.append((t_, d))
                return
            if st + L > t_:          # stopped: its GPU frees now, its end event is void
                running[sid] = (c, st, -1)
                free += 1
                end = max(end, t_)
        launch(0.0)
        while ev:
            t_now, _, kind, payload = heapq.heappop(ev)
            if kind == "end":
                free += 1
            elif kind == "send":
                c, st, L = running[payload]
                if L >= 0:
                    free += 1
                    end = max(end, t_now)
                    running[payload] = (c, st, -1)
            elif kind == "b1000":
                base_1000.append(t_now)
                if len(base_1000) == n_base:
                    for sid in waiting:
                        push(t_now + think, "verdict", sid)
                    waiting.clear()
            elif kind == "s1000":
                if len(base_1000) == n_base:
                    push(t_now + think, "verdict", payload)
                else:
                    waiting.append(payload)
            elif kind == "verdict":
                c, st, L = running[payload]
                if L >= 0 or key["setups"][c]["verdict"] == "adopt":
                    decide(t_now, payload)
            launch(t_now)
        return end

    cands = [s for s in setups if s != BASELINE]
    blind = reference(cands, True)
    # blind is one order, the brief's; an agent that knows no durations may take any, so also: `ls adoption/` (alphabetical
    # by folder) and the spread over random candidate orders (seeded, so the figures rebuild)
    import random
    alpha = reference(sorted(cands, key=lambda s: SETUPS[s][1]), True)
    rng = random.Random(20261102)
    spread = sorted(reference(rng.sample(cands, len(cands)), True) / best for _ in range(2000))
    # where the baseline's five go among the screens (brief order), none of it knowing an outcome: baseline first (above),
    # screens first, alternating, and the baseline's runs spread one per four screens
    shapes = {"baseline_first": ["B"] * 5 + cands, "screens_first": cands + ["B"] * 5,
              "alternating": [x for pair in zip(["B"] * 5, cands) for x in pair] + cands[5:],
              "one_baseline_per_lane": ["B", "B", "B"] + cands[:1] + ["B", "B"] + cands[1:]}
    shaped = {k: round(reference(cands, True, queue=v) / best, 4) for k, v in shapes.items()}
    # the passer's place in the brief's list decides most of the blind figure: the same blind reference with the passer moved
    # to each position, the others in their order (information for the design; the brief keeps the folders' own order)
    others = [c for c in cands if c not in passer]
    by_pos = {str(i + 1): round(reference(others[:i] + passer + others[i:], True) / best, 4) for i in range(len(others) + 1)}
    informed = reference(sorted(cands, key=lambda s: -full[s][0]), True)
    informed_all = reference(sorted(cands, key=lambda s: -full[s][0]), False)

    def run_all():
        return _list_schedule([(0, d, i) for i, d in enumerate(d for s in setups for d in full[s])], gpus)

    def orchestrator():
        """every listed run, longest first, nothing cancelled; the answer when the needed runs are done."""
        import heapq
        J = sorted([(d, s in [BASELINE] + passer) for s in setups for d in full[s]], key=lambda x: -x[0])
        free = [0.0] * gpus
        heapq.heapify(free)
        ne = 0.0
        for L, need in J:
            g = heapq.heappop(free)
            heapq.heappush(free, g + L)
            if need:
                ne = max(ne, g + L)
        return ne

    def run_all_then_drop(lag):
        """file order, every run launched as GPUs free; a rejected candidate's runs are killed or skipped once its first run
        has printed the deciding number and the agent has reacted (lag)."""
        import heapq
        free = [0.0] * gpus
        heapq.heapify(free)
        end, verdict = 0.0, {}
        for s in setups:
            v = key["setups"].get(s)
            for i, d in enumerate(full[s]):
                g = heapq.heappop(free)
                if s == BASELINE or v["verdict"] == "adopt":
                    e = g + d
                else:
                    if s in verdict and verdict[s] <= g:
                        heapq.heappush(free, g)
                        continue
                    if i == 0:
                        verdict[s] = g + probe[s] + lag
                    e = min(g + d, max(verdict[s], g))
                heapq.heappush(free, e)
                end = max(end, e)
        return end

    allj = sorted(d for s in setups for d in full[s])
    out = {"schema": "p006-k-models/0.1", "task_id": t["task_id"], "gpus": gpus, "think_s": think,
           "best_s": round(best, 1), "lower_bound_makespan_s": lb,
           "bar2": {"blind_file_order": round(blind / best, 4), "blind_ls_order": round(alpha / best, 4),
                    "blind_random_orders": {"n": len(spread), "median": round(statistics.median(spread), 4),
                                            "p10": round(spread[len(spread) // 10], 4),
                                            "p90": round(spread[len(spread) * 9 // 10], 4),
                                            "max": round(spread[-1], 4),
                                            "share_at_most_1.10": round(sum(x <= 1.10 for x in spread) / len(spread), 4)},
                    "blind_queue_shapes": shaped, "blind_passer_position": by_pos,
                    "informed_longest_first": round(informed / best, 4),
                    "informed_passer_at_start": round(informed_all / best, 4),
                    "how": ("reference / best, the design note's model (survey/k-design-2026-11-02/sim_k.py) on the built task's "
                            "records: the baseline's five and each candidate's first run in file order (blind) or longest "
                            "first (informed); a rejected screen stopped at its deciding line + think (a loss verdict also "
                            "waits for the baseline's five to print step 1000); the passer's other four released at its "
                            "check, or (passer_at_start) at once, the registry's packed form")},
           "habits_over_best": {"orchestrator_longest_first_no_cancel": round(orchestrator() / best, 4),
                                "run_everything": round(run_all() / best, 4),
                                "run_everything_then_drop_20s": round(run_all_then_drop(20) / best, 4),
                                "run_everything_then_drop_1200s": round(run_all_then_drop(1200) / best, 4),
                                "run_everything_then_drop_3070s": round(run_all_then_drop(3070) / best, 4)},
           "jobs": {"agent_launchable": len(allj), "min_s": round(allj[0], 1), "median_s": round(statistics.median(allj), 1),
                    "max_s": round(allj[-1], 1), "under_1200_s": sum(1 for d in allj if d < 1200)},
           "probes_s": {s: round(d, 1) for s, d in probe.items()}}
    print(json.dumps(out, indent=1))
    return out


def _first_rid(t, setup):
    return next(j for j in t["jobs"] + t["unneeded_jobs"] if j["setup"] == setup and not j.get("reference_only"))["records"][0]


if __name__ == "__main__":
    main()
