"""G4n, pkg-g4n-a and pkg-g4n-b: an optimizer adoption rule over recorded nanogpt runs, the goal-only twin pair of mls-kv-gate's
mechanism on non-KV records (gpu-tasks-kvgate, 2026-10-24; design the harness's experiments/solution/probes/coldcheck/
g4n_design.py, rulings by the analysis 05:46-06:30 MDT).

usage: python3 -I build/gen_g4n.py <registry>      (records and scripts; the tasks are gen_g4n.build(..., twin), called from
                                                    gen_tasks)

The brief states the team's adoption rule: adopt a setup only if a full training run takes no more than 20% longer than the
baseline's, and its three-seed mean final validation loss beats the baseline's. The baseline is contra-soft-muon; five setups
are candidates. Which runs to make is the agent's. SOAP's per-step cost (2.95 s at one GPU against 1.59 for the baseline, 1.76x
the baseline's full run) shows in the first validation line every script prints, step 125, about 6.6 minutes in, so a cheap
look rejects it and its three full runs are work the rule never needed. A scheduler packs that work perfectly; only reading and
cancelling removes it, the behaviour Haiku failed on mls-kv-gate.

Records: the recorded full runs of each setup on their scripts made single-process (build/single_process.py, as H1 and H3' do:
one GPU per run is a fact of the code), served whole as kind `prepared` with the parent as `via`, path_bound to the setup's
folder (build/gen_h1.py's form, which validate.py checks).
Twins split each setup's runs in seed order: A takes the first three, B the next three, so the two are record-disjoint. Each
setup's README names its three seeds; the brief points at them. newton-muon's script takes no seed, so its runs are its first
three (A) or next three (B) by record id and its README says to run it three times.

The scripted reference (bar 2) is brief-only: the needed full runs plus one SOAP launch stopped at its first validation, which
is what an agent that reads step_avg and kills the run pays. plan.py cannot cancel a launch, so that probe is its own job on a
SOAP prefix record at step 125 under probes/, a path the agent's files do not include (analysis, 06:30 MDT: acceptable for bar
2; the agent-facing scripts stay as recorded, with no --stop-at added).
"""
import copy
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import single_process as SP  # noqa: E402

TWINS = {"a": slice(0, 3), "b": slice(3, 6)}
TAG = "g4n"
ROOT = "setups"
BASELINE = "contra-soft-muon"
TIME_BAR = 1.20
PROBE_STEP = 125            # SOAP's first validation, where step_avg first prints (TRAIN_PROGRESS_INTERVAL = 0 in every script)
# setup -> (record-id prefix of its recorded runs, torch the pool is drawn from). One torch per setup.
SETUPS = {
    "contra-soft-muon": ("mng3-20260509-contra-soft-muon-h100", "2.11.0"),
    "soda-fade": ("mng3-20260518-soda-fade-3000-n9-h200", "2.11.0"),
    "radial-damp": ("mng3-20260511-dampen-radial-gradient-component-h100", "2.11.0"),
    "newton-muon": ("mng3-20260505-newton-muon-h100", "2.11.0"),
    "aurora-pruned": ("mng3-20260514-aurora-proj-pruned-extended-contra-h200", "2.10.0"),
    "soap": ("mng3-20260609-soap-f1-auxb2-clean-h100", "2.9.0"),
}
# the folder names the agent sees: the setups' recorded submission names, no speed ranking
FOLDER = {"contra-soft-muon": "contra-soft-muon", "soda-fade": "soda-fade", "radial-damp": "radial-damp",
          "newton-muon": "newton-muon", "aurora-pruned": "aurora-proj-pruned", "soap": "soap-f1-auxb2"}


def pool(recs, setup):
    prefix, torch = SETUPS[setup]
    rs = [r for r in recs.values() if r["identity"]["record_id"].startswith(prefix) and not r["identity"].get("derived_from")
          and r["readiness"]["replay"] and r["steps"] == r.get("planned_steps")
          and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith(torch)]
    gpus = {(r["reference"]["gpu"], r["reference"]["gpu_count"]) for r in rs}
    if len(gpus) != 1:
        raise SystemExit(f"{setup}: its pool spans {sorted(gpus)}; one recorded GPU type per setup")
    return sorted(rs, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                     r["identity"]["record_id"]))


def twin_runs(recs, setup, twin):
    rs = pool(recs, setup)[TWINS[twin]]
    if len(rs) != 3:
        raise SystemExit(f"{setup} twin {twin}: {len(rs)} runs, not 3")
    return rs


def script_text(parent, reg):
    return SP.single_process(open(os.path.join(reg, parent["agent_view"]["script"])).read())


def whole_record(parent, code, script_rel, twin, folder):
    """The recorded run, whole, on its single-process script: kind `prepared` with `via` the parent itself (validate.py's rule
    for build/gen_h1.py's prepared records: entry_hash is the via record's script made single-process, and series, steps and
    timing are the via record's exactly), path_bound to the setup's folder so an edited script stays on its setup's runs. The
    recorded script is not given --stop-at/--resume (gen_segment), so the agent-facing script is the recorded one, single-process
    only."""
    r = copy.deepcopy(parent)
    ident = r["identity"]
    bound = f"{ROOT}/{folder}/"
    key = C.sha256_text(f"{TAG}{twin}\0{folder}")
    seed = ident["seed"]["value"]
    ident.update({
        "record_id": f"{parent['identity']['record_id']}-{TAG}{twin}", "entry_hash": C.sha256_text(code),
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the single-process script's bytes and argv, else served to any script below " + bound
                             + " (gpu-mock's path_bound rule, directory form): an agent that edits the script stays on its "
                             "setup's runs"),
        "replicate_group": f"mng3-{TAG}{twin}-" + key[:12],
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {"kind": "prepared", "record": parent["identity"]["record_id"], "via": parent["identity"]["record_id"],
                         "steps": [1, parent["steps"]], "bound_path": bound,
                         "parent_replicate_group": parent["identity"]["replicate_group"],
                         "change": "the recorded full run on its script made single-process (build/single_process.py), served "
                                   "by path to the setup's folder"},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}{twin}' and the setup folder: one group per setup and twin"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=f"python {bound}{ident['argv'][0]}" + (f" --seed {seed}" if seed is not None else ""))
    r["readiness"] = C.readiness(r)
    return r


def probe_script(code):
    """SOAP's single-process script with its own stop knob's default set to the probe step: `STOP_STEP = int(os.environ.get(
    "STOP_STEP", FINAL_TRAIN_STEPS))` becomes `... PROBE_STEP))`. The script then validates at step 125, prints its step_avg line
    and breaks (its own `if step >= STOP_STEP: break`), as the recorded run's first 125 steps."""
    pat = r'STOP_STEP = int\(os\.environ\.get\("STOP_STEP", FINAL_TRAIN_STEPS\)\)'
    if len(re.findall(pat, code)) != 1:
        raise SystemExit("soap: the STOP_STEP line is not the one probe_script changes")
    return re.sub(pat, f'STOP_STEP = int(os.environ.get("STOP_STEP", {PROBE_STEP}))', code)


def probe_record(parent, code, script_rel, twin):
    """The probe: the parent's first PROBE_STEP steps on the probe script, path_bound under probes/ so only the scripted
    reference's launch reaches it. Its series and timing are the parent's cut at PROBE_STEP (gen_prefix's cut rule)."""
    import gen_prefix as GP
    r = copy.deepcopy(parent)
    ident = r["identity"]
    k = PROBE_STEP
    ident.update({"record_id": f"{parent['identity']['record_id']}-{TAG}{twin}probe{k}", "entry_hash": C.sha256_text(code),
                  "replicate_group": f"mng3-{TAG}{twin}-probe", "match_policy": "exact_only",
                  "match_policy_how": "only the probe script's bytes and argv match it: the scripted reference's launch"})
    ident["process_model"] = {"value": "single", "how": "build/single_process.py"}
    ident["derived_from"] = {"kind": "g4n_probe", "record": parent["identity"]["record_id"], "steps": [1, k], "stop_after_step": k,
                             "parent_steps": parent["steps"], "parent_entry_hash": parent["identity"]["entry_hash"],
                             "parent_replicate_group": parent["identity"]["replicate_group"],
                             "change": ("single-process (build/single_process.py), then STOP_STEP's default set to "
                                        f"{k} (gen_g4n.probe_script): the script's own stop knob, so it validates at step {k} and "
                                        "breaks"), "how": "the recorded run's first steps; every series value at a step <= k is the parent's"}
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
    r["outcome"] = {"status": "completed", "provenance": "derived", "how": f"the probe script stops after its validation at {k}"}
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command="python " + " ".join(ident["argv"]))
    td = r["host"].get("train_data")
    if td:
        import math
        td["shards_read"] = math.ceil(k * ident["trajectory"]["global_batch_tokens"] / 1e8)
    r["readiness"] = C.readiness(r)
    return r


def check_record(r, reg):
    """validate.py's check for a g4n_probe record (delegated as for gen_h4's kind): its script is its parent's made single-process
    and then probe_script, it ends at PROBE_STEP with the parent's values up to there, and it is exact_only in its own group."""
    import json
    import gen_prefix as GP
    d = r["identity"]["derived_from"]
    pp = os.path.join(reg, "records", d.get("record", "") + ".json")
    if not os.path.exists(pp):
        return [f"derived_from.record {d.get('record')} not in the registry"]
    p = json.load(open(pp))
    errs = []
    try:
        want = probe_script(SP.single_process(open(os.path.join(reg, p["agent_view"]["script"])).read()))
    except (ValueError, SystemExit) as e:
        return [f"the parent script does not take the probe change: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the probe script is not its parent's made single-process with only STOP_STEP's default changed")
    k = d.get("stop_after_step")
    if k != PROBE_STEP or r["steps"] != k or any(x["steps"][-1] > k for x in r["series"].values()):
        errs.append(f"the probe must stop at step {PROBE_STEP}")
    for name, sr in r["series"].items():
        ref = GP.cut(p["series"][name], k, "")
        if sr["steps"] != ref["steps"] or sr["values"] != ref["values"]:
            errs.append(f"series {name} is not the parent's cut at step {k}")
    if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
        errs.append("a probe record must be exact_only, in a group apart from its parent's")
    return errs


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    n = 0
    for setup in SETUPS:
        rs = pool(recs, setup)
        code = script_text(rs[0], reg)
        for p in rs[1:6]:
            if script_text(p, reg) != code:
                raise SystemExit(f"{setup}: {p['identity']['record_id']} binds a different script")
        rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{rs[0]['identity']['argv'][0]}"
        C.write_text(os.path.join(reg, rel), code)
        for twin in TWINS:
            for p in twin_runs(recs, setup, twin):
                r = whole_record(p, code, rel, twin, FOLDER[setup])
                C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
                n += 1
            if setup == "soap":
                pcode = probe_script(code)
                prel = f"artifacts/mng3/scripts/{C.sha256_text(pcode)[:12]}/{rs[0]['identity']['argv'][0]}"
                C.write_text(os.path.join(reg, prel), pcode)
                pr = probe_record(twin_runs(recs, setup, twin)[0], pcode, prel, twin)
                C.write_json(os.path.join(reg, "records", pr["identity"]["record_id"] + ".json"), pr)
                n += 1
        print(f"{setup}: twins a and b, 3 runs each, on {rel}")
    print(f"g4n records: {n}")


# ---------------------------------------------------------------------------------------------------------------------------
# The tasks (gen_tasks.main calls build for each twin; finish() then adds stats, horizon and disclosure)

def answer(recs, twin):
    rows = {s: twin_runs(recs, s, twin) for s in SETUPS}
    bt = statistics.mean(C.paced_duration(r, 1) for r in rows[BASELINE])
    bl = statistics.mean(r["result"]["final_val_loss"]["value"] for r in rows[BASELINE])
    out = {}
    for s, rs in rows.items():
        if s == BASELINE:
            continue
        t = statistics.mean(C.paced_duration(r, 1) for r in rs)
        loss = statistics.mean(r["result"]["final_val_loss"]["value"] for r in rs)
        out[s] = {"time_x_baseline": round(t / bt, 4), "mean_final": round(loss, 5), "margin": round(bl - loss, 5),
                  "verdict": "rejected: slower" if t > TIME_BAR * bt else ("adopt" if loss < bl else "rejected: loss")}
    return {"baseline": {"setup": BASELINE, "mean_full_s": round(bt, 1), "mean_final": round(bl, 5)}, "setups": out}


def readme(setup, folder, script, runs):
    seeds = [r["identity"]["seed"]["value"] for r in runs]
    if seeds[0] is None:
        how = (f"Our evaluation protocol for this setup is three runs: `{script}` takes no seed, so run "
               f"`python {ROOT}/{folder}/{script}` three times.")
    else:
        how = (f"Our evaluation protocol for this setup is three runs, with seeds {seeds[0]}, {seeds[1]} and {seeds[2]}: "
               f"`python {ROOT}/{folder}/{script} --seed N`.")
    # no sentence pointing at what the script prints: the step time is the cheap check the agent has to find on its own
    #
    return (f"# {folder}\n\n`{script}` in this folder is the training script, single-process on one GPU, run from the task root. "
            f"It trains the whole schedule. {how}\n")


ADOPTION = """# Adopting an optimizer setup

setups/ has one folder per optimizer setup, each with its training script and a README. contra-soft-muon/ is our baseline;
every other folder is a candidate.

A candidate is adopted only if both hold:

- a full training run takes no more than 20% longer than the baseline's;
- its mean final validation loss over the three seeds its README names is below the baseline's mean over the baseline's three.

Each setup's README says how to run its script and which seeds the evaluation uses.
"""


def build(recs, disclosure_of, reg=None, twin="a", hidden=False, means=False, gpus=4):
    """hidden: the same task with the
    rule moved from the brief into ADOPTION.md at the task root, which names the baseline, says every other folder is a
    candidate and states both criteria. It does not say to read a run's step time or stop one early: that cheap check is the
    agent's to find (analysis: a written "check step time first" is followed and the loss goes, as in H3' and twin A).

    means: the same task with one
    sentence of the brief changed. The twins' brief ends "Report the runs you based the decision on, each setup's mean, and your
    answer", which asks for soap's mean too, so finishing soap's three full runs is what it requests (Opus's review finding; the
    runner counted 19 of 56 answers omitting soap's mean, all Claude Code). It now asks for "the means your decision rests on".
    Files, jobs, records, horizon and reference are unchanged."""
    if twin not in TWINS:
        raise ValueError(f"twin {twin!r}")
    if hidden and means:
        raise ValueError("means is a re-brief of the goal-only brief; the hidden-plan brief is not re-briefed")
    if gpus not in (4, 8):
        raise ValueError(f"gpus {gpus!r}")
    tid = f"pkg-g4n-{twin}" + ("-h" if hidden else "") + ("-means" if means else "") + ("-8gpu" if gpus == 8 else "")
    gpus_text = "four H100s" if gpus == 4 else "eight H100s"
    report = ("Report the runs you based the decision on, the means your decision rests on, and your answer." if means else
              "Report the runs you based the decision on, each setup's mean, and your answer.")
    jobs, files, shards, stats = [], [], 0, {}
    for setup in SETUPS:
        runs = twin_runs(recs, setup, twin)
        whole = [recs[f"{p['identity']['record_id']}-{TAG}{twin}"] for p in runs]
        folder, script = FOLDER[setup], runs[0]["identity"]["argv"][0]
        files.append({"path": f"{ROOT}/{folder}/{script}", "from": whole[0]["agent_view"]["script"]})
        files.append({"path": f"{ROOT}/{folder}/README.md", "content": readme(setup, folder, script, runs)})
        for i, r in enumerate(whole, 1):
            seed = r["identity"]["seed"]["value"]
            jid = f"{setup}-run{i}"
            jobs.append({"job_id": jid, "label": f"{folder}: run {i} of 3" + (f", seed {seed}" if seed is not None else ""),
                         "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                         "draw": "exact_seed" if seed is not None else "without_replacement", "seed": seed,
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": f"python {ROOT}/{folder}/{script}" + (f" --seed {seed}" if seed is not None else ""),
                         "after": [], "consumes": [], "produces": [], "setup": setup})
            shards = max(shards, r["host"]["train_data"]["shards_read"])
        stats[setup] = {"runs": [{"record_id": r["identity"]["record_id"], "seed": r["identity"]["seed"]["value"],
                                  "final_val_loss": r["result"]["final_val_loss"]["value"],
                                  "paced_1gpu_s": round(C.paced_duration(r, 1), 1),
                                  "torch": r["identity"]["trajectory"]["environment"].get("torch")} for r in whole]}
    key = answer(recs, twin)
    needed = {BASELINE} | {s for s, v in key["setups"].items() if not v["verdict"].startswith("rejected: slower")}
    # the deliverable's runs: the baseline's and every setup the speed check does not reject; SOAP's are work the rule never
    # needs, served when launched (its records are task jobs' records, so a launch is tied and scored, not unlisted)
    unneeded = [j for j in jobs if j["setup"] not in needed]
    jobs = [j for j in jobs if j["setup"] in needed]
    soap_runs = twin_runs(recs, "soap", twin)
    probe = recs[f"{soap_runs[0]['identity']['record_id']}-{TAG}{twin}probe{PROBE_STEP}"]
    probe_job = {"job_id": "soap-probe", "label": f"the reference's look at soap-f1-auxb2: its first {PROBE_STEP} steps, to its "
                 "first step_avg line", "replicate_group": probe["identity"]["replicate_group"],
                 "records": [probe["identity"]["record_id"]], "draw": "exact_seed", "seed": probe["identity"]["seed"]["value"],
                 "recorded_gpus": probe["reference"]["gpu_count"], "recorded_gpu": probe["reference"]["gpu"],
                 "batch_policy": probe["launch"]["batch_policy"]["value"], "gpu_options": [1],
                 "launch": f"python probes/soap/{soap_runs[0]['identity']['argv'][0]} --seed {probe['identity']['seed']['value']}",
                 "after": [], "consumes": [], "produces": [], "setup": "soap", "reference_only": True,
                 # the probe script is not in the agent's tree: the scripted reference writes it there first, a 0-second
                 # write (plan.py prep_write, expected_scores.prepare), then launches it
                 "prep": {"reference": probe["agent_view"]["script"],
                          "produces": [f"probes/soap/{soap_runs[0]['identity']['argv'][0]}"], "scripted_s": 0.0,
                          "how": ("the scripted reference's own copy of the probe script, written at run time so the agent's "
                                  "staged tree never holds it")}}
    jobs.append(probe_job)
    for j in jobs + unneeded:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    folders = ", ".join(sorted(f"{FOLDER[s]}/" for s in SETUPS if s != BASELINE))
    if hidden:
        files.append({"path": "ADOPTION.md", "content": ADOPTION})
    brief = ("We're deciding which optimizer setups to adopt for our small-GPT pretraining; ADOPTION.md is our protocol. Which "
             "setups should we adopt? Report the runs you based the decision on, each setup's mean, and your answer. The machine "
             f"has {gpus_text}.") if hidden else ("We're deciding which optimizer setups to adopt for our small-GPT pretraining. setups/ has one folder per setup, "
             f"each with its training script and a README: {FOLDER[BASELINE]}/ is our baseline, and the candidates are {folders}. "
             "Our rule: adopt a setup only if a full training run takes no more than 20% longer than our baseline's, and its "
             "mean final validation loss over the three seeds its README names beats the baseline's over the baseline's three. "
             f"Which setups should we adopt? {report} "
             f"The machine has {gpus_text}.")
    supported = sorted(needed)
    extra = {"box": __import__("gen_tasks").BOX8} if gpus == 8 else {}
    return {**extra,
        "task_id": tid, "rung": 3, "env_kind": "mng3", "goal_level": True, "level": "hidden_plan" if hidden else "goal_only",
        "set": "evaluation",
        "set_how": ("H6 (the analysis, 2026-10-25 04:03 MDT): pkg-g4n-" + twin + " with its rule in ADOPTION.md, "
                    "the pair at the hidden-plan level; the same records, so not a twin by records (disclosed)" if hidden else
                    "the analysis, 2026-10-24 05:46 MDT: package T2 = G4n of the second wave, G4's mechanism on "
                    "nanogpt records with a record-disjoint twin (pkg-g4n-a, pkg-g4n-b)"
                    + ("; pkg-g4n-a-means (RV-001 decided, 2026-10-29 18:20 MDT): twin A with the report sentence re-briefed to "
                       "the means the decision rests on; the same records as pkg-g4n-a, so not a twin by records" if means else "")),
        "title": (f"H6, hidden plan: which optimizer setups pass the team's adoption protocol in ADOPTION.md (twin {twin.upper()}'s "
                  "records); every run one GPU" if hidden else
                  f"Goal-level: which optimizer setups pass the team's adoption rule, from runs the agent chooses (twin {twin.upper()}"
                  + (", re-briefed: the means the decision rests on" if means else "") + "); every run one GPU"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-kvgate from the harness's G4n sketch (2026-10-24): the team's adoption rule, stated "
            "about wall time, never steps (analysis, 05:46 MDT); seeds named per setup in each README, a team's protocol "
            "(analysis, 06:30 MDT); folder names are the setups' recorded names, none ranking speed; no procedure, no deadline or "
            "time words (the study, 2026-10-20); the GPU count stated"
            + ("; H6 (2026-10-25): the folders, baseline, candidates and rule moved from the brief into ADOPTION.md, which states "
               "the rule and not the cheap check" if hidden else "")
            + ("; pkg-g4n-a-means (2026-10-29, RV-001): the report sentence changed from \"each setup's mean\" to \"the means "
               "your decision rests on\", the only change" if means else ""))},
        "files": [f for f in files] + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                                                    "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} "
                                                            "training shards"}],
        "jobs": jobs,
        "unneeded_jobs": unneeded,
        "unneeded_how": ("the rejected-on-speed setup's three full runs: not task jobs, since the deliverable does not need them "
                         "(a scripted plan runs every job, and the scorer counts a job not run as work remaining); their records "
                         "are served when launched, so a model agent that runs them is served them exactly, and the run is work "
                         "the rule never needed"),
        "candidates_stats": stats,
        "candidates_stats_how": "per setup, this twin's three recorded runs: final validation loss and paced one-GPU duration",
        "answer": {"rule": {"time_bar": TIME_BAR, "time_how": "mean paced one-GPU full-run time over the setup's three runs, against "
                                                               "the baseline's three", "loss_how": "three-run mean final val_loss below the baseline's"},
                   **key,
                   "supported_answer_needs": supported,
                   "supported_how": (
                       "a supported answer rests on full runs of the baseline and of every setup the speed check does not reject "
                       "(analysis, 05:46 MDT: skipping the rejected setups is legitimate). newton-muon's margin is "
                       f"{key['setups']['newton-muon']['margin']:+.5f} in this twin: count a call on it as supported when it is on "
                       "the side the agent's own reported means show (analysis, 06:30 MDT)")},
        "measuring_run": {"harness": "scripted", "policy": "packed", "speed": 10.0, "role": "check", "plan": "reference_plan",
                          "how": ("brief-only: plan.py runs reference_plan.instances, the needed full runs and the soap probe; the "
                                  "probe is a reference-only job under probes/, what an agent pays when it kills soap at its first "
                                  "step_avg line")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; recorded full runs converted to one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py), as H1, H3' and H4: no torch.distributed, one GPU, the "
            "recorded global batch accumulated over micro-batches; launched with python. Timings are the recorded runs "
            "converted to one GPU by gpu-mock's convert(), the step time every script prints.",
            "Recorded on other stacks than the box's torch 2.11.0, served on its image: soap-f1-auxb2 on torch 2.9.0+cu128 and "
            "aurora-proj-pruned on 2.10.0+cu128; the rest on 2.11.0. One torch per setup. Whether the agent can see the recorded "
            "version is gpu-mock's question (asked 2026-10-24 11:43Z)",
            "Recorded hardware: contra-soft-muon and soap-f1-auxb2 H100 SXM x8, radial-damp H100 SXM x4, soda-fade H200 x4, "
            "aurora-proj-pruned H200 x8, newton-muon H100 NVL x2; no A100, A40 or L40S record is drawn (gpu-tasks-h3, 11:51Z). "
            "soap-f1-auxb2's 1.76x is its own per-step cost, on the same SXM card as the baseline.",
            "The same scripts serve both twins; twin A draws each setup's first three runs in seed order, twin B the next three, "
            "so the twins share no record. newton-muon is unseeded: its runs are by record id.",
            "The reference's soap probe is the recorded run's first 125 steps on the script with STOP_STEP's default set to 125, "
            "what an agent that kills soap at its first validation pays (about 6 min). The probe script is not staged in the "
            "agent's tree: the scripted reference writes it to probes/soap/ at run time (the job's prep), then launches it."]
        + (["Angle F of the 2026-10-31 package search (the analysis 13:58 MDT): pkg-g4n-" + twin + " on an 8-GPU box. "
            "The brief states eight H100s; the jobs, records, rule and answer are twin " + twin.upper() + "'s, so its runs compare "
            "with the 4-GPU twin on the GPU count alone. The question is whether agents keep fixed lanes or batch launches below "
            "the box's width, which leaves GPUs idle while needed runs wait. It runs only on the 8-GPU image "
            "(environment.image_label), which waits on gpu-mock's device-count release"] if gpus == 8 else [])
        + (["Why this task exists (RV-001, decided by the analysis 2026-10-29 18:20 MDT under the study's "
            "delegation; record in design/EVALUATION-PLAN.md, \"RV-001 decided\"): pkg-g4n-a's brief ends \"Report the runs you "
            "based the decision on, each setup's mean, and your answer\". soap-f1-auxb2 is a setup, so the brief asks for soap's "
            "mean, and finishing soap's three full runs, the work the rule never needs, is what it requests. Opus's adversarial "
            "review raised it; the runner counted 19 of 56 answers that omit soap's mean, all Claude Code, while every Codex and "
            "Grok run reports it. This task changes only that sentence, to \"Report the runs you based the decision on, the means "
            "your decision rests on, and your answer.\" Files, jobs, records, horizon and reference are pkg-g4n-a's, so its "
            "measured host terms transfer and its runs compare with pkg-g4n-a's on the brief alone."] if means else []),
        "reference_plan": {"instances": [{"job_id": j["job_id"]} for j in jobs], "policy": "packed",
                           "rule": ("brief-only: one launch of each setup to its first validation shows its step time; soap's is "
                                    f"{key['setups']['soap']['time_x_baseline']:.2f}x the baseline's, so it is stopped there and "
                                    "rejected; the other setups' three runs and the baseline's run in full"),
                           "how": ("the needed full runs (baseline and every setup the speed check keeps) plus the soap probe, "
                                   "packed longest first. The probe stands in for the look an agent takes; the other setups' "
                                   "looks are the first minutes of their own full runs, which the reference keeps")},
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


if __name__ == "__main__":
    main()
