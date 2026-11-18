"""H1, pkg-h1-study: an optimizer study with preparation, checkpoint continuations and a decision.

usage: python3 -I build/gen_h1.py <registry>      (records and scripts; the task is gen_h1.build, called from gen_tasks)

The package search: a hidden-plan package, where the agent has to find
in the files which setups need preparing, that a continuation needs its screen's checkpoint, and which setups go on. Design
accepted by analysis at 09:50 MDT. Five setups, three seeds each, screened to step 1,000; the two with the lowest mean
validation loss at step 1,000 continue every seed to the end of their schedule from the screen's checkpoint.

- Ready setups (soda-fade, radial-damp, contra-soft-muon): each folder ships its training script, single-process
  (build/single_process.py) and with --stop-at / --resume (build/gen_segment.py segment_script). A screen is the recorded run's
  segment 0-1000; a continuation is its segment 1000-end, which loads checkpoints/seed<S>-step1000.pt.
- Setups to prepare (ema-nesterov-aurora, muonh-tuned): the folder holds what defines the setup (optimizer.py, README.md,
  hparams.md, by gen_prep.setup_files from the setup's step-1,000 prefix script) and the agent writes setups/<setup>/
  train_gpt.py from the baseline train_gpt.py, which is single-process too, and runs it three times. Its screen is the recorded
  prefix to step 1,000, served by path: the records are bound to the folder setups/<setup>/, so a script of any name there serves. These two do not win on the recorded values (3.6465 and 3.6796 against 3.6409 and 3.6426), so neither
  needs a continuation record; a continuation of one would be served by gpu-mock's modelled path, labelled.

Every record here is a recorded run (or its first 1,000 steps) converted to one GPU by gpu-mock's convert(), the timing every
one-GPU job uses. The scripts are made single-process, which the recorded runs were not: one GPU per job is a fact of the
code, so the width rule cannot pay for an agent that runs every job wide.

Seeds (torch 2.11): soda-fade 5-7, radial-damp 2/8/10, contra-soft-muon 11-13 (distinct, since checkpoints are named by
seed), ema-nesterov-aurora 8-10, muonh-tuned (unseeded) its first three runs by record id.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_prefix as GP  # noqa: E402
import gen_prep as P  # noqa: E402
import gen_segment as S  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-h1-study"
SCREEN_STEP = 1000
# A segment script names its checkpoint by seed and step only (checkpoints/seed<S>-step<N>.pt, from the task root), so the ready
# setups' seeds are distinct from one another; the prepared setups write no checkpoint
READY = {"soda-fade": [5, 6, 7], "radial-damp": [2, 8, 10], "contra-soft-muon": [11, 12, 13]}
PREP = {"ema-nesterov-aurora": [8, 9, 10], "muonh-tuned": None}       # None: unseeded, the first three runs by record id
TOP = 2
ROOT = "setups"
TAG = "h1"                     # record ids: <parent>-h1seg<a>-<b> (ready) and <prefix>-h1prep (prepared)


def ready_parents(recs, name):
    g = S.SEGMENT_SPEC[name]
    out = {}
    for r in recs.values():
        if (r["identity"]["replicate_group"] == g and not r["identity"].get("derived_from") and r["readiness"]["replay"]
                and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")):
            out[r["identity"]["seed"]["value"]] = r
    return [out[s] for s in READY[name]]


def prep_prefixes(recs, name):
    g = GP.PREFIX_SPEC[name][0]
    ps = sorted((r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
                 and r["identity"]["derived_from"]["parent_replicate_group"] == g
                 and r["identity"]["derived_from"]["stop_after_step"] == SCREEN_STEP and r["readiness"]["replay"]
                 and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")),
                key=lambda r: r["identity"]["record_id"])
    seeds = PREP[name]
    if seeds is None:
        return ps[:3]
    by = {r["identity"]["seed"]["value"]: r for r in ps}
    return [by[s] for s in seeds]


def ready_script_text(parent, reg):
    """The ready setup's script: the recorded script with save-and-resume, then made single-process (that order: segment_script
    matches the original's distributed lines)."""
    code = open(os.path.join(reg, parent["agent_view"]["script"])).read()
    return SP.single_process(S.segment_script(code))


def ready_records(parent, code, script_rel):
    """The screen (0-1000) and continuation (1000-end) records of one ready seed, by gen_segment.derive_segment on the
    single-process script."""
    out = []
    for a, b in ((0, SCREEN_STEP), (SCREEN_STEP, parent["steps"])):
        r = S.derive_segment(parent, code, a, b)
        ident = r["identity"]
        ident["record_id"] = f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"
        ident["process_model"] = {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                            "`python <script>`"}
        # where gpu-mock reads it (gpu_daemon._serving_info, revision 32c7d1a2): a "single" record paces at one GPU whatever
        # the launcher's WORLD_SIZE, so a torchrun launch of a single-process script gets no speed-up it does not have
        r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
        r["agent_view"] = dict(r["agent_view"], script=script_rel,
                               launch_command="python " + " ".join(ident["argv"]))
        r["derived_from_h1"] = {"how": "a segment of the recorded run (build/gen_segment.py derive_segment) on its single-process "
                                       "script (build/single_process.py); served values the parent's over its steps"}
        r["readiness"] = C.readiness(r)
        out.append(r)
    return out


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    n = 0
    for name in READY:
        parents = ready_parents(recs, name)
        code = ready_script_text(parents[0], reg)
        for p in parents[1:]:
            if ready_script_text(p, reg) != code:
                raise SystemExit(f"{name}: seed {p['identity']['seed']['value']} binds a different script")
        rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{parents[0]['identity']['argv'][0]}"
        C.write_text(os.path.join(reg, rel), code)
        for p in parents:
            for r in ready_records(p, code, rel):
                C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
                n += 1
        print(f"{name}: {len(parents)} seeds, screen and continuation records on {rel}")
    base = open(os.path.join(reg, P.BASE_SCRIPT)).read()
    base_sp = SP.single_process(base)
    base_rel = f"artifacts/mng3/scripts/{C.sha256_text(base_sp)[:12]}/train_gpt.py"
    C.write_text(os.path.join(reg, base_rel), base_sp)
    for name in PREP:
        ps = prep_prefixes(recs, name)
        ref_code = SP.single_process(open(os.path.join(reg, ps[0]["agent_view"]["script"])).read())
        ref_rel = f"artifacts/mng3/scripts/{C.sha256_text(ref_code)[:12]}/train_gpt.py"
        C.write_text(os.path.join(reg, ref_rel), ref_code)
        out = os.path.join(reg, "artifacts", "mng3", "prep-h1", name)
        os.makedirs(out, exist_ok=True)
        for fname, text in P.setup_files(name, ps[0], base_sp, ref_code).items():
            if fname == "README.md":
                text = prep_readme(text, name, ps)
            C.write_text(os.path.join(out, fname), text)
        # One script per setup, setups/<setup>/train_gpt.py, run with each seed (or, unseeded, three times), and every record
        # of the setup bound to the setup's DIRECTORY (gpu-mock 64d1/e035: a bound_path ending in / serves any script below
        # it), so a script under a name of the agent's own serves too (Haiku and Sonnet both named theirs, G-h1-*-L0). One
        # replicate group per setup; the seed picks the record, an unseeded launch draws the setup's next unused one. The
        # scripted preparer's copy of the reference is served at the exact stage (entry hash and argv), on any image.
        # (the study, runner and gpu-tasks, 2026-10-23 21:28Z: W0-h1-check10's per-run folders let identical scripts draw
        # one another's records, which the scorer read as unlisted.)
        bound = f"{ROOT}/{name}/"
        key = C.sha256_text(f"h1-prepared\0{name}")
        for pr in ps:
            r = copy.deepcopy(pr)
            ident = r["identity"]
            seed = ident["seed"]["value"]
            ident.update({
                "record_id": pr["identity"]["record_id"] + f"-{TAG}prep",
                "argv": ["train_gpt.py"] + (["--seed", str(seed)] if seed is not None else []),
                "argv_how": "the single-process reference at setups/<setup>/train_gpt.py: what a scripted preparer runs after "
                            "copying it there",
                "entry_hash": C.sha256_text(ref_code), "bound_path": bound, "match_policy": "path_bound",
                "match_policy_how": ("served to any script below " + bound + ", whatever its name and bytes (gpu-mock's "
                                     "path_bound rule, directory form): the agent writes that script"),
                "replicate_group": "mng3-h1prep-" + key[:12],
                "process_model": {"value": "single", "how": "build/single_process.py"},
                "derived_from": {**pr["identity"]["derived_from"], "kind": "prepared", "via": pr["identity"]["record_id"],
                                 "bound_path": bound, "change": "the step-1,000 prefix record, single-process, served by path"},
            })
            ident["trajectory"]["config_hash"] = key
            r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the reference single-process")
            r["agent_view"] = dict(r["agent_view"], script=ref_rel, launch_command=f"python {bound}train_gpt.py"
                                   + (f" --seed {seed}" if seed is not None else ""))
            r["readiness"] = C.readiness(r)
            C.write_json(os.path.join(reg, "records", ident["record_id"] + ".json"), r)
            n += 1
        print(f"{name}: {len(ps)} prepared records under {ROOT}/{name}/, files in artifacts/mng3/prep-h1/{name}/")
    print(f"h1 records: {n}; baseline {base_rel}")



# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; finish() then adds stats, horizon and disclosure)

NOTES = {
    "train_gpt.py": "the track's baseline training script (modded-nanogpt track 3, train_gpt_simple.py at 4ea6b937), single-process",
}


def readme_ready(name, script, seeds, steps):
    return (f"# {name}\n\nThis setup is ready to run: `{script}` in this folder is its training script, single-process on one GPU "
            f"(`python setups/{name}/{script} --seed <S>`, from the task root). It trains its whole {steps}-step schedule unless "
            "told to stop:\n\n"
            f"- `--stop-at N` stops after the validation at step N and saves `checkpoints/seed<S>-stepN.pt`;\n"
            "- `--resume checkpoints/seed<S>-stepN.pt` loads that checkpoint (model, every optimizer's state and step count) and "
            "carries on from step N, on the same data order and learning-rate schedule.\n\n"
            "A run stopped at step N and resumed from its checkpoint ends where an uninterrupted run would.\n\n"
            f"Seeds for this study: {', '.join(map(str, seeds))}.\n")


def prep_readme(text, name, ps):
    """gen_prep.setup_files' README for a prepared setup, adjusted for this study (runner and the analysis, 2026-10-23
    20:34Z: G-h1-haiku-L0 named its script after the reference and matched no record, because no file it could read said where
    the study keeps each run's script):
    - the reference docstring's bare first line, its own file name, is dropped, so the README does not invite naming the
      script after the reference; a reference to the track's train_gpt_simple.py reads train_gpt.py, this task's name for it;
    - where the study keeps the setup's script, and the seeds it runs, are stated as facts about the study's layout (accepted
      as bar 1 by analysis); one script per setup since 21:28Z (the records are bound to the setup's folder, so a script
      the agent names otherwise is served too)."""
    import re
    lines = text.splitlines()
    if len(lines) > 2 and lines[0].startswith("# ") and re.fullmatch(r"[\w.-]+\.py", lines[2].strip()) and not lines[1].strip():
        del lines[1:3]
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
    text = text.replace("torchrun", "python").replace("`train_gpt_simple.py`", "`train_gpt.py`")
    seeds = [r["identity"]["seed"]["value"] for r in ps]
    if seeds[0] is not None:
        where = (f"This study keeps its script at `{ROOT}/{name}/train_gpt.py` and runs it with seeds "
                 + ", ".join(map(str, seeds[:-1])) + f" and {seeds[-1]}, each launched with `--seed` and its seed.")
    else:
        where = (f"This study keeps its script at `{ROOT}/{name}/train_gpt.py` and runs it {len(ps)} times "
                 "(it takes no seed).")
    return text.rstrip("\n") + "\n\n" + where + "\n"


def recorded_of(r):
    return {"recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"]}


def build(recs, disclosure_of, reg=None):
    jobs, files, cands = [], [{"path": "train_gpt.py", "from": None}], []
    base_sp = None
    shards = 0
    for name, seeds in READY.items():
        screens, conts, sname = [], [], None
        for s in seeds:
            scr = next(r for r in recs.values() if r["identity"]["record_id"].endswith(f"-{TAG}seg0-{SCREEN_STEP}")
                       and r["identity"]["derived_from"]["parent_replicate_group"] == S.SEGMENT_SPEC[name]
                       and r["identity"]["seed"]["value"] == s)
            con = next(r for r in recs.values() if r["identity"]["record_id"] == scr["identity"]["record_id"].replace(
                f"-{TAG}seg0-{SCREEN_STEP}", f"-{TAG}seg{SCREEN_STEP}-{scr['identity']['derived_from']['parent_steps']}"))
            sname = scr["identity"]["argv"][0]
            path = f"{ROOT}/{name}/{sname}"
            jid = f"screen-{name}-seed{s}"
            jobs.append({"job_id": jid, "label": f"screen {name} seed {s}: steps 1-{SCREEN_STEP}, saving a checkpoint",
                         "replicate_group": scr["identity"]["replicate_group"], "records": [scr["identity"]["record_id"]],
                         "draw": "exact_seed", "seed": s, **recorded_of(scr),
                         "batch_policy": scr["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": f"python {path} --seed {s} --stop-at {SCREEN_STEP}", "after": [], "consumes": [],
                         "produces": [f"checkpoints/seed{s}-step{SCREEN_STEP}.pt"]})
            screens.append(jid)
            conts.append((s, con))
            shards = max(shards, con["host"]["train_data"]["shards_read"])
        files.append({"path": f"{ROOT}/{name}/{sname}", "from": recs[jobs[-1]["records"][0]]["agent_view"]["script"]})
        files.append({"path": f"{ROOT}/{name}/README.md", "content": readme_ready(
            name, sname, seeds, conts[0][1]["identity"]["derived_from"]["parent_steps"])})
        cands.append({"variant": name, "screen_jobs": screens, "seeds": seeds,
                      "records": [j["records"][0] for j in jobs if j["job_id"] in screens],
                      "confirm_records": [c["identity"]["record_id"] for _, c in conts],
                      "confirm_by_seed": {str(s): {"record": c["identity"]["record_id"],
                                                   "launch": f"python {ROOT}/{name}/{sname} --seed {s} --resume "
                                                             f"checkpoints/seed{s}-step{SCREEN_STEP}.pt"} for s, c in conts},
                      "kind": "ready"})
    for name, seeds in PREP.items():
        ps = sorted((r for r in recs.values() if r["identity"]["record_id"].endswith(f"-{TAG}prep")
                     and r["identity"]["derived_from"]["parent_replicate_group"] == GP.PREFIX_SPEC[name][0]),
                    # the order the per-run folders gave (seed10, seed8, seed9; run1-3 by record id), so the packed plan and
                    # W0-h1-check10's prediction keep their job order
                    key=lambda r: (f"seed{r['identity']['seed']['value']}" if r["identity"]["seed"]["value"] is not None
                                   else r["identity"]["record_id"]))
        screens = []
        script = f"{ROOT}/{name}/train_gpt.py"
        for k, r in enumerate(ps, 1):
            s = r["identity"]["seed"]["value"]
            sub = f"seed{s}" if s is not None else f"run{k}"
            jid = f"screen-{name}-{sub}"
            # one script per setup: a seeded setup's jobs differ by --seed; an unseeded setup's three are the same launch,
            # each listing one record, which the daemon draws without replacement and the scorer ties by the record served
            jobs.append({"job_id": jid, "label": f"screen {name} {sub}: its script written by the agent, steps 1-{SCREEN_STEP}",
                         "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                         "draw": "exact_seed" if s is not None else "without_replacement", "seed": s, **recorded_of(r),
                         "batch_policy": r["launch"]["batch_policy"]["value"],
                         "gpu_options": [1], "launch": f"python {script}" + (f" --seed {s}" if s is not None else ""),
                         "after": [], "consumes": [], "produces": [],
                         "prep": {"kind": "code", "paths": [f"{ROOT}/{name}/"], "produces": [script],
                                  "reference": r["agent_view"]["script"], "scripted_s": 300.0, "horizon_s": 900.0,
                                  "how": "the agent writes produces[0] from train_gpt.py and setups/<setup>/, one script for "
                                         "the setup's three runs; a scripted preparer copies the reference (the setup's "
                                         "single-process step-1,000 script, never in the agent's tree) there after scripted_s "
                                         "seconds, before each of the three launches (the same bytes each time)"}})
            screens.append(jid)
            shards = max(shards, r["host"]["train_data"]["shards_read"])
        for fname in ("README.md", "optimizer.py", "hparams.md"):
            files.append({"path": f"{ROOT}/{name}/{fname}", "from": f"artifacts/mng3/prep-h1/{name}/{fname}"})
        cands.append({"variant": name, "screen_jobs": screens, "seeds": seeds,
                      "records": [j["records"][0] for j in jobs if j["job_id"] in screens], "kind": "prepare",
                      "confirm_records": [], "confirm_how": "no continuation record: its recorded screen values do not put it "
                                                            "among the two; a continuation would be served on gpu-mock's modelled "
                                                            "path, labelled match: none"})
    base_rel = next(r for r in recs.values() if r["identity"]["record_id"].endswith(f"-{TAG}prep"))["agent_view"]["script"]
    files[0] = {"path": "train_gpt.py", "from": "artifacts/mng3/scripts/" + _base_hash(reg) + "/train_gpt.py",
                "how": NOTES["train_gpt.py"]}
    screens_all = [j["job_id"] for j in jobs]
    follow, pool = [], [r for c in cands for r in c["confirm_records"]]
    for slot in range(1, TOP + 1):
        for k in range(1, 4):
            jid = f"continue-pick{slot}-run{k}"
            follow.append(jid)
            jobs.append({"job_id": jid, "label": f"continue the #{slot} setup's run {k} from its step-{SCREEN_STEP} checkpoint",
                         "records": None, "bind": {"decision": "pick-two", "slot": slot, "run": k},
                         "launch": f"python {ROOT}/<setup>/<script> --seed <its seed> --resume checkpoints/seed<its seed>-step{SCREEN_STEP}.pt",
                         "draw": "the bound setup's continuation record for that seed", "seed": None,
                         **recorded_of(recs[pool[0]]), "batch_policy": "fixed_global_grow_accumulation",
                         "gpu_options": [1], "after": screens_all, "consumes": [], "produces": [], "candidate_pool": pool})
    for j in jobs:
        if j.get("bind"):
            # gen_tasks.disclosure_of gives a single-process pool its paced one-GPU duration; finish() fills the rest
            j["disclosure"] = disclosure_of(j["candidate_pool"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}",
                                            "over every ready setup's continuation, since the winners are not known in advance")
    n_setups = len(READY) + len(PREP)
    brief = ("We're comparing five optimizer setups for our small-GPT pretraining. setups/ has one folder per setup; train_gpt.py "
             "is our baseline training script, and data/fineweb10B/ holds the tokenised data. Some setups' folders have a ready "
             "training script; the others hold only what defines the setup (its optimizer code, hyperparameters and a README), "
             "and their script still has to be written from train_gpt.py. We want to know which two setups lead at step 1,000 "
             "of their schedules, by the mean validation loss over three runs each, and then where those two end up: carry "
             "their six runs on from step 1,000 to the end of their schedules. Report every run's validation loss at step "
             "1,000, the two setups you carried on, and their six final validation losses. The machine has four H100s.")
    return {
        "task_id": TASK_ID, "rung": 2, "env_kind": "mng3", "level": "hidden_plan",
        "title": "H1: five optimizer setups, three runs each to step 1,000 (two of the setups prepared first), then the best two "
                 "carried on to the end from their checkpoints; every job one GPU",
        "brief": {"text": brief, "provenance": ("written by gpu-tasks for the package search (the analysis, "
                                                "2026-10-23): the study, not the runs; which setups need preparing, the seeds, "
                                                "the stop-and-resume mechanics and the continuation's dependency are in the files; "
                                                "no deadline or time words; the GPU count stated")},
        "files": [_with_sha(f, reg) for f in files] + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "decisions": [{"decision_id": "pick-two", "after": screens_all,
                       "rule": f"the two setups with the lowest mean validation loss at step {SCREEN_STEP} over their three runs; "
                               "ties broken by name",
                       "rule_spec": {"metric": "final_val_loss", "aggregate": "mean", "direction": "min", "top": TOP},
                       "observable": f"the validation loss each screen prints at step {SCREEN_STEP}, as served",
                       "launches": follow, "slots": TOP, "shape_independent_of_outcome": False,
                       "shape_how": ("a ready setup's three continuations are one-GPU resumes of 50-56 minutes; a prepared setup "
                                     "has no continuation record, so if one were picked its continuations would be modelled. "
                                     "The recorded values pick two ready setups"),
                       "candidates": cands}],
        "measuring_run": {"harness": "scripted", "policy": "packed", "speed": 10.0, "role": "check",
                          "plan": "reference_plan", "terms_from": ["pkg-prep-p6", "pkg-p20-par"],
                          "how": ("host terms transferred from the preparation and P20 packages (same scripts made "
                                  "single-process, same image), checked by a speed-10 scripted run of reference_plan")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0, converted to one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py): no torch.distributed, one GPU (cuda:0), the recorded global "
            "batch accumulated over micro-batches; launched with python. One GPU per job is a fact of the code, not a brief "
            "instruction (analysis, 2026-10-23 16:08Z). The timings are the records converted to one GPU by gpu-mock's convert().",
            "Ready setups' screens and continuations are one recorded run each, cut at step 1,000 by the save-and-resume change "
            "(build/gen_segment.py); a continuation loads checkpoints/seed<S>-step1000.pt and fails in torch.load if its screen "
            "has not saved it.",
            "Prepared setups' screens are the recorded run's first 1,000 steps, served by path to whatever script the agent "
            "writes under setups/<setup>/ (the README names setups/<setup>/train_gpt.py; the records are bound to the folder, "
            "so another name serves too, on gpu-mock 64d1/e035 and later), by --seed for ema-nesterov-aurora and in turn for "
            "the unseeded muonh-tuned; their READMEs describe the reference run, including its stop after the validation at "
            "step 1,000.",
            f"Recorded mean validation loss at step {SCREEN_STEP}: soda-fade 3.6409, radial-damp 3.6426, ema-nesterov-aurora "
            "3.6465, contra-soft-muon 3.6539, muonh-tuned 3.6796. The two picked are ready setups; second and third differ by "
            "0.0039, 5.8 standard errors of the difference.",
            "If an agent launches a single-process script with torchrun --nproc_per_node>1, gpu-mock's pacing may read the "
            "launcher's WORLD_SIZE (asked of gpu-mock 2026-10-23 15:40Z); such launches are flagged in the run's record."],
        "reference_plan": _reference_plan(jobs, cands),
    }


def _with_sha(f, reg):
    """A files[] entry staged from a registry artifact carries the artifact's sha256 (runner and gpu-tasks, 2026-10-23 20:49Z:
    the READMEs were fixed in the artifacts, the task file's bytes did not move, and run_one's staged-tree cache, keyed by the
    task file's hash, kept serving the old READMEs on four machines). The hash moves the task file whenever an artifact does,
    and build/stage_task.py refuses an artifact whose bytes differ from it."""
    if not str(f.get("from") or "").startswith("artifacts/"):
        return f
    return {**f, "sha256": C.sha256_bytes(open(os.path.join(reg, f["from"]), "rb").read())}


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


def _base_hash(reg):
    base = open(os.path.join(reg, P.BASE_SCRIPT)).read()
    return C.sha256_text(SP.single_process(base))[:12]


def _reference_plan(jobs, cands):
    """The brief's rule as a scripted plan (bar 2): screen every setup (preparing the two that need it), then continue the two
    the screens rank first by mean validation loss at step 1,000. The recorded values put soda-fade and radial-damp there, so
    the plan's continuations are theirs; a scripted agent that read the served screens would launch the same six."""
    ranked = sorted((c for c in cands if c["kind"] == "ready"), key=lambda c: c["variant"])
    return {"rule": "screen all, rank by mean val loss at step 1,000, continue the top two",
            "how": "the brief's rule; plan.py binds the six continuation slots to the recorded winners (decisions.recorded_outcome)"}


if __name__ == "__main__":
    main()
