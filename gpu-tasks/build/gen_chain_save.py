"""D8, pkg-chain-save-8gpu: pkg-chain-side's chain at eight GPUs, with a real checkpoint-save failure on its first piece
(gpu-tasks-dsave, 2026-11-02; package search, angle D x C, notes/2026-11-01-late-failure-shapes.md shape 3; design and
checks in notes/2026-11-02-chain-save-failure.md).

usage: python3 -I build/gen_chain_save.py <registry>      (derives the chain's records and script; the task is
                                                           gen_chain_save.build(recs, disclosure_of, reg), which derives them
                                                           too if they are missing, and returns the task)

The package is pkg-chain-side (build/gen_chain_side.py) on gen_tasks.BOX8, with one change to the chain's script, and the
same 36 side runs on the same records. The chain's script is pkg-chain-side's (the recorded circuit-muon script with
build/gen_segment.py's save-and-resume change, made single-process) without the save's directory creation: the line
`os.makedirs("checkpoints", exist_ok=True)` before `torch.save` is removed, and nothing else. The task tree ships no
checkpoints/ directory. So the chain's first piece trains its 625 steps, prints its step-625 validation, and then torch.save
raises for real, in the box as on hardware:

    RuntimeError: Parent directory checkpoints does not exist.

That is torch's own zip writer (torch.serialization._open_zipfile_writer_file), the same text as the registry's recorded TRL
DPO failure (trl-dpo-h100-sxmx1-c636cfd6-s3-2cfx11s4: "Parent directory dpo_anthropic_hh/tmp-checkpoint-500 does not exist."
at its first checkpoint save). gpu-mock serves nothing here: its patch's torch.save wrapper (cuda_hooks._fix_torch_save) maps
fake-GPU tensors to plain ones and calls the real torch.save, so the error, the traceback and the exit are the script's own.
The fix is the one a researcher makes from the error: create the directory (`mkdir checkpoints`) or add the makedirs back,
and relaunch the same line; the next save finds its directory. No flag, no trigger, nothing the mock keys on. The missing
directory is ours, as C's --micro-batch 256 is, and the task's notes disclose it.

Why the first piece: a missing output directory surfaces at the first checkpoint save of a fresh tree and nowhere else (once
anyone makes it, it exists), and the first piece is the head of the chain, which at eight GPUs is nearly the whole schedule.
The failed attempt runs its whole 625 steps (f = 1); pieces 2-5 wait on the checkpoint it did not write.

Records, all derived here and none edited by hand: the chain's pieces, circuit-muon seed 6's recorded run, steps a+1..b, on
the changed script (build/gen_segment.py derive_segment), record ids <parent>-dsvseg<a>-<b>, kind `segment` with
derived_from.variant `save_dir_not_created` (validate.py delegates them to check_record below), exact_only with a bound_path
(longrun/circuit-muon/<script>) so an agent that edits the script, the makedirs added, is served the same records through
gpu-mock's path_exact stage; plus the whole run in one launch (no --stop-at, so no save), listed by no job. The side records are
pkg-chain-side's own `-csd` records, unchanged (gen_chain_side.derive writes them when they are missing). No record carries a
failure: the failure is not served.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_chain_side as GCS  # noqa: E402
import gen_segment as S  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-chain-save-8gpu"
TAG = "dsv"
VARIANT = "save_dir_not_created"
CHAIN_ROOT = "longrun"                   # a root no other record binds (gpu-tasks AGENTS.md, "Give a new package's script folders
                                         # a root no other task binds"); the side folders stay pkg-chain-side's baselines/
SAVE_DIR = "checkpoints"
MAKEDIRS_RE = re.compile(r'^[ \t]*os\.makedirs\("checkpoints", exist_ok=True\)[ \t]*\n', re.M)
FAILING_PIECE = 1
ERROR = "RuntimeError: Parent directory checkpoints does not exist."
ERROR_HOW = ("torch's own text for a torch.save whose parent directory is missing (torch.serialization's zip writer): checked "
             "on torch 2.13.0 on the controller by gpu-tasks-dsave, 2026-11-02 (`torch.save({...}, 'checkpoints/seed6-step625.pt')` "
             "with no checkpoints/ raises exactly this; after os.makedirs the same call saves); the registry's TRL DPO record "
             "(torch of 2024-01) carries the same wording; in the box image (torch 2.11.0+cu130) gpu-mock ran "
             "`torch.save({}, 'nodir/x.pt')` and got `RuntimeError: Parent directory nodir does not exist.` (gpu-mock, "
             "2026-11-02 18:49Z)")
MIN_JOB_MIN = GCS.MIN_JOB_MIN


def drop_makedirs(code):
    """The one change: the save's `os.makedirs("checkpoints", exist_ok=True)` removed. Refuses a script where that line is not
    exactly one, or where removing it leaves anything else creating the directory."""
    m = list(MAKEDIRS_RE.finditer(code))
    if len(m) != 1:
        raise ValueError(f"drop_makedirs: {len(m)} `os.makedirs(\"checkpoints\", exist_ok=True)` lines, expected 1")
    out = code[:m[0].start()] + code[m[0].end():]
    if re.search(r"(makedirs|mkdir)\([^)]*checkpoints", out) or "Path(\"checkpoints\")" in out:
        raise ValueError("drop_makedirs: something else still creates checkpoints/")
    compile(out, "chain-save", "exec")
    return out


def chain_code(parent, reg):
    """pkg-chain-side's chain script (gen_chain_side.chain_code: segment change, then single-process) without the makedirs."""
    return drop_makedirs(GCS.chain_code(parent, reg))


def chain_script_path(parent):
    variant = GCS.CHAIN[0]
    return f"{CHAIN_ROOT}/{variant}/{parent['identity']['argv'][0]}"


def chain_records(parent, code, script_rel, bounds):
    out = []
    bound = chain_script_path(parent)
    for a, b in zip(bounds, bounds[1:]):
        r = S.derive_segment(parent, code, a, b)
        ident = r["identity"]
        ident["record_id"] = f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"
        ident["process_model"] = {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                            "`python <script>`"}
        ident["bound_path"] = bound
        ident["match_policy_how"] = (ident["match_policy_how"] + "; bound_path names the script's place in the task tree, so an "
                                     "edited copy there (the makedirs added back, say) with this segment's arguments is served "
                                     "this record by gpu-mock's path_exact stage, not the modelled path")
        ident["derived_from"] = dict(ident["derived_from"], variant=VARIANT, change=(
            ident["derived_from"]["change"] + "; then single-process (build/single_process.py); then the save's "
            "`os.makedirs(\"checkpoints\", exist_ok=True)` removed (build/gen_chain_save.py drop_makedirs), so the first "
            "save of a tree without checkpoints/ raises in torch.save, for real"))
        r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
        r["agent_view"] = dict(r["agent_view"], script=script_rel,
                               launch_command=f"python {bound} " + " ".join(ident["argv"][1:]))
        if b < parent["steps"]:
            r["artefacts"]["checkpoint_save_needs_dir"] = {
                "value": SAVE_DIR + "/", "provenance": "derived",
                "how": ("the script's torch.save writes checkpoints/seed<S>-step<N>.pt and does not create checkpoints/ "
                        f"(build/gen_chain_save.py); without it the save raises `{ERROR}` after the step-{b - a} validation, "
                        "for real, and the record's draw is kept (the launch paced every step)")}
        r["readiness"] = C.readiness(r)
        out.append(r)
    return out


def check_design(recs):
    """pkg-chain-side's checks (every job a distinct recorded run, the chain's run serving no side job), and the chain's root
    bound by no record outside this package."""
    parent = GCS.check_design(recs)
    mine = chain_script_path(parent)
    for r in recs.values():
        ident = r["identity"]
        for bp in (ident.get("bound_path"), (ident.get("derived_from") or {}).get("bound_path")):
            if bp and bp.split("/")[0] == CHAIN_ROOT and not ident["record_id"].startswith(parent["identity"]["record_id"] + f"-{TAG}seg"):
                raise SystemExit(f"{ident['record_id']} binds {bp}, under this package's root {CHAIN_ROOT}/ ({mine})")
    return parent


def derive(recs, reg):
    """Write the chain's script and records into <reg>, and add the records to recs; pkg-chain-side's side records too when they
    are missing (gen_chain_side.derive, idempotent). Idempotent: write_json and write_text leave an unchanged file alone."""
    if any(f"{p['identity']['record_id']}-{GCS.TAG}" not in recs for setup in GCS.SIDE for p in GCS.side_prefixes(recs, setup)):
        GCS.derive(recs, reg)
    parent = check_design(recs)
    code = chain_code(parent, reg)
    rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{parent['identity']['argv'][0]}"
    C.write_text(os.path.join(reg, rel), code)
    _, _, bounds = GCS.CHAIN
    n = 0
    for r in chain_records(parent, code, rel, bounds):
        mins = C.paced_duration(r, 1) / 60
        if mins < MIN_JOB_MIN:
            raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, under {MIN_JOB_MIN:g}")
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        recs[r["identity"]["record_id"]] = r
        n += 1
    # the whole run in one launch: no --stop-at, so the script never saves and never fails; listed by no job, served exactly so
    # an agent that runs it whole is served the run (the script's bytes are this package's own, so no other record has its key)
    (w,) = chain_records(parent, code, rel, (0, parent["steps"]))
    key = (w["identity"]["entry_hash"], tuple(w["identity"]["argv"]))
    clash = [x["identity"]["record_id"] for x in recs.values() if x["identity"]["record_id"] != w["identity"]["record_id"]
             and (x["identity"].get("entry_hash"), tuple(x["identity"].get("argv") or ())) == key]
    if clash:
        raise SystemExit(f"the whole-run launch is already served by {clash}")
    C.write_json(os.path.join(reg, "records", w["identity"]["record_id"] + ".json"), w)
    recs[w["identity"]["record_id"]] = w
    return n + 1


def check_record(r, reg):
    """validate.py's delegation for derived_from {kind: segment, variant: save_dir_not_created}: the script is the parent's with
    exactly the documented changes (segment_script, single_process, drop_makedirs), and the record is a segment as
    validate.py's own segment branch requires it, plus its bound_path."""
    import json
    d = r["identity"]["derived_from"]
    errs = []
    ppath = os.path.join(reg, "records", d["record"] + ".json")
    if not os.path.exists(ppath):
        return [f"derived_from.record {d['record']} not in the registry"]
    p = json.load(open(ppath))
    try:
        want = drop_makedirs(SP.single_process(S.segment_script(open(os.path.join(reg, p["agent_view"]["script"])).read())))
    except ValueError as e:
        return [f"the parent script does not take the documented changes: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the script is not its parent's with only the save-and-resume change, single_process and drop_makedirs")
    if (r["identity"].get("process_model") or {}).get("value") != "single":
        errs.append("a save_dir_not_created segment must be single-process")
    a, b = d["steps"][0] - 1, d["steps"][1]
    if r["steps"] != b - a or any(s["steps"][-1] > b - a for s in r["series"].values()):
        errs.append(f"steps or a series runs past the segment's {b - a} steps")
    if r["identity"]["argv"] != S.segment_argv(p["identity"]["argv"][0], p["identity"]["seed"]["value"], a, b, p["steps"]):
        errs.append("argv is not the segment's documented launch")
    if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
        errs.append("a segment record must be exact_only, in a group apart from its parent's")
    if r["identity"].get("bound_path") != f"{CHAIN_ROOT}/{GCS.CHAIN[0]}/{p['identity']['argv'][0]}":
        errs.append(f"bound_path must be the chain script's place in the task tree, under {CHAIN_ROOT}/")
    if r.get("failures"):
        errs.append("the save failure is real, not served: the record carries no failures")
    return errs


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    print(f"chain-save records: {derive(recs, reg)}")


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; complete_task then adds stats, horizon and disclosure)

def chain_readme(variant, script, seed, bounds, steps):
    stops = bounds[1:-1]
    first = f"python {CHAIN_ROOT}/{variant}/{script} --seed {seed} --stop-at {stops[0]}"
    later = ", ".join(f"{b:,}" for b in stops[1:-1]) + f" and {stops[-1]:,}"
    return (f"# {variant}\n\n`{script}` in this folder is {variant}'s training script, a {steps:,}-step schedule, single-process "
            "on one GPU, run from the task root. We train the full run in pieces. A piece stops after the validation at its "
            "`--stop-at` step and saves `checkpoints/seed<S>-step<N>.pt`; the next piece is the same command with `--resume` "
            "pointing at that checkpoint and `--stop-at` at its own step, and the last piece has no `--stop-at` and trains to "
            "the end of the schedule. A piece started before the checkpoint it resumes from exists fails in torch.load before "
            "its first step. A resumed run validates first at the step it resumes from, and ends where an uninterrupted run "
            f"would.\n\nThe write-up's run is seed {seed}. Its first piece is `{first}`. After {stops[0]:,} it stops at {later}, "
            f"and the last piece runs to {steps:,}.\n")


def writeup(side_rows, variant, seed, bounds, steps):
    return GCS.writeup(side_rows, variant, seed, bounds, steps).replace(f"{GCS.CHAIN_ROOT}/{variant}/", f"{CHAIN_ROOT}/{variant}/")


def build(recs, disclosure_of, reg=None):
    import gen_tasks as GT
    reg = reg or GT.REG_DIR
    variant, seed, bounds = GCS.CHAIN
    parent = GCS.chain_parent(recs)
    if any(f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}" not in recs for a, b in zip(bounds, bounds[1:])) or any(
            f"{p['identity']['record_id']}-{GCS.TAG}" not in recs for setup in GCS.SIDE for p in GCS.side_prefixes(recs, setup)):
        derive(recs, reg)
    check_design(recs)
    jobs, files, side_rows, shards = [], [], [], 0
    for setup in GCS.SIDE:
        ps = GCS.side_prefixes(recs, setup)
        whole = [recs[f"{p['identity']['record_id']}-{GCS.TAG}"] for p in ps]
        script = ps[0]["identity"]["argv"][0]
        seeds = [r["identity"]["seed"]["value"] for r in whole]
        files.append({"path": f"{GCS.SIDE_ROOT}/{setup}/{script}", "from": whole[0]["agent_view"]["script"]})
        files.append({"path": f"{GCS.SIDE_ROOT}/{setup}/README.md", "content": GCS.side_readme(setup, script, seeds)})
        for r in whole:
            s = r["identity"]["seed"]["value"]
            jobs.append({"job_id": f"{setup}-seed{s}", "label": f"{setup}, seed {s}: steps 1-{GCS.STEP}",
                         "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                         "draw": "exact_seed", "seed": s,
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": f"python {GCS.SIDE_ROOT}/{setup}/{script} --seed {s}",
                         "after": [], "consumes": [], "produces": [], "role": "side"})
            side_rows.append((setup, s))
            shards = max(shards, r["host"]["train_data"]["shards_read"])
    base = parent["identity"]["argv"][0]
    pieces = [recs[f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"] for a, b in zip(bounds, bounds[1:])]
    files.append({"path": f"{CHAIN_ROOT}/{variant}/{base}", "from": pieces[0]["agent_view"]["script"]})
    files.append({"path": f"{CHAIN_ROOT}/{variant}/README.md",
                  "content": chain_readme(variant, base, seed, bounds, parent["steps"])})
    files.append({"path": "WRITEUP.md", "content": writeup(side_rows, variant, seed, bounds, parent["steps"])})
    if any(f["path"].split("/")[0] == SAVE_DIR for f in files):
        raise SystemExit(f"the task tree must not ship {SAVE_DIR}/")
    prev, n = None, len(pieces)
    for i, (r, (a, b)) in enumerate(zip(pieces, zip(bounds, bounds[1:])), 1):
        jid = f"{variant}-part{i}"
        launch = f"python {CHAIN_ROOT}/{variant}/{base} " + " ".join(r["identity"]["argv"][1:])
        j = {"job_id": jid, "label": f"{variant}, seed {seed}, piece {i} of {n}: steps {a + 1}-{b}",
             "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
             "draw": "exact_seed", "seed": seed,
             "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
             "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
             "launch": launch,
             "after": [prev] if prev else [], "consumes": [f"checkpoints/seed{seed}-step{a}.pt"] if a else [],
             "produces": [f"checkpoints/seed{seed}-step{b}.pt"] if b < parent["steps"] else [],
             "role": "chain", "chain": f"{variant}-seed{seed}", "piece": i}
        if i == FAILING_PIECE:
            j["fails_as_briefed"] = {
                "mode": "crash", "real": True, "served": False, "at": "checkpoint_save", "step": b - a,
                "after": f"its step-{b - a} validation (the parent's step {b}), which prints first",
                "exception": {"type": "RuntimeError", "message": "Parent directory checkpoints does not exist.",
                              "message_re": r"^Parent directory \S*checkpoints/? does not exist\.?$",
                              "frame_file_suffix": "torch/serialization.py", "how": ERROR_HOW},
                # at the top level too, where the runner's scorer rule reads it (runner, 2026-11-02 21:09Z: `fails_as_briefed:
                # {mode, real, served, exception: {type, message_re}, frame_file_suffix, fix}`)
                "frame_file_suffix": "torch/serialization.py",
                "exit_code": 1,
                "cause": ("the script's torch.save writes checkpoints/seed6-step625.pt and does not create checkpoints/; the "
                          "task tree ships none (ours, disclosed: build/gen_chain_save.py drop_makedirs)"),
                "fix": ("create the directory (`mkdir checkpoints`, or `os.makedirs` added before the save) and relaunch the "
                        "same line; the relaunch is served this record again (gpu-mock exact_seed_reused, or path_exact for an "
                        "edited script) and saves"),
                "fixed_launch": launch,
                "how": "real, not served: the record carries no failure (notes/2026-11-02-chain-save-failure.md)"}
        jobs.append(j)
        prev = jid
        shards = max(shards, r["host"]["train_data"]["shards_read"])
    for j in jobs:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    dur = {j["job_id"]: C.paced_duration(recs[j["records"][0]], 1) for j in jobs}
    chain_s = sum(dur[j["job_id"]] for j in jobs if j["role"] == "chain")
    side_s = sum(dur[j["job_id"]] for j in jobs if j["role"] == "side")
    n_side = sum(1 for j in jobs if j["role"] == "side")
    fail_s = dur[f"{variant}-part{FAILING_PIECE}"]
    gpus = GT.BOX8["gpus"]
    brief = (f"We're finishing our write-up on {variant}, an optimizer for our small-GPT pretraining; WRITEUP.md is the draft, "
             "and its blanks are the numbers we still need. Make the runs it needs on this machine, fill in every blank, and "
             "send back the finished WRITEUP.md. The machine has eight H100s.")
    return {
        "box": GT.BOX8,
        "task_id": TASK_ID, "rung": 2, "env_kind": "mng3", "level": "hidden_plan",
        "set": "package-search", "set_how": ("package search, second round (experiments/package-search/AGENTS.md, angles D and "
                                            "C, gpu-tasks-dsave 2026-11-02); not in the paper's benchmark until the study "
                                            "says so"),
        "title": (f"Angle D x C at eight GPUs: a write-up's long run, {variant} carried to the end of its schedule in {n} pieces "
                  f"from saved checkpoints, beside {n_side} independent baseline runs to step {GCS.STEP:,}; the first piece's "
                  "checkpoint save fails for real on a missing directory; every job one GPU"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-chain for angle D (2026-10-31) and carried by gpu-tasks-dsave (2026-11-02) with the GPU count "
            "changed: the request, not the runs. The chain (longrun/circuit-muon/README.md), the side work (WRITEUP.md's table, "
            "baselines/<setup>/README.md) and the failure (the error torch prints) are found in the files and the run; no "
            "procedure, no deadline or time words, the GPU count stated, nothing that names a test")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "shape": {"chain_jobs": n, "side_jobs": n_side, "chain_paced_1gpu_s": round(chain_s, 1),
                  "side_paced_1gpu_s": round(side_s, 1), f"side_over_{gpus - 1}_gpus_s": round(side_s / (gpus - 1), 1),
                  "failing_jobs": [f"{variant}-part{FAILING_PIECE}"], "failed_attempt_paced_1gpu_s": round(fail_s, 1),
                  "chain_with_one_rerun_s": round(chain_s + fail_s, 1),
                  "how": (f"paced one-GPU durations (common.paced_duration). At {gpus} GPUs the chain is longer than the side "
                          f"work spread over the other {gpus - 1}, so it is the critical path; the first piece's failed attempt "
                          "(its whole run, the save failing at its end) and every second until its relaunch move the end of "
                          "the run. Bar 2 and the habits: survey/chain-save-refs-2026-11-02/")},
        "failure_design": {
            "failing_job": f"{variant}-part{FAILING_PIECE}", "error": ERROR, "real": True, "served": False,
            "cause": ("ours: the chain script's save does not create checkpoints/ (pkg-chain-side's script with its "
                      "`os.makedirs(\"checkpoints\", exist_ok=True)` removed) and the task tree has none"),
            "fix": "mkdir checkpoints (or the makedirs back in the script), then the same line again",
            "why_this_piece": ("a missing output directory surfaces at a fresh tree's first checkpoint save and nowhere else; the "
                               "first piece is the head of the critical path"),
            "w0": ("the scripted planner never relaunches, so its piece 1 exits 1 and pieces 2-5 fail in torch.load: a W0 reads "
                   "INVALID by design and checks serving only (gpu-tasks AGENTS.md); bar 2 is computed outside expected_scores "
                   "(survey/chain-save-refs-2026-11-02/)"),
            "how": "notes/2026-11-02-chain-save-failure.md"},
        "measuring_run": {"harness": "scripted", "policy": "packed_cp", "speed": 10.0, "role": "check", "plan": "reference_plan",
                          "terms_from": ["pkg-h3-chains", "pkg-p20-par"],
                          "how": ("host terms transferred (build/transfer_host_terms.py --from, first match wins) from pkg-h3-"
                                  "chains for the chain script (pkg-h3-chains' circuit-muon single-process segment script, of "
                                  "which this one differs by the removed makedirs line only; matched by basename) and from "
                                  "pkg-p20-par for the two baselines' step-875 prefix scripts, as for pkg-chain-side. A speed-10 "
                                  "scripted run of reference_plan checks serving only: its piece 1 fails as briefed")},
        "sources": [f"modded-nanogpt track 3, commit 4ea6b937; {n_side + 1} recorded runs on torch 2.11.0 on H100s, converted to "
                    "one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py): no torch.distributed, one GPU (cuda:0), the recorded global "
            "batch accumulated over micro-batches; launched with python. One GPU per job is a fact of the code. Timings are the "
            "records converted to one GPU by gpu-mock's convert().",
            f"The chain is {variant} seed {seed}'s recorded run cut at steps " + ", ".join(f"{b:,}" for b in bounds[1:-1])
            + " by the save-and-resume change (build/gen_segment.py), as in pkg-chain-side; a piece loads "
            "checkpoints/seed6-step<a>.pt and fails in torch.load if the piece before it has not saved it. Under replay a "
            "checkpoint holds the initial weights (gpu-mock, 2026-10-20); nothing reads its values.",
            "The failure is ours and real, not served (disclosed): the chain script is pkg-chain-side's with the save's "
            "`os.makedirs(\"checkpoints\", exist_ok=True)` removed, and the task tree ships no checkpoints/, so piece 1's save "
            f"raises `{ERROR}` after its step-625 validation, from torch itself; the registry's TRL DPO record failed with the "
            "same text at its first checkpoint save. The fix is `mkdir checkpoints` (or the makedirs back) and the same line "
            "again; no flag or trigger is involved, and nothing in gpu-mock keys on the fix.",
            f"Shape, paced at one GPU: the chain {chain_s / 3600:.2f} h in {n} pieces; the side work {side_s / 3600:.2f} GPU-h in "
            f"{n_side} runs, {side_s / (gpus - 1) / 3600:.2f} h over {gpus - 1} GPUs; piece 1's failed attempt adds "
            f"{fail_s / 60:.0f} min to the chain.",
            "The side runs are pkg-chain-side's records, unchanged (each a recorded run's first 875 steps on its prefix script "
            "made single-process, served by path to the setup's folder); a model run of this package and of pkg-chain-side on one "
            "box would show the same curves twice.",
            "The whole run in one launch (`python longrun/circuit-muon/train_gpt_circuit_muon.py --seed 6`, no --stop-at) is "
            f"served exactly by a record no job lists (<parent>-{TAG}seg0-{parent['steps']}); that launch never saves, so it never "
            "meets the failure.",
            "Hardware: every record is an H100 run (circuit-muon and contra-soft-muon H100 SXM x8, radial-damp H100 SXM x4); "
            "no card-to-card conversion (gpu-tasks AGENTS.md, step 0)."],
        "reference_plan": {"instances": [{"job_id": j["job_id"]} for j in jobs], "policy": "packed_cp",
                           "rule": ("brief-only: every job the write-up needs, on free GPUs, longest remaining critical path "
                                    "first; the chain's pieces in order. With the failure: piece 1's failure noticed at its "
                                    "exit, `mkdir checkpoints` in one model turn, piece 1 relaunched ahead of waiting side work"),
                           "how": ("expected_scores runs packed_cp without the failure; the failure's reference, duration-blind "
                                   "and duration-informed, is survey/chain-save-refs-2026-11-02/sim_refs.json")},
    }


if __name__ == "__main__":
    main()
