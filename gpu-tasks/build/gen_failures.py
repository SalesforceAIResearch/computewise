"""pkg-oom-relaunch: package-search angle C, failures and relaunch (gpu-tasks-failures, 2026-10-31; design sent to gpu-tasks
15:58 MDT and approved with three conditions at 16:10 MDT).

usage: python3 -I build/gen_failures.py <registry>     (scripts and records; the task is gen_failures.build(...), called from
                                                         gen_tasks; build() also derives the records if they are missing)

What the package is. A sweep of 24 one-GPU training runs, listed one launch command per line in sweep/runs.txt. Each run is a
single-process prefix (about 24-26 minutes at one GPU) of a distinct recorded modded-nanogpt run on H100 SXM, torch 2.11.0.
The sweep file gives every run its micro-batch with a flag we added, `--micro-batch`. For nine runs (circuit-muon's one and
radial-damp's eight) it gives 256, which does not fit an 80 GB card: those launches fail with CUDA out of memory at their first
forward. The fix is the one an engineer makes from the traceback: a smaller micro-batch (128 or less). The global batch is fixed
and the loop accumulates over micro-batches, so the fix changes nothing about the run, and the relaunch is served the run.

Why GPUs sit idle (angle C): a launch that dies at its first step frees its GPU at once, and the GPU stays idle until the agent
notices, fixes the line and relaunches. radial-damp's eight lines are the end of the sweep file, so for an agent working in file
order their relaunches are the only work left when they fail. A perfect agent loses nothing (the failure costs seconds), so the
best schedule is the all-clean packing and the waste is entirely the agent's noticing delay.

The script (all three changes are text transforms, each refusing a script where it is not unambiguous):
1. gen_prefix.stop_after(k): the prefix stop (the registry's -stop<k> record's own script);
2. single_process.single_process: one GPU, no torch.distributed (as H1, H3', H4, G4n);
3. add_micro_batch (here): a `--micro-batch` argparse flag, default 64 (the recorded constant), and `mbs = 64` becomes
   `mbs = args.micro_batch`. The recorded run is the default launch.

The records (derived_from.kind `oom_relaunch`; validate.py delegates to check_record below once gpu-tasks merges the 4-line
change written in notes/2026-10-31-angle-C-failures.md): the registry's prefix record (`via`) on the changed script, served
exactly (series, steps, timing are the prefix's), path_bound to sweep/<setup>/ so an edited script stays on its setup's runs,
with `--micro-batch` in argv_exempt so the failing launch and its fixed relaunch draw the same record. Every record carries
one synthesised failure (FAILURE-SERVING.md): mode oom, step 1, where forward, trigger {--micro-batch gt 128}. It is a
property of the micro-batch on the card, so any line raised above 128 fails, and the sweep file's lines at 64 never arm it.

The threshold rests on two assumptions, both named in the failure's `threshold.how` (provenance inferred, never upgraded):
the base peak (memory.peak, 33.44 GiB at micro-batch 64, itself inferred from track_1_short's 2024-10-10 Muon logs: these
scripts log no memory), and that peak memory scales linearly in the micro-batch (activations dominate: weights and optimizer
state are about 1.2 GiB). 128 gives 66.9 GiB and fits; 256 gives 133.7 GiB and does not; the valid micro-batches divide the
512 sequences of the global batch, so none lies between.
"""
import copy
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-oom-relaunch"
TAG = "oomr"
ROOT = "sweep"
KIND = "oom_relaunch"
FLAG = "--micro-batch"
BASE_MBS = 64                     # the recorded constant `mbs = 64` in every script used here
GLOBAL_SEQS = 512                 # batch_size = 8 * 64 * 1024 tokens / sequence length 1024
FAIL_MBS = 256                    # what runs.txt gives the failing lines
OK_MBS = BASE_MBS                 # what it gives the others
CARDS_GIB = {"torch": 79.19, "daemon": 80.0}   # an H100 80GB HBM3 as torch reports it; gpu-mock's daemon card (--capacity-gib)
GIB = 2 ** 30
# setup folder -> (prefix variant in gen_prefix.PREFIX_SPEC, seeds, failing). In sweep-file order: sorted by setup, then seed.
SWEEP = [
    ("circuit-muon", "circuit-muon", [6], True),
    ("contra-soft-muon", "contra-soft-muon", list(range(15)), False),
    ("radial-damp", "radial-damp", [0, 2, 3, 4, 5, 6, 8, 9], True),
]
MICRO_LINE = (f'parser.add_argument("{FLAG}", type=int, default={BASE_MBS})  # sequences per micro-step; the global batch is '
              "accumulated over micro-steps, so this changes memory, not the run")


# ---------------------------------------------------------------------------------------------------------------------------
# The script change and the threshold

def add_micro_batch(code):
    """The third change: a --micro-batch flag that sets the micro-step size, default the recorded 64."""
    seed = list(re.finditer(r'^parser\.add_argument\("--seed", type=int, default=0\)[ \t]*$', code, re.M))
    parse = re.findall(r"^args = parser\.parse_args\(\)[ \t]*$", code, re.M)
    mbs = list(re.finditer(rf"^mbs = {BASE_MBS}[ \t]*$", code, re.M))
    if len(seed) != 1 or len(parse) != 1 or len(mbs) != 1 or FLAG in code:
        raise ValueError(f"add_micro_batch: {len(seed)} --seed argument lines, {len(parse)} parse_args lines, {len(mbs)} "
                         f"`mbs = {BASE_MBS}` lines (each must be 1), or {FLAG} already present")
    if len(re.findall(r"% mbs == 0", code)) != 2:
        raise ValueError("add_micro_batch: the train and validation loops must each assert that mbs divides their batch")
    out = code[:mbs[0].start()] + "mbs = args.micro_batch" + code[mbs[0].end():]
    s = seed[0]
    return out[:s.end()] + "\n" + MICRO_LINE + out[s.end():]


def agent_script(prefix_code):
    return add_micro_batch(SP.single_process(prefix_code))


def peak_at(peak_bytes, mbs):
    """The peak at a micro-batch, linear in it from the recorded base (the assumption threshold.how names)."""
    return peak_bytes * mbs / BASE_MBS


def threshold(peak_bytes):
    """The largest valid micro-batch (a divisor of the global batch's sequences) whose peak fits the card, under both card
    figures; they must agree, or the threshold depends on which card one reads."""
    out = {}
    for name, gib in CARDS_GIB.items():
        fits = [m for m in range(1, GLOBAL_SEQS + 1) if GLOBAL_SEQS % m == 0 and peak_at(peak_bytes, m) <= gib * GIB]
        out[name] = max(fits)
    if len(set(out.values())) != 1:
        raise ValueError(f"the threshold differs between the card figures: {out}")
    return out["daemon"]


def oom_message(peak_bytes, mbs):
    """PyTorch's OOM wording, worded as gpu-mock's own oversubscription OOM (next/src/torch_patch/replay.py
    _oversubscribed_message): the job modelled as having taken the free card and failing on the remainder of its peak."""
    cap = CARDS_GIB["daemon"] * GIB
    mine = peak_at(peak_bytes, mbs)
    tried = mine - cap
    return (f"CUDA out of memory. Tried to allocate {tried / GIB:.2f} GiB. GPU 0 has a total capacity of {cap / GIB:.2f} GiB of "
            f"which 0 bytes is free. Including non-PyTorch memory, this process has {cap / GIB:.2f} GiB memory in use. Of the "
            f"allocated memory {cap / GIB:.2f} GiB is allocated by PyTorch, and 0 bytes is reserved by PyTorch but unallocated. "
            "If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid "
            "fragmentation.  See documentation for Memory Management  "
            "(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)")


def failure_entry(r):
    peak = r["memory"]["peak"]["value"]
    pk = r["memory"]["peak"]
    t = threshold(peak)
    msg = oom_message(peak, FAIL_MBS)
    syn = ("synthesised (gpu-tasks-failures, 2026-10-31; package-search angle C): no recorded run of this script failed. The "
           f"sweep file's {FLAG} {FAIL_MBS} for this run is a misconfiguration we wrote, and this failure is what an 80 GB card "
           f"does with it: CUDA out of memory at the first forward, before any optimizer step. At {FLAG} {t} or less the "
           "launch is not armed and draws this record clean")
    return {
        "mode": "oom", "step": 1, "time_s": 0.0, "at": "during_training", "provenance": "modelled", "how": syn,
        "cause": C.tv(f"torch.OutOfMemoryError: {msg}", "modelled",
                      "synthesised in PyTorch's OOM wording, as gpu-mock's oversubscription OOM words it (the job modelled as "
                      f"having taken the free card and failing on the remainder): this record's peak scaled to {FLAG} {FAIL_MBS}, "
                      f"{peak_at(peak, FAIL_MBS) / GIB:.2f} GiB, against the daemon's {CARDS_GIB['daemon']:.0f} GiB card"),
        "exception": {"type": "torch.OutOfMemoryError", "message": msg},
        "trigger": {"arg": FLAG, "op": "gt", "value": t},
        "threshold": C.tv(t, "inferred",
                          f"the largest micro-batch dividing the global batch's {GLOBAL_SEQS} sequences whose peak fits the card "
                          f"({CARDS_GIB['torch']} GiB as torch reports an H100 80GB HBM3, or the daemon's "
                          f"{CARDS_GIB['daemon']:.0f} GiB; the same under both). It rests on two assumptions: (1) the base peak, "
                          f"memory.peak = {peak / GIB:.2f} GiB at micro-batch {BASE_MBS}, provenance {pk['provenance']}: "
                          f"{pk.get('how')}; (2) that the peak scales linearly in the micro-batch: activations dominate, and "
                          "weights and optimizer state are about 1.2 GiB, so scaling only the activations gives "
                          f"{(1.2 * GIB + (peak - 1.2 * GIB) * 2) / GIB:.1f} GiB at 128 and "
                          f"{(1.2 * GIB + (peak - 1.2 * GIB) * 4) / GIB:.1f} GiB at 256, the same verdicts. Linear: "
                          f"{peak_at(peak, 128) / GIB:.1f} GiB at 128 (fits), {peak_at(peak, 256) / GIB:.1f} GiB at 256 "
                          "(does not)"),
        "where": "forward",
        "replay": {"value": "fire", "how": f"the run's own failure, armed only for a launch with {FLAG} above {t}"},
    }


# ---------------------------------------------------------------------------------------------------------------------------
# Records

def prefix_record(recs, variant, seed):
    import gen_prefix as GP
    g, k = GP.PREFIX_SPEC[variant]
    rs = [r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
          and r["identity"]["derived_from"].get("parent_replicate_group") == g
          and r["identity"]["derived_from"]["stop_after_step"] == k and r["identity"]["seed"]["value"] == seed
          and r["identity"]["record_id"].endswith(f"-stop{k}")]
    if len(rs) != 1:
        raise SystemExit(f"{variant} seed {seed}: {len(rs)} prefix records at step {k}, not 1")
    r = rs[0]
    par = recs[r["identity"]["derived_from"]["record"]]
    torch = str((par["identity"]["trajectory"].get("environment") or {}).get("torch") or "")
    if not (r["readiness"]["replay"] and par["readiness"]["replay"] and torch.startswith("2.11.0")
            and par["reference"]["gpu"] == "NVIDIA H100 80GB HBM3" and "--seed" in r["identity"]["argv"]):
        raise SystemExit(f"{r['identity']['record_id']}: not a replay-ready H100 SXM torch-2.11.0 run seeded by --seed")
    return r


def rid_of(prefix_rid):
    return f"{prefix_rid}-{TAG}"


def folder_group(folder):
    key = C.sha256_text(f"{TAG}\0{folder}")
    return f"mng3-{TAG}-{key[:12]}", key


def derive(prefix, code, script_rel, folder, failing):
    r = copy.deepcopy(prefix)
    ident = r["identity"]
    grp, key = folder_group(folder)
    bound = f"{ROOT}/{folder}/"
    pd = prefix["identity"]["derived_from"]
    seed = ident["seed"]["value"]
    ident.update({
        "record_id": rid_of(prefix["identity"]["record_id"]), "entry_hash": C.sha256_text(code),
        "argv_exempt": [FLAG], "replicate_group": grp, "match_policy": "path_bound", "bound_path": bound,
        "match_policy_how": ("exact on the script's bytes and argv (with --seed matched and " + FLAG + " exempt), else served to "
                             "any script below " + bound + " (gpu-mock's path_bound rule, directory form), drawn by seed: an "
                             "agent that edits the script stays on its setup's runs"),
        "argv_exempt_how": (FLAG + " sets the micro-step size only: the global batch is a constant and the loop accumulates over "
                            "micro-steps, so the optimisation and the step count are the recorded run's at any valid value, and "
                            "gpu-mock paces per optimizer step. It changes memory, which is what the failure's trigger reads "
                            "(FAILURE-SERVING.md: the triggering flag in argv_exempt, so the failing launch and its relaunch "
                            "draw one record)"),
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {"kind": KIND, "record": pd["record"], "via": prefix["identity"]["record_id"], "steps": list(pd["steps"]),
                         "stop_after_step": pd["stop_after_step"], "parent_steps": pd["parent_steps"],
                         "parent_replicate_group": pd["parent_replicate_group"], "bound_path": bound,
                         "change": ("the registry's prefix record on its script made single-process (build/single_process.py) "
                                    f"and given a {FLAG} flag (build/gen_failures.py add_micro_batch: default {BASE_MBS}, the "
                                    "recorded constant, so the recorded run is the default launch), served exactly, with one "
                                    "synthesised failure: an OOM armed only above the threshold (failures[0])"),
                         "how": "the via record's series, steps and timing exactly: the recorded run's first steps"},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}' and the sweep folder: one group per setup"
    r["source"] = dict(r["source"], derived=("a prefix of the run this log records, single-process, with a micro-batch flag "
                                             "(identity.derived_from)"))
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["launch"]["micro_batch_flag"] = {"flag": FLAG, "default": BASE_MBS, "valid": "a divisor of 512 (the scripts assert it)",
                                       "how": "build/gen_failures.py add_micro_batch"}
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    launch = f"python {bound}{ident['argv'][0]} --seed {seed} {FLAG} {FAIL_MBS if failing else OK_MBS}"
    r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command=launch,
                           recognisability="high: a public modded-nanogpt track-3 record, its script changed in two documented ways")
    # every record carries the OOM: it is a property of the micro-batch on the card, not of which sweep line it is, so a
    # line raised above the threshold fails as an 80 GB card would whichever run it is; at the threshold or under it is clean
    r["failures"] = [failure_entry(prefix)]
    r["outcome"] = dict(r["outcome"], how=(r["outcome"].get("how", "") + f"; at {FLAG} {failure_entry(prefix)['trigger']['value']} "
                                           f"or less. A launch above it fails at its first forward (failures[0], synthesised)"))
    r["readiness"] = C.readiness(r)
    return r


def setup_script(recs, reg, variant, seeds):
    """The agent's script of one setup: every seed's prefix record must bind the same prefix script."""
    prefs = [prefix_record(recs, variant, s) for s in seeds]
    texts = {open(os.path.join(reg, p["agent_view"]["script"])).read() for p in prefs}
    if len(texts) != 1:
        raise SystemExit(f"{variant}: its prefix records bind {len(texts)} different scripts")
    code = agent_script(texts.pop())
    rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{prefs[0]['identity']['argv'][0]}"
    return prefs, code, rel


def derive_all(recs, reg, write=True):
    """Every record of the package, written into reg (unchanged files are left alone) and added to recs."""
    out = []
    for folder, variant, seeds, failing in SWEEP:
        prefs, code, rel = setup_script(recs, reg, variant, seeds)
        if write:
            C.write_text(os.path.join(reg, rel), code)
        for p in prefs:
            out.append(derive(p, code, rel, folder, failing))
    C.add_disclosure(out)
    for r in out:
        recs[r["identity"]["record_id"]] = r
        if write:
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
    return out


def check_record(r, reg):
    """validate.py's check for an oom_relaunch record (delegated as for gen_g4n's and gen_h4's kinds): its script is its via
    prefix record's through single_process and add_micro_batch, it serves the via record exactly, it is path_bound with the
    flag exempt, and any failure is the documented OOM at the threshold its own memory figure gives."""
    import json
    d = r["identity"]["derived_from"]
    vp = os.path.join(reg, "records", d.get("via", "") + ".json")
    if not os.path.exists(vp):
        return [f"derived_from.via {d.get('via')} not in the registry"]
    v = json.load(open(vp))
    errs = []
    try:
        want = agent_script(open(os.path.join(reg, v["agent_view"]["script"])).read())
    except ValueError as e:
        return [f"the via script does not take the documented changes: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the script is not its via record's made single-process with only the micro-batch flag added")
    if r["series"] != v["series"] or r["steps"] != v["steps"] or r["timing"] != v["timing"]:
        errs.append("an oom_relaunch record must serve exactly its via record's run")
    if d.get("record") != (v["identity"].get("derived_from") or {}).get("record"):
        errs.append("derived_from.record must be the recorded run its via record cuts")
    if r["identity"]["match_policy"] != "path_bound" or not r["identity"].get("bound_path"):
        errs.append("an oom_relaunch record must be path_bound with a bound_path")
    if r["identity"].get("argv_exempt") != [FLAG]:
        errs.append(f"argv_exempt must be exactly [{FLAG}]")
    if len(r["failures"]) != 1:
        errs.append("an oom_relaunch record carries exactly one failure, the documented OOM")
    for f in r["failures"]:
        t = threshold(r["memory"]["peak"]["value"])
        if (f.get("mode") != "oom" or f.get("where") != "forward" or f.get("step") != 1
                or f.get("trigger") != {"arg": FLAG, "op": "gt", "value": t} or (f.get("threshold") or {}).get("value") != t
                or (f.get("threshold") or {}).get("provenance") != "inferred" or f.get("provenance") != "modelled"):
            errs.append(f"a failure must be the documented OOM: forward, step 1, trigger {FLAG} gt {t}, threshold inferred")
    return errs


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    out = derive_all(recs, reg)
    print(f"{TASK_ID}: {len(out)} records, each with the synthesised OOM above {FLAG} 128; "
          f"{sum(len(ss) for _, _, ss, f in SWEEP if f)} sweep lines at {FAIL_MBS}")


# ---------------------------------------------------------------------------------------------------------------------------
# The task

def sweep_file(jobs):
    head = "# one run per line, from the task root; every script is single-process on one GPU\n"
    return head + "".join(j["launch"] + "\n" for j in jobs)


# The file-order twin (the analysis, 2026-11-01 12:48 MST, shape 1 of notes/2026-11-01-late-failure-shapes.md): the same 24
# lines on the same records, written seed-major (for seed: for setup) instead of setup-major. Both are how a sweep file is written;
# only where the failing lines fall changes, and so how much clean work sits behind them. Same records, scripts and brief
ORDERS = {"setup": TASK_ID, "seed": TASK_ID + "-seedmajor"}


def build(recs, disclosure_of, reg=None, order="setup"):
    import gen_tasks as GT
    task_id = ORDERS[order]
    reg = reg or GT.REG_DIR
    want = [rid_of(prefix_record(recs, v, s)["identity"]["record_id"]) for _, v, ss, _ in SWEEP for s in ss]
    if any(x not in recs for x in want):
        derive_all(recs, reg)
    jobs, files, stats = [], [], []
    for folder, variant, seeds, failing in SWEEP:
        rs = [recs[rid_of(prefix_record(recs, variant, s)["identity"]["record_id"])] for s in seeds]
        script = rs[0]["identity"]["argv"][0]
        files.append({"path": f"{ROOT}/{folder}/{script}", "from": rs[0]["agent_view"]["script"]})
        for r in rs:
            seed = r["identity"]["seed"]["value"]
            j = {"job_id": f"{folder}-s{seed}", "label": f"{folder}, seed {seed}", "replicate_group": r["identity"]["replicate_group"],
                 "records": [r["identity"]["record_id"]], "draw": "exact_seed", "seed": seed,
                 "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                 "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                 "launch": r["agent_view"]["launch_command"], "after": [], "consumes": [], "produces": [], "setup": folder}
            if failing:
                f = r["failures"][0]
                j["fails_as_briefed"] = {
                    "micro_batch": FAIL_MBS, "mode": "oom", "step": 1, "where": "forward", "trigger": f["trigger"],
                    "fix": f"the same line with {FLAG} {f['trigger']['value']} or less, which draws this record clean",
                    "fixed_launch": j["launch"].replace(f"{FLAG} {FAIL_MBS}", f"{FLAG} {OK_MBS}"),
                    "how": "synthesised; failures[0] of the record"}
            jobs.append(j)
            stats.append({"job_id": j["job_id"], "record_id": r["identity"]["record_id"], "source": r["source"].get("path"),
                          "steps": r["steps"], "paced_1gpu_s": round(C.paced_duration(r, 1), 1), "fails_as_briefed": failing})
    srcs = [s["source"] for s in stats]
    if len(set(srcs)) != len(srcs):
        raise SystemExit(f"{task_id}: two jobs draw one recorded run")
    if order == "seed":
        # for seed: for setup (setups in SWEEP's order within a seed); the job list follows the file, as in the setup-major task
        rank = {folder: k for k, (folder, _, _, _) in enumerate(SWEEP)}
        jobs = sorted(jobs, key=lambda j: (j["seed"], rank[j["setup"]]))
        stats = sorted(stats, key=lambda s: [j["job_id"] for j in jobs].index(s["job_id"]))
    files.append({"path": f"{ROOT}/runs.txt", "content": sweep_file(jobs)})
    shards = max(recs[x]["host"]["train_data"]["shards_read"] for j in jobs for x in j["records"])
    files.append({"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                  "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"})
    n_fail = sum(1 for j in jobs if j.get("fails_as_briefed"))
    mins = [s["paced_1gpu_s"] / 60 for s in stats]
    # "final" asked for a value the prefix runs never reach (notes/2026-11-02-final-loss-audit.md). The seed-major twin takes the
    # fixed wording, in the SF fix's form; pkg-oom-relaunch is screened and keeps its own
    ask = ("report the last reported validation loss of each run" if order == "seed"
           else "report each run's final validation loss")
    brief = (f"We're training a sweep of small-GPT pretraining runs. {ROOT}/runs.txt has one launch command per run, run from the "
             f"task root, and each run's folder holds its training script. Run them all and {ask}. The machine has four H100s.")
    return {
        "task_id": task_id, "rung": 1, "env_kind": "mng3",
        "package_search": ("angle C, failures and relaunch (experiments/package-search/AGENTS.md; scope 20-30 jobs \"to "
                           "measure\", the analysis after T-sim's first ranking, relayed by gpu-tasks 2026-10-31). A "
                           "candidate, not one of the paper's packages: it joins them only on the study's word"),
        "title": (f"Package {task_id}: a sweep of {len(jobs)} one-GPU modded-nanogpt prefix runs of {min(mins):.0f}-{max(mins):.0f} "
                  f"minutes, {n_fail} of whose sweep lines OOM at their first step until the micro-batch is fixed"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-failures (2026-10-31): the goal and the sweep file, no procedure, no time words, no deadline "
            "(the study, 2026-10-20), the GPU count stated; nothing says that any line fails")},
        "files": files,
        "jobs": jobs,
        "failure_design": {
            "failing_jobs": [j["job_id"] for j in jobs if j.get("fails_as_briefed")],
            "how": (f"each failing job's sweep line gives {FLAG} {FAIL_MBS}. Every record carries one synthesised OOM armed above "
                    "the threshold, at the first forward, so a line at 64 never arms it and any line raised above 128 does. The relaunch with a micro-batch at or under the threshold draws the same "
                    "record clean: the failed launch ends before gpu-mock's short-run floor, so _end_job returns its draw "
                    "(verified on the daemon, notes/2026-10-31-angle-C-failures.md)"),
            "reference_how": ("build/expected_scores.py does not model a failure (its partial attribution says after_job_failure is "
                              "never simulated), so its policies run each line once, clean: the instant-fix best schedule. The "
                              "brief-only fail-and-relaunch references are computed in the note")},
        "measuring_run": GT.measuring_run(jobs),
        "candidates_stats": stats,
        "candidates_stats_how": "per job: its record, the recorded run behind it (source.path), steps and paced one-GPU duration",
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; records on torch 2.11.0, H100 SXM; prefix records by "
                    "build/gen_prefix.py, served by build/gen_failures.py"],
        "notes": [
            "Every job is the first k steps of a distinct recorded run (k 625, 1000 or 875 by setup: gen_prefix.PREFIX_SPEC), on "
            "the recorded script with three documented changes: the prefix stop (build/gen_prefix.py), single-process "
            "(build/single_process.py), and a --micro-batch flag we added (build/gen_failures.py add_micro_batch).",
            f"The {FLAG} flag is our change to the recorded scripts. Its default is the recorded constant mbs = {BASE_MBS}, so the "
            "recorded run is the default launch. It changes memory and not the run: the global batch (8 x 64 x 1024 tokens) is "
            "a constant and the loop accumulates over micro-steps, so every valid value takes the recorded optimizer steps on "
            "the recorded data. The scripts assert that it divides the batch.",
            *([] if order == "setup" else [
                f"The file-order twin of {TASK_ID} (the analysis, 2026-11-01): the same {len(jobs)} lines on the same "
                "records, written seed-major (for seed: for setup) where that task's runs.txt is setup-major. The failing lines "
                f"fall at positions {[k for k, j in enumerate(jobs) if j.get('fails_as_briefed')]} of {len(jobs)} (0-based), "
                "where the loop nest puts them: none was moved. Disclosed as C's file-order twin, from the same records"]),
            f"Synthesised: the sweep file's {FLAG} {FAIL_MBS} on circuit-muon's and radial-damp's lines is a team's "
            "misconfiguration we wrote, and the OOM those launches raise is synthesised (FAILURE-SERVING.md triggers). No "
            "recorded run of these scripts failed. The OOM's threshold is inferred from the records' memory figure, itself "
            "inferred (failures[0].threshold.how in each failing record).",
            "Recorded on H100 SXM (circuit-muon and contra-soft-muon on 8 GPUs, radial-damp on 4), converted to one GPU by "
            "gpu-mock's convert(). No NVL, H200 or other card is drawn.",
            "Every job is seeded by --seed, so a relaunch of a line is the same run: the failed launch's draw is returned and the "
            "relaunch draws the same record. Each script writes logs/<uuid>.txt, unique per run, and reads only data/fineweb10B/."],
    }


if __name__ == "__main__":
    main()
