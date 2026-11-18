"""Angle D, pkg-chain-side: a long chain plus side work (gpu-tasks-chain, 2026-10-31; package search, angle D,
experiments/package-search/AGENTS.md).

usage: python3 -I build/gen_chain_side.py <registry>      (derives the records and scripts; the task is
                                                           gen_chain_side.build(recs, disclosure_of, reg), which derives
                                                           them too if they are missing, and returns the task)

The team is finishing a write-up on its circuit-muon optimizer. WRITEUP.md is the draft; its blanks are the numbers still
needed. The brief asks for the finished write-up and names neither the chain nor the side work. Both are found in the files:
- the chain: circuit-muon's full run, carried to the end of its 2,900-step schedule in five pieces, each resuming the
  checkpoint the previous one saved (runs/circuit-muon/README.md names the stop steps). The dependency is real: a piece
  started before its checkpoint exists fails in torch.load;
- the side work: the write-up's early-training table, validation loss at step 875 for 27 seeds of contra-soft-muon and 9 of
  radial-damp (baselines/<setup>/README.md names the seeds), 36 runs, each independent of every other and of the chain.
The best schedule runs the chain on one GPU from the start and packs the side work around it. The chain is the one path
longer than a single job (2.05 h of the box's 4.0 h lower bound), so it has to start in the run's first half: an agent that
runs the side work first and the chain last ends 2.05 h after the side work, the chain alone on one GPU while three sit idle.
36 side runs is 9 full waves on 4 GPUs, so no GPU frees for the chain before the side work's last wave starts. File order
puts the chain after the side work, as a repository holds it
(baselines/ sorts before runs/, and the write-up's table comes before its full-run section): T-sim's ranking
(experiments/package-search/sim/RANKING-T-sim-2026-10-31.md, relayed by gpu-tasks 2026-10-31) found that a chain listed after
the side work is the shape whose waste a first-come-first-served scheduler does not remove.

Hardware: every record is an H100 run (circuit-muon and contra-soft-muon on H100 SXM x8, radial-damp on H100 SXM x4), so no
card-to-card conversion is needed (gpu-tasks AGENTS.md, step 0). ema-nesterov-aurora, the one longer setup that takes the
segment change, was recorded on A100s and is left out for that reason. newton-muon (H100 NVL) is left out rather than
converted. Every script is single-process (build/single_process.py), so one GPU per job is a fact of the code (H1, H3', H4,
G4n): without it an agent runs the chain four GPUs wide and the critical path disappears.

Records, all derived here and none edited by hand:
- chain pieces: circuit-muon seed 6's recorded run, steps a+1..b, on its single-process segment script
  (build/gen_segment.py derive_segment, made single-process as build/gen_h3_chains.py does), record ids <parent>-csdseg<a>-<b>;
  plus the whole run in one launch, which no job lists, so an agent that runs it whole is served it exactly: pkg-h3-chains'
  <parent>-h3seg0-2900 already serves that launch (same script bytes, same argv), so <parent>-csdseg0-2900 is written only
  when no record has that exact key;
- side runs: each a recorded run's first 875 steps (gen_prefix's prefix record), served whole on the prefix script made
  single-process, kind `prepared` with the prefix record as `via`, path_bound to its setup's folder (build/gen_g4n.py's and
  build/gen_h1.py's form, which validate.py checks), record ids <prefix>-csd.
Every job is a distinct recorded run (keyed by source.path), and the chain's parent run serves no side job.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_segment as S  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-chain-side"
TAG = "csd"
STEP = 875                               # the side runs' stop: the first validation step at which every side run is >= 20 min
CHAIN = ("circuit-muon", 6, (0, 625, 1250, 1875, 2375, 2900))   # variant, seed, piece bounds (validation steps)
CHAIN_ROOT = "runs"
SIDE_ROOT = "baselines"
# setup -> (parent replicate group, seeds). Seeds in order; contra-soft-muon's 27-29 are left free (rerun siblings). 27 + 9 = 36
# side runs, a multiple of the box's 4 GPUs (see the docstring)
SIDE = {
    "contra-soft-muon": ("mng3-cfg-d649f96e97a1", list(range(27))),
    "radial-damp": ("mng3-cfg-f04ccdc1544e", [0, 2, 3, 4, 5, 6, 8, 9, 10]),
}
MIN_JOB_MIN = 20.0
TORCH = "2.11.0"


def _t211(r):
    return str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith(TORCH)


def chain_parent(recs):
    variant, seed, _ = CHAIN
    g = S.SEGMENT_SPEC[variant]
    p = [r for r in recs.values() if r["identity"]["replicate_group"] == g and not r["identity"].get("derived_from")
         and r["identity"]["seed"]["value"] == seed and r["readiness"]["replay"] and _t211(r)]
    if len(p) != 1:
        raise SystemExit(f"{variant} seed {seed}: {len(p)} replayable torch-{TORCH} parents")
    return p[0]


def side_prefixes(recs, setup):
    """The setup's prefix records at STEP for its seeds, in seed order: one per recorded run."""
    g, seeds = SIDE[setup]
    by = {}
    for r in recs.values():
        d = r["identity"].get("derived_from") or {}
        if (d.get("kind") == "prefix" and d.get("parent_replicate_group") == g and d.get("stop_after_step") == STEP
                and r["readiness"]["replay"] and _t211(r)):
            by.setdefault(r["identity"]["seed"]["value"], []).append(r)
    out = []
    for s in seeds:
        if len(by.get(s) or []) != 1:
            raise SystemExit(f"{setup} seed {s}: {len(by.get(s) or [])} replayable step-{STEP} prefix records")
        out.append(by[s][0])
    return out


def chain_code(parent, reg):
    """The recorded script with save-and-resume, then made single-process (gen_h3_chains.script_text's order)."""
    return SP.single_process(S.segment_script(open(os.path.join(reg, parent["agent_view"]["script"])).read()))


def chain_records(parent, code, script_rel, bounds):
    out = []
    for a, b in zip(bounds, bounds[1:]):
        r = S.derive_segment(parent, code, a, b)
        ident = r["identity"]
        ident["record_id"] = f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"
        ident["process_model"] = {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                            "`python <script>`"}
        # gpu-mock paces a launch of a record whose launch.process_model is single at one GPU whatever WORLD_SIZE says
        # (gpu_daemon._serving_info, revision 32c7d1a2), and gen_tasks.disclosure_of reads it for the one-GPU estimate
        r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
        r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command="python " + " ".join(ident["argv"]))
        r["readiness"] = C.readiness(r)
        out.append(r)
    return out


def side_record(prefix, code, script_rel, setup):
    """A step-STEP prefix record served whole on its script made single-process: kind `prepared`, `via` the prefix record,
    path_bound to the setup's folder (validate.py's rule for prepared records: entry_hash is the via record's script made
    single-process, and series, steps and timing are the via record's exactly)."""
    r = copy.deepcopy(prefix)
    ident = r["identity"]
    bound = f"{SIDE_ROOT}/{setup}/"
    key = C.sha256_text(f"{TAG}\0{setup}\0{STEP}")
    seed = ident["seed"]["value"]
    ident.update({
        "record_id": f"{prefix['identity']['record_id']}-{TAG}", "entry_hash": C.sha256_text(code),
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the single-process script's bytes and argv, else served to any script below " + bound
                             + " (gpu-mock's path_bound rule, directory form): an agent that edits the script stays on its "
                             "setup's runs"),
        "replicate_group": f"mng3-{TAG}-" + key[:12],
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {**prefix["identity"]["derived_from"], "kind": "prepared", "via": prefix["identity"]["record_id"],
                         "bound_path": bound,
                         "change": (f"the recorded run's first {STEP} steps (build/gen_prefix.py's prefix record) on the prefix "
                                    "script made single-process (build/single_process.py), served by path to the setup's "
                                    "folder (build/gen_chain_side.py)")},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}', the setup and the stop step: one group per setup"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=f"python {bound}{ident['argv'][0]}" + (f" --seed {seed}" if seed is not None else ""))
    r["readiness"] = C.readiness(r)
    return r


def check_design(recs):
    """Every job a distinct recorded run (source.path), the chain's run serving no side job; every job >= MIN_JOB_MIN paced at
    one GPU; checkpoint names unique (a segment script names its checkpoint by seed and step only)."""
    parent = chain_parent(recs)
    paths = [parent["source"]["path"]]
    for setup in SIDE:
        paths += [r["source"]["path"] for r in side_prefixes(recs, setup)]
    if len(set(paths)) != len(paths):
        raise SystemExit("two jobs would serve one recorded run")
    return parent


def derive(recs, reg):
    """Write every record and script this task serves into <reg>, and add the records to recs. Idempotent: write_json and
    write_text leave an unchanged file alone."""
    parent = check_design(recs)
    n = 0
    code = chain_code(parent, reg)
    rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{parent['identity']['argv'][0]}"
    C.write_text(os.path.join(reg, rel), code)
    _, _, bounds = CHAIN
    for r in chain_records(parent, code, rel, bounds):
        mins = C.paced_duration(r, 1) / 60
        if mins < MIN_JOB_MIN:
            raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, under {MIN_JOB_MIN:g}")
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        recs[r["identity"]["record_id"]] = r
        n += 1
    # The whole run in one launch (no --stop-at), listed by no job, so an agent that runs it whole is served it exactly. Written
    # only when no record already serves that exact launch: build/gen_h3_chains.py writes <parent>-h3seg0-2900 on the same
    # single-process script bytes with the same argv, and two records of one exact key and one run would let the daemon serve
    # the same curve twice without its exact_seed_reused label (its never-twice set keys a segment by record id).
    (w,) = chain_records(parent, code, rel, (0, parent["steps"]))
    key = (w["identity"]["entry_hash"], tuple(w["identity"]["argv"]))
    if not any((x["identity"].get("entry_hash"), tuple(x["identity"].get("argv") or ())) == key for x in recs.values()
               if x["identity"]["record_id"] != w["identity"]["record_id"]):
        C.write_json(os.path.join(reg, "records", w["identity"]["record_id"] + ".json"), w)
        recs[w["identity"]["record_id"]] = w
        n += 1
    for setup in SIDE:
        ps = side_prefixes(recs, setup)
        scode = SP.single_process(open(os.path.join(reg, ps[0]["agent_view"]["script"])).read())
        for p in ps[1:]:
            if SP.single_process(open(os.path.join(reg, p["agent_view"]["script"])).read()) != scode:
                raise SystemExit(f"{setup}: {p['identity']['record_id']} binds a different prefix script")
        srel = f"artifacts/mng3/scripts/{C.sha256_text(scode)[:12]}/{ps[0]['identity']['argv'][0]}"
        C.write_text(os.path.join(reg, srel), scode)
        for p in ps:
            r = side_record(p, scode, srel, setup)
            mins = C.paced_duration(r, 1) / 60
            if mins < MIN_JOB_MIN:
                raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, under {MIN_JOB_MIN:g}")
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            recs[r["identity"]["record_id"]] = r
            n += 1
    return n


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    print(f"chain-side records: {derive(recs, reg)}")


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; complete_task then adds stats, horizon and disclosure)

def side_readme(setup, script, seeds):
    return (f"# {setup}\n\n`{script}` in this folder is {setup}'s training script, single-process on one GPU, run from the task "
            f"root: `python {SIDE_ROOT}/{setup}/{script} --seed N`. It trains the first {STEP:,} steps of the setup's full "
            f"schedule and stops after its validation at step {STEP:,}; the learning-rate schedule is the full one's. The "
            "write-up's early-training comparison uses seeds "
            + ", ".join(str(s) for s in seeds[:-1]) + f" and {seeds[-1]}.\n")


def chain_readme(variant, script, seed, bounds, steps):
    stops = bounds[1:-1]
    first = f"python {CHAIN_ROOT}/{variant}/{script} --seed {seed} --stop-at {stops[0]}"
    later = ", ".join(f"{b:,}" for b in stops[1:-1]) + f" and {stops[-1]:,}"
    return (f"# {variant}\n\n`{script}` in this folder is {variant}'s training script, a {steps:,}-step schedule, single-process "
            "on one GPU, run from the task root. We train the full run in pieces. A piece stops after the validation at its "
            "`--stop-at` step and saves `checkpoints/seed<S>-step<N>.pt`; the next piece is the same command with `--resume` "
            "pointing at that checkpoint and `--stop-at` at its own step, and the last piece has no `--stop-at` and trains to "
            "the end of the schedule. A piece started before the checkpoint it resumes from exists fails in torch.load before "
            "it touches the GPU. A resumed run validates first at the step it resumes from, and ends where an uninterrupted run "
            f"would.\n\nThe write-up's run is seed {seed}. Its first piece is `{first}`. After {stops[0]:,} it stops at {later}, "
            f"and the last piece runs to {steps:,}.\n")


def writeup(side_rows, variant, seed, bounds, steps):
    rows = "\n".join(f"| {setup} | {s} | — |" for setup, s in side_rows)
    marks = "\n".join(f"| {b:,} | — |" for b in bounds[1:-1]) + f"\n| {steps:,} (end of schedule) | — |"
    return f"""# {variant}: draft write-up

{variant} is our optimizer for small-GPT pretraining on FineWeb (data/fineweb10B/). This draft compares it with two of our
baselines, contra-soft-muon and radial-damp, early in training, and reports where its full schedule ends. Every `—` is a
number we still need, from a run on our machine.

## 1. Early training: validation loss at step {STEP:,}

{SIDE_ROOT}/ has one folder per baseline, with its training script and a README. {variant}'s own value at step {STEP:,} is the
one its full run (section 2) prints there.

| setup | seed | validation loss at step {STEP:,} |
|---|---|---|
{rows}
| {variant} (the full run, section 2) | {seed} | — |

Mean and standard deviation over each baseline's seeds:

| setup | runs | mean | standard deviation |
|---|---|---|---|
""" + "\n".join(f"| {setup} | {sum(1 for x, _ in side_rows if x == setup)} | — | — |" for setup in dict.fromkeys(x for x, _ in side_rows)) + f"""

## 2. The full {variant} run

{CHAIN_ROOT}/{variant}/ has its training script and a README on how we train it. Validation loss of seed {seed}'s run:

| step | validation loss |
|---|---|
{marks}
"""


def build(recs, disclosure_of, reg=None):
    import gen_tasks as GT
    reg = reg or GT.REG_DIR
    variant, seed, bounds = CHAIN
    parent = chain_parent(recs)
    if any(f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}" not in recs for a, b in zip(bounds, bounds[1:])):
        derive(recs, reg)
    jobs, files, side_rows, shards = [], [], [], 0
    for setup in SIDE:
        ps = side_prefixes(recs, setup)
        whole = [recs[f"{p['identity']['record_id']}-{TAG}"] for p in ps]
        script = ps[0]["identity"]["argv"][0]
        seeds = [r["identity"]["seed"]["value"] for r in whole]
        files.append({"path": f"{SIDE_ROOT}/{setup}/{script}", "from": whole[0]["agent_view"]["script"]})
        files.append({"path": f"{SIDE_ROOT}/{setup}/README.md", "content": side_readme(setup, script, seeds)})
        for r in whole:
            s = r["identity"]["seed"]["value"]
            jobs.append({"job_id": f"{setup}-seed{s}", "label": f"{setup}, seed {s}: steps 1-{STEP}",
                         "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                         "draw": "exact_seed", "seed": s,
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": f"python {SIDE_ROOT}/{setup}/{script} --seed {s}",
                         "after": [], "consumes": [], "produces": [], "role": "side"})
            side_rows.append((setup, s))
            shards = max(shards, r["host"]["train_data"]["shards_read"])
    base = parent["identity"]["argv"][0]
    pieces = [recs[f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"] for a, b in zip(bounds, bounds[1:])]
    files.append({"path": f"{CHAIN_ROOT}/{variant}/{base}", "from": pieces[0]["agent_view"]["script"]})
    files.append({"path": f"{CHAIN_ROOT}/{variant}/README.md",
                  "content": chain_readme(variant, base, seed, bounds, parent["steps"])})
    files.append({"path": "WRITEUP.md", "content": writeup(side_rows, variant, seed, bounds, parent["steps"])})
    prev, n = None, len(pieces)
    for i, (r, (a, b)) in enumerate(zip(pieces, zip(bounds, bounds[1:])), 1):
        jid = f"{variant}-part{i}"
        jobs.append({"job_id": jid, "label": f"{variant}, seed {seed}, piece {i} of {n}: steps {a + 1}-{b}",
                     "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                     "draw": "exact_seed", "seed": seed,
                     "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                     "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                     "launch": f"python {CHAIN_ROOT}/{variant}/{base} " + " ".join(r["identity"]["argv"][1:]),
                     "after": [prev] if prev else [], "consumes": [f"checkpoints/seed{seed}-step{a}.pt"] if a else [],
                     "produces": [f"checkpoints/seed{seed}-step{b}.pt"] if b < parent["steps"] else [],
                     "role": "chain", "chain": f"{variant}-seed{seed}", "piece": i})
        prev = jid
        shards = max(shards, r["host"]["train_data"]["shards_read"])
    for j in jobs:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    dur = {j["job_id"]: C.paced_duration(recs[j["records"][0]], 1) for j in jobs}
    chain_s = sum(dur[j["job_id"]] for j in jobs if j["role"] == "chain")
    side_s = sum(dur[j["job_id"]] for j in jobs if j["role"] == "side")
    n_side = sum(1 for j in jobs if j["role"] == "side")
    brief = (f"We're finishing our write-up on {variant}, an optimizer for our small-GPT pretraining; WRITEUP.md is the draft, "
             "and its blanks are the numbers we still need. Make the runs it needs on this machine, fill in every blank, and "
             "send back the finished WRITEUP.md. The machine has four H100s.")
    return {
        "task_id": TASK_ID, "rung": 2, "env_kind": "mng3", "level": "hidden_plan",
        "title": (f"Angle D: a write-up's long run, {variant} carried to the end of its schedule in {n} pieces from saved "
                  f"checkpoints, plus {n_side} independent baseline runs to step {STEP:,} for its comparison table; every job "
                  "one GPU"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-chain for the package search, angle D (2026-10-31): the request, not the runs. The chain "
            "(runs/circuit-muon/README.md) and the side work (WRITEUP.md's table, baselines/<setup>/README.md) are found in "
            "the files; no procedure, no deadline or time words, the GPU count stated, nothing that names a test")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": GT.fineweb_files(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "shape": {"chain_jobs": n, "side_jobs": n_side, "chain_paced_1gpu_s": round(chain_s, 1),
                  "side_paced_1gpu_s": round(side_s, 1), "side_over_three_gpus_s": round(side_s / 3, 1),
                  "how": ("paced one-GPU durations (common.paced_duration). The chain is the one path longer than a single "
                          "job; the best schedule starts it at once and packs the side work around it. Run after the side "
                          "work, it ends chain_paced_1gpu_s after the side work's last wave, alone on one GPU while three "
                          "sit idle")},
        "measuring_run": {"harness": "scripted", "policy": "packed_cp", "speed": 10.0, "role": "check", "plan": "reference_plan",
                          "terms_from": ["pkg-h3-chains", "pkg-p20-par"],
                          "how": ("host terms transferred (build/transfer_host_terms.py --from, first match wins) from pkg-h3-"
                                  "chains, whose circuit-muon single-process segment script is this task's chain script byte "
                                  "for byte (staged at the same path), then pkg-p20-par for the two baselines' step-875 prefix "
                                  "scripts (matched by setup folder; per_event gaps transfer between prefixes). pkg-g4n-a's "
                                  "single-process scripts are not a match: they are full runs and transfer_host_terms keys "
                                  "them by a basename two of its setups share. pkg-p20-par measured torchrun launches of the "
                                  "multi-process scripts, so its start-ups carry launcher time a python launch does not "
                                  "spend (about 2.3 s per job on H3', analysis 2026-10-23 14:10 MDT); the speed-10 scripted "
                                  "run of reference_plan checks them")},
        "sources": [f"modded-nanogpt track 3, commit 4ea6b937; {n_side + 1} recorded runs on torch 2.11.0 on H100s, converted to "
                    "one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py): no torch.distributed, one GPU (cuda:0), the recorded global "
            "batch accumulated over micro-batches; launched with python. One GPU per job is a fact of the code. Timings are the "
            "records converted to one GPU by gpu-mock's convert().",
            f"The chain is {variant} seed {seed}'s recorded run cut at steps " + ", ".join(f"{b:,}" for b in bounds[1:-1])
            + " by the save-and-resume change (build/gen_segment.py); a piece loads checkpoints/seed6-step<a>.pt and fails in "
            "torch.load if the piece before it has not saved it. Under replay a checkpoint holds the initial weights "
            "(gpu-mock, 2026-10-20); nothing reads its values. Its single-process segment script is pkg-h3-chains' circuit-muon "
            "script byte for byte; the cuts differ, so no record is shared.",
            f"The side runs are each a recorded run's first {STEP:,} steps (gen_prefix's prefix records, every one from a "
            "different recorded run), on the prefix script made single-process and served by path to the setup's folder.",
            f"Shape, paced at one GPU: the chain {chain_s / 3600:.2f} h in {n} pieces; the side work {side_s / 3600:.2f} GPU-h in "
            f"{n_side} runs. The best schedule starts the chain at once; run after the side work, the chain runs alone on one "
            "GPU while three sit idle.",
            "The whole run in one launch (`python runs/circuit-muon/train_gpt_circuit_muon.py --seed 6`, no --stop-at) is "
            f"served exactly by a record no job lists: pkg-h3-chains' <parent>-h3seg0-{parent['steps']} (the same script bytes "
            f"and argv), or this builder's <parent>-{TAG}seg0-{parent['steps']} when that one is absent.",
            "Hardware: every record is an H100 run (circuit-muon and contra-soft-muon H100 SXM x8, radial-damp H100 SXM x4); "
            "no card-to-card conversion (gpu-tasks AGENTS.md, step 0)."],
        "reference_plan": {"instances": [{"job_id": j["job_id"]} for j in jobs], "policy": "packed_cp",
                           "rule": ("brief-only: every job the write-up needs, on free GPUs, longest remaining critical path "
                                    "first; the chain's pieces in order"),
                           "how": ("packed_cp orders by the disclosed one-GPU estimates; a duration-blind version that ranks by "
                                   "step counts read from the READMEs and scripts (the chain's 2,900 steps against a side run's "
                                   f"{STEP:,}) picks the same order, since the chain is the only path longer than one job "
                                   "(notes/2026-10-31-angle-D-chain-side.md)")},
    }


if __name__ == "__main__":
    main()
