"""H3', pkg-h3-chains: ten training runs to carry to their end from saved checkpoints, in pieces of uneven length (gpu-tasks-h3,
2026-10-23).

usage: python3 -I build/gen_h3_chains.py <registry>      (records and scripts; the task is gen_h3_chains.build, called from
                                                          gen_tasks)

The package search, hidden-plan level. Design measured before it was
built: ten recorded runs, each cut into
3 to 6 consecutive pieces at uneven stop steps. Nothing lists the pieces. Each run's folder has its script and a README naming
the run's seed, where its first piece stops and where the next one picks up; the agent has to read them to find the chains,
their lengths and what waits on what. One decision: of the six contra-soft-muon runs, only the four with the lowest validation
loss at step 1,000 go on past it.

What it measures, by list scheduling on the records' one-GPU paced durations (/tmp/h3/probe.py, 2026-10-23 15:28Z):
- the best schedule (critical path first, clairvoyant about the rule) is 3.74 h;
- a brief-only rule, most remaining pieces first and the stated continue-rule read at step 1,000, is 3.86 h, 1.032x (bar 2);
- brief order is 1.34x; a random chain order 1.15x median [1.07, 1.46].
The scripts are single-process (build/single_process.py), so one GPU per piece is a fact of the code: the width rule cannot pay
for an agent that runs every piece wide, which made the multi-GPU version 0.998x.

Every record is a recorded run's steps a+1..b (build/gen_segment.py derive_segment) on its single-process segment script, with
process_model single, as build/gen_h1.py writes them and build/validate.py checks them. Seeds: torch 2.11, none of H1's
(contra-soft-muon 10-13, soda-fade 5-8, radial-damp 6/8/10), distinct within the task because a segment script names its
checkpoint by seed and step.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_segment as S  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-h3-chains"
TAG = "h3"                     # record ids: <parent>-h3seg<a>-<b>
RULE_STEP = 1000
RULE_GROUP = "contra-soft-muon"
RULE_TOP = 4
ROOT = "runs"
# (folder, variant, seed, bounds). Each bound is one of the parent's validation steps (every 125); each piece is 20-43 minutes on
# one GPU at speed 1. The contra-soft-muon runs share the rule step 1,000 and differ after it.
CHAINS = [
    ("circuit-muon", "circuit-muon", 6, (0, 500, 1000, 1500, 2125, 2900)),
    ("contra-soft-muon", "contra-soft-muon", 5, (0, 1000, 2000, 3040)),
    ("contra-soft-muon", "contra-soft-muon", 14, (0, 1000, 2250, 3040)),
    ("contra-soft-muon", "contra-soft-muon", 18, (0, 1000, 1875, 3040)),
    ("contra-soft-muon", "contra-soft-muon", 20, (0, 1000, 2125, 3040)),
    ("contra-soft-muon", "contra-soft-muon", 24, (0, 1000, 2000, 3040)),
    ("contra-soft-muon", "contra-soft-muon", 26, (0, 1000, 2250, 3040)),
    ("ema-nesterov-aurora", "ema-nesterov-aurora", 13, (0, 500, 1000, 1375, 1875, 2375, 2900)),
    # radial-damp seed 3, not 0: soda-fade has seed 0 and both stop at 1,000, so both would write checkpoints/seed0-step1000.pt
    # (found by check_design, 2026-10-23 15:57Z; 0 was the 09:33 design). Seed 3 is pkg-sc-q's too; reuse across packages is
    # allowed, only within one task may a record not be served twice
    ("radial-damp", "radial-damp", 3, (0, 1000, 2125, 3020)),
    ("soda-fade", "soda-fade", 0, (0, 1000, 2250, 3040)),
]
H1_SEEDS = {"contra-soft-muon": {10, 11, 12, 13}, "soda-fade": {5, 6, 7, 8}, "radial-damp": {6, 8, 10}}
MIN_PIECE_MIN, MAX_PIECE_MIN = 20.0, 43.0


def parent_of(recs, variant, seed):
    g = S.SEGMENT_SPEC[variant]
    p = [r for r in recs.values() if r["identity"]["replicate_group"] == g and not r["identity"].get("derived_from")
         and r["identity"]["seed"]["value"] == seed and r["readiness"]["replay"]
         and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith("2.11.0")]
    if len(p) != 1:
        raise SystemExit(f"{variant} seed {seed}: {len(p)} replayable torch-2.11 parents")
    return p[0]


def script_text(parent, reg):
    """The run's script: the recorded script with save-and-resume, then made single-process (that order, as gen_h1 does:
    segment_script matches the original's distributed lines)."""
    return SP.single_process(S.segment_script(open(os.path.join(reg, parent["agent_view"]["script"])).read()))


def piece_records(parent, code, script_rel, bounds):
    out = []
    for a, b in zip(bounds, bounds[1:]):
        r = S.derive_segment(parent, code, a, b)
        ident = r["identity"]
        ident["record_id"] = f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"
        ident["process_model"] = {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                            "`python <script>`"}
        # gpu-mock reads launch.process_model (gpu_daemon._serving_info, revision 32c7d1a2) to pace a torchrun
        # --nproc_per_node>1 launch of a single-process script as one GPU, and gen_tasks.disclosure_of reads it for the
        # one-GPU estimate; without it the width leak returns
        r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
        r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command="python " + " ".join(ident["argv"]))
        r["readiness"] = C.readiness(r)
        out.append(r)
    return out


def check_design(recs):
    seeds = [(v, s) for _, v, s, _ in CHAINS]
    if len(set(s for _, s in seeds)) != len(seeds):
        raise SystemExit("two runs share a seed: their checkpoints would collide (checkpoints/seed<S>-step<N>.pt)")
    for v, s in seeds:
        if s in H1_SEEDS.get(v, set()):
            raise SystemExit(f"{v} seed {s} is H1's (build/gen_h1.py)")
    for _, v, _, b in CHAINS:
        if v == RULE_GROUP and RULE_STEP not in b[1:-1]:
            raise SystemExit(f"a {RULE_GROUP} run does not stop at the rule step {RULE_STEP}: {b}")


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    check_design(recs)
    n = 0
    for folder, variant, seed, bounds in CHAINS:
        p = parent_of(recs, variant, seed)
        code = script_text(p, reg)
        rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{p['identity']['argv'][0]}"
        C.write_text(os.path.join(reg, rel), code)
        for r in piece_records(p, code, rel, bounds):
            mins = C.paced_duration(r, 1) / 60
            if not MIN_PIECE_MIN <= mins <= MAX_PIECE_MIN:
                raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, outside {MIN_PIECE_MIN:g}-{MAX_PIECE_MIN:g}")
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            n += 1
        # The whole run in one launch (`python <script> --seed S`, no --stop-at): not a task job, since the brief asks for the
        # pieces, but an agent that runs a run whole is then served its recorded run exactly rather than on the modelled path.
        (w,) = piece_records(p, code, rel, (0, p["steps"]))
        C.write_json(os.path.join(reg, "records", w["identity"]["record_id"] + ".json"), w)
        n += 1
        print(f"{folder} seed {seed}: {len(bounds) - 1} pieces on {rel}")
    print(f"h3 records: {n}")


def readme(folder, variant, script, runs, steps):
    """The folder's README, as a researcher leaves it: the script, each run's seed, where its first piece stops and how the next
    picks up. Stop steps are given in prose, run by run, not as a list of commands. runs: [(seed, bounds)]."""
    head = (f"# {folder}\n\n{variant}: `{script}` in this folder is the training script, a {steps:,}-step schedule, single-process "
            "on one GPU, run from the task root. We train each run in pieces. A piece stops "
            "after the validation at its `--stop-at` step and saves `checkpoints/seed<S>-step<N>.pt`; the next piece is the same "
            "command with `--resume` pointing at that checkpoint and `--stop-at` at its own step, and the last piece has no "
            "`--stop-at` and trains to the end of the schedule. A piece started before the checkpoint it resumes from exists "
            "fails in torch.load before it touches the GPU. A resumed run validates first at the step it resumes from, and "
            "ends where an uninterrupted run would.\n\n")
    body = []
    for seed, bounds in runs:
        stops = bounds[1:-1]
        first = f"python {ROOT}/{folder}/{script} --seed {seed} --stop-at {stops[0]}"
        later = ", then ".join(f"{b:,}" for b in stops[1:])
        body.append(f"Seed {seed}: the first piece is `{first}`. "
                    + (f"After {stops[0]:,} it stops at {later}, and the last piece runs to {steps:,}." if later else
                       f"The last piece runs from {stops[0]:,} to {steps:,}."))
    return head + "\n\n".join(body) + "\n"


def build(recs, disclosure_of, reg=None):
    check_design(recs)
    jobs, files, cands = [], [], []
    shards = 0
    for folder, variant, seed, bounds in CHAINS:
        p = parent_of(recs, variant, seed)
        base = p["identity"]["argv"][0]
        pieces = [recs[f"{p['identity']['record_id']}-{TAG}seg{a}-{b}"] for a, b in zip(bounds, bounds[1:])]
        if not any(f["path"] == f"{ROOT}/{folder}/{base}" for f in files):
            files.append({"path": f"{ROOT}/{folder}/{base}", "from": pieces[0]["agent_view"]["script"]})
            runs = [(sd, bd) for fo, _, sd, bd in CHAINS if fo == folder]
            files.append({"path": f"{ROOT}/{folder}/README.md", "content": readme(folder, variant, base, runs, p["steps"])})
        elif next(f for f in files if f["path"] == f"{ROOT}/{folder}/{base}")["from"] != pieces[0]["agent_view"]["script"]:
            raise SystemExit(f"{folder}: two of its runs bind different scripts")
        run = f"{folder}-seed{seed}"
        prev, n = None, len(pieces)
        chain = []
        for i, (r, (a, b)) in enumerate(zip(pieces, zip(bounds, bounds[1:])), 1):
            argv = r["identity"]["argv"]
            jid = f"{run}-part{i}"
            jobs.append({"job_id": jid, "label": f"{variant}, seed {seed}, piece {i} of {n}: steps {a + 1}-{b}",
                         "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                         "draw": "exact_seed", "seed": seed,
                         # the record's hardware, not the width it runs at now: 4 or 8 GPUs as recorded
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": f"python {ROOT}/{folder}/{base} " + " ".join(argv[1:]),
                         "after": [prev] if prev else [], "consumes": [f"checkpoints/seed{seed}-step{a}.pt"] if a else [],
                         "produces": [f"checkpoints/seed{seed}-step{b}.pt"] if b < p["steps"] else [],
                         "chain": run, "piece": i})
            chain.append(jid)
            prev = jid
            shards = max(shards, r["host"]["train_data"]["shards_read"])
        if variant == RULE_GROUP:
            k = bounds.index(RULE_STEP)
            cands.append({"variant": run, "seed": seed, "screen_job": chain[k - 1],
                          "records": [pieces[k - 1]["identity"]["record_id"]],
                          "after_rule_jobs": chain[k:], "after_rule_records": [x["identity"]["record_id"] for x in pieces[k:]]})
    # The decision: the four contra runs with the lowest validation loss at step 1,000 go on. Their later pieces are gated by it;
    # a run that does not qualify ends at step 1,000. The pieces stay ordinary jobs with records, so a model agent's launch of a
    # piece the rule excludes is served (and scored as work the deliverable did not need, an early or extra run).
    scores = {c["variant"]: recs[c["records"][0]]["result"]["final_val_loss"]["value"] for c in cands}
    seed_of = {c["variant"]: c["seed"] for c in cands}
    keep = sorted(scores, key=lambda v: (scores[v], seed_of[v]))[:RULE_TOP]
    for j in jobs:
        for c in cands:
            if j["job_id"] in c["after_rule_jobs"]:
                j["gated_by"] = {"decision": "contra-best-four", "variant": c["variant"], "continues": c["variant"] in keep}
                if j["job_id"] == c["after_rule_jobs"][0]:
                    j["after"] = sorted(set(j["after"]) | {x["screen_job"] for x in cands})
    # The deliverable is the pieces the rule keeps. The two cut runs' later pieces are not task jobs (a scripted plan runs every
    # job, and the scorer counts a job not run as work remaining); their records stay in the registry, so a model agent that
    # runs one anyway is served it exactly and the run is an extra, listed in decisions[0].stopped.
    stopped = [j for j in jobs if (j.get("gated_by") or {}).get("continues") is False]
    jobs = [j for j in jobs if j not in stopped]
    for j in jobs:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    # 09:33 MDT draft, accepted 10:17 MDT, with one clause corrected at 16:10Z: it said each run "has its first stretch done and
    # saved under checkpoints/", which is false for these files (every run starts at step 0 and no checkpoint is shipped)
    brief = ("We're checking ten of our optimizer setups' long runs to their end. runs/ has one folder per run with its training "
             "script and a README; we train them in pieces, each picking up from the checkpoint the piece before it saved and "
             "saving the next. Bring every run to its last step, except that of the six contra-soft-muon runs only the four with "
             "the lowest validation loss at step 1,000 go on past it. Report each run's final validation loss. The machine has "
             "four H100s.")
    return {
        "task_id": TASK_ID, "rung": 2, "env_kind": "mng3", "level": "hidden_plan",
        "title": "H3': ten runs carried to their end in pieces of uneven length from saved checkpoints, the six contra-soft-muon "
                 "runs cut to the best four at step 1,000; every piece one GPU",
        "brief": {"text": brief, "provenance": ("written by gpu-tasks-h3 for the package search (the analysis, "
                                                "2026-10-23; drafted 09:33 MDT, accepted 10:17 MDT): the request, not the runs; "
                                                "which pieces exist, their stop steps and what waits on what are in each run's "
                                                "README and script; no deadline or time words; the GPU count stated")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "decisions": [{"decision_id": "contra-best-four", "after": [c["screen_job"] for c in cands],
                       "rule": (f"of the six {RULE_GROUP} runs, the {RULE_TOP} with the lowest validation loss at step "
                                f"{RULE_STEP} go on past it; ties broken by seed"),
                       "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": RULE_TOP},
                       "observable": f"the validation loss each contra run's piece prints at step {RULE_STEP}, as served",
                       "kind": "gate", "launches": [], "slots": 0, "shape_independent_of_outcome": False,
                       "shape_how": ("a gate, not a slot: each contra run's pieces after step 1,000 are its own recorded run's, and "
                                     "the rule says which of them the deliverable needs; the recorded values stop "
                                     + " and ".join(sorted(set(scores) - set(keep))) + ". gated_by on each such job says so"),
                       "candidates": [{k: v for k, v in c.items() if k != "after_rule_jobs"} for c in cands],
                       "stopped": [{"job_id": j["job_id"], "records": j["records"], "launch": j["launch"]} for j in stopped],
                       "stopped_how": ("the cut runs' pieces after step 1,000: not task jobs, since the deliverable does not need "
                                       "them; their records are in the registry, so a launch of one is served exactly (exact_only) "
                                       "and is an extra run, which a scorer ties here rather than to a job"),
                       "recorded_gate": {"continue": keep, "stop": sorted(set(scores) - set(keep)),
                                         "scores": {k: round(v, 5) for k, v in scores.items()},
                                         "how": f"final_val_loss of each contra run's piece ending at step {RULE_STEP}, lowest "
                                                f"{RULE_TOP} continue, ties by seed"}}],
        "measuring_run": {"harness": "scripted", "policy": "packed_cp", "speed": 10.0, "role": "check", "plan": "reference_plan",
                          "terms_from": ["pkg-p10-5c5p-q", "pkg-halving-q", "pkg-p20-par"],
                          "how": ("host terms transferred (build/transfer_host_terms.py --from, first match wins) from the "
                                  "measured segment packages: pkg-p10-5c5p-q first, which keys ema-nesterov-aurora's and "
                                  "radial-damp's same-named script by path, then pkg-halving-q (contra-soft-muon, soda-fade, "
                                  "circuit-muon). Those ran the multi-process segment scripts under torchrun; these are "
                                  "single-process under python, so start-up may differ, which the speed-10 scripted run of "
                                  "reference_plan checks")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; ten recorded runs on torch 2.11.0, converted to one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py): no torch.distributed, one GPU (cuda:0), the recorded global "
            "batch accumulated over micro-batches; launched with python. One GPU per piece is a fact of the code (analysis, "
            "2026-10-23 16:08Z). Timings are the records converted to one GPU by gpu-mock's convert().",
            "Each run's pieces are one recorded run cut at validation steps by the save-and-resume change (build/gen_segment.py); "
            "a piece loads checkpoints/seed<S>-step<a>.pt and fails in torch.load if the piece before it has not saved it. Under "
            "replay a checkpoint holds the initial weights (gpu-mock, 2026-10-20); nothing reads its values.",
            "Nothing lists the pieces: each run's README names its seed, the first piece's command and the later stop steps in "
            "prose, so the chains, their lengths and what waits on what are found by reading. The six contra-soft-muon runs share "
            "one folder and one script, as six seeds of one setup do.",
            "The rule's recorded outcome (decisions[0].recorded_gate): " + ", ".join(f"{k} {scores[k]:.5f}" for k in sorted(scores))
            + f"; continue {', '.join(keep)}. The two stopped runs' later pieces are in the task with records, so a launch of one is "
              "served and counted as work the deliverable did not need.",
            "Each run's whole-run launch (`python runs/<setup>/<script> --seed S`, no --stop-at) has a record too, <parent>-h3seg0-"
            "<steps>, listed by no job: an agent that runs a run whole instead of in pieces is served its recorded run exactly, "
            "and the scorer sees a launch the deliverable did not list.",
            "If an agent launches a script with torchrun --nproc_per_node>1, gpu-mock's pacing may read the launcher's WORLD_SIZE "
            "(asked of gpu-mock 2026-10-23 15:40Z; being closed); such launches are flagged in the run's record."],
        "reference_plan": _reference_plan(jobs, keep),
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


def _reference_plan(jobs, keep):
    """The brief-only rule as a scripted plan (bar 2). The jobs are the 31 pieces the rule keeps; the first piece after step 1,000
    of each kept contra run waits on all six contra runs' step-1,000 pieces (its `after`), which is when an agent reading their
    output can apply the cut. The scripted check runs it with plan.py's packed_cp (remaining critical path first, from the
    disclosed one-GPU estimates); a rule that uses only the READMEs, most pieces still to go first, simulates within 2% of it."""
    left = {}
    for j in jobs:
        left[j["chain"]] = left.get(j["chain"], 0) + 1
    return {"instances": [{"job_id": j["job_id"]} for j in jobs],
            "policy": "packed_cp",
            "rule": (f"brief-only: every run's pieces in order on free GPUs, longest remaining run first; a {RULE_GROUP} piece past "
                     f"step {RULE_STEP} only once all six have reached step {RULE_STEP}, and only for the {RULE_TOP} with the lowest "
                     "loss there"),
            "pieces_to_go": left,
            "how": ("the brief's rule and what the files say: the cut is read from the six step-1,000 pieces once they have "
                    "ended (the barrier in `after`), never known in advance. On paced one-GPU durations (build/expected_scores.py) "
                    "packed_cp is 3.78 h against the floor of 3.64 h (work / 4), 1.04x; most-pieces-to-go, which needs only the "
                    "READMEs, 3.86 h, 1.06x (/tmp/h3/probe.py, 2026-10-23 15:28Z)")}

if __name__ == "__main__":
    main()
