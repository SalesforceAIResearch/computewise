"""Angle A, pkg-rounds (and pkg-rounds-8gpu): a two-look triage of thirty training runs, where every look waits for a whole
round of results and the rounds hold stragglers (gpu-tasks-rounds, 2026-10-31; package search, experiments/package-search/
AGENTS.md, angle A "rounds with stragglers").

usage: python3 -I build/gen_rounds.py <registry>      (records and scripts; the tasks are gen_rounds.build(..., gpus=4|8), called
                                                       from gen_tasks)

The study. Thirty recorded runs of three optimizer setups on H100 SXM (contra-soft-muon 21 seeds, radial-damp 8, circuit-muon 1),
each in its own folder under runs/ with a single-process script that saves and resumes (build/gen_segment.py's change, then
build/single_process.py's, as H3' and H1). The team's triage rule, stated in the brief, decides which runs go on at two looks:

- every run trains to step 875 and is reviewed there, all thirty together: the best 5 by validation loss at step 875 are
  promoted and train to the end of their schedule; the next 22 train on to step 1,750; the last 3 stop;
- at step 1,750 the 22 are reviewed together: the best 6 train to the end; the other 16 stop.

63 separately launched GPU jobs: 30 first pieces of 21-36 minutes, 5 promoted runs carried from 875 to the end (55-62 min,
circuit-muon's 85), 22 pieces from 875 to 1,750 (23-36 min), and 6 from 1,750 to the end (34-36 min). Every piece is one GPU
(single-process code), the whole study is about 30 GPU-hours.

Why GPUs go idle, and why it is real work. A look compares runs, so it waits for the slowest run in its round, and the promoted
runs, the longest pieces of the study, cannot start until the look is made. An agent that runs the study in waves (launch a
round, wait for all of it, read, launch the next) leaves every GPU that finishes early idle until the round's straggler ends,
and then starts the long promoted runs last. The best schedule starts each round's work as soon as its look can be made and puts
the promoted runs first among what is released; a review at step 875 still needs all thirty losses, so the bound waits for the
slowest first piece too. What it does not wait for is the agent. Lengths differ for real reasons: circuit-muon's step costs
2.47 s on one GPU against 1.59-1.63 s for the others (its circuit update), and the promoted runs train 2,145-2,165 steps against
875 for a round piece. Nothing is slowed, no sleep is added.

What the agent can know. The brief gives the rule and the GPU count; each folder's README names the run's seed and how a piece
stops and resumes. Durations are never shown (the disclosure flag is not built, notes/2026-10-29-duration-blind-reference.md).
The scripted reference is brief-only and duration-blind: it orders released work by what the files show (pieces still to go in a
run, then steps to go), not by recorded durations; reference_plan says so, and bar 2 is reported both ways in the note.

Records. Every piece is a recorded run's steps a+1..b (gen_segment.derive_segment) on its single-process segment script, with
process_model single, as gen_h3_chains writes them. Record ids carry the tag `rnd`, so they never collide with H3''s `h3seg`.
Seeds are distinct across the whole task, because a segment script names its checkpoint checkpoints/seed<S>-step<N>.pt: two runs
of different setups with one seed would write the same file at a shared review step. The pieces past each look that the rule
does not need (the 3 runs cut at 875, the 16 cut at 1,750, and the extra pieces a run that is promoted at 875 would need if it were
instead taken on to 1,750) are not task jobs; their records are written so that an agent that launches one anyway is served it
exactly, and the scorer counts it as work the deliverable did not need (decisions[].stopped, unneeded_jobs).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_segment as S  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-rounds"
TAG = "rnd"                    # record ids: <parent>-rndseg<a>-<b>
ROOT = "runs"
LOOKS = (875, 1750)            # the two review steps, both validation steps of every parent
PROMOTE_1, CONTINUE_1 = 5, 22  # at step 875: best 5 to the end, next 22 to step 1,750, the rest stop
PROMOTE_2 = 6                  # at step 1,750: best 6 of the 22 to the end, the rest stop
MIN_PIECE_MIN = 20.0
TORCH = "2.11.0"
GPU = "NVIDIA H100 80GB HBM3"
# (setup folder, gen_segment variant, seeds). Every seed once in the task (checkpoint names); radial-damp's seeds are its torch-2.11
# runs (1 and 7 were recorded on 2.10); circuit-muon has one recorded run. soda-fade is left out: its runs are on H200, and the box
# states H100s (a card difference would be a transplant, as G4n's records note found for NVL).
RUNS = [
    ("circuit-muon", "circuit-muon", (6,)),
    ("contra-soft-muon", "contra-soft-muon", (1, 7) + tuple(range(11, 30))),
    ("radial-damp", "radial-damp", (0, 2, 3, 4, 5, 8, 9, 10)),
]


def runs():
    return [(folder, variant, seed) for folder, variant, seeds in RUNS for seed in seeds]


def check_design():
    seeds = [s for _, _, s in runs()]
    if len(set(seeds)) != len(seeds):
        raise SystemExit("two runs share a seed: their checkpoints would collide (checkpoints/seed<S>-step<N>.pt)")
    if len(seeds) != PROMOTE_1 + CONTINUE_1 + (len(seeds) - PROMOTE_1 - CONTINUE_1) or PROMOTE_1 + CONTINUE_1 > len(seeds):
        raise SystemExit("the first look's bands do not fit the runs")
    if PROMOTE_2 > CONTINUE_1:
        raise SystemExit("the second look promotes more runs than it reviews")


def parent_of(recs, variant, seed):
    g = S.SEGMENT_SPEC[variant]
    p = [r for r in recs.values() if r["identity"]["replicate_group"] == g and not r["identity"].get("derived_from")
         and r["identity"]["seed"]["value"] == seed and r["readiness"]["replay"] and r["steps"] == r.get("planned_steps")
         and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith(TORCH)]
    if len(p) != 1:
        raise SystemExit(f"{variant} seed {seed}: {len(p)} replayable full torch-{TORCH} parents")
    if p[0]["reference"]["gpu"] != GPU:
        raise SystemExit(f"{variant} seed {seed}: recorded on {p[0]['reference']['gpu']}, not {GPU}")
    return p[0]


def script_text(parent, reg):
    """The run's script: the recorded script with save-and-resume, then made single-process (gen_h3_chains' order)."""
    return SP.single_process(S.segment_script(open(os.path.join(reg, parent["agent_view"]["script"])).read()))


def loss_at(parent, step):
    vl = parent["series"]["val_loss"]
    m = dict(zip(vl["steps"], vl["values"]))
    if step not in m:
        raise SystemExit(f"{parent['identity']['record_id']}: no validation at step {step}")
    return m[step]


def piece_record(parent, code, script_rel, a, b):
    r = S.derive_segment(parent, code, a, b)
    ident = r["identity"]
    ident["record_id"] = f"{parent['identity']['record_id']}-{TAG}seg{a}-{b}"
    ident["process_model"] = {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                        "`python <script>`"}
    # gpu-mock reads launch.process_model to pace a torchrun --nproc_per_node>1 launch of a single-process script as one GPU,
    # and gen_tasks.disclosure_of reads it for the one-GPU estimate (gen_h3_chains, 2026-10-23 19:38Z)
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command="python " + " ".join(ident["argv"]))
    r["readiness"] = C.readiness(r)
    return r


def bounds_needed(parent):
    """Every piece any decision path of this run can take: (0, 875); (875, end) if promoted at 875; (875, 1750) if continued;
    (1750, end) if promoted at 1,750. Plus (0, end), the whole run in one launch, for an agent that ignores the pieces."""
    end = parent["steps"]
    a, b = LOOKS
    return [(0, a), (a, end), (a, b), (b, end), (0, end)]


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    check_design()
    n = skipped = 0
    # exact key -> the records already serving it (gen_h3_chains writes the whole run of some of these seeds on the same script)
    have = {}
    for r in recs.values():
        have.setdefault((r["identity"].get("entry_hash"), tuple(r["identity"]["argv"])), []).append(r)
    for folder, variant, seed in runs():
        p = parent_of(recs, variant, seed)
        code = script_text(p, reg)
        rel = f"artifacts/mng3/scripts/{C.sha256_text(code)[:12]}/{p['identity']['argv'][0]}"
        C.write_text(os.path.join(reg, rel), code)
        for a, b in bounds_needed(p):
            r = piece_record(p, code, rel, a, b)
            same = [x for x in have.get((r["identity"]["entry_hash"], tuple(r["identity"]["argv"])), [])
                    if x["identity"]["record_id"] != r["identity"]["record_id"]]
            if same:
                # the same launch of the same recorded run already has a record (H3''s whole run, `-h3seg0-<end>`): a second
                # record under one exact key would only duplicate it, so it is not written (run this after gen_h3_chains)
                if any((x["identity"].get("derived_from") or {}).get("record") != p["identity"]["record_id"] for x in same):
                    raise SystemExit(f"{r['identity']['record_id']}: its launch already serves another run's record")
                skipped += 1
                continue
            mins = C.paced_duration(r, 1) / 60
            if mins < MIN_PIECE_MIN:
                raise SystemExit(f"{r['identity']['record_id']}: {mins:.1f} min on one GPU, under {MIN_PIECE_MIN:g}")
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
            n += 1
    print(f"rounds records: {n}, {skipped} not written because an identical record exists")


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build for 4 and 8 GPUs; complete_task then adds stats, horizon, environment and disclosure)

def answer_key(recs):
    """The rule applied to the recorded losses: who is promoted, continued and stopped at each look, with the margins at each
    cut so a close call is visible."""
    rs = [(folder, variant, seed, parent_of(recs, variant, seed)) for folder, variant, seed in runs()]
    a, b = LOOKS
    r1 = sorted(rs, key=lambda x: (loss_at(x[3], a), x[2]))
    promoted1, cont, stop1 = r1[:PROMOTE_1], r1[PROMOTE_1:PROMOTE_1 + CONTINUE_1], r1[PROMOTE_1 + CONTINUE_1:]
    r2 = sorted(cont, key=lambda x: (loss_at(x[3], b), x[2]))
    promoted2, stop2 = r2[:PROMOTE_2], r2[PROMOTE_2:]
    name = lambda x: f"{x[0]}-seed{x[2]}"
    gap = lambda lst, k, step: round(loss_at(lst[k][3], step) - loss_at(lst[k - 1][3], step), 5)
    return {
        "look1": {"step": a, "ranking": [{"run": name(x), "val_loss": loss_at(x[3], a)} for x in r1],
                  "promoted": [name(x) for x in promoted1], "continued": [name(x) for x in cont], "stopped": [name(x) for x in stop1],
                  "margin_promote": gap(r1, PROMOTE_1, a), "margin_stop": gap(r1, PROMOTE_1 + CONTINUE_1, a)},
        "look2": {"step": b, "ranking": [{"run": name(x), "val_loss": loss_at(x[3], b)} for x in r2],
                  "promoted": [name(x) for x in promoted2], "stopped": [name(x) for x in stop2], "margin_promote": gap(r2, PROMOTE_2, b)},
        "finished": sorted([name(x) for x in promoted1 + promoted2]),
        "final_val_loss": {name(x): x[3]["result"]["final_val_loss"]["value"] for x in promoted1 + promoted2},
        "best_final": min((x for x in promoted1 + promoted2), key=lambda x: (x[3]["result"]["final_val_loss"]["value"], x[2]))[2],
        "how": ("the recorded validation loss of each run at each look; ties by seed (none occur). margin_* is the loss gap across "
                "each cut, the distance between the last run in and the first run out"),
    }, {name(x): ("promoted1" if x in promoted1 else "promoted2" if x in promoted2 else "stopped2" if x in stop2 else "stopped1")
        for x in rs}


def readme(folder, variant, script, steps, seeds):
    """The folder's README, as a researcher leaves it: what the script is, how a piece stops and how the next resumes, each run's
    seed. It says nothing about which runs go on: that is the brief's rule, applied to what the runs print."""
    a, b = LOOKS
    head = (f"# {folder}\n\n{variant}: `{script}` in this folder is the training script, a {steps:,}-step schedule, single-process "
            "on one GPU, run from the task root. A run trains in pieces: `--stop-at N` validates at step N, saves "
            "`checkpoints/seed<S>-step<N>.pt` and exits; the next piece is the same command with `--resume` pointing at that "
            "checkpoint, and with no `--stop-at` it trains to the end of the schedule. A piece started before the checkpoint it "
            "resumes from exists fails in torch.load before it touches the GPU. A resumed run validates first at the step it "
            "resumes from, and ends where an uninterrupted run would. The script prints its validation loss at every validation "
            f"step, {a:,} and {b:,} among them.\n\n")
    if len(seeds) == 1:
        body = f"One run here, seed {seeds[0]}: `python {ROOT}/{folder}/{script} --seed {seeds[0]} --stop-at {a}` is its first piece."
    else:
        body = (f"{len(seeds)} runs here, one per seed: " + ", ".join(str(s) for s in seeds[:-1]) + f" and {seeds[-1]}. Each one's "
                f"first piece is `python {ROOT}/{folder}/{script} --seed <S> --stop-at {a}`.")
    return head + body + "\n"


BRIEF = ("We're triaging thirty of our small-GPT pretraining runs, from three optimizer setups. runs/ has one folder per setup with "
         "its training script and a README that names the runs' seeds and how a run is stopped and resumed. Our rule: every run "
         "trains to step 875 and is reviewed there, all thirty together. The five with the lowest validation loss at step 875 "
         "train to the end of their schedule; the next twenty-two train on to step 1,750; the last three stop. At step 1,750 the "
         "twenty-two are reviewed together: the six with the lowest validation loss there train to the end, and the rest stop. "
         "Report every run's last validation loss and which runs finished. The machine has {gpus}.")


def build(recs, disclosure_of, reg=None, gpus=4):
    if gpus not in (4, 8):
        raise ValueError(f"gpus {gpus!r}")
    check_design()
    tid = TASK_ID + ("-8gpu" if gpus == 8 else "")
    a, b = LOOKS
    key, fate = answer_key(recs)
    jobs, files, unneeded, shards = [], [], [], 0
    first_jobs, mid_jobs = [], []
    by_run = {}
    for folder, variant, seed in runs():
        p = parent_of(recs, variant, seed)
        base, end = p["identity"]["argv"][0], p["steps"]
        run = f"{folder}-seed{seed}"
        pid = p["identity"]["record_id"]
        rec = lambda lo, hi, pid=pid: recs[f"{pid}-{TAG}seg{lo}-{hi}"]
        if not any(f["path"] == f"{ROOT}/{folder}/{base}" for f in files):
            files.append({"path": f"{ROOT}/{folder}/{base}", "from": rec(0, a)["agent_view"]["script"]})
            seeds = [s for fo, _, s in runs() if fo == folder]
            files.append({"path": f"{ROOT}/{folder}/README.md", "content": readme(folder, variant, base, end, seeds)})
        elif next(f for f in files if f["path"] == f"{ROOT}/{folder}/{base}")["from"] != rec(0, a)["agent_view"]["script"]:
            raise SystemExit(f"{folder}: two of its runs bind different scripts")

        def job(lo, hi, label, after, stage, rec=rec, run=run, variant=variant, seed=seed, folder=folder, base=base, end=end):
            r = rec(lo, hi)
            return {"job_id": f"{run}-{lo}-{hi}", "label": f"{variant}, seed {seed}: steps {lo + 1}-{hi}" + (f" ({label})" if label else ""),
                    "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                    "draw": "exact_seed", "seed": seed, "recorded_gpus": r["reference"]["gpu_count"],
                    "recorded_gpu": r["reference"]["gpu"], "batch_policy": r["launch"]["batch_policy"]["value"],
                    "gpu_options": [1], "launch": f"python {ROOT}/{folder}/{base} " + " ".join(r["identity"]["argv"][1:]),
                    "after": list(after), "consumes": [f"checkpoints/seed{seed}-step{lo}.pt"] if lo else [],
                    "produces": [f"checkpoints/seed{seed}-step{hi}.pt"] if hi < end else [],
                    "run": run, "setup": folder, "stage": stage}
        first = job(0, a, "to the first review", [], "round1")
        first_jobs.append(first["job_id"])
        by_run[run] = {"first": first, "job": job, "end": end, "fate": fate[run]}
        shards = max(shards, rec(a, end)["host"]["train_data"]["shards_read"])
    # the first look: every promoted and continued piece waits on all thirty first pieces (the review needs every loss)
    for run, x in by_run.items():
        jobs.append(x["first"])
    for run, x in by_run.items():
        j, end, f = x["job"], x["end"], x["fate"]
        own = [x["first"]["job_id"]]
        to_end_1 = j(a, end, "promoted at step 875", own + first_jobs, "promoted1")
        to_mid = j(a, b, "on to the second review", own + first_jobs, "round2")
        if f == "promoted1":
            jobs.append(to_end_1)
            unneeded.append(dict(to_mid, unneeded_how="a promoted run's piece to step 1,750: the rule sends it to the end instead"))
        elif f in ("promoted2", "stopped2"):
            jobs.append(to_mid)
            mid_jobs.append(to_mid["job_id"])
            unneeded.append(dict(to_end_1, unneeded_how="a run continued at step 875, taken to the end without the second review"))
        else:
            unneeded.append(dict(to_mid, unneeded_how="a run the first review stops: its piece to step 1,750"))
            unneeded.append(dict(to_end_1, unneeded_how="a run the first review stops: its piece to the end"))
    for run, x in by_run.items():
        j, end, f = x["job"], x["end"], x["fate"]
        if f not in ("promoted2", "stopped2"):
            continue
        mid = f"{run}-{a}-{b}"
        last = j(b, end, "promoted at step 1,750", [mid] + mid_jobs, "promoted2")
        if f == "promoted2":
            jobs.append(last)
        else:
            unneeded.append(dict(last, unneeded_how="a run the second review stops: its piece to the end"))
    # whole-run launches (`python runs/<setup>/<script> --seed S`, no --stop-at, no --resume): records exist (main()), listed by
    # no job, so an agent that runs a run whole is served its recorded run exactly; the scorer sees a launch no job lists
    for j in jobs + unneeded:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    durs = [C.paced_duration(recs[j["records"][0]], 1) for j in jobs]
    if min(durs) < MIN_PIECE_MIN * 60:
        raise SystemExit(f"{tid}: a job of {min(durs) / 60:.1f} min, under {MIN_PIECE_MIN:g}")
    gpus_text = "four H100s" if gpus == 4 else "eight H100s"
    look_decisions = [
        {"decision_id": "review-875", "after": first_jobs, "kind": "gate",
         "rule": (f"of all thirty runs at step {a}, by validation loss: the best {PROMOTE_1} train to the end, the next {CONTINUE_1} "
                  f"to step {b}, the last {len(by_run) - PROMOTE_1 - CONTINUE_1} stop; ties by seed"),
         # resolve_decision's form: top = the runs that go on past the look (promoted and continued); bands splits them
         "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": PROMOTE_1 + CONTINUE_1,
                       "bands": [PROMOTE_1, CONTINUE_1], "bands_how": "the first band trains to the end, the second to step 1,750"},
         "candidates": [{"variant": run, "seed": x["first"]["seed"], "screen_job": x["first"]["job_id"],
                         "records": x["first"]["records"]} for run, x in by_run.items()],
         "observable": f"the validation loss each run's first piece prints at step {a}, as served",
         "launches": [], "slots": 0, "shape_independent_of_outcome": False,
         "shape_how": ("a gate, not a slot: every later piece is its own run's recorded continuation, and the rule says which the "
                       "deliverable needs; recorded_gate gives the outcome"),
         "recorded_gate": {k: key["look1"][k] for k in ("promoted", "continued", "stopped", "margin_promote", "margin_stop")}},
        {"decision_id": "review-1750", "after": mid_jobs, "kind": "gate",
         "rule": f"of the {CONTINUE_1} runs continued to step {b}, the best {PROMOTE_2} by validation loss there train to the end; the rest stop",
         "rule_spec": {"metric": "final_val_loss", "aggregate": "single", "direction": "min", "top": PROMOTE_2},
         "candidates": [{"variant": j_["run"], "seed": j_["seed"], "screen_job": j_["job_id"], "records": j_["records"]}
                        for j_ in jobs if j_["job_id"] in mid_jobs],
         "observable": f"the validation loss each continued run's piece prints at step {b}, as served",
         "launches": [], "slots": 0, "shape_independent_of_outcome": False,
         "shape_how": "a gate, as review-875",
         "recorded_gate": {k: key["look2"][k] for k in ("promoted", "stopped", "margin_promote")}}]
    extra = {"box": __import__("gen_tasks").BOX8} if gpus == 8 else {}
    n_first, n_p1, n_mid, n_p2 = len(first_jobs), PROMOTE_1, CONTINUE_1, PROMOTE_2
    return {
        **extra,
        "task_id": tid, "rung": 2, "env_kind": "mng3", "level": "goal_only",
        "set": "package-search", "set_how": ("package search, second round (experiments/package-search/AGENTS.md, angle A, "
                                            "gpu-tasks-rounds 2026-10-31); not in the paper's benchmark until the study says so"),
        "title": (f"Angle A: a two-look triage of thirty training runs, every look over a whole round, {len(jobs)} one-GPU pieces "
                  f"of {min(durs) / 60:.0f}-{max(durs) / 60:.0f} minutes" + (", on eight GPUs" if gpus == 8 else "")),
        "brief": {"text": BRIEF.format(gpus=gpus_text),
                  "provenance": ("written by gpu-tasks-rounds for the package search (2026-10-31): the team's triage rule and the "
                                 "report, not the runs; which pieces exist, their stop steps and how they resume are in the "
                                 "READMEs and scripts; no procedure, no deadline or time words (the study, 2026-10-20); the GPU "
                                 "count stated")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "unneeded_jobs": unneeded,
        "unneeded_how": ("pieces the rule does not need, given the recorded losses: a stopped run's later pieces, a promoted run's "
                         f"piece to step {b}, a continued run's piece straight to the end. Not task jobs (a scripted plan runs every "
                         "job, and the scorer counts a job not run as work remaining); their records are served exactly when "
                         "launched, so a run that takes one is served and scored as work the deliverable did not need"),
        "decisions": look_decisions,
        "answer": key,
        "measuring_run": {"harness": "scripted", "policy": "packed_cp", "speed": 10.0, "role": "check", "plan": "jobs",
                          "terms_from": ["pkg-h3-chains"],
                          "how": ("host terms transferred from pkg-h3-chains, whose single-process segment scripts are these "
                                  "scripts' byte-identical siblings for circuit-muon, contra-soft-muon and radial-damp (same "
                                  "transforms of the same recorded scripts); a speed-10 scripted check of the jobs confirms them "
                                  "before the package is declared ready")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; thirty recorded runs on H100 SXM, torch 2.11.0, converted to one GPU"],
        "notes": [
            "Scripts made single-process (build/single_process.py) after the save-and-resume change (build/gen_segment.py): no "
            "torch.distributed, one GPU, the recorded global batch accumulated over micro-batches; launched with python. One GPU per "
            "piece is a fact of the code. Timings are the records converted to one GPU by gpu-mock's convert().",
            "Each run's pieces are one recorded run cut at its validation steps 875 and 1,750; a piece loads "
            "checkpoints/seed<S>-step<a>.pt and fails in torch.load if the piece before it has not saved it. Under replay a "
            "checkpoint holds the initial weights (gpu-mock, 2026-10-20); nothing reads its values. Seeds are distinct across the "
            "task, so no two runs write one checkpoint name.",
            "The rule's recorded outcome is in answer and decisions[].recorded_gate. The margins at the cuts are small (loss gaps "
            f"of {key['look1']['margin_promote']:.5f} and {key['look1']['margin_stop']:.5f} at step {a}, "
            f"{key['look2']['margin_promote']:.5f} at step {b}), as real triage at one seed per run is.",
            f"Job counts: {n_first} first pieces, {n_p1} promoted at step {a}, {n_mid} continued to step {b}, {n_p2} promoted there: "
            f"{len(jobs)} jobs. Pieces the rule does not need are in unneeded_jobs with records, so a launch of one is served and "
            "counted as unneeded work; each run's whole-run launch (`python runs/<setup>/<script> --seed S`) has a record too, "
            "listed by no job.",
            "Lengths are uneven for real reasons only: circuit-muon steps at 2.47 s on one GPU against 1.59-1.63 s for the others, "
            f"and a promoted run trains 2,145-2,165 steps against {a} or {b - a} for a round piece. soda-fade is not used: its runs "
            "were recorded on H200, and the box states H100s.",
        ] + ([("The brief states eight H100s; the jobs, records, rule and answer are pkg-rounds's, so its runs compare with "
               "pkg-rounds's on the box alone. The box is 8 GPUs (gen_tasks.BOX8), so the task runs only on the 8-GPU image "
               "(environment.image_label), which waits on gpu-mock's device-count release")] if gpus == 8 else []),
        "reference_plan": _reference_plan(jobs),
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


def _reference_plan(jobs):
    """The brief-only rule as a scripted plan (bar 2): the jobs, each released when its `after` jobs end (a review needs its
    whole round), taken by the most pieces still to go in the job's run and then the most steps still to go: what the READMEs
    and the rule show, never a recorded duration."""
    return {"instances": [{"job_id": j["job_id"]} for j in jobs], "policy": "packed_cp",
            "rule": ("brief-only: each review once its whole round has ended; released pieces taken by the most pieces still to go "
                     "in their run, then the most steps to go, then file order; no recorded duration used"),
            "how": ("plan.py's packed_cp orders by the disclosed estimates, which this package's agents are never shown; the "
                    "duration-blind version of the same plan, ordered from the files alone, is the bar-2 reference, computed in "
                    "notes/2026-10-31-angle-A-rounds.md with both figures")}


if __name__ == "__main__":
    main()
