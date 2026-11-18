"""Angle I, pkg-matrix-budget: a given-plan matrix of optimizer setup x training budget x seed over recorded nanogpt runs
(gpu-tasks-matrix, 2026-10-31; package search, experiments/package-search/AGENTS.md; design sent to gpu-tasks 18:27 and
corrected 18:39 MDT; rulings relayed by gpu-tasks from the analysis 16:03, 16:19 and 18:50 MDT).

usage: python3 -I build/gen_matrix.py <registry>     (records and scripts; the tasks are gen_matrix.build(..., gpus=4|8),
                                                       called from gen_tasks)

The team question: how each optimizer setup's final validation loss depends on its training budget. optimizers/<setup>/
steps-<NNNN>/ holds one budget's training script, single-process, stopping after its validation at step NNNN, and a README
naming that budget's three seeds. The deliverable is the whole table (every setup, every budget, every seed and the mean), so
every cell is a job the answer needs.

Why GPUs sit idle (angle I): on packages of more than 16 jobs, most no-help Haiku and Sonnet runs deal the work into fixed
per-GPU lists up front (experiments/package-search/sim/NOTE-per-gpu-lists-2026-10-31.md). A list that holds the long cells keeps
its GPU busy after the others have emptied. The lengths here are uneven by budget (the task's title states the range). On
H100 the per-step cost varies only 1.27x across these setups, so the spread is all budget, and the
budget is readable from the folder name, the README and the script (analysis' route 3, 16:19 MDT: build it as a readable-length
package, and say so). The natural order a team writes, setup by setup with short budgets first, puts each setup's full runs
last.

Records. Every cell is its own recorded run (source.path), never one run's steps served twice: the detectability rule
(common.repeated_records) and gpu-tasks' count of distinct runs (98 in 16 setups on H100 and torch 2.11). Four H100 SXM setups
with at least 9 runs each: contra-soft-muon (30), contra-muon (12), muonh-tuned (10), radial-damp (9). newton-muon (15, H100
NVL) is left out, since a card conversion would be needed (gpu-tasks-h3's NVL/SXM 1.245).
- A cell short of the full schedule is the parent's prefix record at that step (gen_prefix's, which this builder never
  edits, or derived here with gen_prefix.derive where gen_prefix has none, same id `<parent>-stop<k>`), served as kind
  `prepared` on that prefix script made single-process (build/single_process.py), path_bound to the cell's folder. validate.py's
  rule for prepared records (gen_h1's form) checks it: entry_hash is single_process of the via record's script, and series,
  steps and timing are the via record's exactly.
- A full-budget cell is the parent itself, served the same way (gen_g4n's form).
- muonh-tuned's script takes no seed, so its three cells are its full budget only: a shorter muonh budget's loss is on its own
  full runs' validation lines, which would make the shorter cells readable without being run (18:39 MDT correction).

The tree is optimizers/, not setups/. G4n's prepared records are bound to setups/<setup>/ as directories, and gpu-mock's
match_path serves the first bound path a script falls under, so a matrix script edited under setups/contra-soft-muon/
steps-1000/ could have been served a G4n full run (found by gpu-tasks-matrix's foreign-bound-path check, 2026-10-31).

Seeds. contra-soft-muon and radial-damp take --seed; contra-muon reads SEED from the environment (gpu-mock reads no
environment seed, so its cells draw without replacement from their folder's three records, like an unseeded script). Each
budget's three seeds are distinct from every other budget's of the same setup. contra-muon's triples are chosen so that
gpu-mock's two draw orders agree: the intended draw takes a group's records in record-id order, the task draw in descending
seed order, and for each triple below the two orders are the same records, so the expected scores and a model run are served
alike (`_contra_muon_triples`).

Bar 2 holds by construction. Duration-informed: packed longest first, the registry's `packed` reference. Duration-blind: packed
by the stated stop step, which the files show. Both figures, and the per-GPU-list figures, are computed by schedule_figures at
build time and written into the task's `schedule` and notes, so they follow the grid; none is stated here.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_prefix as GP  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-matrix-budget"
TAG = "mx"                 # record ids: <via record>-mx
ROOT = "optimizers"  # not setups/: G4n and H1 bind setups/<setup>/ as directories (gpu-mock match_path takes the first hit)
FULL = "full"
# setup -> (submission, script basename, budgets in stop steps, short to long; FULL is the recorded schedule)
SETUPS = {
    # 3 budgets, not 4: only 3 disjoint triples of its runs keep gpu-mock's draw orders in step (_contra_muon_triples)
    "contra-muon": ("20260501_contra_muon", "train_gpt_simple_contra_muon_2.py", [1000, 2000, FULL]),
    "contra-soft-muon": ("20260509_contra_soft_muon", "contra-muon-to-soft-muon.py",
                         [875, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, FULL]),
    "muonh-tuned": ("20260611_muonh_tuned_aux_3250", "train_gpt_simple_muonh.py", [FULL]),
    "radial-damp": ("20260511_dampen_radial_gradient_component", "radial_scale_then_radius_correct.py", [1000, 2000, FULL]),
}
CARD = "NVIDIA H100 80GB HBM3"
TORCH = "2.11.0"
SEEDS = 3
MIN_JOB_S = 20 * 60


def pool(recs, setup):
    """The setup's recorded full runs: H100 SXM, torch 2.11, replay-ready, one per source.path, by seed then record id."""
    sub, script, _ = SETUPS[setup]
    rs = [r for r in recs.values() if r["identity"]["record_id"].startswith("mng3-") and not r["identity"].get("derived_from")
          and r["source"]["submission"] == sub and r["identity"]["argv"][0] == script and r["readiness"]["replay"]
          and r["reference"]["gpu"] == CARD and r["steps"] == r.get("planned_steps")
          and str((r["identity"]["trajectory"].get("environment") or {}).get("torch") or "").startswith(TORCH)]
    paths = [r["source"]["path"] for r in rs]
    if len(set(paths)) != len(paths):
        raise SystemExit(f"{setup}: two records share one recorded run (source.path)")
    if len({r["identity"]["replicate_group"] for r in rs}) != 1:
        raise SystemExit(f"{setup}: its runs span several replicate groups, so one script is not certain")
    return sorted(rs, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                     r["identity"]["record_id"]))


def _consistent(t):
    """A triple on which gpu-mock's two draw orders agree: ascending record id (the intended draw, Registry._draw, which
    expected_scores uses) is descending seed (the task draw, Registry._draw_task, which a model run uses)."""
    by_id = sorted(t, key=lambda r: r["identity"]["record_id"])
    return all(by_id[i]["identity"]["seed"]["value"] > by_id[i + 1]["identity"]["seed"]["value"] for i in range(len(t) - 1))


def _contra_muon_triples(rs, n):
    """n disjoint order-consistent triples of contra-muon's runs (env-seeded: gpu-mock sees three identical launches, so a cell
    draws without replacement and the two draw orders must agree, or expected_scores lists a known divergence). Exhaustive and
    deterministic: the first solution in combination order over runs sorted by seed. On the 12 recorded runs at most 3 such
    triples exist, which caps contra-muon at 3 budgets. Each triple is returned in record-id
    order, and the triples by their lowest seed."""
    import itertools
    rs = sorted(rs, key=lambda r: r["identity"]["seed"]["value"])

    def search(left, acc):
        if len(acc) == n:
            return acc
        if not left:
            return None
        first = left[0]
        for pair in itertools.combinations(left[1:], SEEDS - 1):
            t = (first,) + pair
            if _consistent(t):
                got = search([r for r in left if r not in t], acc + [t])
                if got:
                    return got
        return search(left[1:], acc)            # the lowest seed left sits in no triple
    got = search(rs, [])
    if not got:
        raise SystemExit(f"contra-muon: fewer than {n} order-consistent triples of its {len(rs)} runs")
    return sorted((sorted(t, key=lambda r: r["identity"]["record_id"]) for t in got),
                  key=lambda t: min(r["identity"]["seed"]["value"] for r in t))


def plan(recs):
    """[(setup, budget, [parents])]: each cell's three parent runs, every parent used once in the task."""
    out = []
    for setup, (_, _, budgets) in SETUPS.items():
        rs = pool(recs, setup)
        need = SEEDS * len(budgets)
        if len(rs) < need:
            raise SystemExit(f"{setup}: {len(rs)} recorded runs, {need} needed")
        if setup == "contra-muon":
            groups = _contra_muon_triples(rs, len(budgets))   # by lowest seed: the shortest budget takes the lowest seeds
        else:
            groups = [rs[i * SEEDS:(i + 1) * SEEDS] for i in range(len(budgets))]
        for b, ps in zip(budgets, groups):
            out.append((setup, b, ps))
    return out


def stop_of(parent, budget):
    return parent["steps"] if budget == FULL else budget


def folder(setup, parent, budget):
    return f"{ROOT}/{setup}/steps-{stop_of(parent, budget):04d}/"


def via_record(parent, budget, code, recs, reg, new):
    """The record a cell serves: the parent (full budget) or its prefix at the budget, gen_prefix's where it exists."""
    if budget == FULL:
        return parent, code
    rid = f"{parent['identity']['record_id']}-stop{budget}"
    r, pcode, rel = GP.derive(parent, code, budget, reg)
    if rid in recs:
        if recs[rid]["identity"]["entry_hash"] != r["identity"]["entry_hash"] or recs[rid]["series"] != r["series"]:
            raise SystemExit(f"{rid}: the registry's prefix record is not gen_prefix.derive's")
        return recs[rid], pcode
    new[rid] = (r, pcode, rel)
    return r, pcode


def cell_record(via, sp_code, script_rel, setup, budget):
    """The cell's served record: kind `prepared`, via the prefix (or full) record, on its script made single-process, bound to
    the cell's folder (validate.py's prepared rule)."""
    r = copy.deepcopy(via)
    ident = r["identity"]
    parent_id = (via["identity"].get("derived_from") or {}).get("record") or via["identity"]["record_id"]
    stop = via["steps"]
    bound = f"{ROOT}/{setup}/steps-{stop:04d}/"
    key = C.sha256_text(f"{TAG}\0{setup}\0{stop}")
    seed = ident["seed"]["value"]
    seeded_argv = "--seed" in ident["argv"]
    ident.update({
        "record_id": f"{via['identity']['record_id']}-{TAG}", "entry_hash": C.sha256_text(sp_code),
        "argv": [ident["argv"][0]] + (["--seed", str(seed)] if seeded_argv else []),
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the cell's single-process script bytes and argv, else served to any script below " + bound
                             + " (gpu-mock's path_bound rule, directory form): an edited script stays on its cell's runs"),
        "replicate_group": f"mng3-{TAG}-" + key[:12],
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {"kind": "prepared", "record": parent_id, "via": via["identity"]["record_id"], "steps": [1, stop],
                         "bound_path": bound, "parent_replicate_group": (via["identity"].get("derived_from") or {}).get(
                             "parent_replicate_group") or via["identity"]["replicate_group"],
                         "change": ("the recorded run" + ("" if budget == FULL else f"'s first {stop} steps (gen_prefix's stop "
                                                          "change)") + " on its script made single-process "
                                    "(build/single_process.py), served by path to the cell's folder")},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}', the setup and the stop step: one group per matrix cell"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel, launch_command=launch_line(setup, stop, ident["argv"][0], r))
    r["readiness"] = C.readiness(r)
    return r


def seed_env(r):
    return bool(r["identity"].get("env"))


def launch_line(setup, stop, base, r):
    seed = r["identity"]["seed"]["value"]
    path = f"{ROOT}/{setup}/steps-{stop:04d}/{base}"
    if seed is None:
        return f"python {path}"
    if seed_env(r):
        return f"SEED={seed} python {path}"
    return f"python {path} --seed {seed}"


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    new, vias, scripts = {}, [], {}
    for setup, budget, ps in plan(recs):
        code = open(os.path.join(reg, ps[0]["agent_view"]["script"])).read()
        for p in ps[1:]:
            if open(os.path.join(reg, p["agent_view"]["script"])).read() != code:
                raise SystemExit(f"{setup}: {p['identity']['record_id']} binds a different script")
        for p in ps:
            via, vcode = via_record(p, budget, code, recs, reg, new)
            vias.append((setup, budget, p, via, vcode))
    # prefix records gen_prefix does not make, written as gen_prefix writes its own (its derive, scripts and disclosure), before
    # the cells copy them
    C.add_disclosure([r for r, _, _ in new.values()])
    for rid, (r, pcode, rel) in new.items():
        r["readiness"] = C.readiness(r)
        C.write_text(os.path.join(reg, rel), pcode)
        C.write_json(os.path.join(reg, "records", rid + ".json"), r)
    n = 0
    for setup, budget, p, via, vcode in vias:
        sp = SP.single_process(vcode)
        rel = f"artifacts/mng3/scripts/{C.sha256_text(sp)[:12]}/{p['identity']['argv'][0]}"
        scripts[rel] = sp
        r = cell_record(via, sp, rel, setup, budget)
        if C.paced_duration(r, 1) < MIN_JOB_S:
            raise SystemExit(f"{r['identity']['record_id']}: {C.paced_duration(r, 1) / 60:.1f} min at one GPU, under 20")
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        n += 1
    for rel, sp in scripts.items():
        C.write_text(os.path.join(reg, rel), sp)
    print(f"matrix records: {n} cells, {len(new)} prefix records derived here, {len(scripts)} single-process scripts")


# ---------------------------------------------------------------------------------------------------------------------------
# The tasks (gen_tasks.main calls build for gpus=4 and gpus=8; finish() then adds stats, horizon and disclosure)

def readme(setup, script, stop, full_steps, seeds, env_seed):
    if env_seed:
        how = f"`SEED=N python {ROOT}/{setup}/steps-{stop:04d}/{script}`"
    else:
        how = f"`python {ROOT}/{setup}/steps-{stop:04d}/{script} --seed N`"
    which = (f"the full {full_steps:,}-step schedule" if stop == full_steps else
             f"the first {stop:,} steps of the {full_steps:,}-step schedule; the script stops after its validation at step "
             f"{stop:,} (`stop_after_step`), and the learning-rate schedule is the full one's")
    return (f"# {setup}, {stop:,} steps\n\n`{script}` in this folder is {setup}'s training script, single-process on one GPU, run "
            f"from the task root. It trains {which}. We use seeds {seeds[0]}, {seeds[1]} and {seeds[2]} at this budget: {how}.\n")


def readme_unseeded(setup, script, stop):
    return (f"# {setup}, {stop:,} steps\n\n`{script}` in this folder is {setup}'s training script, single-process on one GPU, run "
            f"from the task root. It trains the full {stop:,}-step schedule. The script takes no seed, so we run it three times: "
            f"`python {ROOT}/{setup}/steps-{stop:04d}/{script}`.\n")


def build(recs, disclosure_of, reg=None, gpus=4):
    if gpus not in (4, 8):
        raise ValueError(f"gpus {gpus!r}")
    tid = TASK_ID + ("-8gpu" if gpus == 8 else "")
    gpus_text = "four H100s" if gpus == 4 else "eight H100s"
    jobs, files, shards, table = [], [], 0, []
    cells = plan(recs)
    for setup, budget, ps in cells:
        rs = []
        for p in ps:
            via_id = p["identity"]["record_id"] if budget == FULL else f"{p['identity']['record_id']}-stop{budget}"
            rs.append(recs[f"{via_id}-{TAG}"])
        stop = rs[0]["steps"]
        base = rs[0]["identity"]["argv"][0]
        seeds = [r["identity"]["seed"]["value"] for r in rs]
        env_seed = seed_env(rs[0])
        files.append({"path": f"{ROOT}/{setup}/steps-{stop:04d}/{base}", "from": rs[0]["agent_view"]["script"]})
        files.append({"path": f"{ROOT}/{setup}/steps-{stop:04d}/README.md",
                      "content": readme_unseeded(setup, base, stop) if seeds[0] is None else
                      readme(setup, base, stop, ps[0]["steps"], seeds, env_seed)})
        exact = seeds[0] is not None and not env_seed
        for i, r in enumerate(rs, 1):
            jid = f"{setup}-{stop:04d}-run{i}"
            jobs.append({"job_id": jid,
                         "label": f"{setup} at {stop:,} steps: run {i} of 3" + (f", seed {seeds[i - 1]}" if seeds[i - 1] is not None else ""),
                         "replicate_group": r["identity"]["replicate_group"],
                         "records": [r["identity"]["record_id"]] if exact else [x["identity"]["record_id"] for x in rs],
                         "draw": "exact_seed" if exact else "without_replacement", "seed": seeds[i - 1] if exact else None,
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                         "launch": launch_line(setup, stop, base, r),
                         "after": [], "consumes": [], "produces": [], "setup": setup, "stop_step": stop,
                         **({"seed_from": "SEED in the environment, which gpu-mock does not read: the cell's three records are "
                                          "drawn without replacement inside its folder"} if env_seed else {})})
            shards = max(shards, r["host"]["train_data"]["shards_read"])
        table.append({"setup": setup, "stop_step": stop, "seeds": seeds,
                      "records": [r["identity"]["record_id"] for r in rs],
                      "parents": [r["identity"]["derived_from"]["record"] for r in rs],
                      "final_val_loss": [r["result"]["final_val_loss"]["value"] for r in rs],
                      "paced_1gpu_s": [round(C.paced_duration(r, 1), 1) for r in rs]})
    for j in jobs:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    setups = list(SETUPS)
    brief = ("We're measuring how each of our optimizer setups' validation loss depends on its training budget, for our "
             "small-GPT pretraining. " + ROOT + "/ has one folder per setup (" + ", ".join(f"{s}/" for s in setups) + "), and inside "
             "each one folder per budget, named by the step its runs stop at. Each budget folder holds the setup's training "
             "script and a README naming the three seeds we use at that budget. Every run follows the setup's standard "
             "learning-rate schedule and stops after its validation at that step; the last budget of each setup is the full "
             "schedule. We need the table: for every setup at every budget, the final validation loss of each of its three "
             f"runs and their mean. The machine has {gpus_text}.")
    durs = [d for row in table for d in row["paced_1gpu_s"]]
    fig = schedule_figures([C.paced_duration(recs[j["records"][0]], 1) for j in jobs], [j["stop_step"] for j in jobs], gpus)
    extra = {"box": __import__("gen_tasks").BOX8} if gpus == 8 else {}
    return {**extra,
        "task_id": tid, "rung": 1, "env_kind": "mng3", "level": "given_plan",
        "level_how": ("given plan: the brief names the deliverable and the files list every cell (folder, script, README with "
                      "its seeds); no cell depends on another's result. The plan is spelled out in the files rather than the "
                      "brief's prose, which keeps the brief free of procedure (the 2026-10-20 rules)"),
        "set": "development",
        "set_how": ("package search, angle I (experiments/package-search/AGENTS.md, 2026-10-31): a new candidate, screened before "
                    "it joins the benchmark (the study's word needed)"),
        "title": (f"Angle I: optimizer setup x training budget x seed, {len(jobs)} one-GPU nanogpt runs of "
                  f"{min(durs) / 60:.0f}-{max(durs) / 60:.0f} minutes, every cell needed for the table"
                  + (", on eight GPUs" if gpus == 8 else "")),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-matrix for the package search (angle I, 2026-10-31): the team's question and its table, not "
            "the runs; which budgets and seeds exist is in each folder's name and README; no procedure, no deadline or time "
            "words (the study, 2026-10-20); the GPU count stated; nothing names a test")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "matrix": {"cells": table,
                   "how": ("per (setup, budget): its three recorded runs, their final validation loss at the budget's stop "
                           "step and their paced one-GPU duration. Every record is a distinct recorded run (source.path); a "
                           "budget short of the full schedule is that run's prefix, a full budget the run itself")},
        "measuring_run": __import__("gen_tasks").measuring_run(jobs),
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; recorded runs on H100 SXM and torch 2.11.0, converted to one GPU; "
                    "prefixes by build/gen_prefix.py's stop change"],
        "notes": [
            "Scripts made single-process (build/single_process.py), as H1, H3', H4 and G4n: no torch.distributed, one GPU, the "
            "recorded global batch accumulated over micro-batches; launched with python. Timings are the recorded runs "
            "converted to one GPU by gpu-mock's convert(), the step time every script prints.",
            "Every cell is its own recorded run (keyed by source.path), never two cells from one run: a prefix and its parent's "
            "full run, or two prefixes of one parent, would repeat losses an agent can see (common.repeated_records). The cap "
            f"is the distinct runs: {sum(len(pool(recs, s)) for s in SETUPS)} in these four H100 SXM setups, of which the "
            f"matrix uses {len(jobs)}.",
            "A budget short of the full schedule is the recorded run's first k steps (build/gen_prefix.py's stop change: "
            "stop_after_step = k, the loop also breaks there; the learning-rate schedule is the full one's). So a shorter "
            "budget's loss is the full schedule's loss at that step, not a run re-scheduled to end there. An agent could read "
            "a seeded setup's shorter budgets off a full run's validation lines, but each budget's README names seeds of its "
            "own, so a supported table needs every cell run. muonh-tuned takes no seed, so its only budget here is the full "
            "schedule: a shorter muonh budget would be readable off its own full runs.",
            "The cells' lengths are readable from their budgets: the folder name, the README and the script's stop step all "
            "state it, and on H100 the per-step cost varies only 1.30-1.64 s across these setups, so the stated stop step ranks "
            f"the true lengths (Spearman {fig['spearman_stop_vs_length']} over the jobs). Analysis' route 3 (2026-10-31 16:19 MDT): build it as a "
            "readable-length package, stated where a team would state it, and say so. An agent that sorts by stated budget "
            "can schedule near the best; the waste this package can show is in how the work is dealt to GPUs, not in "
            "unknowable lengths.",
            "The natural order (glob order of " + ROOT + "/*/steps-*/) is setup by setup, shortest budget first, so every full run is "
            "last within its setup. Not tuned to any lane count.",
            "Recorded hardware: contra-soft-muon, contra-muon and muonh-tuned H100 SXM x8, radial-damp H100 SXM x4; no NVL, A100, "
            "H200 or L40S record is drawn. newton-muon (H100 NVL) is left out rather than converted.",
            "contra-muon reads its seed from SEED in the environment. gpu-mock matches on argv only, so its cells draw without "
            "replacement inside their folder; its triples are chosen so that gpu-mock's intended and task draws take the same "
            "records in the same order.",
            "Computed before the screen, ideal times on these jobs (schedule_figures, in `schedule`): fixed per-GPU lists dealt "
            f"round-robin in the natural order cost {fig['lists_round_robin'][0]}x the best, with idle-while-work-waits "
            f"{fig['lists_round_robin'][1]}; dealt in contiguous chunks {fig['lists_chunked'][0]}x, idle {fig['lists_chunked'][1]}"
            + (f"; four fixed lanes on eight GPUs {fig['four_lanes_round_robin'][0]}x, idle {fig['four_lanes_round_robin'][1]}"
               if gpus == 8 else "")
            + ". T-sim's angle-I prediction (1.53 at L0) was for a lumpier synthetic shape. On this smooth budget spread "
              "round-robin lists barely waste at 4 GPUs, and T-sim's own simulator on this job list (gpu-tasks-matrix, "
              "2026-10-31, notes/2026-10-31-angle-I-matrix.md) puts L0 under bar 4 at both GPU counts."]
        + (["Angle F's dial (the analysis 2026-10-31 13:58 MDT): the same jobs, records and answer on an 8-GPU box. "
            "The brief states eight H100s. It runs only on the 8-GPU image (environment.image_label)."] if gpus == 8 else []),
        "schedule": dict(fig, how=schedule_figures.__doc__.split("\n\n")[0].replace("\n    ", " ")),
        "reference_how": ("bar 2: the registry's packed reference (longest disclosed estimate first) is the duration-informed "
                          "reference; a duration-blind one, packed by the stated stop step the files show, is "
                          f"{fig['blind_by_stop']}x the informed best (schedule.blind_by_stop, ideal times), so the informed "
                          "reference asks nothing the files do not show"),
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


def _rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r, i = [0.0] * len(xs), 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    import math
    import statistics
    a, b = _rank(a), _rank(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    return (sum((x - ma) * (y - mb) for x, y in zip(a, b))
            / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)))


def schedule_figures(durs, stops, gpus):
    """Ideal-time schedules over the jobs in file order (no host terms, no agent turns), as ratios to the informed best (longest
    processing time first on true one-GPU durations): bar 2 duration-blind (packed by the stated stop step, which the files
    show), file-order FCFS, and fixed per-GPU lists dealt round-robin (index mod lanes, the fitted habit) or in contiguous
    chunks, each with its idle while a released job waited (all work is known at the start, so that is GPU idle before the
    last list starts its last job) over GPUs x makespan. At 8 GPUs also four fixed lanes, an agent keeping 4-GPU habits."""
    import heapq

    def lpt(order, g):
        free = [(0.0, i) for i in range(g)]
        heapq.heapify(free)
        for k in order:
            t, i = heapq.heappop(free)
            heapq.heappush(free, (t + durs[k], i))
        return max(t for t, _ in free)

    def lists(lanes, chunked):
        ls = [[] for _ in range(lanes)]
        for n, d in enumerate(durs):
            ls[min(lanes - 1, n * lanes // len(durs)) if chunked else n % lanes].append(d)
        load = [sum(x) for x in ls] + [0.0] * (gpus - lanes)
        mk, last_start = max(load), max(sum(x) - x[-1] for x in ls if x)
        return round(mk / best, 3), round(sum(max(0.0, last_start - x) for x in load) / (gpus * mk), 3)
    idx = list(range(len(durs)))
    best = lpt(sorted(idx, key=lambda k: -durs[k]), gpus)
    out = {"best_s": round(best, 1), "best_utilisation": round(sum(durs) / (gpus * best), 3),
           "blind_by_stop": round(lpt(sorted(idx, key=lambda k: (-stops[k], k)), gpus) / best, 4),
           "fcfs_file": round(lpt(idx, gpus) / best, 3),
           "lists_round_robin": lists(gpus, False), "lists_chunked": lists(gpus, True),
           "spearman_stop_vs_length": round(spearman(stops, durs), 3)}
    if gpus == 8:
        out["four_lanes_round_robin"] = lists(4, False)
        out["four_gpu_pool"] = round(lpt(sorted(idx, key=lambda k: -durs[k]), 4) / best, 3)
    return out


if __name__ == "__main__":
    main()
