"""pkg-obs-latepole: the late-pole package's twin from a different source (lane OBS, 2026-11-02; rulings by
the analysis 15:11 and 15:37 MST; design experiments/package-search/lanes/observed/
OBSERVED-WASTE-2026-11-02.md, "Design 2").

usage: python3 -I build/gen_obs_latepole.py <registry>             (records and scripts; the task is
                                                                     gen_obs_latepole.build(...), called from gen_tasks)
       python3 -I build/gen_obs_latepole.py figures <registry>     (after the records exist: the job list in its natural
                                                                     order, bar 2 by stated steps, and the habit figures)

The mechanism (OBSERVED-WASTE, mechanism 2): every Sol and Astra G1 run launched its one long needed job 31st-40th, in file
order, and it ran alone for its last 2.3-5.5 h. the study's pkg-k2-latepole carries it on MLS-Bench cache and decode
jobs; this is its twin on modded-nanogpt H100 SXM runs, a different source under the selection guard.

The team question: how far along each optimizer setup is at a short training budget (1,000 steps), and whether the setup
that leads there, soap-f1-auxb2, keeps its lead over its full 2,900-step schedule. So every setup runs at 1,000 steps, and
soap-f1-auxb2 also runs its full schedule. Every job is needed: the deliverable is the table of every run's reported loss.

Why GPUs sit idle: the two full runs (about 2.4 h each at one GPU) are about 6.5x the 1,000-step runs (21-31 min). In the
natural order (setup folders sorted by name, budgets ascending, seeds ascending) soap-f1-auxb2 sorts second to last, so its
full runs are the 29th and 30th of 35. An agent that launches in that order starts them last and leaves the other GPUs idle
behind them. The length is readable: each script states its planned steps (stop_after_step for the 1,000-step runs, the
schedule's step count for the full runs), the README says so, and per-step cost differs 1.0-2.0x across these setups, so the
stated steps rank the true lengths. A brief-only reference that ranks by stated steps reaches the best schedule; an agent
that sorts by them escapes the waste, measured fairly.

Records (condition 4 of 15:11, read as ruled at 15:37): every job is its own recorded run (keyed by source.path), H100 80GB
HBM3, replay-ready, its full schedule recorded, not derived, its seed from --seed or none, and no setting read from the
environment except soap's STOP_STEP (its own stop knob). None is a run that the paper's packages or the package-search
candidates serve: the five, their twins and variants, K, K2, SF, C, A, D, filter-yield, the tabular packages and the held
matrix (package-search/lanes/observed/holdings.py's SEARCH; the 15:37 MST ruling: the old development suites, mng3-optimizer-
shootout, mng3-screen-then-confirm and the pkg-p* shapes, are not in the paper, so their runs are free). Those runs are
listed in RUNS below by record id, with what serves them elsewhere stated in the task's notes.
- The root track-3 records (records/track_3_optimization/results/<uuid>.txt, each its own script and group) are named as the
  source names them: by their uuid's first 8 characters.
- soap-f1-auxb2: 9 runs, seeds 11-19. Seeds 11-17 run 1,000 steps and seeds 18-19 the full schedule, so no recorded run is
  served twice.

The tree is latepole/<setup>/steps-<NNNN>/, one folder per setup and budget with its script and a README listing the runs, as
gen_matrix and K lay theirs out. A 1,000-step run is the recorded run's first 1,000 steps (build/gen_prefix.py's stop change, stop_after_step = 1000; for
soap-f1-auxb2, whose loop breaks on `step >= STOP_STEP`, its own STOP_STEP default set to 1000, gen_g4n.probe_script's change,
in a record of kind `olp_stop_step` that check_record below validates, as validate.py delegates G4n's and K's probe kinds).
The learning-rate schedule stays the full one's, as the README says. Each job is served as kind `prepared` on its script made
single-process (build/single_process.py), path_bound to its cell folder; a cell's runs share one script and one replicate group.
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
import gen_prefix as GP  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-obs-latepole"
TAG = "olp"                 # record ids: <via record>-olp
ROOT = "latepole"           # a root no record binds (2026-11-02: adoption, baselines, candidates, methods, precision, quick,
                            # runs, setups, sweep, yieldlab are taken)
SHORT = 1000                # the short budget, a validation step of every script
CARD = "NVIDIA H100 80GB HBM3"
MIN_JOB_S = 20 * 60
GPUS = 8
SOAP = "soap-f1-auxb2"
SOAP_FULL_SEEDS = (18, 19)  # the two full runs; seeds 11-17 run SHORT
# the job list by record id (holdings.free_runs("search") at 15:20 MST, 2026-11-02, every run it returned): folder name ->
# records. The folder is the setup's submission name without its date, or a root record's uuid prefix.
# shampoo-1-4-power (mng3-20260513-shampoo-1-4-power-h100-sxmx8-503575c5-6dd) is left out: its script imports
# distributed_shampoo, which the box image does not carry, so v1's one job never ran (runner, 2026-11-03 01:24 MST; dropped,
# not vendored, by the analysis's ruling of 01:25 MST, so every remaining record runs as recorded).
# IMAGE_PACKAGES is that image's stack (environment.install, constraints.txt); import_check refuses a script needing more.
IMAGE_PACKAGES = {"torch", "numpy", "setuptools", "filelock", "typing_extensions", "sympy", "mpmath", "networkx", "jinja2",
                  "markupsafe", "fsspec"}
RUNS = {
    "1bd8db7a": ["mng3-root-h100-sxmx8-1bd8db7a-f3a"],
    "311d7833": ["mng3-root-h100-sxmx8-311d7833-8df"],
    "478c0427": ["mng3-root-h100-sxmx8-478c0427-06c"],
    "51ece938": ["mng3-root-h100-sxmx8-51ece938-03c"],
    "7b8270c5": ["mng3-root-h100-sxmx8-7b8270c5-a9c"],
    "a63a68d1": ["mng3-root-h100-sxmx8-a63a68d1-24a"],
    "aurora": ["mng3-20260505-aurora-h100-sxmx8-298f02bc-dbb"],
    "muonh": ["mng3-20260430-muonh-h100-sxmx8-9cff801b-4ff", "mng3-20260430-muonh-h100-sxmx8-be7cc5ed-fb1",
              "mng3-20260430-muonh-h100-sxmx8-e69fd378-bcc", "mng3-20260430-muonh-h100-sxmx8-ee63188e-b33",
              "mng3-20260430-muonh-h100-sxmx8-efcac613-8a5"],
    "normuon": ["mng3-20260503-normuon-h100-sxmx8-08ffadff-357"],
    "normuonh": ["mng3-20260430-normuonh-h100-sxmx8-a6251aec-847", "mng3-20260430-normuonh-h100-sxmx8-b27e41ac-f5d",
                 "mng3-20260430-normuonh-h100-sxmx8-f166e3d2-afc", "mng3-20260430-normuonh-h100-sxmx8-f316e57a-213",
                 "mng3-20260430-normuonh-h100-sxmx8-f45b5dcf-16b"],
    "skylight001": ["mng3-20260501-skylight001-h100-sxmx8-e9e6f7ec-2ea", "mng3-20260501-skylight001-h100-sxmx8-f199a697-541",
                    "mng3-20260501-skylight001-h100-sxmx8-f78af80a-2ba"],
    SOAP: ["mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-f55bd90", "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-c0f2fdd",
           "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-726744b", "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-d156aa8",
           "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-2e8ab3b", "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-c5f3e5d",
           "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-be17439", "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-9ccf725",
           "mng3-20260609-soap-f1-auxb2-clean-h100-sxmx8-h100-042fe44"],
    "tuned-baseline-3250": ["mng3-20260610-tuned-baseline-3250-h100-sxmx8-a7c375e8-c0f",
                            "mng3-20260610-tuned-baseline-3250-h100-sxmx8-b199590f-f4a",
                            "mng3-20260610-tuned-baseline-3250-h100-sxmx8-c67512ee-207",
                            "mng3-20260610-tuned-baseline-3250-h100-sxmx8-f0bcca7d-a14",
                            "mng3-20260610-tuned-baseline-3250-h100-sxmx8-fc5b73a0-d67"],
}
# what else serves these runs today (holdings.py, all tasks, 15:20 MST): disclosed in the notes, none a package-search one
ELSEWHERE = "mng3-optimizer-shootout and mng3-screen-then-confirm (development suites, not in the paper)"
SOAP_STOP = r'STOP_STEP = int\(os\.environ\.get\("STOP_STEP", FINAL_TRAIN_STEPS\)\)'
# the brief-only reference: lane OBS's controller, which reads only the staged tree (each script's stated steps and each
# README's launch line) and launches the most stated steps first. Copied into the registry's artifacts by main(); plan.py's
# controller policy writes it into scripted runs only (task.scripted_reference)
REFERENCE_SRC = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "package-search", "lanes",
                                              "observed", "latepole_reference.py"))
REFERENCE_REL = "artifacts/obs-latepole/latepole_reference.py"
# the proved optimum of the job list's paced one-GPU durations on 8 GPUs (package-search/lanes/observed/olp_optimum.py, CP-SAT with
# loads in order as the only symmetry break, OPTIMAL; 9,186.7 s for v1's 36 jobs, 9,068.5 s for these 35 after shampoo's drop,
# 2026-11-03 01:33 MST); figures() checks its own LPT and lower bound against it, so a change of records that moves the optimum
# stops the build rather than reporting a stale ratio
OPTIMUM_S = 9068.5


def eligible(r):
    """The pool test (gen_unneeded_k.eligible's, every condition): why r may not be served, or None."""
    if r["identity"].get("derived_from"):
        return "derived"
    if r["reference"]["gpu"] != CARD:
        return f"card {r['reference']['gpu']}"
    if not r["readiness"]["replay"]:
        return "not replay-ready"
    if r["steps"] != r.get("planned_steps"):
        return "not its full schedule"
    if GP.seed_how(r) not in ("argv", "none"):
        return f"seed {GP.seed_how(r)}"
    return None


def plan(recs, reg=None):
    """[(folder, budget, parent)] in the natural order: folders sorted by name, budget ascending, then seed (or record id).
    Refuses a record that fails the pool test, two records of one recorded run, a folder whose records bind two scripts, or
    a script that imports a package the box image does not carry."""
    reg = reg or REG_ROOT
    out, paths = [], {}
    for folder in sorted(RUNS):
        rs = [recs[x] for x in RUNS[folder]]
        for r in rs:
            why = eligible(r)
            if why:
                raise SystemExit(f"{r['identity']['record_id']}: {why}")
            p = r["source"]["path"]
            if p in paths:
                raise SystemExit(f"{r['identity']['record_id']} and {paths[p]}: one recorded run")
            paths[p] = r["identity"]["record_id"]
        if len({r["identity"]["entry_hash"] for r in rs}) != 1:
            raise SystemExit(f"{folder}: its runs bind more than one script")
        missing = import_check(open(os.path.join(reg, rs[0]["agent_view"]["script"])).read())
        if missing:
            raise SystemExit(f"{folder}: its script imports {missing}, which the box image does not carry")
        rs = sorted(rs, key=lambda r: (r["identity"]["seed"]["value"] is None, r["identity"]["seed"]["value"] or 0,
                                       r["identity"]["record_id"]))
        for r in rs:
            seed = r["identity"]["seed"]["value"]
            full = folder == SOAP and seed in SOAP_FULL_SEEDS
            out.append((folder, r["steps"] if full else SHORT, r))
    return sorted(out, key=lambda x: (x[0], x[1], x[2]["identity"]["seed"]["value"] if x[2]["identity"]["seed"]["value"]
                                      is not None else -1, x[2]["identity"]["record_id"]))


REG_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "registry"))


def import_check(code):
    """The top-level modules a script imports that are neither the standard library nor IMAGE_PACKAGES (ast, every import
    statement, absolute imports only)."""
    import ast
    mods = set()
    for n in ast.walk(ast.parse(code)):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and not n.level:
            mods.add(n.module.split(".")[0])
    return sorted(m for m in mods if m not in sys.stdlib_module_names and m not in IMAGE_PACKAGES)


def stop_script(code, folder, k):
    """The parent's script stopped after its validation at step k: gen_prefix.stop_after, or for soap-f1-auxb2 its own
    STOP_STEP default set to k (gen_g4n.probe_script's change), since its loop breaks on `step >= STOP_STEP`."""
    if folder == SOAP:
        if len(re.findall(SOAP_STOP, code)) != 1:
            raise SystemExit("soap-f1-auxb2: the STOP_STEP line is not the one this change makes")
        return re.sub(SOAP_STOP, f'STOP_STEP = int(os.environ.get("STOP_STEP", {k}))  # this run: the first {k} steps', code)
    return GP.stop_after(code, k)


def short_record(parent, code, k, reg):
    """The parent's first k steps: gen_prefix.derive's record where its stop change applies; for soap-f1-auxb2 the same cut
    on its STOP_STEP form (kind `prefix`, the parent's series and timing cut at k, eval scaled to the validations kept)."""
    if not code.count("STOP_STEP = int(os.environ.get"):
        r, pcode, rel = GP.derive(parent, code, k, reg)
        return r, pcode, rel
    pcode = stop_script(code, SOAP, k)
    r = copy.deepcopy(parent)
    ident = r["identity"]
    pid = parent["identity"]["record_id"]
    sha = C.sha256_text(pcode)
    base = ident["argv"][0]
    rel = f"artifacts/mng3/scripts/{sha[:12]}/{base}"
    ch = GP.config_hash(pcode, base)
    ident.update({"record_id": f"{pid}-stop{k}", "entry_hash": sha, "replicate_group": "mng3-cfg-" + ch[:12],
                  "match_policy": "exact_only",
                  "match_policy_how": "a prefix shares its parent's fingerprint; only the prefix script's exact bytes match it"})
    ident["trajectory"]["config_hash"] = ch
    batch = ident["trajectory"].get("global_batch_tokens") or (r["launch"]["global_batch_tokens"] or {}).get("value")
    ident["trajectory"]["data"]["train_tokens_consumed"] = k * batch
    n_par = len(parent["series"]["val_loss"]["steps"])
    ident["derived_from"] = {
        "kind": "olp_stop_step", "record": pid, "steps": [1, k], "stop_after_step": k, "parent_steps": parent["steps"],
        "parent_entry_hash": parent["identity"]["entry_hash"], "parent_replicate_group": parent["identity"]["replicate_group"],
        "change": (f"STOP_STEP's default set to {k} (build/gen_obs_latepole.py stop_script, gen_g4n.probe_script's change): the "
                   "loop breaks on `step >= STOP_STEP`; nothing else in the script differs"),
        "how": "the first k optimizer steps of the parent run: every series value at a step <= k is the parent's, recorded"}
    r["source"] = dict(r["source"], derived=f"a prefix of the run this log records; its values at steps <= {k} are this log's")
    r["steps"] = r["planned_steps"] = k
    r["series"] = {n: GP.cut(s, k, f"cut at step {k}, the prefix's stop; the parent's values") for n, s in r["series"].items()}
    t = r["timing"]
    tt = r["series"].get("train_time_s")
    t["train_time_s"] = (C.tv(tt["values"][-1], "recorded", f"the parent run's own clock at its validation at step {k}")
                         if tt and tt["steps"] and tt["steps"][-1] == k else C.absent(f"no train_time at step {k}"))
    if (t.get("wall_total_s") or {}).get("value") is not None:
        t["wall_total_s"] = C.absent(f"the parent's wall time is the whole run's; the prefix stops at step {k}")
    ev = (t.get("eval_s") or {}).get("value")
    if ev:
        kept = len(r["series"]["val_loss"]["steps"])
        t["eval_s"] = C.tv(round(ev * kept / n_par, 3), "modelled", f"the parent's modelled eval_s x {kept}/{n_par}")
    r["failures"] = [f for f in parent["failures"] if f.get("step") is not None and f["step"] <= k
                     and f.get("phase") != "after_training"]
    r["result"]["final_val_loss"] = C.tv(r["series"]["val_loss"]["values"][-1], "recorded",
                                         f"the parent's validation loss at step {k}, the last the prefix script prints")
    if "reached_target" in r["result"]:
        r["result"]["reached_target"] = C.tv(False, "derived", f"the prefix stops at step {k}")
    r["outcome"] = {"status": "completed", "provenance": "derived", "how": f"the prefix script stops after its validation at {k}"}
    td = r["host"].get("train_data")
    if td and td.get("shards_read") is not None:
        td["shards_read"] = math.ceil(k * batch / 1e8)
    r["agent_view"] = dict(r["agent_view"], script=rel)
    r["readiness"] = C.readiness(r)
    return r, pcode, rel


def check_record(r, reg):
    """validate.py's check for an olp_stop_step record (delegated as for gen_g4n's g4n_probe and gen_unneeded_k's k_probe): its
    script is its parent's with only STOP_STEP's default set to its stop step, it ends there with the parent's values up to
    it, and it is exact_only in a group apart from its parent's."""
    d = r["identity"]["derived_from"]
    pp = os.path.join(reg, "records", d.get("record", "") + ".json")
    if not os.path.exists(pp):
        return [f"derived_from.record {d.get('record')} not in the registry"]
    p = json.load(open(pp))
    k = d.get("stop_after_step")
    errs = []
    try:
        want = stop_script(open(os.path.join(reg, p["agent_view"]["script"])).read(), SOAP, k)
    except SystemExit as e:
        return [f"the parent script does not take the STOP_STEP change: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the stop-step script is not its parent's with only STOP_STEP's default changed")
    if r["steps"] != k or any(x["steps"][-1] > k for x in r["series"].values()):
        errs.append(f"steps or a series runs past the stop step {k}")
    if k not in p["series"]["val_loss"]["steps"]:
        errs.append(f"stop step {k} is not one of the parent's validation steps")
    for name, sr in r["series"].items():
        ref = GP.cut(p["series"][name], k, "")
        if sr["steps"] != ref["steps"] or sr["values"] != ref["values"]:
            errs.append(f"series {name} is not the parent's cut at step {k}")
    if r["identity"]["match_policy"] != "exact_only" or r["identity"]["replicate_group"] == p["identity"]["replicate_group"]:
        errs.append("an olp_stop_step record must be exact_only, in a group apart from its parent's")
    return errs


def cell_folder(folder, budget):
    """One folder per (setup, budget), as gen_matrix and K lay them out: its runs share one script, one replicate group and
    one bound path, so gpu-mock's exact match (script bytes + argv) and the intended draw take the same pool. An unseeded
    cell's runs are drawn without replacement by record id in both (expected_scores' _draw and the daemon's _draw_task)."""
    return f"{ROOT}/{folder}/steps-{budget:04d}/"


def cell_record(via, sp_code, script_rel, bound):
    """The job's served record: kind `prepared`, via the prefix (or full) record, on its script made single-process, bound to
    the job's cell folder (validate.py's prepared rule)."""
    r = copy.deepcopy(via)
    ident = r["identity"]
    parent_id = (via["identity"].get("derived_from") or {}).get("record") or via["identity"]["record_id"]
    stop = via["steps"]
    key = C.sha256_text(f"{TAG}\0{bound}")
    seed = ident["seed"]["value"]
    seeded = "--seed" in ident["argv"]
    ident.update({
        "record_id": f"{via['identity']['record_id']}-{TAG}", "entry_hash": C.sha256_text(sp_code),
        "argv": [ident["argv"][0]] + (["--seed", str(seed)] if seeded else []),
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the cell's single-process script bytes and argv, else served to any script below " + bound
                             + " (gpu-mock's path_bound rule, directory form): an edited script stays on its cell's runs"),
        "replicate_group": f"mng3-{TAG}-" + key[:12],
        "process_model": {"value": "single", "how": "build/single_process.py: one GPU, no torch.distributed; launched "
                                                    "`python <script>`"},
        "derived_from": {"kind": "prepared", "record": parent_id, "via": via["identity"]["record_id"], "steps": [1, stop],
                         "bound_path": bound, "parent_replicate_group": (via["identity"].get("derived_from") or {}).get(
                             "parent_replicate_group") or via["identity"]["replicate_group"],
                         "change": ("the recorded run" + ("" if via is not None and not via["identity"].get("derived_from")
                                                         else f"'s first {stop} steps") + " on its script made single-process "
                                    "(build/single_process.py), served by path to the job's folder (build/gen_obs_latepole.py)")},
    })
    ident["trajectory"]["config_hash"] = key
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}' and the cell's folder: one group per (setup, budget)"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=f"python {bound}{ident['argv'][0]}" + (f" --seed {seed}" if seeded else ""))
    r["readiness"] = C.readiness(r)
    return r


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    new, vias = {}, []
    for folder, budget, p in plan(recs, reg):
        code = open(os.path.join(reg, p["agent_view"]["script"])).read()
        if budget == p["steps"]:
            via, vcode = p, code
        else:
            rid = f"{p['identity']['record_id']}-stop{budget}"
            via, vcode, rel = short_record(p, code, budget, reg)
            if rid in recs:
                if recs[rid]["identity"]["entry_hash"] != via["identity"]["entry_hash"] or recs[rid]["series"] != via["series"]:
                    raise SystemExit(f"{rid}: the registry's prefix record is not this change's")
                via = recs[rid]
            else:
                new[rid] = (via, vcode, rel)
        vias.append((folder, budget, p, via, vcode))
    # the prefix records this builder derives get their disclosure before any cell copies them, so a first build from a
    # registry without them writes the cells' own durations (gen_matrix's order). Copying first gave every cell its parent's
    # full-run disclosure.duration on a fresh registry
    C.add_disclosure([r for r, _, _ in new.values()])
    for rid, (r, pcode, rel) in new.items():
        r["readiness"] = C.readiness(r)
        C.write_text(os.path.join(reg, rel), pcode)
        C.write_json(os.path.join(reg, "records", rid + ".json"), r)
    scripts, n, cell_hash = {}, 0, {}
    for folder, budget, p, via, vcode in vias:
        sp = SP.single_process(vcode)
        srel = f"artifacts/mng3/scripts/{C.sha256_text(sp)[:12]}/{p['identity']['argv'][0]}"
        scripts[srel] = sp
        bound = cell_folder(folder, budget)
        r = cell_record(via, sp, srel, bound)
        prev = cell_hash.setdefault(bound, r["identity"]["entry_hash"])
        if prev != r["identity"]["entry_hash"]:
            raise SystemExit(f"{bound}: its runs' scripts differ")
        if C.paced_duration(r, 1) < MIN_JOB_S:
            raise SystemExit(f"{r['identity']['record_id']}: {C.paced_duration(r, 1) / 60:.1f} min at one GPU, under 20")
        d = (r.get("disclosure") or {}).get("duration") or {}
        if d.get("true_s") is not None and abs(d["true_s"] - C.job_duration(r)) > 0.01:
            raise SystemExit(f"{r['identity']['record_id']}: disclosure.duration.true_s {d['true_s']} is not its own "
                             f"duration {C.job_duration(r)}")
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        n += 1
    for rel, sp in scripts.items():
        C.write_text(os.path.join(reg, rel), sp)
    # the reference: lane OBS's file where the repository holds it, else the copy already in the registry (a VM snapshot of
    # build/ and registry/ has no package-search/ beside it)
    ref_dest = os.path.join(reg, REFERENCE_REL)
    if os.path.exists(REFERENCE_SRC):
        C.write_text(ref_dest, open(REFERENCE_SRC).read())
    elif not os.path.exists(ref_dest):
        raise SystemExit(f"the brief-only reference is at neither {REFERENCE_SRC} nor {ref_dest}")
    print(f"obs late-pole records: {n} jobs, {len(new)} prefix records derived here, {len(scripts)} single-process scripts")


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; finish() then adds stats, horizon and disclosure)

def readme(folder, script, budget, full_steps, launches, seeded):
    which = (f"its full {full_steps:,}-step schedule" if budget == full_steps else
             f"the first {budget:,} steps of its {full_steps:,}-step schedule, then stops after its validation at step "
             f"{budget:,} (the script's stop setting); the learning-rate schedule is the full one's")
    runs = (f"We use {len(launches)} seed{'s' if len(launches) > 1 else ''} at this budget:" if seeded else
            (f"The script takes no seed, so we run it {len(launches)} times:" if len(launches) > 1 else "One run:"))
    return (f"# {folder}, {budget:,} steps\n\n`{script}` here is {folder}'s training script, single-process on one GPU, run from "
            f"the task root. Each run trains {which}. {runs}\n\n" + "".join(f"- `{x}`\n" for x in launches))


def build(recs, disclosure_of, reg=None):
    jobs, files, shards, table, cells = [], [], 0, [], {}
    for folder, budget, p in plan(recs, reg):
        via_id = p["identity"]["record_id"] if budget == p["steps"] else f"{p['identity']['record_id']}-stop{budget}"
        cells.setdefault((folder, budget), []).append((p, recs[f"{via_id}-{TAG}"]))
    for (folder, budget), members in cells.items():
        rs = [r for _, r in members]
        bound = rs[0]["identity"]["bound_path"]
        base = rs[0]["identity"]["argv"][0]
        seeded = rs[0]["identity"]["seed"]["value"] is not None
        launches = [f"python {bound}{base}" + (f" --seed {r['identity']['seed']['value']}" if seeded else "") for r in rs]
        files.append({"path": f"{bound}{base}", "from": rs[0]["agent_view"]["script"]})
        files.append({"path": f"{bound}README.md", "content": readme(folder, base, budget, members[0][0]["steps"], launches, seeded)})
        for i, ((p, r), launch) in enumerate(zip(members, launches), 1):
            seed = r["identity"]["seed"]["value"]
            jid = f"{folder}--{budget:04d}--run{i}"
            jobs.append({"job_id": jid, "label": f"{folder} at {budget:,} steps: run {i} of {len(rs)}" + (f", seed {seed}" if seed is not None else ""),
                         "replicate_group": r["identity"]["replicate_group"],
                         "records": [r["identity"]["record_id"]] if seeded else [x["identity"]["record_id"] for x in rs],
                         "draw": "exact_seed" if seeded else "without_replacement", "seed": seed if seeded else None,
                         "recorded_gpus": r["reference"]["gpu_count"], "recorded_gpu": r["reference"]["gpu"],
                         "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1], "launch": launch,
                         "after": [], "consumes": [], "produces": [], "setup": folder, "stop_step": budget,
                         "stated_steps": budget})
            shards = max(shards, r["host"]["train_data"]["shards_read"])
            table.append({"job_id": jid, "setup": folder, "steps": budget, "seed": seed, "record": r["identity"]["record_id"],
                          "parent": r["identity"]["derived_from"]["record"], "source_path": p["source"]["path"],
                          "val_loss": r["result"]["final_val_loss"]["value"], "paced_1gpu_s": round(C.paced_duration(r, 1), 1),
                          "torch": p["identity"]["trajectory"]["environment"].get("torch")})
    for j in jobs:
        j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
    fig = figures([t["paced_1gpu_s"] for t in table], [t["steps"] for t in table], GPUS)
    torch_by = {}
    for t in table:
        torch_by.setdefault(t["torch"], set()).add(t["setup"])
    brief = (f"We're comparing optimizer setups for our small-GPT pretraining at a short training budget, and checking one "
             f"result at full length. {ROOT}/ has one folder per setup and inside each one folder per training budget, named "
             "by the step its runs stop at; each budget folder holds the setup's training script and a README listing the "
             f"runs we use. Every setup has runs at {SHORT:,} steps, and {SOAP}, which looks strongest there, "
             f"also has runs of its full schedule. We need the table: for every run, the validation loss it reports at the "
             f"end of its run, and each setup's mean at each budget; and whether {SOAP}'s lead at {SHORT:,} steps holds at its "
             "full schedule. The machine has eight H100s.")
    return {
        "box": __import__("gen_tasks").BOX8,
        "task_id": TASK_ID, "rung": 1, "env_kind": "mng3", "level": "given_plan",
        "level_how": ("given plan: the brief names the deliverable and the files list every run (each budget folder's script "
                      "and its README's launch lines); no run depends on another's result; the plan is in the files, not the "
                      "brief's prose"),
        "set": "package-search",
        "set_how": ("lane OBS of the 2026-11-02 lanes (experiments/package-search/AGENTS.md): the late-pole package's twin from "
                    "a different source (pkg-k2-latepole is MLS-Bench cache and decode jobs; this is modded-nanogpt H100 SXM), "
                    "built by the analysis's rulings of 15:11 and 15:37 MST"),
        "title": (f"Late poles in an optimizer sweep: {len(jobs)} one-GPU nanogpt runs, {fig['short_min_min']}-"
                  f"{fig['short_max_min']} minutes at {SHORT:,} steps and two full runs of about {fig['pole_h']} hours, every "
                  "run needed for the table, on eight GPUs"),
        "brief": {"text": brief, "provenance": (
            "written by lane OBS (2026-11-02): the team's question and its table; which runs exist is in the folders and "
            "READMEs; no procedure, no deadline or time words (the study, 2026-10-20); the GPU count stated; nothing names a "
            "test. It asks for the loss each run reports, not a 'final' loss, since the 1,000-step runs stop early by their "
            "script's stop setting (experiments/gpu-tasks/notes/2026-11-02-final-loss-audit.md)")},
        "files": files + [{"path": "data/fineweb10B/", "hub_files": _fineweb(shards),
                           "from": f"kjj0/fineweb10B-gpt2: the validation shard and the first {shards} training shards"}],
        "jobs": jobs,
        "latepole": {"jobs": table,
                     "how": ("per job in the natural order: its recorded run (source.path), budget, seed, the loss its run "
                             "reports at its last validation, paced one-GPU duration and recorded torch")},
        "schedule": dict(fig, how=FIGURES_HOW),
        "scripted_reference": {"path": "reference/latepole_reference.py", "from": REFERENCE_REL,
                               "args": ["--root", ".", "--gpus", str(GPUS)],
                               "how": ("brief-only controller (lane OBS): reads each run's script for the steps it states it "
                                       "trains and each README's launch line, then launches the most stated steps first as "
                                       "GPUs free; no registry, recorded seconds, losses or table of ours. Staged only for "
                                       "scripted runs, never in a model agent's workspace")},
        "measuring_run": {"harness": "scripted", "policy": "controller", "speed": 10.0, "role": "check",
                          "how": ("brief-only: the task's scripted_reference, the controller plan.py's controller policy "
                                  "writes and runs (pkg-k2-latepole's form); the registry's packed is the informed "
                                  "reference beside it")},
        "sources": ["modded-nanogpt track 3, commit 4ea6b937; recorded runs on H100 SXM (H100 80GB HBM3) x8, converted to one "
                    f"GPU; {SHORT:,}-step runs by build/gen_prefix.py's stop change (soap-f1-auxb2 by its own STOP_STEP)"],
        "notes": [
            "Scripts made single-process (build/single_process.py), as G4n, K, H1, H3' and H4: no torch.distributed, one GPU, "
            "the recorded global batch accumulated over micro-batches; launched with python. Timings are the recorded runs "
            "converted to one GPU by gpu-mock's convert().",
            "Every job is its own recorded run (keyed by source.path), never two jobs from one run: soap-f1-auxb2's seeds 11-17 "
            f"run {SHORT:,} steps and seeds 18-19 its full schedule (common.repeated_records; analysis 15:37 MST).",
            "Under the selection guard: no run here is served by the paper's packages or by any package-search candidate (the "
            "five and their twins and variants, K, K2, SF, C, A, D, filter-yield, the tabular packages and the held matrix; "
            "experiments/package-search/lanes/observed/holdings.py). Some are also served by " + ELSEWHERE + ", by "
            "analysis's ruling of 15:37 MST. The source is the same modded-nanogpt track 3 as K and the G4n twins; the twin is "
            "of pkg-k2-latepole, whose source is MLS-Bench.",
            "The root track-3 records are named as the source names them: their folders are the first 8 characters of the "
            "log's uuid (records/track_3_optimization/results/<uuid>.txt). Each is its own script.",
            "Recorded on other stacks than the box's torch 2.11.0, served on its image: " + "; ".join(
                f"torch {k}: {', '.join(sorted(v))}" for k, v in sorted(torch_by.items())) + ". Whether the agent can see the "
            "recorded version is gpu-mock's question, as for K and G4n.",
            f"The lengths are readable from the stated steps: each script's stop setting or schedule length, the folder name "
            f"and the README state them; per-step cost differs {fig['step_cost_spread']} across these scripts, so the stated "
            f"steps rank the true lengths (Spearman {fig['spearman_steps_vs_length']}). An agent that sorts by them reaches "
            "the best schedule; the package measures an agent that doesn't.",
            "The natural order (folders sorted by name, then budget, then seed or record id; the glob order of latepole/*/steps-*/ "
            "and each README's list) puts "
            f"soap-f1-auxb2's two full runs at {fig['poles_at']} of {len(jobs)}. Nothing is reordered (analysis 15:11 MST, "
            "condition 1).",
            "Computed before the screen, ideal times on these jobs (schedule, at 8 GPUs): file-order FCFS "
            f"{fig['fcfs_file']}x the best, round-robin per-GPU lists {fig['lists_round_robin'][0]}x (idle while work waits "
            f"{fig['lists_round_robin'][1]}), contiguous chunks of ceil({len(jobs)}/{GPUS}) {fig['lists_chunked'][0]}x; brief-only by stated steps "
            f"{fig['blind_by_steps']}x; all over the proved optimum, {fig['best_s']} s. These are predictions, not screens."],
        "reference_how": ("bar 2: the brief-only reference is the scripted_reference controller, which ranks runs by the steps "
                          f"their scripts state; in ideal times it is {fig['blind_by_steps']}x the proved optimum "
                          f"(schedule.blind_by_steps), against {fig['lpt_true_lengths']}x for longest first on true lengths. "
                          "The registry's packed (longest disclosed estimate first) is the informed reference"),
    }


def _fineweb(n):
    import gen_tasks as GT
    return GT.fineweb_files(n)


# schedule.how as a literal: a string built from figures.__doc__ differs by Python version (3.13+ strips docstring
# indentation at compile time), which moved the task hash between the controller and the E2
FIGURES_HOW = ("Ideal-time schedules over the jobs in natural order (no host terms, no agent turns), as ratios to the proved "
               "optimum of the paced one-GPU durations (OPTIMUM_S): bar 2 by stated steps (the files' only length signal), "
               "longest first on true lengths, file-order FCFS, and fixed per-GPU lists dealt round-robin or in contiguous "
               "chunks of ceil(jobs / GPUs), each with its idle while a released job waited, over GPUs x makespan.")


def figures(durs, steps, gpus):
    """Ideal-time schedules over the jobs in natural order (no host terms, no agent turns), as ratios to the proved optimum of
    the paced one-GPU durations (OPTIMUM_S): bar 2 by stated steps (the files' only length signal), longest first on true
    lengths, file-order FCFS, and fixed per-GPU lists dealt round-robin or in contiguous chunks, each with its idle while a
    released job waited, over GPUs x makespan."""
    import heapq
    idx = list(range(len(durs)))

    def lpt(order, g):
        free = [(0.0, i) for i in range(g)]
        heapq.heapify(free)
        for k in order:
            t, i = heapq.heappop(free)
            heapq.heappush(free, (t + durs[k], i))
        return max(t for t, _ in free)

    lpt_best = lpt(sorted(idx, key=lambda k: -durs[k]), gpus)
    s = sorted(durs)
    k = -(-len(durs) // gpus)
    lb = max(sum(durs) / gpus, max(durs), sum(s[:k]))
    if not (lb - 1.0 <= OPTIMUM_S <= lpt_best + 1.0):
        raise SystemExit(f"OPTIMUM_S {OPTIMUM_S} is outside [{lb:.1f}, {lpt_best:.1f}]: the job list moved; re-prove it "
                         "(package-search/lanes/observed/olp_optimum.py)")
    best = OPTIMUM_S

    def lists(chunked):
        # chunks: ceil(n / gpus) consecutive jobs per GPU, the split `split -n` or a loop over fixed-size slices writes
        # (T-sim's OLP-2026-11-02.md; the n * gpus // n_jobs boundaries gen_matrix uses put both poles on one GPU here,
        # 2.605 against this 2.131; floor and numpy array_split give 2.462 and 2.304)
        ls = [[] for _ in range(gpus)]
        per = -(-len(durs) // gpus)
        for n, d in enumerate(durs):
            ls[min(gpus - 1, n // per) if chunked else n % gpus].append(d)
        load = [sum(x) for x in ls]
        mk, last_start = max(load), max(sum(x) - x[-1] for x in ls if x)
        return round(mk / best, 3), round(sum(max(0.0, last_start - x) for x in load) / (gpus * mk), 3)
    short = [d for d, st in zip(durs, steps) if st == SHORT]
    poles = [d for d, st in zip(durs, steps) if st != SHORT]
    per_step = [d / st for d, st in zip(durs, steps)]
    rank = sorted(idx, key=lambda k: (-steps[k], k))
    return {"best_s": round(best, 1), "best_how": "proved optimum (CP-SAT, OPTIMAL; lane OBS olp_optimum.py)",
            "lpt_true_lengths": round(lpt_best / best, 4), "lower_bound_s": round(lb, 1),
            "blind_by_steps": round(lpt(rank, gpus) / best, 4), "fcfs_file": round(lpt(idx, gpus) / best, 3),
            "lists_round_robin": lists(False), "lists_chunked": lists(True),
            "poles_at": [i + 1 for i, st in enumerate(steps) if st != SHORT],
            "short_min_min": round(min(short) / 60), "short_max_min": round(max(short) / 60),
            "pole_h": round(statistics.mean(poles) / 3600, 1) if poles else None,
            "step_cost_spread": f"{min(per_step):.2f}-{max(per_step):.2f} s per step",
            "spearman_steps_vs_length": round(__import__("gen_matrix").spearman(steps, durs), 3)}


def figures_main(reg):
    recs = C.load_records(os.path.join(reg, "records"))
    t = build(recs, lambda rids, recs, on: {}, reg)
    print(json.dumps({"jobs": len(t["jobs"]), "order": [j["job_id"] for j in t["jobs"]], "schedule": t["schedule"]}, indent=1))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "figures":
        figures_main(sys.argv[2])
    else:
        main()
