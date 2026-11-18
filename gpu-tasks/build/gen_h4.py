"""H4, pkg-h4-filter-ablation: a FineWeb-style filter-recipe ablation whose data preparation is real CPU work in the box
.

usage: python3 -I build/gen_h4.py <registry>      (records, scripts and the preparation tool; the task is gen_h4.build, called
                                                   from gen_tasks, or alone through build/finish_one_task.py <registry> gen_h4)

What the agent is asked: which of seven quality-filter recipes trains the best small GPT, three runs per recipe. What it has to
find in the files (README.md, tools/prepare.py): each recipe's data must be made first, by filtering the raw FineWeb pool with
datatrove and tokenising what is kept into runs/<recipe>/data/fineweb10B/, and each recipe's training script reads only its own
recipe's data. The filtering is real: FineWeb's own filters (datatrove 0.10.1, FineWeb's configuration) over 1,160,712
documents, 6-17 minutes per recipe on the box's 42 cores and hours single-process, which is datatrove's default (tasks=1). The
training is replayed: one GPU per job, single-process as a fact of the code.

The jobs. Every recipe's three runs are contra-soft-muon (modded-nanogpt track 3, 20260509) to step 875: the recorded prefix
records (build/gen_prefix.py, k=875, 5 training shards, about 23 minutes on one GPU), each with gen_dataprep's start-up data
check and made single-process (build/single_process.py), in that order. Each (recipe, seed) is served a different recorded run
of that one configuration, so no run is served twice and every recipe's curves are real curves.

What is synthesised, and says so (derived_from.kind = synthesised_filter_variant): a recorded run trained on kjj0's shards,
which no recipe here reproduces. Serving it as the run that trained on a recipe's output makes the recipes' losses, and so the
winning recipe, fiction; the GPU time, the CPU filtering and the data each job needs before it can start are real. The
differences between recipes are run-to-run variation of one configuration (sd 0.0015 in validation loss at step 875 over the 30
recorded runs).

Seeds: the daemon matches a launch's --seed against a record's identity.seed.value, so each
recipe's records carry the argv seed (0, 1, 2) there and the recorded run's own seed in derived_from.seed_map. The nine runs
left over are spare seeds (3 for every recipe, 4 for the first two), so a run the agent adds is a real one it has not seen.
"""
import copy
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_dataprep as DP  # noqa: E402
import single_process as SP  # noqa: E402

TASK_ID = "pkg-h4-filter-ablation"
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT_GROUP = "mng3-cfg-d649f96e97a1"         # 20260509_contra_soft_muon, 30 seeds on torch 2.11
STOP = 875                                     # gen_prefix's k: 5 training shards, about 23 min on one GPU
SEEDS = (0, 1, 2)                              # the brief's seeds, per recipe
# recipe -> (its datatrove filters, the reference preparation's wall time on c3-highmem-44 in seconds)
# Timings: the scripted agent's own preparation steps in W0-h4-measure: c3-8, the 37c9
# mng3-filter box under run_one a66a5302, speed 1, tools/prepare.py <recipe> --tasks 512 --workers 42, wall time from the tool
# call to its result, every step exit 0. That run's training lookups missed (gpu-mock's entry-script fix), which does not touch
# the preparations, since they run before any training. The first measurement, bare in the mng3-dataprep image on c3-16 and -17
# on 2026-10-23 (experiments/cpu-tasks/survey/scripts/h4-2026-10-23/results-2026-10-23/), was 1-8% lower for six recipes and
# 22.6 s lower for 'none' (box and tool-call overhead on a 36 s step): none 36.3, gopher-repetition 440.0, gopher-quality
# 362.9, c4 444.6, fineweb-quality 341.9, gopher 467.6, fineweb 1,055.1.
RECIPES = {
    "none": ([], 58.9),
    "gopher-repetition": (["GopherRepetitionFilter"], 463.6),
    "gopher-quality": (["GopherQualityFilter"], 385.0),
    "c4": (["C4QualityFilter"], 449.4),
    "fineweb-quality": (["FineWebQualityFilter"], 369.2),
    "gopher": (["GopherRepetitionFilter", "GopherQualityFilter"], 485.2),
    "fineweb": (["GopherRepetitionFilter", "GopherQualityFilter", "C4QualityFilter", "FineWebQualityFilter"], 1086.7),
}
# what each recipe keeps of the pool (same run; experiments/cpu-tasks/survey/scripts/h4-2026-10-23/results-2026-10-23/
# recipes-cur.json, on worker-9 with 42 workers): documents and GPT-2 tokens with each document's <|endoftext|>
KEPT = {"none": (1160712, 799997818), "gopher-repetition": (1157249, 797668984), "gopher-quality": (993672, 704740434),
        "c4": (1158733, 799759451), "fineweb-quality": (1155381, 796666115), "gopher": (992235, 703689983),
        "fineweb": (987586, 701230928)}
PREP_CMD = "python tools/prepare.py {recipe} --tasks 512 --workers 42"
PREP_ALLOWANCE = 2.0                           # horizon allowance per recipe: twice the reference preparation, at least 600 s
TAG = "h4"
TOOL_SRC = os.path.join(HERE, "h4_prepare_tool.py")
TOOL_REL = "artifacts/mng3/h4/prepare.py"
CONSTRAINTS_SRC = os.path.join(HERE, "h4_constraints.txt")
CONSTRAINTS_REL = "artifacts/mng3/env/torch-2.11.0-py3.11-filter/constraints.txt"
H4_PACKAGES = ("datatrove[processing]==0.10.1", "spacy==3.8.16", "orjson==3.13.0")
N_POOL_FILES = 512
# build/gen_h4_pool.py's output from kjj0's train shards 1..8 and GPT-2's vocab.json (DATAPREP_VOCAB): filled from its first
# run on a VM; until then stage_task.py cannot stage the pool, and says so
POOL_OUTPUTS_FILE = os.path.join(HERE, "h4_pool_outputs.json")


def recipe_script(prefix_code, files, recipe):
    """The training script of one recipe: the prefix script with gen_dataprep's data check, made single-process, with one header
    line naming the recipe (so each recipe's script is its own bytes, and its records bind to it exactly)."""
    code = SP.single_process(DP.data_check(prefix_code, files))
    anchor = "# Data check (gpu-tasks"
    if code.count(anchor) != 1:
        raise ValueError("the data check is not in the script exactly once")
    chdir = ("# Work in this script's own directory (cpu-tasks, 2026-10-23): its data/ and experiment-logs/ are the recipe's, "
             "wherever it is launched from.\nos.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))\n")
    code = code.replace(anchor, chdir + anchor, 1)
    line = (f"# Filter recipe '{recipe}': trains on runs/{recipe}/data/fineweb10B/, made by `python tools/prepare.py {recipe}` "
            "from the project root.\n")
    return line + code


def prefixes(recs):
    out = sorted((r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "prefix"
                  and r["identity"]["derived_from"]["parent_replicate_group"] == PARENT_GROUP
                  and r["identity"]["derived_from"]["stop_after_step"] == STOP and r["readiness"]["replay"]),
                 key=lambda r: r["identity"]["seed"]["value"])
    if len(out) < len(RECIPES) * len(SEEDS):
        raise SystemExit(f"{len(out)} contra-soft-muon prefix records at step {STOP}; need {len(RECIPES) * len(SEEDS)} "
                         "(run build/gen_prefix.py first)")
    return out


def assignment(ps):
    """(recipe, argv seed, prefix record, role): the primary runs, recipe by recipe and seed by seed (argv seeds 0-2), then the
    left-over runs as spare seeds, round-robin over the recipes in README order: --seed 3 for every recipe, then --seed 4 for
    as many as are left. A launch with a spare seed is served a run of its own; an argv seed with no record falls to the
    recipe's unused records (gpu-mock's seed_substituted), never one served before."""
    rows, i = [], 0
    for recipe in RECIPES:
        for s in SEEDS:
            rows.append((recipe, s, ps[i], "primary"))
            i += 1
    names = list(RECIPES)
    for k, p in enumerate(ps[i:]):
        rows.append((names[k % len(names)], len(SEEDS) + k // len(names), p, "spare"))
    return rows


def record(prefix, recipe, argv_seed, role, code, script_rel):
    r = copy.deepcopy(prefix)
    ident = r["identity"]
    sigma = prefix["identity"]["seed"]["value"]
    bound = f"runs/{recipe}/train_gpt.py"
    files = list(DP.data_files(prefix))
    ident.update({
        "record_id": f"{prefix['identity']['record_id']}-{TAG}-{recipe}",
        "entry_hash": C.sha256_text(code),
        "argv": ["train_gpt.py", "--seed", str(argv_seed)],
        "argv_how": "the recipe's script with the brief's seed: `python runs/<recipe>/train_gpt.py --seed N`, from any directory",
        "bound_path": bound, "match_policy": "path_bound",
        "match_policy_how": ("exact on the recipe script's bytes and argv, else served to a script whose resolved path ends with /"
                             + bound + " (gpu-mock's path_bound rule, 2026-10-23 15:41Z): an agent that edits the script stays "
                             "on its recipe's runs"),
        "replicate_group": f"mng3-{TAG}-{recipe}",
        "seed": {"value": argv_seed, "provenance": "derived",
                 "how": (f"the argv seed this record answers (gpu-mock matches --seed against identity.seed.value); the run served "
                         f"is recorded seed {sigma} (derived_from.seed_map)")},
        "process_model": {"value": "single", "how": "build/single_process.py"},
        "derived_from": {**prefix["identity"]["derived_from"], "kind": "synthesised_filter_variant",
                         "via": prefix["identity"]["record_id"], "bound_path": bound, "recipe": recipe, "role": role,
                         "files": files, "seed_map": {"argv_seed": argv_seed, "recorded_seed": sigma},
                         "change": ("the step-875 prefix script with the start-up data check (build/gen_dataprep.py), made "
                                    "single-process (build/single_process.py), then an os.chdir to the script's own directory "
                                    "before the data check and a one-line header naming the recipe (gen_h4.recipe_script)"),
                         "synthesised": (f"the recorded run trained on kjj0/fineweb10B-gpt2's shards; served as a run on "
                                         f"recipe '{recipe}''s filtered data, which it never saw. Its losses are real values of "
                                         "a real run, and the recipe comparison they make is fiction"),
                         "how": "every series value, step time and memory figure is the prefix record's, recorded or derived"},
    })
    ident["trajectory"]["config_hash"] = C.sha256_text(f"{TAG}\0{recipe}")
    ident["trajectory"]["config_hash_how"] = f"sha256 of '{TAG}' and the recipe: one group per recipe"
    r["launch"]["process_model"] = C.tv("single", "derived", "build/single_process.py made the script single-process")
    r["host"] = dict(r["host"], processes=C.tv(1, "derived", "build/single_process.py: one process on one GPU"))
    r["agent_view"] = dict(r["agent_view"], script=script_rel,
                           launch_command=f"python runs/{recipe}/train_gpt.py --seed {argv_seed}",
                           inputs=[f"runs/{recipe}/data/fineweb10B/ (tools/prepare.py {recipe})"])
    r["readiness"] = C.readiness(r)
    return r


def check_record(r, reg):
    """validate.py's check for a synthesised_filter_variant record: its script is its prefix's through recipe_script, it
    serves exactly its prefix record's run, and it is path_bound to its recipe's script. Returns a list of errors."""
    d = r["identity"]["derived_from"]
    via_p = os.path.join(reg, "records", d.get("via", "") + ".json")
    if not os.path.exists(via_p):
        return [f"derived_from.via {d.get('via')} not in the registry"]
    v = json.load(open(via_p))
    errs = []
    if d.get("recipe") not in RECIPES:
        errs.append(f"derived_from.recipe {d.get('recipe')!r} is not a recipe")
        return errs
    try:
        want = recipe_script(open(os.path.join(reg, v["agent_view"]["script"])).read(), list(DP.data_files(v)), d["recipe"])
    except ValueError as e:
        return [f"the prefix script does not take H4's changes: {e}"]
    if C.sha256_text(want) != r["identity"]["entry_hash"]:
        errs.append("the script is not its prefix's through gen_h4.recipe_script")
    if r["series"] != v["series"] or r["steps"] != v["steps"] or r["timing"] != v["timing"]:
        errs.append("a synthesised_filter_variant record must serve exactly its prefix record's run")
    if r["identity"]["match_policy"] != "path_bound" or r["identity"].get("bound_path") != f"runs/{d['recipe']}/train_gpt.py":
        errs.append("a synthesised_filter_variant record must be path_bound to runs/<recipe>/train_gpt.py")
    if (d.get("seed_map") or {}).get("recorded_seed") != v["identity"]["seed"]["value"]:
        errs.append("derived_from.seed_map.recorded_seed is not the prefix run's seed")
    if (d.get("seed_map") or {}).get("argv_seed") != r["identity"]["seed"]["value"]:
        errs.append("identity.seed.value is not derived_from.seed_map.argv_seed")
    return errs


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    ps = prefixes(recs)
    code0 = open(os.path.join(reg, ps[0]["agent_view"]["script"])).read()
    for p in ps[1:]:
        if p["agent_view"]["script"] != ps[0]["agent_view"]["script"]:
            raise SystemExit(f"seed {p['identity']['seed']['value']} binds another script")
    files = list(DP.data_files(ps[0]))
    scripts = {}
    for recipe in RECIPES:
        code = recipe_script(code0, files, recipe)
        rel = f"artifacts/mng3/{TAG}/{recipe}/train_gpt.py"
        C.write_text(os.path.join(reg, rel), code)
        scripts[recipe] = (code, rel)
    os.makedirs(os.path.dirname(os.path.join(reg, TOOL_REL)), exist_ok=True)
    shutil.copyfile(TOOL_SRC, os.path.join(reg, TOOL_REL))
    os.makedirs(os.path.dirname(os.path.join(reg, CONSTRAINTS_REL)), exist_ok=True)
    shutil.copyfile(CONSTRAINTS_SRC, os.path.join(reg, CONSTRAINTS_REL))
    n = 0
    for recipe, s, p, role in assignment(ps):
        code, rel = scripts[recipe]
        r = record(p, recipe, s, role, code, rel)
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        n += 1
    print(f"h4 records: {n} ({len(RECIPES) * len(SEEDS)} primary); scripts under artifacts/mng3/{TAG}/; tool {TOOL_REL}")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------------------------------------------------------
# The task (gen_tasks.main calls build; complete_task then adds stats, horizon, disclosure and task class)

README = """# Filter-recipe ablation

We're deciding which quality filter to put in front of pretraining. Each recipe below is a set of FineWeb's own text filters
(datatrove, with FineWeb's settings); a recipe's data is the raw pool with the recipe's filters applied, tokenised for GPT-2.

## Layout

- `data/pool/`: the raw pool, {n_files} JSONL files of FineWeb documents (`{{"text": ...}}` per line), 1,160,712 documents in
  all, about 800M GPT-2 tokens.
- `data/val/fineweb_val_000000.bin`: the validation set, GPT-2 tokens, the same for every recipe.
- `tools/prepare.py`: makes one recipe's training data (below).
- `runs/<recipe>/train_gpt.py`: the training script for that recipe. It trains our small GPT (12 layers, 768 wide) on that
  recipe's data and stops after the validation at step 875, printing the validation loss.

## Recipes

| Recipe | Filters |
|---|---|
{recipe_rows}

## Making a recipe's data

From the project root: `python tools/prepare.py <recipe>`. It filters `data/pool/` with the recipe's filters (datatrove's
`LocalPipelineExecutor`), tokenises what is kept, and writes:

- `runs/<recipe>/filtered/`: the documents the recipe keeps;
- `runs/<recipe>/data/fineweb10B/`: `fineweb_train_000001.bin`, `fineweb_train_000002.bin`, ... (100M tokens each) and a copy of
  `fineweb_val_000000.bin`.

`python tools/prepare.py --help` lists its options.

## Training

Each run of a recipe is one GPU and one process:

    python runs/<recipe>/train_gpt.py --seed <seed>

The script works in its own directory, `runs/<recipe>/`, wherever it is launched from. It reads `data/fineweb10B/` there: the
validation shard and training shards 1 to 5. It checks at start-up that those files are there and complete, and stops with the
missing file's name if not. It writes its log under `runs/<recipe>/experiment-logs/`.
"""


def readme():
    rows = "\n".join(f"| `{r}` | " + (", ".join(f"`{f}`" for f in fs) if fs else "none: the pool as it is") + " |"
                     for r, (fs, _) in RECIPES.items())
    return README.format(n_files=N_POOL_FILES, recipe_rows=rows)


def environment(recs, jobs):
    """The mng3-dataprep environment (torch 2.11 CPU, numpy 2.4.4, tiktoken with its offline cache) plus datatrove and what its
    filters need, as resolved in that image on c3-highmem-44."""
    import gen_tasks as GT
    t = {"jobs": jobs}
    GT.add_mng3_dataprep_environment(t, recs)
    env = t["environment"]
    inst = env["install"]
    inst["packages"] = inst["packages"] + list(H4_PACKAGES)
    inst["constraints"] = CONSTRAINTS_REL
    inst["line"] = "pip install -c constraints.txt " + " ".join(inst["packages"])
    inst["cpu_lines"] = inst["cpu_lines"][:1] + [inst["cpu_lines"][1] + " " + " ".join(H4_PACKAGES)]
    inst["how"] += ("; datatrove 0.10.1 with its processing extra (the filters' word tokenizer needs spaCy; spacy.blank, no model "
                    "download) and orjson (its JsonlReader), resolved under numpy 2.4.4 and torch 2.11.0 in the mng3-dataprep "
                    "image: the constraints file is that venv's full pip freeze (cpu-tasks, 2026-10-23)")
    env["verify_imports"] = ["torch", "numpy", "tiktoken", "datatrove", "spacy", "orjson"]
    env["image_label"] = {"value": "mng3-filter",
                          "how": "mng3-dataprep plus datatrove, spaCy and orjson: a thin layer of its own (isolation, 2026-10-23)"}
    return env


def build(recs, disclosure_of, reg=None):
    import gen_tasks as GT
    rows = [r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == "synthesised_filter_variant"]
    if not rows:
        raise SystemExit(f"{TASK_ID}: no synthesised_filter_variant records (run build/gen_h4.py first)")
    by = {}
    for r in rows:
        d = r["identity"]["derived_from"]
        by.setdefault((d["recipe"], d["role"]), []).append(r)
    jobs, files = [], [{"path": "README.md", "content": readme()},
                       {"path": "tools/prepare.py", "from": TOOL_REL,
                        "how": "cpu-tasks' preparation tool: datatrove's LocalPipelineExecutor with the recipe's filters, then "
                               "data/fineweb.py's tokenisation and shard format"}]
    for recipe, (filters, prep_s) in RECIPES.items():
        prim = sorted(by[(recipe, "primary")], key=lambda r: r["identity"]["seed"]["value"])
        files.append({"path": f"runs/{recipe}/train_gpt.py", "from": prim[0]["agent_view"]["script"]})
        group = [f"{recipe}-seed{s}" for s in SEEDS]
        produces = [f"runs/{recipe}/{f}" for f in prim[0]["identity"]["derived_from"]["files"]]
        for i, r in enumerate(prim):
            s = r["identity"]["seed"]["value"]
            jid = f"{recipe}-seed{s}"
            j = {"job_id": jid, "label": f"recipe {recipe}, seed {s}: one GPU, to step {STOP}, on runs/{recipe}/data/",
                 "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                 "draw": "exact_seed", "seed": s, "recorded_gpus": 1, "recorded_gpu": "NVIDIA H100 80GB HBM3",
                 "batch_policy": r["launch"]["batch_policy"]["value"], "gpu_options": [1],
                 "launch": f"python runs/{recipe}/train_gpt.py --seed {s}", "after": [], "consumes": [],
                 "produces": [],
                 "prep": {"kind": "data", "shared_by": group, "paths": [f"runs/{recipe}/"], "produces": produces,
                          "scripted_command": PREP_CMD.format(recipe=recipe), "scripted_s": prep_s,
                          "horizon_s": round(max(600.0, PREP_ALLOWANCE * prep_s), 1) if i == 0 else 0.0,
                          "filters": filters, "kept_docs": KEPT[recipe][0], "kept_tokens": KEPT[recipe][1],
                          "how": ("real CPU work in the box: the recipe's datatrove filters over data/pool/, then tokenisation "
                                  "into runs/<recipe>/data/fineweb10B/; one preparation serves the recipe's three jobs "
                                  "(shared_by). scripted_s is the reference command's wall time in the task's own box on "
                                  "c3-highmem-44 at 42 workers (W0-h4-measure, 2026-10-24); horizon_s, the allowance, sits on "
                                  "the group's first job only, so add_horizon counts each recipe once")}}
            jobs.append(j)
    train_only = GT.fineweb_files(8)
    train_only = dict(train_only, files={k: v for k, v in train_only["files"].items() if k.startswith("fineweb_train_")})
    outputs = json.load(open(POOL_OUTPUTS_FILE)) if os.path.exists(POOL_OUTPUTS_FILE) else {}
    files += [
        {"path": "data/val/", "hub_files": GT.fineweb_files(0),
         "from": "kjj0/fineweb10B-gpt2: the validation shard, unchanged"},
        {"path": "data/pool/", "generated": {"by": "build/gen_h4_pool.py", "args": [str(N_POOL_FILES)],
                                             "inputs": {"shards": train_only, "vocab": GT.DATAPREP_VOCAB},
                                             "outputs": outputs},
         "from": (f"the documents of kjj0/fineweb10B-gpt2's train shards 1..8, decoded with GPT-2's vocabulary, whole documents "
                  f"only, split over {N_POOL_FILES} files in order; build/stage_task.py makes them and checks each sha256")},
    ]
    brief = ("We're choosing a quality-filtering recipe for our pretraining data. README.md describes the project and the seven "
             "recipes we're comparing: which one gives the best model? Train our small GPT on each recipe's data three times "
             "(seeds 0, 1 and 2), and report every run's final validation loss, each recipe's mean, and the recipe you would "
             "pick. The machine has four H100s.")
    return {
        "task_id": TASK_ID, "rung": 2, "level": "hidden_plan", "arm": "data-preparation", "set": "development",
        "set_how": "the analysis, 2026-10-23: package search; its fresh twin draws other records",
        "title": ("H4: seven filter recipes, each prepared by real CPU filtering and tokenisation in the box, then three one-GPU "
                  "training runs per recipe on that recipe's data"),
        "box": {**GT.BOX, "disk": "80g",
                "disk_how": ("the runner's driver passes it to run_one --disk (2026-10-23): the pool is 3.4 GB and every recipe's "
                             "filtered copy and shards about 5 GB, so all seven kept is about 40 GB; 80g leaves room, so a full "
                             "disk is not what this task tests")},
        "brief": {"text": brief, "provenance": (
            "written by cpu-tasks for the package search (the analysis, 2026-10-23): the question and the "
            "deliverable, not the runs; the preparation, the per-recipe directories and the start-up data check are in "
            "README.md and tools/prepare.py; no procedure, no hint about parallelism, no deadline or time words; the GPU count "
            "stated")},
        "files": files,
        "jobs": jobs,
        "environment": environment(recs, jobs),
        "measuring_run": {"harness": "scripted", "policy": "packed", "speed": 1.0, "plan": "jobs",
                          "terms_from": ["pkg-h1-study", "pkg-dataprep-p6-q"],
                          "how": ("the runner's packed preparation pool (plan.py prep_step): prepare one recipe in the "
                                  "foreground, launch its prepared jobs as GPUs free, then the next recipe; recipes in task "
                                  "order. Start-up terms transfer from the single-process mng3 packages; the preparations "
                                  "are measured, not transferred")},
        "sources": ["modded-nanogpt track 3 at 4ea6b937, 20260509_contra_soft_muon: 30 recorded runs on torch 2.11.0, cut at "
                    f"step {STOP} (build/gen_prefix.py) and converted to one GPU by gpu-mock's convert()",
                    "datatrove 0.10.1's GopherRepetitionFilter, GopherQualityFilter, C4QualityFilter(filter_no_terminal_punct="
                    "False) and FineWebQualityFilter: FineWeb's configuration (datatrove examples/fineweb.py)",
                    "kjj0/fineweb10B-gpt2 at 889765ea: the validation shard, and the documents of train shards 1..8"],
        "notes": [
            "Synthesised, disclosed: every job's curve is a real recorded run of one configuration (contra-soft-muon), trained "
            "on kjj0's unfiltered shards and served as a run on a recipe's filtered data (derived_from.kind = "
            "synthesised_filter_variant, derived_from.synthesised). So the recipes' losses and the winning recipe are fiction; "
            "the GPU time, the CPU filtering and tokenisation, and the dependency of each job on its recipe's data are real. "
            "Recipe differences are run-to-run variation (sd 0.0015 at step 875 over the 30 runs).",
            "The preparation is real CPU work, measured in the task's own box on c3-highmem-44 at 42 workers (W0-h4-measure, "
            "2026-10-24): 59 s for 'none' (tokenising only), 369-485 s for the one- and two-filter recipes, 1,087 s for all "
            "four, 3,298 s in all. datatrove's own default, "
            "tasks=1, runs one process: about 40 times longer for the filter stage. The tokeniser defaults to os.cpu_count() - 2 "
            "workers, as data/fineweb.py. Each recipe keeps 7.0 to 8.0 training shards; every job reads 5.",
            "Scripts made single-process (build/single_process.py): one GPU per job is a fact of the code. Each recipe's script "
            "is the step-875 prefix script with gen_dataprep's data check, an os.chdir to its own directory before that check, and "
            "a one-line header naming the recipe, so it stops at start-up with the missing file's name until its recipe's data "
            "exist, wherever it is launched from. Launched `python runs/<recipe>/train_gpt.py --seed N`; a torchrun launch is "
            "paced at one GPU too (gpu-mock's process_model rule).",
            "Every path a record binds is named in README.md: runs/<recipe>/train_gpt.py and runs/<recipe>/data/fineweb10B/ "
            "(the analysis, 2026-10-23 14:35 MDT, from H1's first screen).",
            "Seeds: each recipe's records answer --seed 0, 1 and 2 (identity.seed.value) and serve recorded seeds "
            "3r, 3r+1, 3r+2 for the r-th recipe in README order (derived_from.seed_map). The nine left-over runs are spare "
            "seeds: --seed 3 for every recipe, --seed 4 for the first two (derived_from.role = spare), so an agent that adds a "
            "run gets a real, unseen one. No run is served twice. rerun_siblings come from build/siblings.py, whose pools do "
            "not include this kind; a relaunch with the same seed is served the same run (deterministic, as a real rerun).",
            "Each script writes experiment-logs/.../<uuid>.txt under its recipe's directory and reads only that directory's "
            "data/fineweb10B/, so no two jobs collide; prepare.py rewrites only runs/<recipe>/."],
    }
