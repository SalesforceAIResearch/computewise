"""Yield-constrained filter adoption, owned by independent Sol search.

main() writes a record/artifact overlay; build() returns the unfinished task.
All seven preparations are real. Training curves are synthesised assignments of
distinct existing runs to filtered datasets, disclosed exactly as in H4.
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gen_dataprep as DP
import gen_h4 as H4

TASK_ID = "pkg-filter-yield"
ROOT = "yieldlab"
KIND = "synthesised_yield_variant"
SEEDS = (0, 1, 2, 3)
STOP = 875
MIN_RETENTION = 0.95
RAW_TOKENS = H4.KEPT["none"][1]
TOOL_REL = "artifacts/mng3/sol-yield/prepare.py"
REFERENCE_REL = "artifacts/mng3/sol-yield/filter_yield_reference.py"


def assignments(recs):
    ps = H4.prefixes(recs)
    ps = sorted(ps, key=lambda r: (r["identity"]["seed"]["value"], r["identity"]["record_id"]))
    need = len(H4.RECIPES) * len(SEEDS)
    if len(ps) < need:
        raise ValueError(f"need {need} distinct contra-soft-muon prefix runs, got {len(ps)}")
    ps = ps[:need]
    paths = [r["source"]["path"] for r in ps]
    if len(paths) != len(set(paths)):
        raise ValueError("prefix pool repeats a recorded source.path")
    if any(C.paced_duration(p, 1) < 1200 for p in ps):
        raise ValueError("a training job falls below 20 paced minutes on one GPU")
    return [(recipe, seed, ps[i * len(SEEDS) + seed])
            for i, recipe in enumerate(H4.RECIPES) for seed in SEEDS]


def script_text(prefix, reg, recipe):
    code = open(os.path.join(reg, prefix["agent_view"]["script"])).read()
    return H4.recipe_script(code, list(DP.data_files(prefix)), recipe).replace("runs/", ROOT + "/")


def record(prefix, reg, recipe, seed):
    code = script_text(prefix, reg, recipe)
    rel = f"artifacts/mng3/sol-yield/{recipe}/train_gpt.py"
    r = H4.record(prefix, recipe, seed, "primary", code, rel)
    ident = r["identity"]
    ident["record_id"] = prefix["identity"]["record_id"] + "-solyield-" + recipe
    ident["replicate_group"] = "mng3-solyield-" + recipe
    ident["bound_path"] = f"{ROOT}/{recipe}/train_gpt.py"
    ident["match_policy_how"] = "exact on script and seed, else path-bound to " + ident["bound_path"]
    ident["trajectory"]["config_hash"] = C.sha256_text("solyield\0" + recipe)
    ident["trajectory"]["config_hash_how"] = "sha256 of solyield and the recipe; no shared binding root with H4"
    d = ident["derived_from"]
    d.update(kind=KIND, bound_path=ident["bound_path"],
             change="H4.recipe_script, with runs/ changed to yieldlab/; original data check, single-process conversion, chdir and header",
             synthesised=("Recorded contra-soft-muon run on kjj0's unfiltered shards assigned to this recipe's filtered data. "
                          "No claim that this run trained on that data. Recipe loss differences are synthesised; token retention, "
                          "CPU filtering, dependencies and recorded GPU times are real."))
    r["agent_view"].update(launch_command=f"python {ROOT}/{recipe}/train_gpt.py --seed {seed}",
                           inputs=[f"{ROOT}/{recipe}/data/fineweb10B/ (tools/prepare.py {recipe})"])
    r["readiness"] = C.readiness(r)
    return r, code


def check_record(r, reg):
    d = r["identity"]["derived_from"]
    p = os.path.join(reg, "records", d.get("via", "") + ".json")
    if not os.path.exists(p):
        return ["missing prefix " + p]
    prefix = json.load(open(p))
    recipe = d.get("recipe")
    if recipe not in H4.RECIPES:
        return ["unknown recipe " + str(recipe)]
    seed = r["identity"]["seed"]["value"]
    if seed not in SEEDS:
        return ["seed is not one of the four declared seeds"]
    want, code = record(prefix, reg, recipe, seed)
    errors = []
    for key in ("series", "steps", "timing", "source", "result", "failures"):
        if r[key] != prefix[key]:
            errors.append(f"{key} differs from the recorded prefix")
    if r["identity"] != want["identity"]:
        errors.append("identity does not match the declared transformation")
    path = os.path.join(reg, r["agent_view"]["script"])
    if not os.path.exists(path) or C.sha256_text(open(path).read()) != C.sha256_text(code):
        errors.append("artifact script differs from the declared transformation")
    if C.paced_duration(r, 1) < 1200:
        errors.append("training job shorter than 20 paced minutes")
    return errors


def derive(reg):
    recs = C.load_records(os.path.join(reg, "records"))
    collisions = [r["identity"]["record_id"] for r in recs.values()
                  if (r["identity"].get("derived_from") or {}).get("bound_path", "").startswith(ROOT + "/")
                  and (r["identity"].get("derived_from") or {}).get("kind") != KIND]
    if collisions:
        raise ValueError("binding-root collision: " + str(collisions))
    for recipe, seed, prefix in assignments(recs):
        r, code = record(prefix, reg, recipe, seed)
        C.write_text(os.path.join(reg, r["agent_view"]["script"]), code)
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
    tool = open(H4.TOOL_SRC).read().replace("runs/", ROOT + "/").replace('"runs"', '"' + ROOT + '"')
    C.write_text(os.path.join(reg, TOOL_REL), tool)
    reference_src = os.path.join(os.path.dirname(__file__), "..", "..", "package-search", "sol-independent",
                                 "filter_yield_reference.py")
    C.write_text(os.path.join(reg, REFERENCE_REL), open(reference_src).read())
    print(f"{TASK_ID}: wrote {len(H4.RECIPES) * len(SEEDS)} records and seven scripts; root {ROOT}/")


def build(recs, disclosure_of, reg=None):
    if reg is None:
        raise ValueError("registry path is required")
    base = H4.build(recs, disclosure_of, reg)
    rows = [r for r in recs.values() if (r["identity"].get("derived_from") or {}).get("kind") == KIND]
    if len(rows) != len(H4.RECIPES) * len(SEEDS):
        raise ValueError("run gen_filter_yield.py first: expected 28 derived records")
    by = {}
    for r in rows:
        d = r["identity"]["derived_from"]
        by[d["recipe"], r["identity"]["seed"]["value"]] = r
    accepted_yield = [x for x in H4.RECIPES if H4.KEPT[x][1] / RAW_TOKENS >= MIN_RETENTION]
    jobs, unneeded, preps, files, answer = [], [], [], [], {}
    for recipe, (filters, prep_s) in H4.RECIPES.items():
        rs = [by[recipe, s] for s in SEEDS]
        bound = f"{ROOT}/{recipe}/train_gpt.py"
        files.append({"path": bound, "from": rs[0]["agent_view"]["script"]})
        group = [f"{recipe}-seed{s}" for s in SEEDS]
        prep = {"kind": "data", "shared_by": group, "paths": [f"{ROOT}/{recipe}/"],
                "produces": [f"{ROOT}/{recipe}/{x}" for x in rs[0]["identity"]["derived_from"]["files"]],
                "scripted_command": f"python tools/prepare.py {recipe} --tasks 512 --workers 42",
                "scripted_s": prep_s, "horizon_s": max(600, 2 * prep_s),
                "how": "H4's real preparation, relocated only; timing transferred from its c3-highmem-44 measurement"}
        preps.append({"recipe": recipe, "required": True, **prep,
                      "gpu_independent": True,
                      "observes": f"token counts in {ROOT}/{recipe}/data/fineweb10B/fineweb_train_*.bin headers"})
        losses = []
        for seed, r in zip(SEEDS, rs):
            j = {"job_id": group[seed], "label": f"{recipe}, seed {seed}, step {STOP}",
                 "replicate_group": r["identity"]["replicate_group"], "records": [r["identity"]["record_id"]],
                 "draw": "exact_seed", "seed": seed, "recorded_gpus": r["reference"]["gpu_count"],
                 "recorded_gpu": r["reference"]["gpu"], "batch_policy": r["launch"]["batch_policy"]["value"],
                 "gpu_options": [1], "launch": f"python {bound} --seed {seed}", "after": [],
                 "consumes": [], "produces": [], "setup": recipe,
                 "prep": dict(prep, horizon_s=prep["horizon_s"] if seed == 0 else 0)}
            j["disclosure"] = disclosure_of(j["records"], recs, f"{j['recorded_gpus']}x {j['recorded_gpu']}")
            (jobs if recipe in accepted_yield else unneeded).append(j)
            losses.append(r["result"]["final_val_loss"]["value"])
        answer[recipe] = {"kept_tokens": H4.KEPT[recipe][1], "retention": H4.KEPT[recipe][1] / RAW_TOKENS,
                          "passes_yield": recipe in accepted_yield, "losses": losses,
                          "mean_final_val_loss": sum(losses) / len(losses)}
    # All CPU preparations count in the horizon, including rejected recipes.
    rejected_allowance = sum(p["horizon_s"] for p in preps if p["recipe"] not in accepted_yield)
    jobs[0]["prep"]["horizon_s"] += rejected_allowance
    jobs[0]["prep"]["horizon_how"] = "includes CPU preparation allowances for rejected recipes, which have no required GPU job"
    keep_files = [copy.deepcopy(f) for f in base["files"]
                  if f["path"].startswith("data/")]
    readme = H4.readme().replace("runs/", ROOT + "/")
    readme += ("\n## Adoption criterion\n\nThe raw pool contains " + str(RAW_TOKENS) +
               " GPT-2 tokens, including document separators. A recipe is usable only if its prepared training shards "
               "retain at least 95% of those tokens. Among usable recipes choose the one with the lowest mean "
               f"validation loss at step {STOP} over seeds 0, 1, 2 and 3. Shard headers store the number of tokens; partial final shards count. "
               "The seed list applies to every recipe. The data and code remain available for inspection.\n")
    files += [{"path": "README.md", "content": readme},
              {"path": "tools/prepare.py", "from": TOOL_REL,
               "how": "H4 preparation tool with its output root relocated to yieldlab/"}] + keep_files
    winner = min(accepted_yield, key=lambda x: answer[x]["mean_final_val_loss"])
    return {"task_id": TASK_ID, "rung": 3, "goal_level": True, "level": "goal_only", "arm": "data-preparation",
            "set": "development", "set_how": "independent Sol package exploration, the study 2026-11-02",
            "title": "Yield-constrained FineWeb filter adoption: real CPU preparations and optional GPU training",
            "box": copy.deepcopy(base["box"]),
            "brief": {"text": "We're choosing a quality-filter recipe for our pretraining data. README.md describes the seven "
                      "recipes. A usable recipe must retain at least 95% of the raw pool's GPT-2 tokens, including document "
                      "separators. Among usable recipes, which gives the lowest mean validation loss at "
                      f"step {STOP} over seeds 0, 1, 2 and 3? Report the retention counts, the runs and means your choice rests on, and your recommendation. "
                      "The machine has four H100s.",
                      "provenance": "independent Sol, 2026-11-02: goal and conjunctive adoption criterion; no schedule, deadline or instruction to skip work"},
            "files": files, "jobs": jobs, "unneeded_jobs": unneeded,
            "unneeded_how": "GPU training of a recipe whose real prepared data fails the token-retention criterion; its CPU preparation is still required",
            "preparations": preps,
            "preparations_how": "All seven are necessary CPU observations, including recipes rejected without a GPU launch. The custom scripted reference executes them and reads actual shard headers.",
            "environment": H4.environment(recs, jobs),
            "answer": {"raw_tokens": RAW_TOKENS, "minimum_retention": MIN_RETENTION, "recipes": answer,
                       "supported_answer_needs": accepted_yield, "winner": winner,
                       "loss_provenance": "synthesised assignment of distinct recorded contra-soft-muon curves to recipe outputs, as H4"},
            "reference_plan": {"instances": [{"job_id": j["job_id"]} for j in jobs],
                               "policy": "filter_yield", "how": "custom duration-blind reference: real CPU preparations, observed token headers, free-GPU queue for passing recipes; no source outcomes read at run time"},
            "scripted_reference": {"path": "reference/filter_yield_reference.py", "from": REFERENCE_REL,
                                   "how": "scripted runs stage this file; model-agent runs do not receive it"},
            "measuring_run": {"harness": "scripted", "policy": "filter_yield", "speed": 1.0,
                              "how": "custom controller supplied with the builder; native packed policy alone omits rejected recipes' CPU observations"},
            "sources": base["sources"],
            "notes": ["Training curves assigned to filtered datasets are synthesised, as H4; loss comparisons are not recorded filter-ablation results.",
                      "All 28 potential GPU jobs have distinct source.path values; each is at least 20 paced minutes. Sixteen are needed after yield observations, twelve are not.",
                      "Preparation times transfer from H4's actual c3-highmem-44 box measurements; relocation itself has not yet been timed.",
                      "A reference that simply skips the three rejected CPU preparations is clairvoyant and invalid. Use the supplied controller and report observed CPU completion.",
                      "This candidate misses the 40-GPU-job target. It is a new CPU decision mechanism on the released simulator; larger sources are being developed separately."]}


if __name__ == "__main__":
    derive(sys.argv[1])
