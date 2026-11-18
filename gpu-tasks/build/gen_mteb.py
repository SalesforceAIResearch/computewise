"""mteb-embed-select (twin a) and mteb-embed-select-b (twin b): choose the embedding model for a retrieval service from a
matrix of recorded `mteb run` evaluations (gpu-tasks-mteb, 2026-10-31; package search angle I, survey entry S7; design and
reasons in notes/2026-10-31-mteb-package.md).

Two entry points.
  python3 -I build/gen_mteb.py <results checkout> <char-counts.json> <hub-trees dir> <registry>
      writes one record per (model, task) of both twins under <registry>/records/, the recorded result files the records
      serve under <registry>/artifacts/mteb/results/, and the box's Hub list and pins under <registry>/artifacts/mteb/env/.
      <results checkout>: https://github.com/embeddings-benchmark/results at RESULTS_COMMIT (a sparse checkout of the model
      folders below is enough). <char-counts.json>: count_chars output (the characters each subset hands to the evaluator,
      counted from the pinned datasets). <hub-trees dir>: the Hub API trees (`/api/{models,datasets}/<repo>/revision/<rev>
      ?blobs=true`) of every repo in HUB, one <org>_<name>.json each. Downloads are data: nothing from them is executed.
  build(recs, disclosure_of, reg, twin="a")
      the task, from those records, for gen_tasks.main / finish_one_task.

Every record is one result JSON of the results repository: `evaluation_time` (the wall time of task.evaluate(), mteb
1.18.0 MTEB.py:300-308; the dataset and model loads lie outside it) is the paced duration, and the per-subset score dict is
the result, served verbatim. Hardware is not in the source (reference.gpu is inferred and says so). The records are not
replay-ready: gpu-mock has no serving mode for `mteb run` yet, the entry hash is the box image's own console-script shim,
and that image is not built. Each is a named blocker (materialisation.blockers), tagged generic or MTEB-specific.
"""
import hashlib
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

RESULTS_REPO = "https://github.com/embeddings-benchmark/results"
RESULTS_COMMIT = "964b762b7226bd14c2a9f46e9327d2e028d7d7f8"     # 2026-10-30 21:44:50 +0300, read 2026-10-31
MTEB_VERSION = "1.18.0"                                          # one submission batch: one evaluator, one dataset revision per task
MTEB_WHEEL_SHA256 = "64ba23e422bea411860ea70010fdb082a57af32dc06412849ca1b69937cb40d1"   # PyPI mteb-1.18.0-py3-none-any.whl
READ_ON = "2026-10-31"
CHUNK = 50000                                                    # RetrievalEvaluator corpus_chunk_size, mteb 1.18.0 line 65
SEAM = "mteb.abstasks.AbsTaskRetrieval.AbsTaskRetrieval._evaluate_subset"

# (results-folder name, Hub id, revision): the revision is the folder the 1.18.0 results sit in
MODELS = {
    "nvidia__NV-Embed-v1": ("nvidia/NV-Embed-v1", "570834afd5fef5bf3a3c2311a2b6e0a66f6f4f2c"),
    "nvidia__NV-Embed-v2": ("nvidia/NV-Embed-v2", "7604d305b621f14095a1aa23d351674c2859553a"),
    "bigscience__sgpt-bloom-7b1-msmarco": ("bigscience/sgpt-bloom-7b1-msmarco", "dc579f3d2d5a0795eba2049e16c3e36c74007ad3"),
    "manu__sentence_croissant_alpha_v0.2": ("manu/sentence_croissant_alpha_v0.2", "4610b8cea65d7dd59e0b04af50753933fe5b29b2"),
    "manu__sentence_croissant_alpha_v0.3": ("manu/sentence_croissant_alpha_v0.3", "4ac16754f3d81aba76cc32955dc9ee4122df96eb"),
    "manu__sentence_croissant_alpha_v0.4": ("manu/sentence_croissant_alpha_v0.4", "0ce6372e6a3c21134dcf26dcde13cca869c767fc"),
    "Jaume__gemma-2b-embeddings": ("Jaume/gemma-2b-embeddings", "86431f65d7c3f66b2af096c61e614a2958f191f1"),
    "Linq-AI-Research__Linq-Embed-Mistral": ("Linq-AI-Research/Linq-Embed-Mistral", "0c1a0b0589177079acc552433cad51d7c9132379"),
    "NovaSearch__stella_en_1.5B_v5": ("NovaSearch/stella_en_1.5B_v5", "d03be74b361d4eb24f42a2fe5bd2e29917df4604"),
}
# task -> the dataset repos its 1.18.0 loader reads, at the revisions its TaskMetadata pins
TASK_DATA = {
    "CQADupstackUnixRetrieval": [("mteb/cqadupstack-unix", "6c6430d3a6d36f8d2a829195bc5dc94d7e063e53")],
    "HotpotQAHardNegatives": [("mteb/HotpotQA_test_top_250_only_w_correct-v2", "617612fa63afcb60e3b134bed8b7216a99707c37")],
    "CovidRetrieval": [("C-MTEB/CovidRetrieval", "1271c7809071a13532e05f25fb53511ffce77117"),
                       ("C-MTEB/CovidRetrieval-qrels", "a9f41b7cdf24785531d12417ce0d1157ed4b39ca")],
    "CmedqaRetrieval": [("C-MTEB/CmedqaRetrieval", "cd540c506dae1cf9e9a59c3e06f42030d54e7301"),
                        ("C-MTEB/CmedqaRetrieval-qrels", "279d737f36c731c8ff6e2b055f31fe02216fa23d")],
    "FEVERHardNegatives": [("mteb/FEVER_test_top_250_only_w_correct-v2", "080c9ed6267b65029207906e815d44a9240bafca")],
    "CodeFeedbackMT": [("CoIR-Retrieval/codefeedback-mt", "b0f12fa0c0dd67f59c95a5c33d02aeeb4c398c5f")],
    "CodeFeedbackST": [("CoIR-Retrieval/codefeedback-st", "d213819e87aab9010628da8b73ab4eb337c89340")],
    "Touche2020Retrieval.v3": [("mteb/webis-touche2020-v3", "431886eaecc48f067a3975b70d0949ea2862463c")],
    "WikipediaRetrievalMultilingual": [
        ("ellamind/wikipedia-2023-11-retrieval-multilingual-queries", "3b6ea595c94bac3448a2ad167ca2e06abd340d6e"),
        ("ellamind/wikipedia-2023-11-retrieval-multilingual-corpus", "f20ac0c449c85358d3d5c72a95f92f1eddc98aa5"),
        ("ellamind/wikipedia-2023-11-retrieval-multilingual-qrels", "ec88a7bb2da034d538e98e3122d2c98530ca1c8d")],
    "BelebeleRetrieval": [("facebook/belebele", "75b399394a9803252cfec289d103de462763db7c")],
    "CodeSearchNetCCRetrieval": [("CoIR-Retrieval/CodeSearchNet-ccr", "6e1effa2c03723c5fde48ee912b5ee08d4f211e8")],
}
# The twins: disjoint model sets from the same submission batch (the selection guard: a fresh twin from different records).
# Every cell is a recorded 1.18.0 evaluation of 20 minutes or more that the suspect rule keeps (notes, section 4).
TWINS = {
    "a": {"task_id": "mteb-embed-select",
          "models": ["nvidia__NV-Embed-v1", "nvidia__NV-Embed-v2", "bigscience__sgpt-bloom-7b1-msmarco",
                     "manu__sentence_croissant_alpha_v0.3", "manu__sentence_croissant_alpha_v0.4"],
          "tasks": ["CQADupstackUnixRetrieval", "HotpotQAHardNegatives", "CovidRetrieval", "FEVERHardNegatives", "CodeFeedbackMT",
                    "WikipediaRetrievalMultilingual", "CodeFeedbackST", "BelebeleRetrieval", "Touche2020Retrieval.v3"]},
    "b": {"task_id": "mteb-embed-select-b",
          "models": ["Jaume__gemma-2b-embeddings", "Linq-AI-Research__Linq-Embed-Mistral", "NovaSearch__stella_en_1.5B_v5",
                     "manu__sentence_croissant_alpha_v0.2"],
          "tasks": ["CmedqaRetrieval", "HotpotQAHardNegatives", "CovidRetrieval", "FEVERHardNegatives", "CodeFeedbackMT",
                    "WikipediaRetrievalMultilingual", "CodeFeedbackST", "BelebeleRetrieval", "Touche2020Retrieval.v3",
                    "CodeSearchNetCCRetrieval"]},
}
SUSPECT_FACTOR = 3.0      # a run whose kg CO2 per hour is outside 1/3-3x its model's 1.18.0 median is flagged suspect
MIN_JOB_S = 1200          # p006-jobs-long-enough: every job at least 20 minutes at speed 1

# The hardware statement every record carries: fair to the evidence, not a finding.
GPU_NAME = "NVIDIA H100 80GB HBM3"
GPU_HOW = ("inferred from the batch runner's own statement, not recorded per file. These nine models' mteb 1.18.0 results were "
           "added by one submission, embeddings-benchmark/results PR #83 (Muennighoff, merged 2025-01-03), which addresses "
           "mteb issue #1571; in that issue the same person wrote on 2025-01-08 that some of the batch's task-model "
           "combinations 'took >2 days on H100s' (github.com/embeddings-benchmark/mteb/issues/1571#issuecomment-2578766321), "
           "and the jobs ran on slurm. No result file or model_meta.json names a GPU (searched for H100, A100, V100, L40S, A6000, "
           "RTX, T4, H200 and GPU: none); the only GPU named in these folders, a GPUSpeedTask.json (mteb 1.12.49: Quadro RTX "
           "8000), is another batch's. The batch is one device type (its kg CO2 per hour is steady, each model's median 321-351 "
           "g/h, notes section 4), consistent with the statement. So recorded seconds are paced 1:1 as H100 seconds. Read "
           "2026-10-31 by gpu-tasks; survey/mteb-inputs-2026-10-31/README.md has the sources")

BLOCKERS = [
    # (tag, text): generic = the "pace for T, print the recorded result" core any recorded-total job needs;
    # mteb = specific to `mteb run`
    ("generic", "gpu-mock serving: a job paced for its recorded total with no per-step series, ending in the recorded "
                "result written by the job's own code; not built (gpu-mock next/NOTES.md, \"Serving `mteb run`\", item 5)"),
    ("mteb", "gpu-mock serving, mteb-specific: the library seam " + SEAM + " (pace one unit for the queries and one per "
             "50,000-document chunk, each by its character share; return this record's per-subset score dict); not built "
             "(gpu-mock next/NOTES.md, \"Serving `mteb run`\", items 1, 3, 5)"),
    ("generic", "entry shim: identity.entry_hash is the box image's own `mteb` console-script shim, whose bytes include its "
                "interpreter path; not computable until the image exists (gpu-mock next/NOTES.md, \"Serving `mteb run`\", "
                "item 2). A recorded-total job launched through any installed console script has the same blocker"),
    ("mteb", "box image: mteb 1.18.0, sentence-transformers and pytrec-eval, with this record's model weights and its task's "
             "datasets staged offline at the pinned revisions; not built (isolation; gpu-mock next/NOTES.md, item 8)"),
]


def read_json(p):
    with open(p) as f:
        return json.load(f)


def result_path(root, folder, task):
    return os.path.join(root, "results", folder, MODELS[folder][1], task + ".json")


def record_id(folder, task):
    return "mteb-" + C.slug(folder.replace("__", "-")) + "--" + C.slug(task)


def co2_rates(root, folders):
    """Each model's kg CO2 per hour over its mteb 1.18.0 results of 10 minutes or more, every task in its folder."""
    out = {}
    for f in folders:
        d = os.path.join(root, "results", f, MODELS[f][1])
        rs = []
        for n in sorted(os.listdir(d)):
            if not n.endswith(".json") or n == "model_meta.json":
                continue
            try:
                j = read_json(os.path.join(d, n))
            except (ValueError, OSError):
                continue
            t, co2 = j.get("evaluation_time"), j.get("kg_co2_emissions")
            if isinstance(j, dict) and j.get("mteb_version") == MTEB_VERSION and t and co2 and t >= 600:
                rs.append(co2 / t * 3600.0)
        out[f] = (statistics.median(rs), len(rs)) if rs else (None, 0)
    return out


def hub_entry(trees, kind, repo, rev, keep=None):
    j = read_json(os.path.join(trees, repo.replace("/", "_") + ".json"))
    if j.get("sha") != rev:
        raise SystemExit(f"{repo}: tree is at {j.get('sha')}, not the pinned {rev}")
    files = {}
    for x in j["siblings"]:
        n = x["rfilename"]
        if n.startswith(".") or n.startswith("unused/") or not x.get("size"):
            continue
        if keep and not keep(n):
            continue
        files[n] = int(x["size"])
    return {"kind": kind, "repo": repo, "revision": rev, "files": files, "bytes": sum(files.values()),
            "revision_how": ("the revision mteb 1.18.0's TaskMetadata pins for this task's data" if kind == "dataset" else
                             "the model revision the 1.18.0 results were written under (the results folder's name)")}


def model_files(names):
    """What SentenceTransformer(..., trust_remote_code=True) reads: config, tokenizer, modules and the custom code, and the
    safetensors weights where the repo has them (its .bin copy is then left out)."""
    has_st = any(n.endswith(".safetensors") for n in names)
    return lambda n: not ((has_st and n.endswith(".bin")) or n.endswith((".onnx", ".msgpack", ".h5", ".ot")) or n.startswith("onnx/"))


def make_record(root, counts, trees, folder, task, rate, reg):
    hub_id, rev = MODELS[folder]
    src_path = result_path(root, folder, task)
    raw = open(src_path, "rb").read()
    j = json.loads(raw)
    meta = read_json(os.path.join(root, "results", folder, rev, "model_meta.json"))
    if j.get("mteb_version") != MTEB_VERSION:
        raise SystemExit(f"{folder} {task}: mteb {j.get('mteb_version')}, not {MTEB_VERSION}")
    want_ds = TASK_DATA[task][0][1]
    if j.get("dataset_revision") != want_ds:
        raise SystemExit(f"{folder} {task}: dataset_revision {j.get('dataset_revision')} is not the pinned {want_ds}")
    t_eval = float(j["evaluation_time"])
    co2 = j.get("kg_co2_emissions")
    r_med, r_n = rate
    r_this = co2 / t_eval * 3600.0 if co2 else None
    suspect = bool(r_this and r_med and not (r_med / SUSPECT_FACTOR <= r_this <= r_med * SUSPECT_FACTOR))
    rid = record_id(folder, task)
    art = f"artifacts/mteb/results/{folder}/{rev}/{task}.json"
    C.write_text(os.path.join(reg, art), raw.decode("utf-8"))
    splits = list(j["scores"])
    if len(splits) != 1:
        raise SystemExit(f"{rid}: {len(splits)} splits; the chunk pacing below assumes one")
    split = splits[0]
    subsets = {s["hf_subset"]: s for s in j["scores"][split]}
    cc = counts["tasks"][task]
    if set(cc) != set(subsets):
        raise SystemExit(f"{rid}: character counts cover {len(cc)} subsets, the result {len(subsets)}")
    total = sum(v["query_chars"] + v["doc_chars"] for v in cc.values())
    sub = {}
    for name in sorted(subsets):
        v = cc[name]
        share = (v["query_chars"] + v["doc_chars"]) / total
        sub[name] = {"n_queries": v["n_queries"], "n_docs": v["n_docs"], "query_chars": v["query_chars"],
                     "chunk_chars": v["chunk_chars"], "main_score": subsets[name]["main_score"],
                     "paced_s": C.tv(round(t_eval * share, 3), "modelled",
                                     "evaluation_time x this subset's characters (queries and corpus) over the task's")}
    mains = [s["main_score"] for s in subsets.values()]
    n_par, mem_mb = meta.get("n_parameters"), meta.get("memory_usage_mb")
    weights = hub_entry(trees, "model", hub_id, rev)
    st_bytes = sum(b for n, b in weights["files"].items() if n.endswith(".safetensors"))
    if mem_mb:
        mem, mem_prov = int(mem_mb * 2**20), "recorded"
        mem_how = ("memory_usage_mb of the results repository's model_meta.json for this revision, the weights' memory as "
                   "mteb's model metadata states it, held on the card from load as G1's records do (gpu-mock next/NOTES.md, "
                   "item 4). Activations and the similarity matrix are not counted; a 1.18.0 SentenceTransformer load without "
                   "torch_dtype holds fp32 weights, which can be up to twice this")
    else:
        mem, mem_prov = int(n_par * 4) if n_par else st_bytes, "inferred"
        mem_how = (f"model_meta.json has no memory_usage_mb: n_parameters {n_par:,} (recorded there) x 4 bytes, the fp32 weights "
                   "a SentenceTransformer load without torch_dtype holds; activations and the similarity matrix are not counted"
                   if n_par else "the safetensors weights' size; model_meta.json records neither memory nor parameters")
    cfg = json.dumps([hub_id, rev, task, j["dataset_revision"], MTEB_VERSION], separators=(",", ":"))
    chash = hashlib.sha256(cfg.encode()).hexdigest()
    argv = ["mteb", "run", "-m", hub_id, "--model_revision", rev, "-t", task]
    rec = {
        "schema": C.SCHEMA_RECORD,
        "identity": {
            "record_id": rid, "kind": "eval",
            "entry_hash": None,
            "entry_hash_how": ("the box image's own `mteb` console-script shim (entry point mteb.cli:main of the mteb "
                               f"{MTEB_VERSION} wheel, sha256 {MTEB_WHEEL_SHA256[:12]}...), whose bytes include the image's "
                               "interpreter path; computed once that image exists. Every record of this source shares it, "
                               "and the argv tells them apart"),
            "argv": argv, "argv_provenance": "derived",
            "argv_how": ("mteb 1.18.0's CLI (mteb/cli.py: run -m, --model_revision, -t) for the model, revision and task of the "
                         "result file; one task per launch, as the results repository holds one file per task"),
            "argv_exempt": ["--output_folder", "--batch_size", "-v", "--verbosity", "--co2_tracker"],
            "argv_exempt_how": ("where results are written, the encode batch size, log verbosity and the CO2 tracker change no "
                                "score; the pacing ignores --batch_size, which is disclosed (gpu-mock next/NOTES.md, item 2)"),
            "env": {}, "env_how": "no environment variable changes an evaluation",
            "seed": C.absent("mteb's retrieval evaluation is deterministic encoding and exact search; no seed is taken"),
            "source_key": {"model": hub_id, "model_revision": rev, "task": task, "mteb_version": MTEB_VERSION},
            "replicate_group": "mteb-cfg-" + chash[:12], "match_policy": "exact_only",
            "match_policy_how": "no learning-rate trace; one record per (model, revision, task), matched by argv",
            "trajectory": {
                "config_hash": chash, "config_hash_how": "sha256 of [model, model revision, task, dataset revision, mteb version]",
                "environment": {"mteb_version": MTEB_VERSION, "model": hub_id, "model_revision": rev},
                "lr_schedule": {"identity": None, "source": None, "trace": None},
                "data": {"dataset": task, "revisions": [list(x) for x in TASK_DATA[task]],
                         "how": "the repos and revisions the task's 1.18.0 loader reads"},
                "precision": C.absent("not recorded; mteb 1.18.0 loads a model outside its registry with SentenceTransformer "
                                      "defaults, which do not set torch_dtype"),
                "global_batch_tokens": None},
            "fingerprint": None, "fingerprint_inputs": None,
            "fingerprint_how": "an evaluation through a library entry point; no optimizer, no structural hash",
            "consumes": [],
        },
        "mode": "replay",
        "source": {"name": "MTEB results repository", "repo": RESULTS_REPO, "commit": RESULTS_COMMIT,
                   "path": f"results/{folder}/{rev}/{task}.json", "licence": "CC0-1.0", "read_on": READ_ON,
                   "mteb_version": MTEB_VERSION, "sha256": hashlib.sha256(raw).hexdigest()},
        "reference": {"gpu": GPU_NAME, "gpu_provenance": "inferred", "gpu_how": GPU_HOW,
                      "gpu_count": 1, "gpu_count_provenance": "inferred",
                      "gpu_count_how": ("mteb run puts the model on one device (cli.py: device = 'cuda'), and sentence-"
                                        "transformers encodes on that one device unless encode_multi_process is called, which "
                                        "mteb 1.18.0 does not"),
                      "interconnect": None, "precision": None},
        "launch": {"launcher": "mteb", "world_size": C.tv(1, "inferred", "one device, as reference.gpu_count"),
                   "per_device_microbatch": C.absent("the run's --batch_size was not recorded"),
                   "accumulation_steps": C.absent(), "global_batch_tokens": C.absent(),
                   "batch_policy": C.absent("single-GPU evaluation"), "alternatives": {}},
        "steps": None, "planned_steps": None, "per_step_gpu_s": None,
        "timing": {"pacing": "total",
                   "wall_total_s": C.tv(round(t_eval, 3), "recorded",
                                        "evaluation_time: the wall time of task.evaluate() over the task's splits (mteb 1.18.0 "
                                        "MTEB.py:300-308, 438): encoding, search and scoring. The dataset load and the model "
                                        "load lie outside it; on the box they are real host time"),
                   "setup_s": C.absent("dataset and model loads were not timed by the source"),
                   "eval_s": C.absent(), "checkpoint_s": C.absent()},
        "memory": {"peak_allocated_bytes": mem,
                   "peak": C.tv(mem, mem_prov, mem_how)},
        "series": {},
        "observed": {"kg_co2_emissions": co2, "co2_g_per_h": round(r_this, 1) if r_this else None,
                     "model_co2_g_per_h_median": round(r_med, 1) if r_med else None, "model_co2_runs": r_n,
                     "how": "kept for calibration only, never served: codecarbon's estimate, the device's power proxy"},
        "materialisation": {
            "reads": ["per-subset score dict"], "needs_series": [],
            "blockers": [b for _, b in BLOCKERS],
            "blockers_tagged": [{"tag": t, "blocker": b} for t, b in BLOCKERS],
            "serving": {
                "mode": "per_subset_scores", "seam": SEAM,
                "how": ("gpu-mock next/NOTES.md, \"Serving `mteb run`\" (2026-10-31 15:40 MDT): the seam receives one subset's "
                        "corpus and queries, paces one unit for the queries and one per 50,000-document chunk, each by its share "
                        "of the subset's characters, and returns the subset's recorded score dict, so mteb's own TaskResult "
                        "writes the result file. A full launch paces evaluation_time; a `-l` launch paces its subsets' shares. "
                        "A subset the record does not hold goes to the modelled path. The character share is modelled"),
                "results_file": {"path": f"results/{hub_id.replace('/', '__')}/{rev}/{task}.json", "format": "json",
                                 "recorded": art, "recorded_sha256": hashlib.sha256(raw).hexdigest(),
                                 "how": "the source file, byte for byte: what a full launch must write"},
                "split": split, "chunk_docs": CHUNK,
                "chunk_order": "sorted(corpus ids, reverse=True), RetrievalEvaluator.search line 146",
                "chars_how": counts["how"],
                "subsets": sub,
            }},
        "result": {"main_score_mean": C.tv(round(statistics.mean(mains), 6), "derived",
                                           f"mean of the {len(mains)} subsets' recorded main_score, as the MTEB leaderboard "
                                           "averages a task")},
        "outcome": {"status": "completed", "provenance": "derived", "how": "the result file holds scores for every subset",
                    **({"suspect": {"value": True, "how": f"kg CO2 per hour {r_this:.0f} g/h is outside "
                                                          f"{1 / SUSPECT_FACTOR:.2f}-{SUSPECT_FACTOR:.0f}x the model's 1.18.0 "
                                                          f"median {r_med:.0f} g/h"}} if suspect else {})},
        "failures": [],
        "artefacts": {"produces": [f"results/{hub_id.replace('/', '__')}/{rev}/{task}.json",
                                   f"results/{hub_id.replace('/', '__')}/{rev}/model_meta.json"]},
        "host": {"weights_bytes": C.tv(st_bytes, "recorded", "the Hub's safetensors sizes at the revision"),
                 "data_bytes": C.tv(sum(hub_entry(trees, "dataset", r, v)["bytes"] for r, v in TASK_DATA[task]), "recorded",
                                    "the Hub's file sizes of the task's dataset repos at the pinned revisions")},
        "agent_view": {"script": None, "launch_command": " ".join(argv) + " --output_folder results",
                       "inputs": [hub_id + "@" + rev] + [r + "@" + v for r, v in TASK_DATA[task]],
                       "recognisability": "high: mteb is a public benchmark harness, and the task names are its"},
    }
    return rec, suspect


def main():
    root, counts_p, trees, reg = sys.argv[1:5]
    counts = read_json(counts_p)
    folders = sorted({f for tw in TWINS.values() for f in tw["models"]})
    rates = co2_rates(root, folders)
    recs = []
    for key, tw in TWINS.items():
        for f in tw["models"]:
            for t in tw["tasks"]:
                rec, suspect = make_record(root, counts, trees, f, t, rates[f], reg)
                recs.append(rec)
                if suspect:
                    print(f"suspect: {rec['identity']['record_id']}")
                elif rec["timing"]["wall_total_s"]["value"] < MIN_JOB_S:
                    print(f"under 20 min: {rec['identity']['record_id']}")
    seen = {}
    for r in recs:
        seen[r["identity"]["record_id"]] = r
    recs = list(seen.values())
    C.add_disclosure(recs)
    for r in recs:
        r["readiness"] = C.readiness(r)
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
    hub = []
    for f in folders:
        hid, rev = MODELS[f]
        names = [x["rfilename"] for x in read_json(os.path.join(trees, hid.replace("/", "_") + ".json"))["siblings"]]
        hub.append(hub_entry(trees, "model", hid, rev, model_files(names)))
    for t in sorted({t for tw in TWINS.values() for t in tw["tasks"]}):
        for repo, rev in TASK_DATA[t]:
            if not any(h["repo"] == repo for h in hub):
                hub.append(hub_entry(trees, "dataset", repo, rev,
                                     lambda n: n.endswith((".parquet", ".jsonl", ".json", ".tsv", ".csv")) and n != "data.zip"))
    C.write_json(os.path.join(reg, "artifacts", "mteb", "env", "hub.json"), {"repos": hub, "how": "Hub API trees, read " + READ_ON})
    C.write_text(os.path.join(reg, CONSTRAINTS_REL), "\n".join(PACKAGES) + "\n")
    print(f"mteb: {len(recs)} records ({sum(1 for r in recs if r['readiness']['replay'])} replay-ready), "
          f"{len(hub)} Hub repos, {sum(h['bytes'] for h in hub) / 1e9:.1f} GB")


# The box's packages. Top-level pins only, chosen to resolve with mteb 1.18.0 (2024-10-28) and the NV-Embed custom code, and
# NOT yet resolved or built: isolation resolves the full set when it builds the image (blocker 4). torch is the CPU wheel of
# the version gpu-mock's patch is proven on.
PACKAGES = ("torch==2.11.0", "mteb==1.18.0", "sentence-transformers==3.2.1", "transformers==4.44.2", "datasets==3.0.2",
            "pytrec-eval-terrier==0.5.6", "einops==0.8.0", "huggingface-hub==0.26.2", "safetensors==0.4.5", "tokenizers==0.19.1")
CONSTRAINTS_REL = "artifacts/mteb/env/constraints.txt"

TASK_NOTES = """# Evaluating a candidate on a set

One evaluation is one command, run from the task root:

    mteb run -m <model> --model_revision <revision> -t <set> --output_folder results

It loads the model on one GPU, evaluates it on the set and writes results/<org>__<name>/<revision>/<set>.json. That file's
`scores` holds a `main_score` for each of the set's subsets (one subset for most sets, one per language or language pair for
the multilingual ones); a set's score is the mean of its subsets' main scores. The models and the sets' data are in the local
Hugging Face cache, at the revisions mteb pins; nothing is downloaded.
"""


def build(recs, disclosure_of, reg=None, twin="a"):
    tw = TWINS[twin]
    if reg:
        C.write_text(os.path.join(reg, CONSTRAINTS_REL), "\n".join(PACKAGES) + "\n")
    jobs, stats = [], {}
    for t in tw["tasks"]:
        for f in tw["models"]:
            rid = record_id(f, t)
            r = recs[rid]
            hid, rev = MODELS[f]
            jid = C.slug(f.split("__")[1]) + "--" + C.slug(t)
            jobs.append({"job_id": jid, "label": f"{hid} on {t}", "replicate_group": r["identity"]["replicate_group"],
                         "records": [rid], "draw": "without_replacement", "seed": None,
                         "gpu_options": [1], "recorded_gpus": 1, "recorded_gpu": r["reference"]["gpu"], "batch_policy": None,
                         "after": [], "consumes": [], "produces": r["artefacts"]["produces"][:1],
                         "launch": " ".join(r["identity"]["argv"]) + " --output_folder results"})
            stats[jid] = {"main_score_mean": r["result"]["main_score_mean"]["value"], "wall_s": C.job_duration(r),
                          "paced_s": C.paced_duration(r, 1)}
    means = {}
    for f in tw["models"]:
        hid = MODELS[f][0]
        v = [recs[record_id(f, t)]["result"]["main_score_mean"]["value"] for t in tw["tasks"]]
        means[hid] = round(statistics.mean(v), 6)
    winner = max(means, key=means.get)
    hub_all = read_json(os.path.join(reg, "artifacts", "mteb", "env", "hub.json"))["repos"] if reg else []
    need = {MODELS[f][0] for f in tw["models"]} | {x for t in tw["tasks"] for x, _ in TASK_DATA[t]}
    hub = [h for h in hub_all if h["repo"] in need]
    n_sets, n_cands = len(tw["tasks"]), len(tw["models"])
    words = {4: "four", 5: "five", 9: "nine", 10: "ten"}
    cand = "".join(f"{MODELS[f][0]} {MODELS[f][1]}\n" for f in tw["models"])
    suite = "".join(t + "\n" for t in tw["tasks"])
    brief = (f"We're choosing the embedding model for our document-retrieval service. The {words[n_cands]} candidates, each "
             f"pinned to a revision, are listed in candidates.txt, and our evaluation suite of {words[n_sets]} retrieval sets is "
             "in suite.txt; EVALUATION.md says how one evaluation runs. Our rule: we adopt the candidate with the highest mean "
             f"main score over the {words[n_sets]} sets, each set evaluated in full. Produce the results table, every candidate "
             "on every set with its score, and name the candidate we adopt. The machine has four H100s.")
    return {
        "task_id": tw["task_id"], "rung": 1, "env_kind": "mteb", "goal_level": False, "set": "package-search",
        "set_how": ("package search angle I (the analysis, 2026-10-31 14:46 MDT; survey S7), built by gpu-tasks-mteb; "
                    + ("twin a, the package" if twin == "a" else "twin b, the selection guard's fresh twin from a disjoint "
                       "model set of the same submission batch")),
        "title": (f"Choose an embedding model: {n_cands} candidates x {n_sets} retrieval sets, {len(jobs)} recorded one-GPU "
                  "evaluations whose lengths the set names hide"),
        "brief": {"text": brief, "provenance": (
            "written by gpu-tasks-mteb from the MTEB results repository (CC0) for the 2026-10-20 brief rules: the question and "
            "the rule, not the runs; no procedure, no time words, no deadline; the GPU count stated. It names no benchmark; "
            "EVALUATION.md and the set names do, since the tool is mteb (disclosed in notes)")},
        "files": [{"path": "EVALUATION.md", "content": TASK_NOTES}, {"path": "candidates.txt", "content": cand},
                  {"path": "suite.txt", "content": suite}],
        "jobs": jobs,
        "environment": {
            "python": C.tv("3.11.9", "inferred", "isolation's build.sh pin; mteb 1.18.0 supports 3.9-3.12"),
            "install": {"packages": list(PACKAGES), "constraints": CONSTRAINTS_REL,
                        "line": "pip install -c constraints.txt " + " ".join(PACKAGES),
                        "cpu_lines": ["pip install --index-url https://download.pytorch.org/whl/cpu --no-deps torch==2.11.0",
                                      "pip install " + " ".join(p for p in PACKAGES if not p.startswith("torch=="))],
                        "how": ("top-level pins chosen to resolve with mteb 1.18.0 and the NV-Embed custom code (transformers "
                                "4.44); NOT resolved or installed yet. isolation resolves and builds them (blocker 4)")},
            "hub": hub, "hub_bytes": sum(h["bytes"] for h in hub),
            "hub_how": ("the candidates' weights at their result revisions (safetensors only where a repo has both) and every "
                        "dataset repo the sets' 1.18.0 loaders read, at the pinned revisions; Hub API trees read " + READ_ON),
            "env": {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
            "env_how": "offline, so a missing file fails at once",
            "verified": "nothing: the image is not built (blocker 4)",
            "image_label": {"value": "mteb-1.18", "how": "a label of its own; no image carries it yet"}},
        "candidates_stats": stats,
        "candidates_stats_how": "per job, from its record: the recorded task mean of main_score, the recorded and the paced seconds",
        "recorded_outcome": {"means": means, "winner": winner,
                             "how": "the brief's rule over the recorded scores: mean over the sets of each set's subset-mean "
                                    "main_score; highest wins. For the scorer; the agent sees none"},
        "sources": [f"{RESULTS_REPO} at {RESULTS_COMMIT[:12]}, mteb {MTEB_VERSION} result files (CC0)"],
        "notes": [
            "Every job is one recorded `mteb run` evaluation; its paced length is the result file's evaluation_time, which "
            "excludes the dataset and model loads (the box pays those for real). Lengths are uneven and a set's name hides them: "
            "the corpus size, which sets them, is not in the name; the staged data's byte size and the model's size are readable "
            "from the Hugging Face cache, and ordering by them comes within a few percent of ordering by recorded duration "
            "(notes/2026-10-31-mteb-package.md, section 6).",
            "Hardware is assumed, not recorded (reference.gpu_how): it scales absolute hours, never a schedule ratio.",
            "Not replay-ready by design until gpu-mock's per_subset_scores mode, the box's mteb shim and the image exist; each "
            "record names those blockers, tagged generic or MTEB-specific (materialisation.blockers_tagged).",
            "Width 1 is the code's: mteb run evaluates on one device."],
    }


if __name__ == "__main__":
    main()
