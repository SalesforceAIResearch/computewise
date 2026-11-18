"""gpu-mock's per_example serving block on the 40 MLS-Bench KV records (gpu-tasks, 2026-10-22; contract:
experiments/gpu-mock/docs/MLS-KV-SERVING.md, confirmed by gpu-mock at 16:33Z).

usage: python3 -I build/gen_mlskv_serving.py <registry>

The records come from MLS-Bench's leaderboards (build/gen_mlsbench.py), which hold per-workload aggregates only. gpu-mock
serves them by hooking two of the evaluator's functions by name: the generate seam returns without decoding, and the score
seam returns per-example scores whose mean is the recorded final_score. This adds what the daemon needs to do that to each
record's materialisation.serving, keeping every field gen_mlsbench wrote: the seam names, the workload's full example count,
the score's kind and scale, where a subset limit comes from, and the model's load time (pending a measurement). It runs after
gen_mlsbench.py and is idempotent.

n_examples_full, per workload:
- gsm8k 1319, longbench_v2 503: pinned by the recorded scores. Each is binary (0 or 100 per example), so final_score x n / 100
  is a whole number of hits for every row, and these are the only n in their ranges where it is.
- needlebench_niah 441: the loader's own count, 147 essays x NEEDLE_DEPTHS (0.10, 0.50, 0.90; custom_quant_eval.py
  build_needlebench_examples). The scores fit 147 too (any multiple of 147 does), which is how 147 got here first; W0-mlskv-check10b
  showed the box pacing 441 examples at wall / 147, three times the recorded wall (2026-10-22).
- longbench_hotpotqa, longbench_passage_retrieval, longbench_repobench: 300 each, counted on the staged data.zip as the
  evaluator reads it; the LongBench card's LongBench-E table gives the same, 100 per length band.
  The scores are continuous, so they do not pin n.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

KV_TASKS = {"llm-kv-selection-budgeting": "selection", "llm-kv-adaptive-quantization": "quant"}
GENERATE_SEAM = {"selection": "generate_with_selection", "quant": "generate_with_quantized_cache"}
SUBSET_ENV = {"selection": "SELECTION_KV_MAX_EXAMPLES", "quant": "ADAPTIVE_KV_MAX_EXAMPLES"}
SCORE_KIND = {"gsm8k": "binary", "longbench_v2": "binary", "needlebench_niah": "binary",
              "longbench_passage_retrieval": "retrieval", "longbench_hotpotqa": "continuous", "longbench_repobench": "continuous"}
PINNED_N = {"gsm8k": 1319, "longbench_v2": 503}
NEEDLE_N = 441                    # 147 essays x 3 needle depths (custom_quant_eval.py NEEDLE_DEPTHS)
LONGBENCH_E_N = 300
LOAD_S = 6.0                      # Qwen2.5-3B-Instruct, bf16, through the mock on the E2
SERVING_BLOCKER = "gpu-mock per_example serving, in progress 2026-10-22"
# Peak device memory (gpu-mock and the analysis, 2026-10-23 17:25Z: with no memory figure a KV job held 0 bytes,
# so nvidia-smi showed 0 MiB on a GPU busy for the whole job, and one agent killed four full GSM8K runs as "stalled"). Derived:
# the model's weights, the KV cache at the workload's longest prompt, and one layer's transient prefill working set. Prompt
# lengths are measured (survey/mlskv-prompt-lengths-2026-10-23.json, build/mlskv_prompt_lengths.py).
PROMPT_LENGTHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "survey", "mlskv-prompt-lengths-2026-10-23.json")
PROMPT_KEY = {"longbench_hotpotqa": "hotpotqa_e", "longbench_passage_retrieval": "passage_retrieval_en_e",
              "longbench_repobench": "repobench-p_e", "longbench_v2": "longbench_v2", "needlebench_niah": "needlebench_niah",
              "gsm8k": "gsm8k"}
MAX_NEW = {"longbench_hotpotqa": 32, "longbench_passage_retrieval": 32, "longbench_repobench": 64, "longbench_v2": 128,
           "needlebench_niah": 32, "gsm8k": 256}
WEIGHTS_B = 6171926992              # the two bf16 safetensors shards (the task's hub manifest: 3,968,658,944 + 2,203,268,048)
KV_B_PER_TOKEN = 36 * 2 * 128 * 2 * 2   # 36 layers x 2 KV heads x 128 head_dim x (K, V) x 2 bytes (Qwen2.5-3B config.json)
H, I, NQ = 2048, 11008, 16
CTX_B = 300 * 2 ** 20                # the CUDA context and the caching allocator's slack, a fixed overhead


def peak_memory(fam, w):
    """(bytes, how) for one evaluator family on one workload."""
    import json as _json
    L = _json.load(open(PROMPT_LENGTHS))[PROMPT_KEY[w]]
    T = (L["max_after_selection_truncation"] if fam == "selection" else L["max"]) + MAX_NEW[w]
    copies = 3 if fam == "quant" else 1
    ws = 2 * (2 * T * H + T * (NQ + 4) * 128 + 3 * T * I)
    peak = WEIGHTS_B + copies * KV_B_PER_TOKEN * T + ws + CTX_B
    how = (f"derived: weights {WEIGHTS_B:,} B (bf16 shards) + {copies} x KV cache at the longest prompt plus its new tokens, "
           f"{T:,} tokens x {KV_B_PER_TOKEN:,} B/token (36 layers x 2 KV heads x 128 x K,V x bf16) + one layer's prefill working "
           f"set at that length ({ws:,} B: hidden states and residual, q/k/v, the MLP's gate, up and activation; sdpa, and "
           "logits_to_keep=1 so no vocabulary-sized logits) + 300 MiB for the CUDA context and allocator slack. "
           + ("The quant evaluator holds the live cache, its snapshot and the quantised snapshot during a decode step "
              "(snapshot_cache, quantize_snapshot, restore_cache), hence 3 copies; it does not truncate prompts. "
              if fam == "quant" else
              "The selection evaluator truncates prompts to 32,768 - max_new_tokens - 1 (tokenize_prompt_for_model); its "
              "compressed cache is smaller than this full-length one during decode, but prefill builds the full cache first. ")
           + f"Prompt lengths measured with Qwen's tokenizer (survey/mlskv-prompt-lengths-2026-10-23.json); per-method "
           "differences inside a family are ignored. Not measured on hardware")
    return peak, how


def workload(r):
    a = r["identity"]["argv"]
    return a[a.index("--workload") + 1]


def n_full(w):
    if w == "needlebench_niah":
        return C.tv(NEEDLE_N, "derived", "build_needlebench_examples in custom_quant_eval.py: every essay of opencompass/NeedleBench's "
                    "PaulGrahamEssays.jsonl (147 rows, counted on the staged cache by gpu-mock 2026-10-22) at each of NEEDLE_DEPTHS "
                    "(0.10, 0.50, 0.90): 441 examples. Every row's binary score is a whole number of hits at 441 (237/285/288/288). "
                    "Was 147, the essays alone, until W0-mlskv-check10b paced 441 examples at a third of their time (2026-10-22)")
    if w in PINNED_N:
        how = ("the only example count at which every row's binary final_score is a whole number of hits (final_score x n / 100), "
               "over both KV tasks' leaderboards")
        if w == "longbench_v2":
            how += ("; the staged THUDM/LongBench-v2 train split's data.json holds 503 rows (gpu-mock, counted on the staged cache "
                    "2026-10-22 18:15Z)")
        return C.tv(PINNED_N[w], "derived", how)
    return C.tv(LONGBENCH_E_N, "recorded", "non-empty rows of data/<dataset>.jsonl in THUDM/LongBench data.zip at 5e628be4, as the "
                "evaluator's load_longbench_rows reads them, counted on the staged cache 2026-10-22 18:15Z (gpu-mock: hotpotqa_e, "
                "passage_retrieval_en_e and repobench-p_e 300 each); it agrees with the LongBench card's LongBench-E table, 100 per "
                "length band")


def serving_block(r, fam, w):
    s = dict((r.get("materialisation") or {}).get("serving") or {})
    if isinstance(s.get("decode"), dict):
        # gen_mlsbench's "examples: not recorded" predates n_examples_full; point it there rather than leave two answers
        s["decode"] = dict(s["decode"], examples=C.tv(n_full(w)["value"], n_full(w)["provenance"], "see n_examples_full"))
    s.update({
        "mode": "per_example",
        "generate_seam": GENERATE_SEAM[fam],
        "score_seam": "score_prediction",
        "n_examples_full": n_full(w),
        "score_kind": SCORE_KIND[w],
        "scale": [0, 100],
        "subset_source": {"flag": "--max-examples", "env": SUBSET_ENV[fam],
                          "how": "the evaluator's --max-examples, whose default is the environment variable; the flag wins when given, "
                                 "and 0 means the full workload"},
        "load_s": C.tv(LOAD_S, "recorded", "Qwen2.5-3B-Instruct in bf16 through gpu-mock on the E2 (e2-standard-32, the TRL "
                       "image): from_pretrained 2.0 s plus .to(\"cuda\") 4.0 s (gpu-mock, tests/smoke_qwen_load_in_image.sh, "
                       "2026-10-22 16:26Z). Provisional for the fleet's c3-highmem-44: that machine's measuring run replaces it in "
                       "the host terms (the study, 2026-10-22 16:55Z)"),
        "how": ("gpu-mock's per_example serving (experiments/gpu-mock/docs/MLS-KV-SERVING.md): the generate seam returns without "
                "decoding; the score seam returns the i-th entry of a per-example score vector on the 0-100 scale whose mean over "
                "n_examples_full is the recorded final_score; each example paces (wall_total_s - load_s) / n_examples_full. A "
                "--max-examples k launch takes the first k entries and is labelled synthesised_subset"),
    })
    return s


def main():
    reg = sys.argv[1]
    recs = C.load_records(os.path.join(reg, "records"))
    n = 0
    for r in sorted(recs.values(), key=lambda r: r["identity"]["record_id"]):
        fam = KV_TASKS.get((r["identity"].get("source_key") or {}).get("task"))
        if not fam:
            continue
        w = workload(r)
        # A --max-examples k launch is this record's first k examples (contract §2), so the flag must not change which record
        # matches: exempt it from the exact key. The selection record's argv carries --max-examples 0; the quant scripts never
        # pass it, so an agent that adds it would otherwise miss the record.
        ident = r["identity"]
        ident["argv_exempt"] = sorted(set(ident.get("argv_exempt") or []) | {"--max-examples"})
        # gpu-mock's path_exact stage: an edited copy of the evaluator at
        # this path, launched with the recorded argv, is served this record, both entry hashes logged. The path is the one every
        # KV task lays out for the record (gen_mlskv_task.launch_dir + EVALUATOR); match_policy stays exact_only, since a
        # path_bound record goes to the older path stage, which ignores argv
        import gen_mlskv_task as KT
        method = (r["identity"].get("source_key") or {}).get("baseline")
        ident["bound_path"] = f"{KT.launch_dir(method)}/{KT.EVALUATOR[fam]}"
        ident["bound_path_how"] = ("the evaluator's path from the task root in mls-kv-deploy, mls-kv-report and mls-kv-gate (one "
                                   "per method folder); gpu-mock matches it as a suffix of the live script's resolved path, with "
                                   "the recorded argv (path_exact)")
        ident["argv_exempt_how"] = ("flags whose values are paths the caller chooses (a checkpoint under OUTPUT_DIR) or the model's "
                                    "location, and --max-examples: a k-example launch is this record's first k examples, served "
                                    "as a labelled subset (gpu-mock's MLS-KV-SERVING.md §2), so it must match the same record")
        peak, how = peak_memory(fam, w)
        r["memory"] = dict(r.get("memory") or {}, peak=C.tv(peak, "derived", how), peak_allocated_bytes=peak,
                           # what the card holds once the weights are on it, before the first example (gpu-mock, 2026-10-23
                           # 17:51Z: held from engagement, stepping up to the peak at the first example, so nvidia-smi in the
                           # first seconds of a job shows the weights arriving, not 0 MiB)
                           weights_bytes=C.tv(WEIGHTS_B + CTX_B, "derived",
                                              f"the bf16 weights, {WEIGHTS_B:,} B (the two safetensors shards in the task's hub "
                                              "manifest), plus 300 MiB for the CUDA context and allocator slack: what nvidia-smi "
                                              "shows after from_pretrained(...).to('cuda') and before the first prompt"))
        m = r.setdefault("materialisation", {})
        m["serving"] = serving_block(r, fam, w)
        m["blockers"] = [SERVING_BLOCKER]
        r["readiness"] = C.readiness(r)
        C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
        n += 1
    print(f"MLS-Bench KV records with per_example serving: {n}")


if __name__ == "__main__":
    main()
