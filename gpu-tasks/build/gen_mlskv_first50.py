"""Measured first-K per-example scores on the KV records, for gpu-mock's r6 (gpu-tasks, 2026-10-29; format agreed with gpu-mock
09:02-09:25 MDT, the study's r6 approval via analysis).

usage: python3 -I build/gen_mlskv_first50.py <registry> <harness dir> [--check-only]

<harness dir> holds gpu-mock's decode harness output, one JSON file per record: the record's own evaluator, run for real (real
model, real data, nothing served) on the first K examples with the record's own arguments. Each file carries
  {"record_id", "evaluator": {"path", "sha256"}, "argv": [...], "k": K,
   "examples": [{"i", "example_id", "key", "score", "prediction"}, ...],
   "decode": {"device", "dtype", "torch", "transformers", "model_revision", "attn_implementation", "image", "machine",
              "start", "end", "no_replay": true, "invocation": "...", "slices": [[start, end], ...]}}
argv is the record's own identity.argv, the values the harness gave the evaluator; or, for a command-line run, that argv with
--cpu and --max-examples K. invocation says how the evaluator was called (gpu-mock's harness imports the unmodified evaluator,
loads the model itself and calls the evaluator's own generate and score functions on load_workload(name, 0)[start:end]); slices
are the shards merged into the file, which must tile [0, K). The evaluator file itself must be unmodified: a changed dtype is
stated in decode, never by editing the file.
For each file this writes materialisation.serving.per_example_scores into the record:
  {"k", "key": "sha256(example_id\\0prompt)", "keys", "example_ids", "scores", "provenance": "measured", "how", "source"}
Predictions stay in the harness files (to re-score without re-decoding); the record keeps only what the daemon serves.

Checks, each fatal for that file (nothing is written for it):
- the evaluator's sha256 is the record's own script artifact;
- the argv is the record's own, or it with --cpu and --max-examples K (the selection records carry --max-examples 0, which the
  command-line form replaces); the decode's slices, when given, tile [0, K) in order;
- examples are i = 0..K-1, and their keys and example_ids equal the first K of the family's example-order table, so position i
  here is order position i there;
- each score is within the record's scale, and 0 or 100 for a binary record;
- the tail is feasible: 0 <= final_score x n - sum(scores) <= (n - K) x 100, a whole number of hits for a binary record, so
  gpu-mock can complete the vector to the recorded final_score.
Records with no harness file are left as they are. Standard library; idempotent.
"""
import glob
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FAMILY = {"llm-kv-selection-budgeting": "selection", "llm-kv-adaptive-quantization": "quant"}
KEY = "sha256(example_id\\0prompt)"


def tables(reg):
    out = {}
    for f in sorted(glob.glob(os.path.join(reg, "records", "example-order-*.json"))):
        t = json.load(open(f))
        for fam in t["families"]:
            out[fam, t["workload"]] = (os.path.basename(f), t)
    return out


def sha256_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def strip_flags(argv, k=None):
    """argv without the script path, --cpu and --max-examples (whose value must be k when k is given)."""
    out, i = [], 1
    while i < len(argv):
        a = argv[i]
        if a == "--cpu":
            i += 1
            continue
        if a == "--max-examples":
            if k is not None and argv[i + 1] != str(k):
                raise ValueError(f"--max-examples {argv[i + 1]}, not {k}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def check(reg, r, h, tabs):
    """The per_example_scores block for record r from harness file h, or a list of errors."""
    errs = []
    s = r["materialisation"]["serving"]
    fam = FAMILY[r["identity"]["source_key"]["task"]]
    wl = s["decode"]["workload"]
    n = s["n_examples_full"]["value"]
    k = h.get("k")
    script = os.path.join(reg, r["agent_view"]["script"])
    if (h.get("evaluator") or {}).get("sha256") != sha256_file(script):
        errs.append(f"evaluator sha256 {str((h.get('evaluator') or {}).get('sha256'))[:12]} is not the record's "
                    f"{r['agent_view']['script']}")
    try:
        ha = h.get("argv") or []
        if ha != r["identity"]["argv"]:
            if strip_flags(ha) != strip_flags(r["identity"]["argv"]):
                errs.append(f"argv {ha} is not the record's {r['identity']['argv']} (or it plus --cpu --max-examples {k})")
            else:
                strip_flags(ha, k)                                         # a command-line run's --max-examples must be k
    except (ValueError, IndexError) as e:
        errs.append(f"argv: {e}")
    sl = (h.get("decode") or {}).get("slices")
    if sl is not None and (not all(isinstance(x, list) and len(x) == 2 for x in sl)
                           or [x[0] for x in sl] != [0] + [x[1] for x in sl[:-1]] or sl[-1][1] != k):
        errs.append(f"decode slices {sl} do not tile [0, {k})")
    if (fam, wl) not in tabs:
        return errs + [f"no example-order table for {fam} {wl}"]
    tname, t = tabs[fam, wl]
    ex = h.get("examples") or []
    if not isinstance(k, int) or not 0 < k <= n or len(ex) != k:
        return errs + [f"k {k}, {len(ex)} examples, n {n}"]
    if [e.get("i") for e in ex] != list(range(k)):
        errs.append("examples are not i = 0..k-1 in order")
    keys, ids = [e.get("key") for e in ex], [str(e.get("example_id")) for e in ex]
    if keys != t["order"][:k]:
        bad = next(i for i in range(k) if keys[i] != t["order"][i])
        errs.append(f"keys differ from {tname} order[:{k}], first at {bad}")
    if ids != t["ids"][:k]:
        errs.append(f"example_ids differ from {tname} ids[:{k}]")
    scores = [e.get("score") for e in ex]
    lo, hi = s["scale"]
    if not all(isinstance(x, (int, float)) and lo <= x <= hi for x in scores):
        errs.append(f"a score outside {s['scale']}")
    elif s["score_kind"] == "binary" and not all(x in (lo, hi) for x in scores):
        errs.append("a binary record's score is neither 0 nor 100")
    else:
        rest = r["result"]["final_score"]["value"] * n - sum(scores)
        if not -1e-6 * n <= rest <= (n - k) * hi + 1e-6 * n:
            errs.append(f"tail infeasible: the other {n - k} examples would need to sum to {rest:.3f} (0 to {(n - k) * hi})")
        elif s["score_kind"] == "binary" and abs(rest / hi - round(rest / hi)) > 1e-3:   # final_score is rounded to 6 decimals
            errs.append(f"tail is {rest / hi:.4f} hits, not a whole number")
    if errs:
        return errs
    d = h.get("decode") or {}
    hs = sorted({x.get("harness_sha256") for x in d.get("slice_detail") or [] if x.get("harness_sha256")}
                | ({d["harness_sha256"]} if d.get("harness_sha256") else set()))
    arch = os.path.join(reg, "..", "..", "gpu-mock", "archive", "r6-harness")
    harness = "; harness " + ", ".join(
        f"{x[:12]} ({'archived as experiments/gpu-mock/archive/r6-harness/decode_first_k.' + x + '.py' if os.path.exists(os.path.join(arch, f'decode_first_k.{x}.py')) and sha256_file(os.path.join(arch, f'decode_first_k.{x}.py')) == x else 'not found in gpu-mock/archive/r6-harness'})"
        for x in hs) if hs else "; harness sha256 not stated"
    served = ("nothing served (the harness: " + (d.get("how") or "no_replay true") + ")" if d.get("no_replay") is True
              else "the harness file does not say replay was off")
    how = (f"measured by gpu-mock's decode harness: the record's own unmodified evaluator ({r['agent_view']['script']}) with the "
           f"record's arguments on its first {k} examples, real model and data, {served}; {d.get('invocation') or 'invocation not stated'}"
           f"{'; slices ' + str(d.get('slices')) if d.get('slices') else ''}{harness}; {d.get('device')} {d.get('dtype')}"
           f"{' (' + d['cpu'] + ')' if d.get('cpu') else ''}, torch {d.get('torch')}, "
           f"transformers {d.get('transformers')}, model {d.get('model_revision')}, attention {d.get('attn_implementation')}, "
           f"image {d.get('image')}, {d.get('machine')}, {d.get('start')} to {d.get('end')}. Greedy decoding. The recorded run was "
           "on CUDA (H100) in bf16; this decode's kernels are not those, so a close-call token can differ and so can that example's "
           f"score. Position i is order position i of {tname}; examples past {k} stay synthesised")
    return {"k": k, "key": KEY, "keys": keys, "example_ids": ids, "scores": scores, "provenance": "measured", "how": how,
            "source": None}


def main():
    reg, hdir = sys.argv[1], sys.argv[2]
    check_only = "--check-only" in sys.argv[3:]
    recs = C.load_records(os.path.join(reg, "records"))
    tabs = tables(reg)
    done = bad = 0
    for f in sorted(glob.glob(os.path.join(hdir, "*.json"))):
        h = json.load(open(f))
        r = recs.get(h.get("record_id"))
        if r is None or (r["identity"].get("source_key") or {}).get("task") not in FAMILY:
            bad += 1
            print(f"{os.path.basename(f)}: record {h.get('record_id')!r} is not a KV record")
            continue
        out = check(reg, r, h, tabs)
        if isinstance(out, list):
            bad += 1
            print(f"{os.path.basename(f)}: " + "; ".join(out))
            continue
        prev = r["materialisation"]["serving"].get("per_example_scores") or {}
        j = min(prev.get("k") or 0, out["k"])
        if j and (prev.get("scores")[:j] != out["scores"][:j] or prev.get("keys")[:j] != out["keys"][:j]):
            print(f"{os.path.basename(f)}: NOTE its first {j} scores or keys differ from the block already in the record "
                  f"(k {prev.get('k')}); written anyway, from this file")    # a label, not a refusal: a re-decode may be deliberate
        out["source"] = f"{os.path.relpath(f, os.path.join(reg, '..'))} sha256 {sha256_file(f)[:12]}"
        done += 1
        if not check_only:
            r["materialisation"]["serving"]["per_example_scores"] = out
            C.write_json(os.path.join(reg, "records", r["identity"]["record_id"] + ".json"), r)
    kv = sum(1 for r in recs.values() if (r["identity"].get("source_key") or {}).get("task") in FAMILY)
    print(f"{done} records {'checked' if check_only else 'given measured first-k scores'}, {bad} files refused, "
          f"{kv - done} KV records without a harness file")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
