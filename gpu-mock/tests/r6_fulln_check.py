#!/usr/bin/env python3
"""r6 fidelity: one record's full-n CPU decode against its recorded final_score.
Reads the merged file r6_merge.py writes with --k n (examples i = 0..n-1), checks
every example's key against gpu-tasks' order table at its position, and prints
hits, the full-n mean, the recorded final_score and the difference in points and
hits. That difference is the number to report, as an effect size: both decodes
ran the same examples, so no sampling noise enters it (examples that decode
differently, net). `z` is the difference over the workload's binomial SE. That
answers a different question (would a fresh sample of the workload differ),
and no paired SE is possible: MLS-Bench publishes no per-example H100 vector.

    python3 -I r6_fulln_check.py --merged M --record R --order O
"""
import argparse
import json
import math


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--order", required=True)
    a = ap.parse_args()
    m, r, o = (json.load(open(p)) for p in (a.merged, a.record, a.order))
    n = int(r["materialisation"]["serving"]["n_examples_full"]["value"])
    exs = m["examples"]
    bad = [x["i"] for x in exs if o["order"][x["i"]] != x["key"] or o["ids"][x["i"]] != x["example_id"]]
    assert [x["i"] for x in exs] == list(range(n)), "examples do not cover 0..n-1 in order"
    hits = sum(1 for x in exs if x["score"] >= 50.0)
    mean = sum(x["score"] for x in exs) / n
    rec = float(r["result"]["final_score"]["value"])
    rec_hits = rec * n / 100.0
    se = math.sqrt(rec / 100 * (1 - rec / 100) / n) * 100
    print(json.dumps({"record_id": r["identity"]["record_id"], "n": n, "keys_mismatched": bad[:10], "n_keys_mismatched": len(bad),
                      "cpu_hits": hits, "cpu_mean": round(mean, 6), "recorded_final_score": rec, "recorded_hits": round(rec_hits, 3),
                      "diff_points": round(mean - rec, 6), "diff_hits": round(hits - rec_hits, 3), "se_binomial": round(se, 4),
                      "z": round((mean - rec) / se, 3), "slices": m["decode"]["slices"], "machine": m["decode"]["machine"],
                      "harness_sha256": m["decode"]["harness_sha256"], "evaluator_sha256": m["evaluator"]["sha256"]}, indent=1))


if __name__ == "__main__":
    main()
