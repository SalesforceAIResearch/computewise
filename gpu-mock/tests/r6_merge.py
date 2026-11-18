#!/usr/bin/env python3
"""r6: merge a record's decode slices (decode_first_k.py's <record_id>.<s>-<e>.json)
into the one file per record that gpu-tasks' importer reads
(build/gen_mlskv_first50.py): examples i = 0..k-1 in order, the record's own
argv with `--cpu --max-examples k` appended (what the evaluator would be run
with for the whole first k), the slices' decode facts, and the slices listed.
A record whose slices do not tile [0, k) exactly, or any of whose slices is
incomplete or has a key that does not match its order position, is not
written, and the reason is printed.

    python3 -I r6_merge.py --slices D --out O [--k 50]
"""
import argparse
import collections
import json
import os
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slices", required=True, action="append", help="a dir of slice files; repeat for each machine")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--machine", default=None, help="fills a slice whose machine is a container hostname")
    ap.add_argument("--image", default=None, help="fills a slice that names no image")
    ap.add_argument("--harness-sha256", default=None, help="fills a slice decoded before the harness recorded its own hash")
    ap.add_argument("--harness-before", action="append", default=None, metavar="STAMP=SHA",
                    help="a slice whose decode started before STAMP (UTC, like 2026-10-29T16:22:00Z) ran harness SHA")
    a = ap.parse_args()
    a.harness_before = [tuple(x.split("=", 1)) for x in (a.harness_before or [])]
    by = collections.defaultdict(list)
    # --slices may be given more than once (one dir per machine). A slice decoded on two
    # machines must be identical in its keys, scores and predictions, or the record is
    # refused; the copy kept is the one whose decode started first, so a record's file
    # does not depend on which machine's copy was read last.
    found = collections.defaultdict(list)
    for sdir in a.slices:
        for name in os.listdir(sdir):
            m = re.match(r"^(.+)\.(\d+)-(\d+)\.json$", name)
            if m:
                found[(m.group(1), int(m.group(2)), int(m.group(3)))].append(os.path.join(sdir, name))
    clash = set()
    for (rid, s0, e0), paths in found.items():
        docs = [json.load(open(p)) for p in paths]
        body = {json.dumps([(x["key"], x["score"], x["prediction"]) for x in d["examples"]]) for d in docs}
        if len(body) > 1:
            print(f"REFUSED {rid}: slice [{s0},{e0}) decoded on two machines with different results ({paths})")
            clash.add(rid)
            continue
        keep = min(zip(paths, docs), key=lambda pd: pd[1]["decode"]["start"])[0]
        by[rid].append((s0, e0, keep))
    for rid in clash:
        by.pop(rid, None)
    os.makedirs(a.out, exist_ok=True)
    wrote = refused = 0
    for rid, parts in sorted(by.items()):
        parts.sort()
        docs = [json.load(open(p)) for _, _, p in parts]
        k = min(a.k, docs[0]["n_examples_full"])
        why = None
        cur = 0
        for (s, e, _), d in zip(parts, docs):
            if s != cur:
                why = f"gap or overlap at {cur} (next slice starts at {s})"
                break
            if not d.get("complete") or d.get("keys_mismatched"):
                why = f"slice [{s},{e}) incomplete or keys mismatched {d.get('keys_mismatched')}"
                break
            cur = e
        if why is None and cur != k:
            why = f"slices cover [0,{cur}), not [0,{k})"
        ev = {d["evaluator"]["sha256"] for d in docs}
        if why is None and len(ev) != 1:
            why = f"slices ran different evaluators {ev}"
        if why:
            print(f"REFUSED {rid}: {why}")
            refused += 1
            continue
        exs = [x for d in docs for x in d["examples"]]
        argv = [str(x) for x in docs[0]["argv"]]
        # gpu-tasks' importer accepts the record's own argv exactly, or (selection
        # records, whose argv names --max-examples 0) that with --cpu --max-examples k
        # in its place; a quant record's argv names no --max-examples and stays as is
        if "--max-examples" in argv:
            i = argv.index("--max-examples")
            del argv[i:i + 2]
            argv += ["--cpu", "--max-examples", str(k)]
        dec = dict(docs[0]["decode"])
        dec["start"] = min(d["decode"]["start"] for d in docs)
        dec["end"] = max(d["decode"]["end"] for d in docs)
        # slices decoded before the runner passed R6_MACHINE / R6_IMAGE into the
        # container carry the container's hostname and no image: --machine and
        # --image name them (the batch unit's own environment)
        for d in docs:
            if a.machine and re.fullmatch(r"[0-9a-f]{12}", str(d["decode"].get("machine") or "")):
                d["decode"]["machine"] = a.machine
            if a.image and not d["decode"].get("image"):
                d["decode"]["image"] = a.image
        dec["machine"] = ",".join(sorted({d["decode"]["machine"] for d in docs}))
        dec["image"] = ",".join(sorted({str(d["decode"].get("image")) for d in docs}))
        dec["no_replay"] = all(d["decode"].get("no_replay") for d in docs)
        dec["invocation"] = ("harness import of the unedited evaluator (sha256 = entry_hash); bf16 set by wrapping "
                             "from_pretrained at run time; the evaluator's own main() and run_real_eval under the record's "
                             "argv, on load_workload(name, 0)[start:end]; score_prediction wrapped to record each score")
        # gpu-tasks' importer reads slices as [[start, end], ...], tiling [0, k) in order;
        # each slice's machine, wall time and stamps go beside it in slice_detail
        # each slice's harness file by sha256; slices decoded before the harness
        # recorded its own hash carry none, and --harness-sha256 names the file used
        # the harness version a slice ran, when it predates the harness recording its own
        # hash: --harness-before "<UTC stamp>=<sha256>" names the version staged before a
        # stamp (a slice whose decode started before it ran that file; repeatable, earliest first)
        def harness_of(d):
            h = d["decode"].get("harness_sha256")
            if h:
                return h
            for stamp, sha in sorted(a.harness_before or []):
                if d["decode"]["start"] < stamp:
                    return sha
            return a.harness_sha256 or "unrecorded"
        for d in docs:
            d["decode"]["harness_sha256"] = harness_of(d)
        hs = sorted({d["decode"]["harness_sha256"] for d in docs})
        dec["harness_sha256"] = hs[0] if len(hs) == 1 else ",".join(hs)
        dec["slices"] = [[d["start"], d["end"]] for d in docs]
        dec["slice_detail"] = [{"start": d["start"], "end": d["end"], "machine": d["decode"]["machine"], "wall_s": d["wall_s"],
                                "decode_start": d["decode"]["start"], "decode_end": d["decode"]["end"],
                                "harness_sha256": d["decode"]["harness_sha256"]} for d in docs]
        out = {"record_id": rid, "evaluator": docs[0]["evaluator"], "argv": argv, "k": k,
               "examples": [{"i": x["i"], "example_id": x["example_id"], "key": x["key"], "score": x["score"],
                             "prediction": x["prediction"]} for x in exs],
               "decode": dec}
        fn = os.path.join(a.out, f"{rid}.json")
        with open(fn + ".tmp", "w") as fh:
            json.dump(out, fh)
        os.replace(fn + ".tmp", fn)
        mean = sum(x["score"] for x in exs) / len(exs)
        print(f"wrote {rid}: k {k}, {len(parts)} slice(s), first-{k} mean {mean:.2f}")
        wrote += 1
    print(f"{wrote} written, {refused} refused")
    return 0 if refused == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
