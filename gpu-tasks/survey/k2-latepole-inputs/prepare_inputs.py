"""Prepare original input data forms, on the staging host only.

NIAH uses the upstream deterministic preparation script. The two QA JSONLs
follow upstream prepare_longbench.py's first-100-row transformation, reading
the pinned local LongBench ZIP instead of downloading another copy.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", type=Path, required=True)
    args = ap.parse_args()
    root = args.workspace.resolve()
    original = root / "source/vendor/data_scripts/sparse-attn-eval/prepare_niah.py"
    before = sys.argv
    try:
        sys.argv = [str(original), "--data-root", str(root / "data")]
        runpy.run_path(str(original), run_name="__main__")
    finally:
        sys.argv = before
    dest = root / "data/longbench-qasper"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "data/hf/THUDM/LongBench/data.zip") as z:
        for subset in ("qasper", "multifieldqa_en"):
            with z.open("data/" + subset + ".jsonl") as stream:
                rows = [json.loads(line) for line in stream if line.strip()][:100]
            with (dest / (subset + ".jsonl")).open("w") as out:
                for i, row in enumerate(rows):
                    obj = {"input": row.get("input", ""), "context": row.get("context", ""),
                           "answers": row.get("answers", []), "length": row.get("length", 0),
                           "_id": row.get("_id", str(i))}
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            print(subset, "actual input rows prepared", len(rows))


if __name__ == "__main__":
    main()
