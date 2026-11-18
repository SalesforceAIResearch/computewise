"""Finish one builder's task through gen_tasks' own per-task steps, without regenerating every task.

usage: python3 -I build/finish_one_task.py <registry> <builder module> [--kw '<builder kwargs as JSON>'] [--out <dir>]

It calls gen_tasks.complete_task, the function gen_tasks.main applies to every builder's task (finish, task_class, model_runs,
brief variants, rerun siblings), so the file it writes is the one a full `gen_tasks.py <registry>` writes for that task. For
merging one package quickly while others are live; a full rebuild still confirms it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import gen_tasks as GT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("registry")
    ap.add_argument("module")
    ap.add_argument("--kw", default="{}")
    ap.add_argument("--out")
    a = ap.parse_args()
    GT.REG_DIR = a.registry
    recs = C.load_records(os.path.join(a.registry, "records"))
    import siblings as SIB
    t = __import__(a.module).build(recs, GT.disclosure_of, a.registry, **json.loads(a.kw))
    t = GT.complete_task(t, recs, a.registry, SIB.sibling_pools(recs))
    path = os.path.join(a.out or os.path.join(a.registry, "tasks"), t["task_id"] + ".json")
    C.write_json(path, t)
    print(f"{t['task_id']}: {path}")


if __name__ == "__main__":
    main()
