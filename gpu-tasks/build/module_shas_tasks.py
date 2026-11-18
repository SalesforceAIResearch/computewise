"""materialisation.serving.module_shas for the records the named tasks serve, by build/module_shas.py's own derivation run over
those tasks only (module_shas.py walks every task in the registry, about 15 minutes; this takes seconds).

usage: python3 -I build/module_shas_tasks.py <registry> <task_id> [...]
"""
import json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import module_shas as M  # noqa: E402


def main():
    reg, tids = sys.argv[1], sys.argv[2:]
    maps = {}
    for tid in tids:
        task = json.load(open(os.path.join(reg, "tasks", tid + ".json")))
        staged, roots = M.staged_py(task, reg)
        for j in (task.get("jobs") or []) + (task.get("unneeded_jobs") or []):
            e = M.job_entry(j)
            m = M.entry_map(task, e, staged, roots) if e else None
            if m is None:
                continue
            for rid in j.get("records") or []:
                if rid in maps and maps[rid] != m:
                    sys.exit(f"{rid} gets different maps from different tasks")
                maps.setdefault(rid, m)
    n = 0
    for f in sorted(os.listdir(os.path.join(reg, "records"))):
        if not f.endswith(".json") or f.startswith("example-order"):
            continue
        path = os.path.join(reg, "records", f)
        r = json.load(open(path))
        rid = r["identity"]["record_id"]
        if rid not in maps or rid.startswith("k2-latepole-"):   # K2's builder writes and checks its own
            continue
        sv = r.setdefault("materialisation", {}).setdefault("serving", {"mode": "per_op"})
        if sv.get("module_shas") != maps[rid]:
            sv["module_shas"] = maps[rid]
            C.write_json(path, r)
            n += 1
    print(f"module_shas: {n} records written; {len(maps)} records served by {', '.join(tids)}")


if __name__ == "__main__":
    main()
