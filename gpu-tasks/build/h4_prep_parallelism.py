"""H4's prep_default_parallelism label: did a run filter any recipe with datatrove's default of one task.

usage: python3 -I build/h4_prep_parallelism.py <run dir> [--json]

the analysis' ruling, 2026-10-24 16:41 MDT: an H4 run cut at the horizon fails bar 3 as an early exit, and the default is
reported as its own label. For each recipe the agent prepared, this reads:
  executor   runs/<recipe>/filter-logs/executor.json in the copied workspace: datatrove's own record of the filter stage it
             launched (tasks, workers, world_size). The tool rewrites it on every run, so it is the last preparation's.
  processes  every process whose command line names tools/prepare.py <recipe>, with its span, from isolation's process records
             in resources.jsonl (read by runner/score/prep.py's _procs, the reader the prep scorer uses); a preparation the
             agent started again is two processes.
A recipe is at the default when its executor shows tasks == 1, or, with no executor (the filter stage never wrote one: the run
ended first), when a prepare.py process for it ran without --tasks or with --tasks 1. 'none' has no filters, so its preparation
is tokenising only and never counts as at the default.
Reads files only; runs nothing.
"""
import collections
import json
import os
import re
import sys

PREP = re.compile(r"(?:^|\s)(?:\S*/)?tools/prepare\.py\s+([\w-]+)(.*)$")


def tasks_of(args):
    m = re.search(r"--tasks(?:=|\s+)(\d+)", args)
    return int(m.group(1)) if m else 1


def ts(t):
    import datetime
    return datetime.datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()


def main():
    d = sys.argv[1]
    task = json.load(open(os.path.join(d, "task.json")))
    recipes = sorted({j["job_id"].rsplit("-seed", 1)[0] for j in task["jobs"]})
    root = next((p for p in (os.path.join(d, "workspace", "home", "project"), os.path.join(d, "workspace")) if os.path.isdir(p)), None)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "runner", "score"))
    import prep as _prep                                  # runner/score/prep.py: the box's process spans
    spans = collections.defaultdict(list)                 # recipe -> [(start, end, cmdline)]
    procs_all = _prep._procs(d, full=True)
    prep_pids = {pid for _, _, c, pid, _ in procs_all if PREP.search(str(c or "")) and not str(c).startswith("sh -c ")}
    for a, b, c, pid, ppid in procs_all:
        c = str(c or "")
        if c.startswith("sh -c ") or ppid in prep_pids:
            continue                                      # the shell wrapping it, and the tool's own workers (same cmdline)
        m = PREP.search(c)
        if m and m.group(1) in recipes:
            spans[m.group(1)].append((a, b, c))
    t0 = ts(json.load(open(os.path.join(d, "status.json")))["start_utc"])
    rows = []
    for r in recipes:
        ex = None
        p = os.path.join(root, "runs", r, "filter-logs", "executor.json") if root else None
        if p and os.path.exists(p):
            try:
                e = json.load(open(p))
                ex = {k: e.get(k) for k in ("tasks", "workers", "world_size")}
            except ValueError:
                ex = None
        procs = [{"cmdline": c, "tasks": tasks_of(PREP.search(c).group(2)), "start_s": round(a - t0, 1), "end_s": round(b - t0, 1),
                  "seen_s": round(b - a, 1)} for a, b, c in sorted(spans[r])]
        filters = r != "none"                             # 'none' only tokenises: its filter stage is empty, the default costs nothing
        at_default = filters and ((ex["tasks"] == 1) if ex and ex.get("tasks") is not None else any(x["tasks"] == 1 for x in procs))
        default_s = sum(x["seen_s"] for x in procs if x["tasks"] == 1) if filters else 0.0
        rows.append({"recipe": r, "prepared": bool(procs or ex), "executor": ex, "processes": procs,
                     "at_default": bool(procs or ex) and at_default, "seconds_at_default": round(default_s, 1)})
    out = {"run": os.path.basename(os.path.normpath(d)), "task_id": task["task_id"],
           "prep_default_parallelism": {"recipes_at_default": sum(1 for x in rows if x["at_default"]),
                                        "recipes_prepared": sum(1 for x in rows if x["prepared"]),
                                        "hours_at_default": round(sum(x["seconds_at_default"] for x in rows) / 3600, 2)},
           "recipes": rows,
           "how": "build/h4_prep_parallelism.py: executor.json's tasks, else the prepare.py command lines in resources.jsonl"}
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return
    s = out["prep_default_parallelism"]
    print(f"{out['run']}: {s['recipes_at_default']} of {s['recipes_prepared']} prepared recipes at the default (one task), "
          f"{s['hours_at_default']} h filtering at it")
    for x in rows:
        ex = x["executor"] or {}
        print(f"  {x['recipe']:18s} {'DEFAULT' if x['at_default'] else 'raised ' if x['prepared'] else 'not prepared'} "
              f"executor tasks={ex.get('tasks')} workers={ex.get('workers')} | "
              + "; ".join(f"--tasks {p['tasks']} {p['start_s']:.0f}-{p['end_s']:.0f} s" for p in x["processes"]))


if __name__ == "__main__":
    main()
