"""Write a package's measured feasible-best reference into its speed-1 expected file.

usage: python3 -I build/add_feasible_best.py <registry> <task_id> <feasible-best json> <from-run text>

expected_scores.py's packed_overlap ignores preparation, and the scorer's hindsight releases a preparation package's jobs at 0 s,
so neither is a feasible schedule for H4. Its reference is the best recipe order on a measuring run's own times
(survey/h4-feasible-best-2026-10-24/feasible_best.py; the analysis, 2026-10-24 20:24 MDT, option a). A source file that
carries its own `how` and `use` (mls-kv-report-h, 2026-10-25) is written as it states them. rebuild.sh calls this after writing
the expected files, so a rebuild keeps it.
"""
import json, os, sys
reg, tid, src, run = sys.argv[1:5]
p = os.path.join(reg, "expected", f"{tid}.c3-highmem-44.host.json")
e, fb = json.load(open(p)), json.load(open(src))
e["feasible_best_s"] = fb["feasible_best_s"]
if fb.get("how") and fb.get("use"):
    # a reference whose derivation the source file states itself (mls-kv-report-h, 2026-10-25: the packed policy's makespan)
    e["feasible_best"] = {"value_s": fb["feasible_best_s"], "from": run, "how": fb["how"], "use": fb["use"],
                          **({"best_schedule_s": fb["best_schedule_s"]} if "best_schedule_s" in fb else {})}
    json.dump(e, open(p, "w"), indent=1)
    print(f"{tid}: feasible_best_s {fb['feasible_best_s']} into {p}")
    sys.exit(0)
e["feasible_best"] = {"value_s": fb["feasible_best_s"], "order": fb["feasible_best_order"], "from_run": run,
                      "how": ("the best of all orders of the recipes under runner/scripted/agent.py's one-preparer rule (one blocking "
                              "preparer; after each preparation every prepared job that fits on the four GPUs is launched; a "
                              "recipe's jobs share one command), on that run's own measured preparation spans and job durations; "
                              f"{os.path.relpath(src)} and the feasible_best.py beside it. The run's own order simulates to "
                              f"{fb['run_order_simulated_s']:,} s against its {fb['run_makespan_s']:,} s"),
                      "use": ("bars 2 and 4-6 read makespan to the last job's end / feasible_best_s (the analysis, 2026-10-24 "
                              f"20:24 MDT, option a); bar 2 = {fb['bar2']}. The scorer's hindsight bound releases every job at 0 s, "
                              "since a job's launch names only its training script, so it holds no preparation")}
json.dump(e, open(p, "w"), indent=1)
print(f"{tid}: feasible_best_s {fb['feasible_best_s']} into {p}")
