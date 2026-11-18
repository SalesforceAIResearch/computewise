"""How closely a preparation run's agent-written scripts follow their references.

usage: python3 -I build/prep_fidelity.py <run dir> [--json]

For each job of the run's task that has prep.produces and prep.reference (the preparation arm), reads the script the agent
wrote (<run dir>/workspace/home/project/<produces[0]>, else <run dir>/workspace/<produces[0]>) and the reference the recorded
run used, and reports:
  stop      the reference's stop_after_step and the step constants in the agent's script (train_steps, num_iterations,
            stop_after_step, ...): a script that trains longer or shorter than the recorded run is served the record anyway,
            paced for the steps it runs, so this is what decides whether its timing is the record's;
  numbers   the numeric literals (outside comments and docstrings) found in one script and not the other: a changed learning
            rate, momentum or schedule constant shows here;
  ratio     difflib's line similarity, a rough measure of how much was rewritten.
gpu-mock serves the recorded run to the agent's script whatever its bytes (path_bound), so a departure here is not visible in
the losses the agent sees; this is the label that says whether the served run is the run the script would have produced.

A stop can also be a flag (G-h1-haiku-L0-r2, 2026-10-24: the agent gave its scripts the ready setups' `--stop-at`, kept the
schedule's length as its train_steps, and launched every screen with `--stop-at 1000`). So the stop also matches when a
launch of the job's script passed `--stop-at <the reference's step>` and paced exactly to it, and no launch of that script
paced past it (the occupancy log's job_hello argv and last enqueued step; a shorter smoke launch is no departure);
`stop_by` says which way it matched.
Reads files only; runs nothing.
"""
import difflib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REG = os.path.join(HERE, "..", "registry")
NUM = re.compile(r"(?<![\w.])(\d+\.\d+(?:e-?\d+)?|\d+e-?\d+|\d{3,})(?![\w.])")
STEP = re.compile(r"(?:stop_after_step|train_steps|num_iterations|num_scheduled_iterations|run_stop_step)\s*[:=]\s*(?:int\s*=\s*)?(\d+)")


def code_only(s):
    return re.sub(r'(?s)""".*?"""|#[^\n]*', "", s)


def flag_stops(run, rel):
    """The (--stop-at value, last paced step) of each launch of the script at task path `rel` in the run's occupancy log."""
    p = os.path.join(run, "occupancy.jsonl")
    if not os.path.exists(p):
        return []
    hello, last = {}, {}
    for line in open(p, errors="replace"):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("kind") == "job_hello" and isinstance(e.get("argv"), list) and e["argv"]:
            a = [str(x) for x in e["argv"]]
            if a[0] == rel or a[0].endswith("/" + rel):
                v = next((a[i + 1] for i in range(1, len(a) - 1) if a[i] == "--stop-at"),
                         next((x.split("=", 1)[1] for x in a[1:] if x.startswith("--stop-at=")), None))
                hello[e["job"]] = int(v) if v is not None and str(v).isdigit() else None
        elif e.get("kind") == "enqueue" and e.get("unit") == "step" and isinstance(e.get("step"), (int, float)):
            last[e.get("job")] = max(last.get(e.get("job"), -1), int(e["step"]))
    return [(hello[j], last.get(j)) for j in sorted(hello)]


def main():
    run = sys.argv[1]
    task = json.load(open(os.path.join(run, "task.json")))
    rows = []
    for j in task["jobs"]:
        prep = j.get("prep") or {}
        if not (prep.get("produces") and prep.get("reference")):
            continue
        rel = prep["produces"][0]
        path = next((p for p in (os.path.join(run, "workspace", "home", "project", rel), os.path.join(run, "workspace", rel))
                     if os.path.exists(p)), None)
        ref = open(os.path.join(REG, prep["reference"])).read()
        row = {"job_id": j["job_id"], "produces": rel, "reference": prep["reference"]}
        if path is None:
            row["written"] = False
            rows.append(row)
            continue
        a = open(path, errors="replace").read()
        r_num, a_num = set(NUM.findall(code_only(ref))), set(NUM.findall(code_only(a)))
        row.update({
            "written": True, "identical": a == ref,
            "ratio": round(difflib.SequenceMatcher(None, ref.splitlines(), a.splitlines(), autojunk=False).ratio(), 3),
            "stop_reference": [int(x) for x in re.findall(r"stop_after_step\s*=\s*(\d+)", ref)],
            "step_constants_agent": [int(x) for x in STEP.findall(a)],
            "numbers_only_in_reference": sorted(r_num - a_num), "numbers_only_in_agent": sorted(a_num - r_num),
        })
        row["stop_matches"] = bool(row["stop_reference"]) and row["stop_reference"][0] in row["step_constants_agent"]
        row["stop_by"] = "constant" if row["stop_matches"] else None
        if not row["stop_matches"] and row["stop_reference"]:
            fl = flag_stops(run, rel)
            row["launches_stop_flag"] = [{"stop_at": a, "last_step": b} for a, b in fl]
            want = row["stop_reference"][0]
            # a launch that passed --stop-at <the step> and paced to it, and none that trained past it (a smoke test stopped
            # earlier, Sonnet's --stop-at 125, is not a departure; a launch past the step is)
            if any(a == want and b == want for a, b in fl) and not any(b is not None and b > want for a, b in fl):
                row["stop_matches"], row["stop_by"] = True, "flag"
        rows.append(row)
    if "--json" in sys.argv:
        print(json.dumps({"run": os.path.basename(os.path.normpath(run)), "task_id": task["task_id"], "jobs": rows}, indent=1))
        return
    for r in rows:
        if not r.get("written"):
            print(f"{r['job_id']:24s} not written ({r['produces']})")
            continue
        print(f"{r['job_id']:24s} {'identical' if r['identical'] else 'ratio ' + str(r['ratio'])} | stop {r['stop_reference']} "
              f"agent {r['step_constants_agent']} {('ok by ' + r['stop_by']) if r['stop_matches'] else 'DIFFERS'} | only reference "
              f"{r['numbers_only_in_reference'][:8]} | only agent {r['numbers_only_in_agent'][:8]}")


if __name__ == "__main__":
    main()
