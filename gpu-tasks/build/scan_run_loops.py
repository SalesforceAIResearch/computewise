"""Which registry scripts can make more than one training run in one launch.

usage: python3 -I build/scan_run_loops.py <registry> [--json]

gpu-mock serves one recorded run per launch. A script that trains several runs in one process (newton-muon's `num_trials`, its
last argv) gets one record for the first and, today, the record's frozen final values for the rest. This reads every script
under artifacts/ with ast, runs none of them, and reports per script:
  run_loops   loops that enclose the training step and also rebuild what a run starts from (parameters re-initialised or an
              optimizer, model or trainer constructed inside the loop): each iteration is a fresh run
  count_from  where such a loop's count comes from: sys.argv, an argparse option (its flag), or a constant
  loop_args   argparse options or argv reads whose name says they set a number of runs, trials, seeds or repeats, used or
              not
It then lists the tasks whose jobs serve each exposed script, and the jobs' recorded argv, so a recorded run that itself used
the loop is visible (its record would then hold several runs' values in one series).
"""
import ast
import json
import os
import re
import sys

COUNT_NAME = re.compile(r"(trial|repeat|n_runs|num_runs|runs\b|n_seeds|num_seeds|seeds\b|replicate)", re.I)
STEP_CALL = ("backward", "train")             # loss.backward(), trainer.train()
FRESH_CALL = re.compile(r"^(AdamW|Adam|SGD|Muon|NorMuon|[A-Z]\w*Optim\w*|[A-Z]\w*Trainer|GPT|Model\w*|from_pretrained|"
                        r"\w*init\w*|normal_|zero_|reset_parameters|manual_seed)$")


def call_name(n):
    f = n.func
    return f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None


def calls(node):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call)]


def argv_source(expr, argp):
    """Where a loop's count comes from: 'sys.argv[...]', '--flag', a constant, or the expression's text."""
    s = ast.unparse(expr)
    if "sys.argv" in s:
        return "sys.argv: " + s
    for flag, dest in argp.items():
        if re.search(r"\b(args|opts|cfg|config|a)\." + re.escape(dest) + r"\b", s):
            return f"argparse {flag}: " + s
    return s


def scan(path):
    src = open(path, errors="replace").read()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"parse_error": str(e)}
    argp = {}
    for c in calls(tree):
        if call_name(c) == "add_argument" and c.args and isinstance(c.args[0], ast.Constant) and isinstance(c.args[0].value, str):
            flag = c.args[0].value
            dest = next((k.value.value for k in c.keywords if k.arg == "dest" and isinstance(k.value, ast.Constant)), None)
            argp[flag] = dest or flag.lstrip("-").replace("-", "_")
    # every assignment name -> its value, so `num_trials = int(sys.argv[-1])` resolves a `range(num_trials)`
    assigns = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    assigns.setdefault(t.id, n.value)
    loops = []

    def visit(node, chain):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue                      # loops inside definitions run only when called; the step loop is module level
            if isinstance(ch, (ast.For, ast.While)):
                visit(ch, chain + [ch])
                loops.append((ch, chain))
            else:
                visit(ch, chain)
    visit(tree, [])
    run_loops = []
    for L, outer in loops:
        cs = calls(L)
        steps = [c for c in cs if call_name(c) in STEP_CALL]
        inner_loops = [n for n in ast.walk(L) if isinstance(n, (ast.For, ast.While)) and n is not L]
        # the training step must sit in a loop nested inside L: L repeats whole step loops, not steps
        nested_step = any(any(call_name(c) in STEP_CALL for c in calls(i)) for i in inner_loops)
        fresh = sorted({call_name(c) for c in cs if call_name(c) and FRESH_CALL.match(call_name(c))
                        and not any(c in calls(i) for i in inner_loops if any(call_name(x) in STEP_CALL for x in calls(i)))})
        if steps and nested_step and fresh:
            it = L.iter if isinstance(L, ast.For) else L.test
            cnt = it.args[0] if isinstance(it, ast.Call) and call_name(it) == "range" and it.args else it
            if isinstance(cnt, ast.Name) and cnt.id in assigns:
                cnt = assigns[cnt.id]
            run_loops.append({"line": L.lineno, "header": ast.unparse(L.target if isinstance(L, ast.For) else L.test)
                              + (" in " + ast.unparse(L.iter) if isinstance(L, ast.For) else ""),
                              "count_from": argv_source(cnt, argp), "fresh_inside": fresh})
    loop_args = sorted(f for f in argp if COUNT_NAME.search(f))
    loop_args += [f"sys.argv read at line {n.lineno}: {ast.unparse(n)}" for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript) and ast.unparse(n.value) == "sys.argv"
                  and ast.unparse(n.slice) not in ("0",)]
    return {"run_loops": run_loops, "loop_args": loop_args}


def main():
    reg = sys.argv[1]
    art = os.path.join(reg, "artifacts")
    recs = {}
    for f in os.listdir(os.path.join(reg, "records")):
        if f.endswith(".json"):
            r = json.load(open(os.path.join(reg, "records", f)))
            if str(r.get("schema", "")).startswith("p006-example-order/"):
                continue                                 # an example-order table (gpu-mock r4), not a record
            recs[r["identity"]["record_id"]] = r
    by_script = {}
    for r in recs.values():
        s = (r.get("agent_view") or {}).get("script")
        if s:
            by_script.setdefault(s, []).append(r)
    tasks = {}
    for f in sorted(os.listdir(os.path.join(reg, "tasks"))):
        if f.endswith(".json"):
            t = json.load(open(os.path.join(reg, "tasks", f)))
            for j in t["jobs"]:
                for x in (j.get("records") or []):
                    s = (recs.get(x) or {}).get("agent_view", {}).get("script")
                    if s:
                        tasks.setdefault(s, set()).add((t["task_id"], j["job_id"]))
            for fl in t.get("files") or []:
                if isinstance(fl, dict) and fl.get("from"):
                    tasks.setdefault(fl["from"], set()).add((t["task_id"], "file " + fl["path"]))
    out = {"scripts": 0, "parse_errors": [], "exposed": []}
    for root, _, files in os.walk(art):
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            p = os.path.join(root, f)
            rel = os.path.relpath(p, reg)
            out["scripts"] += 1
            s = scan(p)
            if "parse_error" in s:
                out["parse_errors"].append({"script": rel, "error": s["parse_error"]})
                continue
            if s["run_loops"] or s["loop_args"]:
                rs = by_script.get(rel, [])
                out["exposed"].append({
                    "script": rel, **s,
                    "records": len(rs),
                    "recorded_argv": sorted({" ".join(r["identity"]["argv"]) for r in rs if not r["identity"].get("derived_from")})[:6],
                    "tasks": sorted(f"{a}:{b}" for a, b in tasks.get(rel, ()))})
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return
    print(f"{out['scripts']} scripts read, {len(out['parse_errors'])} do not parse, {len(out['exposed'])} with a run loop or a "
          "count-like argument")
    for e in out["exposed"]:
        print(f"\n{e['script']}  ({e['records']} records)")
        for L in e["run_loops"]:
            print(f"  RUN LOOP line {L['line']}: for {L['header']}  count from {L['count_from']}  fresh inside: {L['fresh_inside']}")
        for a in e["loop_args"]:
            print(f"  arg: {a}")
        if e["recorded_argv"]:
            print(f"  recorded argv: {e['recorded_argv']}")
        print(f"  tasks: {', '.join(e['tasks']) if e['tasks'] else 'none'}")
    for pe in out["parse_errors"]:
        print(f"\nparse error: {pe['script']}: {pe['error']}")


if __name__ == "__main__":
    main()
