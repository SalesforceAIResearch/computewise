"""pkg-obs-latepole's brief-only reference (lane OBS, 2026-11-02; analysis 15:11 MST, condition 3).

It reads only what the agent's staged tree shows: under latepole/<setup>/steps-<NNNN>/, each folder's training script and the
launch lines its README lists. For each run it takes the steps the script itself says it trains (its `stop_after_step = N` or
`STOP_STEP` default where the run stops early, else its schedule's `train_steps` or `FINAL_TRAIN_STEPS`), then launches the
runs with the most stated steps first on the GPUs as they free up. No registry, recorded seconds, losses or table we wrote is
read. Ties keep the tree's own order (sorted paths).

usage in the box:   python reference/latepole_reference.py --root . --gpus 8
check the reading:  python latepole_reference.py --root <staged tree> --gpus 8 --dry-run   (prints the order; launches nothing)
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

STOP = re.compile(r"^\s*stop_after_step\s*=\s*(\d+)\b", re.M)
ENV_STOP = re.compile(r'^\s*STOP_STEP\s*=\s*int\(os\.environ\.get\("STOP_STEP",\s*(\w+)\)\)', re.M)
TRAIN = re.compile(r"^\s*(train_steps|FINAL_TRAIN_STEPS)\s*=\s*(\d+)\b", re.M)
LAUNCH = re.compile(r"`(python [^`]+)`")


def stated_steps(code):
    """The steps a script says it trains: its stop setting where it has one, else its schedule's length."""
    m = STOP.search(code)
    if m:
        return int(m.group(1)), "stop_after_step"
    sched = {name: int(v) for name, v in TRAIN.findall(code)}
    e = ENV_STOP.search(code)
    if e:
        v = e.group(1)
        if v.isdigit():
            return int(v), "STOP_STEP default"
        if v in sched:
            return sched[v], f"STOP_STEP default = {v}"
    for name in ("train_steps", "FINAL_TRAIN_STEPS"):
        if name in sched:
            return sched[name], name
    raise ValueError("no stated step count")


def runs(root):
    """Every launch line the READMEs under latepole/<setup>/steps-<NNNN>/ list, with the steps its script states."""
    out = []
    for readme in sorted(Path(root, "latepole").glob("*/steps-*/README.md")):
        lines = LAUNCH.findall(readme.read_text())
        if not lines:
            raise ValueError(f"{readme}: no launch line")
        for n, line in enumerate(lines, 1):
            argv = shlex.split(line)
            steps, how = stated_steps(Path(root, argv[1]).read_text())
            out.append({"run": f"{readme.parent.relative_to(root)}#{n}", "argv": argv, "stated_steps": steps, "from": how})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rs = runs(a.root)
    order = sorted(range(len(rs)), key=lambda i: (-rs[i]["stated_steps"], i))
    if a.dry_run:
        print(json.dumps({"runs": len(rs), "order": [rs[i]["run"] for i in order],
                          "stated_steps": {rs[i]["run"]: [rs[i]["stated_steps"], rs[i]["from"]] for i in order}}, indent=1))
        return
    os.chdir(a.root)
    free, active, logs = list(range(a.gpus)), {}, Path("reference-logs")
    logs.mkdir(exist_ok=True)
    queue = list(order)
    while queue or active:
        while queue and free:
            i, g = queue.pop(0), free.pop(0)
            log = open(logs / (rs[i]["run"].replace("/", "--") + ".log"), "w")
            p = subprocess.Popen(rs[i]["argv"], stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1"))
            active[p.pid] = (p, g, i, log)
        pid, status = os.wait()
        if pid in active:
            p, g, i, log = active.pop(pid)
            log.close()
            free.append(g)
            if status != 0:
                sys.exit(f"{rs[i]['run']} exited {status}")
    print(json.dumps({"completed": len(rs)}))


if __name__ == "__main__":
    main()
