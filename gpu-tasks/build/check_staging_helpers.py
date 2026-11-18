#!/usr/bin/env python3
"""Helpers for build/check_staging.sh, so the shell script carries no inline Python.

usage: python3 -I build/check_staging_helpers.py field <MANIFEST.json> python|packages|cpu_lines   (cpu_lines: the arguments
                                                                                                   after "pip install", per line)
       python3 -I build/check_staging_helpers.py env <MANIFEST.json> <task root> <HF_HOME>     KEY=VALUE per line
       python3 -I build/check_staging_helpers.py launches <MANIFEST.json>                      job_id TAB launch line
       python3 -I build/check_staging_helpers.py steps <prefix>                                one line per rank that trained
"""
import glob
import json
import sys


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "field":
        e = json.load(open(args[0]))["environment"]
        if args[1] == "python":
            print(e["python"]["value"])
        elif args[1] == "cpu_lines":
            for line in e["install"]["cpu_lines"]:
                if not line.startswith("pip install "):
                    raise SystemExit(f"not a pip install line: {line}")
                print(line[len("pip install "):])
        else:
            print(" ".join(e["install"]["packages"]))
    elif cmd == "env":
        for k, v in json.load(open(args[0]))["environment"]["env"].items():
            print(f"{k}={args[1] if k == 'PYTHONPATH' else args[2] if k == 'HF_HOME' else v}")
    elif cmd == "launches":
        for j in json.load(open(args[0]))["jobs"]:
            print(f"{j['job_id']}\t{j['launch'].split('  #')[0]}")
    elif cmd == "steps":
        rs = [json.load(open(f)) for f in sorted(glob.glob(args[0] + ".*.json"))]
        rs = [r for r in rs if r.get("train_entered") or r.get("patch_error")]
        if not rs:
            print("no rank entered train()")
            return
        out = []
        for r in rs:
            s = (f"rank {r['rank']} of {r['world_size']}: optimizer_steps {r['optimizer_steps']}, global_step {r['global_step']}, "
                 f"max_steps {r['max_steps']}, training_step_calls {r['training_step_calls']}")
            out.append(s + (f", ERROR {r['patch_error']}" if r["patch_error"] else ""))
        print("; ".join(out))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
