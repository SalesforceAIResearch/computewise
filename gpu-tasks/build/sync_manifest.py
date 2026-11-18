"""A sync manifest for one or more tasks: every file the tasks name, path -> sha256.

usage: python3 -I build/sync_manifest.py <out.tsv> <task_id> [<task_id> ...] [--also <path> ...]

Run from gpu-tasks/. It lists, for each task:
- the task file and its expected files (registry/expected/<task>.*json);
- its host terms (survey/host-terms/<task>.*.json);
- every registry-relative "artifacts/..." path named anywhere in the task file: files[].from, scripted_reference.from (the
  W0's controller), environment.install.constraints, prep and reference fields;
- every record its jobs and unneeded_jobs draw, and each record's agent_view.script;
then each `--also` path (a builder, a shared build file).

Why it exists. Hand-built manifests listed files[] only. Twice that dropped a file the task names elsewhere, and the gap
reached a machine:
- pkg-k2-latepole's 38 `artifacts/k2-latepole-inputs/` files (sync 10 omitted them; found when stage_task failed on the
  E2, 2026-11-02 23:16 MST);
- its `scripted_reference.from`, `artifacts/k2-latepole/reference.py` (no sync named it; W0-k2lp-reference was refused on
  c3-8 for want of it, isolation 2026-11-03 13:43Z).
A directory entry (a path ending in "/") is staged from elsewhere (git, hub) and is not listed. A named path that does
not exist stops the script.
"""
import glob
import hashlib
import json
import os
import sys


def named_artifacts(x, out):
    if isinstance(x, dict):
        for v in x.values():
            named_artifacts(v, out)
    elif isinstance(x, list):
        for v in x:
            named_artifacts(v, out)
    elif isinstance(x, str) and x.startswith("artifacts/") and not x.endswith("/"):
        out.add(x)


def task_paths(task_id):
    tp = f"registry/tasks/{task_id}.json"
    t = json.load(open(tp))
    paths = {tp}
    paths |= set(glob.glob(f"registry/expected/{task_id}.json")) | set(glob.glob(f"registry/expected/{task_id}.*.json"))
    paths |= set(glob.glob(f"survey/host-terms/{task_id}.*.json"))
    arts = set()
    named_artifacts(t, arts)
    paths |= {"registry/" + a for a in arts}
    for j in (t.get("jobs") or []) + (t.get("unneeded_jobs") or []):
        for rid in j.get("records") or []:
            rp = f"registry/records/{rid}.json"
            paths.add(rp)
            s = (json.load(open(rp)).get("agent_view") or {}).get("script")
            if s:
                paths.add("registry/" + s)
    return paths


def main():
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    out, rest = args[0], args[1:]
    also = []
    if "--also" in rest:
        i = rest.index("--also")
        rest, also = rest[:i], rest[i + 1:]
    paths = set(also)
    for t in rest:
        paths |= task_paths(t)
    missing = sorted(p for p in paths if not os.path.isfile(p))
    if missing:
        raise SystemExit(f"{len(missing)} named paths do not exist, e.g. {missing[:3]}")
    with open(out, "w") as fh:
        for p in sorted(paths):
            fh.write(p + "\t" + hashlib.sha256(open(p, "rb").read()).hexdigest() + "\n")
    print(f"{out}: {len(paths)} files, sha256 {hashlib.sha256(open(out, 'rb').read()).hexdigest()[:12]}")


if __name__ == "__main__":
    main()
