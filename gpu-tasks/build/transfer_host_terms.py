"""Host terms for one task, carried over from another task's measuring run on the same machine type.

usage: python3 -I build/transfer_host_terms.py <source host-terms file> <task_id> [<task_id> ...] [--out-dir DIR]
       python3 -I build/transfer_host_terms.py --from <file> [--from <file> ...] <task_id> [...] [--out-dir DIR]

With several sources, a script is taken from the first source that has it (the segment packages: a segment package's own
measurement first, then pkg-p20-par's for its standalone prefix jobs).

Writes <out-dir>/<task_id>.<machine type>.json (default out-dir: survey/host-terms/) for each task, in the schema
build/derive_host_terms.py writes, with `transferred_from` naming the source file and run.

Why it is sound. Every modded-nanogpt package draws its scripts from the nine
script groups pkg-p20-par runs, on one image and one machine type. A script's host terms are properties of the script, the image
and the machine, not of the package:
- hello, pre and post: a prefix script and its full script differ only in a stop line that runs after training starts, so
  process start, imports, shard load, model build and exit are the same code;
- in-run host time is per event (per_event: per validation, per shard load, per step, once), so a full run of 3,040 steps gets
  its own count of each, not a prefix's.
What does not carry over is contention between start-ups launched together, which differs by package; the DPO repeats on the
c3 bound its effect at a few seconds per cold start. The long packages are checked by a speed-10 scripted run each before
they are declared ready (backlog W0-<pkg>-check10).

Matching. A task's scripts are keyed as derive_host_terms.py keys them: by path in the task tree where two of the task's
scripts share a basename, else by basename. Each is looked up in the source by its path, then by its path with a `-full`
or `-long` directory read as its prefix variant's (contra-soft-muon-full/ is contra-soft-muon's script run through its whole
schedule, contra-soft-muon-long/ the same script stopped at a later validation), or a `-<step>` directory read the same way
(a stagger package's contra-soft-muon-1500/),
then by its basename when the source has exactly one script of that name. A script with no match stops the transfer, and so
does a match between a segment script (it resumes: checkpoint load and a data-loader skip in its start-up) and one that is not,
told apart by the scripts the source and target task files stage.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def task_scripts(task):
    """The task's script paths, from its launch lines (the first token ending in .py). A per_example task (MLS-Bench KV) launches
    `bash methods/<m>/scripts/<wl>.sh`, which runs the job's script_path; its terms are keyed by that script's basename, as
    build/derive_host_terms.py keys a per_example run's terms, so the job's script_path is used (2026-10-23, mls-kv-report)."""
    out = []
    for j in task["jobs"]:
        p = next((t for t in (j.get("launch") or "").split() if t.endswith(".py")), None)
        if p is None and j.get("n_examples_full") and j.get("script_path"):
            p = j["script_path"]
        if p and os.path.normpath(p) not in out:
            out.append(os.path.normpath(p))
    return out


def keys_for(paths):
    """path -> the key derive_host_terms.py would give it in this task."""
    by_base = collections.defaultdict(set)
    for p in paths:
        by_base[os.path.basename(p)].add(p)
    return {p: (p if len(by_base[os.path.basename(p)]) > 1 else os.path.basename(p)) for p in paths}


def staged_script(task, path, reg):
    """The text of the script a task stages at `path` (its files[] entry), or None."""
    f = next((f for f in task.get("files") or [] if os.path.normpath(f["path"]) == os.path.normpath(path)), None)
    if not f or not str(f.get("from", "")).startswith("artifacts/"):
        return None
    return open(os.path.join(reg, f["from"])).read()


def is_segment(code):
    return code is not None and "--resume" in code and "--stop-at" in code


def find(src_terms, path):
    """The source's terms for a script at `path` in the target task, the source's key for it, and how it was found."""
    if path in src_terms:
        return src_terms[path], path, f"by path {path}"
    d, base = os.path.split(path)
    for suffix, what in (("-full", "run through its whole schedule"), ("-long", "run for a longer prefix")):
        if d.endswith(suffix):
            alt = os.path.join(d[:-len(suffix)], base)
            if alt in src_terms:
                return src_terms[alt], alt, f"by path {alt} ({d}/ is that variant {what})"
    if os.sep in d:                          # <dir>/<setup>/<script> (a goal-level task's candidates/): the setup's own path
        alt = os.path.join(os.path.basename(d), base)
        if alt in src_terms:
            return src_terms[alt], alt, f"by path {alt} ({d}/ is that setup's folder)"
    m = re.fullmatch(r"(.+)-(\d+)", d)
    if m:                                    # <variant>-<stop step>/: a stagger package's prefix of that variant
        alt = os.path.join(m.group(1), base)
        if alt in src_terms:
            return src_terms[alt], alt, f"by path {alt} ({d}/ is that variant stopped at step {m.group(2)})"
        if base in src_terms:
            return src_terms[base], base, f"by basename {base} ({d}/ is {m.group(1)} stopped at step {m.group(2)})"
    m = re.fullmatch(r"(?:setups|runs|quick)/([^/]+)/(?:(?:seed\d+|run\d+)/)?train_gpt\.py", path)
    if m:
        # a prepared script at <root>/<setup>/[<seed or run dir>/]train_gpt.py (pkg-h1-study's setups/<setup>/train_gpt.py
        # since 2026-10-23 21:30Z, the prep packages' runs/): its start-up is its setup's script's, whatever the seed
        # directory; the source's own seed directory for that setup, else its per-setup path
        for k in sorted(src_terms):
            if re.fullmatch(rf"(?:runs|quick)/{re.escape(m.group(1))}/(?:seed\d+/)?train_gpt\.py", k):
                return src_terms[k], k, f"by setup {m.group(1)}: {k} (a prepared script of the same setup, another seed's directory)"
    if base in src_terms:
        return src_terms[base], base, f"by basename {base}"
    hits = [k for k in src_terms if os.path.basename(k) == base]
    if len(hits) == 1:
        return src_terms[hits[0]], hits[0], f"by path {hits[0]}, the source's one script named {base}"
    return None, None, None


def source_path(stask, key):
    """The path in the source task of the script its terms file keys as `key` (a path, or a basename unique in that task)."""
    paths = task_scripts(stask)
    if "/" in key:
        return key if key in paths else None
    hits = [q for q in paths if os.path.basename(q) == key]
    return hits[0] if len(hits) == 1 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("args", nargs="+", metavar="source-or-task")
    ap.add_argument("--from", dest="sources", action="append")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "survey", "host-terms"))
    ap.add_argument("--registry", default=os.path.join(ROOT, "registry"))
    a = ap.parse_args()
    sources = a.sources or [a.args[0]]
    tasks = a.args if a.sources else a.args[1:]
    srcs = [(f, json.load(open(f))) for f in sources]
    for f, src in srcs:
        if float(src.get("speed", 0)) != 1.0:
            raise SystemExit(f"{f}: host terms come from speed-1 runs only")
    if len({src["machine_type"] for _, src in srcs}) != 1:
        raise SystemExit("the sources were measured on different machine types")
    src = srcs[0][1]
    mt = src["machine_type"]
    os.makedirs(a.out_dir, exist_ok=True)
    for tid in tasks:
        task = json.load(open(os.path.join(a.registry, "tasks", tid + ".json")))
        paths = task_scripts(task)
        keys = keys_for(paths)
        terms, how = {}, {}
        # a script the agent writes (prep.produces) is looked up as its reference, the setup's script in its own directory
        # A script the agent writes (prep.produces) is looked up as the same setup's prepared script in a preparation source
        # (runs/<setup>/train_gpt.py: the scripted preparer ran the reference there), else as its reference in its own
        # directory (<setup>/<reference name>) in a control source.
        ref_of = {}
        for j in task["jobs"]:
            prep = j.get("prep") or {}
            for q in prep.get("produces") or []:
                if prep.get("reference"):
                    setup = q.split("/")[1]
                    ref_of[os.path.normpath(q)] = [os.path.normpath(q), f"runs/{setup}/train_gpt.py",
                                                   os.path.join(setup, os.path.basename(prep["reference"]))]
        for p in paths:
            want_seg = is_segment(staged_script(task, p, a.registry))
            t = h = None
            for f, s in srcs:
                for cand in ref_of.get(p, [p]):
                    t, key, h = find(s["terms"], cand)
                    if t is not None:
                        break
                if t is not None and p in ref_of:
                    h = f"{h} (for {p}, which the agent writes)"
                if t is None:
                    continue
                stask = json.load(open(os.path.join(a.registry, "tasks", s["task_id"] + ".json")))
                sp = source_path(stask, key)
                if sp is None and p in task_scripts(stask) and staged_script(stask, p, a.registry) is not None \
                        and staged_script(stask, p, a.registry) == staged_script(task, p, a.registry):
                    # the source keys several scripts by one basename (a per_example task's terms are per evaluator, and
                    # mls-kv-deploy stages one copy per method folder); the target stages the same bytes at the same path
                    sp = p
                    h = f"{h}; {p} is byte-identical at the same path in the source task"
                if sp is None or (staged_script(stask, sp, a.registry) is not None
                                  and is_segment(staged_script(stask, sp, a.registry)) != want_seg):
                    t = h = None
                    continue
                h = f"{h}, from {s['task_id']}"
                break
            if t is None:
                raise SystemExit(f"{tid}: no host terms for {p} ({'segment' if want_seg else 'non-segment'} script) in "
                                 + ", ".join(f for f, _ in srcs))
            terms[keys[p]], how[keys[p]] = t, h
        out = {k: src[k] for k in ("schema", "machine_type", "cpu_platform", "zone", "nproc", "speed", "caps", "image",
                                   "run_dir", "run_task_sha256", "measured_on") if k in src}
        out.update({
            "task_id": tid, "terms": terms,
            "source": (f"transferred from {', '.join(s['task_id'] for _, s in srcs)}'s measuring runs; per script, matched as "
                       "transferred_from.matched says"),
            "transferred_from": {"task_id": src["task_id"], "tasks": [s["task_id"] for _, s in srcs],
                                 "file": os.path.relpath(os.path.abspath(sources[0]), ROOT),
                                 "files": [os.path.relpath(os.path.abspath(f), ROOT) for f, _ in srcs],
                                 "run_dir": src.get("run_dir"), "run_dirs": [s.get("run_dir") for _, s in srcs], "matched": how,
                                 "how": "build/transfer_host_terms.py: a script's host terms are the script's, the image's and the "
                                        "machine's, not the package's; in-run host time is per event"},
            "derived_by": "build/transfer_host_terms.py (gpu-tasks), from build/derive_host_terms.py's output",
        })
        path = os.path.join(a.out_dir, f"{tid}.{mt}.json")
        with open(path, "w") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
        print(f"{tid}: {len(terms)} scripts -> {os.path.relpath(path, ROOT) if path.startswith(ROOT) else path}")


if __name__ == "__main__":
    main()
