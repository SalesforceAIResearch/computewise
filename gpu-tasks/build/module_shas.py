"""materialisation.serving.module_shas on every record: the staged project files a launch can load, by task-root-relative path and
sha256.

usage: python3 -I build/module_shas.py <registry> [--check]

gpu-mock's helper-dependency bind (after r8.6) hashes each listed file on disk when replay engages, on every serving path: a file
missing or changed sends the job to its own code (identity_module_dep). The contract, gpu-mock's:
- the field is materialisation.serving.module_shas, for every serving mode; keys are task-root-relative paths, values sha256 of
  the staged bytes;
- {} means the bind is on and the entry imports no staged project file (gpu-mock still checks that no staged file shadows a
  stdlib or installed module); an absent field means not yet derived, and nothing is checked;
- a staged .py that is loaded but not in the map is ignored, so the map must hold everything the job can load: for a staged
  library tree (TRL's trl/), every .py under it, since its __init__ imports lazily;
- every record kind gets its entry's map (prefix, prepared, segment), since they share scripts.

What the map holds, per record, from the tasks that serve it:
- the staged .py files the entry imports, transitively, resolved by `ast` from the importing file's directory and the task root;
- and every .py under any staged `git` tree that one of those imports reaches.
A record whose builder already writes the field (K2's late pole, gen_k2_latepole.py) is left alone. A record served by several
tasks must get the same map from each; a difference stops the run. A record no task serves gets {} only when its entry script
imports nothing that could be staged, else it is left without the field and listed.

The `git` trees are fetched at their pinned commits (build/stage_task.py fetch_github) into a cache, GPU_TASKS_GIT_CACHE or
<tmp>/gpu-tasks-git-cache, so a rebuild hashes the bytes the box stages.
"""
import ast
import hashlib
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402


def sha256_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def git_tree(spec):
    """{relative path: absolute path} of every file in a `git` files[] entry, fetched once into the cache."""
    import stage_task as ST
    cache = os.environ.get("GPU_TASKS_GIT_CACHE") or os.path.join(tempfile.gettempdir(), "gpu-tasks-git-cache")
    dest = os.path.join(cache, spec["repo"].replace("/", "--"), spec["commit"], spec["keep"].strip("/"))
    if not os.path.isdir(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        tmp = dest + ".part"
        if os.path.isdir(tmp):
            import shutil
            shutil.rmtree(tmp)
        ST.fetch_github(spec["repo"], spec["commit"], spec["keep"], tmp)
        os.replace(tmp, dest)
    out = {}
    for d, _, fs in os.walk(dest):
        for f in fs:
            p = os.path.join(d, f)
            out[os.path.relpath(p, dest)] = p
    return out


def staged_py(task, reg):
    """{task-root-relative path: (absolute source path, git tree root or None)} of every .py the task stages, and the git roots."""
    out, roots = {}, []
    for f in task.get("files") or []:
        p = f["path"]
        if "git" in f:
            root = p.rstrip("/")
            roots.append(root)
            for rel, ab in git_tree(f["git"]).items():
                if rel.endswith(".py"):
                    out[f"{root}/{rel}"] = (ab, root)
        elif p.endswith(".py") and str(f.get("from", "")).startswith("artifacts/"):
            out[p] = (os.path.join(reg, f["from"]), None)
        elif p.endswith(".py") and "content" in f:
            out[p] = (None, None)
    return out, roots


def _source(task, path, staged):
    ab, _ = staged[path]
    if ab is not None:
        return open(ab, encoding="utf-8", errors="replace").read()
    return next(f["content"] for f in task["files"] if f["path"] == path)


def _imports(src):
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(a.name, 0) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.module or "", n.level))
            out += [((n.module + "." if n.module else "") + a.name, n.level) for a in n.names]
    return out


def _resolve(name, level, here, staged):
    """The staged file(s) `import name` (at relative `level`) from the file at `here` can load: the module file or package
    __init__ for the name and each of its parents, looked up from here's directory, then from the task root."""
    if level:
        base = os.path.dirname(here)
        for _ in range(level - 1):
            base = os.path.dirname(base)
        bases = [base]
    else:
        bases = [os.path.dirname(here), ""]
    parts = [x for x in name.split(".") if x]
    hits = []
    for b in bases:
        found = False
        for k in range(1, len(parts) + 1):
            stem = os.path.join(b, *parts[:k])
            for c in (stem + ".py", os.path.join(stem, "__init__.py")):
                c = os.path.normpath(c).lstrip("./") if c.startswith("./") else os.path.normpath(c)
                if c in staged:
                    hits.append(c)
                    found = True
        if found:
            break
    return hits


def entry_map(task, entry, staged, roots):
    """{path: sha256} for one entry script of a task, or None when the entry is not staged as source the build can read."""
    if entry not in staged:
        return None
    seen, todo, trees = set(), [entry], set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        src = _source(task, p, staged)
        for name, level in _imports(src) or []:
            for hit in _resolve(name, level, p, staged):
                if staged[hit][1]:
                    trees.add(staged[hit][1])
                todo.append(hit)
    files = {p for p in seen if p != entry}
    for root in trees:
        files |= {p for p, (_, r) in staged.items() if r == root}
    out = {}
    for p in sorted(files):
        ab, _ = staged[p]
        out[p] = sha256_file(ab) if ab else C.sha256_text(_source(task, p, staged))
    return out


def job_entry(job):
    if job.get("script_path"):
        return job["script_path"]
    return next((w for w in (job.get("launch") or "").split() if w.endswith(".py")), None)


def derive(reg):
    """{record_id: map} from every installed task, and a list of conflicts."""
    maps, conflicts = {}, []
    for tf in sorted(os.listdir(os.path.join(reg, "tasks"))):
        if not tf.endswith(".json"):
            continue
        task = json.load(open(os.path.join(reg, "tasks", tf)))
        jobs = (task.get("jobs") or []) + (task.get("unneeded_jobs") or [])
        if not jobs:
            continue
        staged, roots = staged_py(task, reg)
        for j in jobs:
            e = job_entry(j)
            m = entry_map(task, e, staged, roots) if e else None
            if m is None:
                continue
            for rid in j.get("records") or []:
                if rid in maps and maps[rid] != m:
                    conflicts.append((rid, task["task_id"], sorted(set(maps[rid]) ^ set(m))[:4]))
                maps.setdefault(rid, m)
    return maps, conflicts


def unserved_entry_imports_nothing(rec, reg):
    """True when a record no task serves has an entry script that imports nothing a task could stage beside it: every import
    resolves to the stdlib or an installed name, judged by the script's own text (no relative import, no sys.path change)."""
    p = rec.get("agent_view", {}).get("script")
    if not p or not os.path.exists(os.path.join(reg, p)):
        return False
    src = open(os.path.join(reg, p), encoding="utf-8", errors="replace").read()
    im = _imports(src)
    if im is None or any(level for _, level in im):
        return False
    return not any(s in src for s in ("sys.path.insert", "sys.path.append", "importlib.import_module", "__import__(",
                                      "spec_from_file_location", "runpy."))


def main():
    reg = sys.argv[1]
    check = "--check" in sys.argv[2:]
    maps, conflicts = derive(reg)
    if conflicts:
        for c in conflicts[:20]:
            print("CONFLICT", *c)
        raise SystemExit(f"{len(conflicts)} records get different maps from different tasks")
    n = {"written": 0, "unchanged": 0, "kept": 0, "unserved_empty": 0, "unserved_left": 0}
    left, bad = [], []
    for f in sorted(os.listdir(os.path.join(reg, "records"))):
        if not f.endswith(".json") or f.startswith("example-order"):
            continue
        path = os.path.join(reg, "records", f)
        r = json.load(open(path))
        rid = r["identity"]["record_id"]
        sv = r.setdefault("materialisation", {}).setdefault("serving", {"mode": "per_op"})
        if rid.startswith("k2-latepole-"):
            n["kept"] += 1                       # K2's builder writes and checks its own (gen_k2_latepole.check_record)
            continue
        want = maps.get(rid)
        if want is None:
            if unserved_entry_imports_nothing(r, reg):
                want = {}
                n["unserved_empty"] += 1
            else:
                n["unserved_left"] += 1
                left.append(rid)
                continue
        if sv.get("module_shas") == want:
            n["unchanged"] += 1
            continue
        if check:
            bad.append(rid)
            continue
        sv["module_shas"] = want
        C.write_json(path, r)
        n["written"] += 1
    print("module_shas:", ", ".join(f"{k} {v}" for k, v in n.items()))
    if left:
        print("left without the field (no task serves them and their entry may import a staged file):", ", ".join(left[:10]),
              "..." if len(left) > 10 else "")
    if check and bad:
        raise SystemExit(f"{len(bad)} records' module_shas differ from the derived map, e.g. {bad[:3]}")


if __name__ == "__main__":
    main()
