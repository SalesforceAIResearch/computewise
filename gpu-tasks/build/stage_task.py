#!/usr/bin/env python3
"""Stage a registry task's working tree exactly as the agent should see it, plus a manifest for whoever builds the box.

usage: python3 -I build/stage_task.py <task_id> <dest> [<manifest_dir>]

<dest> must be absent or empty. It receives the task's files[] and nothing else: literals from the task file, scripts from
registry/artifacts (each checked against the entry hash of the records that bind it), and any source checkout a files[] entry
names under `git` (downloaded from GitHub at the pinned commit; only the directory named by `keep` is copied, so no .git,
benchmark/, tests/, docs/ or other examples arrive), and any Hub files a files[] entry names under `hub_files` (downloaded from
the Hugging Face Hub at the pinned revision and each checked against its sha256; a shared cache directory, GPU_TASKS_HUB_CACHE,
keeps one copy across tasks, and the tree gets hard links when it is on the same filesystem), and any files a files[] entry
names under `generated` (made by the named build/ script from its Hub inputs, each output checked against its sha256: a
data-preparation package's data/raw/, by build/gen_raw_text.py). No manifest, record, duration or
registry path enters <dest>.

<manifest_dir> (default <dest>.staging/) receives MANIFEST.json and constraints.txt: Python version, install line, the Hub repos
with revisions and files, environment variables, every job's launch line, the brief and the horizon. It is for the box builder
and must not be copied into the box. build/stage_hub.py reads it to fill a Hugging Face cache.
Standard library only; needs network to GitHub for `git` entries and to the Hugging Face Hub for `hub_files` entries.
"""
import datetime
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
REG = os.path.join(ROOT, "registry")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def fetch_github(repo, commit, keep, dest):
    """Copy directory `keep` of github.com/<repo> at <commit> to <dest>. Returns the number of files copied."""
    url = f"https://codeload.github.com/{repo}/tar.gz/{commit}"
    data = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "gpu-tasks-stage"}), timeout=300).read()
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            t.extractall(tmp, filter="data")
        tops = os.listdir(tmp)
        if len(tops) != 1:
            raise SystemExit(f"unexpected archive layout for {repo}@{commit}: {tops}")
        src = os.path.join(tmp, tops[0], keep)
        if not os.path.isdir(src):
            raise SystemExit(f"{repo}@{commit} has no directory {keep!r}")
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return sum(len(fs) for _, _, fs in os.walk(dest))


def fetch_hub_files(spec, dest):
    """Copy the files a `hub_files` entry names into <dest>, each at the pinned revision and checked against its sha256.
    spec: {repo, kind, revision, files: {name: {bytes, sha256}}}. Returns the bytes placed."""
    cache = os.environ.get("GPU_TASKS_HUB_CACHE") or os.path.join(tempfile.gettempdir(), "gpu-tasks-hub-cache")
    cdir = os.path.join(cache, spec["kind"] + "s", spec["repo"].replace("/", "--"), spec["revision"])
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(dest, exist_ok=True)
    total = 0
    for name, want in sorted(spec["files"].items()):
        c = os.path.join(cdir, name)
        if not (os.path.exists(c) and os.path.getsize(c) == want["bytes"]):
            pre = "datasets/" if spec["kind"] == "dataset" else ""
            url = f"https://huggingface.co/{pre}{spec['repo']}/resolve/{spec['revision']}/{name}"
            tmp = c + ".part"
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
                    fh.write(b)
            if h.hexdigest() != want["sha256"]:
                os.remove(tmp)
                raise SystemExit(f"{spec['repo']}@{spec['revision'][:12]} {name}: sha256 {h.hexdigest()[:16]} is not the recorded {want['sha256'][:16]}")
            os.replace(tmp, c)
        out = os.path.join(dest, name)
        try:
            os.link(c, out)
        except OSError:
            shutil.copyfile(c, out)
        total += want["bytes"]
    return total


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit(__doc__)
    task_id, dest = sys.argv[1], os.path.abspath(sys.argv[2])
    mdir = os.path.abspath(sys.argv[3]) if len(sys.argv) == 4 else dest.rstrip("/") + ".staging"
    tpath = os.path.join(REG, "tasks", task_id + ".json")
    if not os.path.exists(tpath):
        raise SystemExit(f"no task {task_id} in {REG}/tasks")
    task = json.load(open(tpath))
    if os.path.exists(dest) and os.listdir(dest):
        raise SystemExit(f"{dest} is not empty")
    if os.path.commonpath([dest, mdir]) == dest:
        raise SystemExit("the manifest directory must be outside the task tree")
    os.makedirs(dest, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)

    hashes = {}
    for j in task["jobs"]:
        for rid in (j.get("records") or []) + (j.get("candidate_pool") or []):
            r = json.load(open(os.path.join(REG, "records", rid + ".json")))
            s = r["agent_view"].get("script")
            if s:
                hashes.setdefault(s, set()).add(r["identity"]["entry_hash"])

    written = []
    for f in task["files"]:
        out = os.path.join(dest, f["path"])
        if os.path.commonpath([dest, os.path.abspath(out)]) != dest:
            raise SystemExit(f"files[] path escapes the task tree: {f['path']}")
        if "content" in f:
            os.makedirs(os.path.dirname(out) or dest, exist_ok=True)
            open(out, "w").write(f["content"])
            b = f["content"].encode()
            written.append({"path": f["path"], "bytes": len(b), "sha256": sha256_bytes(b), "from": "literal in the task file"})
        elif f.get("git"):
            g = f["git"]
            n = fetch_github(g["repo"], g["commit"], g["keep"], out.rstrip("/"))
            written.append({"path": f["path"], "files": n, "from": f"github.com/{g['repo']}@{g['commit']}, directory {g['keep']}/ only"})
        elif f.get("hub_files"):
            n = fetch_hub_files(f["hub_files"], out.rstrip("/"))
            hf = f["hub_files"]
            written.append({"path": f["path"], "files": len(hf["files"]), "bytes": n,
                            "from": f"huggingface.co/{hf['kind']}s/{hf['repo']}@{hf['revision']}, each file sha256-checked"})
        elif f.get("generated"):
            gsp = f["generated"]
            # the generators: a data-preparation package's raw text (gen_raw_text.py) and H4's document pool (gen_h4_pool.py,
            # cpu-tasks 2026-10-23); both take <shards dir> <vocab.json> <out dir> [args], standard library only
            gens = {"build/gen_raw_text.py": "gen_raw_text.py", "build/gen_h4_pool.py": "gen_h4_pool.py"}
            if gsp.get("by") not in gens:
                raise SystemExit(f"files[] entry {f['path']}: no generator {gsp.get('by')!r}")
            with tempfile.TemporaryDirectory(dir=os.path.dirname(dest)) as tmp:
                fetch_hub_files(gsp["inputs"]["shards"], os.path.join(tmp, "shards"))
                fetch_hub_files(gsp["inputs"]["vocab"], os.path.join(tmp, "vocab"))
                subprocess.run([sys.executable, "-I", os.path.join(HERE, gens[gsp["by"]]), os.path.join(tmp, "shards"),
                                os.path.join(tmp, "vocab", "vocab.json"), out.rstrip("/")] + list(gsp.get("args") or []),
                               check=True, stdout=subprocess.DEVNULL)
            got = sorted(os.listdir(out))
            if got != sorted(gsp["outputs"]):
                raise SystemExit(f"{f['path']}: generated {got}, not the recorded {sorted(gsp['outputs'])}")
            n = 0
            for name, want in gsp["outputs"].items():
                h = hashlib.sha256()
                with open(os.path.join(out, name), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""):
                        h.update(chunk)
                if h.hexdigest() != want["sha256"]:
                    raise SystemExit(f"{f['path']}{name}: sha256 {h.hexdigest()[:16]} is not the recorded {want['sha256'][:16]}")
                n += want["bytes"]
            written.append({"path": f["path"], "files": len(gsp["outputs"]), "bytes": n,
                            "from": f"generated by {gsp['by']} from its inputs, each output sha256-checked"})
        elif str(f.get("from", "")).startswith("artifacts/"):
            src = os.path.join(REG, f["from"])
            b = open(src, "rb").read()
            h = sha256_bytes(b)
            want = hashes.get(f["from"])
            if want and h not in want:
                raise SystemExit(f"{f['from']} does not match the entry hash of the records that bind it")
            if f.get("sha256") and h != f["sha256"]:
                # the task file names the artifact's bytes (build/gen_h1.py, 2026-10-23): a registry whose artifact moved
                # without its task file is refused rather than staged
                raise SystemExit(f"{f['from']}: sha256 {h[:16]} is not the task file's {f['sha256'][:16]}")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, "wb").write(b)
            written.append({"path": f["path"], "bytes": len(b), "sha256": h, "from": f["from"],
                            "entry_hash_checked": bool(want)})
        else:
            raise SystemExit(f"files[] entry {f['path']} cannot be staged: {f.get('from')!r}. The task is not self-contained.")

    env = task.get("environment") or {}
    cons = (env.get("install") or {}).get("constraints")
    if cons:
        shutil.copyfile(os.path.join(REG, cons), os.path.join(mdir, "constraints.txt"))
    manifest = {
        "schema": "p006-gpu-staging/0.1",
        "do_not_copy_into_box": True,
        "task_id": task_id,
        "staged_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "task_root": dest,
        "files": written,
        "brief": task["brief"]["text"],
        "horizon_s": task.get("horizon_s") or task.get("deadline_s"),
        "horizon_how": "the run's hard stop and the end of the scoring window, in recorded seconds; never shown to the agent",
        "launch_from": "the task root (every launch line is relative to it)",
        "jobs": [{"job_id": j["job_id"], "launch": j["launch"], "gpu_options": j["gpu_options"]} for j in task["jobs"]],
        "environment": env,
    }
    json.dump(manifest, open(os.path.join(mdir, "MANIFEST.json"), "w"), indent=1)
    print(f"staged {task_id}: {len(written)} files[] entries into {dest}")
    print(f"manifest: {os.path.join(mdir, 'MANIFEST.json')}")


if __name__ == "__main__":
    main()
