#!/usr/bin/env python3
"""Fill a Hugging Face cache from a staging manifest so the task's scripts load their models and datasets offline.

usage: <the task's python> -I build/stage_hub.py <MANIFEST.json> <HF_HOME>

Run it with the task environment's own interpreter (MANIFEST.json, environment.install), on a host with Hub access. It writes the
cache the box then mounts, with HF_HOME pointing at it (which parts may be read-only: see below). Per Hub entry:
  1. snapshot_download at the pinned revision, only the listed files;
  2. write hub/<kind>s--<repo>/refs/main = revision: offline from_pretrained("gpt2") resolves the bare name through refs/main, and a
     revision-pinned download does not write it (verified 2026-10-19: without it the offline tokenizer load fails);
  3. for datasets, one load_dataset(repo, revision=...) online: offline, `datasets` reads its own processed cache under
     HF_HOME/datasets, not the Hub snapshot (verified 2026-10-19: without it the offline load_dataset fails). That cache is written
     in the format of this interpreter's `datasets`, which is why the task's interpreter must run this script.
  4. after all entries, write hub/version.txt = "1": transformers writes that marker the first time it opens a cache, and on a
     read-only hub/ it cannot, so every process then prints "You are offline and the cache for model files ... has been updated"
     and "There was a problem when trying to write in your cache folder", which the agent reads in its job logs (found by
     the sandbox in the box, 2026-10-20; checked in transformers 4.36.2 and 4.44.2, utils/hub.py).
Then a child process with HF_HUB_OFFLINE, HF_DATASETS_OFFLINE and TRANSFORMERS_OFFLINE set re-opens every entry the way the
scripts name it (bare repo id, no revision) and reports what loaded. Exit status 1 if any entry fails offline.

Mounting the result in the box. HF_HOME ends up with three directories: hub/ (models and dataset snapshots), modules/ and
datasets/ (the processed datasets cache). hub/ and modules/ may be read-only. datasets/ may not: load_dataset takes lock files
there even offline, and map() writes its cache files there, so a wholly read-only HF_HOME fails with PermissionError before
training starts (found 2026-10-19). Give each task run a writable copy of datasets/, with HF_HOME pointing at the whole tree.
Moving the tree to another path keeps working. Checked on 2026-10-19 with the smoke task's load path, datasets 2.20.0.
"""
import json
import os
import subprocess
import sys

CHILD = r'''
import json, sys
from huggingface_hub import hf_hub_download
from datasets import load_dataset
ok = True
for e in json.loads(sys.argv[1]):
    try:
        if e["kind"] == "model":
            for f in e["files"]:
                hf_hub_download(e["repo"], f)
            print("OFFLINE OK  model  ", e["repo"], len(e["files"]), "files resolve by bare name")
        else:
            ds = load_dataset(e["repo"])
            print("OFFLINE OK  dataset", e["repo"], {k: len(v) for k, v in ds.items()})
    except Exception as x:
        ok = False
        print("OFFLINE FAIL", e["kind"], e["repo"], type(x).__name__, str(x)[:300])
sys.exit(0 if ok else 1)
'''


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    manifest = json.load(open(sys.argv[1]))
    hf_home = os.path.abspath(sys.argv[2])
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_CACHE", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE"):
        os.environ.pop(k, None)
    from huggingface_hub import snapshot_download
    hub = (manifest.get("environment") or {}).get("hub") or []
    if not hub:
        raise SystemExit("the manifest names no Hub repos")
    for e in hub:
        files = list(e["files"])
        snapshot_download(e["repo"], revision=e["revision"], repo_type=e["kind"], allow_patterns=files if e["kind"] == "model" else None)
        refs = os.path.join(hf_home, "hub", f"{e['kind']}s--" + e["repo"].replace("/", "--"), "refs")
        os.makedirs(refs, exist_ok=True)
        with open(os.path.join(refs, "main"), "w") as fh:
            fh.write(e["revision"])
        if e["kind"] == "dataset":
            from datasets import load_dataset
            load_dataset(e["repo"], revision=e["revision"])
        print("staged", e["kind"], e["repo"], e["revision"][:12])
    with open(os.path.join(hf_home, "hub", "version.txt"), "w") as fh:
        fh.write("1")
    env = dict(os.environ, HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, json.dumps(hub)], env=env, cwd=hf_home)
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
