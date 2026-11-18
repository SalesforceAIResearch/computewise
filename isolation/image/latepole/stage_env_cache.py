#!/usr/bin/env python3
"""Fill one Python environment's Hugging Face cache for a task with environment.python_environments (the late pole: two
interpreters in one image, isolation's image plan 2026-11-02), with that environment's own interpreter, so the processed
datasets cache is written by the `datasets` the jobs import.

usage (inside the task's image, network on, as root): /opt/envs/<e>/bin/python -I stage_env_cache.py <hub.json> <HF_HOME>
       (offline, as the box's user, over the mounted cache):  ... stage_env_cache.py --check <hub.json>

<hub.json> is {"hub": [...]}: the task's environment.hub entries for this environment only (run_one.py writes it). Per entry,
as stage_mlskv_cache.py beside the KV image, plus the two things the late pole's env b needs:
  model          snapshot_download at the revision, only the listed files; then, when its config.json has an auto_map, each
                 named class through transformers' get_class_from_dynamic_module, so HF_HOME/modules holds the remote code
                 the jobs load and can be mounted read-only (transformers copies a module only when it is missing or differs)
  dataset+split  load_dataset(<load_alias or repo>[, config], split=..., revision=...): the name the evaluator loads by, so the
                 processed cache sits under the directory an offline load of that name reads (openai_humaneval, not
                 openai/openai_humaneval)
  dataset, no split  hf_hub_download of each listed file at the revision
Then refs/main per repo, hub/version.txt = "1", and a child process with HF_HUB_OFFLINE / HF_DATASETS_OFFLINE /
TRANSFORMERS_OFFLINE set that loads each entry the way the jobs do: models by bare repo id (config, tokenizer, every listed
file; remote code classes again, now offline and from the snapshot directory, as the jobs pass it), datasets by load name,
config and split. Exit 1 if any fails offline. Writes STAGED.json, with every file's sha256 and size, beside <HF_HOME> (not
inside it: the cache is mounted into the agent's workspace). --check runs only that offline load, in the caller's
environment: isolation's staging check runs it as the box's user over the read-only mounts, so a load that would write hub/
or modules/ fails.
"""
import hashlib
import json
import os
import subprocess
import sys

CHILD = r'''
import json, os, sys
from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
ok = True
for e in json.loads(sys.argv[1]):
    try:
        if e["kind"] == "model":
            from transformers import AutoConfig, AutoTokenizer
            # the snapshot directory by way of its config.json: huggingface_hub 1.x refuses snapshot_download(local_files_only)
            # on a snapshot holding only the listed files (IncompleteSnapshotError, env a, gpu-mock 2026-11-03 09:19Z)
            d = os.path.dirname(hf_hub_download(e["repo"], "config.json", local_files_only=True))
            remote = bool(json.load(open(os.path.join(d, "config.json"))).get("auto_map"))
            AutoConfig.from_pretrained(e["repo"], local_files_only=True, trust_remote_code=remote)
            AutoTokenizer.from_pretrained(e["repo"], local_files_only=True, trust_remote_code=remote)
            for f in e["files"]:
                hf_hub_download(e["repo"], f, local_files_only=True)
            if remote:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                for ref in sorted(set(json.load(open(os.path.join(d, "config.json")))["auto_map"].values())):
                    get_class_from_dynamic_module(ref, d)
            print("OFFLINE OK  model  ", e["repo"], len(e["files"]), "files, config and tokenizer load", "remote code" if remote else "")
        elif e.get("split"):
            name = e.get("load_alias") or e["repo"]
            args = [name] + ([e["config"]] if e.get("config") else [])
            ds = load_dataset(*args, split=e["split"], download_config=DownloadConfig(local_files_only=True))
            print("OFFLINE OK  dataset", name, e.get("config") or "", e["split"], len(ds), "rows")
        else:
            for f in e["files"]:
                hf_hub_download(e["repo"], f, repo_type="dataset", local_files_only=True)
            print("OFFLINE OK  file   ", e["repo"], list(e["files"]))
    except Exception as x:
        ok = False
        print("OFFLINE FAIL", e["kind"], e["repo"], type(x).__name__, str(x)[:300])
sys.exit(0 if ok else 1)
'''


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    if sys.argv[1] == "--check":
        hub = json.load(open(sys.argv[2]))["hub"]
        sys.exit(subprocess.run([sys.executable, "-I", "-c", CHILD, json.dumps(hub)]).returncode)
    hub = json.load(open(sys.argv[1]))["hub"]
    hf_home = os.path.abspath(sys.argv[2])
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
              "HF_MODULES_CACHE", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        os.environ.pop(k, None)
    from huggingface_hub import hf_hub_download, snapshot_download
    for e in hub:
        files = list(e["files"])
        if e["kind"] == "model":
            d = snapshot_download(e["repo"], revision=e["revision"], allow_patterns=files)
            auto = json.load(open(os.path.join(d, "config.json"))).get("auto_map") or {}
            if auto:
                from transformers.dynamic_module_utils import get_class_from_dynamic_module
                for ref in sorted(set(auto.values())):
                    get_class_from_dynamic_module(ref, d)
        elif e.get("split"):
            from datasets import load_dataset
            args = [e.get("load_alias") or e["repo"]] + ([e["config"]] if e.get("config") else [])
            load_dataset(*args, split=e["split"], revision=e["revision"])
        else:
            for f in files:
                hf_hub_download(e["repo"], f, repo_type="dataset", revision=e["revision"])
        refs = os.path.join(hf_home, "hub", f"{e['kind']}s--" + e["repo"].replace("/", "--"), "refs")
        os.makedirs(refs, exist_ok=True)
        open(os.path.join(refs, "main"), "w").write(e["revision"])
        print("staged", e["kind"], e["repo"], e["revision"][:12], flush=True)
    open(os.path.join(hf_home, "hub", "version.txt"), "w").write("1")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-I", "-c", CHILD, json.dumps(hub)], env=env)
    staged = []
    for root, _, fs in os.walk(hf_home):
        for f in fs:
            p = os.path.join(root, f)
            if os.path.islink(p) or f.endswith(".lock"):
                continue
            staged.append({"path": os.path.relpath(p, hf_home), "bytes": os.path.getsize(p), "sha256": sha256(p)})
    json.dump({"interpreter": sys.executable, "hub": hub, "offline_ok": r.returncode == 0,
               "files": sorted(staged, key=lambda x: x["path"])},
              open(os.path.join(os.path.dirname(hf_home), "STAGED.json"), "w"), indent=1)
    print("STAGED.json:", len(staged), "files,", sum(x["bytes"] for x in staged), "bytes; offline", "OK" if r.returncode == 0 else "FAIL")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
