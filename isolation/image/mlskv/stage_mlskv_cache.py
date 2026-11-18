#!/usr/bin/env python3
"""Fill the MLS-Bench KV task's Hugging Face cache, the way MLS-Bench's own vendor/data_scripts/transformers-kv-lab/
prepare_hf_cache.py does, but at the revisions the task file pins and only the files it lists.

usage (inside the mlsbench-kv image, network on, as root): python3 -I stage_mlskv_cache.py <task.json> <HF_HOME>

stage_hub.py (gpu-tasks) fits the TRL tasks; it does not fit this one: it would download every file of each dataset repo
(no allow_patterns for datasets) and load_dataset(repo) with no config, which fails for openai/gsm8k (config `main`) and
for repos that are not `datasets`-loadable (LongBench is read as data.zip, NeedleBench as one JSONL file). Per hub entry:
  model         snapshot_download at the revision, only the listed files
  dataset+split load_dataset(repo[, config], split=..., revision=...), which writes the arrow cache the evaluators read
                (gsm8k: <HF_HOME>/datasets/openai___gsm8k/main/.../gsm8k-test.arrow; LongBench-v2: its train split)
  dataset, no split  hf_hub_download of each listed file at the revision (LongBench data.zip, NeedleBench JSONL)
Then refs/main is written for every repo (offline lookups by bare name resolve through it), hub/version.txt = "1", and a
child process with HF_HUB_OFFLINE / HF_DATASETS_OFFLINE / TRANSFORMERS_OFFLINE set loads each entry the way the
evaluators do (local_files_only) and reports it. Exit 1 if any entry fails offline. Writes <HF_HOME>/STAGED.json with
every file's sha256 and size, for the manifest.
"""
import hashlib
import json
import os
import subprocess
import sys

CHILD = r'''
import json, sys
from datasets import DownloadConfig, load_dataset
from huggingface_hub import hf_hub_download
ok = True
for e in json.loads(sys.argv[1]):
    try:
        if e["kind"] == "model":
            from transformers import AutoConfig, AutoTokenizer
            AutoConfig.from_pretrained(e["repo"], local_files_only=True)
            AutoTokenizer.from_pretrained(e["repo"], local_files_only=True)
            for f in e["files"]:
                hf_hub_download(e["repo"], f, local_files_only=True)
            print("OFFLINE OK  model  ", e["repo"], len(e["files"]), "files, config and tokenizer load")
        elif e.get("split"):
            dc = DownloadConfig(local_files_only=True)
            args = [e["repo"]] + ([e["config"]] if e.get("config") else [])
            ds = load_dataset(*args, split=e["split"], download_config=dc)
            print("OFFLINE OK  dataset", e["repo"], e.get("config") or "", e["split"], len(ds), "rows")
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
    task = json.load(open(sys.argv[1]))
    hf_home = os.path.abspath(sys.argv[2])
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_CACHE", "HF_DATASETS_CACHE",
              "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        os.environ.pop(k, None)
    from huggingface_hub import hf_hub_download, snapshot_download
    hub = task["environment"]["hub"]
    for e in hub:
        files = list(e["files"])
        if e["kind"] == "model":
            snapshot_download(e["repo"], revision=e["revision"], allow_patterns=files)
        elif e.get("split"):
            from datasets import load_dataset
            args = [e["repo"]] + ([e["config"]] if e.get("config") else [])
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
    json.dump({"hub": [{k: e[k] for k in e if k != "files"} | {"files": e["files"]} for e in hub],
               "offline_ok": r.returncode == 0, "files": sorted(staged, key=lambda x: x["path"])},
              open(os.path.join(hf_home, "STAGED.json"), "w"), indent=1)
    print("STAGED.json:", len(staged), "files,", sum(x["bytes"] for x in staged), "bytes; offline", "OK" if r.returncode == 0 else "FAIL")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
