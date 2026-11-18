"""Fetch the MLS-Bench and third-party inputs of the four MLS-Bench tasks into survey/k2-latepole-inputs/, checking each sha256.

usage: python3 -I build/fetch_mlsbench_inputs.py <MLS-Bench clone at b8739449> [<gpu-tasks root>]

None of these files ships with this repository. build/mlsbench-inputs.json names each one, its sha256 and where it comes from:
- "mls-bench": a path in your MLS-Bench clone (the task configurations, launch scripts, leaderboards, evaluator scaffolds, data
  scripts and the dLLM baselines under its third_party/), which must be at the commit the paper used;
- "url": a file on Hugging Face at a pinned revision (four public datasets' test splits, NeedleBench's essays, two LLaDA model files);
- "tar": a file in GitHub's tarball of dLLM-Cache at a fixed commit.
A file whose sha256 differs is refused and nothing is written for it; the script then exits 1.
"""
import hashlib, io, json, os, subprocess, sys, tarfile, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "computewise-fetch"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    clone = os.path.abspath(sys.argv[1])
    root = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.dirname(HERE)
    man = json.load(open(os.path.join(HERE, "mlsbench-inputs.json")))
    head = subprocess.run(["git", "-C", clone, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head != man["mls_bench"]["commit"]:
        sys.exit(f"{clone} is at {head or 'no git commit'}; check out {man['mls_bench']['commit']} first")
    dest_root = os.path.join(root, "survey", "k2-latepole-inputs")
    tars, done, bad = {}, 0, []
    for e in man["files"]:
        if e["from"] == "mls-bench":
            data = open(os.path.join(clone, e["path"]), "rb").read()
        elif e["from"] == "url":
            data = get(e["url"])
        else:
            if e["url"] not in tars:
                tars[e["url"]] = tarfile.open(fileobj=io.BytesIO(get(e["url"])), mode="r:gz")
            t = tars[e["url"]]
            member = next(m for m in t.getmembers() if m.isfile() and m.name.split("/", 1)[-1] == e["member"])
            data = t.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            bad.append(e["dest"])
            continue
        p = os.path.join(dest_root, e["dest"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)
        done += 1
    print(f"MLS-Bench inputs: {done} of {len(man['files'])} files fetched and checked into {dest_root}")
    if bad:
        print("sha256 mismatch, not written:", *bad, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
