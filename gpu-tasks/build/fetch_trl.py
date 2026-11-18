"""Snapshot TRL's public W&B runs on H100, and each run's entry script at its recorded commit.

usage: python3 -I fetch_trl.py <out W&B dir> <out script dir>

W&B: the public GraphQL endpoint, no API key. For every run in project huggingface/trl whose
runInfo.gpu names an H100, writes <run>.json with metadata, config, summary, full history
and all system-metric events (the API returns every row up to the 20000 asked for; the longest run has under 10000). Scripts: github.com/huggingface/trl contents API at the
run's git commit, written as `<codePath with / as _>@<commit[:8]>`; a path absent at that
commit is skipped. For every run that did not finish, also writes <run>.output.log (the console
log, which holds the error), or <run>.output.log.status when W&B refuses it. Re-running skips files already present.
"""
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

GQL = "https://api.wandb.ai/graphql"


def gql(query):
    err = None
    for _ in range(3):
        try:
            req = urllib.request.Request(GQL, data=json.dumps({"query": query}).encode(), headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=180))
        except Exception as e:  # network errors are retried, then reported
            err = e
            time.sleep(3)
    return {"errors": [str(err)]}


def list_runs():
    runs, cursor = [], None
    while True:
        after = f', after: "{cursor}"' if cursor else ""
        r = gql('query { project(name:"trl", entityName:"huggingface") { runs(first: 200%s) { edges { node { name runInfo { gpu gpuCount } } cursor } pageInfo { hasNextPage } } } }' % after)
        page = r["data"]["project"]["runs"]
        runs += [e["node"] for e in page["edges"]]
        if not page["pageInfo"]["hasNextPage"] or not page["edges"]:
            return runs
        cursor = page["edges"][-1]["cursor"]


def fetch_script(code_path, commit, out_dir):
    out = os.path.join(out_dir, f"{code_path.replace('/', '_')}@{commit[:8]}")
    if os.path.exists(out):
        return "present"
    url = f"https://api.github.com/repos/huggingface/trl/contents/{code_path}?ref={commit}"
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"}), timeout=60))
    except urllib.error.HTTPError as e:
        return f"absent ({e.code})"
    open(out, "wb").write(base64.b64decode(d["content"]))
    return "fetched"


def fetch_output_log(run, out_dir):
    """A run's console log, for the cause of a crash or failure. W&B refuses some files with HTTP 403;
    that answer is kept in <run>.output.log.status so the generator can say why a cause is unknown."""
    out = os.path.join(out_dir, run["name"] + ".output.log")
    st = out + ".status"
    if os.path.exists(out) or os.path.exists(st):
        return "present"
    r = gql('query { project(name:"trl", entityName:"huggingface") { run(name:"%s") { files(names:["output.log"]) { edges { node { directUrl } } } } } }' % run["name"])
    edges = (((r.get("data") or {}).get("project") or {}).get("run") or {}).get("files", {}).get("edges") or []
    if not edges:
        open(st, "w").write("not listed\n")
        return "not listed"
    try:
        data = urllib.request.urlopen(edges[0]["node"]["directUrl"], timeout=120).read()
    except urllib.error.HTTPError as e:
        open(st, "w").write(f"HTTP {e.code}\n")
        return f"HTTP {e.code}"
    open(out, "wb").write(data)
    return f"{len(data)} bytes"


def main():
    wb_dir, sc_dir = sys.argv[1], sys.argv[2]
    os.makedirs(wb_dir, exist_ok=True)
    os.makedirs(sc_dir, exist_ok=True)
    runs = [r for r in list_runs() if "H100" in str((r.get("runInfo") or {}).get("gpu", ""))]
    print("H100 runs:", len(runs))
    for r in runs:
        path = os.path.join(wb_dir, r["name"] + ".json")
        if not os.path.exists(path):
            res = gql('query { project(name:"trl", entityName:"huggingface") { run(name:"%s") { name displayName state createdAt heartbeatAt '
                      'historyLineCount commit config summaryMetrics runInfo { program args codePath os python gpu gpuCount cpuCount git { remote commit } } '
                      'history(samples: 20000) events(samples: 20000) } } }' % r["name"])
            json.dump(res, open(path, "w"))
        run = ((json.load(open(path)).get("data") or {}).get("project") or {}).get("run") or {}
        ri = run.get("runInfo") or {}
        cp, commit = ri.get("codePath"), (ri.get("git") or {}).get("commit")
        status = fetch_script(cp, commit, sc_dir) if cp and commit else "no codePath or commit"
        log = fetch_output_log(run, wb_dir) if run.get("state") != "finished" else "-"
        print(r["name"], run.get("state"), ri.get("gpuCount"), cp, (commit or "")[:8], status, "output.log:", log)


if __name__ == "__main__":
    main()
