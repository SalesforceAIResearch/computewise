#!/usr/bin/env python3
"""r6's decode jobs: the first K examples of each of the 40 KV records, cut into
slices so no one decode process runs much past `--target-s` seconds, longest
first. Cost per example is the record's recorded H100 seconds per example times
`--cpu-ratio` (fit from the fleet's measured decodes). Prints one line per job,
`<record_id>:<start>:<end>\t<estimated s>`, then the totals on stderr.

    python3 -I r6_jobs.py --records R [--k 50] [--target-s 3000] [--cpu-ratio 5]
"""
import argparse
import glob
import json
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--target-s", type=float, default=3000.0)
    ap.add_argument("--cpu-ratio", type=float, default=5.0)
    ap.add_argument("--done", default=None, help="dir of finished slice files to skip")
    ap.add_argument("--from", dest="frm", type=int, default=0, help="decode positions [from, k): extend an earlier first-k")
    ap.add_argument("--workloads", default=None, help="comma list; a record is taken if its record id names one")
    ap.add_argument("--records-only", default=None, help="comma list of record ids: only these")
    a = ap.parse_args()
    done = set(os.listdir(a.done)) if a.done and os.path.isdir(a.done) else set()
    jobs = []
    for p in sorted(glob.glob(os.path.join(a.records, "mls-llm-kv-*.json"))):
        r = json.load(open(p))
        rid = r["identity"]["record_id"]
        if a.workloads and not any(w.strip() and w.strip() in rid for w in a.workloads.split(",")):
            continue
        if a.records_only and rid not in {x.strip() for x in a.records_only.split(",")}:
            continue
        ser = r["materialisation"]["serving"]
        n = int(ser["n_examples_full"]["value"])
        k = min(a.k, n)
        per = float(r["timing"]["wall_total_s"]["value"]) / n * a.cpu_ratio
        size = max(1, min(k, int(a.target_s // max(per, 1e-6))))
        for s0 in range(min(a.frm, k), k, size):
            e = min(k, s0 + size)
            if f"{rid}.{s0}-{e}.json" in done:
                continue
            jobs.append((rid, s0, e, per * (e - s0) + 60.0))
    jobs.sort(key=lambda j: -j[3])
    for rid, s0, e, est in jobs:
        print(f"{rid}:{s0}:{e}\t{est:.0f}")
    tot = sum(j[3] for j in jobs)
    print(f"{len(jobs)} jobs, {tot / 3600:.1f} process-hours estimated, longest {max((j[3] for j in jobs), default=0) / 60:.0f} min",
          file=sys.stderr)


if __name__ == "__main__":
    main()
