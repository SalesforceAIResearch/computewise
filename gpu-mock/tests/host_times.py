#!/usr/bin/env python3
"""Host time per job from a gpu-mock occupancy log: the time a replayed job
spends that the registry cannot know (gpu-tasks' request, 2026-10-20).

    python3 -I tests/host_times.py <occupancy.jsonl>

Per job (one rank is one job):
  start_to_first_enqueue  process start (job_hello.t_start, from /proc) to its
                          first enqueue: interpreter, imports, model loads,
                          tokenising, Trainer setup;
  gaps_beyond_paced       sum over consecutive enqueues of the wall time between
                          them beyond the previous unit's wall_cost: host work the
                          GPU queue did not hide (checkpoint saves, logging);
  last_cursor_to_bye      when its last queued work drains to job_bye: the final
                          save and exit.
Times are wall seconds. Cursors are rebuilt per GPU from the enqueue events
(start = max(previous cursor, t_mono), cursor = start + wall_cost).
"""
from __future__ import annotations
import json, sys
from collections import defaultdict


def main(path):
    ev = [json.loads(l) for l in open(path) if l.strip()]
    hello = {e["job"]: e for e in ev if e["kind"] == "job_hello"}
    bye = {e["job"]: e for e in ev if e["kind"] == "job_bye"}
    enq = defaultdict(list)
    cursor = defaultdict(float)
    last_cursor_wall = {}
    for e in ev:
        if e["kind"] != "enqueue":
            continue
        off = e["t_wall"] - e["t_mono"]
        start = max(cursor[e["gpu"]], e["t_mono"])
        cursor[e["gpu"]] = start + float(e.get("wall_cost") or 0.0)
        enq[e["job"]].append(e)
        last_cursor_wall[e["job"]] = cursor[e["gpu"]] + off
    print(f"{'job':>4} {'gpu':>4} {'steps':>6} {'start->1st enqueue':>19} {'gaps beyond paced':>18} {'paced':>8} {'last cursor->bye':>17}")
    for j in sorted(hello):
        h, es = hello[j], enq.get(j, [])
        t0 = h.get("t_start") or h["t_wall"]
        if not es:
            print(f"{j:>4}  no enqueues (hello at +{h['t_wall'] - t0:.1f} s after process start)")
            continue
        first = es[0]["t_wall"] - t0
        gaps = sum(max(0.0, b["t_mono"] - a["t_mono"] - float(a.get("wall_cost") or 0.0)) for a, b in zip(es, es[1:]))
        paced = sum(float(x.get("wall_cost") or 0.0) for x in es)
        tail = (bye[j]["t_wall"] - last_cursor_wall[j]) if j in bye else float("nan")
        print(f"{j:>4} {es[0]['gpu']:>4} {len(es):>6} {first:>17.1f} s {gaps:>16.1f} s {paced:>6.1f} s {tail:>15.1f} s")


if __name__ == "__main__":
    main(sys.argv[1])
