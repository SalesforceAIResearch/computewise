#!/usr/bin/env python3
"""Known-answer cases for build/expected_scores.py startup_split and partial_attribution: the split of host-phase idle into
job_host_phase (the environment's) and startup_not_overlapped (the agent's), for whole-box idle and per GPU lane for partial
idle, design/IDEA.md, "job host phase" and "Partial idle also carries causes".
Each case is a hand-built schedule whose answer follows from the definition in startup_split's docstring.
usage: python3 -I build/test_startup_split.py   (exit 0 when every case matches)"""
import importlib.util, os, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("expected_scores", os.path.join(HERE, "expected_scores.py"))
ES = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ES)


def sim(*jobs):
    s = types.SimpleNamespace(jobs=[{"job_id": j["id"], "after": j.get("after", [])} for j in jobs], runs={},
                              box=1 + max(g for j in jobs for g in j.get("gpus", [0, 1, 2, 3])),
                              agent=types.SimpleNamespace(turns=[], tools=[]))
    for i, j in enumerate(jobs):
        s.runs[i] = {"i": i, "job_id": j["id"], "launch": j["launch"], "hello": j.get("hello", j["launch"]), "busy": j["busy"],
                     "bye": j["bye"], "gpus": j.get("gpus", [0, 1, 2, 3]), "killed": j.get("killed", False)}
        if "rank_first_busy" in j:
            s.runs[i]["rank_first_busy"] = j["rank_first_busy"]
    return s


A = dict(id="A", launch=0.0, busy=[(100.0, 1000.0)], bye=1010.0)
CASES = [
    ("first launch: nothing was held before it", sim(A), {"A": (0, 100)}),
    ("serial, no dependencies: B's whole start-up was hideable under A", sim(A, dict(id="B", launch=1010.0, busy=[(1110.0, 2000.0)], bye=2010.0)),
     {"A": (0, 100), "B": (100, 0)}),
    ("serial, B depends on A and was launched at A's exit", sim(A, dict(id="B", launch=1010.0, busy=[(1110.0, 2000.0)], bye=2010.0, after=["A"])),
     {"B": (0, 100)}),
    ("launched early, half hidden; A's exit tail inside B's start-up is the agent's, since B's GPU work could have filled it",
     sim(A, dict(id="B", launch=950.0, busy=[(1050.0, 2000.0)], bye=2010.0)), {"B": (50, 0)}),
    ("dependency never ran: never launchable early", sim(A, dict(id="B", launch=1010.0, busy=[(1110.0, 2000.0)], bye=2010.0, after=["Z"])),
     {"B": (0, 100)}),
    ("launchable at its dependency's exit while another lane still held GPUs",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 500.0)], bye=510.0, gpus=[0]), dict(id="C", launch=0.0, busy=[(100.0, 1000.0)], bye=1010.0, gpus=[1]),
         dict(id="B", launch=1010.0, busy=[(1110.0, 2000.0)], bye=2010.0, gpus=[0], after=["A"])), {"B": (100, 0)}),
    ("a predecessor's in-job gap inside the start-up is held, so it stays the environment's",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 500.0), (510.0, 1000.0)], bye=1010.0), dict(id="B", launch=400.0, busy=[(1000.0, 2000.0)], bye=2010.0)),
     {"B": (0, 10)}),
    ("1-GPU jobs one at a time: B could have started up beside A, so all of its start-up is the agent's",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 150.0)], bye=160.0, gpus=[0]), dict(id="B", launch=160.0, busy=[(260.0, 310.0)], bye=320.0, gpus=[1])),
     {"A": (0, 100), "B": (100, 0)}),
    ("launched together, both late: each is charged, neither excuses the other (see the last case for the total)",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 1000.0)], bye=1000.0, gpus=[0, 1]), dict(id="B", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[2]),
         dict(id="C", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[3])), {"B": (100, 0), "C": (100, 0)}),
    ("C launched 0.5 s after B: B carries the agent's share and C's own start-up is covered by B's (cause total: 100, below)",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 1000.0)], bye=1000.0, gpus=[0, 1]), dict(id="B", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[2]),
         dict(id="C", launch=1000.5, busy=[(1100.5, 2000.0)], bye=2000.0, gpus=[3])), {"B": (100, 0), "C": (0, 99.5)}),
    ("per lane: B waits for A on GPU 0 while D keeps GPU 1 busy, so no whole-box idle, but its whole start-up was hideable on its lane",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 500.0)], bye=500.0, gpus=[0]), dict(id="D", launch=0.0, busy=[(100.0, 2000.0)], bye=2000.0, gpus=[1]),
         dict(id="B", launch=500.0, busy=[(600.0, 1000.0)], bye=1000.0, gpus=[0])), {"B": (0, 0, 100)}),
    ("a 4-rank job whose rank 0 ran one unit early: the start-up runs to the slowest rank, and A is capped at its idle",
     sim(dict(id="A", launch=0.0, busy=[(300.0, 1000.0)], bye=1000.0),
         dict(id="B", launch=1000.0, busy=[(1100.0, 1101.0), (1300.0, 2000.0)], bye=2000.0, rank_first_busy=[1100.0, 1300.0, 1300.0, 1300.0])),
     {"B": (299, 0)}),
    ("two start-ups over the same idle: each job carries it, the cause counts it once",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 1000.0)], bye=1000.0, gpus=[0, 1]), dict(id="B", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[2]),
         dict(id="C", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[3])), {"B": (100, 0), "C": (100, 0)}),
]

bad = 0
for name, s, want in CASES:
    got = ES.startup_split(s)
    for j, w in want.items():
        agent, rest, lane = (w + (None,))[:3]
        g = got[j]
        ok = abs(g["agent_s"] - agent) < 1e-6 and abs(g["env_s"] - rest) < 1e-6 and (lane is None or abs(g["lane_s"] - lane) < 1e-6)
        bad += not ok
        print(("ok  " if ok else "FAIL"), f"{name}: {j} startup_not_overlapped {g['agent_s']:.1f} (want {agent}), rest of its "
              f"start-up's whole-box idle {g['env_s']:.1f} (want {rest}), lane_hideable {g['lane_s']:.1f}" + ("" if lane is None else f" (want {lane})"))
# the cause total: agent parts of all jobs, each instant once
for name, s, want in [(n, s, 100.0) for n, s, _ in CASES if n.startswith(("C launched 0.5 s", "two start-ups"))]:
    once = sum(e - x for x, e in ES.union([iv for v in ES.startup_split(s).values() for iv in v["parts"]]))
    bad += abs(once - want) > 1e-6
    print(("ok  " if abs(once - want) < 1e-6 else "FAIL"), f"{name}: the cause's total is {once:.1f} (want {want:.0f})")

# per GPU lane: (name, sim, {job: (whole-box agent s, partial agent GPU-s, lane_hideable)}, w1, agent_end, {partial cause: GPU-s})
LANE = [
    ("packed, a late lane: C follows A on GPU 0 while X holds GPU 1, so its start-up is partial idle the agent could have hidden",
     sim(dict(id="A", launch=0.0, busy=[(100.0, 1000.0)], bye=1000.0, gpus=[0]), dict(id="X", launch=0.0, busy=[(100.0, 2000.0)], bye=2000.0, gpus=[1]),
         dict(id="C", launch=1000.0, busy=[(1100.0, 2000.0)], bye=2000.0, gpus=[0])),
     {"A": (0, 0, 0), "X": (0, 0, 0), "C": (0, 100, 100)}, 2000.0, 2000.0, {"startup_not_overlapped": 100}),
    ("a dependency-limited start-up running from partial into whole-box idle: the lane's 40 s lands on whole-box instants, which "
     "the whole-box rule charges, so the lane adds nothing (charging the partial instants instead would count the delay twice)",
     sim(dict(id="E", launch=0.0, busy=[(0.0, 560.0)], bye=560.0, gpus=[0]), dict(id="A", launch=0.0, busy=[(0.0, 500.0)], bye=500.0, gpus=[1]),
         dict(id="D", launch=0.0, busy=[(0.0, 600.0)], bye=600.0, gpus=[2]),
         dict(id="B", launch=540.0, busy=[(640.0, 1000.0)], bye=1000.0, gpus=[0], after=["A"])),
     {"B": (40, 0, 40)}, 1000.0, 1000.0, {"job_host_phase": 40, "no_work_left": 780, "unattributed": 40}),
    ("launched 300 s late beside the first wave's start-up: its lane could have started it up during that whole-box idle",
     sim(dict(id="A", launch=0.0, busy=[(271.0, 1000.0)], bye=1000.0, gpus=[0]), dict(id="X", launch=0.0, busy=[(271.0, 2000.0)], bye=2000.0, gpus=[1]),
         dict(id="B", launch=300.0, busy=[(571.0, 1500.0)], bye=1500.0, gpus=[2])),
     {"A": (0, 0, 0), "X": (0, 0, 0), "B": (0, 271, 271)}, 2000.0, 2000.0,
     {"unattributed": 29, "startup_not_overlapped": 271, "no_work_left": 1500}),
]
for name, s, want, w1, agent_end, causes in LANE:
    got = ES.startup_split(s)
    for j, (wa, pa, lane) in want.items():
        g = got[j]
        ok = abs(g["agent_s"] - wa) < 1e-6 and abs(g["partial_agent_gs"] - pa) < 1e-6 and abs(g["lane_s"] - lane) < 1e-6
        bad += not ok
        print(("ok  " if ok else "FAIL"), f"{name}: {j} whole-box agent {g['agent_s']:.1f} (want {wa}), partial agent "
              f"{g['partial_agent_gs']:.1f} GPU-s (want {pa}), lane_hideable {g['lane_s']:.1f} (want {lane})")
    pc = ES.partial_attribution(s, got, w1, agent_end)
    ok = set(pc) == set(causes) and all(abs(pc[k] - v) < 1e-6 for k, v in causes.items())
    bad += not ok
    print(("ok  " if ok else "FAIL"), f"{name}: partial causes {({k: round(v, 3) for k, v in sorted(pc.items())})} (want {causes})")
# The battery's fan-out, in miniature (the study, 2026-10-20: "packed on the battery should carry about 270 s per late lane").
# Four 1-GPU lanes start a first wave together at 0, cold, 271 s each. When lane 0's and lane 1's jobs exit, packed launches the
# next job on that lane, and its 270 s warm start-up runs while lanes 2 and 3 still train: partial idle, which an earlier launch
# would have hidden under the lane's previous job. The first wave is A = 0 (nothing ran before it).
BAT = sim(dict(id="W0", launch=0.0, busy=[(271.0, 1000.0)], bye=1000.0, gpus=[0]),
          dict(id="W1", launch=0.0, busy=[(271.0, 1100.0)], bye=1100.0, gpus=[1]),
          dict(id="W2", launch=0.0, busy=[(271.0, 2000.0)], bye=2000.0, gpus=[2]),
          dict(id="W3", launch=0.0, busy=[(271.0, 2000.0)], bye=2000.0, gpus=[3]),
          dict(id="L0", launch=1000.0, busy=[(1270.0, 1900.0)], bye=1900.0, gpus=[0]),
          dict(id="L1", launch=1100.0, busy=[(1370.0, 1900.0)], bye=1900.0, gpus=[1]))
got = ES.startup_split(BAT)
for j, (wa, pa, lane) in {"W0": (0, 0, 0), "W1": (0, 0, 0), "W2": (0, 0, 0), "W3": (0, 0, 0),
                          "L0": (0, 270, 270), "L1": (0, 270, 270)}.items():
    g = got[j]
    ok = abs(g["agent_s"] - wa) < 1e-6 and abs(g["partial_agent_gs"] - pa) < 1e-6 and abs(g["lane_s"] - lane) < 1e-6
    bad += not ok
    print(("ok  " if ok else "FAIL"), f"battery fan-out, packed: {j} whole-box agent {g['agent_s']:.1f} (want {wa}), partial agent "
          f"{g['partial_agent_gs']:.1f} GPU-s (want {pa}), lane_hideable {g['lane_s']:.1f} (want {lane})")
pc = ES.partial_attribution(BAT, got, 2000.0, 2000.0)
sno = pc.get("startup_not_overlapped", 0.0)
bad += abs(sno - 540) > 1e-6
print(("ok  " if abs(sno - 540) < 1e-6 else "FAIL"), f"battery fan-out, packed: partial startup_not_overlapped {sno:.1f} GPU-s "
      f"(want 540, two late lanes of 270 s); all partial causes {({k: round(v, 3) for k, v in sorted(pc.items())})}")
sys.exit(1 if bad else 0)
