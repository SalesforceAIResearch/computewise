"""Where each run's lost time goes: the agent's own delay, the order its jobs ran in, and work its own results had
already ruled out. An offline
replay of each run's own job set on the same 4 GPUs. No agent is re-run.

usage: python3 -I decompose.py [--runs DIR] [--json OUT]
       python3 -I decompose.py --selftest      (known answers; needs no runs)
The clairvoyant optimum needs OR-Tools, which the controller's python lacks: python3 -m venv ~/.venvs/ortools &&
~/.venvs/ortools/bin/pip install ortools. It runs in that venv as a subprocess.

Runs: every valid L0, S1 and A3 run of the five packages (G1 mls-kv-report, H5 mls-kv-report-h, G4 mls-kv-gate, G4n
pkg-g4n-a and pkg-g4n-b) from screen.py's rows, no_bar_reading runs left out. Each run's jobs are its own, from
occupancy.jsonl: each job's length is its share of its GPU (shared_pool.attribute, so a job that ran alone keeps its wall
time), and its identity (method or setup, workload, record, whether it ran in full) from gpu-mock's lookup record.

The floor. Each run is measured against the clairvoyant optimum of its own needed work: each record of the reference
plan once, at its longest instance in the run, on 4 GPUs with every job ready at 0 (P||Cmax). LPT settles an instance
where it meets the lower bound; CP-SAT proves the rest. G4n reading (a) adds SOAP's runs to the needed work.

Variants, each a replay of the run's own job set on 4 GPUs, every job ready at 0, read at its deliverable (the last end
among the jobs that ended by the run's own deliverable, a cancelled one no longer counting):
  V1   as run: screen.py's time to the deliverable, the later of the final report and the work's end
  V1f  the agent's order, no delay: first come, first served in the order the agent first launched each job
  V2   longest first with the true lengths (clairvoyant list scheduling)
  V3   the gpu-harness scheduler (experiments/solution/skills/gpu-harness, pinned by hash below) with no agent: online
       calibration, then critical-path list scheduling, which is longest first for independent jobs. Its priors are what
       the files show: a KV evaluation's workload size (examples x 2 s) and one flat prior for every training run. A KV
       evaluator prints no progress, so a KV job's length is learnt when it ends in full (a screen or a cut run is no
       measurement); a training run prints its step time from step 125, which projects its full length. No checks, no
       gates, no judgement holds
  V4   V3, plus cancelling each job the moment the run's own results had ruled it out (the earliest moment that
       evidence existed in the replay, never the reference plan's set):
         G4   a method is ruled out once one workload the answer key fails it on has both its own full evaluation and
              full_attention's finished; its other jobs stop. full_attention's job of a workload stops once every
              method that ran that workload is ruled out
         G4n  two readings of the brief ("each setup's mean"). (a) SOAP's mean is needed: nothing is ruled out.
              (b) It is not: SOAP is ruled out once any SOAP launch prints its step time (score_all's rejection moment)
         all  a repeat: a job whose record has already finished in full is ruled out by that finish
  V1fc V1f with V4's cancellation (for the split below)
  V5   V3 on the needed work alone: what is left once every unneeded job is gone, ruled out in the run or not
The split of the loss V1 - floor, exact by construction:
  delay       V1 - V1f: the agent's own latency, including any wait to read a result before launching
  ordering    the mean of V1f - V3 and V1fc - V4
  unneeded    the mean of V3 - V4 and V1f - V1fc: work the run's own results had ruled out
              (ordering and unneeded interact, since cancelling saves more under one order than another, so each is the
              mean over the two orders it can be removed in: a Shapley split. The brief's path, V1f - V3 then V3 - V4, is
              kept in the JSON as brief_ordering and brief_unneeded; the two differ by at most 0.05 of a cell's loss)
  unruled     V4 - V5: unneeded work the run's results never ruled out (a long evaluation run without the cheap check
              that would have rejected its method; a cut run later rerun)
  harness gap V5 - floor: the online scheduler's distance from the optimum on the needed work
The planning row replays each package's whole job matrix, true lengths, with the rule applied by a judge who reads
each result as it lands, under FCFS in file order, the harness, LPT with true lengths, and the optimum.

Not modelled: an agent paced by these schedulers would decide at different moments; V4's cancellation is instant (the
harness's own stop takes one tick); every job is ready at 0, as the packages have no data dependencies; the floor counts
only the needed work the run itself ran, so a run that left needed work undone has a lower floor ("short of the reading").
"""
import argparse, heapq, json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.normpath(os.path.join(HERE, "..", ".."))
HARNESS = os.path.join(EXP, "solution", "skills", "gpu-harness")
sys.path.insert(0, HERE)
sys.path.insert(0, HARNESS)
import shared_pool as SP                                                       # noqa: E402
from gpuharness.estimate import Estimator                                      # noqa: E402
from gpuharness.model import Board, Candidate, CandState, Job, JobState, Pace, Role   # noqa: E402
from gpuharness.scheduler import Policy, decide                                # noqa: E402

PACKAGES = SP.PACKAGES
M = 4
B = 1_000_000.0          # the harness clock never starts at 0 (a start time of 0 reads as unset)
N_EXAMPLES = {"gsm8k": 1319, "longbench-hotpotqa": 300, "longbench-passage-retrieval": 300, "longbench-repobench": 300,
              "longbench-v2": 503, "needlebench-niah": 441}
PRINT_STEP = 125         # the training scripts print their step time from this step
FULL_STEPS = {"contra-soft-muon": 3040, "soda-fade": 3040, "radial-damp": 3020, "aurora-pruned": 3020,
              "newton-muon": 3300, "soap": 2900}     # each setup's full run, from the runs' own last steps
HARNESS_POLICY = Policy(calibrate=True, hold_s=0.0, gate_on_checks=False, speculate=False)


def harness_hash():
    import hashlib
    h = hashlib.sha256()
    for f in ("scheduler.py", "estimate.py", "model.py"):
        h.update(open(os.path.join(HARNESS, "gpuharness", f), "rb").read())
    return h.hexdigest()[:12]


class Replay:
    """One replay on m GPUs from 0. Jobs: dicts with id, len, and for the harness group, kind, prior, ref, reveal (the
    fraction of its length at which its pace shows, or None). mode 'order' starts the queue by `rank` (lower first);
    'harness' asks gpuharness.decide. rule(self, now) may call cancel(jid) and reject(group) after each event."""

    def __init__(self, jobs, m=M, mode="order", rank=None, rule=None):
        self.jobs = {j["id"]: j for j in jobs}
        self.m, self.mode, self.rule = m, mode, rule
        self.rank = rank or {}
        self.start, self.end, self.cancelled, self.rejected = {}, {}, set(), set()
        self.running, self.gpu_of, self.free = {}, {}, list(range(m))
        self.ev, self.seq = [], 0
        if mode == "harness":
            self.board = Board(gpus=list(range(m)))
            self.board.gpu_job = {g: None for g in range(m)}
            for j in jobs:
                g = j["group"]
                if g not in self.board.candidates:
                    self.board.candidates[g] = Candidate(id=g, reference=bool(j.get("ref")))
                self.board.jobs[j["id"]] = Job(id=j["id"], command="x", candidate=g, group=g, kind=j["kind"],
                                               role=Role.NEEDED, estimate_s=j["prior"], added_at=float(len(self.board.jobs)))
            self.est = Estimator()
            self.est.load(self.board.jobs.values())

    def push(self, t, kind, jid):
        heapq.heappush(self.ev, (t, kind, self.seq, jid)); self.seq += 1

    def deliverable(self, owed):
        """When the last of the owed jobs (those the answer rested on) still run ends: a cancelled one no longer counts."""
        return max([0.0] + [self.end[j] for j in owed if j in self.end and j not in self.cancelled])

    def done(self, jid):
        return jid in self.end and jid not in self.cancelled

    def cancel(self, jid, now):
        if jid in self.end or jid in self.cancelled:
            return
        self.cancelled.add(jid)
        if jid in self.running:
            del self.running[jid]
            self.end[jid] = now
            self.free.append(self.gpu_of[jid])
            if self.mode == "harness":
                self.board.gpu_job[self.gpu_of[jid]] = None
        if self.mode == "harness":
            self.board.jobs[jid].state = JobState.CANCELLED

    def reject(self, group, now):
        if group in self.rejected:
            return
        self.rejected.add(group)
        if self.mode == "harness":
            self.board.candidates[group].state = CandState.REJECTED
        for jid, j in self.jobs.items():
            if j.get("group") == group:
                self.cancel(jid, now)

    def fill(self, now):
        if self.mode == "order":
            queue = sorted((jid for jid in self.jobs if jid not in self.start and jid not in self.cancelled),
                           key=lambda jid: (self.rank.get(jid, 0), jid))
            starts = []
            for jid in queue:
                if not self.free:
                    break
                starts.append((jid, self.free.pop(0)))
        else:
            self.free.sort()
            dec = decide(self.board, self.est, B + now, HARNESS_POLICY)
            starts = [(jid, g) for jid, g in dec.starts]
            for _, g in starts:
                self.free.remove(g)
        for jid, g in starts:
            j = self.jobs[jid]
            self.start[jid], self.gpu_of[jid] = now, g
            self.running[jid] = now + j["len"]
            self.push(now + j["len"], 0, jid)
            if self.mode == "harness":
                hj = self.board.jobs[jid]
                hj.state, hj.gpu, hj.started_at = JobState.RUNNING, g, B + now
                self.board.gpu_job[g] = jid
            if j.get("reveal") is not None:
                self.push(now + j["reveal"] * j["len"], 1, jid)

    def run(self):
        if self.rule:
            self.rule(self, 0.0)
        self.fill(0.0)
        while self.ev:
            now = self.ev[0][0]
            while self.ev and self.ev[0][0] == now:
                _, kind, _, jid = heapq.heappop(self.ev)
                if jid in self.cancelled or jid not in self.running:
                    continue
                if kind == 0:
                    del self.running[jid]
                    self.end[jid] = now
                    self.free.append(self.gpu_of[jid])
                    if self.mode == "harness":
                        hj = self.board.jobs[jid]
                        hj.state, hj.ended_at = JobState.DONE, B + now
                        self.board.gpu_job[self.gpu_of[jid]] = None
                        # only a completed full job is a measurement: a screen, or a run the agent cut short, says
                        # nothing about its workload's full length (the harness records none for a stopped job)
                        if self.jobs[jid].get("full", True):
                            self.est.measured(hj, self.jobs[jid]["len"])
                else:
                    self.revealed = getattr(self, "revealed", set()) | {jid}
                    if self.mode == "harness":
                        j = self.jobs[jid]
                        hj = self.board.jobs[jid]
                        full_len = j.get("proj") or j["len"]
                        hj.pace = Pace(done=j["reveal"], total=1.0, elapsed_s=now - self.start[jid],
                                       projected_s=full_len, at=B + now)
                        self.est.projected(hj, full_len)
            if self.rule:
                self.rule(self, now)
            self.fill(now)
        return max(self.end.values()) if self.end else 0.0


def repeat_rule(rp, now):
    """A job whose record has already finished (a full run of it) is ruled out by that finish."""
    fin = {rp.jobs[j]["record"] for j in rp.end if rp.done(j) and rp.jobs[j].get("full") and rp.jobs[j].get("record")}
    for jid, j in rp.jobs.items():
        if j.get("full") and j.get("record") in fin and not rp.done(jid) and jid not in rp.cancelled:
            rp.cancel(jid, now)


def g4_rule(fails, ref):
    def rule(rp, now):
        repeat_rule(rp, now)
        have = {(rp.jobs[j]["group"], rp.jobs[j]["kind"]) for j in rp.end if rp.done(j) and rp.jobs[j].get("full")}
        for g, ws in fails.items():
            if g not in rp.rejected and any((g, w) in have and (ref, w) in have for w in ws):
                rp.reject(g, now)
        groups_of = {}
        for j in rp.jobs.values():
            if j["group"] != ref:
                groups_of.setdefault(j["kind"], set()).add(j["group"])
        for jid, j in rp.jobs.items():
            if j["group"] == ref and j["kind"] in groups_of and groups_of[j["kind"]] <= rp.rejected:
                rp.cancel(jid, now)
    return rule


def g4n_rule(reading):
    def rule(rp, now):
        repeat_rule(rp, now)
        if reading == "b" and any(rp.jobs[j]["group"] == "soap" for j in getattr(rp, "revealed", ())):
            rp.reject("soap", now)
    return rule


def split(v, floor, rd=""):
    """The loss over the floor, in four parts that sum to it exactly. The agent's delay comes off first (V1 - V1f). Ordering
    and unneeded work interact (cancelling a job saves more under a good order than a bad one), so each is the mean of
    what it saves over the two orders it can be removed in (the Shapley split of V1f - V4): ordering O = mean(V1f - V3,
    V1fc - V4), unneeded U = mean(V3 - V4, V1f - V1fc). What is left, V4 - floor, is the online cost: work run before
    the results ruled it out, and the harness's own gap to the optimum on the evidence it kept."""
    V1, V1f, V1fc, V3, V4 = v["V1"], v["V1f"], v["V1fc" + rd], v["V3"], v["V4" + rd]
    o = ((V1f - V3) + (V1fc - V4)) / 2
    u = ((V3 - V4) + (V1f - V1fc)) / 2
    out = dict(loss=V1 - floor, delay=V1 - V1f, ordering=o, unneeded=u, online=V4 - floor,
               brief_ordering=V1f - V3, brief_unneeded=V3 - V4)
    if "V1g" in v:              # the delay in two: GPUs left idle, and the agent's own placement against FCFS
        out["idle"] = V1 - v["V1g"]
        out["placement"] = v["V1g"] - V1f
    if "V5" + rd in v:          # the online cost in two: unneeded work the run's results never ruled out, and the rest
        out["unruled"] = V4 - v["V5" + rd]
        out["harness_gap"] = v["V5" + rd] - floor
    return out


def selftest():
    fails = []
    def ok(name, got, want, tol=1e-6):
        good = abs(got - want) <= tol
        print(f"{'ok  ' if good else 'FAIL'} {name}: {got:g} (want {want:g})")
        if not good:
            fails.append(name)
    def four(jobs, agent, better, rule, V1):
        mk = lambda: [dict(x) for x in jobs]
        owed = [j["id"] for j in jobs]
        def dl(**kw):
            rp = Replay(mk(), m=2, **kw); rp.run(); return rp.deliverable(owed)
        return dict(V1=V1, V1f=dl(rank=agent), V1fc=dl(rank=agent, rule=rule), V3=dl(rank=better),
                    V4=dl(rank=better, rule=rule))
    J = lambda i, l: dict(id=i, len=float(l))
    kill_x = lambda rp, now: rp.cancel("X", now) if rp.done("C") else None
    none = lambda rp, now: None
    # (i) delay alone: the agent's order is already the best, no unneeded work; its deliverable came 7 s late
    v = four([J("L", 30), J("S1", 10), J("S2", 10)], {"L": 0, "S1": 1, "S2": 2}, {"L": 0, "S1": 1, "S2": 2}, none, 37)
    sp = split(v, 30)
    ok("(i) delay set to 7", sp["delay"], 7); ok("(i) ordering 0", sp["ordering"], 0); ok("(i) unneeded 0", sp["unneeded"], 0)
    ok("(i) online 0: the floor is reached", sp["online"], 0)
    # (ii) ordering alone: the long job last in the agent's order (S1, S2 at 0, L at 10 to 40), best is 30
    v = four([J("L", 30), J("S1", 10), J("S2", 10)], {"S1": 0, "S2": 1, "L": 2}, {"L": 0, "S1": 1, "S2": 2}, none, 40)
    sp = split(v, 30)
    ok("(ii) ordering set to 10", sp["ordering"], 10); ok("(ii) delay 0", sp["delay"], 0); ok("(ii) unneeded 0", sp["unneeded"], 0)
    # (iii) unneeded alone: one order for both; X (20 s) is ruled out by C (5 s) and stopped when C ends
    jobs = [J("C", 5), J("L", 30), J("X", 20), J("S1", 10), J("S2", 10)]
    best = {"C": 0, "L": 1, "X": 2, "S1": 3, "S2": 4}
    v = four(jobs, best, best, kill_x, 40)
    sp = split(v, 30)
    ok("(iii) V3: C, L at 0; X 5 to 25; S1 25 to 35; S2 30 to 40", v["V3"], 40)
    ok("(iii) V4: X stopped at 5; S1 5 to 15, S2 15 to 25; L to 30", v["V4"], 30)
    ok("(iii) unneeded set to 10", sp["unneeded"], 10); ok("(iii) ordering 0", sp["ordering"], 0)
    ok("(iii) both orders of removal agree", sp["brief_unneeded"], sp["unneeded"])
    # (iv) all three, interacting: the agent launched S1, S2, C, X, L, and delivered at 52 s
    v = four(jobs, {"S1": 0, "S2": 1, "C": 2, "X": 3, "L": 4}, best, kill_x, 52)
    sp = split(v, 30)
    ok("(iv) V1f: S1, S2 at 0; C, X at 10; L 15 to 45", v["V1f"], 45)
    ok("(iv) V1fc: X stopped at 15 when C ends; L 15 to 45 regardless", v["V1fc"], 45)
    ok("(iv) delay = 52 - 45", sp["delay"], 7)
    ok("(iv) the brief's path: ordering 45 - 40 = 5", sp["brief_ordering"], 5)
    ok("(iv) the brief's path: unneeded 40 - 30 = 10", sp["brief_unneeded"], 10)
    ok("(iv) Shapley ordering: mean(45 - 40, 45 - 30) = 10", sp["ordering"], 10)
    ok("(iv) Shapley unneeded: mean(40 - 30, 45 - 45) = 5", sp["unneeded"], 5)
    ok("(iv) the parts sum to the loss", sp["delay"] + sp["ordering"] + sp["unneeded"] + sp["online"], sp["loss"])
    # a running job is stopped, not just a queued one dropped
    rp = Replay([dict(x) for x in jobs], m=2, rank={"X": 0, "C": 1, "L": 2, "S1": 3, "S2": 4}, rule=kill_x)
    ms = rp.run()
    ok("a running X stops at 5 when C ends", rp.end["X"], 5)
    ok("  L and S1 take the two GPUs at 5; L ends at 35", ms, 35)
    # the G4 rule: method m fails on workload w; it is ruled out when m's and the reference's full w have both ended
    g = [dict(id="ref-w", len=10.0, group="ref", kind="w", full=True, record="r1"),
         dict(id="m-w", len=4.0, group="m", kind="w", full=True, record="r2"),
         dict(id="m-long", len=100.0, group="m", kind="v", full=True, record="r3"),
         dict(id="ref-v", len=8.0, group="ref", kind="v", full=True, record="r4")]
    rp = Replay([dict(x) for x in g], m=2, rank={"m-long": 0, "ref-w": 1, "m-w": 2, "ref-v": 3},
                rule=g4_rule({"m": {"w"}}, "ref"))
    rp.run()
    ok("G4: ref-w ends at 10, m-w runs 10 to 14, so m is out at 14 and m-long stops then", rp.end["m-long"], 14)
    ok("G4: the reference's v stops too, since every method that ran v is out", "ref-v" in rp.cancelled, True)
    # G4n reading (b): SOAP is out once any SOAP launch shows its pace
    g = [dict(id="s1", len=100.0, group="soap", kind="train", full=True, record="a", reveal=0.04),
         dict(id="s2", len=100.0, group="soap", kind="train", full=True, record="b", reveal=0.04),
         dict(id="c", len=50.0, group="base", kind="train", full=True, record="c", reveal=0.04)]
    rp = Replay([dict(x) for x in g], m=2, rank={"s1": 0, "c": 1, "s2": 2}, rule=g4n_rule("b"))
    rp.run()
    ok("G4n (b): s1 shows its pace at 4 s and stops; s2 never starts", rp.end["s1"], 4)
    ok("G4n (b): s2 dropped", "s2" in rp.cancelled, True)
    # a repeat of a finished record
    rp = Replay([dict(id="a", len=10.0, full=True, record="r"), dict(id="b", len=30.0, full=True, record="r")],
                m=2, rank={"a": 0, "b": 1}, rule=repeat_rule)
    ok("repeat: b stops when a's record finishes", rp.run(), 10)
    # the harness with true priors on independent jobs is longest first: LPT on 4 GPUs
    hj = [dict(id=f"j{i}", len=float(x), group=f"g{i}", kind="k", prior=float(x), ref=(i == 0), reveal=None)
          for i, x in enumerate([50, 40, 30, 30, 20, 20, 10, 10])]
    ok("harness, true priors, no shared groups: LPT's makespan", Replay([dict(x) for x in hj], mode="harness").run(), 60)
    # the clairvoyant optimum: 3, 3, 2, 2, 2 on 2 machines is 6 (LPT gives 7)
    if os.path.exists(OR_PY):
        ok("CP-SAT optimum of 3, 3, 2, 2, 2 on 2 GPUs", optimum_many([[3, 3, 2, 2, 2]], m=2)[0]["opt"], 6)
        # the longest job need not go on the most loaded machine: {10, 6} and {6, 6, 6} is 18 (two symmetry breaks gave 22)
        o = optimum_many([[10, 6, 6, 6, 6]], m=2)[0]
        ok("CP-SAT optimum of 10, 6, 6, 6, 6 on 2 GPUs (no longest-job break)", o["opt"], 18)
        ok("  proved", o["status"] == "OPTIMAL", True)
        rp = Replay([J(f"j{i}", x) for i, x in enumerate([3, 3, 2, 2, 2])], m=2, rank={f"j{i}": i for i in range(5)})
        ok("  and LPT on it", rp.run(), 7)
    # pinned (V1g): the agent put A (30 s) and B (10 s) on GPU 0 and C (10 s) on GPU 1, with idle gaps; it delivered at 50
    run = dict(jobs=[dict(id="A", gpu=0, s=0, len=30.0), dict(id="B", gpu=0, s=35, len=10.0),
                     dict(id="C", gpu=1, s=2, len=10.0)], owed=["A", "B", "C"])
    ok("pinned: GPU 0 runs A then B back to back, 40 s", pinned(run), 40)
    v = dict(V1=50.0, V1g=40.0, V1f=30.0, V1fc=30.0, V3=30.0, V4=30.0)
    sp = split(v, 30.0)
    ok("idle: 50 - 40", sp["idle"], 10); ok("placement: 40 - 30 (B could have gone to GPU 1)", sp["placement"], 10)
    ok("idle + placement is the delay", sp["idle"] + sp["placement"], sp["delay"])
    # a job stacked onto a busy GPU names it in hold and enqueue only: load_run reads its GPU from job_gpu_map
    occ = [dict(kind="busy_start", job=1, gpu=2), dict(kind="hold", job=1, gpu=2), dict(kind="hold", job=5, gpu=2),
           dict(kind="enqueue", job=5, gpu=2), dict(kind="enqueue", job=6, gpu=3), dict(kind="hold", job=7, gpu=-1)]
    gm, wide = SP.job_gpu_map(occ)
    ok("stacked job 5 read on GPU 2 from its hold", gm.get(5, -9), 2)
    ok("job 6 read on GPU 3 from its enqueue", gm.get(6, -9), 3)
    ok("a gpu -1 record names no GPU", 7 in gm, False)
    ok("busy_start alone drops 5 and 6 (the defect)", len(SP.job_gpu_map(occ, ("busy_start",))[0]), 1)
    print(f"{'PASS' if not fails else 'FAIL'}: {len(fails)} failed")
    return 1 if fails else 0


KEY = os.path.join(EXP, "gpu-tasks", "survey", "mls-kv-gate-answer-2026-10-23.json")
REF_GROUP = {"kv": "full_attention", "g4n": "contra-soft-muon"}
SEC_PER_EX = 2.0         # the harness's prior for a KV evaluation: its examples (what the files show) x this
TRAIN_PRIOR_S = 4500.0   # one flat prior for every training run: the files don't state step times
OWED_TOL_S = 60.0


def fam(task):
    return "g4n" if task.startswith("pkg-g4n") else "kv"


def ident(task, lk, argv):
    """(group, kind) of one launch: the method or setup and the workload, from gpu-mock's lookup, else the command."""
    tj, rec = lk.get("task_job"), str(lk.get("record_id") or "")
    if fam(task) == "g4n":
        if (tj or "").startswith("soap") or "soap" in rec:
            return "soap", ("probe" if "probe" in (tj or "") + rec else "train")
        if tj:
            return tj.rsplit("-run", 1)[0], "train"
        return "_adhoc", "adhoc"
    if tj and "--" in tj:
        g, k = tj.split("--", 1)
        return g, k
    toks = argv.split()
    meth = next((t.split("methods/")[1].split("/")[0] for t in toks if "methods/" in t), None)
    wl = toks[toks.index("--workload") + 1] if "--workload" in toks[:-1] else None
    if meth and wl:
        wl = wl.replace("_", "-")
        return meth, wl if wl in N_EXAMPLES else ("longbench-" + wl if "longbench-" + wl in N_EXAMPLES else wl)
    return "_adhoc", "adhoc"


def load_run(row, base):
    T, S = SP._mods()
    rid, task = row["run_id"], row["task"]
    d = os.path.join(base, rid)
    occ = T.W._jsonl(os.path.join(d, "occupancy.jsonl"))
    st = json.load(open(os.path.join(d, "status.json")))
    t0, tend = st["agent_start_wall"], st.get("teardown_wall")
    hello, bye, look, gpu, printed = {}, {}, {}, {}, {}
    for o in occ:
        k = o.get("kind")
        if k == "job_hello":
            hello[o["job"]] = o
        elif k == "job_bye":
            bye[o["job"]] = o
        elif k == "lookup":
            look.setdefault(o["job"], o)
        elif (k == "enqueue" and o.get("unit") == "step" and isinstance(o.get("step"), (int, float))
              and o["step"] >= PRINT_STEP):
            printed.setdefault(o["job"], o["t_wall"])
    gpu, _ = SP.job_gpu_map(occ)          # busy_start, hold or enqueue: a job stacked onto a busy GPU has no busy_start
    work = SP.job_work(occ)               # each job's own solo length, for dividing a GPU several jobs shared
    w0, _ = S.w0_makespan(task, S.plan_key(json.load(open(os.path.join(d, "task.json")))))
    D = row["over_ref"] * w0
    ref = T.reference_records(task)
    jobs = []
    for j, h in hello.items():
        if j not in gpu:
            continue
        b, lk = bye.get(j) or {}, look.get(j) or {}
        e = (b.get("t_wall") or tend or h["t_wall"]) - t0
        argv = " ".join(h.get("argv") or [])
        g, k = ident(task, lk, argv)
        sub = (lk.get("subset") or {}).get("k") if isinstance(lk.get("subset"), dict) else None
        last = b.get("last_step") or 0
        if fam(task) == "kv":
            full = not sub and b.get("served") == "result" and last >= N_EXAMPLES.get(k, 10 ** 9)
            prior = (sub or N_EXAMPLES.get(k, 300)) * SEC_PER_EX
        else:
            full = k == "train" and last >= 2900
            prior = TRAIN_PRIOR_S
        base_kind = k
        wall = e - (h["t_wall"] - t0)
        rv = ((printed[j] - h["t_wall"]) / wall) if (j in printed and wall > 0 and fam(task) == "g4n") else None
        # what the step-time print projects: the whole run's length, also for a run the agent later cut short
        fs = FULL_STEPS.get(g)
        proj = (wall * fs / last) if (fam(task) == "g4n" and fs and last and not full) else None
        jobs.append(dict(id=str(j), gpu=gpu[j], s=h["t_wall"] - t0, e=e, work=work.get(j), group=g, kind=k, base_kind=base_kind, full=full,
                         record=lk.get("record_id"), needed=lk.get("record_id") in ref, prior=prior,
                         ref=(g == REF_GROUP[fam(task)]), reveal=rv if rv is not None and 0 < rv < 1 else None,
                         proj=proj))
    share, union = SP.attribute(jobs)
    for j in jobs:
        j["len"] = share[j["id"]]
    W = max([D] + [j["e"] for j in jobs])
    return dict(run_id=rid, task=task, package=PACKAGES[task], model=row["model"], arm=row["arm"], D=D, jobs=jobs,
                owed=[j["id"] for j in jobs if j["e"] <= D + OWED_TOL_S], cap=M * W, idle=M * W - union,
                order={j["id"]: j["s"] for j in jobs})


def gate_fails():
    k = json.load(open(KEY))
    return {m: {w.replace("_", "-") for w, v in (ws or {}).items() if isinstance(v, dict) and v.get("passes") is False}
            for m, ws in (k.get("per_method") or {}).items()}


def rules(task):
    """{reading: rule} for V4."""
    if task == "mls-kv-gate":
        return {"": g4_rule(gate_fails(), "full_attention")}
    if fam(task) == "g4n":
        return {"a": g4n_rule("a"), "b": g4n_rule("b")}
    return {"": repeat_rule}


def pinned(run):
    """V1g: the agent's own GPU for each job, its jobs back to back in launch order, every job ready at 0. It keeps the
    agent's placement and removes only its gaps, so V1 - V1g is the time its GPUs sat idle (never negative: a GPU
    can't finish its jobs sooner than their summed length) and V1g - V1f what its placement cost against first come,
    first served on any free GPU (negative where its placement packed better)."""
    by, end = {}, {}
    for j in sorted(run["jobs"], key=lambda j: (j["s"], j["id"])):
        by.setdefault(j["gpu"], []).append(j)
    for js in by.values():
        t = 0.0
        for j in js:
            t += j["len"]
            end[j["id"]] = t
    return max([0.0] + [end[i] for i in run["owed"] if i in end])


def variants(run):
    jobs = lambda: [dict(j) for j in run["jobs"]]
    out = {"V1": run["D"], "V1g": pinned(run)}
    rp = Replay(jobs(), rank=run["order"]); rp.run(); out["V1f"] = rp.deliverable(run["owed"])
    rp = Replay(jobs(), rank={j["id"]: -j["len"] for j in run["jobs"]}); rp.run(); out["V2"] = rp.deliverable(run["owed"])
    rp = Replay(jobs(), mode="harness"); rp.run(); out["V3"] = rp.deliverable(run["owed"])
    for reading, rule in rules(run["task"]).items():
        rp = Replay(jobs(), mode="harness", rule=rule); rp.run()
        out["V4" + reading] = rp.deliverable(run["owed"])
        rp = Replay(jobs(), rank=run["order"], rule=rule); rp.run()
        out["V1fc" + reading] = rp.deliverable(run["owed"])
        # V5: the harness on the needed work alone (each needed record once, its longest instance): what is left once
        # every unneeded job is gone, whether or not the run's own results ruled it out
        keep = needed_jobs(run, reading)
        rp = Replay([dict(j) for j in run["jobs"] if j["id"] in keep], mode="harness"); rp.run()
        out["V5" + reading] = rp.deliverable(keep)
    return out


def needed_jobs(run, reading):
    T, _ = SP._mods()
    ref = T.reference_records(run["task"])
    best = {}
    for j in run["jobs"]:
        if (j["record"] in ref or (reading == "a" and j["group"] == "soap" and j["base_kind"] == "train")) and j["record"]:
            if j["record"] not in best or j["len"] > best[j["record"]]["len"]:
                best[j["record"]] = j
    return {j["id"] for j in best.values()}


def needed_lengths(run, reading):
    """The run's own needed work, for its floor: each needed record once, at its longest instance in the run (a cut run
    that was rerun counts its full rerun; one never rerun counts its cut length, as the deliverable rested on it). G4n
    reading (a) adds SOAP's runs. Also whether the run did all of that reading's work (False: SOAP left unfinished)."""
    T, _ = SP._mods()
    ref = T.reference_records(run["task"])
    best = {}
    for j in run["jobs"]:
        keep = j["record"] in ref or (reading == "a" and j["group"] == "soap" and j["base_kind"] == "train")
        if keep and j["record"]:
            best[j["record"]] = max(best.get(j["record"], 0.0), j["len"])
    short = None
    if reading == "a":
        short = sum(1 for j in run["jobs"] if j["group"] == "soap" and j["base_kind"] == "train" and j["full"]) < 3
    return list(best.values()), short


CPSAT = r"""
import json, sys
from ortools.sat.python import cp_model
out = []
m = int(sys.argv[2])
for L in json.load(open(sys.argv[1])):
    L = [int(round(x)) for x in L]                 # whole seconds
    md = cp_model.CpModel(); x = [[md.NewBoolVar("") for _ in range(m)] for _ in L]
    for r in x: md.AddExactlyOne(r)
    C = md.NewIntVar(0, sum(L), "C")
    load = [sum(L[i] * x[i][k] for i in range(len(L))) for k in range(m)]
    for k in range(m): md.Add(load[k] <= C)
    for k in range(m - 1): md.Add(load[k] >= load[k + 1])      # symmetry: machines in load order (the only break)
    # Not also "the longest job on the first machine": with loads in order that forces the longest job onto the most
    # loaded machine, which an optimum needn't do ({10, 6, 6, 6, 6} on 2: 18 is optimal, the pair of breaks gave 22 and
    # still said OPTIMAL; lane OBS, analysis 2026-11-02 16:06 MST)
    md.Minimize(C); s = cp_model.CpSolver(); s.parameters.max_time_in_seconds = 20; s.parameters.num_workers = 1
    st = s.Solve(md)
    out.append({"opt": s.ObjectiveValue(), "bound": s.BestObjectiveBound(), "status": s.StatusName(st)})
print(json.dumps(out))
"""
OR_PY = os.path.expanduser("~/.venvs/ortools/bin/python")


def lpt_makespan(L, m=M):
    loads = [0.0] * m
    for x in sorted(L, reverse=True):
        loads[loads.index(min(loads))] += x
    return max(loads)


def lower_bound(L, m=M):
    """P||Cmax: the mean load, the longest job, and the (m+1)-th longest job sharing a GPU with the m-th."""
    s = sorted(L, reverse=True)
    lb = max(sum(L) / m, s[0] if s else 0.0)
    if len(s) > m:
        lb = max(lb, s[m - 1] + s[m])
    return lb


def optimum_many(instances, m=M):
    """The clairvoyant optimum of each instance (independent one-GPU jobs on m GPUs, P||Cmax). Where LPT meets the
    lower bound it is optimal and no solver runs; the rest go to CP-SAT (OR-Tools, in a venv: the controller's python has
    none), one worker, 20 s each, with LPT's makespan as the start. The status says whether each was proved."""
    import tempfile
    out, hard = [], []
    for i, L in enumerate(instances):
        lb, ub = lower_bound(L, m), lpt_makespan(L, m)
        if ub <= lb * (1 + 1e-9):          # LPT has met a lower bound, so it is optimal; nothing looser counts
            out.append({"opt": ub, "bound": lb, "status": "OPTIMAL", "how": "LPT meets the lower bound"})
        else:
            out.append(None); hard.append(i)
    if hard:
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "l.json"), "w").write(json.dumps([instances[i] for i in hard]))
            open(os.path.join(td, "m.py"), "w").write(CPSAT)
            p = subprocess.run([OR_PY, "-I", os.path.join(td, "m.py"), os.path.join(td, "l.json"), str(m)],
                               capture_output=True, text=True, timeout=3600)
        if p.returncode:
            raise SystemExit("CP-SAT failed: " + p.stderr[-500:])
        for i, o in zip(hard, json.loads(p.stdout)):
            o["how"] = "CP-SAT"
            out[i] = o
    return out


def needed_set(task, soap_len):
    """The reference plan's jobs with their paced recorded lengths, per reading (G4n (a) adds SOAP's three full runs)."""
    e = json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "expected", f"{task}.json")))
    L = [j["paced_recorded_s"] for j in e["policies"]["packed"]["jobs"]]
    if fam(task) == "g4n":
        return {"a": L + [soap_len] * 3, "b": L}
    return {"": L}


def instance(task, soap_len):
    """The package's whole job matrix as the files offer it (the task file, plus G4n's three SOAP runs), true lengths."""
    t = json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{task}.json")))
    out = []
    for i, j in enumerate(t["jobs"]):
        L = j["disclosure"]["estimate"]["median_s"]
        if fam(task) == "kv":
            g, k = j["job_id"].split("--", 1)
            out.append(dict(id=j["job_id"], len=L, group=g, kind=k, full=True, record=j["job_id"],
                            prior=N_EXAMPLES[k] * SEC_PER_EX, ref=g == "full_attention", reveal=None, order=i))
        else:
            g = "soap" if j["job_id"].startswith("soap") else j["job_id"].rsplit("-run", 1)[0]
            k = "probe" if "probe" in j["job_id"] else "train"
            out.append(dict(id=j["job_id"], len=L, group=g, kind=k, full=k == "train", record=j["job_id"],
                            prior=TRAIN_PRIOR_S, ref=g == "contra-soft-muon", reveal=0.04 if k == "train" else 0.999,
                            order=i))
    if fam(task) == "g4n":
        for s in range(3):
            out.append(dict(id=f"soap-run{s + 1}", len=soap_len, group="soap", kind="train", full=True,
                            record=f"soap-run{s + 1}", prior=TRAIN_PRIOR_S, ref=False, reveal=0.043, order=len(out)))
    return out


def ratio(x, o):
    return f"{x / o:.3f}" if o else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("~/computewise-runs"))
    ap.add_argument("--json")
    ap.add_argument("--gpu-from", default=",".join(SP.GPU_FROM),
                    help="occupancy events that name a job's GPU (as shared_pool.py; busy_start alone reproduces the defect "
                         "fixed 2026-11-02)")
    ap.add_argument("--share", choices=("work", "equal"), default=SP.SHARE,
                    help="a shared GPU's time: by each job's own work (default), or equal slices (the defect fixed 2026-11-02)")
    a = ap.parse_args()
    SP.GPU_FROM, SP.SHARE = tuple(a.gpu_from.split(",")), a.share
    T, S = SP._mods()
    rows, _ = T.screen()
    sel = [r for r in rows if r["task"] in PACKAGES and r["arm"] in ("L0", "S1", "A3") and r["validity"] == "valid"
           and not r.get("no_bar_reading")]
    runs = []
    for r in sorted(sel, key=lambda r: (list(PACKAGES).index(r["task"]), r["arm"], r["model"], r["run_id"])):
        run = load_run(r, a.runs)
        run["v"] = variants(run)
        runs.append(run)
    print(f"harness code {harness_hash()} (scheduler, estimate, model); {len(runs)} runs; CP-SAT on OR-Tools at {OR_PY}")
    soap = [j["len"] for r in runs if fam(r["task"]) == "g4n" for j in r["jobs"] if j["group"] == "soap" and j["full"]]
    soap_len = statistics.median(soap) if soap else 8650.0

    # every floor in one CP-SAT batch: each run's own needed work per reading, and each package's matrix
    inst, keys = [], []
    for r in runs:
        r["floor"], r["short"] = {}, {}
        for rd in rules(r["task"]):
            L, short = needed_lengths(r, rd)
            r["short"][rd] = short
            inst.append(L); keys.append(("run", r["run_id"], rd))
    for t in PACKAGES:
        for rd, L in needed_set(t, soap_len).items():
            inst.append(L); keys.append(("pkg", t, rd))
    sol = optimum_many(inst)
    opt, unproved = {}, []
    for k, o in zip(keys, sol):
        if k[0] == "run":
            next(r for r in runs if r["run_id"] == k[1])["floor"][k[2]] = o["opt"]
        else:
            opt[(k[1], k[2])] = o["opt"]
        if o["status"] != "OPTIMAL":
            unproved.append((k, o))
    print(f"floors: {len(sol)} instances, {sum(1 for o in sol if o['how'] != 'CP-SAT')} settled by LPT meeting the lower "
          f"bound, {sum(1 for o in sol if o['how'] == 'CP-SAT')} by CP-SAT; {len(sol) - len(unproved)} proved optimal"
          + (f"; not proved: {[(k[1], k[2], round(o['opt']), round(o['bound'])) for k, o in unproved]}" if unproved else ""))
    print("package optimum, the reference plan's jobs: " + "; ".join(
        f"{PACKAGES[t]}{'(' + rd + ')' if rd else ''} {o:,.0f} s" for (t, rd), o in opt.items())
        + f". SOAP full run {soap_len:,.0f} s (median of {len(soap)} runs)")

    # check: variant 1's idle reproduces shared_pool's (its 53 L0 runs: Haiku, Sonnet, Opus, Astra, Sol)
    sp = [r for r in runs if r["arm"] == "L0" and r["model"] in SP.MODELS]
    pool, _ = SP.load(a.runs, SP.MODELS)
    print(f"check, V1 idle as shared_pool: {len(sp)} L0 runs, idle {sum(r['idle'] for r in sp) / 3600:,.1f} GPU-h of "
          f"{sum(r['cap'] for r in sp) / 3600:,.1f}; shared_pool.py on the same runs, now: {len(pool)} tenants, idle "
          f"{sum(t['idle_s'] for t in pool) / 3600:,.1f} GPU-h of {sum(t['cap'] for t in pool) / 3600:,.1f}")

    cells = {}
    for r in runs:
        for rd in rules(r["task"]):
            r.setdefault("split", {})[rd] = split(r["v"], r["floor"][rd], rd)
            cells.setdefault((r["package"], rd, r["arm"], r["model"]), []).append(r)
    print("\nEach run against its own floor: the optimum of the needed work it ran (each needed record once). Columns are "
          "medians of variant / floor; the split is of the loss over the floor, summed over the cell's runs.\n")
    print("| package | arm | model | n | V1 as run | V1f agent order, no delay | V2 LPT, true lengths | V3 harness | "
          "V4 harness + cancel | V5 harness, needed only | loss (h, summed) | delay | ordering | unneeded, ruled out | "
          "unneeded, never ruled out | harness gap | short of the reading |\n" + "|---" * 17 + "|")
    for (pk, rd, arm, mdl), rs in cells.items():
        med = lambda k: statistics.median(r["v"][k] / r["floor"][rd] for r in rs)
        tot = lambda k: sum(r["split"][rd][k] for r in rs)
        loss = tot("loss")
        sh = " | ".join(f"{tot(k) / loss:.2f}" if loss > 0 else "-"
                        for k in ("delay", "ordering", "unneeded", "unruled", "harness_gap"))
        short = sum(1 for r in rs if r["short"][rd])
        print(f"| {pk}{' (' + rd + ')' if rd else ''} | {arm} | {mdl} | {len(rs)} | {med('V1'):.3f} | {med('V1f'):.3f} | "
              f"{med('V2'):.3f} | {med('V3'):.3f} | {med('V4' + rd):.3f} | {med('V5' + rd):.3f} | {loss / 3600:,.1f} | {sh} | "
              f"{short or ''} |")

    # pooled by package and family
    fams = (("Claude", ("haiku", "sonnet", "opus")), ("OpenAI", ("astra", "sol")))
    print("\n| package | arm | models | n | loss (h, summed) | delay | ordering | unneeded, ruled out | "
          "unneeded, never ruled out | harness gap |\n" + "|---" * 10 + "|")
    for pk in PACKAGES.values():
        for rd in ("", "a", "b"):
            for arm in ("L0", "S1"):
                for fl, ms in fams:
                    rs = [r for r in runs if r["package"] == pk and r["arm"] == arm and r["model"] in ms and rd in r["split"]]
                    if not rs:
                        continue
                    tot = lambda k: sum(r["split"][rd][k] for r in rs)
                    loss = tot("loss")
                    print(f"| {pk}{' (' + rd + ')' if rd else ''} | {arm} | {fl} | {len(rs)} | {loss / 3600:,.1f} | "
                          + " | ".join(f"{tot(k) / loss:.2f}" for k in ("delay", "ordering", "unneeded", "unruled",
                                                                           "harness_gap")) + " |")

    print("\nThe delay in two (L0, shares of the summed loss): idle, the agent's GPUs empty between its jobs (V1 - V1g), "
          "and placement, its own GPU per job against first come, first served (V1g - V1f).\n")
    print("| package | models | n | delay | idle | placement |\n|---|---|---|---|---|---|")
    for pk in PACKAGES.values():
        for rd in ("", "a", "b"):
            for fl, ms in fams:
                rs = [r for r in runs if r["package"] == pk and r["arm"] == "L0" and r["model"] in ms and rd in r["split"]]
                if not rs:
                    continue
                tot = lambda k: sum(r["split"][rd][k] for r in rs)
                L = tot("loss")
                print(f"| {pk}{' (' + rd + ')' if rd else ''} | {fl} | {len(rs)} | {tot('delay') / L:.2f} | "
                      f"{tot('idle') / L:.2f} | {tot('placement') / L:.2f} |")

    # a planning paper's row: the package's whole matrix, true lengths, the rule applied by an oracle judge
    print("\nThe package's whole matrix, true lengths, the rule applied as results arrive (no agent). Each over the "
          "optimum of the reference plan's jobs.\n")
    print("| package | FCFS, file order | harness (calibration, critical path) | LPT, true lengths | optimum (CP-SAT) | "
          "harness, prior 1 / 5 s per example | A3 runs as run, over each run's floor (n) |\n|---|---|---|---|---|---|---|")
    global SEC_PER_EX
    for t in PACKAGES:
        for rd, rule in rules(t).items():
            o = opt[(t, rd)]
            jobs = instance(t, soap_len)
            owed = [j["id"] for j in jobs]
            res = {}
            for name, kw in (("fcfs", dict(rank={j["id"]: j["order"] for j in jobs})), ("harness", dict(mode="harness")),
                             ("lpt", dict(rank={j["id"]: -j["len"] for j in jobs}))):
                rp = Replay([dict(j) for j in jobs], rule=rule, **kw); rp.run(); res[name] = rp.deliverable(owed)
            sens = []
            for spe in (1.0, 5.0):
                keep, SEC_PER_EX = SEC_PER_EX, spe
                rp = Replay([dict(j) for j in instance(t, soap_len)], mode="harness", rule=rule); rp.run()
                sens.append(rp.deliverable(owed) / o)
                SEC_PER_EX = keep
            a3 = [r["v"]["V1"] / r["floor"][rd] for r in runs if r["task"] == t and r["arm"] == "A3"]
            print(f"| {PACKAGES[t]}{' (' + rd + ')' if rd else ''} | {ratio(res['fcfs'], o)} | {ratio(res['harness'], o)} | "
                  f"{ratio(res['lpt'], o)} | 1.000 ({o:,.0f} s) | {sens[0]:.3f} / {sens[1]:.3f} | "
                  + (", ".join(f"{x:.3f}" for x in sorted(a3)) + f" ({len(a3)})" if a3 else "none") + " |")
    # G4: LPT and the harness with and without the rule, and LPT told the needed set
    t = "mls-kv-gate"
    o = opt[(t, "")]
    jobs = instance(t, soap_len)
    owed = [j["id"] for j in jobs]
    ref = T.reference_records(t)
    recs = {x["job_id"]: set(x.get("records") or []) for x in
            json.load(open(os.path.join(EXP, "gpu-tasks", "registry", "tasks", f"{t}.json")))["jobs"]}
    need = [j for j in jobs if recs.get(j["id"], set()) & ref]
    lpt = lambda js: {j["id"]: -j["len"] for j in js}
    out = []
    for name, js, kw, rule in (("LPT, true lengths, no rule", jobs, dict(rank=lpt(jobs)), None),
                               ("LPT, true lengths, plus the rule", jobs, dict(rank=lpt(jobs)), rules(t)[""]),
                               (f"LPT, true lengths, told the needed set ({len(need)} jobs)", need, dict(rank=lpt(need)), None),
                               ("the harness, no rule", jobs, dict(mode="harness"), None),
                               ("the harness, plus the rule", jobs, dict(mode="harness"), rules(t)[""])):
        rp = Replay([dict(j) for j in js], rule=rule, **kw); rp.run()
        dl = rp.deliverable([j["id"] for j in js])
        out.append(f"{name} {dl / o:.3f} ({dl:,.0f} s)")
    print("\nG4, over the optimum: " + "; ".join(out))

    if a.json:
        json.dump(dict(harness=harness_hash(), opt={f"{t}{rd}": o for (t, rd), o in opt.items()}, soap_len=soap_len,
                       runs=[{k: v for k, v in r.items() if k not in ("jobs", "order")} for r in runs]),
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    main()
