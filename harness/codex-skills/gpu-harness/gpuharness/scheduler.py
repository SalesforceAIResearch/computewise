"""The scheduling policy: decision-aware critical-path list scheduling.

Problem. A decision is to be made from a set of jobs on m identical GPUs. The agent writes the jobs as a graph: each
job belongs to a candidate (or to none), has a role (check, needed, speculative), and may wait for other jobs to
finish (`after`) or for a check's judgement to come in (`needs_pass`). Lengths are uncertain: the plan gives priors,
and the machine measures the rest as jobs run (`estimate.Estimator`). The agent judges; this module never does. A
rejection removes the candidate's whole subtree. The objective is the time until every candidate is decided and
every job the answer rests on has finished, with no GPU-time spent on work a cheaper observation could have ruled
out.

Policy. At every tick, for every free GPU, start the runnable job of highest priority:

  1. Enablers first: a check of a reference candidate. Every other candidate is judged against the reference, so
     its first observation unblocks all of their judgements. When the reference's jobs declare milestones, its first
     SPREAD_RUNS (3) runs of each such kind are enablers too: a single run of a candidate settles a mean rule only
     against a run-to-run spread, and the reference's is the one every candidate shares (G-k-haiku-A4: the baseline's
     second run started 5,800 s in, after every candidate's step-1000 line). Plans without milestones are unchanged.

  2. Everything else that is runnable, by its decision-aware critical path: the length of the longest chain of work
     that would still have to run after this job starts, counting work gated on a check as if the check passes
     (optimistic), plus a judgement latency on each gate:

         chain(j) = remaining(j) + tail(j) + max( max_{c after j} chain(c),
                                                  max_{c gated by j} (judge + chain(c)) )

     A check inherits the long run it gates, so a candidate's cheap checks run before an independent job that is
     shorter than the candidate's eventual long run, and the job that the finish time depends on starts first. With
     no checks this is Graham's critical-path list scheduling; with independent jobs it is longest-processing-time.

  3. Among the checks of one candidate (equal chains), Mitten's ratio rule for testing a series system: the check
     with the highest probability of rejection per second of cost first. A rejection cancels the candidate's other
     checks, so this order minimises the expected cost of reaching its verdict. The agent may give each check a prior
     `p_reject`; otherwise checks are equally likely to fail and the cheapest goes first.

  4. Calibration. A group of jobs (a method, a setup) whose pace nobody has measured yet, and that has at least two
     jobs left, gets its shortest runnable job run first (between enablers and ordinary work), whatever the priors
     say about the spread: the priors are what calibration corrects, so they must not be able to switch it off.
     The reference is calibrated per workload as well: its job of each workload it has not measured runs early. Its
     factor is pinned at 1, so what it shows (a measurement, or a running job past its prediction) is evidence about
     the workload, and a slow method's job of a long workload ranks first as soon as the method is measured. It costs one short job, and its measured pace rescales every other job of the group through the
     duration model, so a slow method's long job is found and started early instead of last. This is the value of a
     cheap measurement that can reorder an expensive decision; it is off for groups that already have a measurement.

     Calibration yields to a job that bounds the finish: one whose chain is at least the remaining GPU work divided
     by the GPUs. The finish is at least the larger of the two (the bound below), so when the chain is the larger,
     every second that job waits is a second on the finish, while calibration only reorders the work that fits
     beside it (G-k2lp-sonnet-A4: the longest job, predicted six times the agent's estimate, waited behind twenty
     calibration starts).

  5. Speculative work runs only on a GPU that no other runnable job wants: jobs the agent marked `speculative`, and,
     if the plan enables `speculate`, jobs whose gate has not passed yet, ranked by their chance of being needed
     times their chain. A speculative job is preempted the moment a non-speculative job is runnable and no GPU is
     free. If its gate passes while it runs, it becomes ordinary needed work and keeps running.

  6. The judgement hold. When a check can be judged (it has finished and its reference result is in) and gates a
     job whose chain is longer than every runnable job's by more than `hold_s`, one free GPU stays free for the agent's
     pass, for at most `hold_s` (default 60 s) from the moment the check became judgeable. Once the pass comes, the
     longest job it opened outranks calibration for `hold_s`, so the GPU goes to the job it was held for. A1's boxes passed checks
     5-14 s after that moment (11 passes in three runs). Without it, the freed GPU goes at once to the best
     runnable job, and the long job the check gates waits for the next GPU to free, which can be hours. The hold is a
     deliberate, bounded break of work conservation: it costs at most `hold_s` GPU-seconds per such check, and only
     checks that gate the longest chain earn one.

Calibration (4) and the judgement hold (6) are one mechanism: the long pole starts on the first free GPU once what
identifies it is known, namely its method's pace, its workload's length (through the reference), and its gate's pass.
Neither half is a separate feature. Workload calibration raises the reference's long workloads in the queue, and
without the hold one of them takes the GPU a check frees, which delays the gated long pole (G1 Sonnet A1's plan,
replayed: 1.014x LPT without the hold, 1.001x with it).

CPU-only jobs (a plan's `"gpu": 0`: data preparation, a CPU metric) hold no GPU. Each runnable one starts at once, best
first, up to the board's `cpu_slots`, and none of them enters a GPU's queue, calibration, a judgement hold or
preemption. The harness doesn't model cores: the agent sizes each CPU job's workers to its share of the machine.

Guarantees, given the agent's judgements and the harness's tick period `tick`:

  * Work-conserving, except for judgement holds: no GPU stays free for longer than one tick while a runnable job
    exists, unless it is held for a check's judgement, which ends within `hold_s` of the check's end.
  * Speculation never delays real work by more than one tick, because a speculative job yields its GPU.
  * Bounded cancellation: every job of a rejected candidate is signalled within one tick of the rejection, and its
    GPU is released within `kill_grace + 2 * tick` (the whole process group is terminated, then killed).
  * CPU-only jobs never hold a GPU, so they never delay a runnable GPU job; they enter the bound below only through the
    chains (`after`) of the GPU jobs that wait on them, not its work term.
  * The list-scheduling bound on needed work. For the jobs that run, with their actual lengths p_j, total work
    W = sum p_j and longest dependency chain L, the makespan of the needed work satisfies
        T <= W / m + (1 - 1/m) * L,
    since at any moment either every GPU is busy or a job on the longest remaining chain is running (Graham 1966).
    Gates add the agent's judgement latencies to L.

What is not guaranteed: the estimates are estimates, so the order is only as good as the priors and the
measurements; and a job that detaches itself from its process group escapes cancellation (the launcher also sweeps
descendants it can see).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .estimate import Estimator
from .model import Candidate, Board, CandState, Job, JobState, Role, TERMINAL

JUDGE_S = 60.0           # assumed latency of an agent's judgement on a gate, seconds
DEFAULT_P_REJECT = 0.5
TIE = 0.01               # predictions within 1% are a tie: flat priors make every job of a group tie
SPREAD_RUNS = 3          # runs of one method that measure its run-to-run spread at a milestone


@dataclass
class Policy:
    speculate: bool = False            # run gated work speculatively on GPUs that would otherwise idle
    preempt_speculative: bool = True   # a speculative job yields its GPU to real work
    judge_s: float = JUDGE_S
    calibrate: bool = True             # run an unmeasured group's short job first (see 4. above)
    hold_s: float = JUDGE_S            # keep a GPU free this long for a check's judgement (see 6. above); 0 = never
    gate_on_checks: bool = True        # a candidate's non-check jobs wait for its checks unless they say otherwise

    @classmethod
    def from_json(cls, d: Optional[dict]) -> "Policy":
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Ranked:
    job: Job
    klass: float          # 3 enabler, 2.5 calibration, 2 ordinary, 1 speculative (marked), 0 speculative (gate open)
    chain: float
    index: float          # ratio rule p_reject / cost for checks; length for others
    speculative: bool
    why: str = ""

    def key(self) -> Tuple[float, float, float, float]:
        return (self.klass, round(self.chain, 3), self.index, -self.job.added_at)


@dataclass
class Decision:
    starts: List[Tuple[str, int]] = field(default_factory=list)   # (job id, GPU)
    cpu_starts: List[str] = field(default_factory=list)            # CPU-only jobs to start now (no GPU slot)
    hold: List[Tuple[str, int, float]] = field(default_factory=list)  # (check id, GPU kept free, until)
    preempt: List[str] = field(default_factory=list)              # speculative jobs to stop
    ranked: List[Ranked] = field(default_factory=list)            # the queue, best first, for status views


# readiness ------------------------------------------------------------------------------------------------------------

def gates(board: Board, job: Job, policy: Policy) -> List[str]:
    """The checks whose judgement this job waits for."""
    if job.needs_pass is not None:
        return list(job.needs_pass)
    cand = board.candidates.get(job.candidate or "")
    if not policy.gate_on_checks or job.role is not Role.NEEDED or cand is None or cand.reference:
        return []
    checks = [j for j in board.jobs.values() if j.candidate == job.candidate and j.role is Role.CHECK and j.id != job.id]
    # a cancelled check that a live check of the same kind has replaced (a re-plan: cancel, then `gpuh add`) no
    # longer gates: G-fyld-haiku-A4-r1b's trainings waited for good on the cancelled checks their re-added ones
    # replaced. A cancelled check with no replacement still gates, so cancel-then-retry behaves as before.
    live_kinds = {j.kind for j in checks if j.state is not JobState.CANCELLED}
    return [j.id for j in checks if not (j.state is JobState.CANCELLED and j.kind in live_kinds)]


def readiness(board: Board, job: Job, policy: Policy) -> Tuple[bool, bool, str]:
    """(runnable now, may run speculatively, reason it waits)."""
    if job.terminal or job.active:
        return False, False, ""
    cand = board.candidates.get(job.candidate or "")
    if cand is not None and cand.state is CandState.REJECTED:
        return False, False, f"candidate {cand.id} rejected"
    for d in job.after:
        dep = board.jobs.get(d)
        if dep is None:
            return False, False, f"unknown dependency {d}"
        if dep.state is JobState.DONE:
            continue
        if dep.terminal:
            return False, False, f"dependency {d} {dep.state.value}"
        return False, False, f"after {d}"
    open_gates = [g for g in gates(board, job, policy) if not board.jobs.get(g) or not board.jobs[g].passed]
    if job.role is Role.SPECULATIVE:
        return False, True, "speculative: only on an idle GPU"
    if open_gates:
        failed = [g for g in open_gates if board.jobs.get(g) and board.jobs[g].state in
                  (JobState.FAILED, JobState.DIED, JobState.CANCELLED)]
        if failed:
            return False, False, f"gate {failed[0]} {board.jobs[failed[0]].state.value}"
        return False, policy.speculate, "awaiting judgement of " + ", ".join(open_gates)
    return True, False, ""


# chains -------------------------------------------------------------------------------------------------------------

def remaining_s(job: Job, est: Estimator, now: float) -> float:
    if job.terminal:
        return 0.0
    total = est.predict(job)
    if job.pace and job.pace.projected_s:
        total = job.pace.projected_s
    if job.active and job.started_at:
        return max(0.0, total - (now - job.started_at))
    return total


def chains(board: Board, est: Estimator, now: float, policy: Policy) -> Dict[str, float]:
    after_children: Dict[str, List[str]] = {j: [] for j in board.jobs}
    gate_children: Dict[str, List[str]] = {j: [] for j in board.jobs}
    for j in board.jobs.values():
        for d in j.after:
            if d in after_children:
                after_children[d].append(j.id)
        for g in gates(board, j, policy):
            if g in gate_children:
                gate_children[g].append(j.id)
    memo: Dict[str, float] = {}
    visiting: set = set()

    def live(jid: str) -> bool:
        job = board.jobs[jid]
        cand = board.candidates.get(job.candidate or "")
        return not job.terminal and not (cand is not None and cand.state is CandState.REJECTED)

    def chain(jid: str) -> float:
        if jid in memo:
            return memo[jid]
        if jid in visiting:          # a cycle slipped past validation: break it
            return 0.0
        visiting.add(jid)
        job = board.jobs[jid]
        own = remaining_s(job, est, now) + (job.tail_s or 0.0) if live(jid) else 0.0
        nxt = 0.0
        for c in after_children[jid]:
            if live(c):
                nxt = max(nxt, chain(c))
        for c in gate_children[jid]:
            if live(c) and c not in after_children[jid]:
                nxt = max(nxt, policy.judge_s + chain(c))
        visiting.discard(jid)
        memo[jid] = own + nxt
        return memo[jid]

    for jid in board.jobs:
        chain(jid)
    return memo


# the decision -------------------------------------------------------------------------------------------------------

def _p_pass(board: Board, job: Job, policy: Policy) -> float:
    p = 1.0
    for g in gates(board, job, policy):
        gj = board.jobs.get(g)
        if gj is not None and not gj.passed:
            p *= 1.0 - (gj.p_reject if gj.p_reject is not None else DEFAULT_P_REJECT)
    return p


def calibration_jobs(board: Board, est: Estimator, now: float, policy: Policy) -> Dict[str, float]:
    """job id -> predicted length, for each unmeasured group's short calibration job."""
    if not policy.calibrate:
        return {}
    groups: Dict[str, List[Job]] = {}
    for j in board.jobs.values():
        if j.terminal or j.cpu:          # calibration measures GPU pace; CPU-only jobs start anyway and measure nothing
            continue
        cand = board.candidates.get(j.candidate or "")
        if cand is not None and cand.state is CandState.REJECTED:
            continue
        groups.setdefault(j.group or j.candidate or "_", []).append(j)
    if len(groups) < 2:
        return {}
    out: Dict[str, float] = {}
    # a tie among a group's shortest jobs goes to a kind no other unmeasured group is running, so one wave measures
    # several kinds at once (G1 replay, flat priors: 1.167x LPT without this, 1.023x with it)
    taken = {j.kind for j in board.jobs.values() if j.active and not est.has_evidence(j)}
    for g, jobs in groups.items():
        if any(j.active for j in jobs) or est.has_evidence(jobs[0]):
            continue
        runnable = [j for j in jobs if readiness(board, j, policy)[0]]
        # whatever the priors' spread: they are the agent's guesses, and flat ones (one number per workload, copied to
        # every method) hide exactly the slow method calibration exists to find (G1 Haiku A1: squat unmeasured for 7.8 h)
        if not runnable or len(jobs) < 2:
            continue
        low = min(est.predict(j) for j in runnable)
        tied = [j for j in runnable if est.predict(j) <= low * (1 + TIE)]
        short = next((j for j in tied if j.kind not in taken), tied[0])
        taken.add(short.kind)
        out[short.id] = est.predict(short)
    # workload calibration: the reference's job of each kind it has not measured runs early. The reference's factor
    # is pinned, so its running lower bound and its measurement are evidence about the kind itself; a slow method's
    # long workload is then predicted long as soon as the method is measured (G1 Haiku A1's replay: no GSM8K had been
    # measured when squat was, so its GSM8K ranked behind its repobench). Its jobs are needed in any case.
    refs: Dict[str, List[Job]] = {}
    for jobs in groups.values():
        for j in jobs:
            cand = board.candidates.get(j.candidate or "")
            if j.kind and cand is not None and cand.reference:
                refs.setdefault(j.kind, []).append(j)
    for kind, js in refs.items():
        group = js[0].group or js[0].candidate or "_"
        if est.cell_has_evidence(group, kind) or any(j.active or j.id in out for j in js):
            continue
        runnable = [j for j in js if readiness(board, j, policy)[0]]
        if runnable:
            pick = min(runnable, key=lambda j: est.predict(j))
            out[pick.id] = est.predict(pick)
    return out


def spread_jobs(board: Board) -> set:
    """The reference's jobs that measure its run-to-run spread at a milestone: for each kind of the reference's jobs
    that declare milestones, its first SPREAD_RUNS of them, counting those already started. A candidate's single run
    can settle a mean rule only against a spread, and the reference's is the one every candidate shares (G-k-haiku-A4:
    the baseline's second run started 5,800 s in, so no branch could see a spread until hours after its candidates'
    step-1000 lines). These are needed jobs in any case; only their order changes. Empty for plans without milestones."""
    out: set = set()
    by_kind: Dict[str, List[Job]] = {}
    for j in board.jobs.values():
        cand = board.candidates.get(j.candidate or "")
        if j.milestones and cand is not None and cand.reference and j.role is not Role.SPECULATIVE and not j.screen:
            by_kind.setdefault(j.kind or "_", []).append(j)
    for js in by_kind.values():
        started = [j for j in js if j.started_at or j.state in TERMINAL]
        left = SPREAD_RUNS - len(started)
        # in plan order: sorting by id put run 10 before run 2 (G-k2p-sonnet-A4: the spread runs were r10 and r2)
        for j in [j for j in js if j not in started][:max(0, left)]:
            out.add(j.id)
    return out


def bounding_jobs(board: Board, est: Estimator, now: float, policy: Policy, ch: Dict[str, float]) -> set:
    """Jobs whose chain is at least the remaining GPU work divided by the GPUs: the finish waits on them second for
    second (see 4. above). The work counts every live GPU job that isn't speculative, gated or not, as the chains do."""
    m = len(board.gpus) or 1
    live = []
    for j in board.jobs.values():
        cand = board.candidates.get(j.candidate or "")
        if j.terminal or j.cpu or j.role is Role.SPECULATIVE or (cand is not None and cand.state is CandState.REJECTED):
            continue
        live.append(j)
    work = sum(remaining_s(j, est, now) for j in live)
    if work <= 0:
        return set()
    return {j.id for j in live if not j.active and ch.get(j.id, 0.0) >= work / m}


def releases_gpu_work(board: Board, cpu_jobs: List[Job], policy: Policy) -> set:
    """The runnable CPU jobs whose end (and, for a check, its pass) is all that stands between some GPU job and a
    start: every other dependency of that GPU job is done, and every other gate is passed or has finished and awaits
    only the agent's pass (seconds: on G-fyld-haiku-A4-r1b each prep's pass came 7-10 s after it ended, and the next
    preparation had taken the CPU slot in that instant)."""
    ids = {j.id for j in cpu_jobs}
    out: set = set()
    for g in board.jobs.values():
        if g.cpu or g.terminal or g.active or g.role is Role.SPECULATIVE:
            continue
        cand = board.candidates.get(g.candidate or "")
        if cand is not None and cand.state is CandState.REJECTED:
            continue
        pending = [d for d in g.after if board.jobs.get(d) is None or board.jobs[d].state is not JobState.DONE]
        pending += [x for x in gates(board, g, policy) if x not in pending
                    and (board.jobs.get(x) is None or not board.jobs[x].passed)
                    and not (board.jobs[x].state is JobState.DONE)]      # finished, awaiting only its pass
        if len(pending) == 1 and pending[0] in ids:
            out.add(pending[0])
    return out


def holds(board: Board, est: Estimator, now: float, policy: Policy, ranked: List[Ranked],
          ch: Dict[str, float]) -> List[Tuple[str, float]]:
    """Checks that just finished and gate work longer than anything runnable: (check id, hold until).

    A check's judgement takes the agent a few seconds. Without a hold, the GPU the check frees goes at once to the
    best runnable job, and the long job the check gates waits for the next GPU to free, which can be hours (G1 Sonnet
    A1: squat's GSM8K, gated on its check, waited 1,790 s behind a longbench_v2 that took the check's GPU)."""
    if policy.hold_s <= 0:
        return []
    top = max((r.chain for r in ranked if not r.speculative and not r.job.cpu), default=0.0)   # what could take the GPU
    out = []
    for j in board.jobs.values():
        if j.role is not Role.CHECK or j.state is not JobState.DONE or j.passed or not j.ended_at:
            continue
        cand = board.candidates.get(j.candidate or "")
        if cand is None or cand.reference or cand.state is not CandState.OPEN:
            continue
        ready = _judgeable_at(board, j)
        if ready is None or now >= ready + policy.hold_s:
            continue
        until = ready + policy.hold_s
        gated = [c for c in board.jobs.values()
                 if not c.terminal and j.id in gates(board, c, policy)]
        if gated and max(ch.get(c.id, 0.0) for c in gated) > top + policy.hold_s:
            out.append((j.id, until))
    return out


def just_opened(board: Board, now: float, policy: Policy, ch: Dict[str, float]) -> set:
    """Jobs a judgement hold was kept for: their gates have all passed within `hold_s`, and each is the longest chain
    its check gated. They outrank calibration for that window, so a short calibration job can't take the held GPU."""
    if policy.hold_s <= 0:
        return set()
    out = set()
    for j in board.jobs.values():
        if j.terminal or j.active or j.role is not Role.NEEDED:
            continue
        gs = [board.jobs[g] for g in gates(board, j, policy) if g in board.jobs]
        if not gs or not all(g.passed and g.judged_at for g in gs):
            continue
        if now - max(g.judged_at for g in gs) > policy.hold_s:
            continue
        sib = [c for c in board.jobs.values() if not c.terminal and not c.active
               and any(g.id in gates(board, c, policy) for g in gs)]
        if ch.get(j.id, 0.0) >= max(ch.get(c.id, 0.0) for c in sib):
            out.add(j.id)
    return out


def _judgeable_at(board: Board, check: Job) -> Optional[float]:
    """When the branch could first judge the check: its end, or its reference's result if that came later (A1's
    boxes: passes came 5-14 s after both were in). None while the reference's job of its kind is unfinished."""
    refs = [j for j in board.jobs.values() if j.kind == check.kind and j.id != check.id
            and (board.candidates.get(j.candidate or "") or Candidate("_")).reference]
    if not refs:
        return check.ended_at
    done = [j.ended_at for j in refs if j.state is JobState.DONE and j.ended_at]
    return max(check.ended_at or 0.0, min(done)) if done else None


def rank(board: Board, est: Estimator, now: float, policy: Policy) -> List[Ranked]:
    ch = chains(board, est, now, policy)
    cal = calibration_jobs(board, est, now, policy)
    opened = just_opened(board, now, policy, ch)
    spread = spread_jobs(board)
    bound = bounding_jobs(board, est, now, policy, ch)
    out: List[Ranked] = []
    for job in board.jobs.values():
        runnable, spec_ok, reason = readiness(board, job, policy)
        if not runnable and not spec_ok:
            continue
        cand = board.candidates.get(job.candidate or "")
        cost = max(1.0, est.predict(job))
        if runnable:
            if job.id in opened:
                klass, why = 2.75, "the long job a GPU was held for: its check has just passed"
            elif job.role is Role.CHECK and cand is not None and cand.reference:
                klass, why = 3.0, "enabler: the reference's check"
            elif job.id in spread:
                klass, why = 3.0, "enabler: the reference's run-to-run spread at a milestone"
            elif job.id in bound:
                klass, why = 2.6, "bounds the finish: its chain is at least the remaining GPU work per GPU"
            elif job.id in cal:
                klass, why = 2.5, "calibration: the group's first measurement"
                # the cheapest measurement first: it is the information, not the work, that is wanted
                out.append(Ranked(job, klass, -cost, 0.0, False, why))
                continue
            else:
                klass, why = 2.0, "critical path"
            if job.role is Role.CHECK:
                p = job.p_reject if job.p_reject is not None else DEFAULT_P_REJECT
                index = p / cost
                why += f", check p_reject {p:.2f} per {cost:.0f}s"
            else:
                index = cost
            out.append(Ranked(job, klass, ch.get(job.id, cost), index, False, why))
        else:
            p = _p_pass(board, job, policy)
            marked = job.role is Role.SPECULATIVE
            klass = 1.0 if marked else 0.0
            out.append(Ranked(job, klass, p * ch.get(job.id, cost), cost, True,
                              ("marked speculative" if marked else f"speculative: gate open, P(needed) {p:.2f}")))
    out.sort(key=lambda r: r.key(), reverse=True)
    return out


def decide(board: Board, est: Estimator, now: float, policy: Policy) -> Decision:
    """Which jobs to start on which free GPUs now, which CPU-only jobs to start, and which speculative jobs to preempt.

    First tells the estimator how long each running job without a pace has lasted (a lower bound on its length), so a
    workload whose jobs all overrun the plan's prior is seen long before any of them ends."""
    est.set_reference(j.group or j.candidate or "_" for j in board.jobs.values()
                      if (board.candidates.get(j.candidate or "") or Candidate("_")).reference)
    for j in board.jobs.values():   # what the machine shows of the running jobs: each has lasted at least this long
        if j.active and j.started_at and not (j.pace and j.pace.projected_s):
            est.running(j, now - j.started_at)
    ranked = rank(board, est, now, policy)
    dec = Decision(ranked=ranked)
    # CPU-only jobs take no GPU: each runnable one starts at once, best first, within the board's CPU slots. A CPU job
    # whose end alone would let GPU work start while GPUs are free goes first (G-fyld-haiku-A4-r1b, cpu_slots 1: each
    # recipe's 1-s retention count, the last gate of its four trainings, waited behind every later preparation)
    cpu_free = board.free_cpu_slots()
    cpu = [r for r in ranked if r.job.cpu and not r.speculative]
    if cpu_free > 0 and cpu:
        first = releases_gpu_work(board, [r.job for r in cpu], policy)
        rival = next((r.job for r in cpu if r.job.id not in first), None)
        # ... when a GPU is free, or will be (by the running GPU jobs' predictions) before the CPU job that would
        # otherwise take the slot ends: a long preparation must not hold the slot past the GPUs' freeing while a
        # 1-s count waits (the same run: the reference's trainings freed all four GPUs at 1,461 s; the counts
        # waited behind a preparation started at 1,371 s, until 1,742 s)
        gpu_free_in = 0.0 if board.free_gpus() else min(
            (remaining_s(j, est, now) for j in board.jobs.values() if j.active and not j.cpu), default=float("inf"))
        if first and rival is not None and gpu_free_in <= remaining_s(rival, est, now):
            cpu = sorted(cpu, key=lambda r: r.job.id not in first)     # stable: the ranking holds within each group
            for r in cpu:
                if r.job.id in first:
                    r.why = "releases GPU work for GPUs free by the time it would wait for; " + r.why
    for r in cpu:
        if cpu_free > 0:
            dec.cpu_starts.append(r.job.id)
            cpu_free -= 1
    ranked_gpu = [r for r in ranked if not r.job.cpu]
    real = [r for r in ranked_gpu if not r.speculative]
    spec = [r for r in ranked_gpu if r.speculative]
    free = list(board.free_gpus())
    for check, until in holds(board, est, now, policy, ranked, chains(board, est, now, policy)):
        if not free:
            break
        dec.hold.append((check, free.pop(0), until))
    for r in real:
        if not free:
            break
        dec.starts.append((r.job.id, free.pop(0)))
    waiting_real = len(real) - len(dec.starts)
    if waiting_real > 0 and policy.preempt_speculative:
        running_spec = sorted(
            (j for j in board.jobs.values()
             if j.state is JobState.RUNNING and not j.cpu and _is_speculative_now(board, j, policy)),
            key=lambda j: _p_pass(board, j, policy))
        for j in running_spec[:waiting_real]:
            dec.preempt.append(j.id)
    for r in spec:
        if not free:
            break
        dec.starts.append((r.job.id, free.pop(0)))
    return dec


def _is_speculative_now(board: Board, job: Job, policy: Policy) -> bool:
    if job.role is Role.SPECULATIVE:
        return True
    return any(not board.jobs[g].passed for g in gates(board, job, policy) if g in board.jobs)


def explain(board: Board, est: Estimator, now: float, policy: Policy, limit: int = 20) -> List[str]:
    lines = []
    for i, r in enumerate(rank(board, est, now, policy)[:limit], 1):
        lines.append(f"{i:2d}. {r.job.id:28s} chain {r.chain:7.0f}s  {r.why}")
    return lines


def subtree(board: Board, candidate: str) -> List[Job]:
    """Every job of a candidate that has not ended: what a rejection cancels."""
    return [j for j in board.jobs.values() if j.candidate == candidate and not j.terminal]


def still_needed(board: Board) -> Iterable[Job]:
    """Jobs whose results the answer may still rest on."""
    for j in board.jobs.values():
        cand = board.candidates.get(j.candidate or "")
        if j.terminal:
            continue
        if cand is not None and cand.state is CandState.REJECTED:
            continue
        yield j
