"""The policy, on boards built by hand: order, gates, speculation, and the list-scheduling bound."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpuharness.estimate import Estimator  # noqa: E402
from gpuharness.model import Board, CandState, Candidate, Job, JobState, Role  # noqa: E402
from gpuharness.scheduler import Policy, chains, decide, rank, readiness  # noqa: E402


def board(jobs, cands=(), gpus=4):
    b = Board(gpus=list(range(gpus)))
    b.gpu_job = {g: None for g in b.gpus}
    for c in cands:
        b.candidates[c.id] = c
    for j in jobs:
        b.jobs[j.id] = j
    return b


def est_for(b):
    e = Estimator()
    e.load(b.jobs.values())
    return e


class Order(unittest.TestCase):
    def test_longest_first_for_independent_jobs(self):
        b = board([Job(id=f"j{i}", command="x", estimate_s=s) for i, s in enumerate([60, 300, 120, 30, 600])])
        d = decide(b, est_for(b), 0, Policy(calibrate=False))
        self.assertEqual([jid for jid, _ in d.starts], ["j4", "j1", "j2", "j0"])

    def test_critical_path_beats_a_long_independent_job(self):
        # a 60-s job that a 500-s job waits on goes before an independent 400-s job
        b = board([Job(id="a", command="x", estimate_s=60), Job(id="b", command="x", estimate_s=500, after=["a"]),
                   Job(id="c", command="x", estimate_s=400)], gpus=1)
        d = decide(b, est_for(b), 0, Policy())
        self.assertEqual(d.starts[0][0], "a")

    def test_reference_check_first(self):
        cands = [Candidate("ref", reference=True), Candidate("m")]
        b = board([Job(id="m_long", command="x", candidate="m", role=Role.NEEDED, estimate_s=9000, needs_pass=[]),
                   Job(id="ref_check", command="x", candidate="ref", role=Role.CHECK, estimate_s=100)], cands, gpus=1)
        d = decide(b, est_for(b), 0, Policy())
        self.assertEqual(d.starts[0][0], "ref_check")

    def test_check_inherits_the_long_run_it_gates(self):
        cands = [Candidate("m")]
        b = board([Job(id="m_check", command="x", candidate="m", role=Role.CHECK, estimate_s=100),
                   Job(id="m_long", command="x", candidate="m", role=Role.NEEDED, estimate_s=5000),
                   Job(id="other", command="x", estimate_s=3000)], cands, gpus=1)
        ch = chains(b, est_for(b), 0, Policy())
        self.assertGreater(ch["m_check"], ch["other"])
        self.assertEqual(decide(b, est_for(b), 0, Policy()).starts[0][0], "m_check")

    def test_ratio_rule_among_one_candidates_checks(self):
        cands = [Candidate("m")]
        b = board([Job(id="cheap_sure", command="x", candidate="m", role=Role.CHECK, estimate_s=100, p_reject=0.5),
                   Job(id="dear_unsure", command="x", candidate="m", role=Role.CHECK, estimate_s=100, p_reject=0.1),
                   Job(id="m_long", command="x", candidate="m", role=Role.NEEDED, estimate_s=1000)], cands, gpus=1)
        self.assertEqual(decide(b, est_for(b), 0, Policy()).starts[0][0], "cheap_sure")


class Calibration(unittest.TestCase):
    def test_unmeasured_group_runs_its_short_job_first(self):
        jobs = []
        for g in ("fast", "slow"):
            jobs.append(Job(id=f"{g}_short", command="x", group=g, kind="short", estimate_s=100))
            jobs.append(Job(id=f"{g}_long", command="x", group=g, kind="long", estimate_s=7000))
        jobs.append(Job(id="other_long", command="x", group="other", kind="long", estimate_s=7200))
        b = board(jobs, gpus=2)
        starts = [j for j, _ in decide(b, est_for(b), 0, Policy()).starts]
        self.assertEqual(sorted(starts), ["fast_short", "slow_short"])
        # once measured, the slow group's long job is predicted long and goes first
        e = est_for(b)
        e.measured(b.jobs["fast_short"], 100)
        e.measured(b.jobs["slow_short"], 400)
        for jid in ("fast_short", "slow_short"):
            b.jobs[jid].state = JobState.DONE
        self.assertEqual(decide(b, e, 0, Policy()).starts[0][0], "slow_long")

    def test_flat_priors_still_calibrate(self):
        # G1 Haiku A1: one prior per kind copied to every method, no job 4x another, so A1 never calibrated at all
        jobs = []
        for g in ("ref", "slow"):
            for k, s in (("qa", 1200), ("code", 1500), ("math", 1200), ("long", 2400)):
                jobs.append(Job(id=f"{g}_{k}", command="x", group=g, kind=k, estimate_s=s))
        b = board(jobs, gpus=2)
        starts = [j for j, _ in decide(b, est_for(b), 0, Policy()).starts]
        # both unmeasured groups calibrate: one job each, and on different kinds so one wave measures two kinds
        self.assertEqual(len(starts), 2)
        self.assertEqual({j.split("_")[0] for j in starts}, {"ref", "slow"})
        self.assertNotEqual(starts[0].split("_", 1)[1], starts[1].split("_", 1)[1])

    def test_off(self):
        jobs = [Job(id="a_s", command="x", group="a", estimate_s=100), Job(id="a_l", command="x", group="a", estimate_s=9000),
                Job(id="b_s", command="x", group="b", estimate_s=100), Job(id="b_l", command="x", group="b", estimate_s=9000)]
        b = board(jobs, gpus=2)
        starts = sorted(j for j, _ in decide(b, est_for(b), 0, Policy(calibrate=False)).starts)
        self.assertEqual(starts, ["a_l", "b_l"])


class Gates(unittest.TestCase):
    def setUp(self):
        self.cands = [Candidate("ref", reference=True), Candidate("m")]
        self.jobs = [Job(id="ref_a", command="x", candidate="ref", role=Role.CHECK, estimate_s=60),
                     Job(id="m_a", command="x", candidate="m", role=Role.CHECK, estimate_s=60),
                     Job(id="m_b", command="x", candidate="m", role=Role.NEEDED, estimate_s=600)]

    def test_needed_waits_for_its_candidates_checks(self):
        b = board(self.jobs, self.cands)
        runnable, _, why = readiness(b, b.jobs["m_b"], Policy())
        self.assertFalse(runnable)
        self.assertIn("m_a", why)

    def test_pass_opens_it(self):
        b = board(self.jobs, self.cands)
        b.jobs["m_a"].state, b.jobs["m_a"].passed = JobState.DONE, True
        self.assertTrue(readiness(b, b.jobs["m_b"], Policy())[0])

    def test_reference_jobs_are_never_gated(self):
        jobs = self.jobs + [Job(id="ref_b", command="x", candidate="ref", role=Role.NEEDED, estimate_s=600)]
        b = board(jobs, self.cands)
        self.assertTrue(readiness(b, b.jobs["ref_b"], Policy())[0])

    def test_rejected_candidate_runs_nothing(self):
        b = board(self.jobs, self.cands)
        b.candidates["m"].state = CandState.REJECTED
        self.assertNotIn("m_a", [r.job.id for r in rank(b, est_for(b), 0, Policy())])

    def test_explicit_empty_gate(self):
        jobs = self.jobs[:2] + [Job(id="m_b", command="x", candidate="m", role=Role.NEEDED, estimate_s=600,
                                    needs_pass=[])]
        b = board(jobs, self.cands)
        self.assertTrue(readiness(b, b.jobs["m_b"], Policy())[0])


class JudgementHold(unittest.TestCase):
    """G1 Sonnet A1: squat's check ended, the GPU it freed went to a 2,589-s longbench_v2, and the 8.1-h GSM8K the
    check gated waited 1,790 s for the next free GPU."""

    def board(self):
        cands = [Candidate("ref", reference=True), Candidate("m")]
        b = board([Job(id="ref_q", command="x", candidate="ref", kind="q", role=Role.CHECK, estimate_s=300),
                   Job(id="m_q", command="x", candidate="m", kind="q", role=Role.CHECK, estimate_s=300),
                   Job(id="m_long", command="x", candidate="m", kind="long", estimate_s=20000),
                   Job(id="other", command="x", estimate_s=2500)], cands, gpus=1)
        for jid in ("ref_q", "m_q"):
            b.jobs[jid].state, b.jobs[jid].started_at, b.jobs[jid].ended_at = JobState.DONE, 1e6 - 300, 1e6
        b.jobs["ref_q"].passed = True
        return b

    def test_the_freed_gpu_waits_for_the_pass(self):
        b = self.board()
        d = decide(b, est_for(b), 1e6 + 5, Policy())
        self.assertEqual(d.starts, [])
        self.assertEqual([(c, g) for c, g, _ in d.hold], [("m_q", 0)])
        b.jobs["m_q"].passed = True                       # the branch passes it 8 s later
        self.assertEqual(decide(b, est_for(b), 1e6 + 8, Policy()).starts[0][0], "m_long")

    def test_the_held_gpu_goes_to_the_job_it_was_held_for(self):
        b = self.board()
        # an unmeasured method with a short job: calibration (class 2.5) would otherwise take the GPU
        b.candidates["n"] = Candidate("n")
        for k, sec in (("q", 300), ("long", 900)):
            b.jobs[f"n_{k}"] = Job(id=f"n_{k}", command="x", candidate="n", kind=k, estimate_s=sec, needs_pass=[])
        b.jobs["m_q"].passed, b.jobs["m_q"].judged_at = True, 1e6 + 8
        self.assertEqual(decide(b, est_for(b), 1e6 + 9, Policy()).starts[0][0], "m_long")

    def test_the_hold_ends(self):
        b = self.board()
        d = decide(b, est_for(b), 1e6 + 61, Policy())     # no pass within hold_s: the GPU goes to other work
        self.assertEqual(d.hold, [])
        self.assertEqual(d.starts[0][0], "other")

    def test_no_hold_for_short_gated_work(self):
        b = self.board()
        b.jobs["m_long"].estimate_s = 1000                # shorter than what is runnable: holding would only cost
        self.assertEqual(decide(b, est_for(b), 1e6 + 5, Policy()).starts[0][0], "other")

    def test_off(self):
        b = self.board()
        self.assertEqual(decide(b, est_for(b), 1e6 + 5, Policy(hold_s=0)).starts[0][0], "other")


class Speculation(unittest.TestCase):
    def test_gated_work_fills_an_idle_gpu_only_when_enabled(self):
        cands = [Candidate("m")]
        jobs = [Job(id="m_a", command="x", candidate="m", role=Role.CHECK, estimate_s=60),
                Job(id="m_b", command="x", candidate="m", role=Role.NEEDED, estimate_s=600)]
        b = board(jobs, cands, gpus=2)
        self.assertEqual([j for j, _ in decide(b, est_for(b), 0, Policy()).starts], ["m_a"])
        self.assertEqual(sorted(j for j, _ in decide(b, est_for(b), 0, Policy(speculate=True)).starts), ["m_a", "m_b"])

    def test_speculative_job_yields_to_real_work(self):
        cands = [Candidate("m")]
        jobs = [Job(id="spec", command="x", role=Role.SPECULATIVE, estimate_s=600, state=JobState.RUNNING, gpu=0),
                Job(id="real", command="x", estimate_s=60)]
        b = board(jobs, cands, gpus=1)
        b.gpu_job[0] = "spec"
        d = decide(b, est_for(b), 0, Policy())
        self.assertEqual(d.preempt, ["spec"])


class Bound(unittest.TestCase):
    def simulate(self, jobs, m):
        """Run the policy to the end with exact lengths; return the makespan."""
        b = board(jobs, gpus=m)
        e = Estimator()
        for j in b.jobs.values():
            e.measured(j, j.estimate_s)
        t, running = 0.0, {}
        while True:
            d = decide(b, e, t, Policy(calibrate=False))
            for jid, g in d.starts:
                j = b.jobs[jid]
                j.state, j.gpu, j.started_at = JobState.RUNNING, g, t
                b.gpu_job[g] = jid
                running[jid] = t + j.estimate_s
            if not running:
                return t
            jid = min(running, key=running.get)
            t = running.pop(jid)
            j = b.jobs[jid]
            j.state, j.ended_at = JobState.DONE, t
            b.gpu_job[j.gpu] = None

    def test_graham_bound_on_random_graphs(self):
        import random
        rnd = random.Random(7)
        for trial in range(40):
            n, m = rnd.randint(4, 18), rnd.randint(1, 4)
            jobs = []
            for i in range(n):
                after = [f"j{k}" for k in range(i) if rnd.random() < 0.15]
                jobs.append(Job(id=f"j{i}", command="x", estimate_s=float(rnd.randint(10, 400)), after=after))
            W = sum(j.estimate_s for j in jobs)
            longest = {}
            for j in jobs:
                longest[j.id] = j.estimate_s + max([longest[a] for a in j.after] or [0])
            L = max(longest.values())
            T = self.simulate([Job(**{k: getattr(j, k) for k in ("id", "command", "estimate_s", "after")}) for j in jobs], m)
            self.assertLessEqual(T, W / m + (1 - 1 / m) * L + 1e-6, f"trial {trial}")
            self.assertGreaterEqual(T, max(W / m, L) - 1e-6)


if __name__ == "__main__":
    unittest.main()
