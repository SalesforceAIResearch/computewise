"""The engine on a fake machine: a rejection cancels the subtree, a lost job is DIED, an undecided check is re-shown."""

import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpuharness import plan as P  # noqa: E402
from gpuharness.daemon import Engine  # noqa: E402
from gpuharness.launcher import Started  # noqa: E402
from gpuharness.ledger import Store  # noqa: E402
from gpuharness.model import CandState, JobState  # noqa: E402


class FakeMachine:
    """Jobs that end when told to, with an exit code or by vanishing."""

    def __init__(self):
        self.next_pid, self.live, self.rc, self.signals = 1000, set(), {}, []

    def start(self, job, gpu, logdir):
        self.next_pid += 1
        pid = self.next_pid
        self.live.add(pid)
        log = os.path.join(logdir, job.id + ".log")
        open(log, "w").write("")
        return Started(pid=pid, log=log, rc_path=os.path.join(logdir, job.id + ".rc"), started_at=time.time())

    def alive(self, pid):
        return pid in self.live

    def exit_code(self, rc_path):
        return self.rc.get(rc_path)

    def terminate(self, pid):
        self.signals.append(("TERM", pid))
        self.live.discard(pid)

    kill = terminate

    def reap(self, pid):
        pass

    def finish(self, job, rc=0):
        self.live.discard(job.pid)
        if rc is not None:
            self.rc[job.rc_path] = rc


def engine(plan_d):
    d = tempfile.mkdtemp()
    board, policy, _ = P.load(plan_d, gpus=[0, 1])
    eng = Engine(board, policy, Store(d), backend=FakeMachine())
    return eng


PLAN = {"rule": "r", "candidates": [{"id": "ref", "branch": "r", "reference": True}, {"id": "m", "branch": "bm"},
                                    {"id": "n", "branch": "bn"}],
        "jobs": [{"id": "ref_a", "candidate": "ref", "kind": "a", "role": "check", "command": "x", "estimate_s": 60},
                 {"id": "m_a", "candidate": "m", "kind": "a", "role": "check", "command": "x", "estimate_s": 60},
                 {"id": "m_b", "candidate": "m", "kind": "b", "command": "x", "estimate_s": 600},
                 {"id": "n_a", "candidate": "n", "kind": "a", "role": "check", "command": "x", "estimate_s": 60}]}


class Engine_(unittest.TestCase):
    def test_reject_cancels_running_and_queued(self):
        eng = engine(PLAN)
        t = 0.0
        eng.tick(t)
        running = [j for j in eng.board.jobs.values() if j.state is JobState.RUNNING]
        self.assertEqual(len(running), 2)
        m_a = eng.board.jobs["m_a"]
        if m_a.state is not JobState.RUNNING:   # make m_a run
            eng.backend.finish(running[0]); eng.tick(t + 1); eng.tick(t + 2)
        eng.store.submit({"op": "reject", "candidate": "m", "why": "slow", "evidence": ["m_a"], "from": "bm"})
        eng.tick(t + 3)
        self.assertIs(eng.board.candidates["m"].state, CandState.REJECTED)
        self.assertIs(eng.board.jobs["m_b"].state, JobState.CANCELLED)
        eng.tick(t + 4)
        self.assertIn(eng.board.jobs["m_a"].state, (JobState.CANCELLED,))
        self.assertNotIn("m_a", eng.board.gpu_job.values())

    def test_refuses_rejection_on_screens(self):
        plan = dict(PLAN, jobs=PLAN["jobs"] + [{"id": "m_slice", "candidate": "m", "kind": "a", "command": "x",
                                                  "screen": True, "role": "check"}])
        eng = engine(plan)
        eng.tick(0)
        ok, text = eng.apply({"op": "reject", "candidate": "m", "why": "slice low", "evidence": ["m_slice"]}, 1)
        self.assertFalse(ok)
        self.assertIn("screen", text)

    def test_vanished_job_is_died_and_its_gpu_freed(self):
        eng = engine(PLAN)
        eng.tick(0)
        j = next(j for j in eng.board.jobs.values() if j.state is JobState.RUNNING)
        eng.backend.finish(j, rc=None)   # gone, no exit code
        eng.tick(1); eng.tick(2)
        self.assertIs(j.state, JobState.DIED)
        self.assertNotIn(j.id, eng.board.gpu_job.values())

    def test_pass_opens_gated_work_and_open_items_track_it(self):
        eng = engine(PLAN)
        eng.tick(0)
        for _ in range(4):
            for j in [j for j in eng.board.jobs.values() if j.state is JobState.RUNNING and j.id != "m_b"]:
                eng.backend.finish(j, rc=0)
            eng.tick(_ + 1)
        self.assertIs(eng.board.jobs["m_a"].state, JobState.DONE)
        self.assertIs(eng.board.jobs["m_b"].state, JobState.WAITING)
        items = dict(eng.open_items().get("bm", []))
        self.assertIn("judge:m_a", items)
        ok, _ = eng.apply({"op": "pass", "jobs": ["m_a"], "from": "bm"}, 9)
        self.assertTrue(ok)
        eng.tick(10)
        self.assertIs(eng.board.jobs["m_b"].state, JobState.RUNNING)
        self.assertNotIn("judge:m_a", dict(eng.open_items().get("bm", [])))


class LiveResults(unittest.TestCase):
    """G-kvgate-haiku-A1: a needed job's result failed the rule, nobody was woken, and the candidate's 8-h job ran on."""

    PLAN = {"rule": "r", "candidates": [{"id": "ref", "branch": "r", "reference": True}, {"id": "m", "branch": "bm"},
                                        {"id": "n", "branch": "bn"}],
            "jobs": [{"id": "ref_a", "candidate": "ref", "kind": "a", "command": "x", "estimate_s": 60},
                     {"id": "ref_b", "candidate": "ref", "kind": "b", "command": "x", "estimate_s": 60},
                     {"id": "m_a", "candidate": "m", "kind": "a", "role": "check", "command": "x", "estimate_s": 60},
                     {"id": "m_b", "candidate": "m", "kind": "b", "command": "x", "estimate_s": 60},
                     {"id": "m_long", "candidate": "m", "kind": "long", "command": "x", "estimate_s": 9000},
                     {"id": "n_a", "candidate": "n", "kind": "a", "role": "check", "command": "x", "estimate_s": 60}]}

    def run_until(self, eng, t, pred, step=1.0, limit=200):
        for _ in range(limit):
            if pred():
                return t
            for j in [j for j in eng.board.jobs.values() if j.state is JobState.RUNNING and j.id != "m_long"]:
                eng.backend.finish(j, rc=0)
            t += step
            eng.tick(t)
        self.fail("condition never held")

    def test_a_needed_result_wakes_its_branch_while_work_remains(self):
        eng = engine(self.PLAN)
        eng.tick(0)
        t = self.run_until(eng, 0, lambda: eng.board.jobs["m_a"].state is JobState.DONE)
        eng.apply({"op": "pass", "jobs": ["m_a"], "from": "bm"}, t)
        t = self.run_until(eng, t, lambda: eng.board.jobs["m_b"].state is JobState.DONE)
        self.assertIs(eng.board.jobs["m_long"].state, JobState.RUNNING)
        done_b = [e for e in eng.store.read_events() if e.type == "DONE" and e.job == "m_b"]
        self.assertTrue(done_b and done_b[0].wake, "a live candidate's needed result must wake its branch")
        self.assertIn("m_long", done_b[0].text)
        self.assertIn("reject m", done_b[0].text)
        # the branch rejects on it, and the long job is stopped at once
        ok, _ = eng.apply({"op": "reject", "candidate": "m", "why": "b fails", "evidence": ["m_b", "ref_b"],
                           "from": "bm"}, t + 1)
        self.assertTrue(ok)
        eng.tick(t + 2)
        self.assertIn(eng.board.jobs["m_long"].state, (JobState.CANCELLED, JobState.STOPPING))

    def test_late_reference_result_makes_needed_results_comparable(self):
        plan = {"rule": "r", "candidates": [{"id": "ref", "branch": "r", "reference": True}, {"id": "m", "branch": "bm"},
                                            {"id": "n", "branch": "bn"}],
                "jobs": [{"id": "ref_b", "candidate": "ref", "kind": "b", "command": "x", "estimate_s": 60},
                         {"id": "m_b", "candidate": "m", "kind": "b", "command": "x", "estimate_s": 60, "needs_pass": []},
                         {"id": "m_c", "candidate": "m", "kind": "c", "command": "x", "estimate_s": 900, "needs_pass": []},
                         {"id": "n_b", "candidate": "n", "kind": "b", "command": "x", "estimate_s": 60, "needs_pass": []}]}
        d = tempfile.mkdtemp()
        board, policy, _ = P.load(plan, gpus=[0, 1, 2, 3])
        eng = Engine(board, policy, Store(d), backend=FakeMachine())
        eng.tick(0)
        eng.backend.finish(eng.board.jobs["m_b"], rc=0)     # the candidate's result lands first
        eng.tick(1); eng.tick(2)
        self.assertIs(eng.board.jobs["m_b"].state, JobState.DONE)
        eng.backend.finish(eng.board.jobs["ref_b"], rc=0)   # then the reference's
        eng.tick(3); eng.tick(4)
        judge = [e for e in eng.store.read_events() if e.type == "JUDGE" and e.job == "m_b"]
        self.assertTrue(judge, "the reference's late result must re-send m_b's comparison")
        self.assertTrue(judge[0].wake)
        self.assertIn("still queued or running", judge[0].text)


T0 = 1_000_000.0   # a box-like clock: the engine reads a job's log only once its started_at is set (non-zero)


class Milestones(unittest.TestCase):
    """pkg-unneeded-k: the rule reads val_loss at step 1000 of a 3,400-step run; A3 showed it only at the end."""

    PAT = r"step:1000/\d+ val_loss:(?P<value>[\d.]+)"
    PLAN = {"rule": "r", "candidates": [{"id": "base", "branch": "rb", "reference": True}, {"id": "psgd", "branch": "c"}],
            "jobs": [{"id": "base_r1", "candidate": "base", "kind": "train", "command": "x", "estimate_s": 4000,
                      "milestones": [{"name": "step1000", "pattern": PAT}]},
                     {"id": "psgd_r1", "candidate": "psgd", "kind": "train", "role": "check", "command": "x",
                      "estimate_s": 5000, "milestones": [{"name": "step1000", "pattern": PAT}]},
                     {"id": "psgd_r2", "candidate": "psgd", "kind": "train", "command": "x", "estimate_s": 5000}]}

    def write(self, eng, jid, text):
        with open(eng.board.jobs[jid].log, "a") as fh:
            fh.write(text)

    def test_the_value_arrives_while_the_job_runs_with_the_reference_beside_it(self):
        eng = engine(self.PLAN)
        eng.tick(T0 + 0)
        self.assertTrue(all(eng.board.jobs[j].state is JobState.RUNNING for j in ("base_r1", "psgd_r1")))
        self.write(eng, "base_r1", "step:1000/3250 val_loss:3.65220 train_time:1235.600s\n")
        eng.tick(T0 + 1)
        self.write(eng, "psgd_r1", "step:999/3400 train_time:1430.5s\nstep:1000/3400 val_loss:3.74245 train_time:1431.9s\n")
        eng.tick(T0 + 2)
        ms = [e for e in eng.store.read_events() if e.type == "MILESTONE" and e.job == "psgd_r1"]
        self.assertEqual(len(ms), 1)
        self.assertTrue(ms[0].wake)
        self.assertEqual(ms[0].to, "c")
        self.assertIn("step1000 = 3.74245", ms[0].text)
        self.assertIn("reference step1000: base_r1 3.6522", ms[0].text)
        self.assertIn("gpuh reject psgd", ms[0].text)
        self.assertIs(eng.board.jobs["psgd_r1"].state, JobState.RUNNING)   # the job is still running
        self.write(eng, "psgd_r1", "step:1000/3400 val_loss:3.74245\n")   # a repeat line: no second event
        eng.tick(T0 + 3)
        self.assertEqual(len([e for e in eng.store.read_events() if e.type == "MILESTONE" and e.job == "psgd_r1"]), 1)

    def test_a_candidate_value_before_the_reference_is_re_sent_when_it_arrives(self):
        eng = engine(self.PLAN)
        eng.tick(T0 + 0)
        self.write(eng, "psgd_r1", "step:1000/3400 val_loss:3.74245\n")
        eng.tick(T0 + 1)
        first = [e for e in eng.store.read_events() if e.type == "MILESTONE" and e.job == "psgd_r1"]
        self.assertIn("not read yet", first[0].text)
        self.write(eng, "base_r1", "step:1000/3250 val_loss:3.65220\n")
        eng.tick(T0 + 2)
        again = [e for e in eng.store.read_events() if e.type == "MILESTONE" and e.job == "psgd_r1"]
        self.assertEqual(len(again), 2)
        self.assertIn("base_r1 3.6522", again[1].text)

    def test_no_milestones_declared_no_event(self):
        plan = {"rule": "r", "candidates": [{"id": "m", "branch": "bm"}],
                "jobs": [{"id": "m_r1", "candidate": "m", "command": "x", "estimate_s": 100}]}
        eng = engine(plan)
        eng.tick(T0 + 0)
        with open(eng.board.jobs["m_r1"].log, "a") as fh:
            fh.write("step:1000/3400 val_loss:3.74245\n")
        eng.tick(T0 + 1)
        self.assertFalse([e for e in eng.store.read_events() if e.type == "MILESTONE"])

    def test_the_plan_refuses_a_pattern_without_a_value_group(self):
        bad = {"rule": "r", "candidates": [{"id": "m", "branch": "bm"}],
               "jobs": [{"id": "m_r1", "candidate": "m", "command": "x",
                         "milestones": [{"name": "s", "pattern": "step:1000"}]}]}
        with self.assertRaises(P.PlanError):
            P.load(bad, gpus=[0])


class CpuJobs(unittest.TestCase):
    """pkg-filter-yield: CPU-only preparations held every GPU for over an hour while 28 training jobs waited."""

    def plan(self, cpu_slots=None):
        recipes = ["none", "c4", "gopher", "fineweb"]
        jobs = []
        for r in recipes:
            jobs.append({"id": f"prep_{r}", "candidate": r, "kind": "prep", "role": "check", "gpu": 0,
                         "command": "x", "estimate_s": 1800})
            for s in range(4):
                jobs.append({"id": f"train_{r}_s{s}", "candidate": r, "kind": "train", "after": [f"prep_{r}"],
                             "command": "x", "estimate_s": 1400})
        d = {"rule": "r", "candidates": [{"id": r, "branch": "b" + r} for r in recipes], "jobs": jobs}
        if cpu_slots:
            d["policy"] = {"cpu_slots": cpu_slots}
        return d

    def test_preps_take_no_gpu_and_training_starts_on_the_freed_gpus(self):
        eng = engine(self.plan())
        eng.board.gpus = [0, 1, 2, 3]
        eng.board.gpu_job = {g: None for g in eng.board.gpus}
        eng.tick(T0 + 0)
        preps = [eng.board.jobs[f"prep_{r}"] for r in ("none", "c4", "gopher", "fineweb")]
        self.assertTrue(all(p.state is JobState.RUNNING and p.gpu is None for p in preps))
        self.assertEqual(eng.board.free_gpus(), [0, 1, 2, 3])            # every GPU free while the CPU works
        eng.backend.finish(eng.board.jobs["prep_none"], rc=0)
        eng.tick(T0 + 1); eng.tick(T0 + 2)
        eng.store.submit({"op": "pass", "jobs": ["prep_none"], "from": "bnone"})
        eng.tick(T0 + 3)
        started = sorted(j.id for j in eng.board.jobs.values() if j.state is JobState.RUNNING and not j.cpu)
        self.assertEqual(started, [f"train_none_s{s}" for s in range(4)])   # none's four seeds, one per GPU
        starts = [e for e in eng.store.read_events() if e.type == "STARTED" and e.job == "prep_c4"]
        self.assertIn("on CPU", starts[0].text)

    def test_cpu_slots_limit_how_many_run_at_once(self):
        eng = engine(self.plan(cpu_slots=1))
        eng.tick(T0 + 0)
        running = [j.id for j in eng.board.jobs.values() if j.cpu and j.state is JobState.RUNNING]
        self.assertEqual(len(running), 1)

    def test_a_cpu_job_sees_no_gpu(self):
        import subprocess, sys as _s
        from gpuharness import launcher as L
        d = tempfile.mkdtemp()
        st = L.start("cpujob", "echo CVD=[$CUDA_VISIBLE_DEVICES]", None, d, d)
        for _ in range(50):
            if L.exit_code(st.rc_path) is not None:
                break
            time.sleep(0.05)
        self.assertIn("CVD=[]", open(st.log).read())
        self.assertIn("start cpujob cpu", open(st.log).read())

    def test_the_plan_refuses_other_gpu_counts(self):
        bad = {"rule": "r", "candidates": [{"id": "m", "branch": "bm"}],
               "jobs": [{"id": "m_r1", "candidate": "m", "command": "x", "gpu": 2}]}
        with self.assertRaises(P.PlanError):
            P.load(bad, gpus=[0])

    def test_a3_board_json_is_unchanged_by_the_new_fields(self):
        plan = {"rule": "r", "candidates": [{"id": "m", "branch": "bm"}],
                "jobs": [{"id": "m_r1", "candidate": "m", "command": "x", "estimate_s": 100}]}
        board, _, _ = P.load(plan, gpus=[0])
        out = board.to_json()
        self.assertNotIn("cpu_slots", out)
        self.assertFalse({"gpus", "milestones", "milestone_values"} & set(out["jobs"][0]))


class SilentJobs(unittest.TestCase):
    """G-k2lp-haiku-A3: squat's GSM8K printed nothing for 5 h (no per-example line), and its branch cancelled it as
    "hung" 5 h into a 29,157-s run. A silent job whose processes use CPU is working, and must say so."""

    def setUp(self):
        import subprocess
        from gpuharness import cli as C
        self.C = C
        self.d = tempfile.mkdtemp()
        plan = {"rule": "r", "candidates": [{"id": "kvq", "branch": "kvq"}],
                "jobs": [{"id": "squat_gsm8k", "candidate": "kvq", "command": "x", "estimate_s": 1800}]}
        board, _, _ = P.load(plan, gpus=[0])
        self.st = Store(self.d)
        self.busy = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: sum(range(10000))"],
                                     start_new_session=True)
        self.idle = subprocess.Popen([sys.executable, "-c", "import time\ntime.sleep(600)"], start_new_session=True)
        j = board.jobs["squat_gsm8k"]
        j.state, j.pid, j.started_at = JobState.RUNNING, self.busy.pid, time.time() - 5 * 3600 - 300
        j.log = os.path.join(self.d, "logs", "squat_gsm8k.log")
        open(j.log, "w").write("Loading weights: 100%|##########| 434/434\n")
        os.utime(j.log, (time.time() - 5 * 3600, time.time() - 5 * 3600))    # its last line, 5 h ago
        self.board = board
        C.CPU_SAMPLE_S = 1.0

    def tearDown(self):
        for p in (self.busy, self.idle):
            p.kill(); p.wait()

    def test_silent_but_working_reads_as_working_not_hung(self):
        h = self.C.silent_health(self.st, self.board, time.time())
        self.assertEqual(len(h), 1)
        self.assertIn("silent squat_gsm8k", h[0])
        self.assertIn("working", h[0])
        self.assertNotIn("STUCK", h[0])
        self.assertIn("running 5h", h[0])

    def test_silent_and_idle_reads_as_stuck(self):
        self.board.jobs["squat_gsm8k"].pid = self.idle.pid
        h = self.C.silent_health(self.st, self.board, time.time())
        self.assertIn("STUCK?", h[0])

    def test_a_job_that_spoke_recently_is_not_listed(self):
        os.utime(self.board.jobs["squat_gsm8k"].log, None)
        self.assertEqual(self.C.silent_health(self.st, self.board, time.time()), [])

    def test_cancel_refuses_a_silent_working_job_without_force(self):
        import io, contextlib, argparse
        self.st.save_board(self.board)
        open(os.path.join(self.d, "plan.json"), "w").write("{}")
        out = io.StringIO()
        args = argparse.Namespace(dir=self.d, as_="kvq", jobs=["squat_gsm8k"], why=None, force=False)
        with contextlib.redirect_stdout(out):
            rc = self.C.cmd_cancel(args)
        self.assertEqual(rc, 2)
        self.assertIn("silent but working, not hung", out.getvalue())


class NoReferenceEstimates(unittest.TestCase):
    """G-k2lp-haiku-A3: no reference, and 11 of 12 kinds run by one candidate. A3's fit let candidate and kind factors
    trade off without bound: the 29,157-s GSM8K was predicted at 116,707 s, then 639 s, then 8,747 s on a retry."""

    def test_prediction_follows_the_measured_kind_not_a_runaway_factor(self):
        from gpuharness.estimate import Estimator
        from gpuharness.model import Job
        e = Estimator(); e.set_reference([])
        J = {}
        for c, kinds in (("kvq", ["gsm8k", "repobench"]), ("kvsel", ["gsm8k", "v2"]), ("dlm", ["math", "humaneval"]),
                         ("mlsys", ["niah", "qasper"])):
            for k in kinds:
                J[c, k] = Job(id=f"{c}_{k}", command="x", candidate=c, kind=k, estimate_s=1800)
                e.prior(J[c, k], 1800)
        e.measured(J["kvq", "repobench"], 2021)
        e.measured(J["mlsys", "niah"], 1784)
        self.assertLess(e.predict(J["kvq", "gsm8k"]), 4000)        # not 116,707 s
        e.measured(J["kvsel", "gsm8k"], 7300)
        self.assertGreater(e.predict(J["kvq", "gsm8k"]), 5000)     # a measured gsm8k carries to the other gsm8k

    def test_a_plan_with_a_reference_is_fitted_as_before(self):
        from gpuharness.estimate import Estimator
        from gpuharness.model import Job
        e = Estimator(); e.set_reference(["ref"])
        r = Job(id="r", command="x", candidate="ref", kind="a", estimate_s=100)
        m = Job(id="m", command="x", candidate="m", kind="a", estimate_s=100)
        m2 = Job(id="m2", command="x", candidate="m", kind="b", estimate_s=100)
        r2 = Job(id="r2", command="x", candidate="ref", kind="b", estimate_s=100)
        for j in (r, m, m2, r2):
            e.prior(j, 100)
        e.measured(r, 100); e.measured(m, 300); e.measured(r2, 1000)
        self.assertAlmostEqual(e.predict(m2), 3000, delta=300)

    def test_the_plan_check_names_the_case(self):
        plan = {"rule": "r", "candidates": [{"id": c, "branch": c} for c in ("a", "b", "c", "d")],
                "jobs": [{"id": f"{c}_{k}", "candidate": c, "kind": f"{c}k{k}", "command": "x", "estimate_s": 1800}
                         for c in ("a", "b", "c", "d") for k in (1, 2)]}
        _, _, warnings = P.load(plan, gpus=[0])
        self.assertTrue(any("no candidate is the reference" in w for w in warnings))



class MilestoneArithmetic(unittest.TestCase):
    """A4.1 (G-k-haiku-A4, c3-19): the MILESTONE line states the arithmetic of a mean rule and the run-to-run spread,
    so a branch never takes the differences between candidates for spread. Values are K's registry records."""

    PAT = r"step:1000/\d+ val_loss:(?P<value>[\d.]+)"

    def plan(self, better=True, ref_runs=5, cand_runs=5):
        ms = {"name": "step1000", "pattern": self.PAT}
        if better:
            ms.update({"better": "lower", "margin": 0.005})
        jobs = [{"id": f"base_r{i}", "candidate": "base", "kind": "train", "command": "x", "estimate_s": 4000,
                 "milestones": [ms]} for i in range(1, ref_runs + 1)]
        jobs += [{"id": f"nm_r{i}", "candidate": "nm", "kind": "train", "role": "check" if i == 1 else "needed",
                  "command": "x", "estimate_s": 4000, "milestones": [ms]} for i in range(1, cand_runs + 1)]
        return {"rule": "r", "candidates": [{"id": "base", "branch": "main", "reference": True},
                                            {"id": "nm", "branch": "b2"}], "jobs": jobs}

    def eng4(self, plan, gpus=4):
        d = tempfile.mkdtemp()
        board, policy, _ = P.load(plan, gpus=list(range(gpus)))
        return Engine(board, policy, Store(d), backend=FakeMachine())

    def write(self, eng, jid, v):
        with open(eng.board.jobs[jid].log, "a") as fh:
            fh.write(f"step:1000/3275 val_loss:{v:.5f} train_time:1186.7s\n")

    def lines(self, eng, jid):
        return [e for e in eng.store.read_events() if e.type == "MILESTONE" and e.job == jid]

    def test_one_reference_run_states_what_the_others_need_and_that_spread_is_unmeasured(self):
        eng = self.eng4(self.plan())
        eng.tick(T0)
        self.write(eng, "base_r1", 3.6522)
        eng.tick(T0 + 1)
        self.write(eng, "nm_r1", 3.6812)
        eng.tick(T0 + 2)
        t = self.lines(eng, "nm_r1")[-1].text
        # bar 3.6522 - 0.005 = 3.6472; the other 4 need (5 * 3.6472 - 3.6812) / 4 = 3.6387, 0.0425 below this run
        self.assertIn("to pass (<= 3.6472, reference mean 3.6522 n 1 - 0.005), its other 4 would have to average 3.6387",
                      t)
        self.assertIn("0.0425 below this run", t)
        self.assertIn("run-to-run sd: not measured yet", t)
        self.assertIn("differences between candidates are not run-to-run spread", t)

    def test_three_reference_runs_give_the_gap_in_sds(self):
        eng = self.eng4(self.plan())
        eng.tick(T0)
        for jid, v in (("base_r1", 3.6522), ("base_r2", 3.65161), ("base_r3", 3.6504)):
            self.write(eng, jid, v)
        eng.tick(T0 + 1)
        self.write(eng, "nm_r1", 3.6812)
        eng.tick(T0 + 2)
        t = self.lines(eng, "nm_r1")[-1].text
        # mean 3.651403, sample sd 0.000918; need (5 * 3.646403 - 3.6812) / 4 = 3.637704, 0.0435 below, 47.4 sd
        self.assertIn("would have to average 3.6377", t)
        self.assertIn("0.0435 below this run", t)
        self.assertIn("run-to-run sd 0.000918 (base, 3 runs): that gap is 47.4 sd", t)

    def test_the_line_is_re_sent_once_when_the_spread_becomes_measurable(self):
        eng = self.eng4(self.plan(), gpus=6)
        eng.tick(T0)
        self.write(eng, "base_r1", 3.6522)
        eng.tick(T0 + 1)
        self.write(eng, "nm_r1", 3.6812)
        eng.tick(T0 + 2)
        self.assertEqual(len(self.lines(eng, "nm_r1")), 1)
        self.write(eng, "base_r2", 3.65161)
        eng.tick(T0 + 3)
        self.assertEqual(len(self.lines(eng, "nm_r1")), 1)          # two runs: no spread yet, nothing new to say
        self.write(eng, "base_r3", 3.6504)
        eng.tick(T0 + 4)
        ls = self.lines(eng, "nm_r1")
        self.assertEqual(len(ls), 2)
        self.assertTrue(ls[1].wake)
        self.assertIn("re-sent: the reference's run-to-run spread is in", ls[1].text)
        self.assertIn("that gap is 47.4 sd", ls[1].text)
        self.write(eng, "base_r4", 3.6524)
        eng.tick(T0 + 5)
        self.assertEqual(len(self.lines(eng, "nm_r1")), 2)          # once only

    def test_without_better_the_line_states_the_tie_and_never_a_verdict(self):
        eng = self.eng4(self.plan(better=False))
        eng.tick(T0)
        self.write(eng, "base_r1", 3.6522)
        eng.tick(T0 + 1)
        self.write(eng, "nm_r1", 3.6812)
        eng.tick(T0 + 2)
        t = self.lines(eng, "nm_r1")[-1].text
        # tie: (5 * 3.6522 - 3.6812) / 4 = 3.64495, 0.03625 below this run
        self.assertIn("to tie the reference mean 3.6522 (n 1), its other 4 would have to average 3.64495", t)
        self.assertNotIn("fail", t.split("| log:")[0].split("stops them now")[-1])

    def test_a_run_already_inside_the_bar_is_said_not_to_fail(self):
        eng = self.eng4(self.plan())
        eng.tick(T0)
        self.write(eng, "base_r1", 3.6522)
        eng.tick(T0 + 1)
        self.write(eng, "nm_r1", 3.6455)            # radial-damp's r1: below the bar 3.6472
        eng.tick(T0 + 2)
        self.assertIn("this value alone does not fail the rule", self.lines(eng, "nm_r1")[-1].text)

    def test_the_reference_runs_that_measure_the_spread_rank_as_enablers(self):
        from gpuharness import scheduler as S
        plan = self.plan()
        for j in plan["jobs"]:
            if j["candidate"] == "nm" and j["id"] != "nm_r1":
                j["role"] = "check"           # every candidate run runnable, so only the ranking decides
        eng = self.eng4(plan)
        ranked = S.rank(eng.board, eng.est, T0, eng.policy)
        top = [r.job.id for r in ranked if r.klass == 3.0]
        self.assertEqual(sorted(top), ["base_r1", "base_r2", "base_r3"])
        self.assertTrue(all("run-to-run spread" in r.why for r in ranked if r.job.id in top))

    def test_without_milestones_the_reference_runs_are_not_enablers(self):
        from gpuharness import scheduler as S
        plan = self.plan()
        for j in plan["jobs"]:
            j.pop("milestones")
        eng = self.eng4(plan)
        self.assertFalse([r for r in S.rank(eng.board, eng.est, T0, eng.policy) if r.klass == 3.0])

    def test_the_plan_refuses_a_bad_better_or_margin(self):
        for bad in ({"better": "up"}, {"margin": 0.005}, {"better": "lower", "margin": -1},
                    {"better": "lower", "margin": "x"}):
            plan = self.plan()
            plan["jobs"][0]["milestones"] = [dict({"name": "s", "pattern": self.PAT}, **bad)]
            with self.assertRaises(P.PlanError):
                P.load(plan, gpus=[0])



class CpuShareNote(unittest.TestCase):
    """A4.2 (G-fyld-haiku-A4): `gpuh check` prints each CPU job's share of the cores, from the plan's structure and the
    machine's affinity only; it never reads a command."""

    def plan(self, slots=1, n=7, gpu=0):
        jobs = [{"id": f"r{i}_prep", "candidate": f"r{i}", "kind": "prep", "gpu": gpu, "role": "check",
                 "command": "python tools/prepare.py r --tasks 8 --workers 8", "estimate_s": 1800} for i in range(n)]
        return {"rule": "r", "candidates": [{"id": f"r{i}", "branch": "b"} for i in range(n)], "jobs": jobs,
                "policy": {"cpu_slots": slots}}

    def notes(self, plan, cores):
        orig = P.machine_cores
        P.machine_cores = lambda: cores
        try:
            return [w for w in P.load(plan, gpus=[0, 1, 2, 3])[2] if "CPU jobs" in w]
        finally:
            P.machine_cores = orig

    def test_one_at_a_time_on_42_cores_gives_each_all_42(self):
        ns = self.notes(self.plan(slots=1), 42)
        self.assertEqual(len(ns), 1)
        self.assertIn('7 CPU jobs ("gpu": 0), at most 1 at once (policy.cpu_slots 1): each has nproc / 1 = 42 of '
                      "this machine's 42 cores", ns[0])
        self.assertIn("at least as many tasks, shards or chunks as workers", ns[0])
        self.assertNotIn("--tasks 8", ns[0])          # the command is never read

    def test_four_at_once_shares_the_cores(self):
        self.assertIn("at most 4 at once (policy.cpu_slots 4): each has nproc / 4 = 10 of this machine's 42 cores",
                      self.notes(self.plan(slots=4), 42)[0])
        self.assertIn("at most 2 at once (policy.cpu_slots 4): each has nproc / 2 = 21",   # fewer jobs than slots
                      self.notes(self.plan(slots=4, n=2), 42)[0])

    def test_no_cpu_jobs_no_note(self):
        self.assertEqual(self.notes(self.plan(gpu=1), 42), [])



class PrepCountTrain(unittest.TestCase):
    """A4.3 (G-fyld-haiku-A4-r1b, c3-16): prep -> count -> train with cpu_slots 1, the plan shapes it ran."""

    RECIPES = ["none", "gopher-repetition", "gopher-quality", "c4"]

    def plan(self, prep_role="check", prefix=("pk", "ck", "tr")):
        pk, ck, tr = prefix
        cands = [{"id": "none", "branch": "main", "reference": True}] + [{"id": m, "branch": "a"} for m in self.RECIPES[1:]]
        return {"rule": "r", "policy": {"cpu_slots": 1}, "candidates": cands, "jobs": [
            {"foreach": {"m": self.RECIPES}, "id": pk + "_{m}", "candidate": "{m}", "kind": "prep", "gpu": 0,
             "command": "x", "estimate_s": 1800, "role": prep_role},
            {"foreach": {"m": self.RECIPES}, "id": ck + "_{m}", "candidate": "{m}", "kind": "count", "gpu": 0,
             "command": "x", "after": [pk + "_{m}"], "estimate_s": 60, "role": "check"},
            {"foreach": {"m": self.RECIPES, "s": ["0", "1", "2", "3"]}, "id": tr + "_{m}_s{s}", "candidate": "{m}",
             "kind": "train", "role": "needed", "command": "x", "after": [pk + "_{m}"], "estimate_s": 1400}]}

    def eng(self, plan):
        board, policy, _ = P.load(plan, gpus=[0, 1, 2, 3])
        return Engine(board, policy, Store(tempfile.mkdtemp()), backend=FakeMachine())

    def test_D1_a_needed_prep_with_a_check_after_it_is_refused_and_named(self):
        with self.assertRaises(P.PlanError) as e:
            P.load(self.plan(prep_role="needed", prefix=("prep", "count", "train")), gpus=[0])
        msg = str(e.exception)
        self.assertIn("dependency cycle", msg)
        self.assertIn("is a needed job, so it waits for every check of its candidate", msg)
        P.load(self.plan(), gpus=[0])                 # the shape the agent re-planned to is accepted

    def test_D1_needs_pass_empty_breaks_the_cycle(self):
        plan = self.plan(prep_role="needed")
        plan["jobs"][0]["needs_pass"] = []
        P.load(plan, gpus=[0])

    def test_D3_the_count_that_releases_idle_gpus_runs_before_the_next_prep(self):
        eng = self.eng(self.plan())
        b, m = eng.board, eng.backend
        t = T0
        eng.tick(t)
        self.assertEqual([j.id for j in b.jobs.values() if j.active], ["pk_none"])
        m.finish(b.jobs["pk_none"]); t += 36; eng.tick(t)      # the reference: count and 4 trainings start
        self.assertTrue(b.jobs["ck_none"].active)
        m.finish(b.jobs["ck_none"]); t += 1; eng.tick(t)
        for k in range(4):
            m.finish(b.jobs[f"tr_none_s{k}"])
        t += 1425; eng.tick(t)                                 # GPUs free; the first recipe's prep runs
        self.assertTrue(b.jobs["pk_gopher-repetition"].active)
        m.finish(b.jobs["pk_gopher-repetition"]); t += 466; eng.tick(t)
        eng.apply({"op": "pass", "jobs": ["pk_gopher-repetition"], "from": "a"}, t); eng.tick(t + 1)
        self.assertTrue(b.jobs["ck_gopher-repetition"].active)     # A4.2 started pk_gopher-quality here
        self.assertFalse(b.jobs["pk_gopher-quality"].active)
        m.finish(b.jobs["ck_gopher-repetition"]); t += 2; eng.tick(t)
        eng.apply({"op": "pass", "jobs": ["ck_gopher-repetition"], "from": "a"}, t); eng.tick(t + 1)
        self.assertEqual(sum(b.jobs[f"tr_gopher-repetition_s{k}"].active for k in range(4)), 4)
        self.assertTrue(b.jobs["pk_gopher-quality"].active)        # and the next prep takes the CPU slot

    def run_to_first_count(self, train_s):
        """The reference's trainings hold all four GPUs; gopher-repetition's prep is done and passed, its count is
        runnable, and gopher-quality's prep is the rival for the one CPU slot (its prior is 1,800 s)."""
        plan = self.plan()
        plan["jobs"][2]["estimate_s"] = train_s
        eng = self.eng(plan)
        b, m = eng.board, eng.backend
        t = T0
        eng.tick(t)
        m.finish(b.jobs["pk_none"]); t += 36; eng.tick(t)
        m.finish(b.jobs["ck_none"]); t += 1; eng.tick(t)
        self.assertEqual(sum(b.jobs[f"tr_none_s{k}"].active for k in range(4)), 4)
        m.finish(b.jobs["pk_gopher-repetition"]); t += 466; eng.tick(t)
        eng.apply({"op": "pass", "jobs": ["pk_gopher-repetition"], "from": "a"}, t); eng.tick(t + 1)
        return b

    def test_D3_gpus_busy_long_after_the_rival_ends_no_reordering(self):
        b = self.run_to_first_count(train_s=20000)      # the GPUs free long after a prep would end
        self.assertTrue(b.jobs["pk_gopher-quality"].active)
        self.assertFalse(b.jobs["ck_gopher-repetition"].active)

    def test_D3_gpus_free_before_the_rival_would_end_the_count_goes_first(self):
        b = self.run_to_first_count(train_s=1400)       # predicted free at ~1,437 s, before a 1,800-s prep would end
        self.assertTrue(b.jobs["ck_gopher-repetition"].active)
        self.assertFalse(b.jobs["pk_gopher-quality"].active)

    def test_D2_a_replaced_cancelled_check_stops_gating(self):
        eng = self.eng(self.plan(prefix=("prep", "count", "train")))
        b = eng.board
        old = [j.id for j in b.jobs.values() if j.candidate == "c4"]
        eng.apply({"op": "cancel", "jobs": old, "from": "main"}, T0)
        frag = self.plan()
        frag["jobs"][0]["foreach"]["m"] = frag["jobs"][1]["foreach"]["m"] = frag["jobs"][2]["foreach"]["m"] = ["c4"]
        ok, msg = eng.apply({"op": "add", "plan": {"jobs": frag["jobs"]}, "from": "main"}, T0)
        self.assertTrue(ok, msg)
        from gpuharness.scheduler import gates
        g = gates(b, b.jobs["tr_c4_s0"], eng.policy)
        self.assertNotIn("count_c4", g)               # cancelled and replaced by ck_c4 (same kind)
        self.assertIn("ck_c4", g)
        self.assertIn("pk_c4", g)

    def test_D2_a_plain_cancel_still_gates_until_retried(self):
        from gpuharness.scheduler import gates, readiness
        eng = self.eng(self.plan())
        b = eng.board
        eng.apply({"op": "cancel", "jobs": ["ck_c4"], "from": "b"}, T0)
        self.assertIn("ck_c4", gates(b, b.jobs["tr_c4_s0"], eng.policy))
        self.assertFalse(readiness(b, b.jobs["tr_c4_s0"], eng.policy)[0])
        eng.apply({"op": "retry", "jobs": ["ck_c4"], "from": "b"}, T0 + 1)
        self.assertIn("ck_c4", gates(b, b.jobs["tr_c4_s0"], eng.policy))

    def test_IDLE_names_the_waiting_jobs(self):
        eng = self.eng(self.plan())
        b, m = eng.board, eng.backend
        eng.apply({"op": "cancel", "jobs": ["pk_c4"], "from": "b"}, T0)     # its count and trainings can never start
        t = T0
        eng.tick(t)
        m.finish(b.jobs["pk_none"]); t += 36; eng.tick(t)
        m.finish(b.jobs["ck_none"]); t += 1; eng.tick(t)
        for k in range(4):
            m.finish(b.jobs[f"tr_none_s{k}"])
        t += 1425; eng.tick(t)                       # the reference is done, as on the box at 1,463 s
        idle = [e for e in eng.store.read_events() if e.type == "IDLE" and e.candidate == "c4"]
        self.assertEqual(len(idle), 1)
        self.assertIn("c4: nothing can start: 5 jobs wait and none is runnable: ck_c4 (dependency pk_c4 cancelled)",
                      idle[0].text)
        self.assertNotIn("every job has ended", idle[0].text)



class OneClockRatio(unittest.TestCase):
    """A4.3 (G-k-haiku-A4-r1b, c3-18): the DONE line compares lengths on one clock each and names them. The branch
    divided radial-damp_s1's wall (5,051 s) by baseline_r1's timer (4,011.5 s), x1.26, and rejected the key's
    winner; on one clock it is x1.155 (timer 4,632.6 / 4,011.5) and x1.166 (wall 5,051 / 4,331)."""

    PLAN = {"rule": "r", "candidates": [{"id": "baseline", "branch": "main", "reference": True},
                                        {"id": "radial-damp", "branch": "c"}],
            "jobs": [{"id": "baseline_r1", "candidate": "baseline", "kind": "train", "command": "x", "estimate_s": 4000},
                     {"id": "radial-damp_s1", "candidate": "radial-damp", "kind": "train", "role": "check",
                      "command": "x", "estimate_s": 4000},
                     {"id": "radial-damp_s7", "candidate": "radial-damp", "kind": "train", "command": "x",
                      "estimate_s": 4000}]}

    def finish(self, eng, jid, at, timer, steps):
        j = eng.board.jobs[jid]
        with open(j.log, "a") as fh:
            fh.write(f"step:{steps}/{steps} val_loss:3.27800 train_time:{timer:.3f}s step_avg:1500.00ms\n")
        eng.backend.finish(j, 0)
        eng.tick(at)              # the exit code is read, and the end stamped, on this tick

    def test_radial_damp_r1b(self):
        board, policy, _ = P.load(self.PLAN, gpus=[0, 1, 2, 3])
        eng = Engine(board, policy, Store(tempfile.mkdtemp()), backend=FakeMachine())
        eng.tick(T0)
        board.jobs["baseline_r1"].started_at = T0
        board.jobs["radial-damp_s1"].started_at = T0 + 1997
        self.finish(eng, "baseline_r1", T0 + 4331, 4011.469, 3250)
        self.finish(eng, "radial-damp_s1", T0 + 1997 + 5051, 4632.581, 3020)
        done = [e for e in eng.store.read_events() if e.type == "DONE" and e.job == "radial-damp_s1"][0].text
        self.assertIn("length against reference baseline_r1, on one clock each: timer 77m13s vs 66m51s -> x1.155; "
                      "wall 84m11s vs 72m11s -> x1.166", done)
        self.assertIn("never a wall time with a timer", done)
        self.assertNotIn("x1.26", done)

    def test_reference_without_a_timer_gives_wall_only(self):
        board, policy, _ = P.load(self.PLAN, gpus=[0, 1, 2, 3])
        eng = Engine(board, policy, Store(tempfile.mkdtemp()), backend=FakeMachine())
        eng.tick(T0)
        board.jobs["baseline_r1"].started_at = T0
        board.jobs["radial-damp_s1"].started_at = T0
        j = board.jobs["baseline_r1"]
        open(j.log, "a").write("done\n"); eng.backend.finish(j, 0); eng.tick(T0 + 100)
        self.finish(eng, "radial-damp_s1", T0 + 115, 100.0, 10)
        done = [e for e in eng.store.read_events() if e.type == "DONE" and e.job == "radial-damp_s1"][0].text
        self.assertIn("on one clock each: wall 115s vs 100s -> x1.150", done)
        self.assertNotIn("timer ", done.split("on one clock each:")[1])


if __name__ == "__main__":
    unittest.main()
