"""Mechanics on real processes: group kill, death detection, pace reading, the estimator, plan validation."""

import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpuharness import launcher as L  # noqa: E402
from gpuharness import plan as P  # noqa: E402
from gpuharness.estimate import Estimator  # noqa: E402
from gpuharness.model import Job  # noqa: E402
from gpuharness.progress import LogReader, parse_line  # noqa: E402


def _ps_has(fragment):
    import subprocess
    out = subprocess.run(["ps", "-A", "-o", "command="], capture_output=True, text=True).stdout
    return any(fragment in line for line in out.splitlines())


class GroupKill(unittest.TestCase):
    def test_wrapper_child_dies_with_the_group(self):
        d = tempfile.mkdtemp()
        tag = f"gpuh-test-{os.getpid()}-{time.time():.0f}"
        cmd = f"bash -c 'python3 -c \"import time; time.sleep(60)\" {tag}; true'"
        st = L.start("w", cmd, 0, d, d)
        time.sleep(0.8)
        self.assertTrue(L.alive(st.pid))
        L.terminate(st.pid)
        for _ in range(50):
            if not L.alive(st.pid):
                break
            time.sleep(0.1)
        L.reap(st.pid)
        self.assertFalse(L.alive(st.pid))
        self.assertFalse(_ps_has(tag), "the wrapper's child survived the group kill")

    def test_exit_code_and_death(self):
        d = tempfile.mkdtemp()
        st = L.start("ok", "exit 3", 1, d, d)
        for _ in range(50):
            if L.exit_code(st.rc_path) is not None:
                break
            time.sleep(0.1)
        self.assertEqual(L.exit_code(st.rc_path), 3)
        st2 = L.start("dies", "sleep 30", 1, d, d)
        time.sleep(0.3)
        L.kill(st2.pid)
        time.sleep(0.5)
        L.reap(st2.pid)
        self.assertFalse(L.alive(st2.pid))
        self.assertIsNone(L.exit_code(st2.rc_path), "a killed job must leave no exit code (it DIED)")

    def test_gpu_env_is_set(self):
        d = tempfile.mkdtemp()
        st = L.start("env", "echo gpu=$CUDA_VISIBLE_DEVICES", 2, d, d)
        for _ in range(50):
            if L.exit_code(st.rc_path) is not None:
                break
            time.sleep(0.1)
        self.assertIn("gpu=2", open(st.log).read())


class Pace(unittest.TestCase):
    def test_shapes(self):
        r = parse_line("step:125/2900 val_loss:4.44817 train_time:344.423s step_avg:2755.38ms")
        self.assertEqual((r.done, r.total, round(r.timer, 1)), (125, 2900, 344.4))
        r = parse_line("step 25/100 train_loss 4.18 step_time 900ms")
        self.assertEqual((r.done, r.per), (25, 0.9))
        r = parse_line(" 12%|#2        | 36/300 [00:41<05:02,  1.14s/it]")
        self.assertEqual((r.done, r.total, r.timer), (36, 300, 41))
        self.assertIsNone(parse_line("TEST_METRICS: final_score=30.0"))

    def test_projection_uses_the_scripts_own_timer(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.log")
        with open(p, "w") as fh:
            fh.write("step:0/2900 val_loss:10.8 train_time:0.000s step_avg:nanms\n")
            fh.write("step:125/2900 val_loss:4.45 train_time:344.4s step_avg:2755ms\n")
        r = LogReader(p)
        pace = r.poll(400.0, 0.0)
        self.assertAlmostEqual(pace.projected_timer_s, 2900 * 344.4 / 125, delta=5)
        self.assertGreater(pace.projected_s, pace.projected_timer_s)

    def test_a_finished_sub_bar_gives_no_pace(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.log")
        open(p, "w").write("Loading weights:   0%|          | 0/434 [00:00<?, ?it/s]\r"
                           "Loading weights: 100%|##########| 434/434 [00:00<00:00, 7929.71it/s]\n")
        self.assertIsNone(LogReader(p).poll(20.0, 0.0))

    def test_result_line(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "x.log")
        open(p, "w").write("loading\nTEST_METRICS: final_score=31.84 mean_retained_fraction=1.0 runtime_seconds=7440\n")
        self.assertTrue(LogReader(p).result_line().startswith("TEST_METRICS: final_score=31.84"))


class Estimates(unittest.TestCase):
    def test_slowdown_carries_across_kinds(self):
        e = Estimator()
        ref_s, ref_l = Job(id="r_s", command="x", candidate="ref", kind="short"), Job(id="r_l", command="x", candidate="ref", kind="long")
        m_s, m_l = Job(id="m_s", command="x", candidate="m", kind="short"), Job(id="m_l", command="x", candidate="m", kind="long")
        e.measured(ref_s, 100)
        e.measured(ref_l, 1000)
        e.measured(m_s, 300)
        self.assertAlmostEqual(e.predict(m_l), 3000, delta=300)


    def test_one_measurement_outweighs_flat_priors(self):
        # G1 Haiku A1: squat's 1,393-s check against the reference's 347 s, with five flat priors per method; A1 kept
        # squat's factor near 1.5 and predicted its GSM8K at 1,796 s (it ran 29,153 s)
        e = Estimator()
        kinds = (("hotpotqa", 1200), ("repobench", 1500), ("v2", 2400), ("gsm8k", 1200), ("passage", 1200))
        jobs = {}
        for c in ("ref", "squat"):
            for k, s in kinds:
                jobs[c, k] = Job(id=f"{c}_{k}", command="x", candidate=c, kind=k, estimate_s=s)
                e.prior(jobs[c, k], s)
        e.measured(jobs["ref", "hotpotqa"], 347)
        e.measured(jobs["squat", "hotpotqa"], 1393)
        self.assertGreater(e.predict(jobs["squat", "gsm8k"]) / e.predict(jobs["ref", "gsm8k"]), 3.0)


    def test_an_overrunning_job_raises_its_cell(self):
        # a job with no pace that has run past its prediction is at least that long: its siblings of the same cell
        # are predicted longer before it ends (KV evaluators print no progress)
        e = Estimator()
        J = {}
        for c in ("ref", "m", "n"):
            for k, s in (("hotpotqa", 1200), ("gsm8k", 1200), ("v2", 2400)):
                J[c, k] = Job(id=f"{c}_{k}", command="x", candidate=c, kind=k, estimate_s=s)
                e.prior(J[c, k], s)
            e.measured(J[c, "hotpotqa"], 350)   # every method's short job at the same pace
        before = e.predict(J["n", "gsm8k"])
        e.running(J["m", "gsm8k"], 6000)
        self.assertGreater(e.predict(J["n", "gsm8k"]), before * 1.3)
        e.measured(J["m", "gsm8k"], 7000)       # the bound gives way to the measurement
        self.assertAlmostEqual(e.predict(J["m", "gsm8k"]), 7000, delta=1)


    def test_the_reference_is_the_yardstick(self):
        # the reference's factor is pinned, so its long GSM8K says GSM8K is long, and a method measured slow on a short
        # workload is predicted slow on GSM8K at once
        e = Estimator()
        e.set_reference(["ref"])
        J = {}
        for c in ("ref", "squat"):
            for k in ("hotpotqa", "gsm8k"):
                J[c, k] = Job(id=f"{c}_{k}", command="x", candidate=c, kind=k, estimate_s=1200)
                e.prior(J[c, k], 1200)
        e.measured(J["ref", "hotpotqa"], 350)
        e.running(J["ref", "gsm8k"], 2000)                 # still running, past its prior
        e.measured(J["squat", "hotpotqa"], 1400)
        self.assertGreater(e.predict(J["squat", "gsm8k"]), 4 * 2000 * 0.8)


class PriorsWithinACell(unittest.TestCase):
    def test_differences_the_plan_gives_are_kept(self):
        e = Estimator()
        jobs = [Job(id=n, command="x", candidate="runs", kind="ablation", estimate_s=s)
                for n, s in (("A", 60), ("B", 120), ("D", 150))]
        for j in jobs:
            e.prior(j, j.estimate_s)
        self.assertGreater(e.predict(jobs[2]), e.predict(jobs[1]))
        self.assertGreater(e.predict(jobs[1]), e.predict(jobs[0]))
        e.measured(jobs[0], 120)   # A took twice its prior: the others scale up with it
        self.assertGreater(e.predict(jobs[2]), 150)


class Plans(unittest.TestCase):
    def plan(self, **over):
        d = {"rule": "r", "candidates": [{"id": "a"}], "jobs": [{"id": "j", "candidate": "a", "command": "true"}]}
        d.update(over)
        return d

    def test_foreach_expands(self):
        b, _, _ = P.load(self.plan(candidates=[{"id": "m1", "branch": "x"}, {"id": "m2", "branch": "y"}], jobs=[
            {"foreach": {"m": ["m1", "m2"], "w": ["qa", "code"]}, "id": "{m}_{w}", "candidate": "{m}", "kind": "{w}",
             "command": "bash jobs/{m}/scripts/{w}.sh", "estimate_s_by": {"qa": 60, "code": 150}}]))
        self.assertEqual(sorted(b.jobs), ["m1_code", "m1_qa", "m2_code", "m2_qa"])
        self.assertEqual(b.jobs["m2_code"].estimate_s, 150)
        self.assertEqual(b.jobs["m2_code"].command, "bash jobs/m2/scripts/code.sh")

    def test_role_by_and_except(self):
        b, _, _ = P.load(self.plan(candidates=[{"id": "m1", "branch": "x"}, {"id": "m2", "branch": "y"}], jobs=[
            {"foreach": {"m": ["m1", "m2"], "w": ["qa", "code"]}, "id": "{m}_{w}", "candidate": "{m}", "kind": "{w}",
             "command": "true", "role_by": {"qa": "check"}, "except": [{"m": "m2", "w": "code"}]}]))
        self.assertEqual(sorted(b.jobs), ["m1_code", "m1_qa", "m2_qa"])
        self.assertEqual(b.jobs["m1_qa"].role.value, "check")
        self.assertEqual(b.jobs["m1_code"].role.value, "needed")

    def test_gpu_assignments_are_stripped(self):
        b, _, warnings = P.load(self.plan(jobs=[{"id": "j", "candidate": "a",
                                                  "command": "CUDA_VISIBLE_DEVICES=3 python3 x.py --seed 1"}]))
        self.assertEqual(b.jobs["j"].command, "python3 x.py --seed 1")
        self.assertTrue(any("CUDA_VISIBLE_DEVICES" in w for w in warnings))

    def test_candidates_need_branches(self):
        with self.assertRaises(P.PlanError):
            P.load(self.plan(candidates=[{"id": "m1"}, {"id": "m2"}],
                             jobs=[{"id": "a", "candidate": "m1", "command": "true"},
                                   {"id": "b", "candidate": "m2", "command": "true"}]))

    def test_refusals(self):
        for bad in (self.plan(jobs=[{"id": "j", "command": ""}]),
                    self.plan(jobs=[{"id": "j", "candidate": "zz", "command": "true"}]),
                    self.plan(jobs=[{"id": "j", "command": "true", "after": ["k"]}]),
                    self.plan(jobs=[{"id": "j", "command": "true", "after": ["k"]},
                                    {"id": "k", "command": "true", "after": ["j"]}]),
                    self.plan(jobs=[{"id": "j", "command": "true"}, {"id": "j", "command": "true"}])):
            with self.assertRaises(P.PlanError):
                P.load(bad)


if __name__ == "__main__":
    unittest.main()
