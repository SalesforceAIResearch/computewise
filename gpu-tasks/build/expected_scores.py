#!/usr/bin/env python3
"""Expected scores for scripted agents, computed in advance from the registry.

usage: python3 -I build/expected_scores.py <registry> <task_id> <policy|all> [--speed 1] [--host <host-terms.json>]
                                          [--emit <dir>] [--json]

A scripted agent uses no model and follows a fixed policy. The policies are the runner's, agreed on 2026-10-20 (the
the study's set in STATE.md, "Measure"). This file implements them independently of the runner's expect.py:
  serial                    task-file order, one job at a time on its widest gpu_options, a 20 s model turn before each launch
  packed                    list scheduling: whenever GPUs are free, launch every job that fits, longest disclosed estimate
                            first (ties by file order), on the lowest-numbered free GPUs, each at its widest option that fits
  packed_cp                 packed, but jobs taken by their remaining critical path first: a job's disclosed estimate plus
                            the longest chain of its `after` descendants (ties: longest estimate, then file order). On a
                            task without dependencies it is packed (the analysis, 2026-10-22: a chain
                            must start first, or a competent agent beats the reference)
  delayed                   packed, after a 300 s sleep
  quit_after_first          the task's first job in file order; the session ends when it exits
  quit_while_running        the task's first job; the session ends at once and the job runs on
  quit_at_first_completion  packed until the first job exits; the session ends then, running jobs run on, nothing more starts
  packed_overlap            the runner's rule, sent 2026-10-20 (runner/scripted/plan.py, overlap_lanes), implemented here
                            from that message without reading plan.py. Every job runs at one width w that divides the box, in
                            box/w lanes of fixed GPUs. Jobs are taken longest first by paced GPU time (the median paced duration
                            of the job's records at width w; ties by file order) and each is put on the lane with the least
                            paced time assigned so far (ties: lowest lane). Within a lane, with pre(k) = the job script's
                            hello_s + pre_s: launch(1) = 0, launch(k) = max(launch(k-1), gpu_free - pre(k)), where gpu_free is
                            the end of the previous job's last paced unit, gaps included. So a successor's start-up runs during
                            its predecessor's GPU work, and the floor is paid once per lane. The agent sleeps between planned
                            launches and ends its session when every job has exited. Without host terms pre(k) is 0 and the
                            policy launches at each predecessor's last GPU unit, as packed does at its exit
The agent runs the task's own launch lines (runs.sh, checks.sh) with two changes: --output_dir=<job_id> (argv_exempt, so replay
still matches, and the scorer maps ranks to task jobs by it), and CUDA_VISIBLE_DEVICES=<its GPUs> for a job narrower than the
box. It waits on its jobs directly: it notices an exit at the next whole millisecond and acts at once. Agent-side durations
(the 20 s turn, the 300 s sleep) are wall seconds, not divided by the speed knob. The box stops at the task's horizon
(horizon_s / speed; the agent never sees it, and no brief states a deadline: the study, 2026-10-20, design/IDEA.md, "No deadline
in the brief"): a job still running then is killed and logs no job_bye, and an agent still waiting is killed with it. A task
file from before the ruling carries deadline_s instead, and is read the same way.

Replay is modelled as gpu-mock implements it, read on 2026-10-20 (experiments/gpu-mock/src/daemon/gpu_daemon.py,
src/torch_patch/replay.py):
  record      gpu-mock's own draw rule (gpu_daemon.Registry._draw, imported by path) over the job's replicate group, with the
              seed gpu-mock's _seed_from_argv reads from the launch, in launch order: the intended exact match. The same
              launches also go through gpu-mock's real lookup(); where today's code gives another record or none, the job
              carries `gpu_mock_today` and the output lists it under known_divergences; so does a launch that gpu-mock
              would match only at the structural stage, since that costs one real optimizer step before replay engages.
  pacing      each rank paces one unit per optimizer step on its GPU (build/common.py:paced_duration states the rule),
              converted by gpu-mock's convert() for the launched GPU count and divided by the speed knob. Unit k is enqueued
              when unit k-1 starts (the client's one-unit lookahead) and runs when the GPU's cursor reaches it.
  GPU index   a rank's daemon GPU is the job's visible device, as CUDA_VISIBLE_DEVICES intends. gpu-mock pacing on LOCAL_RANK
              instead was reported to it on 2026-10-19; until that is fixed, a job narrower than the box paces on GPU 0.
Host time cannot be known from the registry. --host gives it per script basename, in wall seconds, not scaled by the speed
knob (real CPU time does not speed up); they equal recorded seconds only in a speed-1 run. The file may also be the runner's
scripted/host_terms.py output or a registry host-terms file, both of which hold the terms under "terms":
  {"dpo.py": {"hello_s": .., "pre_s": .., "gaps": [[after_step, s], ..], "post_s": ..}}
  Optional per script, from the runner's scripted/host_terms.py: pre_s_first (cold) and pre_s_rest (warm). A job is cold when
  no earlier-launched job of the same script had reached its first paced unit by this job's launch, so the script's cached map
  outputs cannot exist yet; otherwise it is warm. Without the split every job pays pre_s. The rule is per script, because the
  datasets cache is keyed by the dataset and the map function, which the script fixes. The runner's plan.py and expect.py use the
  same per-script rule (agreed 2026-10-20; on trl-regression-battery both make dpo-seed2 cold on lane 3, after the SFT jobs).
  hello_s is launch to the process's start (job_hello's t_start; since 2026-10-20 gpu-mock logs job_hello at the first CUDA use
  and puts the process start in t_start); pre_s is the process's start to its first paced unit; gaps are GPU-idle seconds after a
  step; post_s is the last unit's end to job_bye. Without --host every term is zero: the ideal case, which no real run can beat.

The numbers carry the field names of the runner's scorer (experiments/runner/score/score_run.py and idle.py, read 2026-10-20)
and follow design/IDEA.md, "Idle time", including its cause order and the split of job host phase into job_host_phase and
startup_not_overlapped (startup_split below), but they are computed here from the simulated timeline and
not from an occupancy log. One deliberate difference: a killed job's busy time ends at the kill, while a scorer that rebuilds
busy time from enqueue events counts up to two queued units past it. --emit writes, per policy, a synthetic run directory with
exactly this timeline (occupancy.jsonl, resources.jsonl with one launcher process per launch, status.json in run_one.py's shape, task.json, transcript/session/scripted/<policy>.jsonl in Claude Code's
session shape, timestamps in whole milliseconds), so the scorer can be run on it and compared. Standard library only.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402

DAEMON = os.path.join(C.GPU_MOCK_SRC, "daemon", "gpu_daemon.py")
POLICIES = ("serial", "packed", "packed_cp", "delayed", "quit_after_first", "quit_while_running", "quit_at_first_completion", "packed_overlap")
THINK_S, DELAY_S = 20.0, 300.0
T0_MS, M0 = 1_800_000_000_000, 100_000.0       # fixture clocks at the agent's start: wall (epoch ms, an integer) and monotonic
T0 = T0_MS / 1000.0
EPS = 1e-9
WAIT_RE = re.compile(r"^(sleep|wait)\b")


def val(x):
    return x.get("value") if isinstance(x, dict) else x


def ms_up(t):
    return math.ceil(round(t * 1000.0, 6)) / 1000.0


def iso(t):
    d = datetime.datetime.fromtimestamp(T0_MS // 1000, datetime.timezone.utc) + datetime.timedelta(milliseconds=round(t * 1000))
    return d.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_daemon():
    if not os.path.exists(DAEMON):
        raise SystemExit(f"gpu-mock's daemon is missing: {DAEMON}")
    spec = importlib.util.spec_from_file_location("gpumock_daemon", DAEMON)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gpumock_daemon"] = mod          # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


class GpuMock:
    """The intended draw (gpu-mock's _draw over the job's replicate group) and today's real lookup, side by side."""

    def __init__(self, reg, task_id=None):
        self.gd = load_daemon()
        self.intended = self.gd.Registry(Path(reg))
        self.today = self.gd.Registry(Path(reg))
        if task_id and hasattr(self.today, "load_task"):
            # The daemon loads the run's task file (its draw then keeps to the records the task lists, never one run twice),
            # so "today" is drawn as a task run is.
            self.today.load_task(Path(reg) / "tasks" / f"{task_id}.json")
        self.sha = hashlib.sha256(open(DAEMON, "rb").read()).hexdigest()

    def draw(self, job, argv, entry_hash):
        cands = self.intended._groups.get(job["replicate_group"])
        if not cands:
            raise SystemExit(f"{job['job_id']}: gpu-mock has no records in replicate group {job['replicate_group']}")
        want = self.gd._seed_from_argv(argv)
        rec, draw = self.intended._draw(cands, want)
        if rec is None:
            raise SystemExit(f"{job['job_id']}: no drawable record ({draw})")
        key = job["job_id"]
        r, mode, info = self.today.lookup(None, entry_hash, argv, job_key=key, stage="exact")
        stage = "exact"
        if r is None:
            r, mode, info = self.today.lookup(rec["identity"].get("fingerprint"), entry_hash, argv, job_key=key, stage="structural")
            stage = "structural, after the exact lookup missed"
        today = {"record_id": (r or {}).get("identity", {}).get("record_id"), "mode": mode, "match": info.get("match"),
                 "draw": info.get("draw"), "stage": stage}
        return rec, draw, want, today


def bind_recorded_winners(task, reg):
    """A rung-2 task as an oracle scripted agent runs it: each decision-bound job becomes the job the recorded screen values
    bind it to (recorded_winner: its records and, where the task gives one, its launch), after every screen. Only decisions whose
    winners the records fix (recorded_outcome.winners) can be bound; any other is refused."""
    t = json.loads(json.dumps(task))
    for d in t["decisions"]:
        if not (d.get("recorded_outcome") or {}).get("winners"):
            raise SystemExit(f"{t['task_id']}: decision {d['decision_id']} has no recorded winners, so a scripted plan cannot bind it")
    for j in t["jobs"]:
        w = j.get("recorded_winner")
        if j.get("bind"):
            if not w or not w.get("launch"):
                raise SystemExit(f"{t['task_id']}: {j['job_id']} is decision-bound with no recorded winner launch")
            j["records"], j["launch"] = w["records"], w["launch"]
            j["draw"] = "exact_seed"
            j.pop("candidate_pool", None)
            rec = json.load(open(os.path.join(reg, "records", w["records"][0] + ".json")))
            j["replicate_group"] = rec["identity"]["replicate_group"]
            j["seed"] = rec["identity"]["seed"]["value"]
    return t


def host_terms_of(terms, script):
    """A script's host terms: by its path in the task tree when the terms file keys it so (two scripts of the task share a
    basename, build/derive_host_terms.py), else by basename."""
    terms = terms or {}
    return terms.get(os.path.normpath(script)) or terms.get(os.path.basename(script)) or {}


def isolates_by_output_dir(job):
    """True when the task's launch line has an --output_dir the agent would give each job its own value of (the TRL tasks;
    the records exempt it). A modded-nanogpt script takes no such flag: it writes logs/<uuid>.txt, unique per run."""
    return "--output_dir" in (job.get("launch") or "")


def wrapper_argv(text):
    """The python argv a job's shell wrapper runs: its last line that runs a .py script (`exec python tools/x.py ...`), after
    the interpreter. The TTC twins' `bash jobs/<cell>.sh` (independent Sol, 2026-11-02): the wrapper is staged, so this is what
    runs on the box. None when no line runs one."""
    for line in reversed((text or "").splitlines()):
        if line.lstrip().startswith("#"):
            continue
        try:
            toks = shlex.split(line)
        except ValueError:
            continue
        while toks and not toks[0].endswith(".py"):
            toks.pop(0)
        if toks:
            return toks
    return None


def job_argv(job, files=None):
    """sys.argv of the job's python process: the launch line after its launcher, with --output_dir=<job_id> where the task's
    launch line takes one. Leading VAR=value assignments (a SEED the script reads) are not argv. `files`, the task's staged path
    -> registry file map, lets a `bash <wrapper>.sh` launch with no script_path be read through its wrapper."""
    toks = shlex.split(job["launch"].split("  #")[0])
    raw = list(toks)
    while toks and not toks[0].endswith(".py"):
        toks.pop(0)
    if not toks and job.get("script_path"):
        # a job script (MLS-Bench KV: `bash methods/<m>/scripts/<w>.sh`) runs the method's evaluator with the record's own argv
        # after argv[0]; its --max-examples comes from the environment, which the record's identity exempts
        return [job["script_path"]] + list(job.get("record_argv_tail") or [])
    if not toks and files:
        sh = next((t for t in raw if t.endswith(".sh") and t in files), None)
        got = wrapper_argv(open(files[sh]).read()) if sh else None
        if got:
            return got
    out, i = [], 0
    while i < len(toks):
        if toks[i].startswith("--output_dir="):
            i += 1
            continue
        if toks[i] == "--output_dir":
            i += 2
            continue
        out.append(toks[i])
        i += 1
    return out + ([f"--output_dir={job['job_id']}"] if isolates_by_output_dir(job) else [])


def launch_command(job, gpus, box):
    line = job["launch"].split("  #")[0]
    toks = shlex.split(line)
    out, i = [], 0
    while i < len(toks):
        if toks[i].startswith("--output_dir="):
            i += 1
            continue
        if toks[i] == "--output_dir":
            i += 2
            continue
        out.append(toks[i])
        i += 1
    cmd = " ".join(shlex.quote(t) if "=" not in t or t.startswith("-") else t for t in out) + \
        (f" --output_dir={job['job_id']}" if isolates_by_output_dir(job) else "")
    return cmd if len(gpus) == box else f"CUDA_VISIBLE_DEVICES={','.join(map(str, gpus))} {cmd}"


def paced_units_k(rec, k):
    """A per_example record's paced units for a launch of its first k examples (k None or 0: all of them): one unit per
    example of (wall - load) / n recorded seconds (experiments/gpu-mock/docs/MLS-KV-SERVING.md §1.3). The load is host time
    before them, carried by the measured host terms' start-up, not a unit, as gpu-mock paces it: real on the box."""
    sv = rec["materialisation"]["serving"]
    n = int(sv["n_examples_full"]["value"])
    load = float((sv.get("load_s") or {}).get("value") or 0.0)
    m = n if not k or k >= n else int(k)
    per = max(0.0, C.paced_duration(rec, 1) - load) / n
    return [("example", per)] * m


def whole_job_units(rec):
    """A whole_job record's paced units (gpu-mock next/ r8, replay._pace_whole_job): n_units units inside the seam, each its
    share of the recorded wall_total_s (unit_shares, else equal shares), on the job's GPU. Nothing is subtracted for start-up and
    nothing is converted: the record's wall is the seamed function's time, and everything outside the seam (imports, data and
    model loads) is real host time on the box, carried here by the host terms where a task has them (gpu-tasks, 2026-11-02, for
    the TTC twins, their full-table controls and K2's late pole)."""
    sv = rec["materialisation"]["serving"]
    wall = float(val(rec["timing"].get("wall_total_s")) or 0.0)
    try:
        n = max(1, int(val(sv.get("n_units")) or 1))
    except (TypeError, ValueError):
        n = 1
    shares = sv.get("unit_shares")
    if not (isinstance(shares, list) and len(shares) == n):
        shares = [1.0 / n] * n
    return [("whole_job", max(0.0, wall * float(s))) for s in shares]


def paced_units(rec, n):
    """Recorded seconds per paced unit, in order, as replay.py paces them on n GPUs."""
    tm = rec["timing"]
    vl = (rec.get("series") or {}).get("val_loss")
    ev = val(tm.get("eval_s"))
    vsteps = (vl.get("steps") or []) if isinstance(vl, dict) else []
    sv = (rec.get("materialisation") or {}).get("serving") or {}
    if sv.get("mode") == "per_example":
        # gpu-mock's per_example serving (experiments/gpu-mock/docs/MLS-KV-SERVING.md §1.3): the model load once, as host time
        # before the first paced example, then n_examples_full examples at (wall - load) / n each. `k`, a --max-examples subset,
        # is read from the caller's launch (paced_units_k); here the whole record
        return paced_units_k(rec, None)
    if sv.get("mode") == "whole_job":
        return whole_job_units(rec)
    if tm.get("pacing") not in ("per_step", "per_logged_interval"):
        raise SystemExit(f"{rec['identity']['record_id']}: pacing {tm.get('pacing')!r} is not per step")
    g0 = rec["reference"]["gpu_count"]
    f, prov = C.time_factor(val(rec["launch"]["batch_policy"]), g0, n)
    if f is None:
        raise SystemExit(f"{rec['identity']['record_id']} on {n} GPUs: {prov}")
    steady = val(tm.get("steady_step_s")) or rec.get("per_step_gpu_s")
    first, su = val(tm.get("first_step_s")), val(tm.get("startup_in_train_s")) or 0.0
    units = [("startup", su)] if (not first and su > 0) else []
    # A validation is paced by replay when the script's first eval-mode forward after training begins (replay.py,
    # on_eval_phase_begin: eval_s / validations each), so it follows the step the record validated at, on the GPU.
    per_val = (ev * f / len(vsteps)) if (ev and vsteps) else 0.0
    at = collections.Counter(int(s) for s in vsteps) if per_val else collections.Counter()
    # a validation before the first step (at step 0) comes right after the start-up, as replay paces them: start-up at engage,
    # then the first eval phase (replay.py _engage, on_eval_phase_begin)
    for s in sorted(at):
        if s < 1:
            units.append(("val", per_val * at[s]))
    for k in range(1, rec["steps"] + 1):
        units.append(("step", (max(0.0, first - steady) + steady * f) if (k == 1 and first) else steady * f))
        for _ in range(at.get(k, 0)):
            units.append(("val", per_val))
    for s in sorted(at):
        if s > rec["steps"]:
            units.append(("val", per_val * at[s]))
    total, want = sum(u for _, u in units), C.paced_duration(rec, n)
    if abs(total - want) > 1e-6:
        raise SystemExit(f"{rec['identity']['record_id']}: paced units sum to {total}, common.paced_duration says {want}")
    return units


class Agent:
    """The scripted agent's transcript, in Claude Code's session-JSONL shape, and its activity intervals."""

    def __init__(self, brief, policy):
        self.records, self.turns, self.tools, self.open = [], [], [], {}
        self.n, self.last_input, self.policy = 0, 0.0, policy
        self.records.append({"type": "user", "timestamp": iso(0.0), "scripted": True,
                             "message": {"role": "user", "content": brief}})

    def _assistant(self, t, content):
        self.n += 1
        mid = f"msg_scripted_{self.n:04d}"
        self.turns.append((self.last_input, t))
        self.records.append({"type": "assistant", "timestamp": iso(t), "scripted": True,
                             "message": {"id": mid, "role": "assistant", "content": content}})
        return mid

    def call(self, t, command, background):
        tid = f"toolu_scripted_{self.n + 1:04d}"
        self._assistant(t, [{"type": "tool_use", "id": tid, "name": "Bash",
                             "input": {"command": command, "description": "scripted agent",
                                       **({"run_in_background": True} if background else {})}}])
        kind = "wait_or_poll" if (not background and WAIT_RE.match(command)) else "non_gpu_tool"
        self.open[tid] = (t, kind)
        return tid

    def write(self, t, file_path, content):
        """A Write tool call at t that completes at once: the end of a model turn that wrote a file (the scripted preparer)."""
        tid = f"toolu_scripted_{self.n + 1:04d}"
        self._assistant(t, [{"type": "tool_use", "id": tid, "name": "Write",
                             "input": {"file_path": file_path, "content": content}}])
        self.open[tid] = (t, "non_gpu_tool")
        self.result(t, tid, f"wrote {file_path}")
        return tid

    def result(self, t, tid, text=""):
        t0, kind = self.open.pop(tid)
        self.tools.append((t0, t, kind))
        self.records.append({"type": "user", "timestamp": iso(t), "scripted": True,
                             "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid,
                                                                       "content": text, "is_error": False}]}})
        self.last_input = t

    def end(self, t, killed):
        if killed:                               # killed with the box: an open call stays open in the transcript
            for tid, (t0, kind) in self.open.items():
                self.tools.append((t0, t, kind))
            self.open.clear()
            return
        self._assistant(t, [{"type": "text", "text": "Done."}])


class Sim:
    def __init__(self, task, reg, a, policy):
        self.task, self.a, self.policy = task, a, policy
        self.jobs, self.box = task["jobs"], task["box"]["gpus"]
        self.D = float(task.get("horizon_s") or task["deadline_s"]) / a.speed
        self.gm = GpuMock(reg, task.get("task_id"))
        self.files = {f["path"]: os.path.join(reg, f["from"]) for f in task["files"] if str(f.get("from", "")).startswith("artifacts/")}
        self.literal = {f["path"]: f["content"] for f in task["files"] if "content" in f}
        self.reg = reg
        self.agent = Agent(task["brief"]["text"], policy)
        self.runs = {}
        self.ready = {}                          # script path -> earliest first-paced-unit time among its launched jobs

    def start(self, i, gpus, L):
        job = self.jobs[i]
        argv = job_argv(job, self.files)
        if len(argv) == 1 and job.get("script_path") and not job.get("record_argv_tail"):
            # a job script (`bash methods/.../scripts/<w>.sh`: MLS-Bench KV's per_example jobs, K2's whole_job late pole) runs its
            # evaluator with the record's own flags after argv[0], so they reach gpu-mock's lookup as on the box. Sending the bare
            # path made today's lookup miss every KV job (97-210 false known_divergences per KV package; checked on the daemon
            # b302b632d2e3: bare path -> model/none, path + flags -> replay/exact/exact_seed). Fixed on the analysis's
            # word, 2026-11-02 14:24 MST; makespans never depended on it (they pace the intended record)
            r0 = self.gm.intended._by_id.get((job.get("records") or [None])[0]) or {}
            if ((r0.get("materialisation") or {}).get("serving") or {}).get("mode") in ("whole_job", "per_example"):
                argv = argv + list((r0.get("identity") or {}).get("argv") or [])[1:]
        path = self.files.get(argv[0])
        prep = job.get("prep") or {}
        if not path and argv[0] in (prep.get("produces") or []) and prep.get("reference"):
            path = os.path.join(self.reg, prep["reference"])      # the scripted preparer's copy of the reference
        if path:
            h = C.sha256_bytes(open(path, "rb").read())
        elif argv[0] in self.literal:
            # a script the task file carries as content, staged byte for byte by stage_task (the TTC adapter, independent Sol,
            # 2026-11-04, so its bytes move the task hash)
            h = C.sha256_text(self.literal[argv[0]])
        else:
            raise SystemExit(f"{job['job_id']}: {argv[0]} is not a staged script of the task")
        rec, draw, want, today = self.gm.draw(job, argv, h)
        host = host_terms_of(self.a.host_terms, argv[0])
        gaps = {int(s): float(g) for s, g in host.get("gaps", [])}
        sp = self.a.speed
        hello = L + float(host.get("hello_s", 0.0))
        su_kind, pre = self.startup(i, L)
        sc = os.path.normpath(argv[0])
        self.ready[sc] = min(self.ready.get(sc, math.inf), L + pre)
        t, step, prev_start, after_gap, units = L + pre, 0, None, True, []
        # In-run host time: per gap class where the terms carry it (build/derive_host_terms.py per_event: modded-nanogpt, so it
        # transfers between a prefix and a full run; common.gap_classes is the one rule both use), else the measured lump after
        # one step (gaps). Wall seconds.
        pe = host.get("per_event") or {}
        plan_units = (paced_units_k(rec, job.get("max_examples"))
                      if ((rec.get("materialisation") or {}).get("serving") or {}).get("mode") == "per_example"
                      else paced_units(rec, len(gpus)))
        cls = (C.gap_classes(C.unit_kinds([rs for _, rs in plan_units], [k == "val" for k, _ in plan_units]),
                             set(C.shard_load_steps(rec))) if pe else [None] * len(plan_units))
        # per_example records (MLS-KV): the real host work each example does outside the serving seam (the selection evaluator
        # tokenises every prompt for real: on W0-mlskv-check10b a mean 0.69 s per LongBench-v2 example, median 0.27 s, 13.4 s at
        # most), measured per script and workload into the terms: example_gaps_by_workload, the gap before each example after the
        # first, by index (it is the same prompt's tokenising in every launch: the four v2 launches' gap vectors correlate at
        # 1.000), else example_s_by_workload, one mean for every example. Wall seconds, not scaled by the speed knob. How the box
        # combines it with the paced slot is example_gap_mode: "additive" (gpu-mock d1da and earlier: slot + gap per example, the
        # host time the recorded wall already holds counted twice) or "overlap" (from 73dc, 2026-10-23: the daemon back-dates each
        # unit by its host gap, so an example takes max(slot, gap) and the GPU slot starts when the previous example drained)
        pe_mode = None
        ex_gaps = []
        if ((rec.get("materialisation") or {}).get("serving") or {}).get("mode") == "per_example":
            argv_r = rec["identity"].get("argv") or []
            wl = argv_r[argv_r.index("--workload") + 1].replace("_", "-") if "--workload" in argv_r else None
            vec = (host.get("example_gaps_by_workload") or {}).get(wl)
            one = float((host.get("example_s_by_workload") or {}).get(wl) or 0.0)
            ex_gaps = [float(x) for x in vec] if vec else [one] * max(0, len(plan_units) - 1)
            pe_mode = host.get("example_gap_mode") or "additive"
            if pe_mode not in ("additive", "overlap"):
                raise SystemExit(f"{argv[0]}: example_gap_mode {pe_mode!r} is neither additive nor overlap")
        for n_u, (kind, rs) in enumerate(plan_units):
            g = float(pe.get(f"{cls[n_u]}_s") or 0.0) if (pe and cls[n_u]) else 0.0
            gi = (ex_gaps[n_u - 1] if n_u - 1 < len(ex_gaps) else 0.0) if (pe_mode and n_u > 0) else 0.0
            if pe_mode == "overlap":
                # t is when the previous example drained; the unit is back-dated there, the script resumes at max(slot, gap)
                c = rs / sp
                units.append({"enq": t + gi, "start": t, "end": t + c, "recorded_s": rs, "host_gap_s": gi if n_u > 0 else 0.0})
                prev_start, t, after_gap = t, t + max(c, gi), False
                continue
            g += gi
            if g > 0:
                t += g
                after_gap = True
            c = rs / sp
            enq = t if (after_gap or prev_start is None) else prev_start
            units.append({"enq": enq, "start": t, "end": t + c, "recorded_s": rs})
            prev_start, t, after_gap = t, t + c, False
            if kind == "step":
                step += 1
                if not pe and gaps.get(step, 0.0) > 0:
                    t += gaps[step]
                    after_gap = True
        end = (max(units[-1]["end"], t) if pe_mode == "overlap" else units[-1]["end"]) if units else hello
        bye = end + float(host.get("post_s", 0.0))
        killed = bye > self.D + EPS
        run = {"i": i, "job_id": job["job_id"], "gpus": gpus, "launch": L, "hello": min(hello, self.D), "bye": self.D if killed else bye,
               "killed": killed, "record_id": rec["identity"]["record_id"], "draw": draw, "seed_wanted": want, "argv": argv,
               "units": [u for u in units if u["enq"] <= self.D + EPS], "paced_recorded_s": sum(u["recorded_s"] for u in units),
               "notice": ms_up(bye), "gpu_mock_today": today, "command": launch_command(job, gpus, self.box), "startup": su_kind}
        run["busy"] = [(u["start"], min(u["end"], run["bye"])) for u in units if u["start"] < run["bye"] - EPS]
        self.runs[i] = run
        tid = self.agent.call(L, run["command"], background=True)
        self.agent.result(L, tid, f"started {job['job_id']}")
        return run

    def startup(self, i, L):
        """(which start-up a job launched at L pays, its wall seconds from launch to its first paced unit)."""
        sc = os.path.normpath(job_argv(self.jobs[i], self.files)[0])
        h = host_terms_of(self.a.host_terms, sc)
        if not h:
            return "none", 0.0
        hello = float(h.get("hello_s", 0.0))
        if "pre_s_first" in h and "pre_s_rest" in h:
            first = sc not in self.ready or L < self.ready[sc] - EPS
            return ("first", hello + float(h["pre_s_first"])) if first else ("rest", hello + float(h["pre_s_rest"]))
        return "pre_s", hello + float(h.get("pre_s", 0.0))

    def overlap_launch(self, i, earliest, gpu_free):
        """packed_overlap's launch time for job i: as late as lets its start-up end when the GPUs free, never before `earliest`.
        The start-up it pays depends on the launch time, so try the cheaper one first and keep it if it is consistent."""
        tl = max(earliest, gpu_free - self.startup(i, gpu_free)[1])
        kind, pre = self.startup(i, tl)
        if kind == "first":
            tl = max(earliest, gpu_free - pre)
        return tl

    def overlap_key(self, i, w):
        """A job's paced GPU time for lane assignment: the median paced duration of its records at width w, recorded seconds."""
        ds = [C.paced_duration(self.gm.intended._by_id[rid], w) for rid in (self.jobs[i].get("records") or [])]
        ds = sorted(d for d in ds if d is not None)
        if not ds:
            raise SystemExit(f"{self.jobs[i]['job_id']}: no paced duration at width {w}")
        n = len(ds)
        return ds[n // 2] if n % 2 else (ds[n // 2 - 1] + ds[n // 2]) / 2

    def run_overlap(self):
        ag, D = self.agent, self.D
        n = len(self.jobs)
        widths = {self.widest(i, self.box) for i in range(n)}
        if len(widths) != 1 or None in widths or self.box % next(iter(widths)):
            raise SystemExit(f"packed_overlap needs every job at one width that divides the box; widths here: {sorted(widths, key=str)}")
        w = widths.pop()
        L = self.box // w
        key = {i: self.overlap_key(i, w) for i in range(n)}
        # Lane assignment is LPT within dependency levels: every job is placed after all jobs of a lower level, so the
        # lowest-level unlaunched job is always at the head of a lane with its dependencies launched, and no lane waits on a
        # job queued behind it in another lane.
        ids = {j["job_id"]: k for k, j in enumerate(self.jobs)}
        level = {}
        def lvl(i):
            if i not in level:
                level[i] = 1 + max([lvl(ids[d]) for d in self.jobs[i].get("after") or [] if d in ids], default=-1)
            return level[i]
        lanes, load = [[] for _ in range(L)], [0.0] * L
        for i in sorted(range(n), key=lambda i: (lvl(i), -key[i], i)):
            l = min(range(L), key=lambda l: (load[l], l))
            lanes[l].append(i)
            load[l] += key[i]
        pos, last_launch, gpu_free, t_agent = [0] * L, [0.0] * L, [0.0] * L, self.t0
        while True:
            cands = []
            for l in range(L):
                if pos[l] < len(lanes[l]):
                    i = lanes[l][pos[l]]
                    dep = self.deps_end(i)
                    if dep is None:
                        continue                     # a dependency has not been launched yet
                    tl = self.t0 if pos[l] == 0 else self.overlap_launch(i, last_launch[l], gpu_free[l])
                    cands.append((max(tl, dep), l, i))
            if not cands:
                break
            tl, l, i = min(cands)
            if tl >= D - EPS:
                break
            if tl > t_agent + EPS:
                tid = ag.call(t_agent, f"sleep {tl - t_agent:.3f}", background=False)
                ag.result(tl, tid)
                t_agent = tl
            r = self.start(i, list(range(l * w, (l + 1) * w)), tl)
            r["lane"] = l
            pos[l], last_launch[l] = pos[l] + 1, tl
            gpu_free[l] = r["units"][-1]["end"] if r["units"] else r["hello"]
        end = max(r["notice"] for r in self.runs.values())
        tid = ag.call(t_agent, "wait", background=False)
        if end >= D - EPS:
            ag.end(D, killed=True)
            return D, True
        ag.result(end, tid, "all jobs exited 0")
        ag.end(end, killed=False)
        return end, False

    def prep_s(self, i):
        """The scripted preparer's time for job i (prep.scripted_s): writing its script into place before it can launch."""
        return float(((self.jobs[i].get("prep") or {}).get("scripted_s")) or 0.0)

    def prep_group(self, i):
        """The jobs one preparation serves (prep.shared_by: H4's recipe, one data command for its three runs), else job i alone."""
        return tuple(sorted((self.jobs[i].get("prep") or {}).get("shared_by") or [self.jobs[i]["job_id"]]))

    def group_order(self, order):
        """runner/scripted/plan.py's pool order when preparations are shared: each group's jobs together, at its first
        member's place, so the agent launches a group as soon as its command exits."""
        if not any((self.jobs[i].get("prep") or {}).get("shared_by") for i in order):
            return order
        first = {}
        for k, i in enumerate(order):
            first.setdefault(self.prep_group(i), k)
        return sorted(order, key=lambda i: (first[self.prep_group(i)], order.index(i)))

    def prepare(self, i, t):
        """The scripted preparer prepares job i from t: a foreground call for prep.scripted_s seconds that ends with the
        reference script copied to prep.produces[0], or with its data command's exit (prep.scripted_command). A job whose
        shared preparation already ran costs nothing (plan.py prepares a shared_by group once, for its first job). Returns
        the time it ends. Wall seconds: host time, not scaled by speed."""
        prep = self.jobs[i]["prep"]
        done = getattr(self, "_prepared_groups", None)
        if done is None:
            done = self._prepared_groups = set()
        g = self.prep_group(i)
        if g in done:
            return t
        done.add(g)
        dest = (prep.get("produces") or ["?"])[0]
        t2 = t + self.prep_s(i)
        if prep.get("scripted_command"):
            # a data preparation (cpu-tasks' H4): one foreground command, real CPU work in the box
            tid = self.agent.call(t, prep["scripted_command"], background=False)
            self.agent.result(t2, tid, "exit 0")
            return t2
        # As the runner's scripted agent does it (2026-10-20): prep.scripted_s of model turn, then a Write of the reference to
        # prep.produces[0]. The time is the model's, so the scorer counts it as model_turn, as a model writing the script would be.
        self.agent.write(t2, dest, f"<the reference script {prep.get('reference')}>")
        return t2

    def deps_done(self, i, t):
        """True when every `after` dependency of job i has exited by time t: a scripted agent follows the order the brief
        states and launches a dependent job only once its predecessors have finished (noticed at their exit)."""
        ids = {j["job_id"]: k for k, j in enumerate(self.jobs)}
        for d in self.jobs[i].get("after") or []:
            r = self.runs.get(ids.get(d))
            if r is None or r["killed"] or r["notice"] > t + EPS:
                return False
        return True

    def deps_end(self, i):
        """When job i's last dependency is noticed to have exited, or None if one has not been launched."""
        ids = {j["job_id"]: k for k, j in enumerate(self.jobs)}
        ends = []
        for d in self.jobs[i].get("after") or []:
            r = self.runs.get(ids.get(d))
            if r is None:
                return None
            ends.append(r["notice"])
        return max(ends, default=0.0)

    def widest(self, i, nfree):
        return next((o for o in sorted(self.jobs[i]["gpu_options"], reverse=True) if o <= nfree), None)

    def data_step(self):
        """A data-preparation task's shared step (task.data_prep): the scripted preparer runs the reference preparation in one
        foreground call of data_prep.scripted_s wall seconds before it launches anything. Returns when it ends (0 if none)."""
        dp = self.task.get("data_prep") or {}
        s = float(dp.get("scripted_s") or 0.0)
        if s <= 0:
            return 0.0
        tid = self.agent.call(0.0, dp.get("scripted_command") or "<the reference data preparation>", background=False)
        self.agent.result(min(s, self.D), tid, "done")
        return s

    def run(self):
        p, D, ag = self.policy, self.D, self.agent
        self.t0 = self.data_step()
        if self.t0 >= D - EPS:
            ag.end(D, killed=True)
            return D, True
        if p == "packed_overlap":
            return self.run_overlap()
        if p == "serial":
            t = self.t0
            for i in range(len(self.jobs)):
                L = t + THINK_S
                if self.prep_s(i) > 0:
                    L = self.prepare(i, L)
                if L >= D - EPS:
                    ag.end(D, killed=True)
                    return D, True
                r = self.start(i, list(range(self.widest(i, self.box))), L)
                tid = ag.call(L, f"wait {r['job_id']}", background=False)
                if r["notice"] >= D - EPS:
                    ag.end(D, killed=True)
                    return D, True
                t = r["notice"]
                ag.result(t, tid, f"{r['job_id']} exited 0")
            ag.end(t, killed=False)
            return t, False
        order = list(range(len(self.jobs)))
        if p in ("quit_after_first", "quit_while_running"):
            order = [0]
        else:
            est = lambda i: (((self.jobs[i].get("disclosure") or {}).get("estimate") or {}).get("median_s") or 0.0)
            if p == "packed_cp":
                ids = {j["job_id"]: k for k, j in enumerate(self.jobs)}
                kids = {k: [] for k in range(len(self.jobs))}
                for k, j in enumerate(self.jobs):
                    for d in j.get("after") or []:
                        if d in ids:
                            kids[ids[d]].append(k)
                tail = {}
                def cp(i):
                    if i not in tail:
                        tail[i] = est(i) + max([cp(k) for k in kids[i]], default=0.0)
                    return tail[i]
                order.sort(key=lambda i: (-cp(i), -est(i), i))
            else:
                order.sort(key=lambda i: (-est(i), i))
        order = self.group_order(order)
        pending, running, free, t = list(order), {}, set(range(self.box)), self.t0
        if p == "delayed":
            tid = ag.call(t, f"sleep {DELAY_S:g}", background=False)
            if t + DELAY_S >= D:
                ag.end(D, killed=True)
                return D, True
            t += DELAY_S
            ag.result(t, tid)
        to_prepare = [i for i in order if self.prep_s(i) > 0]
        prepared = {i for i in order if self.prep_s(i) <= 0}
        # launch order among prepared jobs: as runner/scripted/agent.py's run_prep_pool keeps it, jobs that needed no
        # preparation first in the pool's order, then each prepared job in the order its preparation ended (W0-h1-check10
        # and -r2, 2026-10-23: a just-prepared job launched behind ready jobs that had waited for GPUs)
        ready_rank = {i: k for k, i in enumerate(i for i in order if i in prepared)}
        while True:
            for i in sorted(pending, key=lambda i: ready_rank.get(i, math.inf)):
                if i not in prepared or not self.deps_done(i, t):
                    continue
                w = self.widest(i, len(free))
                if w is None:
                    continue
                g = sorted(free)[:w]
                free -= set(g)
                pending.remove(i)
                running[i] = self.start(i, g, t)
            if to_prepare and p != "quit_while_running" or (to_prepare and not running):
                # A scripted preparer works on one job at a time, in launch order, while the launched jobs run; it sees a
                # job's exit when it next looks, at the end of the preparation in hand.
                i = to_prepare.pop(0)
                t = self.prepare(i, t)
                if t >= D - EPS:
                    ag.end(D, killed=True)
                    return D, True
                # the group's other jobs were prepared by the same command (plan.py: prepared by it, no step of their own)
                for k in [i] + [k for k in to_prepare if self.prep_group(k) == self.prep_group(i)]:
                    if k != i:
                        to_prepare.remove(k)
                    prepared.add(k)
                    ready_rank[k] = len(ready_rank)
                for k in [k for k, r in running.items() if r["notice"] <= t + EPS]:
                    free |= set(running.pop(k)["gpus"])
                continue
            if p == "quit_while_running" or not running:
                ag.end(t, killed=False)
                return t, False
            tn = min(r["notice"] for r in running.values())
            tid = ag.call(t, "wait -n", background=False)
            if tn >= D - EPS:
                ag.end(D, killed=True)
                return D, True
            t = tn
            ended = [i for i, r in running.items() if r["notice"] <= t + EPS]
            ag.result(t, tid, " ".join(f"{running[i]['job_id']} exited 0" for i in ended))
            for i in ended:
                free |= set(running.pop(i)["gpus"])
            if p in ("quit_after_first", "quit_at_first_completion"):
                ag.end(t, killed=False)
                return t, False


def union(ivs):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def total(ivs):
    return sum(e - s for s, e in ivs)


def complement(ivs, lo, hi):
    """[lo, hi) minus the union of ivs."""
    out, t = [], lo
    for s, e in union([list(iv) for iv in ivs]):
        if e <= t or s >= hi:
            continue
        if s > t:
            out.append([t, min(s, hi)])
        t = max(t, e)
    if t < hi:
        out.append([t, hi])
    return out


def intersect(A, B):
    out, B = [], union([list(iv) for iv in B])
    for s, e in union([list(iv) for iv in A]):
        for x, y in B:
            lo, hi = max(s, x), min(e, y)
            if hi > lo + EPS:
                out.append([lo, hi])
    return union(out)


def minus(A, B):
    return [iv for s, e in union([list(iv) for iv in A]) for iv in complement(B, s, e)]


def latest(ivs, amount):
    """The latest `amount` seconds of the intervals."""
    out, need = [], amount
    for s, e in reversed(union([list(iv) for iv in ivs])):
        if need <= EPS:
            break
        take = min(need, e - s)
        out.append([e - take, e])
        need -= take
    return union(out)


def hideable_startup(a, p, ready, H):
    """How much more of a start-up [a, a + p) could have been covered by H, had it started anywhere in [ready, a]:
    max over x in [ready, a] of |H ∩ [x, x + p)| - |H ∩ [a, a + p)|, floored at 0. The overlap is piecewise linear in x, so
    the maximum is at an end of the range or where x or x + p meets an edge of H. ready None: never launchable early."""
    ov = lambda lo, hi: sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in H)
    if ready is None or ready >= a - EPS or p <= EPS:
        return 0.0
    cands = {ready, a} | {x for s, e in H for edge in (s, e) for x in (edge, edge - p) if ready <= x <= a}
    return max(0.0, max(ov(x, x + p) for x in cands) - ov(a, a + p))


def startup_split(sim):
    """The split of host-phase idle into job_host_phase (the environment's) and startup_not_overlapped (the agent's), for
    whole-box idle and, per GPU lane, for partial idle: the study, 2026-10-20, design/IDEA.md, "job host phase" and "Partial
    idle also carries causes"; definition agreed with the runner (runner/score/idle.py implements it on real runs).
    Per job j that reached the GPUs:
      a_j    when the job was first alive: its launch, when its launcher starts (idle.py: the launcher's start where the box
             recorded one, else the first rank's t_start).
      f_j    when its GPU work is under way on all its GPUs: the latest first-unit start across its ranks. A rank's isolated
             early unit before f_j stays GPU-busy and lies inside the start-up.
      S_j = [a_j, f_j), p_j = f_j - a_j. held_j = [f_j, its last unit's end). Idle in S_j counts from the first launch on.
      r_j    launchable from: the agent's start without `after` dependencies, else the latest exit among them; a dependency
             that never exited makes the job never launchable early.
    Whole-box idle:
      C_j    every other job's held span, and [a_k, last unit end of k) for every job k launched strictly before j (its exit
             tail excluded: j's GPU work would have filled it). An earlier job that never reached the GPUs covers [a_k, exit).
      A_j = min(hideable_startup(a_j, p_j, r_j, C_j), whole-box idle in S_j); the latest A_j seconds of S_j's whole-box idle
             are startup_not_overlapped, the rest job_host_phase.
    Partial idle, per GPU g of j:
      C_j^g  the same cover restricted to jobs on GPU g, plus whole-box idle inside C_j: instants when no GPU is busy because
             of another job's start-up or gap, so that j's start-up there would have cost GPU g nothing.
      A_j^g = min(hideable_startup(a_j, p_j, r_j, C_j^g), GPU g's idle in S_j). The latest A_j^g seconds of GPU g's idle in
             S_j are the agent's on that lane; the whole-box instants among them are left to the whole-box rule, and the
             partial ones are startup_not_overlapped in GPU-seconds. Partial idle on GPU g while a job on g is alive is
             otherwise job_host_phase.
    Per job: startup_s, whole-box idle in S_j and A_j (wall seconds), its partial agent share (GPU-seconds), and
    lane_hideable_s = the sum of A_j^g over its GPUs (lane-seconds, whole-box instants included). Where start-ups share idle,
    each job carries its own share and the cause totals count every instant once, so there is no per-job job_host_phase.
    A simulated job paces all its ranks alike, so f_j is its first busy instant; a run dict may carry "rank_first_busy" (one
    first-unit start per rank) to give ranks that differ, as build/test_startup_split.py does. Returns {job_id: {...}}."""
    allruns = list(sim.runs.values())
    runs = [r for r in allruns if r["busy"]]
    allbusy = union([list(iv) for r in allruns for iv in r["busy"]])
    t0 = min((r["hello"] for r in allruns), default=0.0)   # before it, whole-box idle is before_first_launch
    t_end = max([r["bye"] for r in allruns] + [e for _, e in allbusy] + [t0])
    whole = complement(allbusy, t0, t_end)
    by_id = {r["job_id"]: r for r in allruns}
    a_of = {r["job_id"]: min(r["launch"], r["hello"]) for r in allruns}
    end_of = {r["job_id"]: (max(e for _, e in r["busy"]) if r["busy"] else r["bye"]) for r in allruns}
    f_of = {r["job_id"]: max(r.get("rank_first_busy") or [min(s for s, _ in r["busy"])]) for r in runs}
    held = {r["job_id"]: [f_of[r["job_id"]], max(e for _, e in r["busy"])] for r in runs}
    lane_busy = {}
    for r in allruns:
        for g in r["gpus"]:
            lane_busy.setdefault(g, []).extend([list(iv) for iv in r["busy"]])
    lane_busy = {g: union(l) for g, l in lane_busy.items()}
    out = {}
    for r in runs:
        j = r["job_id"]
        a, f = a_of[j], f_of[j]
        p = max(0.0, f - a)
        deps = sim.jobs[r["i"]].get("after") or []
        ready = None if any(d not in by_id or by_id[d]["killed"] for d in deps) else max([0.0] + [by_id[d]["bye"] for d in deps])
        others = [q for q in allruns if q is not r]
        def cover(qs):
            return union([list(held[q["job_id"]]) for q in qs if q["job_id"] in held] +
                         [[a_of[q["job_id"]], end_of[q["job_id"]]] for q in qs if a_of[q["job_id"]] < a - EPS])
        C = cover(others)
        lo = max(a, t0)
        idle = intersect(whole, [[lo, f]]) if f > lo + EPS else []
        idle_s = total(idle)
        A = min(idle_s, hideable_startup(a, p, ready, C))
        env_whole = intersect(whole, C)
        lane_parts, lane_s, part_gs = {}, 0.0, 0.0
        for g in r["gpus"]:
            Cg = union(cover([q for q in others if g in q["gpus"]]) + env_whole)
            idle_g = complement(lane_busy.get(g, []), lo, f) if f > lo + EPS else []
            Ag = min(total(idle_g), hideable_startup(a, p, ready, Cg))
            Pg = minus(latest(idle_g, Ag), whole)
            lane_parts[g] = Pg
            lane_s += Ag
            part_gs += total(Pg)
        out[j] = {"startup_s": p, "idle_s": idle_s, "agent_s": A, "env_s": max(0.0, idle_s - A), "lane_s": lane_s,
                  "partial_agent_gs": part_gs, "gpus": len(r["gpus"]), "parts": latest(idle, A), "lane_parts": lane_parts,
                  "t_ready": ready}
    return out


def partial_attribution(sim, split, w1, agent_end):
    """Causes of partial idle, per GPU lane, in wall GPU-seconds: the study, 2026-10-20, design/IDEA.md, "Partial idle also
    carries causes"; agreed with the runner. A lane-instant in [0, w1) is partial idle when its GPU is idle and another
    GPU is busy, so the causes sum to the partial idle reported beside them. First match wins:
      startup_not_overlapped  inside the agent part of a start-up on that lane (startup_split's lane_parts)
      job_host_phase          a job using that GPU is alive: its start-up, an in-job gap or its exit tail
      no_work_left            no task job is both unlaunched and launchable (every `after` dependency exited), so nothing could
                              have run there
      after_agent_ended       after the agent's session ended
      model_turn, wait_or_poll, non_gpu_tool, unattributed   as for whole-box idle
    route_down and after_job_failure are not simulated here, so they never appear."""
    runs = list(sim.runs.values())
    by_id = {r["job_id"]: r for r in runs}
    ag = sim.agent
    anybusy = intersect(union([list(iv) for r in runs for iv in r["busy"]]), [[0.0, w1]])
    events = {r["launch"] for r in runs} | {r["bye"] for r in runs}

    def work_left(m):
        for i, job in enumerate(sim.jobs):
            r = sim.runs.get(i)
            if r is not None and r["launch"] <= m:
                continue
            if all(d in by_id and not by_id[d]["killed"] and by_id[d]["bye"] <= m for d in job.get("after") or []):
                return True
        return False

    cause = {}
    for g in range(sim.box):
        lane = union([list(iv) for r in runs if g in r["gpus"] for iv in r["busy"]])
        idle = intersect(complement(lane, 0.0, w1), anybusy)
        if not idle:
            continue
        agent = union([list(iv) for v in split.values() for iv in v["lane_parts"].get(g, [])])
        alive = [(min(r["launch"], r["hello"]), r["bye"]) for r in runs if g in r["gpus"]]
        cuts_all = events | {agent_end} | {x for iv in alive for x in iv} | {x for iv in agent for x in iv} | \
                   {x for s, e in ag.turns for x in (s, e)} | {x for s, e, _ in ag.tools for x in (s, e)}
        for s, e in idle:
            cuts = sorted({s, e} | {c for c in cuts_all if s < c < e})
            for x, y in zip(cuts, cuts[1:]):
                m = (x + y) / 2
                if any(a <= m < b for a, b in agent):
                    k = "startup_not_overlapped"
                elif any(a <= m < b for a, b in alive):
                    k = "job_host_phase"
                elif not work_left(m):
                    k = "no_work_left"
                elif m > agent_end:
                    k = "after_agent_ended"
                elif any(ts <= m < te for ts, te in ag.turns):
                    k = "model_turn"
                else:
                    k = next((kd for ts, te, kd in ag.tools if ts <= m < te and kd == "wait_or_poll"), None) or \
                        next((kd for ts, te, kd in ag.tools if ts <= m < te), None) or "unattributed"
                cause[k] = cause.get(k, 0.0) + (y - x)
    return cause


def score(sim, agent_end, agent_killed):
    task, a, box = sim.task, sim.a, sim.box
    sp, runs = a.speed, sorted(sim.runs.values(), key=lambda r: r["i"])
    lanes = {g: union([iv for r in runs if g in r["gpus"] for iv in r["busy"]]) for g in range(box)}
    busy_all = [iv for l in lanes.values() for iv in l]
    t0 = min(r["hello"] for r in runs)
    last_busy = max((e for _, e in busy_all), default=t0)
    byes = [r["bye"] for r in runs if not r["killed"]]
    t_last = max(byes + [last_busy])
    clip = lambda lo, hi: sum(max(0.0, min(e, hi) - max(s, lo)) for s, e in busy_all)
    busy, mk = clip(t0, t_last), t_last - t0
    lb = task["stats"]["lower_bound_makespan_s"]
    sr = {"gpus": box, "speed": sp, "makespan_wall_s": mk, "makespan_recorded_s": mk * sp, "busy_gpu_wall_s": busy,
          "utilisation": busy / (box * mk) if mk > 0 else None,
          "makespan_from_agent_start_wall_s": t_last, "makespan_from_agent_start_recorded_s": t_last * sp,
          "before_first_launch_wall_s": t0, "utilisation_from_agent_start": busy / (box * t_last) if t_last > 0 else None,
          "lower_bound_s": lb, "gap_to_lower_bound_s": mk * sp - lb, "makespan_over_lower_bound": mk * sp / lb,
          "gap_to_lower_bound_from_agent_start_s": t_last * sp - lb, "makespan_from_agent_start_over_lower_bound": t_last * sp / lb,
          "unfinished_jobs": [r["job_id"] for r in runs if r["killed"]]}
    done = [r for r in runs if not r["killed"]]
    remaining = len(sim.jobs) - len(done)
    w1, how = sim.D, "the task's horizon_s" if "horizon_s" in task else "the task's deadline_s (earlier design)"
    if remaining == 0 and max(r["bye"] for r in done) < w1:
        w1, how = max(r["bye"] for r in done), how + "; cut at the last job's completion, since no work remained"
    if last_busy > w1 + EPS:
        w1, how = last_busy, how + "; extended to the last busy instant"
    pts = sorted([(max(s, 0.0), 1) for s, e in busy_all if min(e, w1) > max(s, 0.0)] +
                 [(min(e, w1), -1) for s, e in busy_all if min(e, w1) > max(s, 0.0)])
    whole, partial, n, x0 = [], 0.0, 0, 0.0
    for x, d in pts + [(w1, 0)]:
        if x > x0 + EPS:
            if n == 0:
                whole.append((x0, x))
            elif n < box:
                partial += (box - n) * (x - x0)
        n += d
        x0 = max(x0, x)
    ag = sim.agent
    # a job is alive from its launch: its launcher (accelerate launch, torchrun) runs from then, before the ranks say hello,
    # and the runner's idle.py counts it alive from the launcher's start (launcher_intervals)
    alive = [(min(r["launch"], r["hello"]), r["bye"]) for r in runs]
    split = startup_split(sim)
    agent_parts = [iv for v in split.values() for iv in v["parts"]]
    cuts_all = {t0, agent_end} | {x for iv in alive for x in iv} | {x for iv in agent_parts for x in iv} | \
               {x for s, e in ag.turns for x in (s, e)} | \
               {x for s, e, _ in ag.tools for x in (s, e)}
    cause = {}
    for s, e in whole:
        cuts = sorted({s, e} | {c for c in cuts_all if s < c < e})
        for x, y in zip(cuts, cuts[1:]):
            m = (x + y) / 2
            if m < t0:
                k = "before_first_launch"
            elif any(h <= m < b for h, b in alive):
                k = "startup_not_overlapped" if any(s <= m < e for s, e in agent_parts) else "job_host_phase"
            elif m > agent_end:
                k = "after_agent_ended"
            elif any(ts <= m < te for ts, te in ag.turns):
                k = "model_turn"
            else:
                k = next((kd for ts, te, kd in ag.tools if ts <= m < te and kd == "wait_or_poll"), None) or \
                    next((kd for ts, te, kd in ag.tools if ts <= m < te), None) or "unattributed"
            cause[k] = cause.get(k, 0.0) + (y - x) * sp
    ws = sum(e - s for s, e in whole)
    pcause = partial_attribution(sim, split, w1, agent_end)
    idle = {"window": {"from": "agent start", "to": how, "wall_s": w1, "recorded_s": w1 * sp, "speed": sp, "gpus": box},
            "whole_box_idle_share_of_gpu_seconds": ws / w1 if w1 > 0 else None,
            "whole_box_idle_share_of_wall": ws / w1 if w1 > 0 else None,
            "partial_idle_share_of_gpu_seconds": partial / (box * w1) if w1 > 0 else None,
            "time_to_first_launch_recorded_s": t0 * sp,
            "idle_after_agent_stopped_recorded_s": cause.get("after_agent_ended", 0.0),
            "whole_box_idle_by_cause_recorded_s": {k: v for k, v in sorted(cause.items(), key=lambda kv: -kv[1])},
            "partial_idle_recorded_gpu_s": partial * sp,
            "partial_idle_by_cause_recorded_gpu_s": {k: v * sp for k, v in sorted(pcause.items(), key=lambda kv: -kv[1])},
            "startup_split_by_job_recorded_s": {j: {"startup_s": v["startup_s"] * sp,
                                                    "launchable_from_s": None if v["t_ready"] is None else v["t_ready"] * sp,
                                                    "whole_box_idle_in_startup_s": v["idle_s"] * sp,
                                                    "startup_not_overlapped_s": v["agent_s"] * sp,
                                                    "partial_startup_not_overlapped_gpu_s": v["partial_agent_gs"] * sp,
                                                    "lane_hideable_s": v["lane_s"] * sp, "gpus": v["gpus"]}
                                                for j, v in sorted(split.items())},
            "startup_split_how": "build/expected_scores.py startup_split (design/IDEA.md, job host phase; agreed with "
                                 "the runner), whole-box and per GPU lane. Per job: startup_not_overlapped_s, its whole-box share "
                                 "in wall seconds; partial_startup_not_overlapped_gpu_s, its partial share in GPU-seconds, the "
                                 "per-job breakdown of the partial cause; lane_hideable_s, the per-lane counterfactual summed "
                                 "over its GPUs before whole-box instants go to the whole-box rule. Where start-ups share idle "
                                 "the cause totals count each instant once.",
            "work_remaining_at_window_end": {"remaining": remaining, "task_jobs": len(sim.jobs), "done": len(done)},
            "model_turns": len(ag.turns), "tool_calls": len(ag.tools),
            "agent_end_recorded_s": agent_end * sp, "agent_killed_at_horizon": agent_killed}
    per_gpu = {g: {"busy_recorded_s": sum(e - s for s, e in l) * sp, "lanes_wall": [[s, e] for s, e in l]} for g, l in lanes.items()}
    return sr, idle, per_gpu


def emit(path, sim, agent_end, killed=False):
    os.makedirs(os.path.join(path, "transcript", "session", "scripted"), exist_ok=True)
    sp = sim.a.speed
    ev = [{"t_wall": T0 - 1.0, "t_mono": M0 - 1.0, "speed": sp, "kind": "daemon_start", "gpu": -1, "gpus": sim.box}]
    at = lambda t, **f: ev.append({"t_wall": T0 + t, "t_mono": M0 + t, "speed": sp, **f})
    jid = 0
    for r in sorted(sim.runs.values(), key=lambda r: (r["hello"], r["i"])):
        for rank, g in enumerate(r["gpus"]):
            jid += 1
            at(r["hello"], kind="job_hello", gpu=-1, job=jid, pid=20000 + jid, argv=r["argv"], t_start=T0 + r["hello"])
            at(r["hello"], kind="lookup", gpu=-1, job=jid, fingerprint=None, mode="replay", record_id=r["record_id"], match="exact",
               draw=r["draw"], group=sim.jobs[r["i"]]["replicate_group"], stage="exact", rank=rank, job_key=r["job_id"])
            for k, u in enumerate(r["units"], 1):
                at(u["enq"], kind="enqueue", gpu=g, job=jid, op="replay", seq=k, modelled_s=u["recorded_s"], wall_cost=u["recorded_s"] / sp,
                   **({"unit": "example", "host_gap_s": u["host_gap_s"]} if "host_gap_s" in u else {}))
            if not r["killed"]:
                at(r["bye"], kind="job_bye", gpu=-1, job=jid, pid=20000 + jid)
    ev.sort(key=lambda e: (e["t_mono"], e["kind"] != "daemon_start"))
    with open(os.path.join(path, "occupancy.jsonl"), "w") as fh:
        fh.writelines(json.dumps(e) + "\n" for e in ev)
    # one launcher process per launch, as the box's procs records show it (isolation's sampler): the shell the agent's Bash
    # call runs, from the launch to the job's exit, so a scorer reads a job's start from it as on a real run
    procs = []
    for n, r in enumerate(sorted(sim.runs.values(), key=lambda r: (r["launch"], r["i"])), 1):
        pid = 30000 + n
        procs.append((r["launch"], 0, {"kind": "procs", "t_utc": iso(r["launch"]), "full": False, "procs": [], "exited": [],
                                       "started": [{"pid": pid, "ppid": 29999, "comm": "sh", "cmdline": "/bin/sh -c " + r["command"],
                                                    "rss_kb": 0, "oom_score_adj": 0}]}))
        procs.append((r["bye"], 1, {"kind": "procs", "t_utc": iso(r["bye"]), "full": False, "procs": [], "started": [],
                                    "exited": [pid]}))
    with open(os.path.join(path, "resources.jsonl"), "w") as fh:
        fh.writelines(json.dumps(x[2]) + "\n" for x in sorted(procs, key=lambda x: (x[0], x[1])))
    # status.json in the shape isolation/run_one.py writes (monitor() and final_status(), read 2026-10-20): a harness that
    # exited is "completed" with harness_end_wall; one still running at the horizon is "killed-on-time" with teardown
    # "deadline" and no harness_end_wall, and its open tool call stays open in the transcript (Agent.end), as a killed
    # process writes no tool_result. No real run has yet ended that way, so that half is inferred from run_one.py's code.
    # run_one.py 4fd40567 still calls the hard stop "deadline" in these strings; they follow it, not the task field.
    last = max([agent_end] + [r["bye"] for r in sim.runs.values()])
    over = killed or any(r["killed"] for r in sim.runs.values()) or last >= sim.D - EPS
    down = sim.D if over else last
    st = {"status": "killed-on-time" if killed else "completed",
          "reason": "the task's deadline passed with the harness still running" if killed else "harness exited 0",
          "teardown_reason": "deadline" if over else "jobs_ended",
          "agent_start_wall": T0, "harness_end_wall": None if killed else T0 + agent_end,
          "deadline_wall": T0 + sim.D, "teardown_wall": T0 + down, "end_utc": iso(down),
          "synthetic": "gpu-tasks build/expected_scores.py: the expected timeline of a scripted agent; no model, no box"}
    if not killed:
        st["harness_exit_code"] = 0
    json.dump(st, open(os.path.join(path, "status.json"), "w"), indent=1)
    json.dump(sim.task, open(os.path.join(path, "task.json"), "w"), indent=1)
    with open(os.path.join(path, "transcript", "session", "scripted", f"{sim.policy}.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in sim.agent.records)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("registry")
    ap.add_argument("task_id")
    ap.add_argument("policy", choices=POLICIES + ("all",))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--host")
    ap.add_argument("--emit")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    a.host_terms = json.load(open(a.host)) if a.host else {}
    if "terms" in a.host_terms:
        a.host_terms = a.host_terms["terms"]
    task = json.load(open(os.path.join(a.registry, "tasks", a.task_id + ".json")))
    if task.get("decisions"):
        task = bind_recorded_winners(task, a.registry)
    if task.get("reference_plan"):
        task = C.expand_reference_plan(task)     # a goal-level task, as its scripted reference runs it
    gm_sha = hashlib.sha256(open(DAEMON, "rb").read()).hexdigest()
    out = {"schema": "p006-expected-scores/0.1", "task_id": a.task_id, "speed": a.speed,
           "lower_bound_makespan_s": task["stats"]["lower_bound_makespan_s"],
           "horizon_s": task.get("horizon_s") or task["deadline_s"],
           "horizon_how": "the task's horizon_s" if "horizon_s" in task else "the task's deadline_s (earlier design)",
           "host_terms": a.host_terms or "none: the ideal case", "think_s": THINK_S, "delay_s": DELAY_S,
           "gpu_mock_daemon_sha256": gm_sha, "policies": {}, "known_divergences": []}
    for pol in (POLICIES if a.policy == "all" else (a.policy,)):
        sim = Sim(task, a.registry, a, pol)
        agent_end, killed = sim.run()
        sr, idle, per_gpu = score(sim, agent_end, killed)
        jobs = []
        for r in sorted(sim.runs.values(), key=lambda r: r["launch"]):
            today = r["gpu_mock_today"]
            j = {k: r[k] for k in ("job_id", "gpus", "launch", "hello", "bye", "killed", "record_id", "draw", "seed_wanted",
                                   "paced_recorded_s", "command")}
            if a.host_terms:
                j["startup"] = r["startup"]
            # A divergence is today's lookup serving a different recorded run (common.served_unit), not merely another record
            # id: a prepared per-seed record and its per-setup twin share one exact key and one run.
            byid = sim.gm.intended._by_id
            same_run = (today["record_id"] == r["record_id"] or (today["record_id"] in byid and r["record_id"] in byid and
                        C.served_unit(byid[today["record_id"]])[:2] == C.served_unit(byid[r["record_id"]])[:2]))
            if not same_run or today["mode"] != "replay" or today["stage"] != "exact":
                j["gpu_mock_today"] = today
                out["known_divergences"].append({"policy": pol, "job_id": r["job_id"], "intended": r["record_id"], "today": today})
            jobs.append(j)
        out["policies"][pol] = {"score_run": sr, "idle": idle, "per_gpu": per_gpu, "jobs": jobs,
                                "never_launched": [j["job_id"] for k, j in enumerate(sim.jobs) if k not in sim.runs]}
        if pol == "packed_overlap" and any(sim.prep_s(k) > 0 for k in range(len(sim.jobs))):
            # run_overlap launches every job at its lane's time and never prepares one (the analysis, 2026-10-24
            # 05:39 MDT: say so in the expected file)
            out["policies"][pol]["prep_ignored"] = (
                "packed_overlap launches every job without its preparation (prep.scripted_s is not paid), so for a task "
                "with preparation it is a lower bound on a schedule, not a schedule; bar 2 compares the brief-only policy "
                "with the best feasible preparation order instead")
        if a.emit:
            emit(os.path.join(a.emit, pol), sim, agent_end, killed)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    print(f"{a.task_id}: speed {a.speed}, host terms {'from ' + a.host if a.host else 'zero (ideal)'}; lower bound "
          f"{out['lower_bound_makespan_s']} s, horizon {out['horizon_s']} s; recorded seconds")
    hdr = ("makespan", "from start", "util", "util start", "window", "whole idle", "partial", "after stop", "remaining")
    print(f"{'policy':26s}" + "".join(f"{h:>12s}" for h in hdr))
    for pol, s in out["policies"].items():
        r, i = s["score_run"], s["idle"]
        row = (r["makespan_recorded_s"], r["makespan_from_agent_start_recorded_s"], r["utilisation"], r["utilisation_from_agent_start"],
               i["window"]["recorded_s"], i["whole_box_idle_share_of_wall"], i["partial_idle_share_of_gpu_seconds"],
               i["idle_after_agent_stopped_recorded_s"], i["work_remaining_at_window_end"]["remaining"])
        print(f"{pol:26s}" + "".join(f"{x:12.3f}" if isinstance(x, float) else f"{x!s:>12s}" for x in row))
    for pol, s in out["policies"].items():
        print(f"-- {pol}: window to {s['idle']['window']['to']}; causes {json.dumps({k: round(v, 3) for k, v in s['idle']['whole_box_idle_by_cause_recorded_s'].items()})}; never launched {s['never_launched'] or 'none'}")
        for j in s["jobs"]:
            print(f"   {j['job_id']:24s} gpus {j['gpus']} launch {j['launch']:10.3f} bye {j['bye']:10.3f} {'KILLED ' if j['killed'] else ''}"
                  f"{j['record_id']} ({j['draw']})" + (f"  TODAY: {j['gpu_mock_today']['mode']} {j['gpu_mock_today']['record_id']}" if 'gpu_mock_today' in j else ""))
    if out["known_divergences"]:
        print(f"known divergences from gpu-mock as it is today: {len(out['known_divergences'])} job launches (see --json)")


if __name__ == "__main__":
    main()
