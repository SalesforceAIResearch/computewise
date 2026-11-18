"""The engine and the loop that runs it.

The daemon is the single writer of a run. Every tick it:

  1. applies the commands agents sent through the CLI (pass, reject, decide, add, cancel, retry, estimate, finish);
  2. observes every job it holds a GPU for, from the machine: is its process group alive, what did it write to its
     log, what pace does that show, did it leave an exit code;
  3. turns what it saw into events for the agent that owns each job (DONE with the reference's result beside it,
     PROGRESS with the measured pace and the reference's length, JUDGE when a check can be judged, IDLE when a
     candidate has nothing left to run);
  4. asks the scheduler which jobs to start on the free GPUs and which speculative jobs to preempt, and does it;
  5. writes the board to disk.

Nothing in here judges a candidate. Every pass, reject and decision comes from an agent.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import launcher as L
from . import plan as P
from .estimate import Estimator
from .events import Event, WAKES_ALWAYS, duration
from .ledger import Store, pid_alive
from .model import Board, CandState, Job, JobState, Role, TERMINAL
from .progress import LogReader
from .scheduler import SPREAD_RUNS, Policy, decide, gates, readiness, remaining_s


def sample_sd(vals: List[float]) -> float:
    m = sum(vals) / len(vals)
    return (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5

TICK_S = 1.0
PROGRESS_STEP = 0.2         # re-report a check's pace when its projection moves by this fraction
NOWAITER_S = 90.0           # a branch with waking events and no live wait for this long is reported to main
NOWAITER_REPEAT_S = 240.0   # and again this often while it lasts
REMIND_S = 600.0            # an open judgement is re-announced after this long
MAX_REMINDERS = 3


def where(job: Job) -> str:
    """How an event names where a job ran: its GPU, or the CPU for a CPU-only job."""
    return "CPU" if job.cpu else f"GPU {job.gpu}"


class ProcessBackend:
    """Real processes, through `launcher`."""

    def start(self, job: Job, gpu: Optional[int], logdir: str) -> L.Started:
        return L.start(job.id, job.command, gpu, job.cwd or ".", logdir, job.env)

    def alive(self, pid: int) -> bool:
        return L.alive(pid)

    def exit_code(self, rc_path: str) -> Optional[int]:
        return L.exit_code(rc_path)

    def terminate(self, pid: int) -> None:
        L.terminate(pid)

    def kill(self, pid: int) -> None:
        L.kill(pid)

    def reap(self, pid: int) -> None:
        L.reap(pid)


@dataclass
class Engine:
    board: Board
    policy: Policy
    store: Store
    backend: Any = field(default_factory=ProcessBackend)
    kill_grace_s: float = L.KILL_GRACE_S

    def __post_init__(self) -> None:
        self.est = Estimator()
        self.est.load(self.board.jobs.values())
        self.readers: Dict[str, LogReader] = {}
        self.blocked_sent: set = set()
        self.idle_sent: Dict[str, float] = {}
        self.judge_sent: set = set()
        self.held: set = set()               # checks a GPU has been held for (logged once each)
        self.milestone_resent: set = set()   # (job, milestone) pairs re-sent once the reference's value came in
        self.nowaiter_sent: Dict[str, float] = {}
        self.ref_progress_seen: set = set()
        self.cand_ratio_sent: set = set()
        self.nowaiter_count: Dict[str, int] = {}

    # helpers --------------------------------------------------------------------------------------------------------
    def emit(self, type_: str, to: str, text: str, job: Optional[Job] = None, wake: Optional[bool] = None,
             **data: Any) -> None:
        if wake is None:
            wake = type_ in WAKES_ALWAYS
        self.store.emit(Event(type=type_, to=to, text=text, job=job.id if job else None,
                              candidate=job.candidate if job else data.pop("candidate", None), wake=wake, data=data))

    def owner(self, job: Job) -> str:
        return self.board.owner(job)

    def logdir(self) -> str:
        return self.store.path("logs")

    def set_state(self, job: Job, state: JobState, now: float, **extra: Any) -> None:
        job.state = state
        self.store.timeline(job.id, state.value, now, gpu=job.gpu, candidate=job.candidate, **extra)

    def reference_for(self, job: Job) -> Optional[Job]:
        """The reference candidate's job of the same kind, preferring one that has finished."""
        if not job.kind:
            return None
        refs = [j for j in self.board.jobs.values()
                if j.kind == job.kind and j.id != job.id and self.board.candidates.get(j.candidate or "")
                and self.board.candidates[j.candidate].reference]
        if not refs:
            return None
        done = [j for j in refs if j.state is JobState.DONE]
        if done:
            return done[0]
        running = [j for j in refs if j.pace and j.pace.projected_s]
        return running[0] if running else refs[0]

    def length_of(self, job: Job) -> Tuple[Optional[float], Optional[float], str]:
        """(wall seconds, the script's own timer seconds, 'measured' or 'projected') for the whole job."""
        if job.state is JobState.DONE and job.started_at and job.ended_at:
            return job.ended_at - job.started_at, job.final_timer_s, "measured"
        if job.pace and job.pace.projected_s:
            return job.pace.projected_s, job.pace.projected_timer_s, "projected"
        return None, None, ""

    def ratio_text(self, job: Job, ref: Job) -> str:
        """Two finished jobs' lengths, each clock against the same clock, named. G-k-haiku-A4-r1b: a branch divided
        the DONE line's lead (wall, 84m11s) by the reference's result-line timer (66m51s) and rejected the key's winner
        at x1.26; on one clock it was x1.15 (timer) and x1.17 (wall). Empty unless both lengths are measured."""
        mw, mt, _ = self.length_of(job)
        tw, tt, how = self.length_of(ref)
        if not (mw and tw) or how != "measured" or job.state is not JobState.DONE:
            return ""
        out = [f"wall {duration(mw)} vs {duration(tw)} -> x{mw / tw:.3f}"]
        if mt and tt:
            out.insert(0, f"timer {duration(mt)} vs {duration(tt)} -> x{mt / tt:.3f}")
        return (f"length against reference {ref.id}, on one clock each: " + "; ".join(out)
                + " (compare these, never a wall time with a timer)")

    def gated_on(self, check: Job) -> List[Job]:
        return [j for j in self.board.jobs.values()
                if check.id in gates(self.board, j, self.policy) and not j.terminal]

    # commands -------------------------------------------------------------------------------------------------------
    def apply(self, cmd: Dict[str, Any], now: float) -> Tuple[bool, str]:
        op = cmd.get("op")
        who = cmd.get("from") or "main"
        try:
            handler: Callable[[Dict[str, Any], str, float], Tuple[bool, str]] = getattr(self, f"cmd_{op}")
        except AttributeError:
            return False, f"unknown command {op!r}"
        return handler(cmd, who, now)

    def cmd_pass(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        out = []
        for jid in cmd.get("jobs") or []:
            job = self.board.jobs.get(jid)
            if job is None:
                return False, f"no job {jid}"
            cand = self.board.candidates.get(job.candidate or "")
            if cand is not None and cand.state is CandState.REJECTED:
                return False, f"{jid}: candidate {cand.id} is rejected"
            if job.state in (JobState.FAILED, JobState.DIED, JobState.CANCELLED):
                return False, f"{jid} {job.state.value}; nothing to pass (retry it, or reject the candidate)"
            job.passed, job.judged_at = True, now
            opened = [j.id for j in self.board.jobs.values() if jid in gates(self.board, j, self.policy)
                      and not j.terminal and all(self.board.jobs[g].passed for g in gates(self.board, j, self.policy)
                                                 if g in self.board.jobs)]
            self.store.timeline(jid, "passed", now, by=who)
            out.append(f"passed {jid}" + (f"; opens {', '.join(opened)}" if opened else ""))
        return True, "; ".join(out) or "nothing to pass"

    def cmd_reject(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        cid = cmd.get("candidate")
        cand = self.board.candidates.get(cid or "")
        if cand is None:
            return False, f"no candidate {cid}"
        if cand.reference:
            return False, f"{cid} is the reference; it can't be rejected"
        if cand.state is not CandState.OPEN:
            return False, f"{cid} is already {cand.state.value}"
        why = (cmd.get("why") or "").strip()
        if not why:
            return False, "say why: --why '<the observation, in the rule's terms>'"
        evidence = [e for e in (cmd.get("evidence") or []) if e]
        for e in evidence:
            if e not in self.board.jobs:
                return False, f"evidence {e}: no such job"
        if evidence and all(self.board.jobs[e].screen for e in evidence):
            return False, ("every job cited is a screen (a slice, one seed of several): the rule can't act on it. "
                           "Run the evidence the rule names, then reject on it")
        cand.state, cand.reason, cand.evidence, cand.decided_at = CandState.REJECTED, why, evidence, now
        stopped, dropped = self.cancel_jobs([j for j in self.board.jobs.values()
                                             if j.candidate == cid and not j.terminal], now, f"{cid} rejected", who)
        self.emit("REJECTED", "main", f"{cid} by {who}: {why}" + (f" [evidence {', '.join(evidence)}]" if evidence else "")
                  + f" | stopped {len(stopped)}, dropped {len(dropped)} jobs", candidate=cid)
        return True, f"rejected {cid}; stopping {len(stopped)} running, dropped {len(dropped)} queued"

    def cmd_decide(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        cid = cmd.get("candidate")
        cand = self.board.candidates.get(cid or "")
        if cand is None:
            return False, f"no candidate {cid}"
        if cand.state is not CandState.OPEN:
            return False, f"{cid} is already {cand.state.value}"
        verdict = (cmd.get("verdict") or "").strip()
        why = (cmd.get("why") or "").strip()
        if not verdict:
            return False, "give a verdict: --verdict pass|fail|<word>"
        evidence = [e for e in (cmd.get("evidence") or []) if e]
        cand.state, cand.verdict, cand.reason, cand.evidence, cand.decided_at = (
            CandState.DECIDED, verdict, why, evidence, now)
        queued = [j for j in self.board.jobs.values() if j.candidate == cid and not j.terminal and not j.active]
        running = [j for j in self.board.jobs.values() if j.candidate == cid and j.active]
        if cmd.get("stop"):
            self.cancel_jobs(running + queued, now, f"{cid} decided", who)
        else:
            self.cancel_jobs(queued, now, f"{cid} decided", who)
        self.emit("DECIDED", "main", f"{cid} {verdict} by {who}: {why}"
                  + (f" [evidence {', '.join(evidence)}]" if evidence else "")
                  + (f" | {len(running)} of its jobs still running" if running and not cmd.get("stop") else ""),
                  candidate=cid)
        return True, f"decided {cid} {verdict}" + (f"; dropped {len(queued)} queued" if queued else "")

    def cmd_cancel(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        jobs = []
        for jid in cmd.get("jobs") or []:
            job = self.board.jobs.get(jid)
            if job is None:
                return False, f"no job {jid}"
            if not job.terminal:
                jobs.append(job)
        stopped, dropped = self.cancel_jobs(jobs, now, cmd.get("why") or f"cancelled by {who}", who)
        return True, f"stopping {len(stopped)}, dropped {len(dropped)}"

    def cmd_retry(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        out = []
        for jid in cmd.get("jobs") or []:
            job = self.board.jobs.get(jid)
            if job is None:
                return False, f"no job {jid}"
            cand = self.board.candidates.get(job.candidate or "")
            if cand is not None and cand.state is CandState.REJECTED:
                return False, f"{jid}: candidate {cand.id} is rejected"
            if job.state not in (JobState.FAILED, JobState.DIED, JobState.CANCELLED):
                return False, f"{jid} is {job.state.value}; only failed, died or cancelled jobs are retried"
            if cmd.get("command"):
                job.command = cmd["command"]
            job.state, job.gpu, job.pid, job.rc, job.pace, job.reason = JobState.WAITING, None, None, None, None, ""
            job.started_at = job.ended_at = None
            job.pace_reported = None
            self.blocked_sent.discard(jid)
            self.store.timeline(jid, "retry", now, by=who)
            out.append(jid)
        return True, "requeued " + ", ".join(out)

    def cmd_add(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        try:
            new = P.merge(self.board, cmd.get("plan") or {}, now)
        except P.PlanError as e:
            return False, str(e)
        for jid in new:
            self.est.prior(self.board.jobs[jid], self.board.jobs[jid].estimate_s)
            self.store.timeline(jid, "added", now, by=who)
        return True, "added " + ", ".join(new)

    def cmd_estimate(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        job = self.board.jobs.get(cmd.get("job") or "")
        if job is None:
            return False, f"no job {cmd.get('job')}"
        job.estimate_s = float(cmd.get("seconds"))
        self.est.prior(job, job.estimate_s)
        return True, f"{job.id} now estimated at {duration(job.estimate_s)}"

    def cmd_finish(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        running = [j for j in self.board.jobs.values() if j.active]
        if running and not cmd.get("force"):
            keep = cmd.get("keep_running")
            if not keep:
                self.cancel_jobs([j for j in self.board.jobs.values() if not j.terminal], now, "run finished", who)
        else:
            self.cancel_jobs([j for j in self.board.jobs.values() if not j.terminal], now, "run finished", who)
        self.board.finished = True
        self.emit("FINISH", "all", f"the run is over ({who}): {cmd.get('why') or 'finished'}")
        return True, "finishing; every wait returns"

    def cmd_note(self, cmd: Dict[str, Any], who: str, now: float) -> Tuple[bool, str]:
        self.store.timeline("-", "note", now, by=who, text=cmd.get("text", ""))
        return True, "noted"

    # cancellation ---------------------------------------------------------------------------------------------------
    def cancel_jobs(self, jobs: List[Job], now: float, reason: str, who: str) -> Tuple[List[Job], List[Job]]:
        stopped, dropped = [], []
        for job in jobs:
            if job.state is JobState.RUNNING and job.pid:
                self.backend.terminate(job.pid)
                job.stop_deadline = now + self.kill_grace_s
                job.reason = reason
                self.set_state(job, JobState.STOPPING, now, why=reason)
                stopped.append(job)
            elif job.state is JobState.STOPPING:
                stopped.append(job)
            elif not job.terminal:
                job.reason = reason
                self.set_state(job, JobState.CANCELLED, now, why=reason)
                dropped.append(job)
        return stopped, dropped

    # observation ----------------------------------------------------------------------------------------------------
    def observe(self, now: float) -> None:
        for job in [j for j in self.board.jobs.values() if j.active]:
            reader = self.readers.get(job.id)
            if reader is None and job.log:
                reader = self.readers[job.id] = LogReader(job.log, job.progress, milestones=job.milestones)
            if reader is not None and job.started_at:
                pace = reader.poll(now, job.started_at)
                if pace is not None:
                    job.pace = pace
                    if pace.projected_s:
                        self.est.projected(job, pace.projected_s)
            alive = bool(job.pid) and self.backend.alive(job.pid)
            if job.state is JobState.STOPPING:
                if not alive:
                    self.release(job, now)
                    requeue = job.reason.startswith("preempted")
                    if requeue:
                        job.state, job.pid, job.started_at, job.pace = JobState.WAITING, None, None, None
                        self.store.timeline(job.id, "requeued", now, why=job.reason)
                    else:
                        self.set_state(job, JobState.CANCELLED, now, why=job.reason)
                        asked = job.reason.endswith("rejected") or job.reason.endswith("decided") or \
                            job.reason == "run finished"
                        self.emit("CANCELLED", self.owner(job),
                                  f"{job.id} stopped after {duration(job.ended_at - (job.started_at or now))}: {job.reason}"
                                  f"; {where(job)} freed with its whole process group gone",
                                  job, wake=not asked)
                elif job.stop_deadline and now >= job.stop_deadline and not job.killed_hard:
                    self.backend.kill(job.pid)
                    job.killed_hard = True
                    job.stop_deadline = now + self.kill_grace_s
                elif job.killed_hard and job.stop_deadline and now >= job.stop_deadline:
                    self.backend.kill(job.pid)
                    job.stop_deadline = now + self.kill_grace_s
                continue
            if alive:
                self.maybe_progress(job, now)
                self.maybe_milestones(job, reader, now)
                continue
            self.backend.reap(job.pid or 0)
            rc = self.backend.exit_code(job.rc_path) if job.rc_path else None
            if rc is None:
                # the shell may have just exited: give its rc file one more tick before calling it a death
                if not getattr(job, "_grace_tick", False):
                    job._grace_tick = True  # type: ignore[attr-defined]
                    continue
            job.rc = rc
            reader = self.readers.get(job.id)
            self.maybe_milestones(job, reader, now)
            if reader is not None and reader.final is not None and reader.final.timer is not None:
                job.final_timer_s = reader.final.timer
            self.release(job, now)
            if rc == 0:
                self.finished_ok(job, reader, now)
            elif rc is None:
                job.reason = "its process group vanished without an exit code (killed outside the harness)"
                self.set_state(job, JobState.DIED, now)
                self.emit("DIED", self.owner(job), f"{job.id} ({job.candidate or '-'}) {where(job)} after "
                          f"{duration(job.ended_at - job.started_at)}: {job.reason}. "
                          f"`gpuh retry {job.id}` to run it again", job)
            else:
                err = reader.last_error() if reader else ""
                job.reason = err
                self.set_state(job, JobState.FAILED, now, rc=rc)
                self.emit("FAILED", self.owner(job), f"{job.id} ({job.candidate or '-'}) {where(job)} exit {rc} after "
                          f"{duration(job.ended_at - job.started_at)}: {err}  (log: {job.log}; "
                          f"`gpuh retry {job.id} [--command ...]` after fixing it)", job)

    def release(self, job: Job, now: float) -> None:
        job.ended_at = now
        if job.gpu is not None and self.board.gpu_job.get(job.gpu) == job.id:
            self.board.gpu_job[job.gpu] = None

    def finished_ok(self, job: Job, reader: Optional[LogReader], now: float) -> None:
        job.result_line = reader.result_line(job.result) if reader else ""
        self.set_state(job, JobState.DONE, now, seconds=round(now - (job.started_at or now), 1))
        self.est.measured(job, now - (job.started_at or now))
        cand = self.board.candidates.get(job.candidate or "")
        # say what the plan says the job is, so a branch never mistakes a whole workload for a screen (simR, 10-28)
        what = " [screen]" if job.screen else " [the rule's evidence, per the plan]"
        parts = [f"{job.id} ({job.candidate or '-'}) {where(job)} in {duration(now - (job.started_at or now))}"
                 f"{what}: {job.result_line}"]
        ref = self.reference_for(job) if cand is not None and not cand.reference else None
        if ref is not None and ref.state is JobState.DONE:
            parts.append(f"reference {ref.id}: {ref.result_line}")
            ratio = self.ratio_text(job, ref)
            if ratio:
                parts.append(ratio)
        waiting = self.gated_on(job) if job.role is Role.CHECK and not job.passed else []
        if waiting and cand is not None and cand.state is CandState.OPEN:
            if ref is not None and ref.state is not JobState.DONE:
                parts.append(f"{len(waiting)} jobs wait for your judgement once reference {ref.id} is in")
            else:
                parts.append(f"{len(waiting)} jobs wait: `gpuh pass {job.id}` or "
                             f"`gpuh reject {job.candidate} --why '...' --evidence {job.id}"
                             + (f",{ref.id}" if ref else "") + "`")
                self.judge_sent.add(job.id)
        live = cand is not None and not cand.reference and cand.state is CandState.OPEN
        rest = [j for j in self.board.jobs.values() if live and j.candidate == cand.id and not j.terminal]
        if live and rest and not waiting:
            longest = max(rest, key=lambda j: remaining_s(j, self.est, now))
            parts.append(f"{len(rest)} more jobs of {cand.id} queued or running (longest {longest.id}, "
                         f"{duration(remaining_s(longest, self.est, now))} left): if this result fails the rule, "
                         f"`gpuh reject {cand.id} --why '...' --evidence {job.id}" + (f",{ref.id}" if ref else "")
                         + "` now stops them")
        # a branch sees every result of a live candidate: any one of them may settle it (analysis, 2026-10-28)
        wake = live or job.role is Role.CHECK or bool(waiting)
        self.emit("DONE", self.owner(job), " | ".join(parts), job, wake=wake,
                  seconds=round(now - (job.started_at or now), 1))
        # the reference finishing makes earlier checks of the same kind judgeable
        if cand is not None and cand.reference:
            for other in self.board.jobs.values():
                oc = self.board.candidates.get(other.candidate or "")
                if (other.kind == job.kind and other.state is JobState.DONE and oc is not None and not oc.reference
                        and oc.state is CandState.OPEN and other.id not in self.judge_sent):
                    n = len(self.gated_on(other)) if other.role is Role.CHECK and not other.passed else 0
                    self.judge_sent.add(other.id)
                    rest = [j for j in self.board.jobs.values() if j.candidate == oc.id and not j.terminal]
                    ratio = self.ratio_text(other, job)
                    self.emit("JUDGE", self.owner(other),
                              f"{oc.id}: {other.id}{' [screen]' if other.screen else ''} gave {other.result_line} | "
                              f"reference {job.id}: {job.result_line}" + (f" | {ratio}" if ratio else "")
                              + (f" | {n} jobs wait for `gpuh pass {other.id}`" if n else "")
                              + (f" | {len(rest)} of its jobs still queued or running" if rest else "")
                              + f" | if it fails the rule: `gpuh reject {oc.id} --why '...' --evidence "
                                f"{other.id},{job.id}`", other)

    def open_jobs(self, cid: str) -> List[Job]:
        return [j for j in self.board.jobs.values() if j.candidate == cid and not j.terminal
                and (j.active or readiness(self.board, j, self.policy)[0]
                     or not any(g in self.board.jobs and self.board.jobs[g].state is JobState.DONE
                                and not self.board.jobs[g].passed for g in gates(self.board, j, self.policy)))]

    def maybe_progress(self, job: Job, now: float) -> None:
        pace = job.pace
        if pace is None or not pace.projected_s:
            return
        cand = self.board.candidates.get(job.candidate or "")
        is_ref = cand is not None and cand.reference
        first = job.pace_reported is None
        moved = (not first) and abs(pace.projected_s - job.pace_reported) > PROGRESS_STEP * job.pace_reported
        ref = None if is_ref else self.reference_for(job)
        ref_len = self.length_of(ref) if ref else (None, None, "")
        ref_new = ref is not None and ref_len[0] and (job.id, ref.id) not in self.ref_progress_seen
        if not (first or moved or ref_new):
            return
        job.pace_reported = pace.projected_s
        text = (f"{job.id} ({job.candidate or '-'}) {int(pace.done)}/{int(pace.total)} after {duration(pace.elapsed_s)}: "
                f"projected {duration(pace.projected_s)} in all")
        if pace.projected_timer_s:
            text += f" (its own timer: {duration(pace.projected_timer_s)})"
        if ref is not None and ref_len[0]:
            self.ref_progress_seen.add((job.id, ref.id))
            mine = pace.projected_timer_s if (pace.projected_timer_s and ref_len[1]) else pace.projected_s
            theirs = ref_len[1] if (pace.projected_timer_s and ref_len[1]) else ref_len[0]
            text += f" | reference {ref.id} {ref_len[2]} {duration(theirs)} -> x{mine / theirs:.2f}"
        elif ref is not None:
            text += f" | reference {ref.id}: no pace yet"
        text += f" | log: {pace.line}"
        first_of_cand = cand is not None and not is_ref and ref is not None and ref_len[0] and \
            cand.id not in self.cand_ratio_sent
        if first_of_cand:
            self.cand_ratio_sent.add(cand.id)
        wake = (cand is not None and not is_ref and cand.state is CandState.OPEN and not job.passed
                and (job.role is Role.CHECK or first_of_cand))
        if wake:
            text += (f" | a measured pace from the real run: if it settles a time or speed rule, reject now "
                     f"(`gpuh reject {job.candidate} --why '...' --evidence {job.id}`); "
                     f"if it passes the check, `gpuh pass {job.id}` opens its gated runs")
        self.emit("PROGRESS", self.owner(job), text, job, wake=wake)

    # follow-ups -----------------------------------------------------------------------------------------------------
    def maybe_milestones(self, job: Job, reader: Optional[LogReader], now: float) -> None:
        """A value the plan names (a milestone) has appeared in a job's log: send it to the job's owner, with the
        reference's value for the same milestone beside it. The code reads the numbers; it never compares them with
        the rule."""
        if reader is None or not job.milestones:
            return
        for m in job.milestones:
            name = m["name"]
            if name in job.milestone_values or name not in reader.milestones:
                continue
            value, line = reader.milestones[name]
            job.milestone_values[name] = value
            self.store.timeline(job.id, "milestone", now, name=name, value=value)
            self.send_milestone(job, name, value, line, now)

    def milestone_refs(self, job: Job, name: str) -> List[Tuple[str, float]]:
        """The reference's values for this milestone, from its jobs of the same kind (every one read so far)."""
        out = []
        for r in self.board.jobs.values():
            c = self.board.candidates.get(r.candidate or "")
            if c is not None and c.reference and r.kind == job.kind and r.id != job.id and name in r.milestone_values:
                out.append((r.id, r.milestone_values[name]))
        return sorted(out)

    def send_milestone(self, job: Job, name: str, value: float, line: str, now: float) -> None:
        cand = self.board.candidates.get(job.candidate or "")
        is_ref = cand is not None and cand.reference
        refs = [] if is_ref else self.milestone_refs(job, name)
        text = f"{job.id} ({job.candidate or '-'}) {name} = {value:g} after {duration(now - (job.started_at or now))}"
        if refs:
            vals = [v for _, v in refs]
            mean = sum(vals) / len(vals)
            text += (f" | reference {name}: " + ", ".join(f"{j} {v:g}" for j, v in refs)
                     + (f" (mean {mean:g}, n {len(vals)})" if len(vals) > 1 else ""))
        elif not is_ref:
            text += f" | reference {name}: not read yet"
        rest = [] if is_ref or cand is None else \
            [j for j in self.board.jobs.values() if j.candidate == cand.id and not j.terminal and j.id != job.id]
        if not is_ref and cand is not None and cand.state is CandState.OPEN:
            text += (f" | {job.id} is still running" + (f", and {len(rest)} more jobs of {cand.id} queued or running"
                                                         if rest else "")
                     + f": if this value fails the rule, `gpuh reject {cand.id} --why '...' --evidence {job.id}` "
                       f"stops them now")
        if refs and cand is not None and cand.state is CandState.OPEN:
            self.milestone_resent.add((job.id, name))          # this line already has the reference beside it
            arithmetic, has_spread = self.milestone_arithmetic(job, name, refs)
            text += arithmetic
            if has_spread:
                self.milestone_resent.add((cand.id, name, "sd"))   # and its spread: nothing left to re-send
        text += f" | log: {line}"
        wake = not is_ref and cand is not None and cand.state is CandState.OPEN
        self.emit("MILESTONE", self.owner(job), text, job, wake=wake, name=name, value=value)
        if is_ref:   # the reference's value makes earlier milestones of open candidates comparable: re-send them
            spread_now = len(self.milestone_refs(job, name)) + 1 == SPREAD_RUNS
            latest: Dict[str, Job] = {}
            for other in self.board.jobs.values():
                oc = self.board.candidates.get(other.candidate or "")
                if (oc is not None and not oc.reference and oc.state is CandState.OPEN and other.kind == job.kind
                        and name in other.milestone_values):
                    if (other.id, name) not in self.milestone_resent:
                        self.send_milestone(other, name, other.milestone_values[name],
                                            "(re-sent: the reference's value is in)", now)
                    elif spread_now and (oc.id, name, "sd") not in self.milestone_resent:
                        if oc.id not in latest or (other.started_at or 0) > (latest[oc.id].started_at or 0):
                            latest[oc.id] = other
            for cid, other in sorted(latest.items()):    # one line per candidate: it covers all its runs so far
                self.send_milestone(other, name, other.milestone_values[name],
                                    "(re-sent: the reference's run-to-run spread is in)", now)

    def milestone_arithmetic(self, job: Job, name: str, refs: List[Tuple[str, float]]) -> Tuple[str, bool]:
        """The arithmetic a mean rule needs at a milestone, computed here and never judged: the average the candidate's
        other runs of this kind would need for its mean to pass (to tie the reference's mean, if the plan gives no
        margin), how far that is from its runs so far, and that distance in run-to-run sds of one method repeated (the
        reference's runs, or the candidate's own), once SPREAD_RUNS of them are in. G-k-haiku-A4 (c3-19): with no
        spread on the line and one reference run in, a branch took the differences between candidates (about 0.02)
        for run-to-run spread, about 18 times the baseline's, and held two candidates whose first runs were 31 and 44
        baseline sds outside the rule."""
        cand = self.board.candidates[job.candidate or ""]
        runs = [j for j in self.board.jobs.values() if j.candidate == cand.id and j.kind == job.kind and not j.screen
                and any(m.get("name") == name for m in j.milestones)
                and j.state not in (JobState.FAILED, JobState.DIED, JobState.CANCELLED)]
        own = [j.milestone_values[name] for j in runs if name in j.milestone_values]
        rest = len(runs) - len(own)
        if not own or rest <= 0:
            return "", False
        spec = next((m for m in job.milestones if m.get("name") == name), {})
        better, margin = spec.get("better"), float(spec.get("margin") or 0.0)
        ref_vals = [v for _, v in refs]
        rmean, mine = sum(ref_vals) / len(ref_vals), sum(own) / len(own)
        if better in ("lower", "higher"):
            bar = rmean - margin if better == "lower" else rmean + margin
            goal = f"to pass ({'<=' if better == 'lower' else '>='} {bar:.6g}, reference mean {rmean:.6g} n {len(ref_vals)}"
            goal += f" {'-' if better == 'lower' else '+'} {margin:g})"
        else:
            bar, goal = rmean, f"to tie the reference mean {rmean:.6g} (n {len(ref_vals)})"
        need = (len(runs) * bar - sum(own)) / rest
        gap = need - mine
        here = "this run" if len(own) == 1 else f"its {len(own)} runs' mean"
        text = (f" | for the mean of {cand.id}'s {len(runs)} runs {goal}, its other {rest} would have to average"
                f" {need:.6g}: {abs(gap):.3g} {'below' if gap < 0 else 'above'} {here}")
        if better in ("lower", "higher") and (gap >= 0) == (better == "lower"):
            text += " (no better than it: this value alone does not fail the rule)"
        spreads = [(sample_sd(vals), who, len(vals)) for who, vals in
                   ((self.reference_id(), ref_vals), (cand.id, own)) if len(vals) >= SPREAD_RUNS]
        if spreads:
            sd, who, n = max(spreads)
            text += (f" | run-to-run sd {sd:.3g} ({who}, {n} runs): that gap is {abs(gap) / sd:.1f} sd" if sd > 0
                     else f" | run-to-run sd 0 ({who}, {n} runs)")
        else:
            text += (f" | run-to-run sd: not measured yet ({SPREAD_RUNS} runs of one method are needed; the reference"
                     f" has {len(ref_vals)}, {cand.id} {len(own)}; this line is sent again when it is); differences"
                     f" between candidates are not run-to-run spread")
        return text, bool(spreads)

    def reference_id(self) -> str:
        return next((c.id for c in self.board.candidates.values() if c.reference), "the reference")

    def awaits_reference(self, cand) -> bool:
        """True while a reference job of a kind this candidate ran has not finished: its comparison isn't possible yet."""
        kinds = {j.kind for j in self.board.jobs.values() if j.candidate == cand.id and j.kind}
        for j in self.board.jobs.values():
            c = self.board.candidates.get(j.candidate or "")
            if c is not None and c.reference and j.kind in kinds and not j.terminal:
                return True
        return False

    def follow_ups(self, now: float) -> None:
        for job in self.board.jobs.values():
            if job.terminal or job.active or job.id in self.blocked_sent:
                continue
            runnable, spec_ok, why = readiness(self.board, job, self.policy)
            if not runnable and ("dependency" in why or why.startswith("gate ")):
                self.blocked_sent.add(job.id)
                self.emit("BLOCKED", self.owner(job), f"{job.id} can't run: {why}. "
                          f"`gpuh retry <it>` or `gpuh cancel {job.id}`", job)
        for cand in self.board.candidates.values():
            if cand.state is not CandState.OPEN or cand.reference:
                continue
            jobs = [j for j in self.board.jobs.values() if j.candidate == cand.id]
            if not jobs or any(j.active for j in jobs):
                self.idle_sent.pop(cand.id, None)
                continue
            if any(readiness(self.board, j, self.policy)[0] for j in jobs if not j.terminal):
                self.idle_sent.pop(cand.id, None)
                continue
            awaiting = [j for j in jobs if j.role is Role.CHECK and j.state is JobState.DONE and not j.passed
                        and self.gated_on(j) and (self.reference_for(j) is None
                                                  or self.reference_for(j).state is JobState.DONE)]
            if not awaiting and self.awaits_reference(cand):
                continue        # the reference's result will wake the branch (JUDGE); nothing to decide before it
            last = self.idle_sent.get(cand.id)
            if last is not None and now - last < REMIND_S:
                continue
            if last is not None and getattr(self, "_reminders", {}).get(cand.id, 0) >= MAX_REMINDERS:
                continue
            self._reminders = getattr(self, "_reminders", {})
            self._reminders[cand.id] = self._reminders.get(cand.id, 0) + (1 if last is not None else 0)
            self.idle_sent[cand.id] = now
            done = [j.id for j in jobs if j.state is JobState.DONE]
            bad = [f"{j.id} {j.state.value}" for j in jobs if j.state in (JobState.FAILED, JobState.DIED)]
            if awaiting:
                text = (f"{cand.id}: nothing runs until you judge " + ", ".join(j.id for j in awaiting)
                        + ": `gpuh pass <check>` or `gpuh reject " + cand.id + " --why '...'`")
            else:
                waiting = [j for j in jobs if not j.terminal]
                if waiting:   # nothing can start, yet jobs wait: say on what (G-fyld-haiku-A4-r1b: "every job has
                    # ended (0 done)" while six preparations waited on their own checks)
                    why = [f"{j.id} ({readiness(self.board, j, self.policy)[2] or j.state.value})" for j in waiting[:4]]
                    text = (f"{cand.id}: nothing can start: {len(waiting)} jobs wait and none is runnable: "
                            + ", ".join(why) + (f", and {len(waiting) - 4} more" if len(waiting) > 4 else "")
                            + f" ({len(done)} done" + (", " + ", ".join(bad) if bad else "") + "). Fix what they "
                              f"wait on (`gpuh retry`, `gpuh cancel`, `gpuh add`), or decide it: `gpuh decide "
                              f"{cand.id} --verdict <pass|fail> --why '...' --evidence ...`")
                else:
                    text = (f"{cand.id}: every job has ended ({len(done)} done" + (", " + ", ".join(bad) if bad else "")
                            + f"). Decide it: `gpuh decide {cand.id} --verdict <pass|fail> --why '...' --evidence ...`, "
                              f"or add the jobs it still needs")
            self.emit("IDLE", cand.branch, text, candidate=cand.id)
        self.check_all_decided(now)
        self.retire_branches(now)
        self.check_waiters(now)

    def retire_branches(self, now: float) -> None:
        """A branch whose candidates are all decided and whose jobs have all ended is told to stop waiting."""
        if not hasattr(self, "_retired"):
            self._retired: set = set()
        for branch in self.board.branches():
            if branch == "main" or branch in self._retired:
                continue
            cands = [c for c in self.board.candidates.values() if c.branch == branch]
            if not cands or any(c.state is CandState.OPEN for c in cands):
                continue
            if any(not j.terminal for j in self.board.jobs.values() if j.candidate in {c.id for c in cands}):
                continue
            self._retired.add(branch)
            self.emit("RETIRE", branch, "every candidate you judge is decided and their jobs have ended: "
                                        "end your turn without starting another wait")

    def close_references(self, now: float) -> None:
        for cand in self.board.candidates.values():
            if not cand.reference or cand.state is not CandState.OPEN:
                continue
            jobs = [j for j in self.board.jobs.values() if j.candidate == cand.id]
            if jobs and all(j.terminal for j in jobs) and not any(
                    not j.terminal and readiness(self.board, j, self.policy)[0] for j in jobs):
                cand.state, cand.verdict, cand.decided_at = CandState.DECIDED, "reference", now
                cand.reason = f"{sum(1 for j in jobs if j.state is JobState.DONE)} of {len(jobs)} jobs done"

    def open_items(self) -> dict:
        """agent -> [(key, text)]: decisions only that agent can make, which nothing will happen without."""
        out: dict = {}
        for cand in self.board.candidates.values():
            if cand.state is not CandState.OPEN or cand.reference:
                continue
            jobs = [j for j in self.board.jobs.values() if j.candidate == cand.id]
            for j in jobs:
                if j.role is Role.CHECK and j.state is JobState.DONE and not j.passed and self.gated_on(j):
                    ref = self.reference_for(j)
                    if ref is None or ref.state is JobState.DONE:
                        out.setdefault(cand.branch, []).append(
                            (f"judge:{j.id}", f"{cand.id}: {j.id} gave {j.result_line} | "
                             + (f"reference {ref.id}: {ref.result_line} | " if ref else "")
                             + f"{len(self.gated_on(j))} jobs wait for `gpuh pass {j.id}` or "
                               f"`gpuh reject {cand.id} --why '...' --evidence {j.id}`"))
            if jobs and not any(j.active for j in jobs) and not any(
                    not j.terminal and readiness(self.board, j, self.policy)[0] for j in jobs) \
                    and not self.awaits_reference(cand):
                if not any(k.startswith("judge:") and txt.startswith(cand.id + ":") for k, txt in out.get(cand.branch, [])):
                    done = [j.id for j in jobs if j.state is JobState.DONE]
                    out.setdefault(cand.branch, []).append(
                        (f"decide:{cand.id}", f"{cand.id}: nothing is running or waiting ({len(done)} jobs done): "
                         f"`gpuh decide {cand.id} --verdict pass|fail --why '...' --evidence ...`"))
        if self.board.all_decided_sent and not self.board.finished:
            out.setdefault("main", []).append(("finish", "every candidate is decided: `gpuh results`, report, `gpuh finish`"))
        return out

    def write_open(self) -> None:
        items = self.open_items()
        for agent in set(self.board.branches()) | set(items):
            self.store.write_open(agent, items.get(agent, []))

    def check_all_decided(self, now: float) -> None:
        self.close_references(now)
        if self.board.all_decided_sent:
            return
        if any(j.active for j in self.board.jobs.values()):
            return
        open_ = [c for c in self.board.candidates.values() if c.state is CandState.OPEN]
        if open_:
            return
        runnable = [j for j in self.board.jobs.values() if not j.terminal and readiness(self.board, j, self.policy)[0]]
        if runnable:
            return
        self.board.all_decided_sent = True
        lines = []
        for c in self.board.candidates.values():
            if c.reference:
                continue
            lines.append(f"{c.id}={c.verdict or c.state.value}")
        done = sum(1 for j in self.board.jobs.values() if j.state is JobState.DONE)
        text = ("every candidate is decided and every job has ended: " + ", ".join(lines) if lines
                else "every job has ended") + f" ({done} jobs done). `gpuh results` prints them all; then report and " \
                                              "`gpuh finish`"
        self.emit("ALL_DECIDED", "main", text)

    def check_waiters(self, now: float) -> None:
        live = self.store.live_waiters()
        for agent in self.board.branches():
            if agent == "main" or agent in live:
                self.nowaiter_sent.pop(agent, None)
                self.nowaiter_count.pop(agent, None)
                continue
            pend = [e for e in self.store.pending(agent) if e.wake]
            if not pend or now - pend[0].t < NOWAITER_S:
                continue
            n = self.nowaiter_count.get(agent, 0)
            if agent in self.nowaiter_sent and now - self.nowaiter_sent[agent] < NOWAITER_REPEAT_S * (3 ** min(n, 3)):
                continue
            self.nowaiter_sent[agent] = now
            self.nowaiter_count[agent] = n + 1
            self.emit("NOWAITER", "main", f"{agent} has {len(pend)} unread events and no `gpuh wait` running; "
                                          f"oldest: {pend[0].type} {pend[0].text[:120]}. If teammate {agent} exists, "
                                          f"SendMessage it to run `gpuh --as {agent} wait --for {agent}` in the "
                                          f"background; if it was never started or has ended, start it (Agent tool, "
                                          f"name {agent}) as in the skill's section 2", candidate=None)

    # scheduling -----------------------------------------------------------------------------------------------------
    def schedule(self, now: float) -> None:
        if self.board.finished:
            return
        dec = decide(self.board, self.est, now, self.policy)
        for check, gpu, until in dec.hold:   # recorded once per check, so a run's idle cost can be read off the timeline
            if check not in self.held:
                self.held.add(check)
                self.store.timeline(check, "hold", now, gpu=gpu, until=round(until, 1),
                                    why="a GPU kept free for this check's judgement: it gates the longest chain")
        for jid in dec.preempt:
            job = self.board.jobs[jid]
            if job.pid:
                self.backend.terminate(job.pid)
            job.reason = "preempted: a needed job wants the GPU"
            job.stop_deadline = now + self.kill_grace_s
            self.set_state(job, JobState.STOPPING, now, why=job.reason)
        for jid, gpu in dec.starts:
            job = self.board.jobs[jid]
            if self.board.gpu_job.get(gpu) is not None:
                continue
            self.launch(job, gpu, now)
        for jid in dec.cpu_starts:
            self.launch(self.board.jobs[jid], None, now)

    def launch(self, job: Job, gpu: Optional[int], now: float) -> None:
        job.attempts += 1
        try:
            st = self.backend.start(job, gpu, self.logdir())
        except (OSError, ValueError) as e:
            job.reason = f"could not start: {e}"
            self.set_state(job, JobState.FAILED, now)
            self.emit("FAILED", self.owner(job), f"{job.id} could not start: {e}", job)
            return
        job.gpu, job.pid, job.log, job.rc_path, job.started_at = gpu, st.pid, st.log, st.rc_path, now
        job.ended_at, job.rc, job.pace, job.killed_hard, job.stop_deadline = None, None, None, False, None
        self.readers[job.id] = LogReader(st.log, job.progress, milestones=job.milestones)
        if gpu is not None:
            self.board.gpu_job[gpu] = job.id
        self.set_state(job, JobState.RUNNING, now, pid=st.pid)
        self.emit("STARTED", self.owner(job), f"{job.id} ({job.candidate or '-'}) on {where(job)}, pid {st.pid}; "
                  f"expected {self.est.explain(job)}", job, wake=False)

    # one tick -------------------------------------------------------------------------------------------------------
    def tick(self, now: float) -> None:
        for cid, cmd in self.store.take_commands():
            try:
                ok, text = self.apply(cmd, now)
            except Exception as e:  # a bad command must not stop the harness
                ok, text = False, f"{type(e).__name__}: {e}"
            self.store.reply(cid, ok, text)
            if not ok:
                self.emit("ERROR", cmd.get("from") or "main", f"{cmd.get('op')}: {text}", wake=False)
        self.observe(now)
        self.schedule(now)
        self.follow_ups(now)
        self.write_open()
        self.store.save_board(self.board)


def run(run_dir: str, tick_s: float = TICK_S) -> int:
    """The daemon's main loop. Resumes from board.json if one exists."""
    store = Store(run_dir)
    with open(store.path("plan.json")) as fh:
        plan_d = json.load(fh)
    policy = Policy.from_json(plan_d.get("policy"))
    try:
        board = store.load_board()
    except (OSError, ValueError):
        board, policy, _ = P.load(plan_d, gpus=plan_d.get("gpus"))
        board.workdir = plan_d.get("workdir") or board.workdir
        board.started_at = time.time()
    eng = Engine(board, policy, store)
    with open(store.path("daemon.pid"), "w") as fh:
        fh.write(f"{os.getpid()}\n")
    stop = {"now": False}

    def _term(*_: Any) -> None:
        stop["now"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    log = open(store.path("daemon.log"), "a", buffering=1)
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} daemon {os.getpid()} up, {len(board.jobs)} jobs, "
              f"GPUs {board.gpus}\n")
    idle_since: Optional[float] = None
    while not stop["now"]:
        now = time.time()
        try:
            eng.tick(now)
        except Exception:
            log.write(time.strftime("%Y-%m-%dT%H:%M:%S ") + traceback.format_exc())
        if board.finished and not any(j.active for j in board.jobs.values()):
            idle_since = idle_since or now
            if now - idle_since > 5:
                break
        time.sleep(tick_s)
    store.save_board(board)
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} daemon {os.getpid()} down\n")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1]))
