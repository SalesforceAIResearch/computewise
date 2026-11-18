"""The plan's data types: candidates, jobs and their states.

A *candidate* is one option the decision is about (a method, an optimizer setup); one candidate may be the
*reference* that every comparison needs. A *job* is one run on one GPU. Each job has a *role*:

    check        a cheap observation that can settle its candidate early (a short whole workload, a first seed
                 whose pace is read minutes in). A reference job marked `check` is an enabler: the others are
                 judged against it, so it runs first.
    needed       work the answer needs once its gates are open (every job of a full table; the other seeds of a
                 candidate that passed its check).
    speculative  work worth doing only on a GPU that would otherwise sit idle.

Jobs wait for two kinds of predecessor. `after` lists jobs that must have finished successfully (data or a
checkpoint is needed). `needs_pass` lists checks whose *judgement* must be in: the branch that owns the candidate
has run `gpuh pass <check>`. The second is how the model's decision gates expensive work without the code ever
judging anything itself.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


class Role(str, enum.Enum):
    CHECK = "check"
    NEEDED = "needed"
    SPECULATIVE = "speculative"


class JobState(str, enum.Enum):
    WAITING = "waiting"      # predecessors or gates not yet met
    READY = "ready"          # runnable; in the queue
    RUNNING = "running"      # its process group is alive on its GPU
    STOPPING = "stopping"    # being cancelled; the GPU stays held until the whole group is gone
    DONE = "done"            # exited 0
    FAILED = "failed"        # exited non-zero
    DIED = "died"            # vanished with no exit code: killed from outside the harness
    CANCELLED = "cancelled"  # stopped or dropped by the harness on request


TERMINAL = frozenset({JobState.DONE, JobState.FAILED, JobState.DIED, JobState.CANCELLED})
ACTIVE = frozenset({JobState.RUNNING, JobState.STOPPING})


class CandState(str, enum.Enum):
    OPEN = "open"
    REJECTED = "rejected"    # ruled out; every job of it is cancelled at once
    DECIDED = "decided"      # a verdict is in; its remaining jobs run only if still requested


@dataclass
class Pace:
    """A running job's measured progress, read from its own log (see `progress`)."""

    done: float                          # steps or examples completed
    total: float                         # steps or examples in the whole job
    elapsed_s: float                     # wall seconds since the job started
    per_unit_s: Optional[float] = None   # seconds per step, from the log's own timer when it prints one
    projected_s: Optional[float] = None  # projected wall length of the whole job
    projected_timer_s: Optional[float] = None  # projected value of the log's own timer (train_time) at the end
    line: str = ""                       # the log line it was read from
    at: float = 0.0                      # harness clock when it was read


@dataclass
class Job:
    id: str
    command: str
    candidate: Optional[str] = None
    kind: Optional[str] = None           # workload label, for estimates shared across candidates
    group: Optional[str] = None          # the method or setup, for estimates; defaults to the candidate
    role: Role = Role.NEEDED
    after: List[str] = field(default_factory=list)
    needs_pass: Optional[List[str]] = None  # None: gated on the candidate's checks (needed jobs); [] for no gate
    estimate_s: Optional[float] = None   # the plan's prior length
    tail_s: float = 0.0                  # work that will be added after this job ends (a follow-up not yet planned)
    p_reject: Optional[float] = None     # for checks: the model's prior that it rules its candidate out
    screen: bool = False                 # a screen (a slice, one seed of three): never evidence for a rejection
    cwd: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    progress: Optional[str] = None       # regex with named groups done, total and optionally ms, s, timer
    result: Optional[str] = None         # regex picking the result line from the log
    gpus: int = 1                        # 0: a CPU-only job (data preparation, a CPU metric); it holds no GPU slot
    milestones: List[Dict[str, str]] = field(default_factory=list)  # [{name, pattern}]: a value read mid-run (MILESTONE)
    # runtime
    state: JobState = JobState.WAITING
    gpu: Optional[int] = None
    pid: Optional[int] = None
    log: Optional[str] = None
    rc_path: Optional[str] = None
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    rc: Optional[int] = None
    attempts: int = 0
    passed: bool = False
    pace: Optional[Pace] = None
    result_line: str = ""
    reason: str = ""                     # why it was cancelled, failed or blocked
    stop_deadline: Optional[float] = None
    killed_hard: bool = False
    added_at: float = 0.0
    pace_reported: Optional[float] = None  # the projection last sent in a PROGRESS event
    final_timer_s: Optional[float] = None  # the log's own timer at the end (train_time), for comparisons
    judged_at: Optional[float] = None      # when its branch passed it (checks)
    milestone_values: Dict[str, float] = field(default_factory=dict)  # name -> value, as read from the running log
    reminded: int = 0                      # judgement reminders sent

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL

    @property
    def cpu(self) -> bool:
        return self.gpus == 0

    @property
    def active(self) -> bool:
        return self.state in ACTIVE

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        for k, default in (("gpus", 1), ("milestones", []), ("milestone_values", {})):
            if d.get(k) == default:      # fields A4 added: absent when unused, so an A3 plan's board is byte-identical
                d.pop(k, None)
        d["role"] = self.role.value
        d["state"] = self.state.value
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Job":
        d = dict(d)
        pace = d.pop("pace", None)
        d["role"] = Role(d.get("role", "needed"))
        d["state"] = JobState(d.get("state", "waiting"))
        job = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        job.pace = Pace(**pace) if pace else None
        return job


@dataclass
class Candidate:
    id: str
    branch: str = "main"                 # the agent that owns its judgement
    reference: bool = False
    state: CandState = CandState.OPEN
    verdict: str = ""
    reason: str = ""
    evidence: List[str] = field(default_factory=list)
    decided_at: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Candidate":
        d = dict(d)
        d["state"] = CandState(d.get("state", "open"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Board:
    """Everything the scheduler and the engine reason about, in one picklable value."""

    rule: str = ""
    evidence: str = ""
    workdir: str = "."
    gpus: List[int] = field(default_factory=list)
    gpu_job: Dict[int, Optional[str]] = field(default_factory=dict)   # GPU -> job holding it
    cpu_slots: int = 4                   # CPU-only jobs (gpus 0) that may run at once, beside the GPU jobs
    jobs: Dict[str, Job] = field(default_factory=dict)
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    started_at: float = 0.0
    finished: bool = False
    all_decided_sent: bool = False

    def branches(self) -> List[str]:
        names = {c.branch for c in self.candidates.values()} | {"main"}
        return sorted(names)

    def owner(self, job: Job) -> str:
        cand = self.candidates.get(job.candidate or "")
        return cand.branch if cand else "main"

    def free_gpus(self) -> List[int]:
        return [g for g in self.gpus if self.gpu_job.get(g) is None]

    def free_cpu_slots(self) -> int:
        return max(0, self.cpu_slots - sum(1 for j in self.jobs.values() if j.cpu and j.active))

    def to_json(self) -> Dict[str, Any]:
        return {
            "rule": self.rule, "evidence": self.evidence, "workdir": self.workdir, "gpus": self.gpus,
            "gpu_job": {str(k): v for k, v in self.gpu_job.items()},
            "jobs": [j.to_json() for j in self.jobs.values()],
            "candidates": [c.to_json() for c in self.candidates.values()],
            "started_at": self.started_at, "finished": self.finished, "all_decided_sent": self.all_decided_sent,
            **({"cpu_slots": self.cpu_slots} if self.cpu_slots != 4 else {}),
        }

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Board":
        b = cls(rule=d.get("rule", ""), evidence=d.get("evidence", ""), workdir=d.get("workdir", "."),
                gpus=[int(g) for g in d.get("gpus", [])], started_at=d.get("started_at", 0.0),
                finished=d.get("finished", False), all_decided_sent=d.get("all_decided_sent", False),
                cpu_slots=int(d.get("cpu_slots", 4)))
        b.gpu_job = {int(k): v for k, v in (d.get("gpu_job") or {}).items()}
        for g in b.gpus:
            b.gpu_job.setdefault(g, None)
        b.jobs = {j["id"]: Job.from_json(j) for j in d.get("jobs", [])}
        b.candidates = {c["id"]: Candidate.from_json(c) for c in d.get("candidates", [])}
        return b
