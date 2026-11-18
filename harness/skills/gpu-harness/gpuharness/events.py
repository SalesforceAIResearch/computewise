"""Events: what happened, who it is for, and whether it should wake them.

An agent waits for its events with `gpuh wait --for <agent>`, run as a background task. The wait returns as soon as an
event that needs the agent's judgement arrives, and prints every event addressed to it since its last wait, one line
each. Events that need no judgement (a job started, a needed job finished while its siblings still run) are
delivered with the next waking one, so an agent is woken only to decide something.

Types, and who they wake:

    STARTED      a job is running on a GPU (or the CPU, for a CPU-only job), with its pid  owner, no wake
    PROGRESS     a check's measured pace and projected length, with the reference's       owner, wakes
                 length on the same kind of work for comparison
    MILESTONE    a value the plan names appeared in a running job's log (e.g. the loss at  owner, wakes for an
                 step 1000), with the reference's value for the same milestone            open candidate
    DONE         a job exited 0, with its result line and runtime                         owner, wakes for checks
                                                                                         and a candidate's last job
    FAILED       a job exited non-zero, with its last error line                          owner, wakes
    DIED         a job vanished without an exit code (killed from outside)                owner, wakes
    CANCELLED    a job was stopped or dropped                                             owner, wakes if not asked
    BLOCKED      a job can never run (a dependency failed)                                owner, wakes
    JUDGE        a check's result is in and gated work waits on your pass or reject       owner, wakes
    IDLE         every job of an open candidate has ended: decide it                      owner, wakes
    REJECTED     a branch rejected a candidate (its jobs are cancelled)                   main, wakes
    DECIDED      a branch decided a candidate                                             main, wakes
    ALL_DECIDED  every candidate is decided and every remaining job has ended             main, wakes
    RETIRE       every candidate a branch judges is decided: stop waiting                  owner, wakes
    NOWAITER     a branch has events waiting and no live `gpuh wait`                      main, wakes
    ERROR        the harness could not do something it was asked                          asker, wakes
    FINISH       the run is over; every wait returns                                      all, wakes
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

WAKES_ALWAYS = {"PROGRESS", "FAILED", "DIED", "BLOCKED", "JUDGE", "IDLE", "REJECTED", "DECIDED", "ALL_DECIDED",
                "NOWAITER", "ERROR", "FINISH", "RETIRE"}


@dataclass
class Event:
    type: str
    to: str
    text: str
    job: Optional[str] = None
    candidate: Optional[str] = None
    wake: bool = True
    seq: int = 0
    t: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Dict[str, Any]) -> "Event":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def line(self, t0: float = 0.0) -> str:
        stamp = _hms(self.t - t0) if t0 else time.strftime("%H:%M:%S", time.localtime(self.t))
        return f"[{stamp}] {self.type} {self.text}"


def _hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "?"
    seconds = int(round(seconds))
    if seconds < 120:
        return f"{seconds}s"
    if seconds < 7200:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
