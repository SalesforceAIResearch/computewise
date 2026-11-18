"""The harness's state on disk: the board, the event log, cursors, waiters and the command spool.

One run lives in one directory (the *run dir*):

    board.json         the board, rewritten atomically by the daemon (the single writer)
    events.jsonl       every event, append-only, one JSON object per line, numbered by `seq`
    timeline.jsonl     every job state change, for status views and timeline export
    cursor/<agent>     the last event seq delivered to that agent by `gpuh wait`
    waiters/<agent>.<pid>   present while a `gpuh wait --for <agent>` is running
    cmd/<id>.json      a command from the CLI, waiting for the daemon
    cmd/<id>.out       the daemon's reply to it
    logs/<job>.log     each job's output; logs/<job>.rc its exit code
    daemon.pid, daemon.log, plan.json (the plan as loaded)

`~/.gpuh/current` (or `$GPUH_HOME/current`) names the active run dir, so every `gpuh` command finds it from any
working directory.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .events import Event
from .model import Board


def gpuh_home() -> str:
    return os.environ.get("GPUH_HOME") or os.path.join(os.path.expanduser("~"), ".gpuh")


def current_run() -> Optional[str]:
    env = os.environ.get("GPUH_DIR")
    if env:
        return env
    try:
        with open(os.path.join(gpuh_home(), "current")) as fh:
            path = fh.read().strip()
            return path or None
    except OSError:
        return None


def set_current(run_dir: str) -> None:
    os.makedirs(gpuh_home(), exist_ok=True)
    atomic_write(os.path.join(gpuh_home(), "current"), run_dir + "\n")


def atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Store:
    def __init__(self, run_dir: str) -> None:
        self.dir = os.path.abspath(run_dir)
        for sub in ("cursor", "waiters", "cmd", "logs"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self._seq: Optional[int] = None

    def path(self, *parts: str) -> str:
        return os.path.join(self.dir, *parts)

    # board ----------------------------------------------------------------------------------------------------------
    def save_board(self, board: Board) -> None:
        atomic_write(self.path("board.json"), json.dumps(board.to_json(), indent=1))

    def load_board(self) -> Board:
        with open(self.path("board.json")) as fh:
            return Board.from_json(json.load(fh))

    # events ---------------------------------------------------------------------------------------------------------
    def _last_seq(self) -> int:
        if self._seq is None:
            self._seq = 0
            for e in self.read_events():
                self._seq = max(self._seq, e.seq)
        return self._seq

    def emit(self, ev: Event) -> Event:
        ev.seq = self._last_seq() + 1
        self._seq = ev.seq
        with open(self.path("events.jsonl"), "a") as fh:
            fh.write(json.dumps(ev.to_json()) + "\n")
        return ev

    def read_events(self, after: int = 0) -> Iterator[Event]:
        try:
            with open(self.path("events.jsonl")) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = Event.from_json(json.loads(line))
                    except (ValueError, TypeError):
                        continue
                    if ev.seq > after:
                        yield ev
        except FileNotFoundError:
            return

    def cursor(self, agent: str) -> int:
        try:
            with open(self.path("cursor", agent)) as fh:
                return int(fh.read().strip() or 0)
        except (OSError, ValueError):
            return 0

    def set_cursor(self, agent: str, seq: int) -> None:
        atomic_write(self.path("cursor", agent), f"{seq}\n")

    def pending(self, agent: str) -> List[Event]:
        """Events for agent that its waits have not delivered yet."""
        cur = self.cursor(agent)
        return [e for e in self.read_events(cur) if e.to in (agent, "all")]

    # open decisions ---------------------------------------------------------------------------------------------------
    def write_open(self, agent: str, items: List[Tuple[str, str]]) -> None:
        os.makedirs(self.path("open"), exist_ok=True)
        atomic_write(self.path("open", agent), json.dumps(items))

    def read_open(self, agent: str) -> List[Tuple[str, str]]:
        try:
            with open(self.path("open", agent)) as fh:
                return [tuple(x) for x in json.load(fh)]
        except (OSError, ValueError):
            return []

    def shown(self, agent: str) -> Dict[str, float]:
        try:
            with open(self.path("cursor", agent + ".shown")) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def set_shown(self, agent: str, shown: Dict[str, float]) -> None:
        atomic_write(self.path("cursor", agent + ".shown"), json.dumps(shown))

    # timeline -------------------------------------------------------------------------------------------------------
    def timeline(self, job: str, state: str, t: float, **extra: Any) -> None:
        rec = {"t": t, "job": job, "state": state}
        rec.update(extra)
        with open(self.path("timeline.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def read_timeline(self) -> List[Dict[str, Any]]:
        out = []
        try:
            with open(self.path("timeline.jsonl")) as fh:
                for line in fh:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except FileNotFoundError:
            pass
        return out

    # waiters --------------------------------------------------------------------------------------------------------
    def register_waiter(self, agent: str) -> str:
        p = self.path("waiters", f"{agent}.{os.getpid()}")
        atomic_write(p, f"{time.time()}\n")
        return p

    def live_waiters(self) -> Dict[str, List[int]]:
        out: Dict[str, List[int]] = {}
        try:
            names = os.listdir(self.path("waiters"))
        except OSError:
            return out
        for name in names:
            agent, _, pid = name.rpartition(".")
            if not pid.isdigit():
                continue
            if pid_alive(int(pid)):
                out.setdefault(agent, []).append(int(pid))
            else:
                try:
                    os.remove(self.path("waiters", name))
                except OSError:
                    pass
        return out

    # command spool --------------------------------------------------------------------------------------------------
    def submit(self, cmd: Dict[str, Any]) -> str:
        cid = f"{time.time():.6f}-{uuid.uuid4().hex[:8]}"
        cmd = dict(cmd, id=cid, at=time.time())
        atomic_write(self.path("cmd", f"{cid}.json"), json.dumps(cmd))
        return cid

    def take_commands(self) -> List[Tuple[str, Dict[str, Any]]]:
        out = []
        try:
            names = sorted(n for n in os.listdir(self.path("cmd")) if n.endswith(".json"))
        except OSError:
            return out
        for name in names:
            p = self.path("cmd", name)
            try:
                with open(p) as fh:
                    cmd = json.load(fh)
            except (OSError, ValueError):
                continue
            try:
                os.remove(p)
            except OSError:
                continue
            out.append((name[:-5], cmd))
        return out

    def reply(self, cid: str, ok: bool, text: str) -> None:
        atomic_write(self.path("cmd", f"{cid}.out"), json.dumps({"ok": ok, "text": text}))

    def await_reply(self, cid: str, timeout: float = 20.0) -> Optional[Dict[str, Any]]:
        p = self.path("cmd", f"{cid}.out")
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                with open(p) as fh:
                    d = json.load(fh)
                os.remove(p)
                return d
            except (OSError, ValueError):
                time.sleep(0.1)
        return None

    # daemon ---------------------------------------------------------------------------------------------------------
    def daemon_pid(self) -> Optional[int]:
        try:
            with open(self.path("daemon.pid")) as fh:
                pid = int(fh.read().strip())
        except (OSError, ValueError):
            return None
        return pid if pid_alive(pid) else None
