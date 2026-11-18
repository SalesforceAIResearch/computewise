"""Read a running job's pace from its own log.

Training and evaluation scripts print their progress in a handful of shapes. The reader recognises them without
being told, reads only the bytes appended since the last look, and turns two or more readings into a projection of
the whole job's length. A job may also give its own pattern (`progress` in the plan), with named groups `done` and
`total`, and optionally `timer` with `tunit` (the script's own clock, `s` or `ms`).

Recognised shapes, first match wins:

    step:125/2900 val_loss:4.45 train_time:344.4s step_avg:2755ms     the script's own timer
    step 25/100 train_loss 4.18 step_time 900ms                       a per-step time
     12%|#2        | 36/300 [00:41<05:02,  1.14s/it]                    tqdm
    step 25/100 ...  or  37/300 examples                               a bare counter, timed by the wall clock

A reading at step 0, or before 2% of the job, is not trusted on its own: compilation and warm-up make the first steps
slow. Pace is taken from the difference between readings where there are two, so warm-up drops out. A bar that
reaches its total while the job keeps running (loading weights, a data pass) is a sub-task and is set aside; only
a series still in progress gives a pace. A job that prints no progress at all is measured when it ends.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Tuple

from .model import Pace

_NUM = r"[0-9]+(?:\.[0-9]+)?"

PATTERNS: List[Tuple[str, Pattern[str]]] = [
    ("timer", re.compile(
        r"step[:\s]+(?P<done>\d+)\s*/\s*(?P<total>\d+).*?train_time[:=\s]+(?P<timer>" + _NUM + r")\s*(?P<tunit>ms|s)\b")),
    ("step_time", re.compile(
        r"step[:\s]+(?P<done>\d+)\s*/\s*(?P<total>\d+).*?step_time[:=\s]+(?P<per>" + _NUM + r")\s*(?P<punit>ms|s)\b")),
    ("tqdm", re.compile(r"(?P<done>\d+)/(?P<total>\d+)\s*\[(?P<el>[0-9:]+)<")),
    ("counter", re.compile(r"\b(?:step|iter|iteration|batch|epoch)[:\s]+(?P<done>\d+)\s*/\s*(?P<total>\d+)\b", re.I)),
    ("items", re.compile(r"\b(?P<done>\d+)\s*/\s*(?P<total>\d+)\s+(?:examples|samples|items|prompts|questions)\b", re.I)),
]

RESULT_PATTERNS: List[Pattern[str]] = [
    re.compile(r"TEST_METRICS[:\s].*"),
    re.compile(r".*\bval_loss[:=\s]+" + _NUM + r".*"),
    re.compile(r".*\b(?:final|score|accuracy|acc|f1|bleu|ppl|perplexity|metric)\w*\b.*" + _NUM + r".*", re.I),
]

MIN_FRACTION = 0.02


def _seconds(value: str, unit: Optional[str]) -> float:
    v = float(value)
    return v / 1000.0 if unit == "ms" else v


def _clock(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


@dataclass
class Reading:
    done: float
    total: float
    wall: float                    # harness clock
    timer: Optional[float] = None  # the script's own clock, seconds
    per: Optional[float] = None    # seconds per unit, if printed
    line: str = ""


def parse_line(line: str, custom: Optional[Pattern[str]] = None) -> Optional[Reading]:
    """One log line -> a reading, or None. The wall clock is filled in by the caller."""
    pats = ([("custom", custom)] if custom else []) + PATTERNS
    for name, pat in pats:
        m = pat.search(line)
        if not m:
            continue
        g = m.groupdict()
        try:
            done, total = float(g["done"]), float(g["total"])
        except (KeyError, TypeError, ValueError):
            continue
        if total <= 0 or done > total:
            continue
        r = Reading(done=done, total=total, wall=0.0, line=line.strip()[:240])
        if g.get("timer"):
            r.timer = _seconds(g["timer"], g.get("tunit"))
        if g.get("per"):
            r.per = _seconds(g["per"], g.get("punit"))
        if g.get("el"):
            r.timer = _clock(g["el"])
        return r
    return None


class LogReader:
    """Follows one job's log: new bytes only, lines split on newlines and carriage returns."""

    def __init__(self, path: str, custom: Optional[str] = None, keep: int = 40,
                 milestones: Optional[List[Dict[str, str]]] = None) -> None:
        self.path = path
        self.custom = re.compile(custom) if custom else None
        # a milestone is a value the plan names, read from the first line its pattern matches (`value` group)
        self.milestone_rx = [(m["name"], re.compile(m["pattern"])) for m in (milestones or [])]
        self.milestones: Dict[str, Tuple[float, str]] = {}   # name -> (value, the line it came from)
        self.offset = 0
        self.partial = ""
        self.readings: List[Reading] = []   # the current progress series (one bar, one counter)
        self.final: Optional[Reading] = None  # the last reading that reached its total
        self.tail: List[str] = []
        self.keep = keep

    def poll(self, now: float, started_at: float) -> Optional[Pace]:
        """Read what was appended; return the current pace if the log shows any."""
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(1 << 20)
                self.offset += len(chunk)
        except OSError:
            return self.pace(now, started_at)
        if chunk:
            text = self.partial + chunk.decode("utf-8", "replace")
            pieces = re.split(r"[\r\n]", text)
            self.partial = pieces.pop()
            for line in pieces:
                if not line.strip():
                    continue
                self.tail.append(line)
                for name, rx in self.milestone_rx:
                    if name not in self.milestones:
                        m = rx.search(line)
                        if m:
                            try:
                                self.milestones[name] = (float(m.group("value")), line.strip()[:200])
                            except (TypeError, ValueError):
                                pass
                r = parse_line(line, self.custom)
                if r:
                    r.wall = now
                    if r.done >= r.total:
                        # a bar that completed: the job's last line, or a sub-task (loading weights) inside a job that
                        # goes on. Either way it says nothing about the pace of what is still running.
                        self.final = r
                        self.readings = [x for x in self.readings if x.total != r.total]
                    elif self.readings and r.total != self.readings[-1].total:
                        self.readings = [r]          # a different counter: start a new series
                    elif not self.readings or r.done > self.readings[-1].done:
                        self.readings.append(r)
            if len(self.tail) > self.keep:
                self.tail = self.tail[-self.keep:]
            if len(self.readings) > 64:
                self.readings = self.readings[:1] + self.readings[-63:]
        return self.pace(now, started_at)

    def pace(self, now: float, started_at: float) -> Optional[Pace]:
        rs = [r for r in self.readings if 0 < r.done < r.total]
        if not rs:
            return None
        last = rs[-1]
        frac = last.done / last.total
        elapsed = max(0.0, now - started_at)
        per_timer: Optional[float] = None   # seconds of the script's own clock per unit
        per_wall: Optional[float] = None    # wall seconds per unit
        if len(rs) >= 2:
            first = rs[0] if len(rs) < 4 else rs[len(rs) // 2]
            du = last.done - first.done
            if du > 0:
                if last.timer is not None and first.timer is not None:
                    per_timer = (last.timer - first.timer) / du
                per_wall = (last.wall - first.wall) / du if last.wall > first.wall else None
        if per_timer is None:
            if last.per is not None:
                per_timer = last.per
            elif last.timer is not None:
                per_timer = last.timer / last.done
        if len(rs) < 2 and frac < MIN_FRACTION and last.per is None:
            return None
        remaining = last.total - last.done
        projected_timer = (last.timer + remaining * per_timer) if (last.timer is not None and per_timer) else (
            last.total * per_timer if per_timer and last.per is not None else None)
        if per_wall:
            projected = elapsed + remaining * per_wall
        elif per_timer:
            # wall overhead so far (start-up, data loading) plus the remaining units at the script's own pace
            overhead = max(0.0, elapsed - (last.timer if last.timer is not None else last.done * per_timer))
            projected = overhead + last.total * per_timer
        else:
            projected = elapsed / frac if frac > 0 else None
        return Pace(done=last.done, total=last.total, elapsed_s=elapsed, per_unit_s=per_timer or per_wall,
                    projected_s=projected, projected_timer_s=projected_timer, line=last.line, at=now)

    def result_line(self, custom: Optional[str] = None) -> str:
        """The line the job's result is in: the last one matching a result pattern, else the last line."""
        lines = list(self.tail)
        try:
            size = os.path.getsize(self.path)
            with open(self.path, "rb") as fh:
                fh.seek(max(0, size - 65536))
                lines = re.split(r"[\r\n]", fh.read().decode("utf-8", "replace"))
        except OSError:
            pass
        lines = [l.strip() for l in lines if l.strip()]
        pats = ([re.compile(custom)] if custom else []) + RESULT_PATTERNS
        for pat in pats:
            for line in reversed(lines):
                m = pat.search(line)
                if m:
                    return (m.group(0) if pat.pattern.startswith("TEST_METRICS") else line)[:240]
        return lines[-1][:240] if lines else ""

    def last_error(self) -> str:
        lines = [l for l in self.tail if l.strip()]
        for line in reversed(lines):
            if re.search(r"(?i)error|exception|traceback|killed|not found|no such", line):
                return line.strip()[:240]
        return lines[-1].strip()[:240] if lines else ""
