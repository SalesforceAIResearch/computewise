"""Views of a run: status, the queue with the scheduler's reasons, results, and a timeline export.

`status` is what an agent reads to see the machine at a glance. `export_html` writes a self-contained Gantt chart of
every job on every GPU, with rejections and decisions marked, for people reading a run afterwards.
"""

from __future__ import annotations

import html
import json
from typing import Dict, List, Optional

from .estimate import Estimator
from .events import duration
from .ledger import Store
from .model import Board, CandState, JobState
from .scheduler import Policy, explain, readiness

STATE_MARK = {"waiting": ".", "ready": "o", "running": ">", "stopping": "x", "done": "+", "failed": "!",
              "died": "!", "cancelled": "-"}


def status(board: Board, now: float, est: Optional[Estimator] = None, verbose: bool = False) -> str:
    est = est or _est(board)
    out: List[str] = []
    el = now - board.started_at if board.started_at else 0
    busy = sum(1 for g in board.gpus if board.gpu_job.get(g))
    cpu = [j for j in board.jobs.values() if j.cpu and j.active]
    out.append(f"run {duration(el)} | GPUs busy {busy}/{len(board.gpus)} | "
               + (f"CPU jobs {len(cpu)}/{board.cpu_slots} | " if any(j.cpu for j in board.jobs.values()) else "")
               + ", ".join(f"{s} {n}" for s, n in _counts(board).items() if n))
    for g in board.gpus:
        jid = board.gpu_job.get(g)
        if not jid:
            out.append(f"  GPU {g}: free")
            continue
        j = board.jobs[jid]
        pace = ""
        if j.pace and j.pace.projected_s:
            pace = f", {int(j.pace.done)}/{int(j.pace.total)}, projected {duration(j.pace.projected_s)}"
        out.append(f"  GPU {g}: {j.id} ({j.candidate or '-'}) {j.state.value} {duration(now - (j.started_at or now))}"
                   f"{pace}")
    for j in cpu:
        out.append(f"  CPU: {j.id} ({j.candidate or '-'}) {j.state.value} {duration(now - (j.started_at or now))}")
    out.append("candidates:")
    for c in board.candidates.values():
        jobs = [j for j in board.jobs.values() if j.candidate == c.id]
        marks = "".join(STATE_MARK[j.state.value] for j in jobs)
        extra = f" {c.verdict}" if c.verdict else ""
        out.append(f"  {c.id:28s} {c.state.value}{extra:8s} [{marks}]  branch {c.branch}"
                   + (f"  ({c.reason[:80]})" if c.reason and c.state is not CandState.OPEN else ""))
    if verbose:
        out.append("queue (best first):")
        out.extend("  " + l for l in explain(board, est, now, Policy()))
        out.append("waiting:")
        for j in board.jobs.values():
            if j.state is JobState.WAITING:
                r, s, why = readiness(board, j, Policy())
                if why:
                    out.append(f"  {j.id:28s} {why}")
    return "\n".join(out)


def _counts(board: Board) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for j in board.jobs.values():
        c[j.state.value] = c.get(j.state.value, 0) + 1
    return c


def _est(board: Board) -> Estimator:
    e = Estimator()
    e.load(board.jobs.values())
    return e


def results(board: Board) -> str:
    out = []
    for c in list(board.candidates.values()) + [None]:
        jobs = [j for j in board.jobs.values() if (j.candidate == (c.id if c else None))]
        if not jobs:
            continue
        head = (f"{c.id} [{c.state.value}{' ' + c.verdict if c.verdict else ''}]" if c else "(no candidate)")
        if c and c.reason:
            head += f" — {c.reason}"
        out.append(head)
        for j in jobs:
            if j.state is JobState.DONE:
                secs = (j.ended_at or 0) - (j.started_at or 0)
                out.append(f"  {j.id:30s} {duration(secs):>8s}  {j.result_line}" + ("  (screen)" if j.screen else ""))
            elif j.terminal:
                out.append(f"  {j.id:30s} {j.state.value:>8s}  {j.reason[:120]}")
            else:
                out.append(f"  {j.id:30s} {j.state.value:>8s}")
    return "\n".join(out)


def export_html(store: Store, board: Board, path: str) -> str:
    tl = store.read_timeline()
    t0 = board.started_at or (tl[0]["t"] if tl else 0)
    spans = []
    open_: Dict[str, dict] = {}
    for r in tl:
        j = r.get("job")
        if r.get("state") == "running":
            open_[j] = {"job": j, "gpu": r.get("gpu"), "cand": r.get("candidate"), "start": r["t"] - t0}
        elif r.get("state") in ("done", "failed", "died", "cancelled", "requeued") and j in open_:
            s = open_.pop(j)
            s["end"] = r["t"] - t0
            s["how"] = r["state"]
            spans.append(s)
    end = max([s["end"] for s in spans] + [1.0])
    for j, s in open_.items():
        s["end"], s["how"] = end, "running"
        spans.append(s)
    cands = sorted({s["cand"] or "-" for s in spans})
    palette = ["#2a78d4", "#d97a1b", "#3a9b55", "#9a4fc4", "#c4364f", "#1b9aa6", "#8a6d2b", "#5b6b7f"]
    colour = {c: palette[i % len(palette)] for i, c in enumerate(cands)}
    W, rowh = 1100, 34
    gpus = sorted({s["gpu"] for s in spans if s["gpu"] is not None})
    svg = [f'<svg viewBox="0 0 {W + 140} {rowh * len(gpus) + 60}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="system-ui,sans-serif" font-size="11">']
    for i, g in enumerate(gpus):
        y = 20 + i * rowh
        svg.append(f'<text x="8" y="{y + 20}">GPU {g}</text>')
        for s in spans:
            if s["gpu"] != g:
                continue
            x = 70 + W * s["start"] / end
            w = max(2.0, W * (s["end"] - s["start"]) / end)
            op = "0.45" if s["how"] in ("cancelled", "requeued") else "1"
            tip = html.escape(f'{s["job"]} ({s["cand"]}) {duration(s["end"] - s["start"])} {s["how"]}')
            svg.append(f'<rect x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="{rowh - 10}" rx="4" '
                       f'fill="{colour[s["cand"] or "-"]}" opacity="{op}" stroke="white" stroke-width="2">'
                       f'<title>{tip}</title></rect>')
    svg.append(f'<text x="70" y="{20 + len(gpus) * rowh + 20}" fill="#555">0</text>'
               f'<text x="{70 + W - 40}" y="{20 + len(gpus) * rowh + 20}" fill="#555">{duration(end)}</text></svg>')
    legend = " ".join(f'<span style="display:inline-block;margin-right:14px"><span style="display:inline-block;'
                      f'width:12px;height:12px;border-radius:3px;background:{colour[c]};vertical-align:-2px"></span> '
                      f'{html.escape(c)}</span>' for c in cands)
    rows = "".join(f"<tr><td>{html.escape(c.id)}</td><td>{c.state.value} {html.escape(c.verdict)}</td>"
                   f"<td>{html.escape(c.reason)}</td></tr>" for c in board.candidates.values())
    doc = (f"<!doctype html><meta charset='utf-8'><title>gpuh timeline</title><body style='font-family:system-ui;"
           f"margin:24px;color:#222'><h2 style='margin:0 0 6px'>Run timeline</h2><p style='color:#555;margin:0 0 12px'>"
           f"{html.escape(board.rule)}</p>{legend}<div>{''.join(svg)}</div><p style='color:#666'>Faded bars were "
           f"cancelled or preempted. Hover a bar for its job.</p><table style='border-collapse:collapse' "
           f"cellpadding='4'><tr><th align='left'>candidate</th><th align='left'>state</th><th align='left'>why</th>"
           f"</tr>{rows}</table></body>")
    with open(path, "w") as fh:
        fh.write(doc)
    return path


def export_json(store: Store, board: Board) -> str:
    return json.dumps({"board": board.to_json(), "timeline": store.read_timeline()}, indent=1)
