"""Load and validate the job graph the agent writes.

A plan is one JSON file:

    {
      "rule": "deploy a method only if, on the whole workloads, it is within 2 points of full_attention ...",
      "evidence": "whole workloads (--max-examples 0); a slice is a screen and never rejects",
      "workdir": "/home/researcher/project",
      "gpus": [0, 1, 2, 3],                       optional: detected from CUDA_VISIBLE_DEVICES or nvidia-smi
      "policy": {"speculate": true},              optional: see scheduler.Policy
      "candidates": [
        {"id": "full_attention", "branch": "ref", "reference": true},
        {"id": "kivi", "branch": "quant"}
      ],
      "jobs": [
        {"id": "fa_hotpotqa", "candidate": "full_attention", "kind": "hotpotqa", "role": "check",
         "command": "bash methods/full_attention/scripts/longbench_hotpotqa.sh", "estimate_s": 400}
      ]
    }

A job entry may carry `foreach`, a mapping of names to lists, `role_by` (a value of any foreach name -> role, to
mark checks inside a template), and `except` (a list of {name: value} combinations to leave out). It expands to one job per combination, with
`{name}` replaced in every string field (`id`, `command`, `candidate`, `kind`, `cwd`, env values, `after`,
`needs_pass`), and an optional `estimate_s_by` mapping a kind to its estimate:

    {"foreach": {"m": ["kivi", "squat"], "w": ["hotpotqa", "gsm8k"]},
     "id": "{m}_{w}", "candidate": "{m}", "kind": "{w}", "command": "bash methods/{m}/scripts/{w}.sh",
     "estimate_s_by": {"hotpotqa": 400, "gsm8k": 3600}}

Validation refuses: duplicate ids, unknown candidates or dependencies, dependency cycles, empty commands, and a
job gated on a check that is not a check.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .model import Board, Candidate, Job, Role
from .scheduler import Policy, gates

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+=-]{0,127}$")
STRING_FIELDS = ("id", "command", "candidate", "kind", "group", "cwd", "progress", "result")
LIST_FIELDS = ("after", "needs_pass")


class PlanError(ValueError):
    pass


def _fill(value: Any, names: Dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**names)
        except (KeyError, IndexError, ValueError):
            out = value
            for k, v in names.items():
                out = out.replace("{" + k + "}", v)
            return out
    if isinstance(value, list):
        return [_fill(v, names) for v in value]
    if isinstance(value, dict):
        return {k: _fill(v, names) for k, v in value.items()}
    return value


def expand(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in entries:
        fe = e.get("foreach")
        if not fe:
            out.append(dict(e))
            continue
        if not isinstance(fe, dict) or not fe:
            raise PlanError(f"foreach must map names to lists: {fe!r}")
        keys = list(fe)
        for combo in itertools.product(*[list(fe[k]) for k in keys]):
            names = {k: str(v) for k, v in zip(keys, combo)}
            j = {k: _fill(v, names) for k, v in e.items() if k not in ("foreach", "estimate_s_by", "role_by", "except")}
            if any(all(names.get(k) == str(v) for k, v in ex.items()) for ex in (e.get("except") or [])):
                continue
            by = e.get("estimate_s_by") or {}
            kind = j.get("kind")
            if by and kind in by and "estimate_s" not in j:
                j["estimate_s"] = by[kind]
            # role_by: {"<value of any foreach name>": "check"} marks those combinations' role
            for val, role in (e.get("role_by") or {}).items():
                if val in names.values():
                    j["role"] = role
            out.append(j)
    return out


_GPU_ASSIGN = re.compile(r"(?:^|(?<=[\s;&|(]))(?:export\s+)?(?:CUDA_VISIBLE_DEVICES|NVIDIA_VISIBLE_DEVICES|HIP_VISIBLE_DEVICES)=\S*\s*")


def strip_gpu(cmd: str) -> Tuple[str, bool]:
    """Remove GPU assignments from a command: the harness sets the GPU, and one written in the command would override
    it and put two jobs on one GPU."""
    out = _GPU_ASSIGN.sub("", cmd).strip()
    return out, out != cmd.strip()


def job_from(d: Dict[str, Any], default_cwd: str) -> Job:
    if "id" not in d or not d["id"]:
        raise PlanError(f"a job has no id: {d!r}")
    jid = str(d["id"])
    if not _ID.match(jid):
        raise PlanError(f"job id {jid!r}: use letters, digits and _ . : + = - only")
    cmd, _ = strip_gpu(str(d.get("command") or ""))
    if not cmd:
        raise PlanError(f"job {jid}: empty command")
    try:
        role = Role(d.get("role", "needed"))
    except ValueError:
        raise PlanError(f"job {jid}: role must be check, needed or speculative, not {d.get('role')!r}")
    est = d.get("estimate_s")
    if est is not None:
        try:
            est = float(est)
        except (TypeError, ValueError):
            raise PlanError(f"job {jid}: estimate_s must be a number of seconds")
    p = d.get("p_reject")
    if p is not None and not (0.0 <= float(p) <= 1.0):
        raise PlanError(f"job {jid}: p_reject must be between 0 and 1")
    needs_pass = d.get("needs_pass")
    for pat_field in ("progress", "result"):
        if d.get(pat_field):
            try:
                re.compile(d[pat_field])
            except re.error as e:
                raise PlanError(f"job {jid}: {pat_field} is not a valid regex: {e}")
    g = d.get("gpu", d.get("gpus", 1))
    if g not in (0, 1, "0", "1"):
        raise PlanError(f"job {jid}: gpu must be 1 (a GPU job, the default) or 0 (a CPU-only job)")
    milestones = []
    for m in d.get("milestones") or []:
        if not isinstance(m, dict) or not m.get("name") or not m.get("pattern"):
            raise PlanError(f"job {jid}: each milestone needs a name and a pattern")
        try:
            rx = re.compile(m["pattern"])
        except re.error as e:
            raise PlanError(f"job {jid}: milestone {m['name']} pattern is not a valid regex: {e}")
        if "value" not in rx.groupindex:
            raise PlanError(f"job {jid}: milestone {m['name']} pattern needs a named group (?P<value>...)")
        spec = {"name": str(m["name"]), "pattern": str(m["pattern"])}
        if m.get("better") is not None:
            if m["better"] not in ("lower", "higher"):
                raise PlanError(f"job {jid}: milestone {m['name']} better must be \"lower\" or \"higher\"")
            spec["better"] = m["better"]
        if m.get("margin") is not None:
            try:
                spec["margin"] = float(m["margin"])
            except (TypeError, ValueError):
                raise PlanError(f"job {jid}: milestone {m['name']} margin must be a number")
            if spec["margin"] < 0 or "better" not in spec:
                raise PlanError(f"job {jid}: milestone {m['name']} margin must be >= 0, and needs better")
        milestones.append(spec)
    return Job(
        id=jid, command=cmd, candidate=d.get("candidate"), kind=d.get("kind"), group=d.get("group"), role=role,
        after=[str(x) for x in (d.get("after") or [])],
        needs_pass=None if needs_pass is None else [str(x) for x in needs_pass],
        estimate_s=est, tail_s=float(d.get("tail_s") or 0.0),
        p_reject=None if p is None else float(p), screen=bool(d.get("screen", False)),
        cwd=d.get("cwd") or default_cwd, env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
        progress=d.get("progress"), result=d.get("result"),
        gpus=int(g), milestones=milestones,
    )


def validate(board: Board) -> List[str]:
    """Raise PlanError on a broken graph; return warnings."""
    warnings: List[str] = []
    for j in board.jobs.values():
        if j.candidate and j.candidate not in board.candidates:
            raise PlanError(f"job {j.id}: unknown candidate {j.candidate!r}")
        for d in j.after:
            if d not in board.jobs:
                raise PlanError(f"job {j.id}: after unknown job {d!r}")
        for g in (j.needs_pass or []):
            if g not in board.jobs:
                raise PlanError(f"job {j.id}: needs_pass unknown job {g!r}")
            if board.jobs[g].role is not Role.CHECK:
                warnings.append(f"job {j.id} waits for a pass on {g}, which is not a check")
    # cycles over after, needs_pass and the default gates (a needed job waits for its candidate's checks): a needed
    # preparation with a check `after` it waits for its own check (G-fyld-haiku-A4-r1b: refused nothing, ran nothing)
    pol = Policy()
    graph = {j.id: list(j.after) + [g for g in gates(board, j, pol) if g not in j.after] for j in board.jobs.values()}
    state: Dict[str, int] = {}

    def visit(n: str, path: List[str]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            loop = path[path.index(n):] + [n] if n in path else path + [n]
            hint = ""
            for a, b in zip(loop, loop[1:]):
                if b not in board.jobs[a].after and b not in (board.jobs[a].needs_pass or []):
                    hint = (f" ({a} is a needed job, so it waits for every check of its candidate, {b} among them; "
                            f"make {a} a check, or give it \"needs_pass\": [] or the checks it really waits for)")
                    break
            raise PlanError("dependency cycle: " + " -> ".join(loop) + hint)
        state[n] = 1
        for m in graph.get(n, []):
            visit(m, path + [n])
        state[n] = 2

    for n in graph:
        visit(n, [])
    judged = [c for c in board.candidates.values() if not c.reference]
    on_main = [c.id for c in judged if c.branch == "main"]
    if len(judged) >= 2 and on_main:
        raise PlanError("give each candidate a branch teammate to judge it (\"branch\": \"<name>\", one per candidate, "
                        "at most four; group similar ones): " + ", ".join(on_main) + " are on main. Each branch is "
                        "started as its own teammate (Agent tool), never judged from main")
    refs = [c for c in board.candidates.values() if c.reference]
    if len(refs) > 1:
        warnings.append("more than one reference candidate: " + ", ".join(c.id for c in refs))
    for c in board.candidates.values():
        if not any(j.candidate == c.id for j in board.jobs.values()):
            warnings.append(f"candidate {c.id} has no jobs")
    # 'job_from' strips GPU assignments; say so once, so the agent stops writing them
    ests = [j.estimate_s for j in board.jobs.values()]
    if any(e is None for e in ests):
        warnings.append("some jobs have no estimate_s; the scheduler assumes the mean until they are measured")
    known = [e for e in ests if e is not None]
    if len(known) >= 4 and len(set(known)) == 1:
        warnings.append(f"every job is estimated at {known[0]:.0f}s. Jobs that look alike often differ: read each job's "
                        "settings (steps x time per step, examples x time per example) and give each its own estimate, "
                        "or the longest job may start last")
    # the duration model learns how long a kind of work takes by comparing candidates on it. A kind only one candidate
    # runs teaches it nothing it can carry, and with no reference to anchor the scale its predictions can drift far
    # from both the plan and the machine (G-k2lp-haiku-A3: no reference, 11 of 12 kinds run by one candidate each; the
    # harness predicted 116,707 s for a 1,800-s prior, and 8,747 s on a retry, for one 29,157-s job)
    kinds: dict = {}
    for j in board.jobs.values():
        if j.kind and j.gpus != 0:
            kinds.setdefault(j.kind, set()).add(j.candidate)
    lone = sorted(k for k, cs in kinds.items() if len(cs) == 1)
    if not refs and len(kinds) >= 4 and len(lone) * 2 > len(kinds):
        warnings.append(f"no candidate is the reference, and {len(lone)} of {len(kinds)} kinds of work are run by one "
                        "candidate only, so measured lengths can't be carried between jobs reliably; the harness will "
                        "lean on your estimates. Read each job's settings for its own length (examples x time per "
                        "example), especially the long ones, and mark a reference if the brief names a baseline")
    # CPU jobs: the arithmetic of their share of the cores, never a reading of their commands (G-fyld-haiku-A4: seven
    # preps one at a time, each with `--tasks 8 --workers 8` on 42 cores, so 8 ran at once and each prep took 2.7x)
    cpu = [j for j in board.jobs.values() if j.cpu]
    if cpu:
        n = machine_cores()
        at_once = min(board.cpu_slots, len(cpu))
        share = max(1, n // at_once)
        warnings.append(f"{len(cpu)} CPU jobs (\"gpu\": 0), at most {at_once} at once (policy.cpu_slots "
                        f"{board.cpu_slots}): each has nproc / {at_once} = {share} of this machine's {n} cores. Check "
                        f"that each command's own options use about {share}: as many workers, and at least as many "
                        f"tasks, shards or chunks as workers, since a tool runs no more at once than it has")
    return warnings


def machine_cores() -> int:
    """The cores this process may run on: its CPU affinity (a container's cpuset), as `nproc` reports it."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def load(path_or_dict: Any, gpus: Optional[List[int]] = None) -> Tuple[Board, Policy, List[str]]:
    if isinstance(path_or_dict, dict):
        d = path_or_dict
        base = os.getcwd()
    else:
        with open(path_or_dict) as fh:
            try:
                d = json.load(fh)
            except json.JSONDecodeError as e:
                raise PlanError(f"{path_or_dict}: not valid JSON: {e}")
        base = os.path.dirname(os.path.abspath(path_or_dict))
    workdir = os.path.abspath(os.path.join(base, d.get("workdir") or "."))
    board = Board(rule=str(d.get("rule", "")), evidence=str(d.get("evidence", "")), workdir=workdir)
    board.gpus = [int(g) for g in (gpus if gpus is not None else d.get("gpus") or [])]
    board.gpu_job = {g: None for g in board.gpus}
    cs = (d.get("policy") or {}).get("cpu_slots")
    if cs is not None:
        if not isinstance(cs, int) or cs < 1:
            raise PlanError("policy.cpu_slots must be a whole number of at least 1")
        board.cpu_slots = cs
    for c in d.get("candidates") or []:
        if isinstance(c, str):
            c = {"id": c}
        cid = str(c.get("id") or "")
        if not cid or not _ID.match(cid):
            raise PlanError(f"candidate id {cid!r}: use letters, digits and _ . : + = - only")
        if cid in board.candidates:
            raise PlanError(f"duplicate candidate {cid}")
        board.candidates[cid] = Candidate(id=cid, branch=str(c.get("branch") or "main"),
                                          reference=bool(c.get("reference", False)))
    stripped = 0
    for e in expand(d.get("jobs") or []):
        stripped += strip_gpu(str(e.get("command") or ""))[1]
        job = job_from(e, workdir)
        if job.id in board.jobs:
            raise PlanError(f"duplicate job id {job.id}")
        board.jobs[job.id] = job
    if not board.jobs:
        raise PlanError("the plan has no jobs")
    warnings = validate(board)
    if stripped:
        warnings.insert(0, f"removed CUDA_VISIBLE_DEVICES from {stripped} commands: the harness gives each job its GPU")
    return board, Policy.from_json(d.get("policy")), warnings


def merge(board: Board, d: Dict[str, Any], now: float) -> List[str]:
    """Add the candidates and jobs of a plan fragment to a running board. Returns the new job ids."""
    for c in d.get("candidates") or []:
        if isinstance(c, str):
            c = {"id": c}
        cid = str(c.get("id"))
        if cid not in board.candidates:
            board.candidates[cid] = Candidate(id=cid, branch=str(c.get("branch") or "main"),
                                              reference=bool(c.get("reference", False)))
    new: List[Job] = []
    for e in expand(d.get("jobs") or []):
        job = job_from(e, board.workdir)
        if job.id in board.jobs:
            raise PlanError(f"job {job.id} already exists")
        job.added_at = now
        new.append(job)
    trial = Board.from_json(board.to_json())
    for job in new:
        trial.jobs[job.id] = job
    validate(trial)
    for job in new:
        board.jobs[job.id] = job
    return [j.id for j in new]
