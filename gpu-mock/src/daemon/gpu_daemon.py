#!/usr/bin/env python3
"""gpu-mock host daemon: the brain of the fake GPU box.

One daemon per box. It owns the virtual GPUs: a per-GPU queue with a wall-clock
cursor that paces modelled work, proportional-share contention when two jobs
share a GPU, memory accounting with real OOM, the occupancy log that the paper's
utilisation number is computed from, registry lookup by fingerprint, and failure
injection. Job processes talk to it over a Unix socket; the bridge
(bridge/overrides_bridge.py) reads its published state and writes Mokka's
overrides.yaml.

Design: docs/ARCHITECTURE.md. This file has no torch and no GPU dependency; it is
pure Python and runs on the host (inside the container, host-of-the-fake).

Protocol: newline-delimited JSON over AF_UNIX. One request object per line, one
response object per line. Requests carry a "cmd" field; see _dispatch.

Run:  python3 gpu_daemon.py --socket /var/run/nvidia-persistenced/socket \\
                            --registry /path/to/registry --gpus 4 --speed 1.0
"""

from __future__ import annotations

import argparse
import re
import json
import math
import os
import socket
import threading

try:
    import surrogate as _sg
except ImportError:
    # `python -I path/to/gpu_daemon.py` (isolated mode, how the sidecar and the
    # in-image tests start it) does not put the script's directory on sys.path
    # (-P); the module sits beside this file
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    import surrogate as _sg
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Clock. One monotonic source for the whole daemon. Wall-clock in seconds.
# The speed knob scales *modelled* durations before they ever enter a queue, so
# everything downstream (cursor, occupancy, bridge) is already in real wall-time.
# ---------------------------------------------------------------------------


def now() -> float:
    return time.monotonic()


# how early a score may be read before its example's unit ends: the client's
# wait and the daemon's clock are the same CLOCK_MONOTONIC, so this covers only
# the read's own round trip
READ_SLACK_S = 0.05


# ---------------------------------------------------------------------------
# Virtual GPU: a queue with a cursor. The cursor is the wall-clock time at which
# all work enqueued so far will have finished. Enqueuing pushes the cursor out;
# a sync blocks the caller until the cursor is reached. Contention is handled by
# slowing the rate at which every sharer's work drains (proportional share).
# ---------------------------------------------------------------------------


@dataclass
class WorkUnit:
    job_id: int
    seq: int
    op: str                 # "forward" | "backward" | "step" | "kernel" | ...
    modelled_s: float       # already divided by the speed knob
    mem_delta_bytes: int     # change in this job's allocation on this GPU
    enqueued_at: float


@dataclass
class VirtualGPU:
    index: int
    capacity_bytes: int
    # cursor: wall-clock time the queue is busy until. <= now() means idle.
    cursor: float = field(default_factory=now)
    # jobs currently holding memory on this GPU: job_id -> bytes
    mem: dict[int, int] = field(default_factory=dict)
    # jobs that have enqueued unfinished work (for contention counting)
    active_jobs: set[int] = field(default_factory=set)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def used_bytes(self) -> int:
        return sum(self.mem.values())

    def free_bytes(self) -> int:
        return max(0, self.capacity_bytes - self.used_bytes())

    def is_busy(self, t: Optional[float] = None) -> bool:
        return self.cursor > (t if t is not None else now())


# ---------------------------------------------------------------------------
# Occupancy log. The daemon's own record of every state transition, at full
# resolution, in both wall-clock and modelled (knob-undone) time. This is the
# dataset the best-possible-schedule computation and the paper's utilisation
# number read. Append-only JSONL.
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    """The daemon's own log line (stderr, which the sidecar keeps as daemon.log)."""
    import sys as _sys
    print(f"[gpu_daemon] {msg}", file=_sys.stderr, flush=True)


class OccupancyLog:
    def __init__(self, path: Path, speed: float):
        self.path = path
        self.speed = speed
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", buffering=1)  # line-buffered

    def event(self, kind: str, gpu: int, **fields: Any) -> None:
        rec = {
            "t_wall": time.time(),
            "t_mono": now(),
            "speed": self.speed,
            "kind": kind,
            "gpu": gpu,
            **fields,
        }
        line = json.dumps(rec, separators=(",", ":"))
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


# ---------------------------------------------------------------------------
# Registry. Thin for now: a directory of JSON records keyed by fingerprint and
# by (entry_hash, argv). The daemon only reads; gpu-tasks writes. The record
# schema is their RECORD-FORMAT.md; here we touch only the fields the daemon
# needs to pace and serve. convert() (src/convert) adapts a record to the
# requested gpu_count; until it lands the daemon uses the record as-is and notes
# gpu_count==record.gpu_count.
# ---------------------------------------------------------------------------


import os as _os


def _basename(s: str) -> str:
    return _os.path.basename(str(s))


def _takes_value(tok: Optional[str]) -> bool:
    """Whether the token after a bare `--flag` is that flag's value. A token
    starting with "-" is the next flag, unless it is a number (`--k -1`).
    Without this rule a bare boolean exempt flag (`--push_to_hub`) ate the flag
    after it, so a record with `--push_to_hub --gradient_checkpointing` and a
    launch with only `--gradient_checkpointing` got different exact keys, and
    `--logging_first_step --bf16` / `--fp16` collided."""
    if tok is None:
        return False
    if not tok.startswith("-"):
        return True
    try:
        float(tok)
        return True
    except ValueError:
        return False


_PATHLIKE_EXT = (".pt", ".pth", ".bin", ".safetensors", ".json", ".jsonl", ".txt", ".csv", ".parquet",
                 ".yaml", ".yml", ".toml", ".npy", ".npz", ".tar", ".gz", ".zip", ".log", ".md")


def _norm_path_token(tok: str, cwd: Optional[str]) -> str:
    """A path-valued argv token in the form a record stores it: an absolute
    path under the job's working directory becomes cwd-relative, `./a/b`
    becomes `a/b`; everything else is left alone. G-h3-haiku-L3 (2026-10-23)
    resumed with `--resume /home/researcher/project/checkpoints/x.pt` where
    the record held `checkpoints/x.pt`: the same file, and the exact key
    missed on the spelling, so four jobs ran the modelled path."""
    t = str(tok)
    if not t or t.startswith("-"):
        return t
    if "/" not in t and not t.startswith(".") and not t.lower().endswith(_PATHLIKE_EXT):
        return t
    if t.startswith("/") and cwd:
        root = str(cwd).rstrip("/")
        if root and (t == root or t.startswith(root + "/")):
            t = t[len(root) + 1:] or "."
    if "/" in t or t.startswith("."):
        n = os.path.normpath(t)
        if t.endswith("/") and n != "/":
            n += "/"
        t = n
    return t


def _normalise_argv_paths(argv: Optional[list], cwd: Optional[str]) -> list:
    """Every value token (and the value of `--k=v`) through _norm_path_token;
    argv[0] untouched (its basename is what keys use)."""
    out = []
    for i, a in enumerate(argv or []):
        a = str(a)
        if i == 0 or not a:
            out.append(a)
            continue
        if a.startswith("-") and "=" in a and not a.startswith("--="):
            k, v = a.split("=", 1)
            out.append(k + "=" + _norm_path_token(v, cwd))
        else:
            out.append(_norm_path_token(a, cwd))
    return out


def _strip_argv(argv: list, exempt: Optional[list]) -> list:
    """argv with the seed args and the record's exempt args removed, argv[0]
    as basename. Exempt args change logging/output paths, not the trajectory
    (RECORD-FORMAT §5); the seed is matched separately. `--k=v` is one token;
    a bare `--k` takes the next token as its value only if _takes_value says
    so, so boolean exempt flags never swallow the flag after them."""
    # --max-examples is a per_example launch's subset size, not its identity
    # (docs/MLS-KV-SERVING.md §2: a k-subset is the full record's prefix)
    exempt = set(exempt or []) | {"--seed", "--max-examples"}
    argv = list(argv or [])
    out = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if i == 0:
            out.append(_basename(a))
            i += 1
            continue
        key = a.split("=", 1)[0]
        if key in exempt:
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            i += 2 if ("=" not in a and _takes_value(nxt)) else 1
            continue
        out.append(a)
        i += 1
    return out


def _strip_stop_at(argv: list) -> tuple[list, Optional[int]]:
    """argv without its `--stop-at N` / `--stop_at N` / `--stop-at=N`, and N."""
    out, stop, i = [], None, 0
    argv = list(argv or [])
    while i < len(argv):
        a = argv[i]
        key, eq, val = a.partition("=")
        if key in ("--stop-at", "--stop_at"):
            v = val if eq else (argv[i + 1] if i + 1 < len(argv) else None)
            try:
                stop = int(v)
            except (TypeError, ValueError):
                stop = None
            i += 1 if eq else 2
            continue
        out.append(a)
        i += 1
    return out, stop


def _seed_from_argv(argv: list) -> Optional[int]:
    for i, a in enumerate(argv or []):
        if a == "--seed" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except ValueError:
                return None
        if a.startswith("--seed="):
            try:
                return int(a.split("=", 1)[1])
            except ValueError:
                return None
    return None


class Registry:
    """Records in RECORD-FORMAT (gpu-tasks). Indexed by exact key (entry_hash +
    stripped argv), by fingerprint, by replicate group and by record_id.
    Replicate groups are drawn WITHOUT replacement per group; a launch naming a
    seed with a record gets it, one naming a seed without a record gets another
    seed's record (seed_substituted); an exhausted group recycles and logs it.
    All ranks of one job (same job_key) get the same draw."""

    def __init__(self, root: Optional[Path]):
        self.root = root
        self._by_fp: dict[str, list] = {}
        self._by_exact: dict[str, list] = {}
        self._by_entry: dict[str, list] = {}    # entry_hash -> records of that script (near-miss reports)
        self.example_order: dict = {}          # (family, workload) -> {"index": {key: position}, "n": n, "ids": [...]} (gpu-tasks' tables)
        self._by_path: dict[str, list] = {}     # bound_path -> path_bound records
        self._by_bound: dict[str, list] = {}    # bound_path -> records of any other policy (the path_exact stage)
        self._by_basename: dict[str, set] = {}  # script basename -> the bound_paths that end in it
        self._by_id: dict[str, dict] = {}
        self._groups: dict[str, list] = {}
        self._used: dict[str, set] = {}
        self._job_draws: dict[str, dict] = {}
        # Option (b), the study 2026-10-21: a rerun is served a sibling run.
        # _siblings: primary record id -> ordered sibling ids (the task file's
        # jobs[].rerun_siblings, gpu-tasks' build/siblings.py); _served: record
        # id -> job keys it went to in this run. A lookup that lands on a record
        # already served to a DIFFERENT job key is a rerun (all ranks of one
        # job share a key) and gets the first unused sibling, logged
        # draw=sibling; none left -> the primary again, draw=reused.
        self._siblings: dict[str, list] = {}
        self._served: dict[str, set] = {}
        self._task_primaries: set = set()        # record ids any task job lists in `records`
        # surrogate families (docs/SURROGATE-SERVING.md): synthesised serving for a
        # configuration no record covers, only for scripts a family names and,
        # with a task loaded, only for families a task job declares
        self.families: dict[str, _sg.Family] = {}
        self._surrogate_jobs: dict[str, list] = {}       # family -> job ids declaring it
        self._forced_families: set = set()               # families synthesised even where a record covers the launch
        self._launch_ordinals: dict[tuple, int] = {}    # (family, config hash) -> launches so far
        self._recycle: dict[str, int] = {}               # replicate group -> recycled draws so far (task runs)
        self._used_units: set = set()                    # task runs: every serving unit served so far, run-wide
        # which task job a served record ties to (the runner's scorer ties a rank
        # to its job by it): records/rerun_siblings -> the one job that lists them;
        # a decision slot's pool is shared by the decision's slots, so a slot
        # launch binds to the first slot of that decision not yet bound in this
        # run, in launch order, and the lookup event says which
        self._job_of_record: dict[str, str] = {}
        self._slots: dict[str, list] = {}       # decision id -> [(slot, job_id, set(pool ids))]
        self._cand_of_record: dict[str, tuple] = {}   # record id -> (decision id, candidate variant)
        self._bound: dict[str, dict] = {}       # job_key -> bind info
        self._slot_launches: dict[tuple, int] = {}    # (decision, slot) -> launches bound so far
        self.task_id: Optional[str] = None
        self._lock = threading.Lock()
        if root and root.exists():
            self._load()

    def load_surrogates(self, root: Optional[Path]) -> int:
        self.families = _sg.load_families(root)
        return len(self.families)

    def load_task(self, path: Path) -> int:
        """Read a task file's jobs[].rerun_siblings; returns how many primary
        records gained a sibling list. Missing file or field: nothing changes."""
        try:
            task = json.loads(Path(path).read_text())
        except (OSError, ValueError) as e:
            log(f"task file {path} not read: {e!r}")
            return 0
        self.task_id = task.get("task_id")
        n = 0
        for job in task.get("jobs") or []:
            jid = str(job.get("job_id") or "")
            for rid in job.get("records") or []:
                self._task_primaries.add(str(rid))
                self._job_of_record.setdefault(str(rid), jid)
            # a decision slot (`records: null`, `bind: {decision, slot}`) lists
            # every record a candidate's continuation could be in
            # `candidate_pool`; those count as listed
            fam = (job.get("surrogate") or {}).get("family")
            if fam:
                self._surrogate_jobs.setdefault(str(fam), []).append(jid)
                if job.get("surrogate_force"):
                    # the plausible-world test (INTERPRETED-SERVING §5.0): the same
                    # brief on records and on the synthesised tier for the same
                    # configurations; served jobs are labelled surrogate_forced
                    self._forced_families.add(str(fam))
            pool = [str(x) for x in (job.get("candidate_pool") or [])]
            for rid in pool:
                self._task_primaries.add(rid)
            bind = job.get("bind") or {}
            if pool and bind.get("decision") is not None:
                self._slots.setdefault(str(bind["decision"]), []).append((int(bind.get("slot") or 0), jid, set(pool)))
            sibs = [str(x) for x in (job.get("rerun_siblings") or []) if x]
            for sid in sibs:
                self._job_of_record.setdefault(sid, jid)
            if not sibs:
                continue
            for rid in job.get("records") or []:
                self._siblings[str(rid)] = list(sibs)
                n += 1
        for lst in self._slots.values():
            lst.sort(key=lambda t: t[0])
        # the screen-then-confirm tasks also name each candidate's continuation
        for dec in task.get("decisions") or []:
            did = str(dec.get("decision_id") or dec.get("id") or "")
            for cand in dec.get("candidates") or []:
                for rid in cand.get("confirm_records") or []:
                    self._task_primaries.add(str(rid))
                    self._cand_of_record.setdefault(str(rid), (did, cand.get("variant")))
        return n

    def _tie(self, rid: Optional[str], job_key: Optional[str]) -> dict:
        """Which task job a served record ties to, for the lookup event: {task_job,
        bind: {decision, slot} | None, candidate, slot_launches}."""
        if not rid or not self._task_primaries:
            return {}
        if job_key and job_key in self._bound:
            return self._bound[job_key]
        out: dict = {}
        if rid in self._job_of_record:
            out = {"task_job": self._job_of_record[rid], "bind": None}
        else:
            for did, slots in self._slots.items():
                holding = [(slot, jid) for slot, jid, pool in slots if rid in pool]
                if not holding:
                    continue
                free = [(slot, jid) for slot, jid in holding if not self._slot_launches.get((did, slot))]
                slot, jid = (free or holding)[0]
                self._slot_launches[(did, slot)] = self._slot_launches.get((did, slot), 0) + 1
                out = {"task_job": jid, "bind": {"decision": did, "slot": slot},
                       "slot_launches": self._slot_launches[(did, slot)]}
                break
        if rid in self._cand_of_record:
            out["candidate"] = self._cand_of_record[rid][1]
        if job_key and out:
            self._bound[job_key] = out
        return out

    def _task_filter(self, cands: list) -> list:
        """With a task loaded, a first launch may only be served a record the
        task lists for one of its jobs (`records`); siblings come only through
        the rerun rule. The registry can bind many derived records to one path
        (the Q prep packages: every base run's `-prepq` record on one path),
        and drawing from that whole pool rotated records a job was never
        listed for, which the runner scores as a defect. Keys the task does
        not list at all keep the registry's pool (an unlisted script behaves
        as today)."""
        if not self._task_primaries:
            return cands
        listed = [r for r in cands if self._rid(r) in self._task_primaries]
        return listed or cands

    def full_run_for(self, rec: dict) -> Optional[dict]:
        """The full-schedule record a prefix / prepared record was derived from,
        when the registry has it, it is drawable, and it is longer. A launch
        whose stop does not fit the prefix is served the parent: a prepared
        setup launched without `--stop-at` ran the full schedule (muonh 3,250
        steps) and was served the stop-1000 prefix — about 1,500 s where
        hardware takes about 4,900 s."""
        d = (rec.get("identity") or {}).get("derived_from")
        if not isinstance(d, dict) or d.get("kind") not in ("prefix", "prepared") or not d.get("record"):
            return None
        par = self._by_id.get(str(d["record"]))
        if par is None or not _drawable(par):
            return None
        try:
            if int(par.get("steps") or 0) <= int(rec.get("steps") or 0):
                return None
        except (TypeError, ValueError):
            return None
        return par

    def return_draw(self, rec: dict, job_key: Optional[str]) -> None:
        """Give a drawn record back: the launch ended before the short-run
        floor (a smoke test). The next launch of that configuration draws the
        same record as a first draw, and a relaunch under the same job key is
        not a rerun of it."""
        rid = self._rid(rec)
        # the draw registers the rid under the POOL's used set (keyed by the
        # first candidate's group), which need not be this record's own group
        # when a pool mixes groups: discard it wherever it was registered
        for used in self._used.values():
            used.discard(rid)
        self._used_units.discard(self._unit(rec))
        if job_key is not None:
            srv = self._served.get(rid)
            if srv is not None:
                srv.discard(job_key)
                if not srv:
                    self._served.pop(rid, None)

    def _sibling_if_rerun(self, rec: Optional[dict], draw: str, job_key: Optional[str]) -> tuple[Optional[dict], str, dict]:
        """Apply the rerun rule to a draw. Returns (record, draw, extra info)."""
        if rec is None or not job_key:
            return rec, draw, {}
        rid = self._rid(rec)
        served = self._served.setdefault(rid, set())
        if not served or job_key in served:
            served.add(job_key)
            return rec, draw, {}
        if rid not in self._siblings:
            # no sibling list for this record (no task file, or none listed):
            # the draw's own label stands (exact_seed_reused, recycled, ...)
            served.add(job_key)
            return rec, draw, {}
        sibs = self._siblings.get(rid) or []
        for k, sid in enumerate(sibs):
            s = self._by_id.get(sid)
            if s is None or not (s.get("readiness") or {}).get("replay", True):
                continue
            if self._served.get(sid):
                continue        # already someone's record or an earlier rerun's sibling
            self._served.setdefault(sid, set()).add(job_key)
            return s, "sibling", {"sibling_of": rid, "sibling_rank": k}
        served.add(job_key)
        return rec, "reused", {"sibling_of": rid, "siblings_exhausted": len(sibs)}

    def _load_example_order(self, p, doc: dict) -> None:
        """A workload's example order (gpu-tasks' `example-order-*.json`, in
        records/): position i of the record's per-example vector is the example
        whose key is order[i]. A table whose keys are not unique, or whose length
        is not n_examples_full, is refused and logged, never half-used."""
        order, n = doc.get("order") or [], doc.get("n_examples_full")
        fams = doc.get("families") or ([doc["family"]] if doc.get("family") else [])
        wl = doc.get("workload")
        bad = None
        if not wl or not fams or not order:
            bad = "missing workload, families or order"
        elif n is not None and len(order) != int(n):
            bad = f"{len(order)} keys for n_examples_full {n}"
        elif len(set(order)) != len(order):
            bad = f"{len(order) - len(set(order))} duplicate keys"
        elif not ("example_id" in str(doc.get("key")) and "prompt" in str(doc.get("key"))):
            bad = f"key {doc.get('key')!r}, expected {EXAMPLE_ORDER_KEY!r}"
        if bad:
            log(f"example order {p.name} refused: {bad}")
            return
        import hashlib
        try:
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError:
            sha = None
        ent = {"index": {k: i for i, k in enumerate(order)}, "n": len(order), "ids": doc.get("ids") or [], "file": p.name, "sha256": sha}
        for fam in fams:
            self.example_order[(str(fam), str(wl))] = ent

    def entry_has_order(self, entry_hash: Optional[str]) -> bool:
        """True when some per_example record of this script has an example-order
        table: only then may a module the launch imported stand in for its entry
        script (an agent's shard runner), since a shard is served by content."""
        return any(_per_example(r) and self.order_for(r) is not None for r in self._by_entry.get(entry_hash or "") or [])

    def order_for(self, rec: Optional[dict]) -> Optional[dict]:
        """The example-order table of a per_example record: its family (the
        record's MLS-Bench task) and workload."""
        if not rec:
            return None
        task = ((rec.get("identity") or {}).get("source_key") or {}).get("task") or ""
        fam = "quant" if "quant" in task else ("selection" if "selection" in task else None)
        wl = (((rec.get("materialisation") or {}).get("serving") or {}).get("decode") or {}).get("workload")
        if fam is None or not wl:
            return None
        return self.example_order.get((fam, wl))

    def _load(self) -> None:
        paths = sorted(self.root.glob("*.json"))
        if not paths and (self.root / "records").exists():
            paths = sorted((self.root / "records").glob("*.json"))
        for p in paths:
            if p.name.startswith("._"):
                continue        # macOS AppleDouble sidecars, left by a tarball
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    rec = json.load(fh)
            except (ValueError, OSError):
                # ValueError covers JSONDecodeError and UnicodeDecodeError: a
                # binary file among the records crashed the daemon at start
                #
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get("schema") == EXAMPLE_ORDER_SCHEMA:
                self._load_example_order(p, rec)
                continue
            ident = rec.get("identity", {})
            rid = ident.get("record_id") or p.stem
            self._by_id[rid] = rec
            fp = ident.get("fingerprint")
            if fp:
                self._by_fp.setdefault(fp, []).append(rec)
            exact = ident.get("entry_hash")
            if ident.get("argv") is not None:
                ident["argv"] = _normalise_argv_paths(ident.get("argv"), None)
            argv = ident.get("argv")
            if exact and argv is not None:
                key = _exact_key(exact, _strip_argv(argv, ident.get("argv_exempt")))
                self._by_exact.setdefault(key, []).append(rec)
                self._by_entry.setdefault(exact, []).append(rec)
            grp = ident.get("replicate_group") or exact or rid
            self._groups.setdefault(grp, []).append(rec)
            bp = ident.get("bound_path")
            if ident.get("match_policy") == "path_bound" and bp:
                self._by_path.setdefault(_norm_bound_path(bp), []).append(rec)
            elif bp:
                # path_exact: an exact_only
                # record that also names its path serves an EDITED copy of its
                # script at that path with its argv — a print added to an evaluator
                # is the recorded evaluator, not the modelled path at 10x
                nb = _norm_bound_path(bp)
                self._by_bound.setdefault(nb, []).append(rec)
                self._by_basename.setdefault(nb.rsplit("/", 1)[-1], set()).add(nb)

    @staticmethod
    def _rid(rec: dict) -> str:
        return rec.get("identity", {}).get("record_id", "?")

    @staticmethod
    def _seed(rec: dict) -> Optional[int]:
        s = rec.get("identity", {}).get("seed")
        if isinstance(s, dict):
            s = s.get("value")
        try:
            return int(s) if s is not None else None
        except (TypeError, ValueError):
            return None

    def _draw(self, cands: list, want_seed: Optional[int]) -> tuple[Optional[dict], str]:
        """Pick a record from candidates sharing a replicate group.

        Records whose readiness.replay is false are never drawn (RECORD-FORMAT
        §4): a run cut short by a suppressed failure with continuation
        `sibling_required`, or any later blocker, stays out of the pool even
        though it shares the group. If nothing drawable remains, the caller
        sends the launch to the modelled path."""
        cands = [r for r in cands if _drawable(r)]
        if not cands:
            return None, "no_drawable_record"
        grp = cands[0].get("identity", {}).get("replicate_group") \
            or cands[0].get("identity", {}).get("entry_hash") or self._rid(cands[0])
        used = self._used.setdefault(grp, set())
        if self.task_id is not None:
            return self._draw_task(cands, want_seed, grp, used)
        if want_seed is not None:
            # A launch naming a seed that has a record always gets that record,
            # even when a seedless launch already took it as a substitute: the
            # seed is the stronger identity, and two launches of one seed are
            # one run. Before this, the substitute consumed it and the exact
            # launch got a "recycled" other record (gpu-tasks' pilot, 2026-10-20).
            for r in cands:
                if self._seed(r) == want_seed:
                    reused = self._rid(r) in used
                    used.add(self._rid(r))
                    return r, "exact_seed_reused" if reused else "exact_seed"
            # a seed with no record: substitute another seed's record
            for r in sorted(cands, key=self._rid):
                if self._rid(r) not in used:
                    used.add(self._rid(r))
                    return r, "seed_substituted"
        for r in sorted(cands, key=self._rid):
            if self._rid(r) not in used:
                used.add(self._rid(r))
                return r, "without_replacement"
        # exhausted: recycle
        used.clear()
        r = sorted(cands, key=self._rid)[0]
        used.add(self._rid(r))
        return r, "recycled"

    @staticmethod
    def _unit(rec: dict) -> str:
        """The underlying run a record serves: `common.served_unit` when the
        record names it (a step-1250 prefix and the path-bound full record of
        one run share it, so a task run never shows one run twice as two
        records; gpu-tasks, 15:39Z 2026-10-21), else the record id."""
        u = ((rec.get("common") or {}).get("served_unit"))
        if u:
            return str(u)
        # lineage fallback: a prefix (`-stopN`) or prepared/path-bound copy of a
        # parent run serves that parent; a chain segment is its own unit
        d = (rec.get("identity") or {}).get("derived_from")
        if isinstance(d, dict) and d.get("kind") in ("prefix", "prepared") and d.get("record"):
            return str(d["record"])
        return Registry._rid(rec)

    def _draw_task(self, cands: list, want_seed: Optional[int], grp: str, used: set) -> tuple[Optional[dict], str]:
        """The draw with a task loaded (gpu-tasks, 15:07Z 2026-10-21, for the
        goal-level packages where the agent picks its own seeds): no record is
        served twice while an unused one of the setup remains, because two
        launches showing one curve is a visible tell and a substituted seed is
        not. The exact seed gets its record only while that record is unused;
        otherwise an unused record, `seed_substituted`, taken in DESCENDING
        seed order so the low seeds agents launch next stay available. When
        the pool is spent the draw is `recycled` round-robin without forgetting
        what was served, and _sibling_if_rerun then hands out the job's
        rerun_siblings and labels exhaustion `reused`. A repeated exact seed whose
        record is spent gets that record again, `exact_seed_reused` (2026-10-24):
        substitution is for seeds with no record, never for a seed that has one."""
        # the "never twice" set is RUN-WIDE and keyed by serving unit, not per
        # replicate group: a prefix record carries the task's group and the
        # path-bound full record of the same run carries the parent's, and a
        # per-group set let one run be served once through each (G1
        # simulation, 16:20Z 2026-10-21)
        order = sorted(cands, key=lambda r: (-(self._seed(r) if self._seed(r) is not None else -1), self._rid(r)))

        def take(r: dict) -> None:
            used.add(self._rid(r))
            self._used_units.add(self._unit(r))

        if want_seed is not None:
            own = [r for r in cands if self._seed(r) == want_seed]
            for r in own:
                if self._unit(r) not in self._used_units:
                    take(r)
                    return r, "exact_seed"
            if own:
                # The seed HAS a record and it is spent: a second launch of one
                # seed is one run on hardware (the same curve again), so serve
                # that record, labelled. Until 2026-10-24 this fell through to
                # the substitution below, which gave a duplicate `--seed 1`
                # launch seed 0's record and then recycled that record to the
                # genuine `--seed 0` launch: two different seeds showing one
                # curve, the very tell the substitution exists to avoid
                #. The no-task
                # draw has done this since 2026-10-20.
                r = own[0]
                take(r)
                return r, "exact_seed_reused"
        for r in order:
            if self._unit(r) not in self._used_units:
                take(r)
                # a seed with NO record of its own: another seed's record
                return r, "seed_substituted" if want_seed is not None else "without_replacement"
        # the pool is spent. A launch that names a seed one of the records
        # carries is a rerun of THAT configuration: recycle that record, so the
        # rerun rule hands out its own job's siblings. The round-robin below
        # ignored the seed and took the pool in descending seed order, so a
        # seed-3 rerun of run4 was served run5's sibling (Q-p20tail-sonnet-
        # L0-bashdefault-r2b, the study 2026-10-23 00:15Z).
        if want_seed is not None:
            for r in order:
                if self._seed(r) == want_seed:
                    return r, "recycled"
        k = self._recycle.get(grp, 0)
        self._recycle[grp] = k + 1
        return order[k % len(order)], "recycled"

    def _effective_candidates(self, entry_hash: str, argv: list, eff: dict,
                              defaults: Optional[dict] = None) -> list:
        """Records of this script whose explicit, non-exempt flags all equal the
        launch's effective values, and whose explicit flags cover the launch's
        explicit non-exempt flags. A launch flag the record never named ran at
        the record's default, which the launch may have changed, so it is a
        miss — unless the launch's value IS the script's own default (the
        patch sends the parser's defaults as `argv_defaults`): then the
        record's silent run used that value too, the script bytes being the
        same. Until 2026-10-25 every such flag missed: G-kvrep-astra-L0-r4's
        `--model-id Qwen/Qwen2.5-3B-Instruct`, the quant script's default,
        against a record whose argv omits it, and every H5 agent passes it
       . Flag `--k` is the dest `k` with
        dashes as underscores; a bare flag means True; numbers compare as
        numbers. A record with a multi-value flag group is left to the
        literal key."""
        recs = self._by_entry.get(entry_hash) or []
        if not recs:
            return []
        mine = _flag_groups(_strip_argv(argv, None))
        if mine is None:
            return []
        return [r for r in recs if self._effective_match(r, mine, eff, defaults)]

    @staticmethod
    def _effective_match(r: dict, mine: dict, eff: dict, defaults: Optional[dict] = None) -> bool:
        """One record against the launch's flag groups and parsed values (the
        rule of _effective_candidates)."""
        return Registry._effective_defaulted(r, mine, eff, defaults) is not None

    @staticmethod
    def _effective_defaulted(r: dict, mine: dict, eff: dict, defaults: Optional[dict] = None) -> Optional[list]:
        """The rule of _effective_candidates; returns the launch flags accepted
        at the script's default (possibly empty) when the record matches, None
        when it does not."""
        ident = r.get("identity", {})
        theirs = _flag_groups(_strip_argv(ident.get("argv"), ident.get("argv_exempt")))
        if theirs is None:
            return None
        for flag, val in theirs.items():
            dest = flag.lstrip("-").replace("-", "_")
            if dest not in eff or not _same_value(eff[dest], val):
                return None
        ex = set(ident.get("argv_exempt") or []) | {"--seed", "--max-examples"}
        defaulted: list = []
        for flag in mine:
            if flag in theirs or flag in ex:
                continue
            dest = flag.lstrip("-").replace("-", "_")
            if defaults and dest in defaults and dest in eff and defaults[dest] is not None \
                    and _same_value(eff[dest], defaults[dest]):
                defaulted.append(flag)
                continue
            return None
        return defaulted

    def _argv_matches(self, r: dict, argv: Optional[list], eff: Optional[dict], mine: Optional[dict],
                      defaults: Optional[dict] = None) -> Optional[str]:
        """Does the launch's argv name this record's computation: literally (the
        exact key's rule, the record's exempt flags stripped from both) or
        effectively (parsed values against the record's explicit flags)."""
        ident = r.get("identity", {})
        ex = ident.get("argv_exempt")
        if argv is not None and _canonical_argv(_strip_argv(argv, ex)) == _canonical_argv(_strip_argv(ident.get("argv") or [], ex)):
            return "literal"
        if eff and mine is not None and self._effective_match(r, mine, eff, defaults):
            return "effective"
        return None

    def _path_exact_candidates(self, script_path: Optional[str], argv: Optional[list],
                               eff: Optional[dict], defaults: Optional[dict] = None) -> tuple[Optional[str], list, Optional[str]]:
        """Records of an exact_only family that names its path, for a launch
        whose script bytes missed (an edited copy): (bound_path, records, rung).
        Rung "suffix": the live path ends with a record's bound_path and the argv
        matches; a known path with no argv match returns no records (a new
        config runs the modelled path, disclosed, never a guess). Rung
        "basename": a flattened copy (the family directory gone) matches by
        script name + argv only when exactly one family has such records;
        two → "ambiguous", no records."""
        if not script_path or not self._by_bound:
            return None, [], None
        sp = str(script_path).replace("\\", "/")
        mine = _flag_groups(_strip_argv(argv, None)) if argv is not None else None

        def matching(recs: list) -> list:
            return [r for r in recs if self._argv_matches(r, argv, eff, mine, defaults)]
        for bp, lst in self._by_bound.items():
            if _path_under(sp, bp):
                m = matching(lst)
                return bp, m, ("suffix" if m else "suffix_argv_miss")
        bn = sp.rsplit("/", 1)[-1]
        groups = [(bp, matching(self._by_bound[bp])) for bp in sorted(self._by_basename.get(bn) or ())]
        groups = [(bp, m) for bp, m in groups if m]
        if len(groups) == 1:
            return groups[0][0], groups[0][1], "basename"
        return None, [], ("ambiguous" if len(groups) > 1 else None)

    def near_miss(self, entry_hash: Optional[str], argv: Optional[list], recs: Optional[list] = None) -> Optional[dict]:
        """When a launch of a script the registry knows misses every match: the
        closest record of that script (fewest differing argv tokens) and the
        tokens that differ, for the occupancy log. A `match: none` on a task
        job's script is a launch that will run the modelled path for real and
        print nothing for a long time; the runner should see it at once rather
        than read a hang into it."""
        # `recs`: the records to compare against when the script's bytes are not
        # the registry's (an edited copy at a known path, path_exact's suffix_argv_miss)
        recs = recs if recs is not None else (self._by_entry.get(entry_hash or "") or [])
        if not recs:
            return None
        mine = _canonical_argv(_strip_argv(argv, None))
        best = None
        for r in recs:
            ident = r.get("identity", {})
            theirs = _canonical_argv(_strip_argv(ident.get("argv"), ident.get("argv_exempt")))
            a, b = set(mine[1:]), set(theirs[1:])
            d = len(a ^ b)
            if best is None or d < best[0]:
                best = (d, ident.get("record_id"), sorted(a - b), sorted(b - a))
        return {"record_id": best[1], "only_launch": best[2], "only_record": best[3],
                "records_of_script": len(recs)}

    def match_path(self, script_path: Optional[str]) -> tuple[Optional[str], list]:
        """The path_bound records whose bound_path the script's resolved path
        ends with (the agent may run from any cwd, so the task root is never
        needed): (bound_path, records), or (None, [])."""
        if not script_path or not self._by_path:
            return None, []
        for bp, lst in self._by_path.items():
            if _path_under(script_path, bp):
                return bp, lst
        return None, []

    def _surrogate_for(self, script_path: Optional[str]) -> Optional[_sg.Family]:
        """The family that synthesises this script, if any. With a task file
        loaded only families a task job declares (`surrogate: {family}`) serve;
        without one every loaded family does (gate runs and probes)."""
        for fam in self.families.values():
            if fam.serves(script_path) and (self.task_id is None or fam.name in self._surrogate_jobs):
                return fam
        return None

    def _forced_for(self, script_path: Optional[str]) -> Optional[_sg.Family]:
        for name in self._forced_families:
            fam = self.families.get(name)
            if fam is not None and fam.serves(script_path):
                return fam
        return None

    def _surrogate_record(self, fam: _sg.Family, argv: Optional[list], fingerprint_inputs: Optional[dict],
                          pending: bool) -> dict:
        cfg = fam.resolve(argv, fingerprint_inputs)
        h = fam.config_hash(cfg)
        if pending:
            ordinal = self._launch_ordinals.get((fam.name, h), 0)
        else:
            ordinal = self._launch_ordinals.get((fam.name, h), 0)
            self._launch_ordinals[(fam.name, h)] = ordinal + 1
        return fam.record(cfg, fam.seed(cfg, ordinal), ordinal, pending=pending)

    def peek_whole_job(self, entry_hash: Optional[str], argv: Optional[list],
                       script_path: Optional[str] = None,
                       argv_effective: Optional[dict] = None,
                       argv_defaults: Optional[dict] = None) -> Optional[dict]:
        """The whole_job record a launch's exact or path lookup would draw from,
        READ-ONLY: no draw, no used-set, no served set, no cached job draw
        (next/ r8.7). The pre-import peek (whole_job_probe.arm) runs in every
        python launch with an entry script, before the launch's own lookup. It
        called lookup(), which DRAWS: every launch consumed one record before its
        real lookup, so seedless launches skipped records, a named seed's record
        came back exact_seed_reused and a one-record group recycled (trials
        known answer 10/15 on r8.1-r8.6; 15/15 with the peek off). The peek only
        needs the seam, which every record of one script and argv shares, so it
        reads the candidate pools the exact, effective, path_exact and path
        stages read, and returns the first drawable whole_job record, else None."""
        with self._lock:
            cands: list = []
            if entry_hash and argv is not None:
                cands = list(self._by_exact.get(_exact_key(entry_hash, _strip_argv(argv, None))) or [])
                if not cands:
                    for key, lst in self._by_exact.items():
                        ex = lst[0].get("identity", {}).get("argv_exempt")
                        if key == _exact_key(entry_hash, _strip_argv(argv, ex)):
                            cands = list(lst)
                            break
                if not cands and argv_effective:
                    cands = self._effective_candidates(entry_hash, argv, argv_effective, argv_defaults)
            if not cands and script_path and self._by_bound:
                _, cands, _ = self._path_exact_candidates(script_path, argv, argv_effective, argv_defaults)
            if not cands and script_path:
                _, cands = self.match_path(script_path)
            for r in self._task_filter(list(cands or [])):
                if _drawable(r) and _whole_job(r) is not None:
                    return r
            return None

    def lookup(self, fingerprint: Optional[str], entry_hash: Optional[str],
               argv: Optional[list], job_key: Optional[str] = None,
               stage: str = "exact", script_path: Optional[str] = None,
               fingerprint_inputs: Optional[dict] = None,
               argv_effective: Optional[dict] = None,
               argv_defaults: Optional[dict] = None,
               modules: Optional[list] = None) -> tuple[Optional[dict], str, dict]:
        """Return (record, mode, info). mode: replay | model | refuse.
        info: match (exact|path|structural|none), draw, group. Order: exact
        (script bytes + argv), then path (a path_bound record whose bound_path
        the script's path ends with: the agent wrote the script, the brief
        named the path; gpu-tasks' preparation packages, 2026-10-20), then
        structural (fingerprint, at the first step)."""
        with self._lock:
            if job_key and job_key in self._job_draws:
                d = self._job_draws[job_key]
                sur = (d["rec"] or {}).get("surrogate") or {}
                if sur.get("pending") and stage == "structural":
                    # the live fingerprint is in: the final configuration, seed
                    # and timing for this launch
                    fam = self.families.get(sur.get("family"))
                    if fam is not None:
                        forced = bool(sur.get("forced"))
                        rec = self._surrogate_record(fam, argv, fingerprint_inputs, pending=False)
                        if forced:
                            rec["surrogate"]["forced"] = True
                        info = dict(d["info"], match="surrogate_forced" if forced else "surrogate", draw="synthesised",
                                    surrogate=rec["surrogate"], group=rec["identity"]["replicate_group"])
                        self._job_draws[job_key] = {"rec": rec, "mode": "replay", "info": info}
                        return rec, "replay", info
                return d["rec"], d["mode"], d["info"]
            rec, info = None, {"match": "none", "draw": None, "group": None}
            want_seed = _seed_from_argv(argv or [])
            forced = self._forced_for(script_path) if script_path else None
            if forced is not None:
                rec = self._surrogate_record(forced, argv, fingerprint_inputs, pending=(stage != "structural"))
                rec["surrogate"]["forced"] = True
                info = {"match": ("surrogate_forced" if stage == "structural" else "surrogate_forced_pending"),
                        "draw": "synthesised", "group": rec["identity"]["replicate_group"],
                        "surrogate": rec["surrogate"], "script_path": script_path}
            if rec is None and entry_hash and argv is not None:
                cands = self._by_exact.get(_exact_key(entry_hash, _strip_argv(argv, None)))
                if not cands:
                    # the record's own exempt list may strip more than the default
                    for key, lst in self._by_exact.items():
                        ex = lst[0].get("identity", {}).get("argv_exempt")
                        if key == _exact_key(entry_hash, _strip_argv(argv, ex)):
                            cands = lst
                            break
                if cands:
                    rec, draw = self._draw(self._task_filter(cands), want_seed)
                    info = {"match": "exact" if rec else "exact_undrawable",
                            "draw": draw,
                            "group": (rec or {}).get("identity", {}).get("replicate_group")}
            if rec is None and info.get("match") == "none" and entry_hash and argv is not None and argv_effective:
                # the literal key missed: a record of the same script whose every
                # explicit flag equals the launch's parsed value (its defaults
                # filled in) is the same computation
                cands = self._effective_candidates(entry_hash, argv, argv_effective, argv_defaults)
                if cands:
                    rec, draw = self._draw(self._task_filter(cands), want_seed)
                    info = {"match": "exact_effective" if rec else "exact_effective_undrawable",
                            "draw": draw,
                            "group": (rec or {}).get("identity", {}).get("replicate_group")}
                    if rec is not None:
                        # the launch flags accepted at the script's own default (the
                        # record never named them): disclosed on the lookup event
                        mine = _flag_groups(_strip_argv(argv, None))
                        dflt = self._effective_defaulted(rec, mine, argv_effective, argv_defaults) if mine is not None else None
                        if dflt:
                            info["defaulted_flags"] = dflt
            if rec is None and info.get("match") == "none" and entry_hash and argv:
                # exact_prefix: `--stop-at N` is the one flag an agent adds to a
                # recorded launch to smoke-test it (G-h1-haiku-L0 `--stop-at 4`,
                # G-h1-sonnet-L0 `--stop-at 125`); the first N steps of the
                # recorded run ARE that launch, and the script stops itself. A
                # segment record that names --stop-at itself matched above.
                stripped, stop_at = _strip_stop_at(argv)
                if stop_at is not None:
                    cands = self._by_exact.get(_exact_key(entry_hash, _strip_argv(stripped, None)))
                    if not cands:
                        for r in self._by_entry.get(entry_hash) or []:
                            ex = r.get("identity", {}).get("argv_exempt")
                            if ex and _exact_key(entry_hash, _strip_argv(r["identity"].get("argv"), ex)) == _exact_key(entry_hash, _strip_argv(stripped, ex)):
                                cands = (cands or []) + [r]
                    cands = [r for r in (cands or []) if not r.get("steps") or int(r["steps"]) >= stop_at]
                    if cands:
                        rec, draw = self._draw(self._task_filter(cands), want_seed)
                        info = {"match": "exact_prefix" if rec else "exact_prefix_undrawable", "draw": draw,
                                "group": (rec or {}).get("identity", {}).get("replicate_group"),
                                "stop_at": stop_at}
            if rec is None and info.get("match") == "none" and modules and argv is not None:
                # exact_module: the launch's entry script is not a recorded one,
                # but a module it imported is (an agent's runner that loads the
                # evaluator with importlib and calls its main(): astracc's shard
                # runner, Fable's debug harness, Opus's score printer, 2026-10-26
                # to 28). Only a per_example script with an example-order table
                # may stand in, since such a launch is served by example content,
                # whatever slice of the workload it runs. argv[0] is the module's
                # own name: the runner set sys.argv to the evaluator's arguments
                # behind an argv[0] of its choosing ("x").
                for m in modules:
                    mh, mp = (m or {}).get("sha256"), (m or {}).get("path")
                    if not mh or not mp or not self.entry_has_order(mh):
                        continue
                    argv_m = [_basename(mp)] + list(argv[1:])
                    cands = self._by_exact.get(_exact_key(mh, _strip_argv(argv_m, None)))
                    how = "exact_module"
                    if not cands:
                        for key, lst in self._by_exact.items():
                            ex = lst[0].get("identity", {}).get("argv_exempt")
                            if ex and key == _exact_key(mh, _strip_argv(argv_m, ex)):
                                cands = lst
                                break
                    if not cands and argv_effective:
                        cands = self._effective_candidates(mh, argv_m, argv_effective, argv_defaults)
                        how = "exact_effective_module"
                    if cands:
                        rec, draw = self._draw(self._task_filter(cands), want_seed)
                        info = {"match": how if rec else how + "_undrawable", "draw": draw,
                                "group": (rec or {}).get("identity", {}).get("replicate_group"),
                                "entry_module": mp, "entry_module_hash": mh}
                        break
            if rec is None and script_path and self._by_bound:
                bp, cands, rung = self._path_exact_candidates(script_path, argv, argv_effective, argv_defaults)
                if cands:
                    rec, draw = self._draw(self._task_filter(cands), want_seed)
                    info = {"match": (("path_exact" if rung == "suffix" else "path_exact_basename") if rec
                                      else "path_exact_undrawable"), "draw": draw,
                            "group": (rec or {}).get("identity", {}).get("replicate_group"),
                            "bound_path": bp, "script_path": script_path}
                elif rung:
                    info = dict(info, path_exact=rung, bound_path=bp, script_path=script_path)
            if rec is None and script_path:
                bp, cands = self.match_path(script_path)
                if cands:
                    rec, draw = self._draw(self._task_filter(cands), want_seed)
                    info = {"match": "path" if rec else "path_undrawable", "draw": draw,
                            "group": (rec or {}).get("identity", {}).get("replicate_group"),
                            "bound_path": bp, "script_path": script_path}
            if rec is None and fingerprint and stage == "structural":
                cands = [r for r in self._by_fp.get(fingerprint, [])
                         if r.get("identity", {}).get("match_policy") == "structural_verified"]
                if cands:
                    rec, draw = self._draw(self._task_filter(cands), want_seed)
                    info = {"match": "structural" if rec else "structural_undrawable",
                            "draw": draw,
                            "group": (rec or {}).get("identity", {}).get("replicate_group")}
            if rec is None and script_path:
                fam = self._surrogate_for(script_path)
                if fam is not None:
                    # no record covers this launch: synthesise, provisionally from
                    # the argv knobs (the fingerprint refines it at the first step)
                    rec = self._surrogate_record(fam, argv, fingerprint_inputs,
                                                 pending=(stage != "structural"))
                    info = {"match": "surrogate" if stage == "structural" else "surrogate_pending",
                            "draw": "synthesised", "group": rec["identity"]["replicate_group"],
                            "surrogate": rec["surrogate"], "script_path": script_path}
            if rec is not None and _per_example(rec) is not None:
                # a deterministic evaluation: a repeated launch of the same
                # method and workload is the same record again, labelled
                # (docs/MLS-KV-SERVING.md §2.1), never a sibling or recycled
                rid = self._rid(rec)
                served = self._served.setdefault(rid, set())
                if served and job_key not in served:
                    # through a loaded module (next/ r4) the launch may be another
                    # part of the workload rather than a rerun: job_bye.positions
                    # says which examples it decoded
                    info = dict(info, draw=("module_repeat" if info.get("entry_module") else "deterministic_rerun"))
                if job_key:
                    served.add(job_key)
                tie = self._tie(rid, job_key)
                if tie:
                    info = dict(info, **tie)
            elif rec is not None and not rec.get("surrogate"):
                rec, draw2, extra = self._sibling_if_rerun(rec, info.get("draw") or "", job_key)
                if extra:
                    info = dict(info, draw=draw2, **extra)
                    info["group"] = (rec or {}).get("identity", {}).get("replicate_group")
                tie = self._tie(self._rid(rec), job_key)
                if tie:
                    info = dict(info, **tie)
            elif rec is not None and rec.get("surrogate"):
                for jid in self._surrogate_jobs.get(rec["surrogate"]["family"], [])[:1]:
                    info = dict(info, task_job=jid)
            if rec is None:
                mode = "model"
            else:
                ready = _drawable(rec)
                mode = rec.get("mode", "replay") if ready else "model"
                if ready and not (rec.get("readiness") or {}).get("replay", True):
                    info = dict(info, readiness_overridden="per_example")
            # Cache a successful draw only: that is what keeps every rank of one
            # job on the same record. Caching a miss pinned a job that missed at
            # interpreter start to that miss, so its structural lookup at the
            # first optimizer step returned the cached None and the job ran
            # unpaced for good.
            if job_key and rec is not None:
                self._job_draws[job_key] = {"rec": rec, "mode": mode, "info": info}
            return rec, mode, info


def _norm_failure(f: dict) -> dict:
    """A recorded failure with `replay` as a plain "fire" | "suppress". The
    registry writes it as a value with provenance ({"value": "suppress", "how":
    ...}). The checks compared that dict with the string "suppress", so a
    suppressed failure fired: sft-seed1 of the battery exited 1 at its last step
    on a Hub-push crash the record suppresses (2026-10-20). A missing value
    means fire, as before."""
    r = f.get("replay")
    v = r.get("value") if isinstance(r, dict) else r
    return dict(f, replay=v if v in ("fire", "suppress") else "fire")


# -- served failures: triggers and the exception they raise -------------------
#
# A record's failure may carry (docs/FAILURE-SERVING.md; analysis 2026-10-23,
# the H2 package: a configured OOM at step k, the GPU freed, a relaunch under
# the limit clean):
#   trigger    {arg, op, value} or a list of them (all must hold): the failure
#              is ARMED for a launch only when the launch's value of `arg`
#              satisfies it. op: eq ne ge gt le lt in. No trigger: always armed.
#   exception  {type, message}: what the job raises (else parsed from cause.value,
#              RECORD-FORMAT's last exception line "Type: message").
#   stderr     the run's real stderr text, printed verbatim in place of the live
#              traceback; a value with provenance or a plain string.
#   exit_code  for crash/killed (default 137 for killed, 1 otherwise).
#   rank       only the job at this launch rank raises (default: every rank).

def _trigger_value(arg, argv: list, eff: Optional[dict]):
    """The launch's value of a trigger's argument: the script's parsed value
    when the patch sent its Namespace (defaults filled in), else the literal
    flag token; None when neither names it."""
    dest = str(arg).lstrip("-").replace("-", "_")
    if eff and dest in eff:
        return eff[dest]
    groups = _flag_groups(_strip_argv(argv, None)) or {}
    flag = str(arg) if str(arg).startswith("-") else "--" + str(arg).replace("_", "-")
    return groups.get(flag)


def _trigger_fires(trigger, argv: list, eff: Optional[dict]) -> Optional[bool]:
    """True: the failure is armed for this launch. False: disarmed. None: the
    trigger names an argument the launch does not resolve (not armed; the
    lookup logs it as unresolved)."""
    if not trigger:
        return True
    conds = trigger if isinstance(trigger, list) else [trigger]
    for c in conds:
        if not isinstance(c, dict) or "arg" not in c:
            return None
        v = _trigger_value(c["arg"], argv, eff)
        if v is None:
            return None
        op = str(c.get("op", "eq")).lower()
        want = c.get("value")
        if op == "in":
            if not any(_same_value(v, w) for w in (want if isinstance(want, list) else [want])):
                return False
            continue
        if op in ("eq", "ne"):
            if _same_value(v, want) != (op == "eq"):
                return False
            continue
        try:
            a, b = float(v), float(want)
        except (TypeError, ValueError):
            return None
        ok = {"ge": a >= b, "gt": a > b, "le": a <= b, "lt": a < b}.get(op)
        if ok is None:
            return None
        if not ok:
            return False
    return True


def _arm_failures(rec: Optional[dict], argv: list, eff: Optional[dict]) -> dict:
    """The record's failures by index: armed / disarmed / unresolved for this launch."""
    out = {"armed": [], "disarmed": [], "unresolved": []}
    for i, f in enumerate((rec or {}).get("failures") or []):
        r = _trigger_fires((f or {}).get("trigger"), argv, eff)
        (out["armed"] if r else out["unresolved"] if r is None else out["disarmed"]).append(i)
    return out


def _failure_exception(f: dict) -> Optional[dict]:
    """What a served oom / xid / nan failure raises in the job: the record's
    explicit `exception` {type, message}, else `cause.value` parsed as
    "Type: message" (RECORD-FORMAT: the last exception line of the run's
    console log). crash and killed end the process without raising, so they
    get None unless the record names an exception explicitly."""
    e = f.get("exception")
    if isinstance(e, dict) and e.get("type"):
        return {"type": str(e["type"]), "message": str(e.get("message") or "")}
    if f.get("mode") not in ("oom", "xid", "nan"):
        return None
    c = f.get("cause")
    c = c.get("value") if isinstance(c, dict) else c
    if not isinstance(c, str) or not c.strip():
        return None
    line = c.strip().splitlines()[-1].strip()
    head, sep, msg = line.partition(":")
    if not re.match(r"^[A-Za-z_][\w.]*$", head.strip()):
        return None
    return {"type": head.strip(), "message": msg.strip() if sep else ""}


# A launch that ends before this many optimizer steps (or served examples)
# gives its record back to the pool: an agent's smoke test (`--stop-at 4`) must
# not cost its real run the record, or the careful behaviour is the one the
# scorer voids as "recycled". The floor
# is also capped at a quarter of the record's length, so a short record keeps
# the rule. Every real stop seen so far is >= 125.
SHORT_RUN_STEPS = int(os.environ.get("NVRT_SHORT_RUN_STEPS", "50"))


def _short_run_floor(rec: Optional[dict]) -> int:
    total = None
    if rec:
        total = rec.get("steps")
    try:
        total = int(total) if total else None
    except (TypeError, ValueError):
        total = None
    return min(SHORT_RUN_STEPS, max(1, total // 4)) if total else SHORT_RUN_STEPS


def _served_full(prefix: dict, parent: dict) -> dict:
    """The parent as served for a launch that drew the prefix: the parent's
    steps, series, timing and memory; the prefix's identity fields that steer
    serving where the parent has none (process_model, bound_path, seed,
    argv_exempt), and `served_for` naming the drawn record."""
    out = dict(parent)
    ident = dict(parent.get("identity") or {})
    pid = prefix.get("identity") or {}
    for k in ("process_model", "bound_path", "seed", "argv_exempt"):
        if pid.get(k) is not None and ident.get(k) is None:
            ident[k] = pid[k]
    ident["served_for"] = pid.get("record_id")
    out["identity"] = ident
    return out


def _norm_bound_path(bp: str) -> str:
    bp = str(bp).replace("\\", "/")
    while bp.startswith("./"):
        bp = bp[2:]
    return bp.lstrip("/")


def _path_under(script_path: str, bp: str) -> bool:
    """Does a script's resolved path fall under a bound path: a file suffix
    (`runs/x/train_gpt.py`) matches the file; a DIRECTORY suffix, written with
    a trailing slash (`setups/muonh-tuned/`), matches any script anywhere
    below that directory — the prep packages, where the agent writes the
    setup's script under a name of its own choosing (G-h1-haiku-L0 named it
    after the README's reference, runner 2026-10-23)."""
    sp = str(script_path).replace("\\", "/")
    if bp.endswith("/"):
        return sp.startswith(bp) or ("/" + bp) in sp
    return sp == bp or sp.endswith("/" + bp)


def _fp_value(x):
    return x.get("value") if isinstance(x, dict) else x


def _derived_from(rec: Optional[dict]) -> Optional[dict]:
    """A compact form of identity.derived_from for the lookup event: a prefix
    record's kind, parent and step range (gpu-tasks' derived records,
    2026-10-20). None for a record that is not derived."""
    d = ((rec or {}).get("identity") or {}).get("derived_from")
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in ("kind", "record", "steps", "stop_after_step") if k in d}


def _is_flag(tok) -> bool:
    """A token that starts a flag group: `-x`, `--k`, `--k=v`; not a negative
    number (`-1`, `-0.5`, `-1e-4`), not a lone `-`."""
    t = str(tok)
    if not t.startswith("-") or t == "-":
        return False
    try:
        float(t)
        return False
    except ValueError:
        return True


def _canonical_argv(argv: list) -> list:
    """The order-insensitive form of a stripped argv, for the exact key only:
    argv[0] and any leading positionals in place, then the flag groups (a flag
    and the values that follow it up to the next flag) sorted by flag name,
    stably, with `--k=v` split into `--k v` so both spellings key alike.
    argparse treats `--stop-at 500 --resume X` and `--resume X --stop-at 500`
    as the same launch; until 2026-10-22 the key did not, and two Astra runs
    sat 658 s each on a `match: none` launch written in the brief's order
   . Positionals after a flag stay in that flag's group,
    which is the one layout argparse would also accept the other way."""
    argv = [str(a) for a in (argv or [])]
    if not argv:
        return argv
    head = [argv[0]]
    i = 1
    while i < len(argv) and not _is_flag(argv[i]):
        head.append(argv[i]); i += 1
    groups = []
    while i < len(argv):
        a = argv[i]; i += 1
        if "=" in a and not a.startswith("--="):
            k, v = a.split("=", 1); g = [k, v]
        else:
            g = [a]
        while i < len(argv) and not _is_flag(argv[i]):
            g.append(argv[i]); i += 1
        groups.append(g)
    groups.sort(key=lambda g: g[0])
    return head + [t for g in groups for t in g]


def _flag_groups(argv: list) -> Optional[dict]:
    """{flag: value} over a stripped argv's flag groups: a bare flag → True, one
    value → that string; a group with several values → None (not handled)."""
    out = {}
    can = _canonical_argv(argv)
    i = 1
    while i < len(can) and not _is_flag(can[i]):
        i += 1                      # leading positionals carry no flag; ignored here
    while i < len(can):
        flag = can[i]; i += 1
        vals = []
        while i < len(can) and not _is_flag(can[i]):
            vals.append(can[i]); i += 1
        if len(vals) > 1:
            return None
        out[flag] = True if not vals else vals[0]
    return out


def _same_value(eff, rec) -> bool:
    """The launch's parsed value against a record's argv token. Path values
    compare by suffix after normpath: `/home/researcher/project/checkpoints/a.pt`
    and `../../checkpoints/a.pt` are the record's `checkpoints/a.pt`."""
    if rec is True:
        return eff is True or str(eff).lower() in ("true", "1")
    if isinstance(eff, bool):
        return str(eff).lower() == str(rec).lower()
    if eff is None:
        return False
    try:
        return abs(float(eff) - float(rec)) <= 1e-12 * max(1.0, abs(float(rec)))
    except (TypeError, ValueError):
        pass
    a, b = str(eff), str(rec)
    if a == b:
        return True
    if "/" in a or "/" in b:
        a2 = os.path.normpath(a); b2 = os.path.normpath(b)
        a2 = a2.lstrip("./") if a2.startswith("./") else a2
        return a2 == b2 or a2.endswith("/" + b2) or b2.endswith("/" + a2)
    return False


def _exact_key(entry_hash: str, argv: list) -> str:
    return entry_hash + "\0" + "\0".join(_canonical_argv(argv))


def _convert(serving: dict, gpu_count: int) -> dict:
    """The shared converter (src/convert/convert.py), imported lazily so the
    daemon runs from any cwd; the reference-schedule solver imports the same
    function, so both produce the same numbers."""
    import importlib.util
    here = Path(__file__).resolve().parent.parent / "convert" / "convert.py"
    spec = importlib.util.spec_from_file_location("gm_convert", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.convert(serving, gpu_count)


def _val(v, default=None):
    """A RECORD-FORMAT value-with-provenance ({value, provenance, how}) or a
    plain number."""
    if isinstance(v, dict) and "value" in v:
        return v["value"] if v["value"] is not None else default
    return v if v is not None else default


def _serving_info(rec: Optional[dict]) -> Optional[dict]:
    """What the replay client needs to pace and verify a job, lifted from the
    record (RECORD-FORMAT §5–§8). Times are recorded, unconverted."""
    if not rec:
        return None
    ident = rec.get("identity", {})
    timing = rec.get("timing", {}) or {}
    steps = rec.get("steps")
    val_series = (rec.get("series") or {}).get("val_loss")
    n_validations = None
    if isinstance(val_series, dict):
        n_validations = len(val_series.get("steps") or [])
    elif isinstance(val_series, list):
        n_validations = len(val_series)
    eval_total = _val(timing.get("eval_s"))
    sur = rec.get("surrogate") or {}
    if sur and not n_validations:
        n_validations = sur.get("n_validations")
    traj = ident.get("trajectory") or {}
    lr = (traj.get("lr_schedule") or {})
    trace = lr.get("trace") if isinstance(lr.get("trace"), dict) else None
    return {
        "record_id": ident.get("record_id"),
        "match_policy": ident.get("match_policy", "exact_only"),
        "argv_exempt": ident.get("argv_exempt") or [],
        "steps": steps,
        "planned_steps": rec.get("planned_steps", steps),
        "per_step_gpu_s": rec.get("per_step_gpu_s"),
        "steady_step_s": _val(timing.get("steady_step_s"), rec.get("per_step_gpu_s")),
        "first_step_s": _val(timing.get("first_step_s")),
        "startup_in_train_s": _val(timing.get("startup_in_train_s"), 0.0),
        "setup_s": _val(timing.get("setup_s"), 0.0),
        "eval_s_total": eval_total,
        "n_validations": n_validations,
        "eval_s_per_validation": (float(eval_total) / n_validations
                                  if eval_total and n_validations else None),
        "pacing": timing.get("pacing"),
        "series_roles": (sorted((rec.get("series") or {}).keys()) if not sur
                         else ["train_loss", "val_loss"]),
        "surrogate": ({"family": sur.get("family"), "config_hash": sur.get("config_hash"),
                       "pending": sur.get("pending")} if sur else None),
        "serving_mode": ((rec.get("materialisation") or {}).get("serving") or {}).get("mode", "per_op"),
        "per_example": _per_example_reply(rec),
        "whole_job": _whole_job_reply(rec),
        # next/ r8.9: the project files the record binds, on every serving path
        # ({path relative to the task root: sha256}); {} = declared none, so the
        # patch's shadow and in-process checks run; absent (None) = not derived,
        # nothing checked (r8 behaviour). Hashes and paths, no result.
        "module_shas": (dict(((rec.get("materialisation") or {}).get("serving") or {}).get("module_shas"))
                        if isinstance(((rec.get("materialisation") or {}).get("serving") or {}).get("module_shas"), dict)
                        else None),
        "wall_total_s": _val(timing.get("wall_total_s")),
        # no final_score and no outcome here: the patch reads neither, and a
        # client that is not a job used the reply as an answer oracle (Grok
        # canary, 2026-10-26: 47 lookups printed serving.final_score for every
        # method and workload without running one); a job's score reaches it
        # only through the score seam, one paced example at a time
        "process_model": _val((rec.get("launch") or {}).get("process_model")) or ((rec.get("materialisation") or {}).get("serving") or {}).get("process_model"),
        "world_size_recorded": _val((rec.get("launch") or {}).get("world_size")),
        "batch_policy": _val((rec.get("launch") or {}).get("batch_policy")),
        "lr_trace": ({"alignment": trace.get("alignment"), "steps": trace.get("steps"),
                      "values": trace.get("values"), "tolerance": trace.get("tolerance"),
                      "per_group": trace.get("per_group")} if trace else None),
        "failures": [_failure_reply(_norm_failure(f)) for f in (rec.get("failures") or [])],
        # the record's device memory: peak bytes (max over GPUs) and, where the
        # source has it, each GPU's peak as a percentage of the card
        "memory": {"peak_bytes": _val((rec.get("memory") or {}).get("peak"),
                                      (rec.get("memory") or {}).get("peak_allocated_bytes")),
                   "per_gpu_peak_pct": (rec.get("memory") or {}).get("per_gpu_peak_pct"),
                   # the model as loaded (weights + context), held from engagement
                   # and stepped up to the peak at the first paced unit
                   "weights_bytes": _val((rec.get("memory") or {}).get("weights_bytes")),
                   # per-GPU bytes of state sharded across the recorded GPUs
                   # (ZeRO-3 / FSDP); convert() scales them by width
                   "sharded_bytes": _val((rec.get("memory") or {}).get("sharded_bytes"))},
    }


def _nearest_series_value(series, step: int):
    """The value at the latest recorded step <= `step`, else at the earliest
    recorded step after it (so a stand-in exists before a record's first
    validation). None only for an empty series."""
    v = _series_value_at(series, step)
    if v is None:
        v = _series_value_at(series, step, pick="covering")
    return v


def _per_example(rec: Optional[dict]) -> Optional[dict]:
    """The record's `materialisation.serving` block when its mode is
    per_example (MLS-Bench KV evaluations, docs/MLS-KV-SERVING.md), else None."""
    ser = ((rec or {}).get("materialisation") or {}).get("serving") or {}
    return ser if ser.get("mode") == "per_example" else None


# The per_example fields the lookup reply carries to the patch: what it needs to
# install the seams and pace (next/ r7.1, 2026-11-02). An ALLOWLIST, so that a
# result field added to a record's serving block later never reaches a client
# unpaced. r6 added `per_example_scores` (the measured first-k scores) to the
# block, and the reply carried it whole, so any client that sent a lookup read
# 50 real scores per KV record without running anything: the answer oracle r2
# closed for final_score (2026-10-27), reopened. Results reach a job only through
# `materialise`, gated on its paced unit.
_PER_EXAMPLE_REPLY_KEYS = ("mode", "generate_seam", "score_seam", "n_examples_full", "score_kind",
                           "subset_source", "process_model")


def _whole_job(rec: Optional[dict]) -> Optional[dict]:
    """The record's `materialisation.serving` block when its mode is whole_job
    (next/ r8): a job that runs its recorded script, paced for its recorded
    duration in units the record defines, and returns its recorded result at a
    seam the record names. Else None."""
    ser = ((rec or {}).get("materialisation") or {}).get("serving") or {}
    return ser if ser.get("mode") == "whole_job" else None


# The whole_job fields the lookup reply carries: the seam and the unit plan the
# patch needs to install and pace. The result itself (`result_key`'s value, the
# record's `result.*`) never travels in the reply: the patch fetches it through
# `materialise` role whole_job_result, which the daemon serves only once every
# unit has been paced and its GPU time has passed.
_WHOLE_JOB_REPLY_KEYS = ("mode", "seam", "unit", "n_units", "result_kind", "result_shape", "process_model",
                         # live_fields (next/ r8.3): {result_dict_key: "<allowlisted live getter>"}. After
                         # the units pace, the patch reads the named LIVE instrument (e.g.
                         # torch.cuda.max_memory_allocated) IN the seamed context and merges it into the
                         # served dict -- but ONLY for a key the recorded result does not carry, so a
                         # recorded field is never overridden by a live one. For a source whose terminal
                         # print includes a live stat (DLM peak_memory_mb, a paced-seam elapsed) that the
                         # CSV never recorded: the quality is recorded, the live stat is the instrument's
                         # own reading, nothing is invented.
                         "live_fields")


def _whole_job_reply(rec: Optional[dict]) -> Optional[dict]:
    wj = _whole_job(rec)
    if wj is None:
        return None
    out = {k: wj[k] for k in _WHOLE_JOB_REPLY_KEYS if k in wj}
    # The identity a whole_job record was recorded under (next/ r8.5). The patch
    # checks it in the job's own interpreter, which the daemon (a sidecar with
    # its own python) cannot see: the trajectory env the job runs with, the
    # package versions it has installed, and the bytes of the seam module it
    # loaded. Mismatch on any -> no served aggregate.
    ident = (rec or {}).get("identity") or {}
    out["identity"] = {
        "env": ident.get("env") or {},
        "env_exempt": list(ident.get("env_exempt") or []),
        "environment": (ident.get("trajectory") or {}).get("environment") or {},
        "seam_sha256": wj.get("seam_sha256"),
        # next/ r8.5.1: the project modules the seam module imports (the method
        # file an agent edits may be one of them, e.g. sparse's custom_sparse_attn)
        "module_shas": dict(wj.get("module_shas") or {}),
    }
    return out


def _whole_job_units(wj: dict) -> int:
    try:
        n = int(_val(wj.get("n_units")) or 1)
    except (TypeError, ValueError):
        n = 1
    return max(1, n)


def _whole_job_unit_cost_floor(rec: Optional[dict]) -> float:
    """The least a whole_job unit may cost (recorded seconds) and still count
    toward the result: the record's wall over its units, with a 0.1% float
    tolerance. The patch prices each unit at exactly that (or its recorded
    share, see `unit_shares`), so a job that runs its script always meets it."""
    wj = _whole_job(rec) or {}
    wall = _val(((rec or {}).get("timing") or {}).get("wall_total_s"))
    try:
        wall = float(wall)
    except (TypeError, ValueError):
        return 0.0
    shares = wj.get("unit_shares")
    n = _whole_job_units(wj)
    if isinstance(shares, list) and len(shares) == n and all(isinstance(x, (int, float)) for x in shares):
        return max(0.0, min(float(x) for x in shares) * wall * 0.999)
    return max(0.0, wall / n * 0.999)


# The fields of a recorded failure the lookup reply carries: what the patch reads
# to decide whether to poll `fail?` and at which seam (_live_failure_wheres). The
# failure's text (cause, exception, stderr) and its step reach a job only through
# `fail?`, when the job has paced to that step (next/ r7.1).
_FAILURE_REPLY_KEYS = ("mode", "where", "replay", "at")


def _failure_reply(f: dict) -> dict:
    """A recorded failure as the lookup reply carries it."""
    return {k: f[k] for k in _FAILURE_REPLY_KEYS if k in f}


def _per_example_reply(rec: Optional[dict]) -> Optional[dict]:
    """The record's per_example block as the lookup reply carries it: the
    allowlisted fields only."""
    pe = _per_example(rec)
    if pe is None:
        return None
    return {k: pe[k] for k in _PER_EXAMPLE_REPLY_KEYS if k in pe}


def _measured_scores(rec: Optional[dict]) -> Optional[dict]:
    """The record's measured per-example scores (next/ r6): the real scores of
    its first k examples in the example-order table's order, decoded with the
    record's own evaluator (gpu-tasks' `materialisation.serving.per_example_scores`
    block). None when the record has none, or the block is not usable."""
    pe = _per_example(rec) or {}
    b = pe.get("per_example_scores")
    if not isinstance(b, dict):
        return None
    sc = b.get("scores")
    k = b.get("k")
    if not isinstance(sc, list) or not sc or k is None or int(k) != len(sc):
        return None
    try:
        return {"k": int(k), "scores": [float(x) for x in sc], "keys": list(b.get("keys") or []),
                "example_ids": list(b.get("example_ids") or [])}
    except (TypeError, ValueError):
        return None


def _synth(n: int, mean: float, ser: dict, rng) -> list:
    """A synthesised per-example vector of length n, on the serving block's
    scale, with mean exactly `mean` (see _score_vector)."""
    lo, hi = (ser.get("scale") or [0, 100])[:2]
    span = float(hi) - float(lo)
    kind = ser.get("score_kind") or "binary"
    if kind in ("binary", "retrieval"):
        # hits in example units; a binary record's is an integer (that pins n),
        # a retrieval record's can be fractional, because retrieval_score is
        # hits / numbers found when an answer holds several numbers (passage
        # retrieval sums like 183.25 or 169.9 of 300; gpu-tasks 2026-10-22
        # 23:23Z, every passage score was off by up to 0.08 points when this
        # rounded). The whole part is served as hits, the remainder as one
        # example at the fractional value, so the mean is the recorded
        # final_score exactly. A remainder under 1e-3 is the six-decimal
        # rounding of the recorded mean, not a fraction.
        total = (mean - float(lo)) / span * n
        hits = int(math.floor(total + 1e-3))
        rem = total - hits
        vec = [float(hi)] * hits + [float(lo)] * (n - hits)
        if rem > 1e-3 and hits < n:
            vec[hits] = float(lo) + span * rem
        rng.shuffle(vec)
        return vec
    m = min(max((mean - float(lo)) / span, 1e-6), 1 - 1e-6)
    conc = 2.0
    vec = [float(lo) + span * rng.betavariate(m * conc, (1 - m) * conc) for _ in range(n)]
    for _ in range(8):     # pin the mean exactly, keeping every value in range
        resid = mean * n - sum(vec)
        if abs(resid) < 1e-9:
            break
        free = [i for i, v in enumerate(vec) if (float(lo) < v < float(hi))]
        if not free:
            break
        d = resid / len(free)
        for i in free:
            vec[i] = min(float(hi), max(float(lo), vec[i] + d))
    return vec




def _score_vector(rec: dict) -> list:
    """The per-example score vector of a per_example record: length
    n_examples_full, on the script's scale (0–100), mean exactly the recorded
    final_score, seeded by the record id so every launch and every daemon
    sees the same vector. binary / retrieval: a 0/100 vector with the recorded
    hit count in a seeded order; continuous: Beta samples with the recorded
    mean and concentration 2, rescaled to the exact mean (a synthesised
    shape, disclosed as subset_model on every served value).

    When the record carries measured scores (next/ r6: the real scores of its
    first k examples, decoded with its own evaluator), positions 0..k-1 are
    those measured scores and only the tail k..n-1 is synthesised, to the mean
    that keeps the whole vector's mean at the recorded final_score. If the
    measured head makes that tail mean fall outside the scale, the tail is
    clamped to the scale and the full vector's mean is then off by the excess,
    logged once (_score_vector_note)."""
    import hashlib
    import random
    ser = _per_example(rec) or {}
    n = int(_val(ser.get("n_examples_full")) or 0)
    mean = _val(((rec.get("result") or {}).get("final_score")))
    if n <= 0 or mean is None:
        return []
    mean = float(mean)
    rng = random.Random(int(hashlib.sha256(str(rec.get("identity", {}).get("record_id")).encode()).hexdigest()[:16], 16))
    meas = _measured_scores(rec)
    if not meas:
        return _synth(n, mean, ser, rng)
    k = min(meas["k"], n)
    head = meas["scores"][:k]
    if k >= n:
        return head
    lo, hi = (ser.get("scale") or [0, 100])[:2]
    tail_n = n - k
    tail_mean = (mean * n - sum(head)) / tail_n
    clamped = min(float(hi), max(float(lo), tail_mean))
    if abs(clamped - tail_mean) > 1e-9:
        _score_vector_note[str((rec.get("identity") or {}).get("record_id"))] = {
            "tail_mean_wanted": tail_mean, "tail_mean_served": clamped,
            "full_mean_served": (sum(head) + clamped * tail_n) / n, "final_score": mean}
    return head + _synth(tail_n, clamped, ser, rng)


# record id -> the r6 tail that could not keep the recorded mean (measured head outside what the tail can balance)
_score_vector_note: dict = {}


# the most start-up a launch may claim against the record's wall when its
# example units are priced, as a fraction of the wall: a client that claims more
# saves at most this share of the evaluation's time; a real launch's start-up is
# far below it (measured across collected runs: median 1.2 s, p99 3.7 s, max
# 5.0 s against walls of 323–29,153 s, 2026-10-27), so it is never priced short
MAX_CLAIMED_STARTUP_FRAC = 0.5


def _example_cost_floor(rec: Optional[dict], startup_s) -> float:
    """The least a per_example unit may cost (recorded seconds) and still open
    its example's score: the record's (wall − start-up) / n_examples_full, the
    start-up being what the launch's first unit claims, capped. The patch
    prices every unit at exactly (wall − its measured start-up) / n_full, so it
    is always at or above the floor; a 0.1% tolerance covers float rounding."""
    pe = _per_example(rec) if rec else None
    if not pe:
        return 0.0
    n = _val(pe.get("n_examples_full"))
    wall = _val(((rec or {}).get("timing") or {}).get("wall_total_s"))
    try:
        n = int(n); wall = float(wall)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0 or wall <= 0:
        return 0.0
    try:
        claim = float(startup_s) if startup_s is not None else 0.0
    except (TypeError, ValueError):
        claim = 0.0
    claim = min(max(0.0, claim), MAX_CLAIMED_STARTUP_FRAC * wall)
    return max(0.0, wall - claim) / n * 0.999


def _positions_summary(pos: list) -> Optional[dict]:
    """The record positions a per_example job opened: how many, the first
    and last, whether they are one contiguous run, and whether they are the
    workload's prefix (a whole launch or a --max-examples k launch)."""
    if not pos:
        return None
    srt = sorted(pos)
    return {"n": len(pos), "first": srt[0], "last": srt[-1],
            "contiguous": srt == list(range(srt[0], srt[0] + len(srt))),
            "prefix": pos == list(range(len(pos)))}


PER_EXAMPLE_BLOCKER = "gpu-mock per_example serving"
# a whole_job record whose only blocker is this serving mode is drawable once a
# daemon has it (next/ r8); a record with another blocker (a box image not yet
# built, an adapter-specific seam) is not
WHOLE_JOB_BLOCKER = "gpu-mock whole_job serving"

# gpu-tasks' example-order tables (registry/records/example-order-*.json,
# agreed 2026-10-28): order[i] is the key of the record's example i
EXAMPLE_ORDER_SCHEMA = "p006-example-order/0.1"
EXAMPLE_ORDER_KEY = "sha256(example_id\\0prompt)"


def _drawable(rec: dict) -> bool:
    """readiness.replay, except that a per_example record whose only blocker
    is the per_example serving itself is drawable by a daemon that has it
    (gpu-tasks set `readiness.missing` to name this build, 2026-10-22, so the
    capability flips readiness instead of a hand edit of 40 records)."""
    rd = rec.get("readiness") or {}
    if rd.get("replay", True):
        return True
    missing = rd.get("missing") or []
    if bool(_whole_job(rec)) and bool(missing) and all(str(m).startswith(WHOLE_JOB_BLOCKER) for m in missing):
        return True       # the same rule for whole_job records (next/ r8)
    return bool(_per_example(rec)) and bool(missing) and all(str(m).startswith(PER_EXAMPLE_BLOCKER) for m in missing)


def _subset_of(ser: dict, argv: Optional[list], subset_env: Optional[dict]) -> Optional[dict]:
    """A launch's --max-examples k (the flag wins, else the env the serving
    block names; 0 means the full workload): {k, n_full} or None."""
    src = ser.get("subset_source") or {}
    flag = src.get("flag") or "--max-examples"
    k = None
    argv = [str(a) for a in (argv or [])]
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            k = argv[i + 1]
        elif a.startswith(flag + "="):
            k = a.split("=", 1)[1]
    if k is None and src.get("env"):
        k = (subset_env or {}).get(src["env"])
    try:
        k = int(k) if k is not None else 0
    except (TypeError, ValueError):
        k = 0
    n = int(_val(ser.get("n_examples_full")) or 0)
    if k <= 0 or (n and k >= n):
        return None
    return {"k": k, "n_full": n}


def _series_points(series) -> list:
    """(step, value) pairs of a series in either form, numeric values only."""
    out = []
    if isinstance(series, list):
        out = [(i, v) for i, v in enumerate(series)]
    elif isinstance(series, dict):
        out = list(zip(series.get("steps") or [], series.get("values") or []))
    pts = []
    for st, v in out:
        try:
            st, v = int(st), float(v)
        except (TypeError, ValueError):
            continue
        if v == v:
            pts.append((st, v))
    return pts


def _extrapolate(series, step: int, seed_key: str) -> Optional[float]:
    """A labelled continuation of a series past its last recorded step
    (2026-10-22): a single trial that outlasts its record (the agent lengthened
    the schedule) used to be served the last value for every later read, a
    frozen tail an agent can tell. The continuation fits the last six points
    as a + b·ln(step), never rising (b clipped at 0), and adds a deterministic
    Gaussian at the fit's residual spread (floor 0.0005, 0.002 with under
    three points), seeded by record and step so a re-read is the same value.
    The event that serves it says `tier: extrapolated`; it is a guess, labelled
    as one, never a recorded number."""
    import math
    import random
    pts = _series_points(series)
    pts = [(st, v) for st, v in pts if st > 0]
    if not pts:
        return None
    tail = pts[-6:]
    xs = [math.log(st) for st, _ in tail]
    ys = [v for _, v in tail]
    n = len(tail)
    if n >= 3:
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx) if sxx > 0 else 0.0
        b = min(b, 0.0)
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = max(0.0005, (sum(r * r for r in resid) / max(1, n - 2)) ** 0.5)
    else:
        a, b, sd = ys[-1], 0.0, 0.002
        a = ys[-1] - b * xs[-1]
    pred = a + b * math.log(max(1, int(step)))
    pred = min(pred, ys[-1] + 3 * sd)          # never a jump above the recorded tail
    rng = random.Random(f"{seed_key}:{int(step)}")
    return pred + rng.gauss(0.0, sd)


# Eval metrics: roles whose value before the series' first recorded step must
# NOT be back-filled from a future point or from result.* (next/ r8, gpu-mock
# 2026-11-02). An eval accuracy or validation loss read before the first
# validation has no honest value: a run that stops at the first eval passing a
# threshold must not see its final or future score early (MLPerf LoRA stops at
# val_loss < 0.925 around step 192). A reported metric, unlike a NaN guard's
# train-loss read, gets None, so the script's own "not evaluated yet" path runs.
EVAL_METRIC_ROLES = ("val_loss", "eval_loss", "eval_accuracy", "accuracy", "val_accuracy",
                     "main_score", "ndcg_at_10", "map_at_10")


def _is_eval_metric_role(role: Optional[str]) -> bool:
    r = str(role or "")
    return r in EVAL_METRIC_ROLES or r.startswith("eval.") or r.startswith("eval_") \
        or r.endswith("_accuracy") or r.endswith("_score")


def _before_first_step(series, step: int) -> bool:
    """True when `step` precedes a sparse series' first recorded step (so there
    is no past value). A plain list has a value at every step from 0, so never."""
    if isinstance(series, dict):
        steps = series.get("steps") or []
        return bool(steps) and int(step) < int(steps[0])
    return False


def _series_value_at(series, step: int, pick: str = "at_or_before"):
    """Read a series value at an optimizer step. Handles two forms:
    - a plain list: values at steps 0..n-1 (shorthand).
    - the object form {"steps":[...],"values":[...]} (sparse, RECORD-FORMAT §8):
      pick "at_or_before" (default): the value at the largest recorded step
      <= `step`, None if `step` precedes the first recorded step;
      pick "covering": the value at the smallest recorded step >= `step`, the
      log that covers this step, else the last value. The Trainer seam serves
      this way, so every step of a logging interval carries the logged value and
      the interval mean the Trainer prints equals the record."""
    if series is None:
        return None
    if isinstance(series, list):
        if not series:
            return None
        return series[min(step, len(series) - 1)]
    if isinstance(series, dict):
        steps = series.get("steps") or []
        values = series.get("values") or []
        if not steps or not values:
            return None
        if pick == "covering":
            for st, v in zip(steps, values):
                if st >= step:
                    return v
            return values[-1]
        # largest recorded step <= step
        chosen = None
        for s, v in zip(steps, values):
            if s <= step:
                chosen = v
            else:
                break
        return chosen
    return None


# ---------------------------------------------------------------------------
# Job: one connected process. Holds its record, its per-GPU sequence counters,
# and a cursor into each served metric series.
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: int
    pid: int
    argv: list
    cwd: str
    record: Optional[dict] = None
    mode: str = "model"
    fingerprint: Optional[str] = None
    record_id: Optional[str] = None
    seq: int = 0
    step: int = 0
    series_cursor: dict[str, int] = field(default_factory=dict)
    observer: bool = False   # the bridge: logged as observer_hello/bye, never a job
    t_start: Optional[float] = None   # the process's own start, wall time
    fired: set = field(default_factory=set)   # indices of this job's record failures that fired
    armed: Optional[set] = None                 # indices of the record's failures whose trigger holds for THIS launch (None: no triggers, all armed)
    job_key: Optional[str] = None               # the launch's job key (its label), from the lookup
    drawn: Optional[dict] = None                # the record the draw picked when `record` is its full-schedule parent (prefix upgrade)
    draw: Optional[str] = None                  # how its record was drawn (exact_seed | seed_substituted | without_replacement | recycled | ...)
    rank: Optional[int] = None                  # the launch rank the lookup named (a failure may carry `rank`: only that rank raises)
    fingerprint_differs: Optional[bool] = None   # set by the structural check
    lr_history: list = field(default_factory=list)   # [[step, [lr per param group]], ...] from the patch's enqueues
    lr_last: Optional[list] = None
    diverged_logged: bool = False               # a surrogate's first NaN was logged as a failure
    trial: int = 1                              # in-process trial (a script that loops trials: new model, new optimizer, steps from 0)
    exhausted_logged: bool = False              # record_exhausted logged for this trial
    subset: Optional[dict] = None               # per_example launch with --max-examples k: {k, n_full}
    startup_s: Optional[float] = None           # per_example launch's start-up on the box's clock (sent with its first example)
    startup_real_s: Optional[float] = None      # the same in real seconds
    overrun_logged: bool = False                # per_example_overrun logged (a read at or past n_full)
    example_ends: dict = field(default_factory=dict)   # per_example: example index -> monotonic time its paced unit's GPU work ends
    pe_units: int = 0                           # per_example: examples paced in order that opened their score (the gate's position)
    pe_gpu: Optional[int] = None                # per_example: the GPU the job's examples are paced on
    pe_last_end: Optional[float] = None         # per_example: when the job's previous example unit ended (clamps the host gap)
    pe_order: Optional[dict] = None             # per_example: the record's example-order table, if gpu-tasks staged one
    pe_positions: list = field(default_factory=list)   # per_example: record positions opened, in order
    pe_order_misses: int = 0                    # per_example: example keys the table did not hold (served by local index)
    entry_module: Optional[str] = None          # the imported module that stood in for the entry script
    read_ahead_logged: int = 0                  # read_ahead refusals logged for this job (capped)
    wj_units: int = 0                           # whole_job: units paced in order at or above the floor
    wj_last_end: Optional[float] = None         # whole_job: when the job's last counted unit's GPU work ends
    wj_served: bool = False                     # whole_job: the recorded result was served


# ---------------------------------------------------------------------------
# The daemon.
# ---------------------------------------------------------------------------


class GPUDaemon:
    def __init__(self, socket_path: Path, registry: Registry,
                 gpus: int, capacity_bytes: int, speed: float,
                 occupancy_path: Path):
        self.socket_path = socket_path
        self._score_vectors: dict[str, list] = {}       # per_example records: record id -> score vector
        self.occupancy_path = Path(occupancy_path)
        self.registry = registry
        self.speed = max(1e-6, speed)
        self.gpus = [VirtualGPU(i, capacity_bytes) for i in range(gpus)]
        self.occ = OccupancyLog(occupancy_path, speed)
        self.jobs: dict[int, Job] = {}
        self._next_job_id = 1
        self._lock = threading.Lock()
        self._server: Optional[socket.socket] = None

    # -- job lifecycle ----------------------------------------------------

    def _new_job(self, pid: int, argv: list, cwd: str, observer: bool = False,
                 t_start: Optional[float] = None) -> Job:
        """A connection that said hello. A job is a process that will enqueue
        GPU work; an observer (the overrides bridge) only reads snapshots and is
        logged as observer_hello/observer_bye, so a run's occupancy log begins
        with its first real job and the scorer's makespan is not fooled.
        t_start is the process's own start time (wall), sent by the client, so
        the log keeps when the job began even though it says hello only at its
        first CUDA use."""
        with self._lock:
            jid = self._next_job_id
            self._next_job_id += 1
            job = Job(job_id=jid, pid=pid, argv=argv, cwd=cwd, observer=observer,
                      t_start=t_start)
            self.jobs[jid] = job
        kind = "observer_hello" if observer else "job_hello"
        self.occ.event(kind, gpu=-1, job=jid, pid=pid, argv=argv, t_start=t_start)
        return job

    def _write_lr_stream(self, job: Job) -> None:
        """One line per job and trial next to the occupancy log: the per-step
        learning rates of every param group (the analysis, gpu-tasks' fits)."""
        if not job.lr_history:
            return
        try:
            with open(self.occupancy_path.parent / "lr_stream.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job": job.job_id, "pid": job.pid, "record_id": job.record_id, "trial": job.trial,
                                     "groups": len(job.lr_history[0][1]), "steps": len(job.lr_history),
                                     "lr": job.lr_history}) + "\n")
        except OSError as e:
            log(f"lr_stream not written for job {job.job_id}: {e!r}")

    def _begin_trial(self, job: Job, trial: int) -> None:
        """A script that loops trials in one process (modded-nanogpt's
        `num_trials = int(sys.argv[-1])`, 2026-10-22: newton-muon on c3-12
        showed a dead replay from trial 2 on): the patch sends a new lookup
        under `job_key#t<n>` when a new optimizer over new parameters appears
        after steps. The job's per-trial state restarts; the record is replaced
        by the dispatch that called this; the finished trial's LR stream is
        written first."""
        self._write_lr_stream(job)
        job.lr_history, job.lr_last = [], None
        job.step = 0
        job.fired = set()
        job.diverged_logged = False
        job.exhausted_logged = False
        job.fingerprint_differs = None
        job.trial = int(trial)

    def _end_job(self, job: Job, reason: str = "bye") -> None:
        # a launch that did not run its record to the end gives the record back:
        # a smoke test (ended before the short-run floor) and, from 2026-10-24,
        # any incomplete run — a job the agent scancelled at step 384 and
        # resubmitted under the same seed was served another seed's record
        #; on hardware the same seed shows the
        # same curve again, and the "never twice" rule guards concurrent
        # launches, not a run that is over. A completed run keeps its draw.
        returned = None
        incomplete = None
        if job.record is not None and job.mode == "replay" and not job.observer \
                and _whole_job(job.record) is not None \
                and job.draw in ("exact_seed", "seed_substituted", "without_replacement"):
            # whole_job (next/ r8): a job that never paced its last unit did not
            # finish the run, so the next launch of the configuration is a first
            # draw, as a short run of a step-paced record is
            if job.wj_units < _whole_job_units(_whole_job(job.record)):
                incomplete = True
                self.registry.return_draw(job.drawn or job.record, job.job_key)
        elif job.record is not None and job.mode == "replay" and not job.observer \
                and not _per_example(job.record) \
                and job.draw in ("exact_seed", "seed_substituted", "without_replacement"):
            step = int(job.step or 0)
            total = job.record.get("steps")
            try:
                total = int(total) if total else None
            except (TypeError, ValueError):
                total = None
            # completeness is measured against what was DRAWN as well as what
            # was served: a prefix launch upgraded to its parent whose script
            # stops at the prefix's own last step is complete, not cut short
            # (gpu-tasks 2026-10-24 14:01Z: measured against the parent, the
            # prefix went back and a seedless pool served one record twice)
            drawn_total = None
            if job.drawn is not None and job.drawn is not job.record:
                try:
                    drawn_total = int(job.drawn.get("steps") or 0) or None
                except (TypeError, ValueError):
                    drawn_total = None
            # exactly the drawn record's last step (one past it tolerated): a
            # kill between the prefix's stop and the parent's end still returns
            complete_at_drawn = drawn_total is not None and drawn_total <= step <= drawn_total + 1
            if step < _short_run_floor(job.record):
                returned = True
            elif total is not None and step < total and not complete_at_drawn:
                incomplete = True
            if returned or incomplete:
                self.registry.return_draw(job.drawn or job.record, job.job_key)
        for g in self.gpus:
            with g.lock:
                if job.job_id in g.mem:
                    del g.mem[job.job_id]
                g.active_jobs.discard(job.job_id)
                self._recompute_idle(g)
        self.jobs.pop(job.job_id, None)
        self._write_lr_stream(job)
        self.occ.event("observer_bye" if job.observer else "job_bye", gpu=-1,
                       job=job.job_id, pid=job.pid,
                       # None until a structural check ran; True when the live
                       # fingerprint disagreed with the record's (the scorer's flag)
                       fingerprint_differs=getattr(job, "fingerprint_differs", None),
                       surrogate=bool((job.record or {}).get("surrogate")),
                       lr_steps=len(job.lr_history) or None,
                       # how the connection ended: bye | eof | error:<type>
                       reason=reason,
                       trials=(job.trial if job.trial > 1 else None),
                       served=("result" if _per_example(job.record) else
                               ("result" if job.wj_served else "none") if _whole_job(job.record) else None),
                       whole_job_units=((job.wj_units, _whole_job_units(_whole_job(job.record)))
                                        if _whole_job(job.record) else None),
                       subset=job.subset, startup_s=job.startup_s, startup_real_s=job.startup_real_s,
                       # which of the record's examples the job decoded: a shard or a
                       # hand-picked list shows here (the served mean is the record's
                       # only when n equals the record's n_examples_full)
                       positions=(_positions_summary(job.pe_positions) if _per_example(job.record) else None),
                       example_order_misses=(job.pe_order_misses or None),
                       entry_module=job.entry_module,
                       # the last optimizer step the job paced (its enqueue units carry
                       # the step): how far it got, whatever the record or family planned
                       last_step=job.step or None,
                       # a smoke test: ended before the short-run floor, its record returned
                       draw_returned_short_run=returned,
                       short_run_floor=(_short_run_floor(job.record) if returned else None),
                       # ended after the floor but before the record's last step: the
                       # record goes back too, so a resubmit of the seed shows the same curve
                       draw_returned_incomplete=incomplete)

    # -- pacing -----------------------------------------------------------

    def _sharers(self, g: VirtualGPU) -> int:
        return max(1, len(g.active_jobs))

    def _enqueue(self, job: Job, gpu_index: int, op: str,
                 modelled_s: float, mem_delta: int,
                 step: Optional[int] = None, unit: Optional[str] = None,
                 lr: Optional[list] = None, startup_s: Optional[float] = None,
                 startup_real_s: Optional[float] = None, host_gap_s: Optional[float] = None,
                 oom_aware: bool = False, key: Optional[str] = None) -> dict:
        """Push one work unit onto a GPU's queue. Returns the new cursor.

        `key` (per_example units): the example's content key, sha256 of its
        example_id, a NUL and its prompt. With the record's example-order
        table it names the record position the unit decodes, so a launch that
        runs a slice of the workload (a shard, a hand-picked list) opens the
        record's own examples, whichever they are (_example_position).

        `host_gap_s` (per_example units): the real seconds the script spent on
        the host since its previous unit drained (tokenising, scoring). The
        record's wall already holds the recorded node's own host time once,
        so the unit is back-dated by that gap, clamped to the GPU's cursor:
        the real time per example becomes max(slot, gap) instead of slot +
        gap, a speed-1 launch takes and prints the wall, and the virtual
        clock, which advances by the recorded share, is unchanged (the runner,
        2026-10-22 23:54Z: a speed-1 measure was double counting it).

        Contention comes from the single shared cursor: a GPU does one thing at a
        time, so two jobs' work serialises on the same cursor and each job's own
        completion slips behind the other's queued work. That already gives the
        proportional-share 1/N slowdown without any extra multiplier — the GPU's
        total throughput is fixed and the two jobs divide it. Multiplying the
        per-unit cost by the sharer count on top of serialising would double-count
        the contention, so we do not. `sharers` is logged for the occupancy record
        but does not scale the cost.
        """
        g = self.gpus[gpu_index]
        wall_cost = max(0.0, modelled_s) / self.speed
        opens = None
        if unit == "example":
            # A unit opens its example's score only if it is the job's next
            # example, on the job's one GPU, and costs at least the record's own
            # per-example share; and its back-date is clamped to the job's real
            # gap since its previous example ended (none for the first). The
            # patch's seams do exactly this, so a job that runs its script is
            # unaffected; a client that labels, prices or back-dates its own
            # units opens nothing (next/ r2b, 2026-10-27).
            if host_gap_s is not None:
                if job.pe_last_end is None:
                    host_gap_s = 0.0
                else:
                    host_gap_s = min(max(0.0, float(host_gap_s)), max(0.0, now() - job.pe_last_end))
            # the start-up the launch claimed with its first unit (only that
            # unit carries it; job.startup_s keeps it for the rest)
            floor = _example_cost_floor(job.record, startup_s if startup_s is not None else job.startup_s)
            seq = step is not None and int(step) == job.pe_units + 1
            same_gpu = job.pe_gpu is None or job.pe_gpu == gpu_index
            if seq and same_gpu and modelled_s >= floor:
                opens = self._example_position(job, key, int(step) - 1)
        wj_counts = False
        if unit == "whole_job":
            # a unit counts toward the result only in order, on the job's one GPU,
            # at or above the record's own per-unit share: the patch's seam does
            # exactly this, so a job that runs its script always counts every unit
            seq = step is not None and int(step) == job.wj_units + 1
            same_gpu = job.pe_gpu is None or job.pe_gpu == gpu_index
            wj_counts = seq and same_gpu and modelled_s >= _whole_job_unit_cost_floor(job.record)
        with g.lock:
            was_idle = not g.is_busy()
            g.active_jobs.add(job.job_id)
            sharers = self._sharers(g)
            base = max(g.cursor, now() - max(0.0, float(host_gap_s or 0.0)))
            if was_idle:
                self.occ.event("busy_start", gpu=gpu_index, job=job.job_id,
                               sharers=sharers)
            g.cursor = base + wall_cost
            # memory
            if mem_delta:
                g.mem[job.job_id] = max(0, g.mem.get(job.job_id, 0) + mem_delta)
            cursor = g.cursor
            used = g.used_bytes()
            free = g.free_bytes()
            # oversubscription (docs/FAILURE-SERVING.md): the job whose arrival
            # pushes the GPU's held memory past the card is the one that OOMs,
            # as the second process to allocate does on hardware. The detail
            # names the other holders by pid so the message reads as the real
            # one does; an oom-aware patch raises and the job frees on exit.
            # Two jobs' peaks that fit together run together, as they would.
            oom_detail = self._oom_detail_locked(g, job, gpu_index) if (mem_delta > 0 and used > g.capacity_bytes) else None
        job.seq += 1
        if wj_counts:
            job.wj_units += 1
            job.wj_last_end = cursor
            if job.pe_gpu is None:
                job.pe_gpu = gpu_index
        if unit == "example":
            job.pe_last_end = cursor
            if opens is not None:
                # record position `opens` is served its score once this unit's
                # GPU time has passed (_materialise_example_score)
                job.example_ends[opens] = cursor
                job.pe_positions.append(opens)
                job.pe_units += 1
                if job.pe_gpu is None:
                    job.pe_gpu = gpu_index
        # the learning rate every param group carried at this optimizer step
        # (gpu-tasks, 2026-10-21: the schedule the agent's code chose, read from
        # the run, drives the surrogate's cumulative-LR law); kept per step on
        # the job, logged on the event only when it changed
        if step is not None and int(step) > job.step:
            job.step = int(step)        # how far the job got, for job_bye.last_step
        lr_changed = None
        if lr is not None and step is not None:
            try:
                vals = [float(x) for x in lr]
            except (TypeError, ValueError):
                vals = None
            if vals is not None:
                job.lr_history.append([int(step), vals])
                if job.lr_last != vals:
                    lr_changed = vals
                    job.lr_last = vals
        if startup_s is not None:
            # a per_example launch's start-up (box clock, and real), stated once by
            # the patch with the first example; repeated on every served score
            # and on job_bye
            job.startup_s = float(startup_s)
        if startup_real_s is not None:
            job.startup_real_s = float(startup_real_s)
        self.occ.event("enqueue", gpu=gpu_index, job=job.job_id, op=op,
                       seq=job.seq, modelled_s=modelled_s, wall_cost=wall_cost,
                       sharers=sharers, used_bytes=used, free_bytes=free,
                       step=step, unit=unit, lr=lr_changed,
                       host_gap_s=(round(float(host_gap_s), 4) if host_gap_s is not None else None),
                       startup_s=(float(startup_s) if startup_s is not None else None),
                       startup_real_s=(float(startup_real_s) if startup_real_s is not None else None),
                       trial=(job.trial if job.trial > 1 else None))
        # OOM check: if this job just pushed total allocation over capacity.
        oom = used > g.capacity_bytes
        if oom_detail is not None and oom_aware:
            self._log_oversubscribed(job, gpu_index, step, used, oom_detail)
        # wall_cost lets the client pace with lookahead: block only until the
        # PREVIOUS work drains (cursor - wall_cost), so host overhead overlaps
        # this unit's GPU time as on a real async GPU.
        return {"cursor": cursor, "seq": job.seq, "used_bytes": used,
                "free_bytes": free, "oom": oom, "wall_cost": wall_cost,
                "oom_detail": oom_detail}

    def _example_position(self, job: Job, key: Optional[str], local: int) -> int:
        """The record position a job's `local`-th example unit decodes: the
        position of its content key in the record's example-order table, else
        `local` itself, which is what a launch that runs the workload from its
        first example (the whole of it, or --max-examples k) decodes. A key the
        table does not hold (an edited prompt, a different dataset) falls back
        to `local`, counted on job_bye and logged three times a job."""
        idx = (job.pe_order or {}).get("index")
        if idx is None or not key:
            return local
        p = idx.get(key)
        if p is not None:
            return p
        job.pe_order_misses += 1
        if job.pe_order_misses <= 3:
            self.occ.event("example_order_miss", gpu=-1, job=job.job_id, step=local + 1,
                           record_id=job.record_id, table=job.pe_order.get("file"),
                           detail=f"example {local}'s key is not in the record's example order; served as position {local}")
        return local

    def _oom_detail_locked(self, g: VirtualGPU, job: Job, gpu_index: int) -> dict:
        """Under g.lock: the card, this job's held bytes, the other holders by pid."""
        others = []
        for j, b in g.mem.items():
            if j != job.job_id and b > 0:
                oj = self.jobs.get(j)
                others.append({"pid": (oj.pid if oj else None), "bytes": int(b)})
        return {"gpu": gpu_index, "capacity_bytes": g.capacity_bytes,
                "mine_bytes": int(g.mem.get(job.job_id, 0)), "others": others}

    def _log_oversubscribed(self, job: Job, gpu_index: int, step, used: int, detail: dict) -> None:
        # logged as a failure only when the patch will act on it (oom_aware), so
        # the log never claims a failure the job did not have
        self.occ.event("failure", gpu=gpu_index, job=job.job_id, mode="oom",
                       label="oversubscribed", step=step, used_bytes=used,
                       capacity_bytes=self.gpus[gpu_index].capacity_bytes, mine_bytes=detail["mine_bytes"],
                       others=detail["others"], record_id=job.record_id)

    def _hold(self, job: Job, gpu_index: int, delta: int, oom_aware: bool = False) -> dict:
        """Hold (or release) bytes on a GPU without queueing work: the record's
        weights from engagement, so nvidia-smi shows the model arriving before
        the first paced unit brings the rest of the peak.
        The same oversubscription rule as _enqueue: the arrival that crosses
        the card is the job that OOMs."""
        g = self.gpus[gpu_index]
        with g.lock:
            if delta:
                g.mem[job.job_id] = max(0, g.mem.get(job.job_id, 0) + int(delta))
            used = g.used_bytes()
            free = g.free_bytes()
            oom_detail = self._oom_detail_locked(g, job, gpu_index) if (delta > 0 and used > g.capacity_bytes) else None
        self.occ.event("hold", gpu=gpu_index, job=job.job_id, bytes=int(delta),
                       used_bytes=used, free_bytes=free, record_id=job.record_id)
        if oom_detail is not None and oom_aware:
            self._log_oversubscribed(job, gpu_index, None, used, oom_detail)
        return {"used_bytes": used, "free_bytes": free, "oom": used > g.capacity_bytes,
                "oom_detail": oom_detail}

    def _recompute_idle(self, g: VirtualGPU) -> None:
        # Called under g.lock already held by caller in _end_job; here we only
        # log a transition to idle if the queue has drained.
        if not g.is_busy():
            self.occ.event("busy_end", gpu=g.index)

    def _sync_wait_s(self, gpu_index: int) -> float:
        """Seconds the caller must sleep for the GPU cursor to pass now()."""
        g = self.gpus[gpu_index]
        with g.lock:
            wait = max(0.0, g.cursor - now())
        return wait

    # -- published state for the bridge ----------------------------------

    def snapshot(self) -> dict:
        """What the bridge turns into overrides.yaml. Sampled, not streamed."""
        t = now()
        out = {"gpus": [], "t_wall": time.time()}
        for g in self.gpus:
            with g.lock:
                busy = g.is_busy(t)
                used = g.used_bytes()
                free = g.free_bytes()
                procs = [{"pid": self.jobs[j].pid, "used_bytes": b}
                         for j, b in g.mem.items() if j in self.jobs]
            out["gpus"].append({
                "index": g.index,
                "util_gpu": 100 if busy else 0,   # bridge smooths over 1s
                "used_bytes": used,
                "free_bytes": free,
                "capacity_bytes": g.capacity_bytes,
                "processes": procs,
            })
        return out

    # -- request dispatch -------------------------------------------------

    def _dispatch(self, job: Optional[Job], req: dict) -> dict:
        cmd = req.get("cmd")
        if cmd == "hello":
            job = self._new_job(req.get("pid", -1), req.get("argv", []),
                                 req.get("cwd", ""), observer=bool(req.get("observer")),
                                 t_start=req.get("t_start"))
            return {"ok": True, "job_id": job.job_id, "speed": self.speed}
        if job is None:
            # Stateless pre-import whole_job peek (next/ r8.1, whole_job_probe.arm):
            # a lookup at stage whole_job_probe with a cwd and NO prior hello is a
            # read-only registry match that creates NO job, logs NO event, and
            # commits NO draw. It returns only whether the record is a whole_job
            # library seam, so the arm can decide to install the seam at the seam
            # module's import (before a lazily-imported torch makes the torch-time
            # probe fire too late). Requiring a hello here instead made the peek
            # register a phantom job that corrupted per_example multi-launch
            # accounting (17/17 -> 12/5, 2026-11-02); a stateless peek touches
            # nothing the real launch's own hello/lookup will do.
            if cmd == "lookup" and req.get("stage") == "whole_job_probe" and req.get("cwd") is not None:
                argv_n = _normalise_argv_paths(req.get("argv"), req.get("cwd")) if req.get("argv") is not None else None
                # read-only (next/ r8.7): registry.lookup() here drew a record per launch
                rec = self.registry.peek_whole_job(
                    req.get("entry_hash"), argv_n, script_path=req.get("script_path"),
                    argv_effective=req.get("argv_effective"), argv_defaults=req.get("argv_defaults"))
                mode = (rec.get("mode", "replay") if rec is not None else None)
                wj = _whole_job_reply(rec) if (rec is not None and mode == "replay") else None
                return {"ok": True, "mode": (mode if wj is not None else None),
                        "record_id": (rec or {}).get("identity", {}).get("record_id") if wj is not None else None,
                        "serving": (_serving_info(rec) if wj is not None else None)}
            return {"ok": False, "error": "no hello"}
        if cmd == "lookup":
            fp = req.get("fingerprint")
            if req.get("trial") is not None and int(req["trial"]) > job.trial:
                self._begin_trial(job, int(req["trial"]))
            # path-valued tokens in the record's form (cwd-relative, normpath'd)
            argv_n = _normalise_argv_paths(req.get("argv"), job.cwd) if req.get("argv") is not None else None
            probe = req.get("stage") == "whole_job_probe"
            rec, mode, info = self.registry.lookup(
                fp, req.get("entry_hash"), argv_n,
                job_key=req.get("job_key"), stage="exact" if probe else req.get("stage", "exact"),
                script_path=req.get("script_path"),
                fingerprint_inputs=req.get("fingerprint_inputs"),
                argv_effective=req.get("argv_effective"),
                argv_defaults=req.get("argv_defaults"),
                modules=req.get("modules"))
            if probe:
                # The whole_job early probe (next/ r8, replay.try_whole_job_early):
                # it engages the patch early ONLY for a whole_job record. The draw
                # is cached by job key, so the job's own first-CUDA-touch exact
                # lookup re-commits the same record. Here, commit nothing to the
                # job and log a `whole_job_probe` event, not a `lookup`: a probe
                # that returns no whole_job serving must leave the job exactly as
                # a job that never probed, so step/per_example/structural records
                # engage on their normal path. Serving is returned only for a
                # whole_job record.
                wj = _whole_job_reply(rec) if (rec is not None and mode == "replay") else None
                self.occ.event("whole_job_probe", gpu=-1, job=job.job_id, record_id=(rec or {}).get("identity", {}).get("record_id"),
                               match=info.get("match"), is_whole_job=bool(wj), job_key=req.get("job_key"))
                if wj is None:
                    return {"ok": True, "mode": mode, "record_id": None, "has_record": rec is not None,
                            "match": info.get("match"), "serving": None}
                job.record = rec; job.drawn = rec; job.mode = mode
                job.record_id = (rec.get("identity") or {}).get("record_id")
                job.rank = (int(req["rank"]) if req.get("rank") is not None else None)
                job.job_key = req.get("job_key"); job.draw = info.get("draw")
                serving = _serving_info(rec)
                if serving is not None and req.get("world_size"):
                    conv = _convert(serving, int(req["world_size"]))
                    if conv.get("curve_applies") is not False:
                        serving = dict(serving, converted=conv)
                return {"ok": True, "mode": mode, "record_id": job.record_id, "speed": self.speed,
                        "has_record": True, "match": info.get("match"), "serving": serving}
            # prefix upgrade: a draw that landed on a prefix / prepared record
            # whose stop the launch does not fit (no --stop-at, or one past the
            # prefix's steps) is served the record's full-schedule parent — same
            # unit, same trajectory, the whole schedule's pacing. The drawn
            # (listed) record keeps the job's identity for the scorer.
            drawn = rec
            upgraded = None
            if rec is not None and mode == "replay":
                _, stop_at = _strip_stop_at(argv_n or [])
                par = self.registry.full_run_for(rec)
                if par is not None and (stop_at is None or stop_at > int(rec.get("steps") or 0)):
                    upgraded = par
                    rec = _served_full(drawn, par)
            job.record = rec
            job.drawn = drawn
            job.mode = mode
            job.fingerprint = fp
            job.record_id = (drawn or {}).get("identity", {}).get("record_id")
            job.rank = (int(req["rank"]) if req.get("rank") is not None else None)
            job.job_key = req.get("job_key")
            job.draw = info.get("draw")
            # served failures: a failure with a trigger is armed only when the
            # launch's argument satisfies it (the H2 relaunch under the limit
            # draws the same record and runs clean)
            arm = _arm_failures(rec, argv_n, req.get("argv_effective"))
            job.armed = (set(arm["armed"]) if any((f or {}).get("trigger") for f in ((rec or {}).get("failures") or [])) else None)
            if job.armed is not None:
                self.occ.event("failures_armed", gpu=-1, job=job.job_id, record_id=job.record_id, **arm)
            pe = _per_example(rec)
            job.subset = _subset_of(pe, req.get("argv"), req.get("subset_env")) if pe else None
            job.pe_order = self.registry.order_for(rec) if pe else None
            job.entry_module = info.get("entry_module")
            fallback_from = None
            if info.get("match") == "none" and req.get("entry_hash"):
                nm = self.registry.near_miss(req.get("entry_hash"), argv_n)
                if nm is None and info.get("path_exact") == "suffix_argv_miss" and info.get("bound_path"):
                    # an edited copy at a known path with a new config: the near
                    # miss is against that path's records, not the live bytes'
                    nm = self.registry.near_miss(None, argv_n,
                                                 recs=self.registry._by_bound.get(info["bound_path"]))
                if nm:
                    self.occ.event("near_miss", gpu=-1, job=job.job_id, stage=req.get("stage"), **nm)
                    fallback_from = "near_miss"     # the modelled path runs real compute: the box's miss, not the workload's cost
            self.occ.event("lookup", gpu=-1, job=job.job_id, fingerprint=fp,
                           mode=mode, record_id=job.record_id,
                           match=info.get("match"), draw=info.get("draw"),
                           group=info.get("group"), stage=req.get("stage"),
                           rank=req.get("rank"), job_key=req.get("job_key"),
                           derived_from=_derived_from(rec),
                           bound_path=info.get("bound_path"), script_path=info.get("script_path"),
                           sibling_of=info.get("sibling_of"), sibling_rank=info.get("sibling_rank"),
                           siblings_exhausted=info.get("siblings_exhausted"),
                           task_job=info.get("task_job"), bind=info.get("bind"), fallback_from=fallback_from,
                           candidate=info.get("candidate"), slot_launches=info.get("slot_launches"),
                           surrogate=info.get("surrogate"),
                           trial=(job.trial if job.trial > 1 else None),
                           rebuilt=(req.get("trial_rebuilt") if job.trial > 1 else None),
                           subset=job.subset,
                           readiness_overridden=info.get("readiness_overridden"),
                           path_exact=info.get("path_exact"), stop_at=info.get("stop_at"),
                           defaulted_flags=info.get("defaulted_flags"),
                           entry_module=info.get("entry_module"), entry_module_hash=info.get("entry_module_hash"),
                           example_order=((job.pe_order or {}).get("file") if pe else None),
                           served_from=((upgraded.get("identity") or {}).get("record_id") if upgraded else None),
                           served_steps=(rec.get("steps") if upgraded else None),
                           prefix_steps=((drawn or {}).get("steps") if upgraded else None),
                           entry_hash_live=(req.get("entry_hash") if str(info.get("match") or "").startswith("path") else None),
                           entry_hash_record=(((rec or {}).get("identity") or {}).get("entry_hash")
                                              if str(info.get("match") or "").startswith("path") else None))
            serving = _serving_info(rec)
            if serving is not None and job.armed is not None and serving.get("failures"):
                serving = dict(serving, failures=[dict(f, armed=(i in job.armed))
                                                  for i, f in enumerate(serving["failures"])])
            conv = None
            if serving is not None and req.get("world_size"):
                # GPU-count conversion (unit 5): the record's times are on its
                # recorded GPU count; pace for the count the agent launched.
                # a single-process record (gpu-tasks' single_process scripts, H1/H4):
                # the script never goes distributed, so N torchrun copies are N
                # independent whole-run copies on one GPU, not an N-GPU speed-up.
                # Pace at 1 GPU whatever WORLD_SIZE says; the copies then contend
                # on one cursor.
                _ws = int(req["world_size"])
                if str((serving.get("process_model") or (_per_example(rec) or {}).get("process_model") or "")).lower() == "single":
                    _ws = 1
                conv = _convert(serving, _ws)
                if conv.get("curve_applies") is False:
                    # the recorded curve does not describe this launch
                    # (fixed_per_device): modelled path, logged.
                    mode = "model"
                    job.mode = mode
                else:
                    serving = dict(serving, recorded_steady_step_s=serving.get("steady_step_s"),
                                   recorded_first_step_s=serving.get("first_step_s"),
                                   steady_step_s=conv["steady_step_s"],
                                   first_step_s=conv["first_step_s"],
                                   eval_s_per_validation=conv["eval_s_per_validation"],
                                   converted=conv)
                self.occ.event("convert", gpu=-1, job=job.job_id,
                               record_id=job.record_id, **{k: v for k, v in conv.items()
                                                           if k in ("from_gpus", "to_gpus",
                                                                    "batch_policy", "time_factor",
                                                                    "memory_factor", "weights_factor",
                                                                    "memory_provenance", "curve_applies",
                                                                    "provenance")})
            return {"ok": True, "mode": mode, "record_id": job.record_id, "speed": self.speed,
                    "has_record": rec is not None, "match": info.get("match"),
                    "draw": info.get("draw"), "subset": job.subset, "serving": serving,
                    "converted": conv, "entry_module": info.get("entry_module")}
        if cmd == "enqueue":
            st = req.get("step")
            res = self._enqueue(job, int(req.get("gpu", 0)),
                                 req.get("op", "kernel"),
                                 float(req.get("modelled_s", 0.0)),
                                 int(req.get("mem_delta", 0)),
                                 step=int(st) if st is not None else None,
                                 unit=req.get("unit"), lr=req.get("lr"),
                                 startup_s=req.get("startup_s"), startup_real_s=req.get("startup_real_s"),
                                 host_gap_s=req.get("host_gap_s"),
                                 oom_aware=bool(req.get("oom_aware")),
                                 key=req.get("key"))
            return {"ok": True, **res}
        if cmd == "hold":
            return {"ok": True, **self._hold(job, int(req.get("gpu", 0)), int(req.get("bytes", 0)),
                                             oom_aware=bool(req.get("oom_aware")))}
        if cmd == "exception":
            # the job's own uncaught exception, reported by the patch's
            # excepthook before the process dies: a job that ends before its
            # record's last step carries its cause (runner's scorer, 2026-10-21)
            self.occ.event("exception", gpu=-1, job=job.job_id,
                           etype=str(req.get("etype", ""))[:80],
                           message=str(req.get("message", ""))[:300],
                           step=req.get("step"), record_id=job.record_id,
                           # innermost traceback frame (where it was raised)
                           frame_file=str(req.get("frame_file", ""))[:300] or None,
                           frame_line=req.get("frame_line"),
                           frame_func=str(req.get("frame_func", ""))[:120] or None)
            return {"ok": True}
        if cmd == "sync":
            wait = self._sync_wait_s(int(req.get("gpu", 0)))
            return {"ok": True, "sleep_s": wait}
        if cmd == "materialise":
            return self._materialise(job, req)
        if cmd == "fail?":
            return self._pending_failure(job, req)
        if cmd == "snapshot":
            return {"ok": True, "snapshot": self.snapshot()}
        if cmd == "fingerprint_check":
            return self._fingerprint_check(job, req)
        if cmd == "disengage":
            # the job withdrew replay (lr_verify mismatch, or a launch that sees
            # no CUDA device and runs on the CPU): serve nothing more
            self.occ.event("replay_disengaged", gpu=-1, job=job.job_id,
                           record_id=job.record_id, step=req.get("step"),
                           reason=req.get("reason"),
                           cuda_visible_devices=req.get("cuda_visible_devices"),
                           detail=req.get("detail"))
            if str(req.get("reason") or "").startswith(("whole_job_identity_", "identity_")) and job.record is not None:
                # next/ r8.5: the launch is not the recorded configuration (a cut
                # env, an edited seam module, another runtime; r8.9: an edited bound
                # file, a shadowing module, an evaluator patched in memory on any
                # path, checked before the first served unit). Nothing was served
                # and nothing will be, so the draw goes back: the next matching
                # launch draws the record as a first draw, not a rerun.
                self.registry.return_draw(job.drawn or job.record, job.job_key)
            job.record = None
            job.mode = "model"
            return {"ok": True}
        if cmd == "bye":
            self._end_job(job)
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd {cmd!r}"}

    def _materialise(self, job: Job, req: dict) -> dict:
        """Serve a value from the record by (role, step, kind).

        Serving order (REPLAY-INTERFACE §8): record series by role → final
        `result.*` for the final-metrics mode → derive from meta/memory → derive
        from loss (last resort). Series may be the object form
        {"steps":[...],"values":[...]} (sparse, RECORD-FORMAT §8) or a plain list.
        Logged with job identity and the derivation tier.
        """
        sur = (job.record or {}).get("surrogate")
        if sur:
            return self._materialise_surrogate(job, req, sur)
        pe = _per_example(job.record)
        if pe is not None and req.get("role") == "example_score":
            return self._materialise_example_score(job, req, pe)
        wj = _whole_job(job.record)
        if wj is not None and req.get("role") == "whole_job_result":
            return self._materialise_whole_job(job, req, wj)
        if req.get("roles") is not None:
            # many roles at one step, one round trip (the Trainer seam asks once
            # per optimizer step for every series the record has)
            pick = req.get("pick", "at_or_before")
            step = int(req.get("step", job.step))
            series_all = (job.record or {}).get("series", {}) or {}
            tier = "series"
            if self._beyond_record(job, step):
                vals = {r: (_extrapolate(series_all.get(r), step, f"{job.record_id}:{r}") if series_all.get(r) is not None else None) for r in req["roles"]}
                tier = "extrapolated"
            else:
                def _batched(r):
                    ser = series_all.get(r)
                    if _is_eval_metric_role(r):
                        # an eval metric is read at_or_before whatever pick was asked:
                        # None before its first validation, else the latest validation
                        # at or before this step, never a future one (a threshold check
                        # must not see a later validation's score)
                        return None if _before_first_step(ser, step) else _series_value_at(ser, step, "at_or_before")
                    return _series_value_at(ser, step, pick)
                vals = {r: _batched(r) for r in req["roles"]}
            self.occ.event("materialise", gpu=-1, job=job.job_id, trial=(job.trial if job.trial > 1 else None), roles=len(vals), step=step,
                           pick=pick, value=vals.get("train_loss"), tier=tier,
                           record_id=job.record_id)
            return {"ok": True, "values": vals}
        role = req.get("role", "loss")
        step = int(req.get("step", job.step))
        kind = req.get("kind", "scalar")
        value, tier = None, "none"
        rec = job.record or {}

        beyond = self._beyond_record(job, step)
        series = rec.get("series", {}).get(role)
        # an eval metric read before its series' first recorded step has no
        # honest value: None, and it must not reach the result.* or loss-derived
        # fallbacks below, which would hand back a future or final score
        eval_unreached = _is_eval_metric_role(role) and series is not None and not beyond and _before_first_step(series, step)
        if eval_unreached:
            self.occ.event("materialise", gpu=-1, job=job.job_id, role=role, step=step, mkind=kind,
                           value=None, tier="eval_unreached", record_id=job.record_id,
                           detail=f"{role} read at step {step}, before its first recorded validation")
            return {"ok": True, "value": None, "tier": "eval_unreached"}
        if series is not None:
            value = _extrapolate(series, step, f"{job.record_id}:{role}") if beyond else _series_value_at(series, step)
            if value is not None:
                tier = "extrapolated" if beyond else "series"

        # final-metrics mode: a text-decode / eval job serves result.* at its
        # final materialisation (RECORD-FORMAT §11 "Output-level results").
        if value is None:
            result = rec.get("result", {})
            if role in result:
                rv = result[role]
                value = rv.get("value") if isinstance(rv, dict) else rv
                tier = "result"

        if value is None:
            loss = rec.get("series", {}).get("loss") or rec.get("series", {}).get("train_loss")
            if loss is not None:
                value = _extrapolate(loss, step, f"{job.record_id}:loss") if beyond else _nearest_series_value(loss, step)
                tier = ("extrapolated" if beyond else "loss_derived") if value is not None else tier
        if value is None and role in ("train_loss", "loss"):
            # a record with no train loss (track 3 never prints it): "serve a
            # finite value" — the nearest recorded val loss is the honest stand-in,
            # in EITHER direction: before the first validation it is the first
            # validation's value, never the default 0.0 (2026-10-21: 33 mng3
            # scripts' NaN guards read the train loss at step 0, and records whose
            # validations start at 125 were served 0.0)
            vl = rec.get("series", {}).get("val_loss")
            if vl is not None:
                value = _extrapolate(vl, step, f"{job.record_id}:val_loss") if beyond else _nearest_series_value(vl, step)
                tier = ("extrapolated" if beyond else "val_derived") if value is not None else tier
        if value is None:
            value, tier = 0.0, "default"

        self.occ.event("materialise", gpu=-1, job=job.job_id, trial=(job.trial if job.trial > 1 else None), role=role,
                       step=step, mkind=kind, value=value, tier=tier,
                       record_id=job.record_id, fingerprint=job.fingerprint)
        return {"ok": True, "value": value, "tier": tier}

    def _materialise_whole_job(self, job: Job, req: dict, wj: dict) -> dict:
        """A whole_job record's recorded result, served once: only after the job
        has paced every one of its n_units in order and the last unit's GPU time
        has passed. A job that ran its script to the end gets the record's result
        exactly; one that stopped early, or a client that never paced, gets
        nothing (and a read_ahead event), so reading the result costs what
        running the job costs (the rule r2b set for per-example scores)."""
        n = _whole_job_units(wj)
        rid = job.record_id or ""
        end = job.wj_last_end
        if job.wj_units < n or end is None or end > now() + READ_SLACK_S:
            if job.read_ahead_logged < 3:
                job.read_ahead_logged += 1
                self.occ.event("read_ahead", gpu=-1, job=job.job_id, role="whole_job_result", record_id=rid,
                               paced=job.wj_units, n_units=n,
                               wait_s=(round(end - now(), 3) if end is not None and job.wj_units >= n else None),
                               detail=(f"result asked after {job.wj_units} of {n} units" if job.wj_units < n
                                       else f"result asked {end - now():.2f} s before the last unit ends"))
            return {"ok": True, "value": None, "tier": "none"}
        key = wj.get("result_key")
        res = (job.record or {}).get("result") or {}
        value = None
        if key:
            v = res.get(key)
            value = _val(v) if isinstance(v, dict) and "value" in v else v
        job.wj_served = True
        self.occ.event("materialise", gpu=-1, job=job.job_id, role="whole_job_result", tier="result",
                       record_id=rid, n_units=n, result_key=key, result_kind=wj.get("result_kind"),
                       provenance=((res.get(key) or {}).get("provenance") if isinstance(res.get(key), dict) else None))
        return {"ok": True, "value": value, "tier": "result"}

    def _materialise_example_score(self, job: Job, req: dict, pe: dict) -> dict:
        """The i-th entry of the record's per-example score vector
        (docs/MLS-KV-SERVING.md §1.2), labelled result; a subset launch's
        reads carry provenance synthesised_subset and the subset model."""
        rid = job.record_id or ""
        vec = self._score_vectors.get(rid)
        if vec is None:
            vec = _score_vector(job.record or {})
            self._score_vectors[rid] = vec
        i = int(req.get("step", 0))
        # the record position of the job's i-th example: its content key's
        # position when the table holds it, else the position the job's i-th
        # paced unit opened, else i (a launch from the workload's first example)
        idx = (job.pe_order or {}).get("index")
        key = req.get("key")
        if idx is not None and key and key in idx:
            pos = idx[key]
        elif 0 <= i < len(job.pe_positions):
            pos = job.pe_positions[i]
        else:
            pos = i
        # A score exists once its example has been decoded: serve example i
        # only to a job that paced example i and whose unit's GPU time has
        # passed. The patch's seams always pace an example before scoring it,
        # so a job that runs its script is never refused; a client that asks
        # for scores without pacing them (the Grok canary read whole vectors in
        # a second, 2026-10-26) gets none, and reading the vector costs what
        # running the evaluation costs.
        end = job.example_ends.get(pos)
        if end is None or end > now() + READ_SLACK_S:
            if job.read_ahead_logged < 3:
                job.read_ahead_logged += 1
                # its own kind, not "failure": the runner's scorers read any
                # failure event as the job having failed (idle.py, hindsight.py)
                self.occ.event("read_ahead", gpu=-1, job=job.job_id, step=i, position=pos,
                               paced=len(job.pe_positions),
                               wait_s=(round(end - now(), 3) if end is not None else None), record_id=rid,
                               detail=(f"example {i} score asked before its example was paced" if end is None
                                       else f"example {i} score asked {end - now():.2f} s before its paced unit ends"))
            return {"ok": True, "value": None, "tier": "none"}
        value = vec[pos] if 0 <= pos < len(vec) else None
        if value is None and pos >= len(vec) and not job.overrun_logged:
            # the script has more examples than the record's n_examples_full
            # says: a record/loader mismatch (NeedleBench's 147 essays were
            # 441 examples, gpu-tasks 2026-10-22 23:19Z). Each example was paced
            # at wall / n_full, so the launch overruns the wall by the ratio;
            # say so once, where the scorer reads
            job.overrun_logged = True
            self.occ.event("failure", gpu=-1, job=job.job_id, mode="per_example_overrun",
                           step=i, n_full=len(vec), record_id=rid,
                           detail=f"example {i} requested; the record's n_examples_full is {len(vec)}")
        kind = pe.get("score_kind") or "binary"
        model = {"kind": "beta", "concentration": 2} if kind == "continuous" else {"kind": "bernoulli"}
        # next/ r6: a position inside the record's measured head is a real score,
        # decoded with the record's own evaluator; past it, the synthesised tail
        meas = _measured_scores(job.record)
        measured = bool(meas) and 0 <= pos < meas["k"]
        self.occ.event("materialise", gpu=-1, job=job.job_id, trial=(job.trial if job.trial > 1 else None),
                       role="example_score", step=i, mkind="scalar", value=value, tier="result",
                       example=i, position=(pos if pos != i else None), n_full=len(vec), score_kind=kind, subset=job.subset,
                       # a value served out of the workload's prefix order, or to a
                       # launch that loaded the evaluator as a module, belongs to a slice
                       # the launch may have cut itself: its mean is a subset mean unless
                       # job_bye's positions cover n_full
                       provenance=("measured" if measured else
                                   ("synthesised_subset" if (job.subset or pos != i or job.entry_module) else "result")),
                       subset_model=(None if measured else (model if (job.subset or pos != i or job.entry_module) else None)),
                       measured_k=(meas["k"] if meas else None), startup_s=job.startup_s,
                       startup_real_s=job.startup_real_s, record_id=rid)
        return {"ok": True, "value": value, "tier": "result"}

    def _beyond_record(self, job: Job, step: int) -> bool:
        """True when a read lies past the served record's last step; the first
        time per (job, trial) a `record_exhausted` event says so."""
        rec_steps = (job.record or {}).get("steps")
        try:
            rec_steps = int(rec_steps) if rec_steps is not None else None
        except (TypeError, ValueError):
            rec_steps = None
        if not rec_steps or int(step) <= rec_steps:
            return False
        if not job.exhausted_logged:
            job.exhausted_logged = True
            self.occ.event("record_exhausted", gpu=-1, job=job.job_id, record_id=job.record_id,
                           record_steps=rec_steps, step=int(step),
                           trial=(job.trial if job.trial > 1 else None))
        return True

    def _materialise_surrogate(self, job: Job, req: dict, sur: dict) -> dict:
        """A synthesised job's reads come from the family's curve; every event
        says `tier: surrogate`; the first NaN of a divergent configuration is
        logged once as a failure of kind surrogate_divergence."""
        fam = self.registry.families.get(sur.get("family"))
        step = int(req.get("step", job.step))
        cfg, seed = sur.get("config") or {}, int(sur.get("seed") or 0)
        roles = req.get("roles") if req.get("roles") is not None else [req.get("role", "loss")]
        vals = {}
        for r in roles:
            v = fam.value(r, step, cfg, seed, lr_history=job.lr_history) if fam is not None else None
            vals[r] = v
            if v is not None and isinstance(v, float) and v != v and not job.diverged_logged:
                job.diverged_logged = True
                self.occ.event("failure", gpu=-1, job=job.job_id, mode="surrogate_divergence",
                               xid=None, step=step, record_id=job.record_id)
        if req.get("roles") is not None:
            self.occ.event("materialise", gpu=-1, job=job.job_id, trial=(job.trial if job.trial > 1 else None), roles=len(vals), step=step,
                           pick=req.get("pick"), value=vals.get("train_loss"), tier="surrogate",
                           record_id=job.record_id)
            return {"ok": True, "values": vals}
        role = roles[0]
        self.occ.event("materialise", gpu=-1, job=job.job_id, trial=(job.trial if job.trial > 1 else None), role=role, step=step,
                       mkind=req.get("kind", "scalar"), value=vals[role], tier="surrogate",
                       record_id=job.record_id, fingerprint=job.fingerprint)
        return {"ok": True, "value": vals[role], "tier": "surrogate"}

    def _fingerprint_check(self, job: Job, req: dict) -> dict:
        """Log how the job's live fingerprint compares with its record's, for a
        job matched by path (the agent wrote the script): never a gate, only the
        disclosure's evidence of what differed."""
        rec = job.record or {}
        ident = rec.get("identity") or {}
        rec_fp = _fp_value(ident.get("fingerprint"))
        live_fp = req.get("fingerprint")
        rec_in = ident.get("fingerprint_inputs") or {}
        live_in = req.get("inputs") or {}
        keys = ("model", "num_layers", "model_dim", "vocab_size", "optimizers", "world_size",
                "precision", "torch_compile", "num_parameters", "accumulation")
        differ = {}
        for k in keys:
            a, b = rec_in.get(k), live_in.get(k)
            if a is None and b is None:
                continue
            if k == "optimizers":
                a = sorted(str(x).lower() for x in (a or [])); b = sorted(str(x).lower() for x in (b or []))
            if a != b:
                differ[k] = {"record": a, "live": b}
        mb_rec = rec_in.get("per_device_batch", rec_in.get("micro_batch"))
        mb_live = live_in.get("micro_batch", live_in.get("per_device_batch"))
        if mb_rec is not None and mb_live is not None and int(mb_rec) != int(mb_live):
            differ["micro_batch"] = {"record": mb_rec, "live": mb_live}
        agree = bool(rec_fp) and rec_fp == live_fp
        job.fingerprint_differs = not agree
        self.occ.event("fingerprint_check", gpu=-1, job=job.job_id, record_id=job.record_id,
                       match=req.get("match"), agree=agree, fingerprint_record=rec_fp,
                       fingerprint_live=live_fp, differs=differ)
        return {"ok": True, "agree": agree, "differs": differ}

    def _pending_failure(self, job: Job, req: dict) -> dict:
        """Return a failure the record schedules at or before this step/time."""
        rec = job.record
        if not rec:
            return {"ok": True, "failure": None}
        step = int(req.get("step", job.step))
        phase = req.get("phase", "training")
        where = req.get("where") or "step"      # forward | backward | step: which seam is asking
        # The fired state lives on the JOB. Every job that draws a record shares
        # the registry's own dict, so a flag written into the failure entry made
        # a firing failure fire once per daemon: never again for a later job
        # drawing the same record (recycled or seed-substituted), and for only
        # one rank of a multi-rank job.
        for i, f in enumerate(rec.get("failures", [])):
            # RECORD-FORMAT: failures carry replay = fire | suppress. Suppressed
            # ones (a cause-matched collision in the original project, or a Hub
            # push after training) are recorded but must not fire.
            if _norm_failure(f)["replay"] == "suppress":
                continue
            # One that happened after training fires when training has ended
            # (the replay's Trainer seam asks with phase=after_training), never
            # at a step.
            if (f.get("at") == "after_training") != (phase == "after_training"):
                continue
            # a failure surfaces at one seam of its step: the forward (where a
            # real OOM mostly does), the backward, or the optimizer step (default)
            if phase != "after_training" and (f.get("where") or "step") != where:
                continue
            # a triggered failure fires only for the launch that armed it; one
            # that names a rank fires only on that rank (torchrun ends the rest)
            if job.armed is not None and i not in job.armed:
                continue
            if f.get("rank") is not None and job.rank is not None and int(f["rank"]) != job.rank:
                continue
            if (phase == "after_training" or (f.get("step") is not None and f["step"] <= step)) \
                    and i not in job.fired:
                job.fired.add(i)
                exc = _failure_exception(f)
                text = _val(f.get("stderr"))
                text = text if isinstance(text, str) and text else None
                code = _val(f.get("exit_code"))
                self.occ.event("failure", gpu=int(f.get("gpu", 0)),
                               job=job.job_id, mode=f.get("mode"),
                               xid=f.get("xid"), step=step,
                               label="served_failure", record_id=job.record_id,
                               trigger=f.get("trigger"), provenance=f.get("provenance"),
                               exception=(exc or {}).get("type"),
                               stderr_bytes=(len(text) if text else None), rank=f.get("rank"))
                return {"ok": True, "failure": {
                    "mode": f.get("mode"), "xid": f.get("xid"),
                    "gpu": f.get("gpu", 0), "label": "served_failure",
                    "exception": exc, "stderr": text,
                    "exit_code": (int(code) if code is not None else None),
                    "trigger": f.get("trigger")}}
        return {"ok": True, "failure": None}

    # -- server loop ------------------------------------------------------

    def _handle_conn(self, conn: socket.socket) -> None:
        """One client connection: requests until end of stream. The job it
        registered with `hello` ends when the stream ends, HOWEVER it ends:
        an explicit `bye` (reason bye), the socket closing on a clean exit, a
        crash or a SIGKILL (eof), or an error inside this handler, e.g. the
        reply write failing because the client died between its request and
        the reply (error:<type>). Before 2026-10-22 the last case left the
        job registered with its memory held and no job_bye (isolation's
        idle_after_harness rule counts such a job as connected)."""
        job: Optional[Job] = None
        reason = "eof"
        with conn:
            try:
                f = conn.makefile("rwb", buffering=0)
                for raw in f:
                    try:
                        req = json.loads(raw.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        f.write(b'{"ok":false,"error":"bad json"}\n')
                        continue
                    resp = self._dispatch(job, req)
                    if req.get("cmd") == "hello" and resp.get("ok"):
                        job = self.jobs[resp["job_id"]]
                    if req.get("cmd") == "bye":
                        reason = "bye"
                    f.write((json.dumps(resp, separators=(",", ":")) + "\n").encode())
            except Exception as e:          # noqa: BLE001 - the job must end whatever broke
                reason = f"error:{type(e).__name__}"
                log(f"connection of job {job.job_id if job else '?'} ended by {e!r}")
            finally:
                if job is not None and job.job_id in self.jobs:
                    self._end_job(job, reason=reason)

    def serve(self) -> None:
        # AF_UNIX sun_path is 104 bytes on macOS, 108 on Linux, NUL included.
        if len(str(self.socket_path).encode()) > 103:
            raise SystemExit(
                f"gpu-mock daemon: socket path is {len(str(self.socket_path))} bytes; "
                f"AF_UNIX allows at most 103. Use a short path such as "
                f"/var/run/nvidia-persistenced/socket or /tmp/<name>.sock: "
                f"{self.socket_path}")
        if self.socket_path.exists():
            self.socket_path.unlink()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.socket_path))
        srv.listen(64)
        self._server = srv
        self.occ.event("daemon_start", gpu=-1, gpus=len(self.gpus),
                       speed=self.speed)
        # what the per_example serving reads beyond the records (next/ r6): every
        # example-order table by file and sha256, and how many records carry
        # measured scores, so a run's provenance names them
        tabs = {}
        for ent in self.registry.example_order.values():
            tabs[ent.get("file")] = ent.get("sha256")
        meas = sorted((r.get("identity") or {}).get("record_id") for r in self.registry._by_id.values() if _measured_scores(r))
        self.occ.event("registry_loaded", gpu=-1, example_order=tabs or None,
                       measured_records=len(meas) or None,
                       measured_k=sorted({_measured_scores(r)["k"] for r in self.registry._by_id.values() if _measured_scores(r)}) or None)
        try:
            while True:
                conn, _ = srv.accept()
                threading.Thread(target=self._handle_conn, args=(conn,),
                                 daemon=True).start()
        finally:
            srv.close()
            if self.socket_path.exists():
                self.socket_path.unlink()
            self.occ.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", default="/var/run/nvidia-persistenced/socket")
    ap.add_argument("--registry", default=None)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--capacity-gib", type=float, default=80.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="knob: >1 runs modelled time faster")
    ap.add_argument("--occupancy", default="/tmp/gpu-mock-occupancy.jsonl")
    ap.add_argument("--surrogates", default=None,
                    help="directory of surrogate families (registry/surrogates); default: the registry's sibling")
    ap.add_argument("--task", default=None,
                    help="the run's task.json: jobs[].rerun_siblings makes a rerun of a job serve a sibling run")
    args = ap.parse_args()

    reg = Registry(Path(args.registry) if args.registry else None)
    sdir = Path(args.surrogates) if args.surrogates else (Path(args.registry).parent / "surrogates" if args.registry else None)
    if sdir and sdir.is_dir():
        log(f"surrogate families: {reg.load_surrogates(sdir)} from {sdir}")
    if args.task:
        n = reg.load_task(Path(args.task))
        log(f"task {reg.task_id}: rerun siblings for {n} primary records")
    cap = int(args.capacity_gib * (1024 ** 3))
    d = GPUDaemon(Path(args.socket), reg, args.gpus, cap, args.speed,
                  Path(args.occupancy))
    d.serve()


if __name__ == "__main__":
    main()
