"""Post-draw verification of a replayed job against its record's learning-rate
trace (RECORD-FORMAT §5, `structural_verified`). Torch-free so it is unit-testable
and so the expected-scores calculator can import it beside the daemon.

The fingerprint narrows candidates; it never authorises replay by itself. A
structural draw is confirmed, step by step, by comparing the learning rate the
job's optimizer actually carries at its k-th optimizer.step() call with the
record's trace:

  alignment "after_step"  trace value at step k-1 is the rate step k uses
                          (the HF Trainer logs after lr_scheduler.step());
  alignment "at_step"     trace value at step k.

A step absent from the trace constrains nothing, so a different logging cadence
is never a mismatch. Tolerance: relative 1e-6 or absolute 1e-12 (either passes).
`per_group`, when present, holds one value list per param-group index.
"""

from __future__ import annotations

from typing import Optional


def _index(trace: dict) -> dict:
    idx = trace.get("_index")
    if idx is None:
        idx = {int(s): i for i, s in enumerate(trace.get("steps") or [])}
        trace["_index"] = idx
    return idx


def expected_lr(trace: Optional[dict], k: int) -> Optional[list]:
    """Expected rate(s) at the k-th optimizer.step() call (k is 1-based), as a
    list with one entry per param group when the trace is per-group, else a
    one-element list. None when the trace places no constraint on step k."""
    if not trace or not trace.get("steps"):
        return None
    align = trace.get("alignment") or "after_step"
    want = k - 1 if align == "after_step" else k
    i = _index(trace).get(want)
    if i is None:
        return None
    per_group = trace.get("per_group")
    if per_group:
        return [float(g[i]) for g in per_group]
    return [float(trace["values"][i])]


def _close(a: float, b: float, tol: dict) -> bool:
    rel = float(tol.get("rel", 1e-6))
    ab = float(tol.get("abs", 1e-12))
    return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))


def mismatch(trace: Optional[dict], k: int, observed: list) -> Optional[dict]:
    """Compare the observed per-param-group rates at the k-th step with the
    trace. Returns None when they agree or the trace says nothing about step k,
    else a dict naming the step, group, expected and observed values."""
    exp = expected_lr(trace, k)
    if exp is None or not observed:
        return None
    tol = trace.get("tolerance") or {}
    for g, ob in enumerate(observed):
        want = exp[g] if len(exp) > 1 else exp[0]
        if len(exp) > 1 and g >= len(exp):
            break
        if not _close(float(ob), want, tol):
            return {"step": k, "group": g, "expected": want, "observed": float(ob),
                    "alignment": trace.get("alignment") or "after_step"}
    return None
