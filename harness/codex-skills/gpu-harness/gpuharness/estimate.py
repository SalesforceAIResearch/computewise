"""Online duration model.

A job's length is modelled as the product of a factor for its candidate and a factor for its kind of work:

    log d(c, k) = a_c + b_k

so one measured job of a candidate tells the scheduler about that candidate's other jobs, and one measured job of a
kind tells it about the other candidates' jobs of that kind. This is how a method that is 3x slower than the
reference on a short workload is predicted 3x slower on its long workload before that long run has finished.

Observations, weighted:
    measured durations of finished jobs         weight 1
    projections from a running job's own pace   weight PROJECTION_WEIGHT (they move as the job runs)
    a running job's elapsed time, once it passes its prediction (a lower bound)   weight PROJECTION_WEIGHT
    the plan's estimates (the agent's priors)   weight PRIOR_WEIGHT

The reference's factor is pinned at 1: it is the yardstick, so what it shows about a kind is about the kind. Once the
reference has measured a kind (or a running reference job has passed its prediction), that kind's factor comes from
the machine's observations alone, not the agent's priors.
Every other candidate's factor comes from its priors only until the machine has measured it; after that, from its measurements
and projections alone. Priors keep shaping the kind factors and the differences between jobs within a cell.

The factors are fitted by alternating weighted least squares in log space with a small ridge towards zero, which
keeps a cell with no evidence of its own at the product of its row and column. A finished job's own measurement is
always used for that job itself; the model is for the jobs that have not run.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

PRIOR_WEIGHT = 0.3
PROJECTION_WEIGHT = 0.6
RIDGE = 0.05
DEFAULT_S = 600.0
NO_CANDIDATE = "_"


def _kind(job) -> str:
    return job.kind or job.id


def _cand(job) -> str:
    g = job.group or job.candidate or NO_CANDIDATE
    # a CPU-only job's length says nothing about its group's GPU pace: it gets its own row in the model
    return "cpu:" + g if getattr(job, "gpus", 1) == 0 else g


class Estimator:
    """Predict job lengths from the plan's priors and what the machine has measured so far."""

    def __init__(self, default_s: float = DEFAULT_S) -> None:
        self.default_s = default_s
        self._obs: Dict[Tuple[str, str, str], Tuple[float, float]] = {}  # (cand, kind, source) -> (log d, weight)
        self._a: Dict[str, float] = {}
        self._b: Dict[str, float] = {}
        self._dirty = True
        self.reference: frozenset = frozenset()   # groups whose factor is pinned at 1: the yardstick
        self.reference_known = False              # set_reference has been told the plan's reference (or its absence)

    # evidence -------------------------------------------------------------------------------------------------------
    def prior(self, job, seconds: Optional[float]) -> None:
        if seconds and seconds > 0:
            self._put(_cand(job), _kind(job), "prior:" + job.id, seconds, PRIOR_WEIGHT)

    def measured(self, job, seconds: float) -> None:
        if seconds > 0:
            self._put(_cand(job), _kind(job), "measured:" + job.id, seconds, 1.0)
            self._obs.pop((_cand(job), _kind(job), "projected:" + job.id), None)
            self._obs.pop((_cand(job), _kind(job), "running:" + job.id), None)

    def running(self, job, elapsed: float) -> None:
        """A running job with no pace of its own has lasted `elapsed` so far, so it is at least that long. Once that
        passes its prediction, the bound is evidence that its cell is longer than predicted (weighted like a
        projection): a workload whose jobs all overrun their priors is found long before any of them ends."""
        if elapsed <= 0 or (_cand(job), _kind(job), "projected:" + job.id) in self._obs:
            return
        key = (_cand(job), _kind(job), "running:" + job.id)
        if key in self._obs:
            if elapsed > math.exp(self._obs[key][0]) * 1.05:
                self._put(*key, elapsed, PROJECTION_WEIGHT)
        elif elapsed > self.predict(job):
            self._put(*key, elapsed, PROJECTION_WEIGHT)

    def projected(self, job, seconds: Optional[float]) -> None:
        if seconds and seconds > 0:
            self._put(_cand(job), _kind(job), "projected:" + job.id, seconds, PROJECTION_WEIGHT)

    def _put(self, cand: str, kind: str, source: str, seconds: float, weight: float) -> None:
        self._obs[(cand, kind, source)] = (math.log(seconds), weight)
        self._dirty = True

    # model ----------------------------------------------------------------------------------------------------------
    def _fit(self, iterations: int = 25) -> None:
        cands = {c for c, _, _ in self._obs}
        kinds = {k for _, k, _ in self._obs}
        a = {c: self._a.get(c, 0.0) for c in cands}
        b = {k: self._b.get(k, 0.0) for k in kinds}
        if not self._obs:
            self._a, self._b, self._dirty = a, b, False
            return
        # a kind the reference has measured (or bounded) gets its factor from what the machine shows alone: the
        # reference's factor is pinned, so that evidence identifies the kind, and the agent's per-kind prior copied to
        # every method would otherwise drag a long workload back towards its guess. A kind only other methods have
        # run keeps its priors, which anchor it so a single slow method's run is blamed on the method, not the kind.
        ruled = {k for c, k, src in self._obs if c in self.reference and not src.startswith("prior:")}
        # a plan that names no reference has no yardstick at all, so a kind's own measurements (and running bounds) are
        # the only evidence of its length: fitted from those once it has any, not anchored by the priors, or the
        # machine's long runs of it are blamed on the methods that ran them (G-k2lp-sonnet-A4-r4, no reference: the
        # workload fitted at 785 s against runs of 12,664 s, the methods at x2.3-9.0, and its longest job, 29,158 s,
        # predicted at 667 s). With a reference, a kind it doesn't run keeps its priors as the anchor, as before.
        if self.reference_known and not self.reference:
            ruled = {k for c, k, src in self._obs if not src.startswith("prior:")}
        # no reference at all, and few kinds shared: the case where the factors are not identified. Where candidates
        # share most kinds (a sweep of methods over the same workloads), the shared kinds identify them; keep the fit
        shared = [k for k in kinds if len({c for c, kk, _ in self._obs if kk == k}) > 1]
        anchorless = self.reference_known and not self.reference and len(shared) * 2 < len(kinds)
        if anchorless:   # no yardstick: a kind the machine has measured is fitted from what it showed, not the priors
            ruled = {k for c, k, src in self._obs if not src.startswith("prior:")}

        def kind_obs(k):
            return [(c, y, w) for (c, kk, src), (y, w) in self._obs.items()
                    if kk == k and (k not in ruled or not src.startswith("prior:"))]

        # start the kind factors at their weighted mean, so the candidate factors start as relative slowdowns
        for k in kinds:
            ob = kind_obs(k)
            b[k] = sum(w * y for _, y, w in ob) / sum(w for _, _, w in ob)
        # a candidate with a measurement or a live projection gets its factor from those alone: the plan's priors are
        # the agent's guesses, usually one per kind copied across candidates, so they say how kinds differ, not how
        # fast a candidate is, and four flat priors would otherwise outvote one measured 4x slowdown (G1 Haiku A1)
        seen = {c for c, _, src in self._obs if not src.startswith("prior:")}
        # with no reference there is no yardstick: a candidate's speed and a kind's length are only told apart through
        # kinds two candidates share, and the alternating fit lets them trade off without bound (G-k2lp-haiku-A3, no
        # reference: one candidate's factor fitted at 1,444, another's at 0.07; a 29,157-s job predicted at 116,707 s,
        # then 639 s, then 8,747 s). So without a reference every candidate factor is 1, and each (candidate, kind)
        # cell is predicted from that kind's own measurements and running bounds, and from its prior until it has any.
        for _ in range(iterations):
            for c in cands:
                if c in self.reference or anchorless:
                    a[c] = 0.0
                    continue
                use = [(k, y, w) for (cc, k, src), (y, w) in self._obs.items()
                       if cc == c and (c not in seen or not src.startswith("prior:"))]
                num = sum(w * (y - b[k]) for k, y, w in use)
                den = sum(w for _, _, w in use) + RIDGE
                a[c] = num / den
            for k in kinds:
                ob = kind_obs(k)
                den = sum(w for _, _, w in ob)
                b[k] = sum(w * (y - a[c]) for c, y, w in ob) / den if den else 0.0
        self._a, self._b, self._dirty = a, b, False

    def predict(self, job) -> float:
        """Expected wall seconds for the whole job.

        Its own measurement if it has run. Otherwise the model's prediction for its (group, kind) cell, scaled by how
        this job's prior compares with the mean prior in that cell: jobs the plan says differ keep their differences,
        and what the machine measured about the cell moves them all together."""
        own = self._obs.get((_cand(job), _kind(job), "measured:" + job.id))
        if own:
            return math.exp(own[0])
        if self._dirty:
            self._fit()
        c, k = _cand(job), _kind(job)
        if k in self._b:
            cell = math.exp(self._a.get(c, 0.0) + self._b[k])
            if job.estimate_s and job.estimate_s > 0:
                priors = [y for (cc, kk, src), (y, w) in self._obs.items()
                          if cc == c and kk == k and src.startswith("prior:")]
                if len(priors) > 1:
                    mean_prior = sum(priors) / len(priors)
                    return cell * math.exp(math.log(job.estimate_s) - mean_prior)
            return cell
        if job.estimate_s:
            return float(job.estimate_s)
        if c in self._a and self._b:
            # an unseen kind: the candidate's factor on the mean kind
            return math.exp(self._a[c] + sum(self._b.values()) / len(self._b))
        return self.default_s

    def has_evidence(self, job) -> bool:
        """True once the job's group has a measured run or a live projection."""
        g = _cand(job)
        return any(c == g and not src.startswith("prior:") for (c, _, src) in self._obs)

    def set_reference(self, groups: Iterable[str]) -> None:
        """The reference's factor is 1 by definition, so everything it shows about a kind is about the kind: a long
        workload is seen as long, not as the reference being slow (G1 Haiku A1's replay, where a non-reference
        method's GSM8K overran and the model blamed the method)."""
        groups = frozenset(groups)
        if groups != self.reference or not self.reference_known:
            self.reference, self.reference_known, self._dirty = groups, True, True

    def cell_has_evidence(self, group: str, kind: str) -> bool:
        """True once a job of this group and kind has a measured run, a live projection or a running lower bound."""
        return any(c == group and k == kind and not src.startswith("prior:") for (c, k, src) in self._obs)

    def factor(self, candidate: str) -> float:
        """The candidate's speed factor relative to the fitted baseline (1.0 = average)."""
        if self._dirty:
            self._fit()
        return math.exp(self._a.get(candidate, 0.0))

    def explain(self, job) -> str:
        own = self._obs.get((_cand(job), _kind(job), "measured:" + job.id))
        if own:
            return f"measured {math.exp(own[0]):.0f}s"
        src = [s.split(":")[0] for (c, k, s) in self._obs if c == _cand(job) and k == _kind(job)]
        basis = ", ".join(sorted(set(src))) or "candidate and kind factors"
        return f"{self.predict(job):.0f}s from {basis} (candidate x{self.factor(_cand(job)):.2f})"

    def load(self, jobs: Iterable) -> None:
        """Rebuild the model from the board: priors from the plan, measurements from finished jobs."""
        self._obs.clear()
        for j in jobs:
            self.prior(j, j.estimate_s)
            if j.state.value == "done" and j.started_at and j.ended_at:
                self.measured(j, j.ended_at - j.started_at)
            elif j.pace and j.pace.projected_s:
                self.projected(j, j.pace.projected_s)
        self._dirty = True
