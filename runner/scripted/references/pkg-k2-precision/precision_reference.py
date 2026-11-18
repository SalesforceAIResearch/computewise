"""P1 sequential reference, using only RUNS.tsv and actual training output.

Portable standard-library controller. No registry, recorded endpoint values,
winner or resolving sample count is read. The finite family is three arms by
three looks (4, 8, all available); iid Gaussian outcomes are assumed.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import signal
import subprocess
import tempfile
import time


# Same incomplete-beta Student-t calculation as gen_k2_precision.py.
def beta_fraction(a, b, x):
    tiny, eps = 1e-300, 2e-14
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 401):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d, c = (d if abs(d) > tiny else tiny), (c if abs(c) > tiny else tiny)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d, c = 1 + aa * d, 1 + aa / c
        d, c = (d if abs(d) > tiny else tiny), (c if abs(c) > tiny else tiny)
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps:
            return h
    raise ArithmeticError("incomplete beta did not converge")


def regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * beta_fraction(a, b, x) / a
    return 1 - factor * beta_fraction(b, a, 1 - x) / b


def student_t_cdf(t, degrees):
    if t < 0:
        return 1 - student_t_cdf(-t, degrees)
    return 1 - 0.5 * regularized_beta(degrees / (degrees + t * t), degrees / 2, 0.5)


def student_t_quantile(probability, degrees):
    if not 0.5 < probability < 1 or degrees < 1:
        raise ValueError((probability, degrees))
    lo, hi = 0.0, 1.0
    while student_t_cdf(hi, degrees) < probability:
        hi *= 2
    for _ in range(80):
        mid = (lo + hi) / 2
        if student_t_cdf(mid, degrees) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def assess(values):
    probability = 1 - 0.05 / (3 * 3) / 2
    intervals = {}
    for arm, samples in values.items():
        critical = student_t_quantile(probability, len(samples) - 1)
        mean = statistics.mean(samples)
        half = critical * statistics.stdev(samples) / math.sqrt(len(samples))
        intervals[arm] = {"n": len(samples), "mean": mean, "low": mean - half,
                          "high": mean + half, "critical": critical}
    winners = [a for a in intervals if all(a == b or intervals[a]["high"] < intervals[b]["low"] for b in intervals)]
    return intervals, winners[0] if len(winners) == 1 else None


def endpoint(path):
    pattern = re.compile(r"\bstep:(\d+)/\d+\s+val_loss:([+\-\d.eE]+)")
    matches = list(pattern.finditer(path.read_text(errors="replace")))
    if not matches or int(matches[-1][1]) != 1500:
        raise RuntimeError(f"no completed step1500 validation in {path}")
    value = float(matches[-1][2])
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite validation in {path}")
    return value


def run_batch(rows, workspace, gpus, directory):
    waiting, active, completed = list(rows), {}, []
    try:
        while waiting or active:
            for gpu in range(gpus):
                if gpu in active or not waiting:
                    continue
                row = waiting.pop(0)
                path = directory / (row["optimizer"] + "-" + row["replication"] + ".log")
                stream = path.open("w")
                proc = subprocess.Popen(row["command"], shell=True, cwd=workspace,
                                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                active[gpu] = proc, row, path, stream
            for gpu, (proc, row, path, stream) in list(active.items()):
                code = proc.poll()
                if code is None:
                    continue
                stream.close()
                del active[gpu]
                if code:
                    raise RuntimeError(f"training exit{code}: {path}")
                completed.append({"optimizer": row["optimizer"], "replication": int(row["replication"]),
                                  "val_loss": endpoint(path), "log": str(path)})
            if active:
                time.sleep(0.2)
    finally:
        for proc, row, path, stream in active.values():
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            stream.close()
    return completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workspace", "--root", type=Path, default=Path.cwd())
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()
    workspace = args.workspace.resolve()
    if args.gpus < 1:
        raise ValueError("positive GPU count required")
    rows = list(csv.DictReader((workspace / "RUNS.tsv").open(), delimiter="\t"))
    arms = list(dict.fromkeys(r["optimizer"] for r in rows))
    if len(arms) != 3:
        raise ValueError("this P1 family is defined for three optimizers")
    if any(len([r for r in rows if r["optimizer"] == arm]) < 8 for arm in arms):
        raise ValueError("the declared 4/8/all family requires at least eight replications per arm")
    completed, looks = [], []
    directory = Path(tempfile.mkdtemp(prefix="precision-reference-", dir=workspace))
    for target in (4, 8, None):
        done = {(r["optimizer"], r["replication"]) for r in completed}
        batch = [r for r in rows if (r["optimizer"], int(r["replication"])) not in done
                 and (target is None or int(r["replication"]) <= target)]
        completed.extend(run_batch(batch, workspace, args.gpus, directory))
        values = {arm: [r["val_loss"] for r in completed if r["optimizer"] == arm] for arm in arms}
        intervals, winner = assess(values)
        look = {"look": target if target is not None else "all_available", "intervals": intervals, "winner": winner,
                "jobs_completed": len(completed), "familywise_alpha": 0.05,
                "model": "iid Gaussian outcomes within each optimizer; simultaneous Bonferroni across9 intervals"}
        looks.append(look)
        print(json.dumps(look), flush=True)
        if winner is not None:
            break
    result = {"looks": looks, "completed": completed,
              "conclusion": {"winner": looks[-1]["winner"], "unresolved": looks[-1]["winner"] is None}}
    (directory / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
