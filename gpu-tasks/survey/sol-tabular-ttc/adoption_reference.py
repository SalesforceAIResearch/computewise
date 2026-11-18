"""Brief-only reference: observed paired means decide whether to continue.

The controller receives the same dataset manifest and published cost profiles as
the agent. It reads no registry, source score CSV or prospective answer.
"""
import argparse
import concurrent.futures
import json
import math
import os
import pathlib
import statistics
import subprocess


def run_cell(root, cell, gpu):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    subprocess.run(["bash", cell["script"]], cwd=root, env=env, check=True)
    value = json.loads((root / cell["result"]).read_text())
    for field in ("dataset", "arm", "split"):
        if value[field] != cell[field]:
            raise ValueError(f"result identity mismatch for {cell['job_id']}: {field}")
    if not math.isfinite(float(value["metric_error"])):
        raise ValueError(f"non-finite result for {cell['job_id']}")
    return value


def paired_means(results, arms, splits):
    expected = {(a, s) for a in arms for s in splits}
    if {(r["arm"], r["split"]) for r in results} != expected or len(results) != len(expected):
        raise ValueError("paired means require each prescribed cell exactly once")
    if len({r["metric"] for r in results}) != 1:
        raise ValueError("cannot average different error metrics")
    return {a: statistics.mean(float(r["metric_error"]) for r in results if r["arm"] == a) for a in arms}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path("."))
    ap.add_argument("--gpus", type=int, default=4)
    args = ap.parse_args()
    root = args.root.resolve()
    spec = json.loads((root / "datasets.json").read_text())
    cells = json.loads((root / "cost_profiles.json").read_text())["cells"]
    arms = spec["arms"]
    order = sorted(spec["datasets"], key=lambda d: (sum(c["fit_predict_s"] for c in cells if c["dataset"] == d), d))
    observed, recommendation = {}, "adopt"
    for dataset in order:
        pending = sorted((c for c in cells if c["dataset"] == dataset), key=lambda c: (-c["fit_predict_s"], c["job_id"]))
        results, running = [], {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.gpus) as pool:
            for gpu in range(args.gpus):
                if pending:
                    cell = pending.pop(0)
                    running[pool.submit(run_cell, root, cell, gpu)] = gpu
            while running:
                done, _ = concurrent.futures.wait(running, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    gpu = running.pop(future)
                    results.append(future.result())
                    if pending:
                        cell = pending.pop(0)
                        running[pool.submit(run_cell, root, cell, gpu)] = gpu
        means = paired_means(results, arms, spec["splits"])
        passes = means[arms[1]] <= means[arms[0]] * (1 - spec["minimum_relative_error_reduction"])
        observed[dataset] = {"means": means, "passes": passes, "metric": results[0]["metric"]}
        print(json.dumps({"dataset": dataset, **observed[dataset]}), flush=True)
        if not passes:
            recommendation = "reject"
            break
    report = {"recommendation": recommendation, "observed_datasets": observed}
    output = root / "results" / "adoption_decision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
