"""Run one prescribed TTC dataset/split/arm using the pinned project's code.

Requires the original TTC environment and its pinned TabArena dependency. It
calls the original runner's _run directly: the process owns one cell and exits
after writing the metrics.
"""
import argparse
import json
import math
import os
import pathlib
import shutil
import tempfile

ARMS = {
    "tabpfn3_aggregation_96": "tabpfn3-agg",
    "tabpfn3_diagscale_aggregation_96": "tabpfn3-agg-ft",
}
AGGREGATION = {"n_configs": 96, "seed": 0, "n_est_val": 2, "n_est_test": 8,
               "val_folds": 8, "n_iterations": 40}


def writable_cache_copy(source, destination):
    """OpenML 0.15.1 rewrites a pickle on read; keep the source snapshot intact.

    Copies and permissions are real CPU/filesystem work before the GPU seam.
    copytree also copies read-only directory modes, so normalize the owned copy.
    """
    destination = pathlib.Path(destination)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    destination.chmod(0o700)
    for path in destination.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)


def execute(dataset, split, arm):
    snapshot = os.environ.get("OPENML_CACHE_DIR")
    if snapshot:
        with tempfile.TemporaryDirectory(prefix="tabular-inputs-") as cache:
            writable_cache_copy(snapshot, cache)
            return execute_on_cache(dataset, split, arm, cache)
    return execute_on_cache(dataset, split, arm, None)


def execute_on_cache(dataset, split, arm, cache):
    import openml
    if cache:
        openml.config.set_root_cache_directory(cache)
    from tabarena.benchmark.experiment import AGModelOuterExperiment
    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
    from tabarena.nips2025_utils.tabarena_context import TabArenaContext
    from ttc.registry import MODELS, resolve_model_class

    model_key = ARMS[arm]
    spec = MODELS[model_key]
    context = TabArenaContext()
    grid = context.task_metadata_collection.task_grid()
    rows = grid[(grid["dataset"] == dataset) & (grid["split"] == split)]
    if len(rows) != 1:
        raise ValueError(f"expected one prescribed split, found {len(rows)}: {dataset}/{split}")
    coordinates = rows.iloc[0]
    metadata = [x for x in context.task_metadata_collection.tasks if x.tabarena_task_name == dataset]
    task_ids = {x.task_id_str for x in metadata}
    if len(task_ids) != 1:
        raise ValueError(f"ambiguous OpenML task identity: {task_ids}")
    task_id = int(next(iter(task_ids)))
    task = OpenMLTaskWrapper.from_task_id(task_id)
    hp = {**spec.hyperparameters, "aggregation": dict(AGGREGATION)}
    if arm == "tabpfn3_diagscale_aggregation_96":
        hp.update(ft_mode="diagscale", ft_lr=1e-2, ft_epochs=30)
    experiment = AGModelOuterExperiment(name=spec.display, model_cls=resolve_model_class(model_key),
                                        model_hyperparameters=hp)
    # This is the original Experiment.run setup, with a one-cell process rather
    # than a cached whole-suite process. CPU task loading and splitting run here.
    with experiment.task_cache_scope(task=task, cache_task_key=task_id):
        fit_args = experiment.init_method_kwargs(task=task)
        runner = experiment.experiment_cls(method_cls=experiment.method_cls, task=task,
                                          fold=int(coordinates["fold"]), repeat=int(coordinates["repeat"]),
                                          sample=0, task_name=dataset, method=spec.display,
                                          fit_args=fit_args, cleanup=False)
        # r8's record names this original library method. The seam includes
        # scoring, whose cost the source did not publish; that omission is
        # disclosed in the record and candidate specification.
        result = runner._run()
    output = {"dataset": dataset, "split": split, "arm": arm,
              **{key: result[key] for key in ("metric", "metric_error", "time_train_s", "time_infer_s")}}
    if not math.isfinite(float(output["metric_error"])):
        raise ValueError("non-finite measured error")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    args = ap.parse_args()
    output = execute(args.dataset, args.split, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
