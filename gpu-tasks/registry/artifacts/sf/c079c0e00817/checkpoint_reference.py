"""Brief-only ensemble reference, run only inside a staged W0 workspace.

Reads the run list and blank ensemble index. Gives each member a unique
ordinary --out-dir; no record IDs, expected losses or hidden durations are read.
No checkpoint is made except by the real training script's torch.save.
"""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import time

EVAL = re.compile(r"step:(\d+)/(\d+) val_loss:([^ ]+)")
SAVED = re.compile(r"^saved checkpoint to (.+) save_time_s:([0-9.eE+-]+)$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(args.root)
    todo = deque()
    for line in Path("ensemble_train/runs.txt").read_text().splitlines():
        argv = shlex.split(line, comments=True)
        if not argv:
            continue
        recipe = Path(argv[1]).parent.name
        seed = int(argv[argv.index("--seed") + 1])
        out = Path("members") / recipe / ("seed" + str(seed))
        todo.append({"argv": argv + ["--out-dir", str(out)], "recipe": recipe, "seed": seed,
                     "checkpoint": str(out / "ckpt.pt")})
    index = json.loads(Path("ensemble.json").read_text())
    expected = {(m["recipe"], m["seed"]): m for m in index["members"]}
    assert len(expected) == len(todo) and {(j["recipe"], j["seed"]) for j in todo} == set(expected)
    if args.dry_run:
        print(json.dumps({"jobs": len(todo), "gpus": args.gpus,
                          "unique_destinations": len({j["checkpoint"] for j in todo})}))
        return
    if args.gpus < 1:
        raise ValueError("GPU count must be positive")
    logs = Path("reference_logs")
    logs.mkdir(exist_ok=True)
    free = deque(range(args.gpus))
    active = {}
    selector = selectors.DefaultSelector()
    started = time.monotonic()
    save_observations = []

    def write_save_observations():
        observed_starts = sorted(x["save_announcement_received_s"] for x in save_observations
                                 if x["save_announcement_received_s"] is not None)
        payload = {
            "scope": "unique-directory W0; no shared-path collision/corruption experiment",
            "save_elapsed_how": "seconds printed by the training process around its real torch.save, including the replay wrapper; no added delay",
            "receipt_time_how": "controller wall seconds when stdout was received; approximate observation times, not exact filesystem write boundaries",
            "observations": save_observations,
            "observed_announcement_gaps_s": [b - a for a, b in zip(observed_starts, observed_starts[1:])],
        }
        (logs / "save-timings.json").write_text(json.dumps(payload, indent=2) + "\n")

    def close(state, stop=False):
        proc = state["proc"]
        if stop and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        try:
            selector.unregister(proc.stdout)
        except KeyError:
            pass
        proc.stdout.close()
        state["log"].close()
        active.pop(proc.pid, None)
        free.append(state["gpu"])

    def consume(state, text):
        if text.startswith("saving checkpoint to "):
            state["save_observation"]["save_announcement_received_s"] = time.monotonic() - started
        saved = SAVED.match(text)
        if saved:
            state["save_observation"].update(
                reported_path=saved[1], save_elapsed_s=float(saved[2]),
                save_completion_received_s=time.monotonic() - started)
        match = EVAL.search(text)
        if match:
            state["loss"] = float(match[3])

    try:
        while todo or active:
            while todo and free:
                job = todo.popleft()
                gpu = free.popleft()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                proc = subprocess.Popen(job["argv"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True, bufsize=0)
                os.set_blocking(proc.stdout.fileno(), False)
                state = dict(job, proc=proc, gpu=gpu, buffer="", loss=None,
                             log=(logs / f"{job['recipe']}-s{job['seed']}.log").open("w"))
                observation = {"recipe": job["recipe"], "seed": job["seed"], "checkpoint": job["checkpoint"],
                               "launch_received_s": time.monotonic() - started,
                               "save_announcement_received_s": None, "save_completion_received_s": None,
                               "save_elapsed_s": None, "process_exit_received_s": None, "returncode": None}
                state["save_observation"] = observation
                save_observations.append(observation)
                active[proc.pid] = state
                selector.register(proc.stdout, selectors.EVENT_READ, state)
            for key, _ in selector.select(timeout=0.25):
                state = key.data
                chunk = os.read(key.fd, 65536)
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    state["log"].write(text)
                    state["buffer"] += text
                    while "\n" in state["buffer"]:
                        text, state["buffer"] = state["buffer"].split("\n", 1)
                        consume(state, text)
                elif state["proc"].poll() is not None:
                    consume(state, state["buffer"])
                    state["save_observation"].update(process_exit_received_s=time.monotonic() - started,
                                                     returncode=state["proc"].returncode)
                    checkpoint = Path(state["checkpoint"])
                    if state["proc"].returncode != 0 or state["loss"] is None or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                        raise RuntimeError(f"member {state['recipe']}/{state['seed']} did not produce a completed checkpoint")
                    expected[state["recipe"], state["seed"]].update(checkpoint=state["checkpoint"], validation_loss=state["loss"])
                    # The index is an output locator, never a substitute for
                    # isolation's real checkpoint digest and occupancy evidence.
                    Path("ensemble.json.tmp").write_text(json.dumps(index, indent=2) + "\n")
                    os.replace("ensemble.json.tmp", "ensemble.json")
                    write_save_observations()
                    close(state)
    finally:
        for state in list(active.values()):
            close(state, stop=True)
        write_save_observations()
        selector.close()


if __name__ == "__main__":
    main()
