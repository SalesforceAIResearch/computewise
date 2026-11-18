"""Brief-only SF reference; execute in a staged workspace, never on the laptop.

Only inputs: sf_train/runs.txt, four visible GPUs, stdout from launched scripts.
No registry, expected losses, job lengths or list of broken jobs is read.
It repairs each job independently at the first effective-LR print showing zero
rates. --observe first_eval retains the second stated reference, which waits
for a post-update evaluation. Neither makes a queue-wide inference. No
artificial repair delay is inserted.

Usage in the box: python reference_zero_lr.py --gpus 4
Read-only syntax/launch-list check: add --dry-run.
"""
import argparse
import ast
from collections import deque
import json
import math
import os
from pathlib import Path
import re
import selectors
import shlex
import signal
import subprocess
import time

EVAL = re.compile(r"step:(\d+)/(\d+) val_loss:([^ ]+)")
LRS = re.compile(r"effective_lrs:(\[[^\]]*\])")


def repair(argv):
    out = list(argv)
    if "--lr-scale" not in out:
        raise RuntimeError("zero effective rates but no LR-scale argument to repair")
    k = out.index("--lr-scale")
    if k + 1 >= len(out) or float(out[k + 1]) != 0:
        raise RuntimeError("zero effective rates not explained by the launch's --lr-scale")
    out[k + 1] = "1"
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--runs", type=Path, default=Path("sf_train/runs.txt"))
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("sf-reference"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--observe", choices=["first_update", "first_eval"], default="first_update")
    args = parser.parse_args()
    os.chdir(args.root)
    commands = [shlex.split(line, comments=True) for line in args.runs.read_text().splitlines()]
    commands = [x for x in commands if x]
    if args.gpus < 1 or not commands:
        raise ValueError("need a positive GPU count and at least one launch")
    for argv in commands:
        if len(argv) < 2 or Path(argv[0]).name not in {"python", "python3"}:
            raise ValueError("expected standalone Python launch lines")
    if args.dry_run:
        print(json.dumps({"jobs": len(commands), "gpus": args.gpus, "runs": str(args.runs), "observe": args.observe}))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    todo = deque((i, argv, 0) for i, argv in enumerate(commands))
    free = deque(range(args.gpus))
    active = {}
    completed = {}
    events = []
    selector = selectors.DefaultSelector()
    start = time.monotonic()

    def event(kind, job, **extra):
        events.append({"kind": kind, "job": job, "wall_s": time.monotonic() - start, **extra})

    def close(state, stopped=False):
        proc = state["proc"]
        if stopped and proc.poll() is None:
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
        free.appendleft(state["gpu"])

    def restart(state, evidence, step, loss=None):
        if state["attempt"] != 0:
            raise RuntimeError("rates remained zero after repair")
        fixed = repair(state["argv"])
        event("zero_lr_observed", state["job"], step=step, val_loss=loss, evidence=evidence,
              pid=state["proc"].pid, repair=fixed)
        close(state, stopped=True)
        todo.appendleft((state["job"], fixed, state["attempt"] + 1))
        return True

    def line(state, text):
        lr = LRS.search(text)
        if lr:
            values = ast.literal_eval(lr.group(1))
            state["zero_rates"] = bool(values) and all(float(x) == 0 for x in values)
            if state["zero_rates"] and args.observe == "first_update":
                step = re.search(r"step:(\d+)", text)
                return restart(state, "effective_lrs", int(step[1]) if step else None)
        evaluation = EVAL.search(text)
        if evaluation:
            step, total, loss = int(evaluation[1]), int(evaluation[2]), float(evaluation[3])
            state["last_eval"] = {"step": step, "declared_schedule_steps": total, "val_loss": loss}
            if step > 0 and state["zero_rates"]:
                return restart(state, "validation", step, loss)
        return False

    try:
        while todo or active:
            while todo and free:
                job, argv, attempt = todo.popleft()
                gpu = free.popleft()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
                proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        env=env, start_new_session=True, bufsize=0)
                os.set_blocking(proc.stdout.fileno(), False)
                state = {"proc": proc, "job": job, "argv": argv, "gpu": gpu, "attempt": attempt,
                         "zero_rates": False, "last_eval": None, "buffer": "",
                         "log": (args.output / f"job-{job:03d}-attempt-{attempt}.log").open("w")}
                active[proc.pid] = state
                selector.register(proc.stdout, selectors.EVENT_READ, state)
                event("launch", job, argv=argv, gpu=gpu, attempt=attempt, pid=proc.pid)
            # This waits on child output; it never sleeps or delays GPU work.
            for key, _ in selector.select(timeout=0.25):
                state = key.data
                if state["proc"].pid not in active:
                    continue
                chunk = os.read(key.fd, 65536)
                if chunk:
                    text = chunk.decode("utf-8", errors="replace")
                    state["log"].write(text)
                    state["log"].flush()
                    state["buffer"] += text
                    while "\n" in state["buffer"]:
                        text, state["buffer"] = state["buffer"].split("\n", 1)
                        if line(state, text):
                            break
                elif state["proc"].poll() is not None:
                    if state["buffer"] and line(state, state["buffer"]):
                        continue
                    code = state["proc"].returncode
                    last = state["last_eval"]
                    if code != 0 or not last or not math.isfinite(last["val_loss"]) or state["zero_rates"]:
                        raise RuntimeError(f"job {state['job']} did not finish trained: exit={code}, last={last}")
                    completed[state["job"]] = {"command": state["argv"], "attempt": state["attempt"], **last}
                    event("completed", state["job"], **last)
                    close(state)
        (args.output / "results.json").write_text(json.dumps({"observe": args.observe, "completed": completed,
                                                             "events": events}, indent=2) + "\n")
    finally:
        for state in list(active.values()):
            close(state, stopped=True)
        selector.close()


if __name__ == "__main__":
    main()
