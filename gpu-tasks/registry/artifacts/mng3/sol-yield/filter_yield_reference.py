"""Duration-blind reference for pkg-filter-yield, run inside its staged box.

One real preparation at a time, as H4's measured 42-worker command. Passing
recipes release their four GPU jobs immediately. Every preparation is completed,
including ones which release no GPU work. Never reads registry scores or times.
"""
import argparse
import concurrent.futures
import json
import os
import pathlib
import queue
import signal
import struct
import subprocess
import threading
import time

RECIPES = ("none", "gopher-repetition", "gopher-quality", "c4", "fineweb-quality", "gopher", "fineweb")
RAW_TOKENS = 799997818
MIN_RETENTION = 0.95
SEEDS = (0, 1, 2, 3)


def token_count(root, recipe):
    files = sorted((root / "yieldlab" / recipe / "data" / "fineweb10B").glob("fineweb_train_*.bin"))
    if not files:
        raise ValueError(f"no prepared training shards for {recipe}")
    total = 0
    for path in files:
        with path.open("rb") as f:
            header = f.read(1024)
        if len(header) != 1024:
            raise ValueError(f"short shard header: {path}")
        magic, version, n = struct.unpack_from("<iii", header)
        if (magic, version) != (20240520, 1) or n <= 0 or path.stat().st_size != 1024 + n * 2:
            raise ValueError(f"corrupt or incomplete shard: {path}")
        total += n
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--workers", type=int, default=42)
    a = ap.parse_args()
    root = a.root.resolve()
    logs = root / "reference-logs"
    logs.mkdir(exist_ok=True)
    pending = queue.Queue()
    events, errors, children = [], [], set()
    lock = threading.Lock()
    start = time.monotonic()

    def event(kind, **fields):
        row = {"t_s": time.monotonic() - start, "event": kind, **fields}
        with lock:
            events.append(row)
            with (logs / "events.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")

    def run(cmd, logfile, env=None):
        with logfile.open("w") as out:
            p = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            with lock:
                children.add(p)
            try:
                rc = p.wait()
                if rc:
                    raise RuntimeError(f"exit {rc}: {cmd}; output {logfile}")
            finally:
                with lock:
                    children.discard(p)

    def lane(gpu):
        while True:
            item = pending.get()
            try:
                if item is None:
                    return
                recipe, seed = item
                event("gpu_start", recipe=recipe, seed=seed, gpu=gpu)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                run(["python", f"yieldlab/{recipe}/train_gpt.py", "--seed", str(seed)],
                    logs / f"{recipe}-seed{seed}.log", env)
                event("gpu_end", recipe=recipe, seed=seed, gpu=gpu)
            except Exception as e:
                with lock:
                    errors.append(str(e))
            finally:
                pending.task_done()

    def stop(signum, frame):
        with lock:
            active = list(children)
        for p in active:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    observed = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.gpus) as pool:
        lanes = [pool.submit(lane, gpu) for gpu in range(a.gpus)]
        try:
            for recipe in RECIPES:
                event("prep_start", recipe=recipe)
                run(["python", "tools/prepare.py", recipe, "--tasks", "512", "--workers", str(a.workers)],
                    logs / f"prepare-{recipe}.log")
                count = token_count(root, recipe)
                passed = count >= MIN_RETENTION * RAW_TOKENS
                observed[recipe] = {"tokens": count, "retention": count / RAW_TOKENS, "passes": passed}
                event("prep_end", recipe=recipe, **observed[recipe])
                if passed:
                    for seed in SEEDS:
                        pending.put((recipe, seed))
        finally:
            for _ in lanes:
                pending.put(None)
        pending.join()
        for future in lanes:
            future.result()
    result = {"observed": observed, "gpu_jobs_completed": sum(x["event"] == "gpu_end" for x in events),
              "elapsed_s": time.monotonic() - start, "errors": errors,
              "policy": "one real preparation at a time, enqueue passing seeds on completion; no recorded duration or outcome read"}
    (logs / "reference.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
