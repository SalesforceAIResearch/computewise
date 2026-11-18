"""Prepare one filter recipe's training data: filter data/pool/ with the recipe's datatrove filters, then tokenise what is kept
into GPT-2 training shards for the recipe's run directory.

usage (from the project root): python tools/prepare.py <recipe> [--tasks N] [--workers N] [--tok-workers N]

Reads data/pool/*.jsonl ({"text": ...} per line) and data/val/fineweb_val_000000.bin. Writes
  runs/<recipe>/filtered/       the kept documents (datatrove's JsonlWriter, one file per task)
  runs/<recipe>/filter-logs/    datatrove's logs and completion markers
  runs/<recipe>/data/fineweb10B/ fineweb_val_000000.bin and fineweb_train_000001.bin, ... (1e8 tokens each, the last one
                                 shorter): a 256-int32 header, then uint16 GPT-2 tokens, an <|endoftext|> before each document,
                                 as modded-nanogpt's data/fineweb.py writes them
The filter stage is datatrove's LocalPipelineExecutor: --tasks splits the pool's files among that many tasks (default 1, as
datatrove's), --workers runs that many tasks at once (default -1: all of them, as datatrove's). The tokeniser runs
--tok-workers processes (default os.cpu_count() - 2, as data/fineweb.py). Rerunning a recipe starts it again from scratch.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import time

import numpy as np

RECIPES = {
    "none": [],
    "gopher-repetition": ["GopherRepetitionFilter"],
    "gopher-quality": ["GopherQualityFilter"],
    "c4": ["C4QualityFilter"],
    "fineweb-quality": ["FineWebQualityFilter"],
    "gopher": ["GopherRepetitionFilter", "GopherQualityFilter"],
    "fineweb": ["GopherRepetitionFilter", "GopherQualityFilter", "C4QualityFilter", "FineWebQualityFilter"],
}
SHARD = 10**8
_enc = _eot = None


def make_filter(name):
    from datatrove.pipeline import filters
    if name == "C4QualityFilter":
        return filters.C4QualityFilter(filter_no_terminal_punct=False)   # as FineWeb's pipeline sets it
    return getattr(filters, name)()


def header(n):
    h = np.zeros(256, dtype=np.int32)
    h[0], h[1], h[2] = 20240520, 1, n
    return h.tobytes()


def _init():
    global _enc, _eot
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    _eot = _enc._special_tokens["<|endoftext|>"]


def tokenize(text):
    return np.array([_eot] + _enc.encode_ordinary(text), dtype=np.uint16)


def texts(files):
    for p in files:
        with open(p, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)["text"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("recipe", choices=sorted(RECIPES))
    ap.add_argument("--tasks", type=int, default=1)
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("--tok-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.join("runs", a.recipe)
    shards = os.path.join(out, "data", "fineweb10B")
    for d in ("filtered", "filter-logs"):
        shutil.rmtree(os.path.join(out, d), ignore_errors=True)
    shutil.rmtree(shards, ignore_errors=True)
    os.makedirs(shards)
    steps = RECIPES[a.recipe]
    if steps:
        from datatrove.executor import LocalPipelineExecutor
        from datatrove.pipeline.readers import JsonlReader
        from datatrove.pipeline.writers import JsonlWriter
        LocalPipelineExecutor(
            pipeline=[JsonlReader("data/pool", glob_pattern="*.jsonl")] + [make_filter(s) for s in steps]
                     + [JsonlWriter(os.path.join(out, "filtered"), output_filename="${rank}.jsonl", compression=None)],
            tasks=a.tasks, workers=a.workers, logging_dir=os.path.join(out, "filter-logs")).run()
        kept = sorted(glob.glob(os.path.join(out, "filtered", "*.jsonl")))
    else:
        kept = sorted(glob.glob("data/pool/*.jsonl"))
    t1 = time.time()
    buf, n, idx, ntok = np.empty(SHARD, dtype=np.uint16), 0, 1, 0
    with mp.Pool(a.tok_workers, initializer=_init) as p:
        for tok in p.imap(tokenize, texts(kept), chunksize=16):
            ntok += len(tok)
            while len(tok):
                take = min(SHARD - n, len(tok))
                buf[n:n + take] = tok[:take]
                n += take
                tok = tok[take:]
                if n == SHARD:
                    with open(os.path.join(shards, f"fineweb_train_{idx:06d}.bin"), "wb") as f:
                        f.write(header(SHARD))
                        f.write(buf.tobytes())
                    idx += 1
                    n = 0
    if n:
        with open(os.path.join(shards, f"fineweb_train_{idx:06d}.bin"), "wb") as f:
            f.write(header(n))
            f.write(buf[:n].tobytes())
    shutil.copy("data/val/fineweb_val_000000.bin", shards)
    print(f"{a.recipe}: filter {t1 - t0:.0f} s, tokenise {time.time() - t1:.0f} s; {ntok:,} tokens kept, "
          f"{idx - (0 if n else 1)} training shards in {shards}", flush=True)


if __name__ == "__main__":
    main()
