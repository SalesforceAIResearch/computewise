"""The raw document pool of H4, pkg-h4-filter-ablation: FineWeb text for the agent to filter.

usage: python3 -I build/gen_h4_pool.py <shard dir> <gpt2 vocab.json> <out dir> [<n files>]

Reads fineweb_train_000001..8.bin (kjj0/fineweb10B-gpt2; the validation shard, if present in <shard dir>, is not read: it is the
evaluation set and the task ships it as is), splits the token stream into documents at <|endoftext|>, decodes each with GPT-2's
byte-level vocabulary (build/gen_raw_text.py's decoder) and writes {"text": ...} lines to <out dir>/pool_NNN.jsonl, the
documents in order, split evenly over <n files> files (default 512). The stream's first piece (the tail of a document that began
in the validation shard) and its last (a document that runs on into shard 9) are left out, so every line is one whole
document. Prints each file's size and sha256.

512 files: the datatrove reference preparation (cpu-tasks' timing on c3-highmem-44, 2026-10-23) runs one task per file on 42
workers. Standard library only, so build/stage_task.py can run it where it stages the task.
"""
import array
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_raw_text as RT  # noqa: E402

TRAIN_SHARDS = 8
N_FILES = 512


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    sdir, vocab, out = sys.argv[1:4]
    nfiles = int(sys.argv[4]) if len(sys.argv) == 5 else N_FILES
    tok = array.array("H")
    for i in range(1, TRAIN_SHARDS + 1):
        tok.extend(RT.shard_tokens(os.path.join(sdir, f"fineweb_train_{i:06d}.bin")))
    eots = [i for i, t in enumerate(tok) if t == RT.EOT]
    spans = list(zip(eots[:-1], eots[1:]))          # whole documents only
    table = RT.decoder(vocab)
    os.makedirs(out, exist_ok=True)
    per = -(-len(spans) // nfiles)
    files = []
    for k in range(nfiles):
        p = os.path.join(out, f"pool_{k:03d}.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            for s, e in spans[k * per:(k + 1) * per]:
                text = b"".join([table[t] for t in tok[s + 1:e]]).decode("utf-8")
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        files.append(p)
    print(f"{len(spans):,} documents from train shards 1..{TRAIN_SHARDS} ({len(tok):,} tokens), {nfiles} files of {per:,}")
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print(f"{os.path.basename(p)} {os.path.getsize(p)} {h.hexdigest()}")


if __name__ == "__main__":
    main()
