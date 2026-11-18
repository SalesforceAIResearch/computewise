"""The documents behind modded-nanogpt's FineWeb shards, as JSONL: the raw text of a data-preparation package (gpu-tasks,
2026-10-21).

usage: python3 -I build/gen_raw_text.py <shard dir> <gpt2 vocab.json> <out dir> [<last train shard>]

Reads fineweb_val_000000.bin and fineweb_train_000001..N.bin (kjj0/fineweb10B-gpt2; N defaults to 8), splits their token stream
into documents at <|endoftext|> (each shard starts with one, as data/fineweb.py writes them), decodes each document with GPT-2's
byte-level vocabulary to UTF-8 text and writes {"text": ...} lines, 100,000 documents a file, to <out dir>/fineweb_docs_NNN.jsonl.
The last document runs on into shard N+1, which is not read, so it is left out. Prints each file's sha256.

Tokenising these documents in order with data/fineweb.py's rule (an <|endoftext|> and encode_ordinary per document, 1e8-token
shards, shard 0 the validation shard, a straddling document split) gives back the validation shard and train shards 1..N-1
byte for byte, and shard N on the tokens it has (cpu-tasks, 2026-10-21, experiments/cpu-tasks/survey/PREP-STEPS-2026-10-21.md,
with tiktoken's decode; build/check_dataprep.sh repeats it on these files).

Standard library only, so build/stage_task.py can run it where it stages the task. A document that is not valid UTF-8 stops it.
"""
import array
import hashlib
import json
import os
import sys

EOT = 50256
MAGIC, VERSION = 20240520, 1
DOCS_PER_FILE = 100_000


def bytes_to_unicode():
    """GPT-2's map from bytes to the printable characters its vocabulary spells them with (openai/gpt-2 encoder.py)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def decoder(vocab_path):
    """Token id -> its bytes."""
    enc = json.load(open(vocab_path, encoding="utf-8"))
    bd = {c: b for b, c in bytes_to_unicode().items()}
    table = [b""] * (max(enc.values()) + 1)
    for tok, i in enc.items():
        table[i] = bytes(bd[c] for c in tok) if i != EOT else b""
    return table


def shard_tokens(path):
    with open(path, "rb") as f:
        head = array.array("i")
        head.frombytes(f.read(1024))
        body = array.array("H")
        body.frombytes(f.read())
    if sys.byteorder != "little":
        head.byteswap()
        body.byteswap()
    if head[0] != MAGIC or head[1] != VERSION or len(body) != head[2]:
        raise SystemExit(f"{path}: not a complete data shard")
    return body


def main():
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    sdir, vocab, out = sys.argv[1:4]
    last = int(sys.argv[4]) if len(sys.argv) == 5 else 8
    names = ["fineweb_val_000000.bin"] + [f"fineweb_train_{i:06d}.bin" for i in range(1, last + 1)]
    tok = array.array("H")
    for n in names:
        tok.extend(shard_tokens(os.path.join(sdir, n)))
    if tok[0] != EOT:
        raise SystemExit("the stream does not start with <|endoftext|>")
    table = decoder(vocab)
    os.makedirs(out, exist_ok=True)
    files, fh, ndoc, start = [], None, 0, 0
    while True:
        try:
            end = tok.index(EOT, start + 1)
        except ValueError:
            break                                  # the last document runs on into the next shard
        if ndoc % DOCS_PER_FILE == 0:
            if fh:
                fh.close()
            files.append(os.path.join(out, f"fineweb_docs_{ndoc // DOCS_PER_FILE:03d}.jsonl"))
            fh = open(files[-1], "w", encoding="utf-8", newline="\n")
        text = b"".join([table[t] for t in tok[start + 1:end]]).decode("utf-8")
        fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        ndoc += 1
        start = end
    if fh:
        fh.close()
    print(f"{ndoc:,} documents from {len(names)} shards ({len(tok):,} tokens; the last document, from token {start:,}, left out)")
    for p in files:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        print(f"{os.path.basename(p)} {os.path.getsize(p)} {h.hexdigest()}")


if __name__ == "__main__":
    main()
